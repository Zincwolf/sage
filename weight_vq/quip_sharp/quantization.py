"""QuIP# E8P quantization over shared teacher replay and block reconstruction."""

from __future__ import annotations

from dataclasses import asdict
from itertools import groupby
from operator import attrgetter
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from weight_vq import utils
from weight_vq.blockwise.artifacts import projections
from weight_vq.blockwise.distributed import broadcast_quantized, quantization_attempts
from weight_vq.blockwise.hessian import load_quantization_hessian
from weight_vq.blockwise.models import LinearExperts
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.quip_sharp.config import QuIPSharpConfig


def quantize_layer(
    layer: nn.Module,
    index: int,
    config: QuIPSharpConfig,
    reconstruction: DistributedReconstruction,
    *,
    projection_owners: dict[str, int] | None = None,
) -> None:
    # Method-environment boundary: use upstream LDLQ, packing, and train-mode Linear.
    from quip_sharp import codebook
    from quip_sharp import utils as quip_utils
    from quip_sharp.algo import quip
    from quip_sharp.linear import QuantizedLinear

    device = reconstruction.distributed.device
    orig_dtype = next(layer.parameters()).dtype
    layer.float()
    torch.manual_seed(config.seed + index)
    # Upstream FP64 retry mutates a copied args object; our public config is frozen.
    args = SimpleNamespace(
        **asdict(config),
        lora_rank=0,
        incoh_mode="had",
        rescale_WH=False,
        resid_scale_override=-1,
        no_use_buffered=False,
        lowmem_ldlq=False,
    )
    cb = codebook.get_codebook(config.codebook).to(device)
    saved_projections = {}
    experts = getattr(layer.mlp, "experts", None)
    expert_count = len(experts) if isinstance(experts, LinearExperts) else 0
    specs = projections(expert_count, getattr(layer, "block_type", "full_attention"))
    distributed = reconstruction.distributed
    # Attention keeps Algorithm FT; independent experts join at gate/up/down boundaries.
    for _, entries in groupby(
        specs,
        key=lambda spec: (
            spec[0].rsplit(".", 1)[1] if spec[0].startswith("mlp.experts.") else spec[1]
        ),
    ):
        group = list(entries)
        owners = {
            short: projection_owners[short]
            for _, short, *_ in group
            if projection_owners and short in projection_owners
        }
        retries = []
        payloads = {}
        sharded = False
        for sharded in quantization_attempts(distributed, bool(owners), retries):
            payloads.clear()
            for attribute, short, hessian_name, _parallel in group:
                linear = attrgetter(attribute)(layer)
                with torch.no_grad():
                    if sharded and owners[short] != distributed.rank:
                        # Match upstream's initial Hadamard draws without loading H/weight.
                        torch.randn(linear.in_features, device=device)
                        torch.randn(linear.out_features, device=device)
                        continue
                    hessian, metadata = load_quantization_hessian(
                        config.hessians / f"{index}_{hessian_name}.pt", config.min_expert_coverage
                    )
                    hessian = quip_utils.regularize_H(hessian, hessian.shape[0], config.sigma_reg)
                    dtype = torch.float64 if config.use_fp64 else torch.float32
                    weight = linear.weight.detach().to(device="cpu", dtype=dtype)
                    scale = weight.square().mean().sqrt()
                    _decoded, saved = quip.quantize(
                        hessian,
                        weight / scale,
                        0,
                        cb,
                        args,
                        str(device),
                        reuse_codebook=True,
                        on_retry=retries.append,
                    )
                    saved["SV"] *= scale
                    saved.update(fused=False, shapes=[weight.shape], scales=[1.0], **metadata)
                    payloads[short] = saved
                    del hessian, weight, _decoded
        if sharded:
            broadcast_quantized(payloads, owners, distributed)
        saved_projections.update(payloads)
        for attribute, short, *_ in group:
            linear = attrgetter(attribute)(layer)
            quantized = QuantizedLinear(
                linear.in_features,
                linear.out_features,
                cb.codesz,
                cb.packsz,
                cb.pack_out,
                str(cb.idx_dtype),
                cb.version,
                rank=0,
                bias=linear.bias is not None,
                train_mode=True,
                grad_ckpt=config.gradient_checkpointing,
            )
            quip_utils.unpack_quip(
                quantized, payloads[short], codebook.get_id(config.codebook), cb.codesz
            )
            if linear.bias is not None:
                quantized.bias = nn.Parameter(linear.bias.detach().float().cpu().clone())
            quantized.SU = nn.Parameter(quantized.get_buffer("SU").float())
            quantized.SV = nn.Parameter(quantized.get_buffer("SV").float())
            parent, name = attribute.rsplit(".", 1)
            setattr(attrgetter(parent)(layer), name, quantized)
            del linear
        reconstruction(layer, f"{index}_{group[-1][1]}", orig_dtype)

    if reconstruction.distributed.is_primary:
        for attribute, short, *_ in specs:
            quantized = attrgetter(attribute)(layer)
            saved = saved_projections[short]
            saved["SU"] = quantized.SU.detach().cpu().clone()
            saved["SV"] = quantized.SV.detach().cpu().clone()
            utils.atomic_torch_save(saved, config.checkpoints / f"{index}_{short}.pt")
    layer.cpu()
    cb.cpu()
    utils.clear_cuda_cache()


@torch.no_grad()
def reconstruct_dense_weight(saved: dict[str, Any], device: torch.device) -> torch.Tensor:
    # Packed E8P decoding is supplied by upstream's CUDA extension.
    from quip_sharp import codebook
    from quip_sharp.utils.matmul_had import matmul_hadU

    cb = codebook.get_codebook("E8P12").to(device)
    raw = cb.by_idxs(saved["Qidxs"].to(device)).float()
    transformed = matmul_hadU(matmul_hadU(raw / 32).T).T
    su = saved["SU"].to(device).float()
    sv = saved["SV"].to(device).float()
    return ((transformed * su) * (sv * 32).unsqueeze(1)).contiguous().cpu()
