"""Author's identical-codebook LDLQ, whole-block reconstruction, and dense decode."""

from __future__ import annotations

from dataclasses import asdict
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
from weight_vq.unisvq.config import UniSVQConfig
from weight_vq.unisvq.reconstruction import UniSVQReconstruction


def quantize_layer(
    layer: nn.Module,
    index: int,
    config: UniSVQConfig,
    reconstruction: UniSVQReconstruction,
    *,
    projection_owners: dict[str, int] | None = None,
) -> None:
    # Method-environment boundary: do not substitute QuIP#'s LDLQ or codebook.
    from unisvq import codebook
    from unisvq import utils as upstream_utils
    from unisvq.algo import quip
    from unisvq.linear import QuantizedLinear

    device = reconstruction.distributed.device
    original_dtype = next(layer.parameters()).dtype
    layer.float()
    # Upstream deliberately resets the per-block seed to the block index.
    torch.manual_seed(index)
    args = SimpleNamespace(
        **asdict(config),
        use_fp64=False,
        lora_rank=0,
        incoh_mode="had",
        rescale_WH=False,
        resid_scale_override=-1,
        no_use_buffered=False,
        lowmem_ldlq=False,
    )
    experts = getattr(layer.mlp, "experts", None)
    specs = projections(len(experts) if isinstance(experts, LinearExperts) else 0)
    distributed = reconstruction.distributed
    owners = projection_owners or {}
    retries = []
    saved_projections = {}
    modules = {}
    workspace = None
    sharded = False
    # UniSVQ has no intermediate reconstruction: join once after the entire block.
    for sharded in quantization_attempts(distributed, bool(owners), retries):
        saved_projections.clear()
        modules.clear()
        for attribute, short, hessian_name, _ in specs:
            # Both constructors consume CPU RNG even for this fixed grid. Keep their
            # original order on every rank, including experts owned by another rank.
            cb = codebook.get_codebook(config.codebook, codebook_bit=config.codebook_bit)
            if workspace is None:
                workspace = cb.to(device=device, dtype=torch.float32)
            linear = attrgetter(attribute)(layer)
            with torch.no_grad():
                if sharded and short in owners and owners[short] != distributed.rank:
                    torch.randn(linear.in_features, device=device)
                    torch.randn(linear.out_features, device=device)
                else:
                    hessian, metadata = load_quantization_hessian(
                        config.hessians / f"{index}_{hessian_name}.pt", config.min_expert_coverage
                    )
                    hessian = upstream_utils.regularize_H(
                        hessian, hessian.shape[0], config.sigma_reg
                    )
                    weight = linear.weight.detach().float().cpu()
                    scale = weight.square().mean().sqrt()
                    _decoded, saved = quip.quantize(
                        hessian,
                        weight / scale,
                        0,
                        workspace,
                        args,
                        str(device),
                        reuse_codebook=True,
                        on_retry=retries.append,
                    )
                    saved["SV"] *= scale
                    saved.update(
                        fused=False,
                        shapes=[weight.shape],
                        scales=[1.0],
                        bias=None if linear.bias is None else linear.bias.detach().cpu().clone(),
                        **metadata,
                    )
                    saved_projections[short] = saved
                    del hessian, weight, _decoded
                modules[short] = QuantizedLinear(
                    linear.in_features,
                    linear.out_features,
                    cb.codesz,
                    cb.packsz,
                    cb.pack_out,
                    cb.version,
                    rank=0,
                    bias=linear.bias is not None,
                    train_mode=False,
                    grad_ckpt=config.gradient_checkpointing,
                    codebook_id=codebook.get_id(config.codebook),
                    codebook_bit=config.codebook_bit,
                    idx_dtype=str(cb.idx_dtype),
                )
            del linear, cb
    if sharded:
        broadcast_quantized(saved_projections, owners, distributed)
    for attribute, short, *_ in specs:
        quantized = modules[short]
        upstream_utils.unpack_quip(
            quantized,
            saved_projections[short],
            codebook.get_id(config.codebook),
            quantized.codebook_class.codebook.codesz,
        )
        quantized.SU = nn.Parameter(quantized.get_buffer("SU").float())
        quantized.SV = nn.Parameter(quantized.get_buffer("SV").float())
        parent, name = attribute.rsplit(".", 1)
        setattr(attrgetter(parent)(layer), name, quantized)
    # The released PTQ reconstructs once, after replacing the entire block.
    reconstruction(layer, str(index), original_dtype)
    if reconstruction.distributed.is_primary:
        for attribute, short, *_ in specs:
            quantized = attrgetter(attribute)(layer)
            saved = saved_projections[short]
            # Preserve the released checkpoint's FP16 SU/SV storage, not QuIP#'s FP32.
            saved["SU"] = quantized.SU.detach().half().cpu()
            saved["SV"] = quantized.SV.detach().half().cpu()
            utils.atomic_torch_save(saved, config.checkpoints / f"{index}_{short}.pt")
    layer.cpu()
    del workspace
    utils.clear_cuda_cache()


@torch.no_grad()
def reconstruct_dense_weight(saved: dict[str, Any], device: torch.device) -> torch.Tensor:
    from unisvq import codebook
    from unisvq.utils.matmul_had import matmul_hadU

    # IndexLinear's released grid is BF16 even outside training autocast.
    cb = codebook.get_codebook("identical", codebook_bit=2).to(device=device, dtype=torch.bfloat16)
    raw = cb.by_idxs(saved["Qidxs"].to(device)).float()
    transformed = matmul_hadU(matmul_hadU(raw / 32).T).T
    su = saved["SU"].to(device).float()
    sv = saved["SV"].to(device).float()
    return ((transformed * su) * (sv * 32).unsqueeze(1)).contiguous().cpu()
