"""QuIP# E8P quantization over shared teacher replay and block reconstruction."""

from __future__ import annotations

from dataclasses import asdict
from operator import attrgetter
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from weight_vq import utils
from weight_vq.blockwise.artifacts import PROJECTIONS
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.quip_sharp.config import QuIPSharpConfig


def quantize_layer(
    layer: nn.Module, index: int, config: QuIPSharpConfig, reconstruction: DistributedReconstruction
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
    cb = codebook.get_codebook(config.codebook)
    saved_projections = {}
    for attribute, short, hessian_name, _parallel in PROJECTIONS:
        linear = attrgetter(attribute)(layer)
        with torch.no_grad():
            # Match upstream quantize_linear without its per-file restart/write behavior:
            # an incomplete block must restart from the teacher on every rank.
            statistics = torch.load(
                config.hessians / f"{index}_{hessian_name}.pt",
                map_location="cpu",
                weights_only=True,
            )
            hessian = quip_utils.flat_to_sym(statistics["flatH"], statistics["n"])
            mu = statistics["mu"]
            hessian.add_(mu[:, None] * mu[None, :])
            hessian = quip_utils.regularize_H(hessian, statistics["n"], config.sigma_reg)
            dtype = torch.float64 if config.use_fp64 else torch.float32
            weight = linear.weight.detach().to(device="cpu", dtype=dtype)
            scale = weight.square().mean().sqrt()
            _decoded, saved = quip.quantize(hessian, weight / scale, 0, cb, args, str(device))
            saved["SV"] *= scale
            saved.update(fused=False, shapes=[weight.shape], scales=[1.0])
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
            quip_utils.unpack_quip(quantized, saved, codebook.get_id(config.codebook), cb.codesz)
            if linear.bias is not None:
                quantized.bias = nn.Parameter(linear.bias.detach().float().cpu().clone())
            saved_projections[short] = saved
            del statistics, hessian, mu, weight, _decoded
        quantized.SU = nn.Parameter(quantized.get_buffer("SU").float())
        quantized.SV = nn.Parameter(quantized.get_buffer("SV").float())
        parent, name = attribute.rsplit(".", 1)
        setattr(attrgetter(parent)(layer), name, quantized)
        del linear
        # Paper Algorithm FT reconstructs after every projection, including the last.
        reconstruction(layer, f"{index}_{short}", orig_dtype)

    if reconstruction.distributed.is_primary:
        for attribute, short, *_ in PROJECTIONS:
            quantized = attrgetter(attribute)(layer)
            saved = saved_projections[short]
            saved["SU"] = quantized.SU.detach().cpu().clone()
            saved["SV"] = quantized.SV.detach().cpu().clone()
            utils.atomic_torch_save(saved, config.checkpoints / f"{index}_{short}.pt")
    layer.cpu()


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
