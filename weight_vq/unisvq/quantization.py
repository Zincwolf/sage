"""Author's identical-codebook LDLQ, whole-block reconstruction, and dense decode."""

from __future__ import annotations

from dataclasses import asdict
from operator import attrgetter
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from weight_vq import utils
from weight_vq.blockwise.artifacts import PROJECTIONS
from weight_vq.unisvq.config import UniSVQConfig
from weight_vq.unisvq.reconstruction import UniSVQReconstruction


def quantize_layer(
    layer: nn.Module, index: int, config: UniSVQConfig, reconstruction: UniSVQReconstruction
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
    saved_projections = {}
    for attribute, short, hessian_name, _ in PROJECTIONS:
        cb = codebook.get_codebook(config.codebook, codebook_bit=config.codebook_bit)
        linear = attrgetter(attribute)(layer)
        with torch.no_grad():
            statistics = torch.load(
                config.hessians / f"{index}_{hessian_name}.pt",
                map_location="cpu",
                weights_only=True,
            )
            hessian = upstream_utils.flat_to_sym(statistics["flatH"], statistics["n"])
            mu = statistics["mu"]
            hessian.add_(mu[:, None] * mu[None, :])
            hessian = upstream_utils.regularize_H(hessian, statistics["n"], config.sigma_reg)
            weight = linear.weight.detach().float().cpu()
            scale = weight.square().mean().sqrt()
            _decoded, saved = quip.quantize(hessian, weight / scale, 0, cb, args, str(device))
            saved["SV"] *= scale
            saved.update(
                fused=False,
                shapes=[weight.shape],
                scales=[1.0],
                bias=None if linear.bias is None else linear.bias.detach().cpu().clone(),
            )
            quantized = QuantizedLinear(
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
            upstream_utils.unpack_quip(
                quantized, saved, codebook.get_id(config.codebook), cb.codesz
            )
            saved_projections[short] = saved
            del statistics, hessian, mu, weight, _decoded
        quantized.SU = nn.Parameter(quantized.get_buffer("SU").float())
        quantized.SV = nn.Parameter(quantized.get_buffer("SV").float())
        parent, name = attribute.rsplit(".", 1)
        setattr(attrgetter(parent)(layer), name, quantized)
        del linear
    # Unlike the existing QuIP#/QTIP path, reconstruct only after all seven replacements.
    reconstruction(layer, str(index), original_dtype)
    if reconstruction.distributed.is_primary:
        for attribute, short, *_ in PROJECTIONS:
            quantized = attrgetter(attribute)(layer)
            saved = saved_projections[short]
            # Preserve the released checkpoint's FP16 SU/SV storage, not QuIP#'s FP32.
            saved["SU"] = quantized.SU.detach().half().cpu()
            saved["SV"] = quantized.SV.detach().half().cpu()
            utils.atomic_torch_save(saved, config.checkpoints / f"{index}_{short}.pt")
    layer.cpu()


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
