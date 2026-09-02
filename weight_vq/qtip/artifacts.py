"""QTIP Hessian and dense checkpoint artifact contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lib.codebook.bitshift import bitshift_codebook
from lib.utils.matmul_had import matmul_hadU

from weight_vq import utils

PROJECTIONS = (
    ("self_attn.v_proj", "v", "qkv", "col"),
    ("self_attn.q_proj", "q", "qkv", "col"),
    ("self_attn.k_proj", "k", "qkv", "col"),
    ("self_attn.o_proj", "o", "o", "row"),
    ("mlp.up_proj", "up", "up", "col"),
    ("mlp.gate_proj", "gate", "up", "col"),
    ("mlp.down_proj", "down", "down", "row"),
)
HESSIAN_NAMES = ("qkv", "o", "up", "down")


def quant_order() -> list[tuple[str, str, str, str, str]]:
    return [
        (attribute, short, hessian, short, parallel)
        for attribute, short, hessian, parallel in PROJECTIONS
    ]


def projection_paths(layer_index: int, checkpoint_dir: str | Path) -> tuple[Path, ...]:
    checkpoint_dir = Path(checkpoint_dir)
    return tuple(
        checkpoint_dir / f"{layer_index}_{short}.pt"
        for _attribute, short, _hessian, _parallel in PROJECTIONS
    )


def layer_complete(layer_index: int, checkpoint_dir: str | Path) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    return (checkpoint_dir / f"{layer_index}.complete").is_file() and all(
        path.is_file() for path in projection_paths(layer_index, checkpoint_dir)
    )


def hessian_complete(layer_index: int, hessian_dir: str | Path) -> bool:
    hessian_dir = Path(hessian_dir)
    return all((hessian_dir / f"{layer_index}_{name}.pt").is_file() for name in HESSIAN_NAMES)


def atomic_write_completion(layer_index: int, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    missing = [path for path in projection_paths(layer_index, checkpoint_dir) if not path.is_file()]
    if missing:
        raise RuntimeError(f"layer {layer_index} is missing QTIP checkpoints: {missing}")
    marker = checkpoint_dir / f"{layer_index}.complete"
    utils.atomic_write_text(marker, "complete\n")


def reconstruct_dense_weight(saved: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    """Decode one native bitshift checkpoint into its dense QDQ weight."""
    if int(saved.get("rcp", 0)) != 0:
        raise NotImplementedError("dense QTIP export does not use TP packing")
    quant = config.get("quantization", config)
    su = saved["SU"].float()
    sv = saved["SV"].float() * saved["Wscale"].float()
    out_features, in_features = len(sv), len(su)
    codebook = bitshift_codebook(
        L=quant["L"],
        K=quant["K"],
        V=quant["V"],
        tlut_bits=quant["tlut_bits"],
        decode_mode=quant["decode_mode"],
        tlut=saved.get("tlut"),
    )
    trellis = codebook.unpack_trellis(saved["trellis"], quant["td_x"] * quant["td_y"])
    raw = (
        codebook.recons(trellis)
        .transpose(0, 1)
        .transpose(1, 2)
        .reshape(
            out_features // quant["td_x"],
            in_features // quant["td_y"],
            quant["td_x"],
            quant["td_y"],
        )
        .transpose(1, 2)
        .reshape(out_features, in_features)
    )
    transformed = matmul_hadU(matmul_hadU(raw.float() / 32).T).T
    return ((transformed * su) * (sv * 32).unsqueeze(1)).contiguous()
