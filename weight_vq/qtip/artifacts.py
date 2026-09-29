"""Decode QTIP checkpoints into dense QDQ weights."""

from __future__ import annotations

from typing import Any

import torch
from lib.codebook.bitshift import bitshift_codebook
from lib.utils.matmul_had import matmul_hadU


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
    packed = saved["trellis"]
    kernel_layout = quant.get("packed_inference_kernel", False)
    if kernel_layout:
        # Undo only the storage swizzle; these are already MMA-order trellis
        # states produced at quantization time, not converted legacy states.
        packed = (
            packed.view(torch.uint8)
            .reshape(out_features // 32, in_features // 32, 32, 2, 2, quant["K"])
            .permute(0, 4, 1, 3, 2, 5)
            .flip((-1,))
            .reshape(-1, 2)
            .flip((-1,))
            .contiguous()
            .view(torch.int16)
            .reshape(-1, 16 * quant["K"])
        )
    trellis = codebook.unpack_trellis(packed, quant["td_x"] * quant["td_y"])
    tiles = codebook.recons(trellis).transpose(0, 1).transpose(1, 2)
    if kernel_layout:
        order = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
        tiles = tiles.reshape(-1, 256)[:, torch.argsort(order)]
    raw = (
        tiles.reshape(
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
