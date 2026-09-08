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
