"""Decoder projection selection over the shared Hessian collectors."""

from __future__ import annotations

from operator import attrgetter
from typing import cast

import torch
from torch import nn

from weight_vq.hessian import LinearHessianCollectors, MaskedHessianCollector


class LayerHessianCollectors(LinearHessianCollectors):
    def __init__(
        self, layer: nn.Module, *, include_padding: bool = False, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__(collector_modules(layer), include_padding=include_padding, dtype=dtype)


def collector_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    return {
        "qkv": cast(nn.Linear, attrgetter("self_attn.q_proj")(layer)),
        "o": cast(nn.Linear, attrgetter("self_attn.o_proj")(layer)),
        "up": cast(nn.Linear, attrgetter("mlp.up_proj")(layer)),
        "down": cast(nn.Linear, attrgetter("mlp.down_proj")(layer)),
    }


__all__ = [
    "LayerHessianCollectors",
    "LinearHessianCollectors",
    "MaskedHessianCollector",
    "collector_modules",
]
