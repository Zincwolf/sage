"""Padding-aware QTIP Hessian collection."""

from __future__ import annotations

from operator import attrgetter
from typing import cast

import torch
from torch import nn


class MaskedHessianCollector:
    """Accumulate E[x x.T] over valid sequence positions."""

    def __init__(self, module: nn.Linear, *, include_padding: bool = False) -> None:
        self.size = module.in_features
        self.include_padding = include_padding
        self.matrix = torch.zeros(
            self.size, self.size, dtype=torch.float32, device=module.weight.device
        )
        self.count = 0
        self._position_mask: torch.Tensor | None = None
        self._handle = module.register_forward_pre_hook(self._hook)

    def set_position_mask(self, mask: torch.Tensor) -> None:
        if mask.ndim != 2:
            raise ValueError(f"Expected a [batch, sequence] mask, got {tuple(mask.shape)}")
        self._position_mask = mask.detach().to(device=self.matrix.device, dtype=torch.bool)

    @torch.no_grad()
    def _hook(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        values = inputs[0]
        if values.shape[-1] != self.size:
            raise RuntimeError(f"Projection input width {values.shape[-1]} != {self.size}")
        flattened = values.reshape(-1, self.size).float()
        if not self.include_padding:
            if self._position_mask is None:
                raise RuntimeError("Position mask must be set before every layer forward")
            flat_mask = self._position_mask.reshape(-1)
            if flat_mask.numel() != flattened.shape[0]:
                raise RuntimeError(
                    "Projection input and position mask are misaligned: "
                    f"{flattened.shape[0]} rows vs {flat_mask.numel()} mask entries"
                )
            flattened = flattened[flat_mask]
        if flattened.numel() == 0:
            return
        self.matrix.addmm_(flattened.T, flattened)
        self.count += flattened.shape[0]

    def close(self) -> dict[str, torch.Tensor | int | str]:
        self._handle.remove()
        if self.count == 0:
            raise RuntimeError("Hessian collector observed no valid inputs")
        matrix = (self.matrix / self.count).cpu()
        rows, columns = torch.tril_indices(self.size, self.size)
        flattened = matrix[rows, columns].contiguous()
        self.matrix = torch.empty(0)
        return {
            "flatH": flattened,
            "n": self.size,
            "ct": self.count,
            "position_mask": "all" if self.include_padding else "attention_mask",
        }


def collector_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    return {
        "qkv": cast(nn.Linear, attrgetter("self_attn.q_proj")(layer)),
        "o": cast(nn.Linear, attrgetter("self_attn.o_proj")(layer)),
        "up": cast(nn.Linear, attrgetter("mlp.up_proj")(layer)),
        "down": cast(nn.Linear, attrgetter("mlp.down_proj")(layer)),
    }


class LayerHessianCollectors:
    def __init__(self, layer: nn.Module, *, include_padding: bool = False) -> None:
        self.collectors = {
            name: MaskedHessianCollector(module, include_padding=include_padding)
            for name, module in collector_modules(layer).items()
        }

    def set_position_mask(self, mask: torch.Tensor) -> None:
        for collector in self.collectors.values():
            collector.set_position_mask(mask)

    def close(self) -> dict[str, dict[str, torch.Tensor | int | str]]:
        return {name: collector.close() for name, collector in self.collectors.items()}
