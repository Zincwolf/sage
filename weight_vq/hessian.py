"""Distributed-friendly second-moment collectors for linear quantization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MaskedHessianCollector:
    """Accumulate an uncentered second moment for one linear projection."""

    def __init__(
        self,
        module: nn.Linear,
        *,
        include_padding: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.size = module.in_features
        self.include_padding = include_padding
        self.matrix = torch.zeros(self.size, self.size, dtype=dtype, device=module.weight.device)
        self.first_moment = torch.zeros(self.size, dtype=dtype, device=module.weight.device)
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
        flattened = values.reshape(-1, self.size).to(self.matrix.dtype)
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
        self.first_moment.add_(flattened.sum(dim=0))
        self.count += flattened.shape[0]

    def close_statistics(
        self, distributed: Any | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        self._handle.remove()
        count = self.count
        if distributed is not None:
            distributed.sum_(self.matrix)
            distributed.sum_(self.first_moment)
            count = int(distributed.global_count(count))
        matrix = self.matrix.cpu()
        first_moment = self.first_moment.cpu()
        self.matrix = torch.empty(0)
        self.first_moment = torch.empty(0)
        return matrix, first_moment, count

    def close_raw(self, distributed: Any | None = None) -> tuple[torch.Tensor, int]:
        matrix, _first_moment, count = self.close_statistics(distributed)
        return matrix, count

    def close(
        self, distributed: Any | None = None, *, center: bool = False
    ) -> dict[str, torch.Tensor | int | str]:
        matrix, first_moment, count = self.close_statistics(distributed)
        matrix.div_(count)
        if center:
            # Center the global second moment, not the unnormalized sum.
            first_moment.div_(count)
            matrix.sub_(first_moment[:, None] * first_moment[None, :])
        rows, columns = torch.tril_indices(self.size, self.size)
        flattened = matrix[rows, columns].contiguous()
        result: dict[str, torch.Tensor | int | str] = {
            "flatH": flattened,
            "n": self.size,
            "ct": count,
            "position_mask": "all" if self.include_padding else "attention_mask",
        }
        if center:
            result["mu"] = first_moment
        return result


class LinearHessianCollectors:
    """Collect Hessians for a caller-provided mapping of linear projections."""

    def __init__(
        self,
        modules: dict[str, nn.Linear],
        *,
        include_padding: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.collectors = {
            name: MaskedHessianCollector(module, include_padding=include_padding, dtype=dtype)
            for name, module in modules.items()
        }

    def set_position_mask(self, mask: torch.Tensor) -> None:
        for collector in self.collectors.values():
            collector.set_position_mask(mask)

    def close(
        self, distributed: Any | None = None
    ) -> dict[str, dict[str, torch.Tensor | int | str]]:
        return {name: collector.close(distributed) for name, collector in self.collectors.items()}
