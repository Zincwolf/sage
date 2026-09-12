"""Decoder Hessian collection, projection selection, and quantization metrics."""

from __future__ import annotations

from operator import attrgetter
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from weight_vq.blockwise.artifacts import projections
from weight_vq.blockwise.models import LinearExperts


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

    def close(
        self, distributed: Any | None = None, *, center: bool = False
    ) -> dict[str, torch.Tensor | int | str]:
        """Finalize a complete sample set for numerical tests and quantizer fixtures.

        The production pipeline uses close_statistics to merge raw chunks on disk.
        """
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


class LayerHessianCollectors:
    def __init__(
        self, layer: nn.Module, *, include_padding: bool = False, dtype: torch.dtype = torch.float32
    ) -> None:
        self.collectors = {
            name: MaskedHessianCollector(module, include_padding=include_padding, dtype=dtype)
            for name, module in collector_modules(layer).items()
        }

    def set_position_mask(self, mask: torch.Tensor) -> None:
        for collector in self.collectors.values():
            collector.set_position_mask(mask)

    def close(
        self, distributed: Any | None = None
    ) -> dict[str, dict[str, torch.Tensor | int | str]]:
        return {name: collector.close(distributed) for name, collector in self.collectors.items()}


def collector_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    experts = getattr(layer.mlp, "experts", None)
    count = len(experts) if isinstance(experts, LinearExperts) else 0
    # q/k/v and expert gate/up share inputs. Keep one collector per Hessian.
    modules = {}
    for attribute, _short, name, _parallel in projections(count):
        if attribute.rsplit(".", 1)[1] in ("q_proj", "o_proj", "up_proj", "down_proj"):
            modules[name] = cast(nn.Linear, attrgetter(attribute)(layer))
    return modules


def load_quantization_hessian(
    path: str | Path, min_expert_coverage: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    # All methods use the upstream symmetric unpacking convention.
    from lib.utils import flat_to_sym

    statistics = torch.load(path, map_location="cpu", weights_only=True)
    metadata = {}
    fallback = False
    if "total_tokens" in statistics:
        count = statistics["ct"]
        total = statistics["total_tokens"]
        coverage = count / total
        fallback = count == 0 or coverage < min_expert_coverage
        metadata = {
            "observed_tokens": count,
            "total_tokens": total,
            "coverage": coverage,
            "fallback": fallback,
            "metric": "identity" if fallback else "empirical",
        }
    if fallback:
        hessian = torch.eye(statistics["n"], dtype=statistics["flatH"].dtype)
    else:
        hessian = flat_to_sym(statistics["flatH"], statistics["n"])
        if "mu" in statistics:
            mu = statistics["mu"]
            hessian.add_(mu[:, None] * mu[None, :])
    return hessian, metadata


__all__ = ["LayerHessianCollectors", "MaskedHessianCollector", "collector_modules"]
