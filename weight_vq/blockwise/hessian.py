"""Decoder projection selection over the shared Hessian collectors."""

from __future__ import annotations

from operator import attrgetter
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from weight_vq.blockwise.artifacts import projections
from weight_vq.blockwise.models import LinearExperts
from weight_vq.hessian import LinearHessianCollectors, MaskedHessianCollector


class LayerHessianCollectors(LinearHessianCollectors):
    def __init__(
        self, layer: nn.Module, *, include_padding: bool = False, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__(collector_modules(layer), include_padding=include_padding, dtype=dtype)


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
    # Both methods use the upstream symmetric unpacking convention.
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


__all__ = [
    "LayerHessianCollectors",
    "LinearHessianCollectors",
    "MaskedHessianCollector",
    "collector_modules",
]
