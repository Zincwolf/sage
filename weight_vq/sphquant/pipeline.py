"""SPHQuant stage adapters."""

from __future__ import annotations

from weight_vq._pending import stage
from weight_vq.config import PendingPipelineConfig


def optimize(config: PendingPipelineConfig) -> None:
    stage("sphquant", "optimize", config)


def export(config: PendingPipelineConfig) -> None:
    stage("sphquant", "export", config)


def validate(config: PendingPipelineConfig) -> None:
    stage("sphquant", "validate", config)
