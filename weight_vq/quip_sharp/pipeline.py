"""QuIP-Sharp stage adapters."""

from __future__ import annotations

from weight_vq._pending import stage
from weight_vq.config import PendingPipelineConfig


def hessian(config: PendingPipelineConfig) -> None:
    stage("quip-sharp", "hessian", config)


def quantize(config: PendingPipelineConfig) -> None:
    stage("quip-sharp", "quantize", config)


def export(config: PendingPipelineConfig) -> None:
    stage("quip-sharp", "export", config)


def validate(config: PendingPipelineConfig) -> None:
    stage("quip-sharp", "validate", config)
