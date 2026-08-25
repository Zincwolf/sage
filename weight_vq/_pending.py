"""Helpers for pipeline stages that have not been migrated yet."""

from __future__ import annotations

from weight_vq.config import PendingPipelineConfig


def stage(method: str, name: str, config: PendingPipelineConfig) -> None:
    raise NotImplementedError(
        f"{method}/{name} has not been migrated yet (output: {config.output})"
    )
