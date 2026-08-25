"""Typed configuration contracts for weight-vq commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    path: Path
    revision: str


@dataclass(frozen=True)
class CalibrationConfig:
    tokenizer: TokenizerConfig
    dataset_cache: Path
    output: Path
    seed: int
    train_samples: int
    validation_samples: int
    sequence_length: int
    force: bool = False
    expected_train_sha256: str | None = None
    expected_validation_sha256: str | None = None


@dataclass(frozen=True)
class PendingPipelineConfig:
    """Temporary common shape replaced by each verified method adapter."""

    model: Path
    calibration: Path
    output: Path
