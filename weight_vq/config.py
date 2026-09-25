"""Typed configuration contracts for weight-vq commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer identity and local artifact."""

    name: str
    path: Path
    revision: str


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration dataset generation configuration."""

    tokenizer: TokenizerConfig
    dataset_cache: Path
    output: Path
    seed: int
    train_samples: int
    validation_samples: int
    sequence_length: int
    force: bool = False
    expected_sha256: str | None = None


@dataclass(frozen=True)
class CalibrationMetadata:
    train_samples: int
    validation_samples: int
    sequence_length: int
