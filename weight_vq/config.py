"""Typed configuration contracts used by the initial package framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    revision: str | None
    split: str


@dataclass(frozen=True)
class TokenizerConfig:
    model: Path


@dataclass(frozen=True)
class CalibrationConfig:
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    output: Path
    manifest: Path
    seed: int
    samples: int
    sequence_length: int


@dataclass(frozen=True)
class PendingPipelineConfig:
    """Temporary common shape replaced by each verified method adapter."""

    model: Path
    calibration: Path
    output: Path
