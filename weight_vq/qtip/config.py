"""Typed public configuration for the QTIP Qwen3-VL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QTIPConfig:
    """Shared configuration for Hessian, quantize, export, and validate."""

    model: Path
    calibration: Path
    hessians: Path
    checkpoints: Path
    exported_model: Path
    devices: list[int]
    train_samples: int = 512
    validation_samples: int = 128
    batch_size: int = 16
    reconstruction_epochs: int = 10
    learning_rate: float = 3e-6
    seed: int = 0
    layer_start: int = 0
    layer_end: int | None = None
