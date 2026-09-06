"""Typed public configuration for the SPHQuant pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SPHQuantConfig:
    """Shared configuration for optimize, export, and validate."""

    model: Path
    calibration: Path
    output_dir: Path
    device: int = 0
    sample_size: int = 65536
    kmeans_iterations: int = 20
    codebook_bits: int = 6
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    cache_shards: int = 8
    resume: bool = True
    checkpointing: bool = True
    seed: int = 0
    max_layers: int | None = None
    calibration_limit: int | None = None
    use_wandb: bool = False

    @property
    def checkpoints(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def exported_model(self) -> Path:
        return self.output_dir / "exported_model"
