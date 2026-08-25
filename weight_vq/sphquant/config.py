"""Typed public configuration for SPHQuant optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SPHQuantInitializationConfig:
    """Spherical codebook initialization controls."""

    sample_size: int = 65536
    kmeans_iterations: int = 20


@dataclass(frozen=True)
class SPHQuantRecoveryConfig:
    """Layerwise recovery controls."""

    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    cache_shards: int = 8
    resume: bool = True
    checkpointing: bool = True


@dataclass(frozen=True)
class SPHQuantOptimizeConfig:
    """Inputs and user-facing controls for the SPHQuant optimize stage."""

    model: Path
    calibration: Path
    output: Path
    initialization: SPHQuantInitializationConfig = field(
        default_factory=SPHQuantInitializationConfig
    )
    recovery: SPHQuantRecoveryConfig = field(default_factory=SPHQuantRecoveryConfig)
    seed: int = 0
    max_layers: int | None = None
    calibration_limit: int | None = None
    use_wandb: bool = False
