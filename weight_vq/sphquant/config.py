"""Typed public configuration for the SPHQuant pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SPHQuantConfig:
    """SPHQuant: run optimizes, exports, and validates with one shared configuration.

    Args:
        model: Local Hugging Face model directory.
        calibration: Tokenized train/validation artifact from wvq data calibration,
            with its companion metadata JSON file.
        output_dir: Root for checkpoints/<model.name>/ and exported_model/.
        device: Physical CUDA device index for single-GPU layerwise reconstruction.
        sample_size: Maximum subvectors sampled for spherical codebook initialization.
        kmeans_iterations: Spherical k-means iterations during initialization.
        codebook_bits: Direction-index bits; radius bits equal 10 minus this value.
            Six selects the published 64-direction, 4-bit-radius split.
        epochs: Reconstruction epochs per decoder layer, using quantized student
            inputs and full-precision teacher targets.
        learning_rate: Learning rate for radii and codebook angles.
        batch_size: Calibration sequences per reconstruction microbatch.
        gradient_accumulation_steps: Microbatches accumulated per optimizer step.
        cache_shards: Activation-cache shards; more shards reduce GPU memory use.
        resume: Reuse complete layer checkpoints; incomplete layers are rebuilt.
        checkpointing: Enable gradient checkpointing during reconstruction.
        seed: Initialization seed.
        max_layers: Number of leading decoder layers to quantize; null selects all.
        calibration_limit: Cap calibration sequences for smoke tests; null uses all.
        use_wandb: Log reconstruction metrics to Weights & Biases.
    """

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
