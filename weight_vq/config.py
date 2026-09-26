"""Typed configuration contracts for weight-vq commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer identity and local artifact.

    Args:
        name: Hugging Face tokenizer identifier for this calibration configuration.
        path: Local directory containing the tokenizer files for the target model.
        revision: Revision passed to the Hugging Face tokenizer loader.
    """

    name: str
    path: Path
    revision: str


@dataclass(frozen=True)
class CalibrationConfig:
    """Build shared SPHQuant/SAGE text calibration from WikiText-2, C4, and RedPajama.

    Args:
        tokenizer: Identity and local path of the target model's tokenizer.
        dataset_cache: Existing Hugging Face datasets cache directory. Missing
            datasets may be downloaded when this command runs.
        output: Destination .pt file containing train and validation token tensors.
            A companion metadata JSON file is written alongside it.
        seed: Seed for dataset shuffling and mixing.
        train_samples: Number of training sequences.
        validation_samples: Number of held-out validation sequences.
        sequence_length: Tokens per sequence.
        force: Overwrite existing calibration tensors and metadata.
        expected_sha256: Optional expected SHA-256 of the tensor inventory in the manifest.
    """

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
