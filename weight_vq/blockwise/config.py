"""Shared monolithic configuration for blockwise language-model quantization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BlockwiseConfig:
    model: Path
    dataset: Path
    output_dir: Path
    devices: list[int]

    seed: int = 0
    layer_start: int = 0
    layer_end: int | None = None
    num_cpu_threads: int = 8
    model_dtype: str = "bfloat16"

    hessian_samples: int = 4096
    hessian_sequence_length: int = 8192
    train_samples: int = 256
    validation_samples: int = 128
    sequence_length: int = 2048
    hessian_chunk_size: int = 256
    forward_microbatch_size: int = 1
    reconstruction_microbatch_size: int = 1
    replay_device: Literal["cpu", "cuda"] = "cuda"

    reconstruction_epochs: int = 5
    learning_rate: float = 5e-5
    susv_learning_rate: float = 5e-4
    batch_size: int = 8
    gradient_checkpointing: bool = False

    @property
    def calibration(self) -> Path:
        return self.output_dir / "calibration.pt"

    @property
    def hessians(self) -> Path:
        return self.output_dir / "hessians"

    @property
    def checkpoints(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def exported_model(self) -> Path:
        return self.output_dir / "exported_model"

    def calibration_recipe(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset": str(self.dataset.resolve()),
            "model": str(self.model.resolve()),
            "seed": self.seed,
            "split": "train",
            "chat_template": False,
            "hessian": {
                "shape": [self.hessian_samples, self.hessian_sequence_length],
                "sampling": "replacement-full-documents-concatenate-reshape",
            },
            "train": {"shape": [self.train_samples, self.sequence_length]},
            "validation": {"shape": [self.validation_samples, self.sequence_length]},
            "reconstruction_sampling": "replacement-long-document-prefixes-train-then-validation",
        }

    def hessian_recipe(self, end: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "calibration": self.calibration_recipe(),
            "layer_range": [self.layer_start, end],
            "model_dtype": self.model_dtype,
            "statistic": "uncentered-token-second-moment",
            "attention": "sdpa",
        }

    def recipe(self, *, model_family: str, layer_end: int) -> dict[str, object]:
        payload = asdict(self)
        if model_family == "qwen3_moe":
            payload["moe_protocol"] = {
                "coverage_denominator": "original-calibration-tokens-including-padding",
                "fallback": "identity-metric-with-method-codebook",
                "reconstruction": "attention-per-projection-experts-gate-up-down-groups",
                "router": "unquantized",
            }
        else:
            # The MoE-only parameter does not invalidate existing dense checkpoints.
            payload.pop("min_expert_coverage", None)
        for key in ("model", "dataset"):
            payload[key] = str(getattr(self, key).resolve())
        for key in (
            "output_dir",
            "devices",
            "hessian_chunk_size",
            "forward_microbatch_size",
            "reconstruction_microbatch_size",
            "replay_device",
            "num_cpu_threads",
            "layer_start",
            "layer_end",
        ):
            payload.pop(key)
        return {
            "model_family": model_family,
            "layer_range": [self.layer_start, layer_end],
            "calibration_protocol": self.calibration_recipe(),
            **payload,
        }
