"""Shared monolithic configuration for blockwise language-model quantization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from transformers import AutoConfig

from weight_vq.blockwise.artifacts import projection_layout


@dataclass(frozen=True)
class BlockwiseConfig:
    """Shared calibration, Hessian, quantization, export, and validation configuration.

    Args:
        model: Local Hugging Face model directory.
        dataset: Cached Hugging Face Arrow training dataset directory, containing
            dataset_info.json and *-train-*-of-*.arrow shards.
        output_dir: Root for calibration.pt, hessians/, checkpoints/, and exported_model/.
        devices: CUDA device indices used by the blockwise pipeline.
        seed: Seed for calibration sampling and quantization.
        layer_start: First decoder layer to quantize, inclusive.
        layer_end: Last decoder layer to quantize, exclusive; null selects model depth.
        num_cpu_threads: CPU threads per worker.
        model_dtype: Floating-point dtype used to load the source model.
        hessian_samples: Sequences used to collect activation statistics.
        hessian_sequence_length: Tokens per activation-statistics sequence.
        train_samples: Sequences used for layer reconstruction.
        validation_samples: Held-out sequences used for reconstruction validation.
        sequence_length: Tokens per reconstruction sequence.
        hessian_chunk_size: Calibration sequences processed per statistics chunk.
        forward_microbatch_size: Sequences per teacher forward pass.
        reconstruction_microbatch_size: Sequences per reconstruction replay microbatch.
        replay_device: Storage device for cached replay tensors.
        reconstruction_epochs: Reconstruction epochs using dense teacher inputs and targets.
        learning_rate: Learning rate for ordinary reconstruction parameters.
        susv_learning_rate: Learning rate for SU/SV scale parameters.
        batch_size: Reconstruction batch size.
        gradient_checkpointing: Enable gradient checkpointing during reconstruction.
    """

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
        if model_family == "qwen3_5":
            payload.update(
                projection_layout(
                    AutoConfig.from_pretrained(self.model, local_files_only=True), layer_end
                )
            )
            payload["recurrent_auxiliaries"] = "frozen-source-weights"
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
            "gradient_checkpointing",
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
