"""Actual defaults of upstream unisvq_qwen3_1.7B.sh and its PTQ parser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from weight_vq.blockwise.config import BlockwiseConfig


@dataclass(frozen=True)
class UniSVQConfig(BlockwiseConfig):
    hessian_samples: int = 1024
    hessian_sequence_length: int = 4096
    # Statistics accumulation/storage only; upstream quantizer precision is unchanged.
    hessian_dtype: Literal["float32", "float64"] = "float32"
    train_samples: int = 896
    validation_samples: int = 128
    sequence_length: int = 4096
    reconstruction_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    early_stop: int = 3
    sigma_reg: float = 0.01
    scale_override: float = 0.83
    quip_tune_iters: int = 10

    # Scope is the released W2 PTQ entry, not the paper's trainable affine grid.
    codebook = "identical"
    codebook_bit = 2

    def calibration_recipe(self) -> dict[str, object]:
        return {
            **super().calibration_recipe(),
            "schema_version": 2,
            "hessian": {
                "shape": [self.hessian_samples, self.hessian_sequence_length],
                "sampling": "replacement-truncated-documents-concatenate-reshape",
            },
            "reconstruction_sampling": "independent-truncated-documents-concatenate-reshape",
        }

    def hessian_recipe(self, end: int) -> dict[str, object]:
        return {
            **super().hessian_recipe(end),
            "statistic": "centered-token-covariance-and-mean",
            "statistics_dtype": self.hessian_dtype,
        }

    def recipe(self, *, model_family: str, layer_end: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "format": "unisvq-blockwise-full-state-dense-qdq",
            "upstream_revision": "449e8e4f317d2c539d7135c39f66817adce3cf4c",
            "codebook": self.codebook,
            "codebook_bit": self.codebook_bit,
            "hadamard": "upstream-128-block",
            "hessian_protocol": self.hessian_recipe(layer_end),
            "reconstruction": "whole-block-teacher-inputs-fp16-autocast-summed-microbatch-means",
            "saved_susv_dtype": "float16",
            "saved_norms": ["input_layernorm", "post_attention_layernorm"],
            **super().recipe(model_family=model_family, layer_end=layer_end),
        }
