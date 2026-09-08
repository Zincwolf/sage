"""QuIP# W2 E8P configuration; statistics precision is method-owned."""

from __future__ import annotations

from dataclasses import dataclass

from weight_vq.blockwise.config import BlockwiseConfig


@dataclass(frozen=True)
class QuIPSharpConfig(BlockwiseConfig):
    hessian_samples: int = 6144
    hessian_sequence_length: int = 2048
    sigma_reg: float = 0.01
    scale_override: float = 0.9
    quip_tune_iters: int = 10
    use_fp64: bool = False

    # This adapter implements the paper's W2 Hadamard E8P path, without E2E/LoRA.
    codebook = "E8P12"
    lora_rank = 0
    incoh_mode = "had"
    rescale_WH = False
    resid_scale_override = -1
    no_use_buffered = False
    lowmem_ldlq = False

    def hessian_recipe(self, end: int) -> dict[str, object]:
        return {
            **super().hessian_recipe(end),
            "statistic": "centered-token-covariance-and-mean",
            "statistics_dtype": "float64",
        }

    def recipe(self, *, model_family: str, layer_end: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "format": "quip-sharp-blockwise-full-state-dense-qdq",
            "codebook": self.codebook,
            "hessian_protocol": self.hessian_recipe(layer_end),
            "hadamard": "upstream-with-512-block-fallback",
            **super().recipe(model_family=model_family, layer_end=layer_end),
        }
