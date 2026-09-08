"""QTIP algorithm parameters over the shared blockwise configuration."""

from __future__ import annotations

from dataclasses import dataclass

from weight_vq.blockwise.config import BlockwiseConfig


@dataclass(frozen=True)
class QTIPConfig(BlockwiseConfig):
    td_x: int = 16
    td_y: int = 16
    L: int = 16
    K: int = 2
    V: int = 2
    tlut_bits: int = 9
    decode_mode: str = "quantlut_sym"
    sigma_reg: float = 0.01
    scale_override: float = 0.9
    use_fp64: bool = False
    train_lut: bool = True

    # The upstream interface also supports packed TP inference, outside dense QDQ.
    packed_inference_kernel = False
    split_for_tp = False
    tp_rank = 8

    def recipe(self, *, model_family: str, layer_end: int) -> dict[str, object]:
        return {
            "schema_version": 4,
            "format": "qtip-blockwise-full-state-dense-qdq",
            "lut_learning_rate_policy": "ordinary-group-migration-choice",
            **super().recipe(model_family=model_family, layer_end=layer_end),
        }
