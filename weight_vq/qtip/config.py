"""QTIP algorithm parameters over the shared blockwise configuration."""

from __future__ import annotations

from dataclasses import dataclass

from weight_vq.blockwise.config import BlockwiseConfig


@dataclass(frozen=True)
class QTIPConfig(BlockwiseConfig):
    """QTIP blockwise quantization from a cached Arrow dataset.

    Args:
        packed_inference_kernel: Quantize in upstream MMA/trellis order. False
            preserves the paper's QDQ recipe; changing it requires new checkpoints.
        export_compressed: Export Qwen3 packed HF weights instead of dense QDQ.
            Requires packed_inference_kernel; does not change quantization.
        min_expert_coverage: Minimum fraction of original calibration tokens needed
            to use an expert's measured Hessian. Insufficient coverage uses an identity metric.
    """

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
    # Fraction of original calibration tokens needed for an expert Hessian.
    min_expert_coverage: float = 0.0

    packed_inference_kernel: bool = False
    export_compressed: bool = False

    # Tensor-parallel packing remains outside this pipeline.
    split_for_tp = False
    tp_rank = 8

    def __post_init__(self) -> None:
        if self.export_compressed and not self.packed_inference_kernel:
            raise ValueError("export_compressed requires packed_inference_kernel=True")
        if self.export_compressed and (
            self.L,
            self.K,
            self.V,
            self.tlut_bits,
            self.td_x,
            self.td_y,
            self.decode_mode,
        ) != (16, 2, 2, 9, 16, 16, "quantlut_sym"):
            raise ValueError(
                "packed QTIP inference supports only 2-bit L16/V2/9-bit LUT/16x16 quantlut_sym"
            )

    def recipe(self, *, model_family: str, layer_end: int) -> dict[str, object]:
        shared = super().recipe(model_family=model_family, layer_end=layer_end)
        shared.pop("export_compressed")
        return {
            "schema_version": 4,
            "format": "qtip-blockwise-full-state-dense-qdq",
            "lut_learning_rate_policy": "ordinary-group-migration-choice",
            **shared,
        }
