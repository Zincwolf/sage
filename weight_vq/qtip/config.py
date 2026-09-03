"""Monolithic public configuration for QTIP pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class QTIPConfig:
    """Shared configuration for Hessian, quantize, export, and validate."""

    model: Path
    calibration: Path
    hessians: Path
    checkpoints: Path
    exported_model: Path

    devices: list[int]
    seed: int = 0
    layer_start: int = 0
    layer_end: int | None = None
    num_cpu_threads: int = 8
    model_dtype: str = "bfloat16"

    train_samples: int = 512
    validation_samples: int = 128
    batch_size: int = 16
    replay_device: Literal["cpu", "cuda"] = "cuda"
    include_padding_in_hessian: bool = False

    td_x: int = 16
    td_y: int = 16
    L: int = 16
    K: int = 2
    V: int = 2
    tlut_bits: int = 9
    decode_mode: str = "quantlut_sym"

    sigma_reg: float = 1e-2
    scale_override: float = 0.9
    use_fp64: bool = False

    reconstruction_epochs: int = 10
    learning_rate: float = 3e-6
    update_frequency: int = 1
    validation_frequency: int = 1
    early_stop: int = 1
    gradient_checkpointing: bool = False
    train_lut: bool = False
    mask_padding: bool = True

    packed_inference_kernel: bool = False
    split_for_tp: bool = False
    tp_rank: int = 8

    def recipe(self, *, model_family: str, layer_end: int) -> dict[str, object]:
        payload = asdict(self)
        payload["model"] = str(self.model.resolve())
        payload["calibration"] = str(self.calibration.resolve())
        for name in (
            "hessians",
            "checkpoints",
            "exported_model",
            "devices",
            "batch_size",
            "replay_device",
            "num_cpu_threads",
            "layer_start",
            "layer_end",
        ):
            payload.pop(name)
        payload["layer_range"] = [self.layer_start, layer_end]
        return {
            "schema_version": 3,
            "format": f"qtip-bitshift-trellis-{model_family.replace('_', '-')}-dense-qdq",
            "model_family": model_family,
            **payload,
        }
