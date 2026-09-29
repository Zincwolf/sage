"""Register weight-vq's pre-packed format with native Transformers loading.

Import this module before AutoModelForCausalLM.from_pretrained. No custom model,
remote code, source checkpoint, or training package is needed during inference.
The first format supports Qwen3 dense decoders; CUDA is required for execution,
not for serialization or CPU inspection.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from weight_vq.inference.linear import (
    QTIPLinear,
    QuIPSharpLinear,
    SAGELinear,
    SPHQuantLinear,
    UniSVQLinear,
)

Method = Literal["sage", "sphquant", "qtip", "quip-sharp", "unisvq"]


@dataclass
class WVQConfig(QuantizationConfigMixin):
    """On-disk inference ABI, not a calibration or optimization recipe.

    Module dimensions come from the native architecture. Only SAGE's variable
    bit split/permutation and QuIP#'s trained FP32 scales need per-module metadata.
    Fixed codebooks, packing and transform conventions are defined by format_version.
    """

    method: Method = "sage"
    modules: dict[str, dict[str, int | bool]] = field(default_factory=dict)
    format_version: int = 1
    synthetic_seed: int | None = None
    # HF accepts registered strings, though its annotation only lists built-ins.
    quant_method: str = "wvq"  # pyright: ignore[reportIncompatibleVariableOverride]

    def __post_init__(self):
        if self.format_version != 1:
            raise ValueError(f"Unsupported weight-vq format version: {self.format_version}")


def packed_linear(
    method: Method,
    packed: dict[str, torch.Tensor],
    out_features: int,
    bias: torch.Tensor | None = None,
) -> nn.Module:
    """Shared module construction for export and HF skeleton replacement."""
    if method == "sage":
        return SAGELinear(packed, out_features, bias)
    if method == "sphquant":
        return SPHQuantLinear(packed, bias)
    constructor = {"qtip": QTIPLinear, "quip-sharp": QuIPSharpLinear, "unisvq": UniSVQLinear}[
        method
    ]
    return constructor(packed, bias, cached_dense_weight=False)


def prepare_for_inference(model: nn.Module, *, cached_dense_weight: bool = False) -> None:
    """Prepare derived buffers outside compile/capture; never serialize dense caches.

    Call again after moving a CPU-loaded model to CUDA if dense caching is wanted.
    The cache policy is runtime-only and independent of the checkpoint format.
    """
    for module in model.modules():
        if isinstance(module, (QTIPLinear, QuIPSharpLinear, UniSVQLinear)):
            module.cached_dense_weight = cached_dense_weight
            module.dense = None
            module.prepare_for_inference()
            if cached_dense_weight:
                module.warm_weight_cache()


class WVQQuantizer(HfQuantizer):
    requires_calibration = True
    quantization_config: WVQConfig

    def _process_model_before_weight_loading(self, model, **kwargs):
        if model.config.model_type != "qwen3":
            raise ValueError("weight-vq packed loading currently supports Qwen3 dense decoders")
        method = self.quantization_config.method
        for name, spec in self.quantization_config.modules.items():
            linear = model.get_submodule(name)
            k, n = linear.in_features, linear.out_features
            # Explicit ABI dtypes survive HF's generic dtype=auto loading. Build
            # only a meta skeleton: never allocate random or dense target weights.
            with torch.device("meta"):
                if method == "sage":
                    entries = 1 << int(spec["codebook_bits"])
                    tiles, groups = (n + 15) // 16, k // 128
                    packed = {
                        "packed_codes": torch.empty(tiles, groups, 16, 8, dtype=torch.int32),
                        "radius_codes": torch.empty(tiles, groups, 16, dtype=torch.int32),
                        "radius_grid": torch.empty(tiles, groups, 16, 2, dtype=torch.float16),
                        "codebook": torch.empty(4, entries, 2, dtype=torch.float16),
                        "permutation": torch.empty(k if spec["permuted"] else 0, dtype=torch.int32),
                    }
                elif method == "sphquant":
                    packed = {
                        "packed_codes": torch.empty(n, k // 8, dtype=torch.int16),
                        "radius_extra": torch.empty(n, k // 128, dtype=torch.int32),
                        "radius_scales": torch.empty(n, k // 128, dtype=torch.float16),
                        "radius_mins": torch.empty(n, k // 128, dtype=torch.float16),
                        "codebook": torch.empty(64, 8, dtype=torch.float16),
                    }
                elif method == "qtip":
                    packed = {
                        "trellis": torch.empty(n * k // 256, 32, dtype=torch.int16),
                        "SU": torch.empty(k, dtype=torch.float16),
                        "SV": torch.empty(n, dtype=torch.float32),
                        "tlut": torch.empty(512, 2, dtype=torch.float16),
                    }
                elif method == "quip-sharp":
                    scale_dtype = torch.float32 if spec.get("fp32_scales", False) else torch.float16
                    packed = {
                        "Qidxs": torch.empty(n, k // 32, dtype=torch.int64),
                        "SU": torch.empty(k, dtype=scale_dtype),
                        "SV": torch.empty(n, dtype=scale_dtype),
                    }
                elif method == "unisvq":
                    packed = {
                        "Qidxs": torch.empty(n, k // 4, dtype=torch.uint8),
                        "SU": torch.empty(k, dtype=torch.bfloat16),
                        "SV": torch.empty(n, dtype=torch.bfloat16),
                        "cb_scale": torch.empty((), dtype=torch.bfloat16),
                        "cb_zero": torch.empty((), dtype=torch.bfloat16),
                    }
                else:
                    raise ValueError(f"Unsupported weight-vq method: {method}")
                replacement = packed_linear(method, packed, n, linear.bias)
            model.set_submodule(name, replacement)

    def _process_model_after_weight_loading(self, model, **kwargs):
        # HF materializes non-persistent meta buffers with empty storage; rebuild
        # mathematical constants here rather than saving duplicate derived data.
        prepare_for_inference(model)
        return model

    # HF's abstract declarations infer None, but the documented API is boolean.
    def is_serializable(self) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return True

    @property
    def is_trainable(self) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return False

    @property
    def is_compileable(self):
        return True


# Registration decorators in HF erase subclass types; retain the concrete names.
register_quantization_config("wvq")(WVQConfig)
register_quantizer("wvq")(WVQQuantizer)
