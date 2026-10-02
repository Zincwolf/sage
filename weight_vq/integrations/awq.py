"""Runtime-only HF backend for standard Qwen3 LLM_AWQ checkpoints.

Assign LLMAWQConfig(**config.quantization_config).to_dict() to an
AutoConfig's quantization_config before passing it to AutoModelForCausalLM.from_pretrained.
This uses HF's normal weight loader without overriding its built-in AWQ registry
or importing GPTQModel.
The source checkpoint remains standard AWQ; this runtime config is not an export.
"""

import torch
from torch import nn
from transformers import AwqConfig
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizers_utils import should_convert_module

from weight_vq.inference.awq import AWQLinear


class LLMAWQConfig(AwqConfig):
    """Select the frozen W4G128 quantization_new kernel, not an auto-selected backend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.bits != 4 or self.group_size != 128 or self.format != "llm-awq":
            raise ValueError("WVQ AWQ inference requires bits=4, group_size=128, format=llm-awq")
        self.quant_method = "wvq-awq"  # pyright: ignore[reportAttributeAccessIssue]


class LLMAWQQuantizer(HfQuantizer):
    requires_calibration = True
    quantization_config: LLMAWQConfig

    def _process_model_before_weight_loading(self, model, **kwargs):
        if model.config.model_type != "qwen3":
            raise ValueError("WVQ AWQ inference currently supports Qwen3 dense decoders")
        skipped = self.get_modules_to_not_convert(
            model,
            self.quantization_config.modules_to_not_convert,
            model._keep_in_fp32_modules,
            add_default_skips=True,
        )
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and should_convert_module(name, skipped):
                with torch.device("meta"):
                    replacement = AWQLinear(module.in_features, module.out_features, module.bias)
                model.set_submodule(name, replacement)

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    def is_serializable(self) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        # Avoid exporting this runtime selector instead of the standard AWQ config.
        return False

    @property
    def is_trainable(self) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return False

    @property
    def is_compileable(self):
        return True


register_quantization_config("wvq-awq")(LLMAWQConfig)
register_quantizer("wvq-awq")(LLMAWQQuantizer)
