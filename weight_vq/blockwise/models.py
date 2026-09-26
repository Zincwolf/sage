"""Native language backbones and private Qwen3 MoE work layers."""

from typing import Any

from paroquant.optim.moe import LinearExperts as LinearExperts
from paroquant.optim.moe import projection_weight as projection_weight
from paroquant.optim.moe import split_work_experts as split_work_experts


def language_backbone(model: Any) -> Any:
    if model.config.model_type in ("mistral3", "qwen3_5"):
        return model.model.language_model
    return model.model


def layer_attention_type(config: Any, index: int) -> str:
    text = getattr(config, "text_config", config)
    types = getattr(text, "layer_types", None)
    return types[index] if types is not None else "full_attention"


def layer_expert_count(config: Any, index: int) -> int:
    """Args:
    config: Huggingface AutoConfig
    index: 0-based decoder layer index
    """
    if (
        config.model_type == "qwen3_moe"
        and index not in config.mlp_only_layers
        and (index + 1) % config.decoder_sparse_step == 0
    ):
        return config.num_experts
    return 0
