"""Native language backbones and private Qwen3 MoE work layers."""

from typing import Any

import torch
from torch import nn


def language_backbone(model: Any) -> Any:
    if model.config.model_type == "mistral3":
        return model.model.language_model
    return model.model


def layer_expert_count(config: Any, index: int) -> int:
    if (
        config.model_type == "qwen3_moe"
        and index not in config.mlp_only_layers
        and (index + 1) % config.decoder_sparse_step == 0
    ):
        return config.num_experts
    return 0


class LinearExperts(nn.ModuleList):
    """Private work-layer layout; the native block still selects and weights routes."""

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros_like(hidden_states)
        # Match the native eager expert dispatch, exposing Linear calls for statistics/FT.
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=len(self)).permute(2, 1, 0)
            hits = mask.sum(dim=(-1, -2)).nonzero().flatten().tolist()
        for index in hits:
            slot, token = torch.where(mask[index])
            output = self[index](hidden_states[token]) * top_k_weights[token, slot, None]
            result.index_add_(0, token, output.to(result.dtype))
        return result


def split_work_experts(layer: Any) -> None:
    """Call only on a private decoder copy, never on the shared teacher."""
    experts = getattr(layer.mlp, "experts", None)
    if experts is None:
        return
    # Transformers 5's fused weights have no Linear replacement interface.
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeMLP

    if not isinstance(experts, Qwen3MoeExperts):
        return
    fused: Any = experts
    modules = LinearExperts()
    for index in range(fused.num_experts):
        with torch.device("meta"):
            expert = Qwen3MoeMLP(fused.config, intermediate_size=fused.intermediate_dim)
        gate, up = fused.gate_up_proj[index].chunk(2, dim=0)
        expert.gate_proj.weight = nn.Parameter(gate, requires_grad=gate.requires_grad)
        expert.up_proj.weight = nn.Parameter(up, requires_grad=up.requires_grad)
        expert.down_proj.weight = nn.Parameter(
            fused.down_proj[index], requires_grad=fused.down_proj.requires_grad
        )
        modules.append(expert)
    layer.mlp.experts = modules


def projection_weight(layer: Any, attribute: str) -> torch.Tensor:
    """Access a dense projection in native HF storage, including fused experts."""
    if attribute.startswith("mlp.experts."):
        _, _, expert, projection = attribute.split(".")
        experts = layer.mlp.experts
        if projection == "down_proj":
            return experts.down_proj[int(expert)]
        gate, up = experts.gate_up_proj[int(expert)].chunk(2, dim=0)
        return gate if projection == "gate_proj" else up
    return layer.get_parameter(f"{attribute}.weight")
