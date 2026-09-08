"""Text-only decoder replay using the native HF RoPE and masking APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from weight_vq.blockwise.hessian import LayerHessianCollectors
from weight_vq.blockwise.models import language_backbone


@dataclass
class ReplayState:
    model_config: Any
    rotary_emb: nn.Module
    inputs: torch.Tensor
    microbatch_size: int

    def forward(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask_factory = (
            create_sliding_window_causal_mask
            if (
                getattr(layer, "attention_type", None) == "sliding_attention"
                or (
                    self.model_config.model_type == "ministral3"
                    and self.model_config.sliding_window is not None
                )
            )
            else create_causal_mask
        )
        mask = mask_factory(
            config=self.model_config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        embeddings = self.rotary_emb.to(hidden.device)(hidden, positions)
        return layer(
            hidden,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=embeddings,
            use_cache=False,
        )

    @torch.no_grad()
    def forward_original(
        self,
        layer: nn.Module,
        *,
        device: torch.device | str,
        collectors: LayerHessianCollectors | None = None,
    ) -> LayerReplay:
        inputs = self.inputs
        outputs = torch.empty_like(inputs)
        for start in range(0, len(inputs), self.microbatch_size):
            hidden = inputs[start : start + self.microbatch_size].to(device)
            if collectors is not None:
                collectors.set_position_mask(
                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                )
            outputs[start : start + len(hidden)].copy_(self.forward(layer, hidden))
        self.inputs = outputs
        return LayerReplay(self, inputs, outputs)


@dataclass
class LayerReplay:
    state: ReplayState
    inputs: torch.Tensor
    targets: torch.Tensor

    def loss_parts(
        self, layer: nn.Module, indices: torch.Tensor, device: torch.device | str
    ) -> tuple[torch.Tensor, int]:
        source = self.inputs[indices.to(self.inputs.device)].to(device)
        target = self.targets[indices.to(self.targets.device)].to(device)
        difference = self.state.forward(layer, source).float() - target.float()
        return difference.square().sum(), difference.numel()

    @torch.no_grad()
    def local_totals(
        self, layer: nn.Module, device: torch.device | str, microbatch_size: int
    ) -> tuple[torch.Tensor, int]:
        total = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        for indices in torch.arange(len(self.inputs)).split(microbatch_size):
            error, elements = self.loss_parts(layer, indices, device)
            total += error
            count += elements
        return total, count


@torch.no_grad()
def capture_replay(
    model: Any,
    tokens: torch.Tensor,
    *,
    device: torch.device | str,
    microbatch_size: int,
    placement: str = "cuda",
) -> ReplayState:
    backbone = language_backbone(model)
    embedding = backbone.embed_tokens.to(device)
    inputs = torch.empty(
        (*tokens.shape, embedding.embedding_dim),
        dtype=embedding.weight.dtype,
        device=device if placement == "cuda" else "cpu",
    )
    try:
        for start in range(0, len(tokens), microbatch_size):
            batch = tokens[start : start + microbatch_size].to(device)
            inputs[start : start + len(batch)].copy_(embedding(batch))
    finally:
        embedding.cpu()
    return ReplayState(backbone.config, backbone.rotary_emb, inputs, microbatch_size)
