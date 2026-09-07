"""Compact execution context and layer replay for QTIP."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from accelerate.utils import send_to_device
from torch import nn
from transformers import __version__ as transformers_version
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from weight_vq.qtip.calibration import CocoCalibrationBatches, local_batch_slice
from weight_vq.qtip.config import QTIPConfig

ReplayPlacement = Literal["cpu", "cuda"]


class PositionMaskCollector(Protocol):
    def set_position_mask(self, mask: torch.Tensor) -> None: ...


def _hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    return output.last_hidden_state


def _cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu()


def _store(
    value: torch.Tensor, placement: ReplayPlacement, device: torch.device | str
) -> torch.Tensor:
    return value.detach().to(device if placement == "cuda" else "cpu")


def _prefetch(
    groups: tuple[list[torch.Tensor], ...],
    indices: Iterable[int],
    *,
    placement: ReplayPlacement,
    device: torch.device | str,
) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
    target = torch.device(device)
    order = iter(indices)
    try:
        first = next(order)
    except StopIteration:
        return
    if placement == "cuda" or target.type != "cuda":
        yield first, tuple(group[first].to(target) for group in groups)
        for index in order:
            yield index, tuple(group[index].to(target) for group in groups)
        return

    stream = torch.cuda.Stream(device=target)
    buffers: list[list[torch.Tensor | None]] = [
        [None for _group in groups],
        [None for _group in groups],
    ]
    slot_events: list[torch.cuda.Event | None] = [None, None]

    def enqueue(slot: int, index: int) -> tuple[int, tuple[torch.Tensor, ...], torch.cuda.Event]:
        previous = slot_events[slot]
        if previous is not None:
            previous.synchronize()
        staged: list[torch.Tensor] = []
        for group_index, group in enumerate(groups):
            source = group[index]
            buffer = buffers[slot][group_index]
            if buffer is None or buffer.shape != source.shape or buffer.dtype != source.dtype:
                buffer = torch.empty_like(source, device="cpu", pin_memory=True)
                buffers[slot][group_index] = buffer
            buffer.copy_(source)
            with torch.cuda.stream(stream):
                staged.append(buffer.to(target, non_blocking=True))
        event = torch.cuda.Event()
        event.record(stream)
        slot_events[slot] = event
        return index, tuple(staged), event

    current = enqueue(0, first)
    slot = 1
    while True:
        try:
            following = enqueue(slot, next(order))
        except StopIteration:
            following = None
        current[2].wait()
        yield current[0], current[1]
        if following is None:
            break
        current = following
        slot = 1 - slot


@dataclass
class BatchContext:
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    valid_mask: torch.Tensor
    visual_pos_mask: torch.Tensor | None = None
    deepstack_visual_embeds: list[torch.Tensor] | None = None


@dataclass
class ReplayState:
    family: Literal["qwen3", "qwen3_vl"]
    model_config: Any
    rotary_emb: nn.Module
    inputs: list[torch.Tensor]
    contexts: list[BatchContext]
    placement: ReplayPlacement = "cpu"

    def _layer_kwargs(
        self, hidden: torch.Tensor, context: BatchContext, layer: nn.Module
    ) -> dict[str, Any]:
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = context.position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        else:
            position_ids = position_ids.to(hidden.device)
        text_position_ids = position_ids[0] if self.family == "qwen3_vl" else position_ids
        attention_mask = (
            None if context.attention_mask is None else context.attention_mask.to(hidden.device)
        )
        mask_factory: Any = (
            create_sliding_window_causal_mask
            if getattr(layer, "attention_type", None) == "sliding_attention"
            else create_causal_mask
        )
        mask_kwargs = {
            "config": self.model_config,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        if int(transformers_version.split(".", 1)[0]) >= 5:
            mask_kwargs["inputs_embeds"] = hidden
        else:
            mask_kwargs["input_embeds"] = hidden
            mask_kwargs["cache_position"] = cache_position
        causal_mask = mask_factory(**mask_kwargs)
        position_embeddings = self.rotary_emb.to(hidden.device)(hidden, position_ids)
        return {
            "attention_mask": causal_mask,
            "position_ids": text_position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }

    def _forward_device(
        self, layer: nn.Module, hidden: torch.Tensor, batch_index: int
    ) -> torch.Tensor:
        kwargs = self._layer_kwargs(hidden, self.contexts[batch_index], layer)
        return _hidden(layer(hidden, **kwargs))

    def _forward(
        self, layer: nn.Module, source: torch.Tensor, batch_index: int, device: torch.device | str
    ) -> torch.Tensor:
        return self._forward_device(layer, source.to(device), batch_index)

    def _apply_deepstack(
        self, output: torch.Tensor, context: BatchContext, layer_index: int
    ) -> torch.Tensor:
        embeds = context.deepstack_visual_embeds
        if embeds is None or layer_index >= len(embeds):
            return output
        mask = cast(torch.Tensor, context.visual_pos_mask).to(output.device, dtype=torch.bool)
        advanced = output.clone()
        advanced[mask] += embeds[layer_index].to(output.device, output.dtype)
        if layer_index == len(embeds) - 1:
            context.deepstack_visual_embeds = None
            context.visual_pos_mask = None
        return advanced

    @torch.no_grad()
    def forward_original(
        self,
        layer: nn.Module,
        layer_index: int,
        *,
        device: torch.device | str,
        collectors: PositionMaskCollector | None = None,
        autocast_dtype: torch.dtype | None = None,
        retain_layer: bool = True,
    ) -> LayerReplay | None:
        layer_inputs = self.inputs
        outputs: list[torch.Tensor] = []
        autocast = (
            torch.autocast(torch.device(device).type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with autocast:
            for batch_index, (hidden,) in _prefetch(
                (layer_inputs,), range(len(layer_inputs)), placement=self.placement, device=device
            ):
                if collectors is not None:
                    collectors.set_position_mask(self.contexts[batch_index].valid_mask)
                output = self._forward_device(layer, hidden, batch_index)
                outputs.append(_store(output, self.placement, device))
        self.inputs = [
            self._apply_deepstack(output, context, layer_index)
            for output, context in zip(outputs, self.contexts, strict=True)
        ]
        if not retain_layer:
            return None
        return LayerReplay(self, layer_index, layer_inputs, outputs)


@dataclass
class LayerReplay:
    state: ReplayState
    layer_index: int
    inputs: list[torch.Tensor]
    targets: list[torch.Tensor]

    def __len__(self) -> int:
        return len(self.inputs)

    def forward(
        self, layer: nn.Module, batch_index: int, device: torch.device | str
    ) -> torch.Tensor:
        return self.state._forward(layer, self.inputs[batch_index], batch_index, device)

    def loss_parts(
        self, layer: nn.Module, batch_index: int, device: torch.device | str, *, mask_padding: bool
    ) -> tuple[torch.Tensor, int]:
        source = self.inputs[batch_index].to(device)
        target = self.targets[batch_index].to(device)
        return self._loss_parts(layer, batch_index, source, target, mask_padding=mask_padding)

    def _loss_parts(
        self,
        layer: nn.Module,
        batch_index: int,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        mask_padding: bool,
    ) -> tuple[torch.Tensor, int]:
        difference = self.state._forward_device(layer, source, batch_index).float() - target.float()
        if mask_padding:
            difference = difference[
                self.state.contexts[batch_index].valid_mask.to(
                    device=source.device, dtype=torch.bool
                )
            ]
        return difference.square().sum(), difference.numel()

    def iter_loss_parts(
        self,
        layer: nn.Module,
        indices: Iterable[int],
        device: torch.device | str,
        *,
        mask_padding: bool,
    ) -> Iterator[tuple[int, torch.Tensor, int]]:
        for batch_index, (source, target) in _prefetch(
            (self.inputs, self.targets), indices, placement=self.state.placement, device=device
        ):
            squared_error, count = self._loss_parts(
                layer, batch_index, source, target, mask_padding=mask_padding
            )
            yield batch_index, squared_error, count

    def loss(
        self, layer: nn.Module, batch_index: int, device: torch.device | str, *, mask_padding: bool
    ) -> torch.Tensor:
        squared_error, count = self.loss_parts(
            layer, batch_index, device, mask_padding=mask_padding
        )
        return squared_error / count

    @torch.no_grad()
    def local_totals(
        self, layer: nn.Module, device: torch.device | str, *, mask_padding: bool
    ) -> tuple[torch.Tensor, int]:
        was_training = layer.training
        layer.eval()
        squared_error = torch.zeros((), device=device, dtype=torch.float64)
        element_count = 0
        try:
            for _index, batch_error, batch_count in self.iter_loss_parts(
                layer, range(len(self)), device, mask_padding=mask_padding
            ):
                squared_error += batch_error
                element_count += batch_count
        finally:
            layer.train(was_training)
        return squared_error, element_count


def _capture_text(
    model: nn.Module,
    config: QTIPConfig,
    split: str,
    device: torch.device | str,
    rank: int,
    world_size: int,
) -> ReplayState:
    sample_count = config.train_samples if split == "train" else config.validation_samples
    samples = torch.load(config.calibration, map_location="cpu", weights_only=True)[split][
        :sample_count
    ]
    backbone = cast(Any, model).model
    embedding = backbone.embed_tokens.to(device)
    try:
        inputs = []
        for global_batch in samples.split(config.batch_size):
            local_batch = global_batch[local_batch_slice(len(global_batch), rank, world_size)]
            inputs.append(_store(embedding(local_batch.to(device)), config.replay_device, device))
    finally:
        embedding.cpu()
    contexts = [
        BatchContext(
            attention_mask=None,
            position_ids=None,
            valid_mask=torch.ones(hidden.shape[:2], dtype=torch.bool),
        )
        for hidden in inputs
    ]
    return ReplayState(
        "qwen3", backbone.config, backbone.rotary_emb, inputs, contexts, config.replay_device
    )


def _capture_vl(
    model: nn.Module,
    config: QTIPConfig,
    split: str,
    device: torch.device | str,
    rank: int,
    world_size: int,
) -> ReplayState:
    sample_count = config.train_samples if split == "train" else config.validation_samples
    source = CocoCalibrationBatches(
        config.calibration,
        model_path=config.model,
        split=split,
        batch_size=config.batch_size,
        limit=sample_count,
        rank=rank,
        world_size=world_size,
    )
    inputs: list[torch.Tensor] = []
    contexts: list[BatchContext] = []
    language_model = cast(Any, model).model.language_model
    vision = cast(Any, model).model.visual

    class CaptureComplete(Exception):
        pass

    def capture(_module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        inputs.append(_store(kwargs["inputs_embeds"], config.replay_device, device))
        deepstack = kwargs.get("deepstack_visual_embeds")
        contexts.append(
            BatchContext(
                attention_mask=_cpu(kwargs.get("attention_mask")),
                position_ids=_cpu(kwargs.get("position_ids")),
                valid_mask=current_valid_mask,
                visual_pos_mask=_cpu(kwargs.get("visual_pos_masks")),
                deepstack_visual_embeds=(
                    None if deepstack is None else [value.detach().cpu() for value in deepstack]
                ),
            )
        )
        raise CaptureComplete

    current_valid_mask = torch.empty(0, dtype=torch.bool)
    handle = language_model.register_forward_pre_hook(capture, with_kwargs=True)
    vision.to(device)
    language_model.embed_tokens.to(device)
    try:
        for batch in source:
            current_valid_mask = batch["attention_mask"].detach().cpu().to(torch.bool)
            with suppress(CaptureComplete):
                model(**send_to_device(batch, device), use_cache=False)
    finally:
        handle.remove()
        vision.cpu()
        language_model.embed_tokens.cpu()
    return ReplayState(
        "qwen3_vl",
        language_model.config,
        language_model.rotary_emb,
        inputs,
        contexts,
        config.replay_device,
    )


@torch.no_grad()
def capture_replay(
    model: nn.Module,
    config: QTIPConfig,
    *,
    family: Literal["qwen3", "qwen3_vl"],
    split: str,
    device: torch.device | str,
    rank: int = 0,
    world_size: int = 1,
) -> ReplayState:
    if family == "qwen3":
        return _capture_text(model, config, split, device, rank, world_size)
    return _capture_vl(model, config, split, device, rank, world_size)
