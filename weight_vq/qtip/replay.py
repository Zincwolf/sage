"""Qwen3-VL capture and layer replay state for QTIP."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from accelerate.utils import send_to_device
from paroquant.optim.multimodal import CocoCalibrationBatches, apply_deepstack_context
from paroquant.optim.util import capture_layer_inputs_and_args
from torch import nn

from weight_vq.qtip.config import QTIPConfig


class PositionMaskCollector(Protocol):
    def set_position_mask(self, mask: torch.Tensor) -> None: ...


def _hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported decoder output: {type(output).__name__}")


@dataclass
class LayerReplay:
    state: ReplayState
    layer_index: int
    inputs: list[torch.Tensor]
    targets: list[torch.Tensor]
    retained_kwargs: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.inputs)

    def forward(
        self, layer: nn.Module, batch_index: int, device: torch.device | str
    ) -> torch.Tensor:
        kwargs = cast(
            dict[str, Any],
            send_to_device(
                deepcopy(self.state.kwargs_by_layer[self.layer_index][batch_index]), device
            ),
        )
        for key, value in self.retained_kwargs[batch_index].items():
            kwargs[key] = send_to_device(deepcopy(value), device)
        other_args = send_to_device(
            self.state.other_args_by_layer[self.layer_index][batch_index], device
        )
        return _hidden(layer(self.inputs[batch_index].to(device), *other_args, **kwargs))

    def loss(
        self, layer: nn.Module, batch_index: int, device: torch.device | str, *, mask_padding: bool
    ) -> torch.Tensor:
        difference = (
            self.forward(layer, batch_index, device).float()
            - self.targets[batch_index].to(device).float()
        )
        if mask_padding:
            difference = difference[
                self.state.valid_masks[batch_index].to(device=device, dtype=torch.bool)
            ]
        if difference.numel() == 0:
            raise ValueError("Cannot compute MSE without valid positions")
        return difference.square().mean()

    @torch.no_grad()
    def mean_loss(
        self, layer: nn.Module, device: torch.device | str, *, mask_padding: bool
    ) -> float:
        was_training = layer.training
        layer.eval()
        squared_error = torch.zeros((), device=device)
        element_count = 0
        try:
            for batch_index in range(len(self)):
                difference = (
                    self.forward(layer, batch_index, device).float()
                    - self.targets[batch_index].to(device).float()
                )
                if mask_padding:
                    difference = difference[
                        self.state.valid_masks[batch_index].to(device=device, dtype=torch.bool)
                    ]
                squared_error += difference.square().sum()
                element_count += difference.numel()
        finally:
            layer.train(was_training)
        if element_count == 0:
            raise ValueError("Replay contains no scored elements")
        return (squared_error / element_count).cpu().item()


@dataclass
class ReconstructionAdapter:
    train: LayerReplay
    validation: LayerReplay
    seed: int
    mask_padding: bool

    def iter_train(self, epoch: int) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        yield from torch.randperm(len(self.train), generator=generator).tolist()

    def loss(self, layer: nn.Module, batch_index: int, device: torch.device | str) -> torch.Tensor:
        return self.train.loss(layer, batch_index, device, mask_padding=self.mask_padding)

    def validation_loss(self, layer: nn.Module, device: torch.device | str) -> float:
        return self.validation.mean_loss(layer, device, mask_padding=self.mask_padding)

    @torch.no_grad()
    def materialize(self, layer: nn.Module, device: torch.device | str) -> None:
        self.train.forward(layer, 0, device)


@dataclass
class ReplayState:
    kwargs_by_layer: list[list[dict[str, Any]]]
    other_args_by_layer: list[list[tuple[torch.Tensor, ...]]]
    deepstack_contexts: list[dict[str, Any]]
    valid_masks: list[torch.Tensor]
    sample_ids: list[str]
    inputs: list[torch.Tensor]
    retained_kwargs: list[dict[str, Any]]

    @torch.no_grad()
    def forward_original(
        self,
        layer: nn.Module,
        layer_index: int,
        *,
        device: torch.device | str,
        collectors: PositionMaskCollector | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> LayerReplay:
        batch_count = len(self.inputs)
        related = (
            len(self.kwargs_by_layer[layer_index]),
            len(self.other_args_by_layer[layer_index]),
            len(self.retained_kwargs),
            len(self.valid_masks),
        )
        if any(count != batch_count for count in related):
            raise ValueError(
                f"Layer {layer_index} replay batch mismatch: "
                f"inputs={batch_count}, related={related}"
            )

        layer_inputs = self.inputs
        layer_retained = deepcopy(self.retained_kwargs)
        updated_retained = deepcopy(self.retained_kwargs)
        outputs: list[torch.Tensor] = []
        autocast = (
            torch.autocast(torch.device(device).type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with autocast:
            for batch_index, hidden in enumerate(layer_inputs):
                if collectors is not None:
                    collectors.set_position_mask(self.valid_masks[batch_index])
                kwargs = cast(
                    dict[str, Any],
                    send_to_device(
                        deepcopy(self.kwargs_by_layer[layer_index][batch_index]), device
                    ),
                )
                for key, value in updated_retained[batch_index].items():
                    kwargs[key] = send_to_device(value, device)
                other_args = send_to_device(
                    self.other_args_by_layer[layer_index][batch_index], device
                )
                outputs.append(_hidden(layer(hidden.to(device), *other_args, **kwargs)).cpu())
                for key in updated_retained[batch_index]:
                    if key in kwargs:
                        updated_retained[batch_index][key] = send_to_device(kwargs[key], "cpu")

        self.inputs = [
            apply_deepstack_context(output, context, layer_index)
            for output, context in zip(outputs, self.deepstack_contexts, strict=True)
        ]
        self.retained_kwargs = updated_retained
        return LayerReplay(
            state=self,
            layer_index=layer_index,
            inputs=layer_inputs,
            targets=outputs,
            retained_kwargs=layer_retained,
        )


class _RecordingCalibration:
    def __init__(self, source: CocoCalibrationBatches) -> None:
        self.source = source
        self.valid_masks: list[torch.Tensor] = []

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.source:
            self.valid_masks.append(batch["attention_mask"].detach().cpu().to(torch.bool))
            yield batch


def capture_replay(
    model: nn.Module,
    layers: nn.ModuleList,
    config: QTIPConfig,
    *,
    split: str,
    device: torch.device | str,
) -> ReplayState:
    sample_count = config.train_samples if split == "train" else config.validation_samples
    source = CocoCalibrationBatches(
        config.calibration,
        model_path=config.model,
        split=split,
        batch_size=config.batch_size,
        limit=sample_count,
    )
    if len(source.records) != sample_count:
        raise RuntimeError(f"Requested {sample_count} {split} samples, found {len(source.records)}")
    recorder = _RecordingCalibration(source)
    (inputs, kwargs_by_layer, other_args_by_layer, retained_kwargs, deepstack_contexts) = (
        capture_layer_inputs_and_args(model, layers, recorder, batch_size=None, device=device)
    )
    return ReplayState(
        kwargs_by_layer=kwargs_by_layer,
        other_args_by_layer=other_args_by_layer,
        deepstack_contexts=deepstack_contexts,
        valid_masks=recorder.valid_masks,
        sample_ids=[str(record["id"]) for record in source.records],
        inputs=inputs,
        retained_kwargs=retained_kwargs,
    )
