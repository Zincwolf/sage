"""Qwen3-VL language-backbone contract for dense QTIP."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from weight_vq.qtip.artifacts import PROJECTIONS, layer_complete, reconstruct_dense_weight

EXPECTED_LAYERS = 36


def get_language_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    language_model = getattr(inner, "language_model", None)
    if language_model is None:
        language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise TypeError(f"{type(model).__name__} does not expose a Qwen3-VL language model")
    return language_model


def get_layers(model: nn.Module) -> nn.ModuleList:
    layers = getattr(get_language_model(model), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Qwen3-VL language_model.layers is not a ModuleList")
    return layers


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def audit_qwen3_vl(
    model: nn.Module,
    *,
    expected_layers: int = EXPECTED_LAYERS,
    tile_rows: int = 16,
    tile_columns: int = 16,
) -> dict[str, Any]:
    config = model.config
    model_type = _config_value(config, "model_type")
    if model_type != "qwen3_vl":
        raise RuntimeError(f"Expected model_type=qwen3_vl, got {model_type!r}")

    layers = get_layers(model)
    if len(layers) != expected_layers:
        raise RuntimeError(f"Expected {expected_layers} decoder layers, found {len(layers)}")

    text_config = _config_value(config, "text_config")
    configured_layers = _config_value(text_config, "num_hidden_layers")
    if configured_layers is not None and int(configured_layers) != len(layers):
        raise RuntimeError(
            "text_config.num_hidden_layers does not match language backbone: "
            f"{configured_layers} vs {len(layers)}"
        )

    projection_shapes: dict[str, list[int]] = {}
    for index, layer in enumerate(layers):
        for attribute, _short, _hessian, _parallel in PROJECTIONS:
            try:
                module = attrgetter(attribute)(layer)
            except AttributeError as error:
                raise AttributeError(
                    f"layer {index} is missing QTIP projection {attribute}"
                ) from error
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"layer {index} {attribute} is {type(module).__name__}, not nn.Linear"
                )
            if module.out_features % tile_rows:
                raise ValueError(
                    f"layer {index} {attribute} output width {module.out_features} "
                    f"is not divisible by QTIP tile rows {tile_rows}"
                )
            if module.in_features % tile_columns:
                raise ValueError(
                    f"layer {index} {attribute} input width {module.in_features} "
                    f"is not divisible by QTIP tile columns {tile_columns}"
                )
            projection_shapes.setdefault(attribute, [module.out_features, module.in_features])

    vision_config = _config_value(config, "vision_config")
    deepstack_indexes = _config_value(vision_config, "deepstack_visual_indexes", [])
    if not deepstack_indexes:
        raise RuntimeError("Qwen3-VL vision_config has no deepstack_visual_indexes")

    return {
        "model_type": model_type,
        "num_hidden_layers": len(layers),
        "projection_count": len(layers) * len(PROJECTIONS),
        "projections_per_layer": len(PROJECTIONS),
        "projection_shapes": projection_shapes,
        "deepstack_visual_indexes": list(deepstack_indexes),
        "deepstack_language_injection_layers": list(range(len(deepstack_indexes))),
    }


def projection_state_key(layer_index: int, projection: str) -> str:
    return f"model.language_model.layers.{layer_index}.{projection}.weight"


def inject_dense_checkpoints(
    model: nn.Module,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    *,
    layer_start: int,
    layer_end: int,
    reconstruct: Callable[[dict[str, Any], dict[str, Any]], torch.Tensor] | None = None,
) -> list[str]:
    """Inject fake-quantized weights into the Qwen3-VL language backbone."""
    checkpoint_dir = Path(checkpoint_dir)
    layers = get_layers(model)
    reconstruct = reconstruct_dense_weight if reconstruct is None else reconstruct
    modified: list[str] = []
    with torch.no_grad():
        for layer_index in range(layer_start, layer_end):
            if not layer_complete(layer_index, checkpoint_dir):
                raise RuntimeError(f"QTIP layer {layer_index} is not complete")
            layer = layers[layer_index]
            for attribute, short, _hessian, _parallel in PROJECTIONS:
                path = checkpoint_dir / f"{layer_index}_{short}.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                dense = reconstruct(saved, config)
                linear = attrgetter(attribute)(layer)
                if tuple(dense.shape) != tuple(linear.weight.shape):
                    raise RuntimeError(
                        f"{path.name}: {tuple(dense.shape)} != {tuple(linear.weight.shape)}"
                    )
                linear.weight.copy_(dense.to(linear.weight.dtype))
                modified.append(projection_state_key(layer_index, attribute))
    return modified
