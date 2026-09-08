"""Language backbone shared by QTIP replay, quantization and dense export."""

from typing import Any


def language_backbone(model: Any) -> Any:
    if model.config.model_type == "mistral3":
        return model.model.language_model
    return model.model
