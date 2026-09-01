"""Single jsonargparse entry point used by all project environments."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

from jsonargparse import ArgumentParser

from weight_vq.config import (
    CalibrationConfig,
    PendingPipelineConfig,
    VLMCalibrationConfig,
)
from weight_vq.qtip.config import QTIPConfig
from weight_vq.sphquant.config import SPHQuantConfig

Stage = Callable[[Any], None]

STAGES: dict[tuple[str, str], str] = {
    ("data", "calibration"): "weight_vq.calibration:main",
    ("data", "vlm-calibration"): "weight_vq.vlm_calibration:main",
    ("qtip", "hessian"): "weight_vq.qtip.pipeline:hessian",
    ("qtip", "quantize"): "weight_vq.qtip.pipeline:quantize",
    ("qtip", "export"): "weight_vq.qtip.pipeline:export",
    ("qtip", "validate"): "weight_vq.qtip.pipeline:validate",
    ("quip-sharp", "hessian"): "weight_vq.quip_sharp.pipeline:hessian",
    ("quip-sharp", "quantize"): "weight_vq.quip_sharp.pipeline:quantize",
    ("quip-sharp", "export"): "weight_vq.quip_sharp.pipeline:export",
    ("quip-sharp", "validate"): "weight_vq.quip_sharp.pipeline:validate",
    ("sphquant", "optimize"): "weight_vq.sphquant.pipeline:optimize",
    ("sphquant", "export"): "weight_vq.sphquant.pipeline:export",
    ("sphquant", "validate"): "weight_vq.sphquant.pipeline:validate",
}

CONFIG_TYPES = {
    ("data", "calibration"): CalibrationConfig,
    ("data", "vlm-calibration"): VLMCalibrationConfig,
    ("qtip", "hessian"): QTIPConfig,
    ("qtip", "quantize"): QTIPConfig,
    ("qtip", "export"): QTIPConfig,
    ("qtip", "validate"): QTIPConfig,
    ("sphquant", "optimize"): SPHQuantConfig,
    ("sphquant", "export"): SPHQuantConfig,
    ("sphquant", "validate"): SPHQuantConfig,
}


def load_stage(specification: str) -> Stage:
    module_name, attribute = specification.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    stage: Any = getattr(module, attribute)
    return stage


def parser(key: tuple[str, str]) -> ArgumentParser:
    cli = ArgumentParser(prog="python -m weight_vq.worker")
    cli.add_argument("method", choices=[key[0]])
    cli.add_argument("stage", choices=[key[1]])
    cli.add_argument("--config", action="config", required=True)
    cli.add_class_arguments(CONFIG_TYPES.get(key, PendingPipelineConfig))
    return cli


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2 or (arguments[0], arguments[1]) not in STAGES:
        methods = ", ".join(f"{method}/{stage}" for method, stage in STAGES)
        raise SystemExit(f"unsupported method/stage; expected one of: {methods}")
    key = (arguments[0], arguments[1])
    cli = parser(key)
    parsed = cli.instantiate(cli.parse_args(arguments))
    config_type = CONFIG_TYPES.get(key, PendingPipelineConfig)
    config = config_type(**{field.name: parsed[field.name] for field in fields(config_type)})
    load_stage(STAGES[key])(config)


if __name__ == "__main__":
    main()
