"""Single jsonargparse entry point used by all project environments."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from typing import Any

from jsonargparse import ArgumentParser

from weight_vq.config import CalibrationConfig, PendingPipelineConfig

Stage = Callable[[Any], None]

STAGES: dict[tuple[str, str], str] = {
    ("data", "calibration"): "weight_vq.calibration:main",
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
    config_type = CalibrationConfig if key[0] == "data" else PendingPipelineConfig
    cli.add_class_arguments(config_type, "pipeline")
    return cli


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2 or (arguments[0], arguments[1]) not in STAGES:
        methods = ", ".join(f"{method}/{stage}" for method, stage in STAGES)
        raise SystemExit(f"unsupported method/stage; expected one of: {methods}")
    key = (arguments[0], arguments[1])
    cli = parser(key)
    parsed = cli.instantiate(cli.parse_args(arguments))
    load_stage(STAGES[key])(parsed.pipeline)


if __name__ == "__main__":
    main()
