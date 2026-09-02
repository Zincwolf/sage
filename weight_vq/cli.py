"""Command line interface for weight-vq pipelines."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

from jsonargparse import ArgumentParser, Namespace

from weight_vq import calibration, vlm_calibration
from weight_vq.config import CalibrationConfig, PendingPipelineConfig, VLMCalibrationConfig
from weight_vq.qtip import pipeline as qtip_pipeline
from weight_vq.qtip.config import QTIPConfig
from weight_vq.quip_sharp import pipeline as quip_sharp_pipeline
from weight_vq.sphquant import pipeline as sphquant_pipeline
from weight_vq.sphquant.config import SPHQuantConfig

Stage = Callable[[Any], None]
ConfigType = type[Any]

STAGES: dict[tuple[str, str], tuple[Stage, ConfigType]] = {
    ("data", "calibration"): (calibration.main, CalibrationConfig),
    ("data", "vlm-calibration"): (vlm_calibration.main, VLMCalibrationConfig),
    ("qtip", "hessian"): (qtip_pipeline.hessian, QTIPConfig),
    ("qtip", "quantize"): (qtip_pipeline.quantize, QTIPConfig),
    ("qtip", "export"): (qtip_pipeline.export, QTIPConfig),
    ("qtip", "validate"): (qtip_pipeline.validate, QTIPConfig),
    ("quip-sharp", "hessian"): (quip_sharp_pipeline.hessian, PendingPipelineConfig),
    ("quip-sharp", "quantize"): (quip_sharp_pipeline.quantize, PendingPipelineConfig),
    ("quip-sharp", "export"): (quip_sharp_pipeline.export, PendingPipelineConfig),
    ("quip-sharp", "validate"): (quip_sharp_pipeline.validate, PendingPipelineConfig),
    ("sphquant", "optimize"): (sphquant_pipeline.optimize, SPHQuantConfig),
    ("sphquant", "export"): (sphquant_pipeline.export, SPHQuantConfig),
    ("sphquant", "validate"): (sphquant_pipeline.validate, SPHQuantConfig),
}


def parser() -> ArgumentParser:
    cli = ArgumentParser(prog="wvq", description=__doc__)
    methods = cli.add_subcommands(required=True)
    method_parsers = {
        method: ArgumentParser() for method in dict.fromkeys(method for method, _stage in STAGES)
    }
    for method, method_parser in method_parsers.items():
        methods.add_subcommand(method, method_parser)

    for method, method_parser in method_parsers.items():
        stages = method_parser.add_subcommands(required=True)
        for (stage_method, stage), (_function, config_type) in STAGES.items():
            if stage_method != method:
                continue
            stage_parser = ArgumentParser()
            stages.add_subcommand(stage, stage_parser)
            stage_parser.add_argument("--config", action="config", required=True)
            stage_parser.add_class_arguments(config_type)
    return cli


def _selected(parsed: Namespace) -> tuple[tuple[str, str], Namespace]:
    method = parsed.subcommand
    method_values = parsed[method]
    stage = method_values.subcommand
    return (method, stage), method_values[stage]


def main(argv: Sequence[str] | None = None) -> None:
    cli = parser()
    parsed = cli.instantiate(cli.parse_args(list(sys.argv[1:] if argv is None else argv)))
    key, values = _selected(parsed)
    function, config_type = STAGES[key]
    config = config_type(**{field.name: values[field.name] for field in fields(config_type)})
    function(config)


if __name__ == "__main__":
    main()
