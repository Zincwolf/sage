from dataclasses import replace
from pathlib import Path

import pytest

from weight_vq import cli
from weight_vq.sage.config import SAGEConfig
from weight_vq.sphquant import pipeline
from weight_vq.sphquant.config import SPHQuantConfig


@pytest.mark.parametrize("esg,adaptive_bits", [(False, True), (True, False), (False, False)])
def test_sage_component_ablation_options(monkeypatch, tmp_path, esg, adaptive_bits):
    path = tmp_path / "sage.yaml"
    path.write_text(
        "model: /models/qwen3\ncalibration: /data/calibration.pt\noutput_dir: outputs/sage\n"
        "energy_power: 0.625\nallow_partial: true\n"
    )
    received = []
    monkeypatch.setitem(cli.STAGES, ("sage", "optimize"), (received.append, SAGEConfig))
    cli.main(
        [
            "sage",
            "optimize",
            "--config",
            str(path),
            f"--esg={str(esg).lower()}",
            f"--adaptive_bits={str(adaptive_bits).lower()}",
            "--export_compressed=true",
        ]
    )
    assert received == [
        SAGEConfig(
            model=Path("/models/qwen3"),
            calibration=Path("/data/calibration.pt"),
            output_dir=Path("outputs/sage"),
            esg=esg,
            adaptive_bits=adaptive_bits,
            energy_power=0.625,
            allow_partial=True,
            export_compressed=True,
        )
    ]


@pytest.mark.parametrize("config_type", [SPHQuantConfig, SAGEConfig])
def test_spherical_run_shares_one_config_for_every_stage(monkeypatch, tmp_path, config_type):
    config = config_type(
        model=tmp_path / "model", calibration=tmp_path / "calibration.pt", output_dir=tmp_path
    )
    calls = []
    for stage in ("optimize", "export", "validate"):
        monkeypatch.setattr(
            pipeline, stage, lambda value, stage=stage: calls.append((stage, value))
        )
    pipeline.run(config)
    assert [stage for stage, _ in calls] == ["optimize", "export", "validate"]
    assert all(value is config for _, value in calls)


def test_sage_defaults_are_independent_of_sphquant():
    baseline = SPHQuantConfig(
        model=Path("model"), calibration=Path("calibration.pt"), output_dir=Path("sphquant")
    )
    sage = SAGEConfig(
        model=baseline.model, calibration=baseline.calibration, output_dir=Path("sage")
    )
    assert sage.esg and sage.adaptive_bits
    assert sage.energy_power == 0.75
    assert not sage.allow_partial
    assert baseline.codebook_bits == 6
    assert not hasattr(baseline, "esg")
    assert replace(sage, esg=False, adaptive_bits=False).checkpoints == Path("sage/checkpoints")
