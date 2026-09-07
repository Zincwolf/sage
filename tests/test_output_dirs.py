from dataclasses import asdict, replace
from pathlib import Path

import pytest

from weight_vq import cli
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.pipeline import _initialize_run
from weight_vq.sphquant.config import SPHQuantConfig


@pytest.mark.parametrize("config_type", [QTIPConfig, SPHQuantConfig])
def test_output_paths_are_derived_without_filesystem_side_effects(
    tmp_path: Path, config_type: type[QTIPConfig] | type[SPHQuantConfig]
) -> None:
    root = tmp_path / "nested" / "experiment"
    if config_type is QTIPConfig:
        config = QTIPConfig(model=tmp_path, dataset=tmp_path, output_dir=root, devices=[0])
        assert config.hessians == root / "hessians"
    else:
        config = SPHQuantConfig(model=tmp_path, calibration=tmp_path, output_dir=root)
    assert config.checkpoints == root / "checkpoints"
    assert config.exported_model == root / "exported_model"
    assert not root.exists()
    assert not {"hessians", "checkpoints", "exported_model"} & asdict(config).keys()
    moved = replace(config, output_dir=Path("relative/run"))
    assert moved.checkpoints == Path("relative/run/checkpoints")
    assert moved.exported_model == Path("relative/run/exported_model")
    if isinstance(moved, QTIPConfig):
        assert moved.hessians == Path("relative/run/hessians")


@pytest.mark.parametrize(
    ("method", "stage"),
    [(method, stage) for method, stage in cli.STAGES if method in ("qtip", "sphquant")],
)
def test_cli_output_dir_override_controls_all_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, stage: str
) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "model: /models/model\noutput_dir: original/run\n"
        + (
            "dataset: /data/rp\ndevices: [0]\n"
            if method == "qtip"
            else "calibration: /data/calibration\n"
        )
    )
    config_type = QTIPConfig if method == "qtip" else SPHQuantConfig
    received: list[QTIPConfig | SPHQuantConfig] = []
    monkeypatch.setitem(cli.STAGES, (method, stage), (received.append, config_type))
    cli.main([method, stage, "--config", str(path), "--output_dir=other/run"])
    assert received[0].output_dir == Path("other/run")
    assert received[0].checkpoints == Path("other/run/checkpoints")
    assert received[0].exported_model == Path("other/run/exported_model")
    if isinstance(received[0], QTIPConfig):
        assert received[0].hessians == Path("other/run/hessians")


@pytest.mark.parametrize(
    ("method", "field"),
    [
        ("qtip", "hessians"),
        ("qtip", "checkpoints"),
        ("qtip", "exported_model"),
        ("sphquant", "checkpoints"),
        ("sphquant", "exported_model"),
    ],
)
def test_cli_rejects_independent_artifact_paths(tmp_path: Path, method: str, field: str) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "model: /models/model\noutput_dir: outputs/run\n"
        + (
            "dataset: /data/rp\ndevices: [0]\n"
            if method == "qtip"
            else "calibration: /data/calibration\n"
        )
        + f"{field}: /independent/path\n"
    )
    with pytest.raises(SystemExit) as error:
        cli.main([method, "export", "--config", str(path)])
    assert error.value.code == 2


def test_qtip_initializes_nested_output_and_keeps_recipe_relocatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "nested" / "run"
    config = QTIPConfig(model=tmp_path, dataset=tmp_path, output_dir=root, devices=[0])
    monkeypatch.setattr("weight_vq.qtip.pipeline.bitshift_codebook", lambda **_kwargs: None)
    _initialize_run(config, "qwen3", 2)
    assert not config.hessians.exists()
    assert config.checkpoints.is_dir()
    assert (config.checkpoints / "qtip_config.json").is_file()
    assert not config.exported_model.exists()
    assert config.recipe(model_family="qwen3", layer_end=2) == replace(
        config, output_dir=tmp_path / "moved"
    ).recipe(model_family="qwen3", layer_end=2)
