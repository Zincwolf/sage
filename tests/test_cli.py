from pathlib import Path

import pytest

from weight_vq import cli
from weight_vq.config import CalibrationConfig, TokenizerConfig, VLMCalibrationConfig
from weight_vq.qtip.config import QTIPConfig
from weight_vq.sphquant.config import SPHQuantConfig


def test_cli_options_after_config_override_same_named_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "calibration.yaml"
    config.write_text(
        "tokenizer:\n"
        "  name: original\n"
        "  path: /models/original\n"
        "  revision: revision\n"
        "dataset_cache: /datasets\n"
        "output: outputs/calibration.pt\n"
        "seed: 0\n"
        "train_samples: 1920\n"
        "validation_samples: 128\n"
        "sequence_length: 2048\n"
    )
    received: list[CalibrationConfig] = []
    monkeypatch.setitem(
        cli.STAGES,
        ("data", "calibration"),
        (received.append, CalibrationConfig),
    )

    cli.main(
        [
            "data",
            "calibration",
            "--config",
            str(config),
            "--tokenizer.path=/models/override",
            "--train_samples=1",
            "--force=true",
        ]
    )

    assert received == [
        CalibrationConfig(
            tokenizer=TokenizerConfig(
                name="original",
                path=Path("/models/override"),
                revision="revision",
            ),
            dataset_cache=Path("/datasets"),
            output=Path("outputs/calibration.pt"),
            seed=0,
            train_samples=1,
            validation_samples=128,
            sequence_length=2048,
            force=True,
        )
    ]


def test_cli_loads_and_forwards_qtip_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "qtip.yaml"
    config.write_text(
        "model: /models/qwen3-vl\n"
        "calibration: /data/coco.manifest.json\n"
        "hessians: outputs/qtip/hessians\n"
        "checkpoints: outputs/qtip/trellis\n"
        "exported_model: outputs/qtip/hf-dense-qdq\n"
        "devices: [0, 2]\n"
        "train_samples: 16\n"
        "validation_samples: 8\n"
        "batch_size: 4\n"
        "reconstruction_epochs: 1\n"
        "layer_start: 0\n"
        "layer_end: 4\n"
    )
    received: list[QTIPConfig] = []
    monkeypatch.setitem(
        cli.STAGES,
        ("qtip", "export"),
        (received.append, QTIPConfig),
    )

    cli.main(["qtip", "export", "--config", str(config)])

    assert received == [
        QTIPConfig(
            model=Path("/models/qwen3-vl"),
            calibration=Path("/data/coco.manifest.json"),
            hessians=Path("outputs/qtip/hessians"),
            checkpoints=Path("outputs/qtip/trellis"),
            exported_model=Path("outputs/qtip/hf-dense-qdq"),
            devices=[0, 2],
            train_samples=16,
            validation_samples=8,
            batch_size=4,
            reconstruction_epochs=1,
            layer_start=0,
            layer_end=4,
        )
    ]


def test_cli_loads_vlm_calibration_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "vlm-calibration.yaml"
    config.write_text(
        "output_dir: /datasets/sharegpt4v-coco\n"
        "workers: 8\n"
        "image_archive: /datasets/train2017.zip\n"
        "force: true\n"
    )
    received: list[VLMCalibrationConfig] = []
    monkeypatch.setitem(
        cli.STAGES,
        ("data", "vlm-calibration"),
        (received.append, VLMCalibrationConfig),
    )

    cli.main(["data", "vlm-calibration", "--config", str(config)])

    assert received == [
        VLMCalibrationConfig(
            output_dir=Path("/datasets/sharegpt4v-coco"),
            workers=8,
            image_archive=Path("/datasets/train2017.zip"),
            force=True,
        )
    ]


@pytest.mark.parametrize("stage", ["optimize", "export", "validate"])
def test_cli_loads_shared_flat_sphquant_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    config = tmp_path / "sphquant.yaml"
    config.write_text(
        "model: /models/qwen3-vl\n"
        "calibration: /data/coco.manifest.json\n"
        "checkpoints: outputs/sphquant/layerwise\n"
        "exported_model: outputs/sphquant/hf-layerwise\n"
        "device: 2\n"
        "sample_size: 512\n"
        "kmeans_iterations: 2\n"
        "epochs: 0\n"
        "cache_shards: 2\n"
        "max_layers: 1\n"
        "calibration_limit: 32\n"
    )
    received: list[SPHQuantConfig] = []
    monkeypatch.setitem(
        cli.STAGES,
        ("sphquant", stage),
        (received.append, SPHQuantConfig),
    )

    cli.main(["sphquant", stage, "--config", str(config)])

    assert received == [
        SPHQuantConfig(
            model=Path("/models/qwen3-vl"),
            calibration=Path("/data/coco.manifest.json"),
            checkpoints=Path("outputs/sphquant/layerwise"),
            exported_model=Path("outputs/sphquant/hf-layerwise"),
            device=2,
            sample_size=512,
            kmeans_iterations=2,
            epochs=0,
            cache_shards=2,
            max_layers=1,
            calibration_limit=32,
        )
    ]


def test_cli_has_nested_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["qtip", "--help"])

    assert exit_info.value.code == 0
    assert "hessian" in capsys.readouterr().out


def test_cli_rejects_invalid_pair() -> None:
    with pytest.raises(SystemExit):
        cli.main(["data", "export", "--config", "data.yaml"])
