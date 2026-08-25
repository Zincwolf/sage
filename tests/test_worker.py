from pathlib import Path

import pytest

from weight_vq import worker
from weight_vq.config import CalibrationConfig, PendingPipelineConfig, TokenizerConfig
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
    monkeypatch.setattr(worker, "load_stage", lambda _specification: received.append)

    worker.main(
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


def test_worker_loads_and_forwards_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "qtip.yaml"
    config.write_text(
        "model: /models/qwen3\n"
        "calibration: outputs/calibration.pt\n"
        "output: outputs/qtip\n"
    )
    received: list[PendingPipelineConfig] = []
    monkeypatch.setattr(worker, "load_stage", lambda _specification: received.append)

    worker.main(["qtip", "export", "--config", str(config)])

    assert received == [
        PendingPipelineConfig(
            model=Path("/models/qwen3"),
            calibration=Path("outputs/calibration.pt"),
            output=Path("outputs/qtip"),
        )
    ]


@pytest.mark.parametrize("stage", ["optimize", "export", "validate"])
def test_worker_loads_shared_flat_sphquant_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    config = tmp_path / "sphquant.yaml"
    config.write_text(
        "model: /models/qwen3\n"
        "calibration: outputs/calibration.pt\n"
        "checkpoints: outputs/sphquant/layerwise\n"
        "exported_model: outputs/sphquant/hf-layerwise\n"
        "sample_size: 512\n"
        "kmeans_iterations: 2\n"
        "epochs: 0\n"
        "cache_shards: 2\n"
        "max_layers: 1\n"
        "calibration_limit: 32\n"
    )
    received: list[SPHQuantConfig] = []
    monkeypatch.setattr(worker, "load_stage", lambda _specification: received.append)

    worker.main(["sphquant", stage, "--config", str(config)])

    assert received == [
        SPHQuantConfig(
            model=Path("/models/qwen3"),
            calibration=Path("outputs/calibration.pt"),
            checkpoints=Path("outputs/sphquant/layerwise"),
            exported_model=Path("outputs/sphquant/hf-layerwise"),
            sample_size=512,
            kmeans_iterations=2,
            epochs=0,
            cache_shards=2,
            max_layers=1,
            calibration_limit=32,
        )
    ]


def test_worker_rejects_invalid_pair() -> None:
    with pytest.raises(SystemExit):
        worker.main(["data", "export", "--config", "data.yaml"])
