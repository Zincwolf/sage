from pathlib import Path

import pytest

from weight_vq import worker
from weight_vq.config import PendingPipelineConfig
from weight_vq.sphquant.config import (
    SPHQuantInitializationConfig,
    SPHQuantOptimizeConfig,
    SPHQuantRecoveryConfig,
)


def test_worker_loads_and_forwards_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "qtip.yaml"
    config.write_text(
        "pipeline:\n"
        "  model: /models/qwen3\n"
        "  calibration: outputs/calibration.pt\n"
        "  output: outputs/qtip\n"
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


def test_worker_loads_typed_sphquant_optimize_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "sphquant.yaml"
    config.write_text(
        "pipeline:\n"
        "  model: /models/qwen3\n"
        "  calibration: outputs/calibration.pt\n"
        "  output: outputs/sphquant\n"
        "  initialization:\n"
        "    sample_size: 512\n"
        "    kmeans_iterations: 2\n"
        "  recovery:\n"
        "    epochs: 0\n"
        "    cache_shards: 2\n"
        "  max_layers: 1\n"
        "  calibration_limit: 32\n"
    )
    received: list[SPHQuantOptimizeConfig] = []
    monkeypatch.setattr(worker, "load_stage", lambda _specification: received.append)

    worker.main(["sphquant", "optimize", "--config", str(config)])

    assert received == [
        SPHQuantOptimizeConfig(
            model=Path("/models/qwen3"),
            calibration=Path("outputs/calibration.pt"),
            output=Path("outputs/sphquant"),
            initialization=SPHQuantInitializationConfig(
                sample_size=512,
                kmeans_iterations=2,
            ),
            recovery=SPHQuantRecoveryConfig(epochs=0, cache_shards=2),
            max_layers=1,
            calibration_limit=32,
        )
    ]


def test_worker_rejects_invalid_pair() -> None:
    with pytest.raises(SystemExit):
        worker.main(["data", "export", "--config", "data.yaml"])
