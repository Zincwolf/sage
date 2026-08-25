from pathlib import Path

import pytest

from weight_vq import worker
from weight_vq.config import PendingPipelineConfig


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


def test_worker_rejects_invalid_pair() -> None:
    with pytest.raises(SystemExit):
        worker.main(["data", "export", "--config", "data.yaml"])
