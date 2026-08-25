import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from weight_vq import cli
from weight_vq.cli import command, parser


def test_dispatch_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UV", raising=False)
    args = Namespace(method="quip-sharp", stage="quantize")
    worker_args = [
        "--config",
        "configs/quip-sharp/qwen3.example.yaml",
        "--output=outputs/test",
    ]

    assert command(args, worker_args, root=Path("/repo")) == [
        "uv",
        "run",
        "--project",
        "/repo/projects/quip-sharp",
        "--no-sync",
        "python",
        "-m",
        "weight_vq.worker",
        "quip-sharp",
        "quantize",
        *worker_args,
    ]


def test_dispatch_drops_parent_virtual_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")

    def run(command: list[str], *, check: bool, env: dict[str, str]):
        assert check is False
        assert "VIRTUAL_ENV" not in env
        assert command[-2:] == ["--config", "config.yaml"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["data", "calibration", "--config", "config.yaml"])

    assert exit_info.value.code == 0


def test_stage_help_and_overrides_are_left_for_worker() -> None:
    args, worker_args = parser().parse_known_args(
        [
            "data",
            "calibration",
            "--help",
            "--config",
            "config.yaml",
            "--train_samples=1",
        ]
    )

    assert (args.method, args.stage) == ("data", "calibration")
    assert worker_args == [
        "--help",
        "--config",
        "config.yaml",
        "--train_samples=1",
    ]


def test_invalid_stage_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["sphquant", "hessian", "--config", "config.yaml"])
