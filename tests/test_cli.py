import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from weight_vq import cli
from weight_vq.cli import command, parser


def test_dispatch_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UV", raising=False)
    args = Namespace(
        method="quip-sharp",
        stage="quantize",
        config=Path("configs/quip-sharp/qwen3.example.yaml"),
    )

    assert command(args, root=Path("/repo")) == [
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
        "--config",
        "configs/quip-sharp/qwen3.example.yaml",
    ]


def test_dispatch_drops_parent_virtual_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")

    def run(command: list[str], *, check: bool, env: dict[str, str]):
        assert check is False
        assert "VIRTUAL_ENV" not in env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["data", "calibration", "--config", "config.yaml"])

    assert exit_info.value.code == 0


def test_config_is_required() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["qtip", "hessian"])


def test_invalid_stage_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["sphquant", "hessian", "--config", "config.yaml"])
