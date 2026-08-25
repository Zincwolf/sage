"""Lightweight dispatcher for isolated weight-vq environments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

METHOD_PROJECTS = ["data", "qtip", "quip-sharp", "sphquant"]
METHOD_STAGES = {
    "data": ("calibration",),
    "qtip": ("hessian", "quantize", "export", "validate"),
    "quip-sharp": ("hessian", "quantize", "export", "validate"),
    "sphquant": ("optimize", "export", "validate"),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="wvq", description=__doc__)
    methods = cli.add_subparsers(dest="method", required=True)
    for method in METHOD_PROJECTS:
        method_parser = methods.add_parser(method)
        stage_parsers = method_parser.add_subparsers(dest="stage", required=True)
        for stage in METHOD_STAGES[method]:
            stage_parser = stage_parsers.add_parser(stage)
            stage_parser.add_argument("--config", type=Path, required=True)
    return cli


def command(args: argparse.Namespace, *, root: Path | None = None) -> list[str]:
    root = repository_root() if root is None else root
    project = root / "projects" / args.method
    return [
        os.environ.get("UV", "uv"),
        "run",
        "--project",
        str(project),
        "--no-sync",
        "python",
        "-m",
        "weight_vq.worker",
        args.method,
        args.stage,
        "--config",
        str(args.config),
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        completed = subprocess.run(command(args), check=False)
    except FileNotFoundError as error:
        raise SystemExit(f"uv executable not found: {command(args)[0]}") from error
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
