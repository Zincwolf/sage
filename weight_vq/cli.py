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
            # Stage options belong to the jsonargparse worker and must be forwarded
            # verbatim so config values can be overridden from the command line.
            stage_parsers.add_parser(stage, add_help=False)
    return cli


def command(
    args: argparse.Namespace,
    worker_args: Sequence[str],
    *,
    root: Path | None = None,
) -> list[str]:
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
        *worker_args,
    ]


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args, worker_args = parser().parse_known_args(arguments)
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    try:
        completed = subprocess.run(
            command(args, worker_args), check=False, env=environment
        )
    except FileNotFoundError as error:
        raise SystemExit(
            f"uv executable not found: {command(args, worker_args)[0]}"
        ) from error
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
