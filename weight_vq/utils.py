"""Shared artifact and integrity utilities."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def check_hash(name: str, actual: str, expected: str | None) -> None:
    if expected is not None and actual != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: {actual} != {expected}")


def temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    return Path(name)
