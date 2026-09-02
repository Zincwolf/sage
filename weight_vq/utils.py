"""Shared artifact and integrity utilities."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def tensor_inventory_sha256(inventory: dict[str, Any]) -> str:
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def atomic_write_text(path: Path, content: str) -> None:
    temporary = temporary_path(path)
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = temporary_path(path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
