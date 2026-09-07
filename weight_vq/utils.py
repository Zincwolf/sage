"""Shared artifact and integrity utilities."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
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


def validate_dense_model(
    path: Path,
    *,
    model_loader: Any,
    weights: Callable[[torch.nn.Module], Iterable[tuple[str, torch.Tensor]]],
    device: int | str = 0,
) -> Any:
    """Load through the HF API and report selected dense weights as CSV.

    Quantiles use at most one million regularly spaced values per weight;
    finiteness and the absolute maximum cover the full weight.
    """
    model = model_loader.from_pretrained(
        path,
        dtype="auto",
        device_map=device,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("layer", "p01", "p50", "p99", "abs_p99", "abs_p999", "abs_max"))
    with torch.no_grad():
        for name, parameter in weights(model):
            weight = parameter.detach().flatten()
            if not torch.isfinite(weight).all():
                raise ValueError(f"non-finite weight: {name}")
            stride = max(1, (weight.numel() + 999_999) // 1_000_000)
            sample = weight[::stride].float()
            quantiles = torch.quantile(
                sample, torch.tensor([0.01, 0.5, 0.99], device=sample.device)
            )
            abs_quantiles = torch.quantile(
                sample.abs(), torch.tensor([0.99, 0.999], device=sample.device)
            )
            writer.writerow(
                (
                    name,
                    *(f"{value.item():.5e}" for value in quantiles),
                    *(f"{value.item():.5e}" for value in abs_quantiles),
                    f"{weight.abs().max().item():.5e}",
                )
            )
    return model


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
