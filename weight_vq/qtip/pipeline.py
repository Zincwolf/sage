"""QTIP stage adapters for dense Qwen3-VL QDQ export."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from weight_vq.qtip.config import QTIPConfig

EXPECTED_LAYERS = 36
PROJECTION_SHORT_NAMES = ("v", "q", "k", "o", "up", "gate", "down")
HESSIAN_NAMES = ("qkv", "o", "up", "down")


def layer_range(config: QTIPConfig) -> tuple[int, int]:
    end = EXPECTED_LAYERS if config.layer_end is None else config.layer_end
    if not 0 <= config.layer_start < end <= EXPECTED_LAYERS:
        raise ValueError("layer range must be within [0, 36]")
    return config.layer_start, end


def split_layer_ranges(
    start: int,
    end: int,
    devices: list[int],
) -> list[tuple[int, int, int]]:
    if not devices:
        raise ValueError("devices must contain at least one CUDA device")
    if len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise ValueError("devices must be distinct non-negative integers")
    worker_count = min(len(devices), end - start)
    width, remainder = divmod(end - start, worker_count)
    ranges = []
    cursor = start
    for worker_index, device in enumerate(devices[:worker_count]):
        next_cursor = cursor + width + (worker_index < remainder)
        ranges.append((device, cursor, next_cursor))
        cursor = next_cursor
    return ranges


def hessian_complete(index: int, directory: Path) -> bool:
    return all((directory / f"{index}_{name}.pt").is_file() for name in HESSIAN_NAMES)


def layer_complete(index: int, directory: Path) -> bool:
    return (directory / f"{index}.complete").is_file() and all(
        (directory / f"{index}_{name}.pt").is_file()
        for name in PROJECTION_SHORT_NAMES
    )


def _validate_config(config: QTIPConfig) -> tuple[int, int]:
    start, end = layer_range(config)
    split_layer_ranges(start, end, config.devices)
    for name in ("train_samples", "validation_samples", "batch_size"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.train_samples % config.batch_size:
        raise ValueError("train_samples must be divisible by batch_size")
    if config.validation_samples % config.batch_size:
        raise ValueError("validation_samples must be divisible by batch_size")
    if config.reconstruction_epochs < 0:
        raise ValueError("reconstruction_epochs must be non-negative")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return start, end


def _recipe(config: QTIPConfig) -> dict[str, Any]:
    start, end = layer_range(config)
    return {
        "format": "qtip-bitshift-trellis-qwen3-vl-dense-qdq",
        "model_family": "qwen3_vl",
        "model": str(config.model.resolve()),
        "calibration": str(config.calibration.resolve()),
        "layer_range": [start, end],
        "quantization": {
            "td_x": 16,
            "td_y": 16,
            "L": 16,
            "K": 2,
            "V": 2,
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
            "scale_override": 0.9,
            "sigma_reg": 1e-2,
        },
        "calibration_recipe": {
            "train_samples": config.train_samples,
            "validation_samples": config.validation_samples,
            "batch_size": config.batch_size,
            "include_padding_in_hessian": False,
        },
        "reconstruction": {
            "epochs": config.reconstruction_epochs,
            "learning_rate": config.learning_rate,
            "mask_padding": True,
        },
        "model_dtype": "bfloat16",
        "seed": config.seed,
        "packed_inference_kernel": False,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _initialize_run(config: QTIPConfig) -> None:
    config.hessians.mkdir(parents=True, exist_ok=True)
    config.checkpoints.mkdir(parents=True, exist_ok=True)
    recipe = _recipe(config)
    path = config.checkpoints / "qtip_config.json"
    if path.is_file():
        if json.loads(path.read_text()) != recipe:
            raise RuntimeError(
                "configuration differs from checkpoints/qtip_config.json; "
                "use a new checkpoint directory"
            )
    else:
        _atomic_write_json(path, recipe)

    # Constructing quantlut_sym once in the coordinator pins one run-local
    # TLUT; workers only load it and never race through shared /tmp state.
    tlut = config.checkpoints / "tlut-l16-b9-v2.pt"
    if not tlut.is_file():
        from lib.codebook.bitshift import bitshift_codebook  # pyright: ignore[reportMissingImports]

        previous = os.environ.get("QTIP_TLUT_PATH")
        os.environ["QTIP_TLUT_PATH"] = str(tlut)
        try:
            bitshift_codebook(
                L=16,
                K=2,
                V=2,
                tlut_bits=9,
                decode_mode="quantlut_sym",
            )
        finally:
            if previous is None:
                os.environ.pop("QTIP_TLUT_PATH", None)
            else:
                os.environ["QTIP_TLUT_PATH"] = previous


def _worker_command(
    config: QTIPConfig,
    stage: str,
    start: int,
    end: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "weight_vq.qtip.worker",
        stage,
        "--model",
        str(config.model),
        "--calibration",
        str(config.calibration),
        "--hessians",
        str(config.hessians),
        "--checkpoints",
        str(config.checkpoints),
        "--train-samples",
        str(config.train_samples),
        "--validation-samples",
        str(config.validation_samples),
        "--batch-size",
        str(config.batch_size),
        "--reconstruction-epochs",
        str(config.reconstruction_epochs),
        "--learning-rate",
        str(config.learning_rate),
        "--seed",
        str(config.seed),
        "--layer-start",
        str(start),
        "--layer-end",
        str(end),
    ]


def _run_workers(config: QTIPConfig, stage: str) -> None:
    start, end = _validate_config(config)
    _initialize_run(config)
    processes: list[tuple[int, int, int, subprocess.Popen[bytes]]] = []
    for device, shard_start, shard_end in split_layer_ranges(start, end, config.devices):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        process = subprocess.Popen(
            _worker_command(config, stage, shard_start, shard_end),
            env=environment,
        )
        processes.append((device, shard_start, shard_end, process))

    failures = []
    for device, shard_start, shard_end, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append((device, shard_start, shard_end, returncode))
    if failures:
        raise RuntimeError(f"QTIP {stage} workers failed: {failures}")

    if stage == "hessian":
        missing = [
            index
            for index in range(start, end)
            if not hessian_complete(index, config.hessians)
        ]
    else:
        missing = [
            index
            for index in range(start, end)
            if not layer_complete(index, config.checkpoints)
        ]
    if missing:
        raise RuntimeError(f"QTIP {stage} did not complete layers: {missing}")


def hessian(config: QTIPConfig) -> None:
    _run_workers(config, "hessian")


def quantize(config: QTIPConfig) -> None:
    _run_workers(config, "quantize")


def export(config: QTIPConfig) -> None:
    start, end = _validate_config(config)
    missing = [
        index
        for index in range(start, end)
        if not layer_complete(index, config.checkpoints)
    ]
    if missing:
        raise RuntimeError(f"QTIP checkpoints are incomplete for layers: {missing}")
    if config.exported_model.exists():
        raise FileExistsError(f"refusing to overwrite {config.exported_model}")

    from lib.qwen3_vl.common import (  # pyright: ignore[reportMissingImports]
        audit_qwen3_vl,
        inject_dense_checkpoints,
    )
    from transformers import (  # pyright: ignore[reportMissingImports]
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        config.model,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    ).eval()
    audit_qwen3_vl(model)
    recipe = json.loads((config.checkpoints / "qtip_config.json").read_text())
    modified = inject_dense_checkpoints(
        model,
        config.checkpoints,
        recipe,
        layer_start=start,
        layer_end=end,
    )
    expected = (end - start) * len(PROJECTION_SHORT_NAMES)
    if len(modified) != expected:
        raise RuntimeError(f"export modified {len(modified)} projections, expected {expected}")

    staging = config.exported_model.with_name(f".{config.exported_model.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        staging.mkdir(parents=True)
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        processor = AutoProcessor.from_pretrained(
            config.model,
            local_files_only=True,
            trust_remote_code=True,
        )
        processor.save_pretrained(staging)
        shutil.copy2(config.checkpoints / "qtip_config.json", staging / "qtip_config.json")
        config.exported_model.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, config.exported_model)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate(config: QTIPConfig) -> None:
    """Load the dense artifact through the evaluation-facing HF API."""
    start, end = _validate_config(config)
    import torch  # pyright: ignore[reportMissingImports]
    from lib.qwen3_vl.common import PROJECTIONS, get_layers  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForImageTextToText  # pyright: ignore[reportMissingImports]

    model = AutoModelForImageTextToText.from_pretrained(
        config.exported_model,
        torch_dtype="auto",
        device_map=0,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    layers = get_layers(model)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("layer", "p01", "p50", "p99", "abs_p99", "abs_p999", "abs_max"))
    with torch.no_grad():
        for index in range(start, end):
            layer = layers[index]
            for attribute, _short, _hessian, _parallel in PROJECTIONS:
                module: Any = layer
                for part in attribute.split("."):
                    module = getattr(module, part)
                weight = module.weight.detach().flatten().float()
                if not torch.isfinite(weight).all():
                    raise ValueError(f"non-finite weight: language layer {index} {attribute}")
                stride = max(1, (weight.numel() + 999_999) // 1_000_000)
                sample = weight[::stride]
                quantiles = torch.quantile(
                    sample,
                    torch.tensor([0.01, 0.5, 0.99], device=sample.device),
                )
                abs_quantiles = torch.quantile(
                    sample.abs(),
                    torch.tensor([0.99, 0.999], device=sample.device),
                )
                writer.writerow(
                    (
                        f"model.language_model.layers.{index}.{attribute}",
                        *(f"{value.item():.5e}" for value in quantiles),
                        *(f"{value.item():.5e}" for value in abs_quantiles),
                        f"{weight.abs().max().item():.5e}",
                    )
                )
