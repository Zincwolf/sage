"""Build the fixed Qwen3 mixed-text calibration recipe."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from weight_vq import utils
from weight_vq.config import CalibrationConfig, CalibrationMetadata

DATASETS = {
    "wikitext2": {
        "name": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "subset": "wikitext-2-raw-v1",
        "splits": {"train": "train", "validation": "validation"},
        "data_files": {"train": None, "validation": None},
        "partition": "native splits",
    },
    "c4": {
        "name": "allenai/c4",
        "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "subset": None,
        "splits": {"train": "train", "validation": "validation"},
        "data_files": {
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00001-of-00008.json.gz",
        },
        "partition": "native splits",
    },
    "redpajama": {
        "name": "liang2kl/RedPajama-Data-1T-Sample-Backup",
        "revision": "c2c1504dcb5887121b04787e36055e32aeb9017c",
        "subset": None,
        "splits": {"train": "train", "validation": "train"},
        "data_files": {"train": None, "validation": None},
        "partition": "seeded shuffle, then 70% train / 10% validation / 20% test",
    },
}


def load_calibration_metadata(path: Path) -> CalibrationMetadata:
    tensors = json.loads(utils.manifest_path(path).read_text())["tensors"]
    train_shape = tensors["train"]["shape"]
    validation_shape = tensors["validation"]["shape"]
    return CalibrationMetadata(
        train_samples=train_shape[0],
        validation_samples=validation_shape[0],
        sequence_length=train_shape[1],
    )


def _redpajama_split(dataset: Any, split: str) -> Any:
    test_size = int(len(dataset) * 0.2)
    validation_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size - validation_size
    if split == "train":
        return dataset.select(range(train_size))
    return dataset.select(range(train_size, train_size + validation_size))


def _load_rows(cache: Path, key: str, split: str, seed: int) -> Any:
    from datasets import load_dataset  # pyright: ignore[reportMissingImports]

    recipe = DATASETS[key]
    source_split = recipe["splits"][split]
    data_file = recipe["data_files"][split]
    dataset = load_dataset(
        recipe["name"],
        recipe["subset"],
        split=source_split,
        revision=recipe["revision"],
        data_files={source_split: data_file} if data_file is not None else None,
        cache_dir=str(cache),
        trust_remote_code=key == "redpajama",
    ).shuffle(seed=seed)
    return _redpajama_split(dataset, split) if key == "redpajama" else dataset


def _text_blocks(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    samples: int,
    sequence_length: int,
) -> list[list[int]]:
    """Strip, tokenize, filter, concatenate, then cut fixed-length blocks."""
    tokens: list[int] = []
    for row in rows:
        encoded = tokenizer.encode(row["text"].strip())
        if not encoded or len(encoded) > sequence_length:
            continue
        tokens.extend(encoded)
        if len(tokens) >= samples * sequence_length:
            break
    if len(tokens) < samples * sequence_length:
        raise RuntimeError(f"only {len(tokens) // sequence_length} of {samples} blocks available")
    return [
        tokens[start : start + sequence_length]
        for start in range(0, samples * sequence_length, sequence_length)
    ]


def _allocation(samples: int) -> dict[str, int]:
    per_dataset = samples // len(DATASETS)
    return {
        key: per_dataset if index < len(DATASETS) - 1 else samples - per_dataset * index
        for index, key in enumerate(DATASETS)
    }


def _mixed_blocks(
    tokenizer: Any,
    cache: Path,
    split: str,
    samples: int,
    sequence_length: int,
    seed: int,
) -> list[list[int]]:
    allocation = _allocation(samples)
    blocks = []
    for key, quota in allocation.items():
        blocks.extend(
            _text_blocks(
                _load_rows(cache, key, split, seed),
                tokenizer,
                quota,
                sequence_length,
            )
        )
    random.Random(seed).shuffle(blocks)
    return blocks


def main(config: CalibrationConfig) -> None:
    if not config.dataset_cache.is_dir():
        raise FileNotFoundError(f"dataset cache does not exist: {config.dataset_cache}")
    if not config.tokenizer.path.is_dir():
        raise FileNotFoundError(f"tokenizer directory does not exist: {config.tokenizer.path}")
    if min(config.train_samples, config.validation_samples, config.sequence_length) < 1:
        raise ValueError("sample counts and sequence length must be positive")

    manifest_path = utils.manifest_path(config.output)
    existing = [path for path in (config.output, manifest_path) if path.exists()]
    if existing and not config.force:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")

    import torch  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.path,
        revision=config.tokenizer.revision,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    train = torch.tensor(
        _mixed_blocks(
            tokenizer,
            config.dataset_cache,
            "train",
            config.train_samples,
            config.sequence_length,
            config.seed,
        ),
        dtype=torch.int64,
    ).contiguous()
    validation = torch.tensor(
        _mixed_blocks(
            tokenizer,
            config.dataset_cache,
            "validation",
            config.validation_samples,
            config.sequence_length,
            config.seed,
        ),
        dtype=torch.int64,
    ).contiguous()
    tensors = {"train": train, "validation": validation}
    manifest = {
        "tensors": {
            name: {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": utils.tensor_sha256(tensor),
            }
            for name, tensor in tensors.items()
        }
    }
    actual_sha256 = utils.tensor_inventory_sha256(manifest["tensors"])
    utils.check_hash("calibration", actual_sha256, config.expected_sha256)

    output_temp = utils.temporary_path(config.output)
    manifest_temp = utils.temporary_path(manifest_path)
    try:
        torch.save(tensors, output_temp)
        manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(output_temp, config.output)
        os.replace(manifest_temp, manifest_path)
    finally:
        output_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
