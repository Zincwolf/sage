"""Independent RedPajama Hessian and block-reconstruction samples."""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset, DatasetInfo, concatenate_datasets
from transformers import AutoTokenizer

from weight_vq import utils
from weight_vq.blockwise.config import BlockwiseConfig
from weight_vq.unisvq.config import UniSVQConfig


def sample_documents(
    dataset: Dataset,
    tokenizer: Any,
    *,
    samples: int,
    length: int,
    concatenate: bool,
    generator: torch.Generator,
    truncate_documents: bool = False,
) -> torch.Tensor:
    result = torch.empty(samples * length, dtype=torch.int64)
    written = 0
    # Process a document at a time: full-document tokenization never creates a
    # padded batch as large as its longest outlier, or stores a corpus-sized list.
    while written < result.numel():
        index = torch.randint(len(dataset), (1,), generator=generator).item()
        encoded = tokenizer(
            dataset[int(index)]["text"],
            truncation=truncate_documents or not concatenate,
            **({} if concatenate and not truncate_documents else {"max_length": length}),
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        if not concatenate and len(encoded) < length:
            continue
        count = min(len(encoded), result.numel() - written)
        result[written : written + count] = torch.tensor(encoded[:count], dtype=torch.int64)
        written += count
    return result.reshape(samples, length)


def calibration(config: BlockwiseConfig) -> None:
    protocol = config.calibration_recipe()
    if config.calibration.exists():
        load_calibration(config)
        return
    info = DatasetInfo.from_directory(str(config.dataset))
    files = sorted(config.dataset.glob("*-train-*-of-*.arrow"))
    dataset = concatenate_datasets([Dataset.from_file(str(path), info=info) for path in files])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, local_files_only=True, fix_mistral_regex=True
    )
    generator = torch.Generator().manual_seed(config.seed)
    hessian = sample_documents(
        dataset,
        tokenizer,
        samples=config.hessian_samples,
        length=config.hessian_sequence_length,
        concatenate=True,
        generator=generator,
        truncate_documents=isinstance(config, UniSVQConfig),
    )
    reconstruction = sample_documents(
        dataset,
        tokenizer,
        samples=config.train_samples + config.validation_samples,
        length=config.sequence_length,
        concatenate=isinstance(config, UniSVQConfig),
        generator=generator,
        truncate_documents=isinstance(config, UniSVQConfig),
    )
    utils.atomic_torch_save(
        {
            "protocol": protocol,
            "source": {
                "dataset_name": info.dataset_name,
                "description": info.description,
                "homepage": info.homepage,
                "download_sources": list(info.download_checksums or {}),
                "rows": len(dataset),
                "upstream_sample_identity_verified": False,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_name_or_path": tokenizer.name_or_path,
            },
            "hessian": hessian,
            "train": reconstruction[: config.train_samples].clone(),
            "validation": reconstruction[config.train_samples :].clone(),
        },
        config.calibration,
    )


def load_calibration(config: BlockwiseConfig) -> dict[str, Any]:
    artifact = torch.load(config.calibration, map_location="cpu", weights_only=True, mmap=True)
    if artifact["protocol"] != config.calibration_recipe():
        raise RuntimeError("calibration protocol differs; use a new output_dir")
    return artifact
