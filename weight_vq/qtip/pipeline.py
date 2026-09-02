"""QTIP stages for dense Qwen3-VL QDQ export."""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal, cast

import torch
from lib.algo import finetune
from lib.codebook.bitshift import bitshift_codebook
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from weight_vq import utils
from weight_vq.qtip.artifacts import (
    PROJECTIONS,
    atomic_write_completion,
    hessian_complete,
    layer_complete,
    projection_paths,
    quant_order,
)
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.hessian import LayerHessianCollectors
from weight_vq.qtip.qwen3_vl import (
    EXPECTED_LAYERS,
    audit_qwen3_vl,
    get_layers,
    inject_dense_checkpoints,
)
from weight_vq.qtip.replay import ReconstructionAdapter, capture_replay

Stage = Literal["hessian", "quantize"]


def layer_range(config: QTIPConfig) -> tuple[int, int]:
    end = EXPECTED_LAYERS if config.layer_end is None else config.layer_end
    return config.layer_start, end


def split_layer_ranges(start: int, end: int, devices: list[int]) -> list[tuple[int, int, int]]:
    if not devices:
        raise ValueError("devices must contain at least one CUDA device")
    worker_count = min(len(devices), end - start)
    width, remainder = divmod(end - start, worker_count)
    ranges = []
    cursor = start
    for worker_index, device in enumerate(devices[:worker_count]):
        next_cursor = cursor + width + (worker_index < remainder)
        ranges.append((device, cursor, next_cursor))
        cursor = next_cursor
    return ranges


def _tlut_path(config: QTIPConfig) -> Path:
    return config.checkpoints / (f"tlut-l{config.L}-b{config.tlut_bits}-v{config.V}.pt")


def _initialize_run(config: QTIPConfig) -> None:
    config.hessians.mkdir(parents=True, exist_ok=True)
    config.checkpoints.mkdir(parents=True, exist_ok=True)
    _, end = layer_range(config)
    recipe = config.recipe(model_family="qwen3_vl", layer_end=end)
    recipe_path = config.checkpoints / "qtip_config.json"
    if recipe_path.is_file():
        if json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError(
                "configuration differs from checkpoints/qtip_config.json; "
                "use a new checkpoint directory"
            )
    else:
        utils.atomic_write_text(recipe_path, json.dumps(recipe, indent=2) + "\n")

    tlut_path = _tlut_path(config)
    if not tlut_path.is_file():
        torch.manual_seed(config.seed)
        bitshift_codebook(
            L=config.L,
            K=config.K,
            V=config.V,
            tlut_bits=config.tlut_bits,
            decode_mode=config.decode_mode,
            tlut_path=tlut_path,
        )


def _run_shard(
    config: QTIPConfig, stage: Stage, physical_device: int, start: int, end: int
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QTIP hessian and quantize stages require CUDA")
    if stage == "quantize":
        missing = [
            index for index in range(start, end) if not hessian_complete(index, config.hessians)
        ]
        if missing:
            raise FileNotFoundError(f"missing QTIP Hessians for layers: {missing}")

    torch.manual_seed(config.seed)
    device = torch.device(f"cuda:{physical_device}")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(config.seed)
    dtype = cast(torch.dtype, getattr(torch, config.model_dtype))
    codebook = bitshift_codebook(
        L=config.L,
        K=config.K,
        V=config.V,
        tlut_bits=config.tlut_bits,
        decode_mode=config.decode_mode,
        tlut_path=_tlut_path(config),
    )
    model = cast(
        nn.Module,
        AutoModelForImageTextToText.from_pretrained(
            config.model,
            device_map="cpu",
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager",
        ),
    ).eval()
    audit_qwen3_vl(model)
    layers = get_layers(model)
    model.to(device)
    train = capture_replay(model, layers, config, split="train", device=device)
    validation = (
        capture_replay(model, layers, config, split="validation", device=device)
        if stage == "quantize" and config.reconstruction_epochs
        else None
    )
    model.cpu()
    utils.clear_cuda_cache()

    for index in range(end):
        layer = layers[index].to(device).eval()
        selected = start <= index < end
        collectors = (
            LayerHessianCollectors(layer, include_padding=config.include_padding_in_hessian)
            if stage == "hessian" and selected and not hessian_complete(index, config.hessians)
            else None
        )
        train_layer = train.forward_original(
            layer, index, device=device, collectors=collectors, autocast_dtype=dtype
        )
        if collectors is not None:
            for name, payload in collectors.close().items():
                utils.atomic_torch_save(payload, config.hessians / f"{index}_{name}.pt")
            utils.clear_cuda_cache()

        validation_layer = (
            validation.forward_original(layer, index, device=device, autocast_dtype=dtype)
            if validation is not None
            else None
        )
        if stage == "quantize" and selected and not layer_complete(index, config.checkpoints):
            (config.checkpoints / f"{index}.complete").unlink(missing_ok=True)
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
            reconstruction = (
                ReconstructionAdapter(
                    train=train_layer,
                    validation=validation_layer,
                    seed=config.seed + index,
                    mask_padding=config.mask_padding,
                )
                if validation_layer is not None
                else None
            )
            finetune.quantize_finetune_decoder_layer(
                layer,
                quant_order(),
                index,
                codebook,
                config,
                device,
                None,
                None,
                reconstruction_adapter=reconstruction,
            )
            missing = [
                path for path in projection_paths(index, config.checkpoints) if not path.is_file()
            ]
            if missing:
                raise RuntimeError(f"layer {index} missing checkpoints: {missing}")
            atomic_write_completion(index, config.checkpoints)

        layers[index] = nn.Identity()
        utils.clear_cuda_cache()

    # TODO: recover this only when unified logging has a clear design
    # print(
    #     json.dumps(
    #         {
    #             "stage": stage,
    #             "physical_device": physical_device,
    #             "layer_range": [start, end],
    #             "train_sample_ids": train.sample_ids,
    #             "validation_sample_ids": validation.sample_ids if validation is not None else [],
    #         },
    #         indent=2,
    #     )
    # )


def _run_workers(config: QTIPConfig, stage: Stage) -> None:
    start, end = layer_range(config)
    ranges = split_layer_ranges(start, end, config.devices)
    _initialize_run(config)
    context = multiprocessing.get_context("spawn")
    processes = [
        (
            device,
            shard_start,
            shard_end,
            context.Process(
                target=_run_shard, args=(config, stage, device, shard_start, shard_end)
            ),
        )
        for device, shard_start, shard_end in ranges
    ]
    for _device, _start, _end, process in processes:
        process.start()

    failures = []
    for device, shard_start, shard_end, process in processes:
        process.join()
        if process.exitcode:
            failures.append((device, shard_start, shard_end, process.exitcode))
    if failures:
        raise RuntimeError(f"QTIP {stage} workers failed: {failures}")

    complete = hessian_complete if stage == "hessian" else layer_complete
    directory = config.hessians if stage == "hessian" else config.checkpoints
    missing = [index for index in range(start, end) if not complete(index, directory)]
    if missing:
        raise RuntimeError(f"QTIP {stage} did not complete layers: {missing}")


def hessian(config: QTIPConfig) -> None:
    _run_workers(config, "hessian")


def quantize(config: QTIPConfig) -> None:
    _run_workers(config, "quantize")


def export(config: QTIPConfig) -> None:
    start, end = layer_range(config)
    if config.packed_inference_kernel or config.split_for_tp:
        raise ValueError("dense export cannot represent packed or tensor-parallel checkpoints")
    missing = [
        index for index in range(start, end) if not layer_complete(index, config.checkpoints)
    ]
    if missing:
        raise RuntimeError(f"QTIP checkpoints are incomplete for layers: {missing}")
    if config.exported_model.exists():
        raise FileExistsError(f"refusing to overwrite {config.exported_model}")

    model = AutoModelForImageTextToText.from_pretrained(
        config.model,
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    ).eval()
    audit_qwen3_vl(model)
    recipe = json.loads((config.checkpoints / "qtip_config.json").read_text())
    modified = inject_dense_checkpoints(
        model, config.checkpoints, recipe, layer_start=start, layer_end=end
    )
    expected = (end - start) * len(PROJECTIONS)
    if len(modified) != expected:
        raise RuntimeError(f"export modified {len(modified)} projections, expected {expected}")

    staging = config.exported_model.with_name(f".{config.exported_model.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        staging.mkdir(parents=True)
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        processor = AutoProcessor.from_pretrained(
            config.model, local_files_only=True, trust_remote_code=True
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
    start, end = layer_range(config)
    model = AutoModelForImageTextToText.from_pretrained(
        config.exported_model,
        dtype="auto",
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
                    sample, torch.tensor([0.01, 0.5, 0.99], device=sample.device)
                )
                abs_quantiles = torch.quantile(
                    sample.abs(), torch.tensor([0.99, 0.999], device=sample.device)
                )
                writer.writerow(
                    (
                        f"model.language_model.layers.{index}.{attribute}",
                        *(f"{value.item():.5e}" for value in quantiles),
                        *(f"{value.item():.5e}" for value in abs_quantiles),
                        f"{weight.abs().max().item():.5e}",
                    )
                )
