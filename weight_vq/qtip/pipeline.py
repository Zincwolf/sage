"""QTIP language-model calibration through HF dense QDQ export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from operator import attrgetter
from pathlib import Path
from typing import Literal, cast

import torch
from lib.algo import finetune
from lib.codebook.bitshift import bitshift_codebook
from torch import nn
from torch.multiprocessing.spawn import spawn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weight_vq import utils
from weight_vq.qtip.artifacts import (
    PROJECTIONS,
    atomic_write_completion,
    hessian_complete,
    layer_complete,
    quant_order,
    reconstruct_dense_weight,
)
from weight_vq.qtip.calibration import calibration, load_calibration
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.distributed import process_group
from weight_vq.qtip.hessian import LayerHessianCollectors
from weight_vq.qtip.reconstruction import DistributedReconstruction
from weight_vq.qtip.replay import capture_replay

Stage = Literal["hessian", "quantize"]


def _model_contract(model: Path) -> tuple[str, int]:
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError(f"QTIP currently supports qwen3, got {config.model_type!r}")
    return "qwen3", int(config.num_hidden_layers)


def layer_range(config: QTIPConfig, model_layers: int) -> tuple[int, int]:
    return config.layer_start, model_layers if config.layer_end is None else config.layer_end


def _tlut_path(config: QTIPConfig) -> Path:
    return config.checkpoints / f"tlut-l{config.L}-b{config.tlut_bits}-v{config.V}.pt"


def _initialize_run(config: QTIPConfig, family: str, end: int) -> None:
    config.checkpoints.mkdir(parents=True, exist_ok=True)
    recipe = config.recipe(model_family=family, layer_end=end)
    recipe_path = config.checkpoints / "qtip_config.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError("configuration differs from checkpoints; use a new output_dir")
    else:
        utils.atomic_write_text(recipe_path, json.dumps(recipe, indent=2) + "\n")
    if not _tlut_path(config).exists():
        torch.manual_seed(config.seed)
        bitshift_codebook(
            L=config.L,
            K=config.K,
            V=config.V,
            tlut_bits=config.tlut_bits,
            decode_mode=config.decode_mode,
            tlut_path=_tlut_path(config),
        )


def _run_rank(
    rank: int,
    config: QTIPConfig,
    stage: Stage,
    family: str,
    start: int,
    end: int,
    rendezvous: Path,
    hessian_dir: Path,
) -> None:
    torch.set_num_threads(config.num_cpu_threads)
    with process_group(rank, config.devices, rendezvous) as distributed:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        dtype = cast(torch.dtype, getattr(torch, config.model_dtype))
        # Each worker loads independent CPU storage; no model tensors cross spawn.
        model = (
            AutoModelForCausalLM.from_pretrained(
                config.model,
                device_map="cpu",
                dtype=dtype,
                local_files_only=True,
                attn_implementation="sdpa",
            )
            .eval()
            .requires_grad_(False)
        )
        layers = model.model.layers
        data = load_calibration(config)
        if stage == "hessian":
            accumulators: dict[tuple[int, str], tuple[torch.Tensor, int]] = {}
            for chunk in data["hessian"].split(config.hessian_chunk_size):
                state = capture_replay(
                    model,
                    chunk[rank :: distributed.world_size],
                    device=distributed.device,
                    microbatch_size=config.forward_microbatch_size,
                    placement="cuda",
                )
                for index in range(end):
                    layer = layers[index].to(distributed.device)
                    collectors = (
                        LayerHessianCollectors(layer, include_padding=True)
                        if index >= start
                        else None
                    )
                    replay = state.forward_original(
                        layer, device=distributed.device, collectors=collectors
                    )
                    del replay
                    layer.cpu()
                    if collectors is not None:
                        for name, collector in collectors.collectors.items():
                            matrix, count = collector.close_raw(distributed)
                            if distributed.is_primary:
                                key = (index, name)
                                if key in accumulators:
                                    total, tokens = accumulators[key]
                                    total.add_(matrix)
                                    accumulators[key] = total, tokens + count
                                else:
                                    accumulators[key] = matrix, count
                            del matrix
                    del collectors, layer
                    utils.clear_cuda_cache()
                del state
                utils.clear_cuda_cache()
            if distributed.is_primary:
                for (index, name), (matrix, count) in accumulators.items():
                    matrix.div_(count)
                    # Avoid two quadratic int64 index arrays for the packed triangle.
                    flat = torch.cat([row[: i + 1] for i, row in enumerate(matrix)])
                    utils.atomic_torch_save(
                        {"flatH": flat, "n": matrix.shape[0], "ct": count, "position_mask": "all"},
                        hessian_dir / f"{index}_{name}.pt",
                    )
                utils.atomic_write_text(
                    hessian_dir / "complete.json",
                    json.dumps(config.hessian_recipe(end), indent=2) + "\n",
                )
            distributed.barrier()
            return

        codebook = bitshift_codebook(
            L=config.L,
            K=config.K,
            V=config.V,
            tlut_bits=config.tlut_bits,
            decode_mode=config.decode_mode,
            tlut_path=_tlut_path(config),
        )
        train = capture_replay(
            model,
            data["train"][rank :: distributed.world_size],
            device=distributed.device,
            microbatch_size=config.forward_microbatch_size,
            placement=config.replay_device,
        )
        validation = capture_replay(
            model,
            data["validation"][rank :: distributed.world_size],
            device=distributed.device,
            microbatch_size=config.forward_microbatch_size,
            placement=config.replay_device,
        )
        for index in range(end):
            layer = layers[index].to(distributed.device).eval()
            # Both boundaries advance through the unmodified teacher before training.
            train_layer = train.forward_original(layer, device=distributed.device)
            validation_layer = validation.forward_original(layer, device=distributed.device)
            if index >= start and not layer_complete(index, config.checkpoints):
                if distributed.is_primary:
                    (config.checkpoints / f"{index}.complete").unlink(missing_ok=True)
                layer.requires_grad_(True)
                reconstruction = DistributedReconstruction(
                    train_layer, validation_layer, config, distributed, config.seed + index
                )
                finetune.quantize_finetune_decoder_layer(
                    layer,
                    quant_order(),
                    index,
                    codebook,
                    config,
                    distributed.device,
                    None,
                    None,
                    reconstruction_callback=reconstruction,
                    write_checkpoints=distributed.is_primary,
                )
                if distributed.is_primary:
                    prefixes = tuple(f"{attribute}." for attribute, *_ in PROJECTIONS)
                    auxiliary = {
                        name: value.detach().cpu().clone()
                        for name, value in layer.state_dict().items()
                        if not name.startswith(prefixes)
                    }
                    utils.atomic_torch_save(auxiliary, config.checkpoints / f"{index}.auxiliary.pt")
                    atomic_write_completion(index, config.checkpoints)
                distributed.barrier()
                del reconstruction
            del train_layer, validation_layer
            layers[index] = nn.Identity()
            del layer
            utils.clear_cuda_cache()


def _run_workers(config: QTIPConfig, stage: Stage) -> None:
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    load_calibration(config)
    marker = config.hessians / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text()) != config.hessian_recipe(end):
            raise RuntimeError("Hessian protocol differs; use a new output_dir")
        if all(hessian_complete(index, config.hessians) for index in range(start, end)):
            if stage == "hessian":
                return
        else:
            raise RuntimeError("published Hessian artifact is incomplete")
    elif stage == "quantize":
        raise FileNotFoundError("a fully published Hessian stage is required")
    if stage == "quantize":
        _initialize_run(config, family, end)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Staging lives beside the final artifact, on the configured large-data filesystem.
    with tempfile.TemporaryDirectory(prefix=f".{stage}-", dir=config.output_dir) as temporary:
        working = Path(temporary)
        hessian_dir = working / "hessians"
        hessian_dir.mkdir()
        spawn(
            _run_rank,
            args=(config, stage, family, start, end, working / "store", hessian_dir),
            nprocs=len(config.devices),
            join=True,
        )
        if stage == "hessian":
            os.replace(hessian_dir, config.hessians)
    if stage == "quantize":
        missing = [
            index for index in range(start, end) if not layer_complete(index, config.checkpoints)
        ]
        if missing:
            raise RuntimeError(f"QTIP quantize did not complete layers: {missing}")


def hessian(config: QTIPConfig) -> None:
    _run_workers(config, "hessian")


def quantize(config: QTIPConfig) -> None:
    _run_workers(config, "quantize")


def export(config: QTIPConfig) -> None:
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    recipe = json.loads((config.checkpoints / "qtip_config.json").read_text())
    if recipe != config.recipe(model_family=family, layer_end=end):
        raise RuntimeError("export configuration differs from checkpoint recipe")
    missing = [
        index for index in range(start, end) if not layer_complete(index, config.checkpoints)
    ]
    if missing:
        raise RuntimeError(f"QTIP checkpoints are incomplete for layers: {missing}")
    if config.exported_model.exists():
        raise FileExistsError(f"refusing to overwrite {config.exported_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype="auto", device_map="cpu", local_files_only=True
    ).eval()
    layers = model.model.layers
    with torch.no_grad():
        for index in range(start, end):
            layer = layers[index]
            for attribute, short, *_ in PROJECTIONS:
                saved = torch.load(
                    config.checkpoints / f"{index}_{short}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                linear = attrgetter(attribute)(layer)
                linear.weight.copy_(reconstruct_dense_weight(saved, recipe).to(linear.weight.dtype))
            auxiliary = torch.load(
                config.checkpoints / f"{index}.auxiliary.pt", map_location="cpu", weights_only=True
            )
            state = layer.state_dict()
            for name, value in auxiliary.items():
                state[name].copy_(value)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=config.output_dir) as temporary:
        staging = Path(temporary) / "model"
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(config.model, local_files_only=True).save_pretrained(staging)
        shutil.copy2(config.checkpoints / "qtip_config.json", staging / "qtip_config.json")
        os.replace(staging, config.exported_model)


def validate(config: QTIPConfig) -> None:
    _, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    utils.validate_dense_model(
        config.exported_model,
        model_loader=AutoModelForCausalLM,
        weights=lambda model: (
            (name, model.get_parameter(f"{name}.weight"))
            for index in range(start, end)
            for attribute, *_ in PROJECTIONS
            for name in (f"model.layers.{index}.{attribute}",)
        ),
    )


def run(config: QTIPConfig) -> None:
    calibration(config)
    hessian(config)
    quantize(config)
    export(config)
    validate(config)
