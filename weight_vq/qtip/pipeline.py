"""QTIP stages for dense Qwen3 and Qwen3-VL QDQ export."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from operator import attrgetter
from pathlib import Path
from typing import Any, Literal, cast

import torch
from lib.algo import finetune
from lib.codebook.bitshift import bitshift_codebook
from torch import nn
from torch.multiprocessing.spawn import spawn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from weight_vq import utils
from weight_vq.qtip.artifacts import (
    PROJECTIONS,
    atomic_write_completion,
    hessian_complete,
    layer_complete,
    projection_paths,
    quant_order,
    reconstruct_dense_weight,
)
from weight_vq.qtip.calibration import local_batch_slice
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.distributed import process_group
from weight_vq.qtip.hessian import LayerHessianCollectors
from weight_vq.qtip.reconstruction import DistributedReconstruction
from weight_vq.qtip.replay import LayerReplay, capture_replay

Stage = Literal["hessian", "quantize"]
ModelFamily = Literal["qwen3", "qwen3_vl"]


def _model_contract(model: Path) -> tuple[ModelFamily, int]:
    """Return model family and number of hidden layers."""
    hf_config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    if hf_config.model_type == "qwen3":
        return "qwen3", int(hf_config.num_hidden_layers)
    if hf_config.model_type == "qwen3_vl":
        return "qwen3_vl", int(hf_config.text_config.num_hidden_layers)
    raise ValueError(f"QTIP supports only qwen3 and qwen3_vl, got {hf_config.model_type!r}")


def _get_layers(model: nn.Module, family: ModelFamily) -> nn.ModuleList:
    if family == "qwen3":
        return cast(Any, model).model.layers
    return cast(Any, model).model.language_model.layers


def layer_range(config: QTIPConfig, model_layers: int) -> tuple[int, int]:
    end = model_layers if config.layer_end is None else config.layer_end
    return config.layer_start, end


def _tlut_path(config: QTIPConfig) -> Path:
    return config.checkpoints / (f"tlut-l{config.L}-b{config.tlut_bits}-v{config.V}.pt")


def _initialize_run(config: QTIPConfig, family: ModelFamily, end: int) -> None:
    config.hessians.mkdir(parents=True, exist_ok=True)
    config.checkpoints.mkdir(parents=True, exist_ok=True)
    recipe = config.recipe(model_family=family, layer_end=end)
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


def _run_rank(
    rank: int,
    config: QTIPConfig,
    stage: Stage,
    family: ModelFamily,
    start: int,
    end: int,
    rendezvous: Path,
    model: nn.Module,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QTIP hessian and quantize stages require CUDA")

    with process_group(rank, config.devices, rendezvous) as distributed:
        torch.manual_seed(config.seed)
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
        layers = _get_layers(model, family)
        train = capture_replay(
            model,
            config,
            family=family,
            split="train",
            device=distributed.device,
            rank=rank,
            world_size=distributed.world_size,
        )
        validation = (
            capture_replay(
                model,
                config,
                family=family,
                split="validation",
                device=distributed.device,
                rank=rank,
                world_size=distributed.world_size,
            )
            if stage == "quantize" and config.reconstruction_epochs
            else None
        )
        utils.clear_cuda_cache()

        for index in range(end):
            layer = layers[index].to(distributed.device).eval()
            selected = start <= index < end
            collectors = (
                LayerHessianCollectors(layer, include_padding=config.include_padding_in_hessian)
                if stage == "hessian" and selected and not hessian_complete(index, config.hessians)
                else None
            )
            quantizing = (
                stage == "quantize" and selected and not layer_complete(index, config.checkpoints)
            )
            retain_layer = quantizing and config.reconstruction_epochs > 0
            train_layer = train.forward_original(
                layer,
                index,
                device=distributed.device,
                collectors=collectors,
                autocast_dtype=dtype,
                retain_layer=retain_layer,
            )
            if collectors is not None:
                payloads = collectors.close(distributed)
                if distributed.is_primary:
                    for name, payload in payloads.items():
                        utils.atomic_torch_save(payload, config.hessians / f"{index}_{name}.pt")
                distributed.barrier()
                del payloads
                utils.clear_cuda_cache()
            del collectors

            validation_layer = (
                validation.forward_original(
                    layer,
                    index,
                    device=distributed.device,
                    autocast_dtype=dtype,
                    retain_layer=retain_layer,
                )
                if validation is not None
                else None
            )
            if quantizing:
                if distributed.is_primary:
                    (config.checkpoints / f"{index}.complete").unlink(missing_ok=True)
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)
                resume_path = config.checkpoints / f"{index}.resume.pt"
                resume_state = (
                    torch.load(resume_path, map_location="cpu", weights_only=True)
                    if resume_path.is_file()
                    else None
                )

                def save_progress(
                    completed: int, projections: dict[str, dict[str, Any]], path: Path = resume_path
                ) -> None:
                    if distributed.is_primary:
                        utils.atomic_torch_save(
                            {"completed": completed, "projections": projections}, path
                        )
                    distributed.barrier()

                reconstruction = (
                    DistributedReconstruction(
                        train=cast(LayerReplay, train_layer),
                        validation=cast(LayerReplay, validation_layer),
                        config=config,
                        distributed=distributed,
                        seed=config.seed + index,
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
                    distributed.device,
                    None,
                    None,
                    reconstruction_callback=reconstruction,
                    resume_state=resume_state,
                    progress_callback=save_progress,
                    write_checkpoints=distributed.is_primary,
                )
                if distributed.is_primary:
                    missing = [
                        path
                        for path in projection_paths(index, config.checkpoints)
                        if not path.is_file()
                    ]
                    if missing:
                        raise RuntimeError(f"layer {index} missing checkpoints: {missing}")
                    resume_path.unlink(missing_ok=True)
                    atomic_write_completion(index, config.checkpoints)
                distributed.barrier()
                del reconstruction, resume_state

            del train_layer, validation_layer
            layers[index] = nn.Identity()
            del layer
            utils.clear_cuda_cache()


def _run_workers(config: QTIPConfig, stage: Stage) -> None:
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    if not config.devices:
        raise ValueError("devices must contain at least one CUDA device")
    local_batch_slice(config.batch_size, 0, len(config.devices))
    if stage == "quantize":
        missing = [
            index for index in range(start, end) if not hessian_complete(index, config.hessians)
        ]
        if missing:
            raise FileNotFoundError(f"missing QTIP Hessians for layers: {missing}")
    _initialize_run(config, family, end)
    model_loader = AutoModelForCausalLM if family == "qwen3" else AutoModelForImageTextToText
    model = (
        cast(
            nn.Module,
            model_loader.from_pretrained(
                config.model,
                device_map="cpu",
                dtype=cast(torch.dtype, getattr(torch, config.model_dtype)),
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
                attn_implementation="eager",
            ),
        )
        .eval()
        .requires_grad_(False)
    )
    # Spawn shares CPU storage; moving a worker's layer to CUDA gives it private weights.
    # Keep the parent model alive until all workers finish using the shared storage.
    model.share_memory()
    with tempfile.TemporaryDirectory(prefix="wvq-qtip-") as temporary:
        rendezvous = Path(temporary, "store")
        spawn(
            _run_rank,
            args=(config, stage, family, start, end, rendezvous, model),
            nprocs=len(config.devices),
            join=True,
        )

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
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    if config.packed_inference_kernel or config.split_for_tp:
        raise ValueError("dense export cannot represent packed or tensor-parallel checkpoints")
    missing = [
        index for index in range(start, end) if not layer_complete(index, config.checkpoints)
    ]
    if missing:
        raise RuntimeError(f"QTIP checkpoints are incomplete for layers: {missing}")
    if config.exported_model.exists():
        raise FileExistsError(f"refusing to overwrite {config.exported_model}")

    model_loader = AutoModelForCausalLM if family == "qwen3" else AutoModelForImageTextToText
    model = model_loader.from_pretrained(
        config.model,
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    ).eval()
    layers = _get_layers(model, family)
    recipe = json.loads((config.checkpoints / "qtip_config.json").read_text())
    with torch.no_grad():
        for layer_index in range(start, end):
            layer = layers[layer_index]
            for attribute, short, _hessian, _parallel in PROJECTIONS:
                saved = torch.load(
                    config.checkpoints / f"{layer_index}_{short}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                linear = attrgetter(attribute)(layer)
                linear.weight.copy_(reconstruct_dense_weight(saved, recipe).to(linear.weight.dtype))

    staging = config.exported_model.with_name(f".{config.exported_model.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        staging.mkdir(parents=True)
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        if family == "qwen3":
            assets = AutoTokenizer.from_pretrained(
                config.model, local_files_only=True, trust_remote_code=True
            )
        else:
            assets = AutoProcessor.from_pretrained(
                config.model, local_files_only=True, trust_remote_code=True
            )
        assets.save_pretrained(staging)
        shutil.copy2(config.checkpoints / "qtip_config.json", staging / "qtip_config.json")
        config.exported_model.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, config.exported_model)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate(config: QTIPConfig) -> None:
    """Load the dense artifact through the evaluation-facing HF API."""
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    model_loader = AutoModelForCausalLM if family == "qwen3" else AutoModelForImageTextToText
    model = model_loader.from_pretrained(
        config.exported_model,
        dtype="auto",
        device_map=0,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    layers = _get_layers(model, family)
    layer_prefix = "model.layers" if family == "qwen3" else "model.language_model.layers"
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
                        f"{layer_prefix}.{index}.{attribute}",
                        *(f"{value.item():.5e}" for value in quantiles),
                        *(f"{value.item():.5e}" for value in abs_quantiles),
                        f"{weight.abs().max().item():.5e}",
                    )
                )
