"""Shared language-model calibration through HF dense QDQ export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from copy import deepcopy
from operator import attrgetter
from pathlib import Path
from typing import Literal, cast

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
from weight_vq.blockwise.artifacts import (
    PROJECTIONS,
    atomic_write_completion,
    hessian_complete,
    layer_complete,
    quant_order,
)
from weight_vq.blockwise.calibration import calibration, load_calibration
from weight_vq.blockwise.distributed import process_group
from weight_vq.blockwise.hessian import LayerHessianCollectors
from weight_vq.blockwise.models import language_backbone
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.artifacts import reconstruct_dense_weight
from weight_vq.qtip.config import QTIPConfig
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.quip_sharp.quantization import quantize_layer
from weight_vq.quip_sharp.quantization import reconstruct_dense_weight as reconstruct_quip_weight
from weight_vq.unisvq.config import UniSVQConfig
from weight_vq.unisvq.quantization import quantize_layer as quantize_unisvq_layer
from weight_vq.unisvq.quantization import reconstruct_dense_weight as reconstruct_unisvq_weight
from weight_vq.unisvq.reconstruction import UniSVQReconstruction

Config = QTIPConfig | QuIPSharpConfig | UniSVQConfig
Stage = Literal["hessian", "quantize"]


def _model_contract(model: Path) -> tuple[str, int]:
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type == "qwen3":
        return "qwen3", int(config.num_hidden_layers)
    if config.model_type == "mistral3" and config.text_config.model_type == "ministral3":
        return "mistral3", int(config.text_config.num_hidden_layers)
    raise ValueError(
        f"Blockwise pipeline supports qwen3 and Ministral-3, got {config.model_type!r}"
    )


def model_loader(family: str):
    return AutoModelForImageTextToText if family == "mistral3" else AutoModelForCausalLM


def layer_range(config: Config, model_layers: int) -> tuple[int, int]:
    return config.layer_start, model_layers if config.layer_end is None else config.layer_end


def _tlut_path(config: QTIPConfig) -> Path:
    return config.checkpoints / f"tlut-l{config.L}-b{config.tlut_bits}-v{config.V}.pt"


def _initialize_run(config: Config, family: str, end: int) -> None:
    config.checkpoints.mkdir(parents=True, exist_ok=True)
    recipe = config.recipe(model_family=family, layer_end=end)
    method = {QTIPConfig: "qtip", QuIPSharpConfig: "quip_sharp", UniSVQConfig: "unisvq"}[
        type(config)
    ]
    recipe_path = config.checkpoints / f"{method}_config.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError("configuration differs from checkpoints; use a new output_dir")
    else:
        utils.atomic_write_text(recipe_path, json.dumps(recipe, indent=2) + "\n")
    if isinstance(config, QTIPConfig) and not _tlut_path(config).exists():
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
    config: Config,
    stage: Stage,
    model: nn.Module,
    start: int,
    end: int,
    rendezvous: Path,
    hessian_dir: Path,
) -> None:
    torch.set_num_threads(config.num_cpu_threads)
    with process_group(rank, config.devices, rendezvous) as distributed:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        # Spawn gives each rank its own module shell, backed by the parent's
        # shared CPU storage. Never move or train these teacher modules in place.
        layers = language_backbone(model).layers
        data = load_calibration(config)
        if stage == "hessian":
            centered = isinstance(config, (QuIPSharpConfig, UniSVQConfig))
            statistics_dtype = (
                cast(torch.dtype, getattr(torch, config.hessian_dtype))
                if isinstance(config, UniSVQConfig)
                else torch.float32
            )
            accumulators: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, int]] = {}
            for chunk in data["hessian"].split(config.hessian_chunk_size):
                state = capture_replay(
                    model,
                    chunk[rank :: distributed.world_size],
                    device=distributed.device,
                    microbatch_size=config.forward_microbatch_size,
                    placement="cuda",
                )
                for index in range(end):
                    layer = deepcopy(layers[index]).to(distributed.device)
                    collectors = (
                        LayerHessianCollectors(layer, include_padding=True, dtype=statistics_dtype)
                        if index >= start
                        else None
                    )
                    replay = state.forward_original(
                        layer, device=distributed.device, collectors=collectors
                    )
                    del replay
                    if collectors is not None:
                        for name, collector in collectors.collectors.items():
                            matrix, first_moment, count = collector.close_statistics(distributed)
                            if distributed.is_primary:
                                key = (index, name)
                                if key in accumulators:
                                    total, first_total, tokens = accumulators[key]
                                    total.add_(matrix)
                                    first_total.add_(first_moment)
                                    accumulators[key] = total, first_total, tokens + count
                                else:
                                    accumulators[key] = matrix, first_moment, count
                            del matrix, first_moment
                    del collectors, layer
                    utils.clear_cuda_cache()
                del state
                utils.clear_cuda_cache()
            if distributed.is_primary:
                for (index, name), (matrix, first_moment, count) in accumulators.items():
                    matrix.div_(count)
                    if centered:
                        first_moment.div_(count)
                        matrix.sub_(first_moment[:, None] * first_moment[None, :])
                    # Avoid two quadratic int64 index arrays for the packed triangle.
                    flat = torch.cat([row[: i + 1] for i, row in enumerate(matrix)])
                    utils.atomic_torch_save(
                        {
                            "flatH": flat,
                            "n": matrix.shape[0],
                            "ct": count,
                            "position_mask": "all",
                            **({"mu": first_moment} if centered else {}),
                        },
                        hessian_dir / f"{index}_{name}.pt",
                    )
                utils.atomic_write_text(
                    hessian_dir / "complete.json",
                    json.dumps(config.hessian_recipe(end), indent=2) + "\n",
                )
            distributed.barrier()
            return

        codebook = None
        if isinstance(config, QTIPConfig):
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
            # Only this private work layer is replaced/trained by the quantizer.
            layer = deepcopy(layers[index]).to(distributed.device).eval()
            # Both boundaries advance through the unmodified teacher before training.
            train_layer = train.forward_original(layer, device=distributed.device)
            validation_layer = validation.forward_original(layer, device=distributed.device)
            if index >= start and not layer_complete(index, config.checkpoints):
                if distributed.is_primary:
                    (config.checkpoints / f"{index}.complete").unlink(missing_ok=True)
                layer.requires_grad_(True)
                trainer = (
                    UniSVQReconstruction
                    if isinstance(config, UniSVQConfig)
                    else DistributedReconstruction
                )
                reconstruction = trainer(
                    train_layer, validation_layer, config, distributed, config.seed + index
                )
                if isinstance(config, QTIPConfig):
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
                elif isinstance(config, UniSVQConfig):
                    quantize_unisvq_layer(
                        layer, index, config, cast(UniSVQReconstruction, reconstruction)
                    )
                else:
                    quantize_layer(layer, index, config, reconstruction)
                if distributed.is_primary:
                    prefixes = tuple(f"{attribute}." for attribute, *_ in PROJECTIONS)
                    auxiliary = {
                        name: value.detach().cpu().clone()
                        for name, value in layer.state_dict().items()
                        if not name.startswith(prefixes) or name.endswith(".bias")
                    }
                    if isinstance(config, UniSVQConfig):
                        # Released quantize_llama_layer/hfize persist only these norms.
                        # q/k norm parameters are trained upstream but discarded at export;
                        # preserve that behavior rather than silently improve the method.
                        auxiliary = {
                            name: value
                            for name, value in auxiliary.items()
                            if name in ("input_layernorm.weight", "post_attention_layernorm.weight")
                            or name.endswith(".bias")
                        }
                    utils.atomic_torch_save(auxiliary, config.checkpoints / f"{index}.auxiliary.pt")
                    atomic_write_completion(index, config.checkpoints)
                distributed.barrier()
                del reconstruction
            del train_layer, validation_layer
            del layer
            utils.clear_cuda_cache()


def _run_workers(config: Config, stage: Stage) -> None:
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
    torch.set_num_threads(config.num_cpu_threads)
    # Load once per stage. PyTorch's CPU shared storage survives spawn without
    # per-rank weight copies; the parent retains it until every worker exits.
    model = (
        model_loader(family)
        .from_pretrained(
            config.model,
            device_map="cpu",
            dtype=cast(torch.dtype, getattr(torch, config.model_dtype)),
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
        .share_memory()
    )
    # Staging lives beside the final artifact, on the configured large-data filesystem.
    with tempfile.TemporaryDirectory(prefix=f".{stage}-", dir=config.output_dir) as temporary:
        working = Path(temporary)
        hessian_dir = working / "hessians"
        hessian_dir.mkdir()
        spawn(
            _run_rank,
            args=(config, stage, model, start, end, working / "store", hessian_dir),
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
            raise RuntimeError(f"quantize did not complete layers: {missing}")


def hessian(config: Config) -> None:
    _run_workers(config, "hessian")


def quantize(config: Config) -> None:
    _run_workers(config, "quantize")


def export(config: Config) -> None:
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    method = {QTIPConfig: "qtip", QuIPSharpConfig: "quip_sharp", UniSVQConfig: "unisvq"}[
        type(config)
    ]
    recipe_name = f"{method}_config.json"
    recipe = json.loads((config.checkpoints / recipe_name).read_text())
    if recipe != config.recipe(model_family=family, layer_end=end):
        raise RuntimeError("export configuration differs from checkpoint recipe")
    missing = [
        index for index in range(start, end) if not layer_complete(index, config.checkpoints)
    ]
    if missing:
        raise RuntimeError(f"checkpoints are incomplete for layers: {missing}")
    if config.exported_model.exists():
        raise FileExistsError(f"refusing to overwrite {config.exported_model}")
    model = (
        model_loader(family)
        .from_pretrained(config.model, dtype="auto", device_map="cpu", local_files_only=True)
        .eval()
    )
    layers = language_backbone(model).layers
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
                if isinstance(config, QTIPConfig):
                    weight = reconstruct_dense_weight(saved, recipe)
                elif isinstance(config, UniSVQConfig):
                    weight = reconstruct_unisvq_weight(
                        saved, torch.device(f"cuda:{config.devices[0]}")
                    )
                else:
                    weight = reconstruct_quip_weight(
                        saved, torch.device(f"cuda:{config.devices[0]}")
                    )
                linear.weight.copy_(weight.to(linear.weight.dtype))
                del weight
            auxiliary = torch.load(
                config.checkpoints / f"{index}.auxiliary.pt", map_location="cpu", weights_only=True
            )
            state = layer.state_dict()
            for name, value in auxiliary.items():
                state[name].copy_(value)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=config.output_dir) as temporary:
        staging = Path(temporary) / "model"
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        if family == "mistral3":
            AutoProcessor.from_pretrained(
                config.model, local_files_only=True, fix_mistral_regex=True
            ).save_pretrained(staging)
            shutil.copy2(config.model / "tekken.json", staging / "tekken.json")
        else:
            AutoTokenizer.from_pretrained(config.model, local_files_only=True).save_pretrained(
                staging
            )
        shutil.copy2(config.checkpoints / recipe_name, staging / recipe_name)
        os.replace(staging, config.exported_model)


def validate(config: Config) -> None:
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    prefix = "model.language_model" if family == "mistral3" else "model"
    utils.validate_dense_model(
        config.exported_model,
        model_loader=model_loader(family),
        weights=lambda model: (
            (name, model.get_parameter(f"{name}.weight"))
            for index in range(start, end)
            for attribute, *_ in PROJECTIONS
            for name in (f"{prefix}.layers.{index}.{attribute}",)
        ),
    )


def run(config: Config) -> None:
    calibration(config)
    hessian(config)
    quantize(config)
    export(config)
    validate(config)
