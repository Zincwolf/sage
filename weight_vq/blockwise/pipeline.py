"""Shared language-model calibration through HF dense QDQ export."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Literal, cast

import torch
from lib.algo import finetune
from lib.codebook.bitshift import bitshift_codebook
from safetensors import safe_open
from torch import nn
from torch.multiprocessing.spawn import spawn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from weight_vq import utils
from weight_vq.blockwise.artifacts import (
    atomic_write_completion,
    hessian_complete,
    layer_complete,
    projection_layout,
    projections,
    quant_order,
)
from weight_vq.blockwise.calibration import calibration, load_calibration
from weight_vq.blockwise.distributed import process_group
from weight_vq.blockwise.hessian import LayerHessianCollectors, load_quantization_hessian
from weight_vq.blockwise.models import (
    language_backbone,
    layer_attention_type,
    layer_expert_count,
    projection_weight,
    split_work_experts,
)
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
    if config.model_type in ("qwen3", "qwen3_moe"):
        return config.model_type, int(config.num_hidden_layers)
    if config.model_type == "mistral3" and config.text_config.model_type == "ministral3":
        return "mistral3", int(config.text_config.num_hidden_layers)
    if config.model_type == "qwen3_5":
        return "qwen3_5", int(config.text_config.num_hidden_layers)
    raise ValueError(
        "Blockwise pipeline supports qwen3, qwen3_moe, Qwen3.5 and Ministral-3, "
        f"got {config.model_type!r}"
    )


def model_loader(family: str):
    return (
        AutoModelForImageTextToText if family in ("mistral3", "qwen3_5") else AutoModelForCausalLM
    )


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
        saved_recipe = json.loads(recipe_path.read_text())
        # Older recipes included this execution-only memory/compute tradeoff.
        saved_recipe.pop("gradient_checkpointing", None)
        if saved_recipe != recipe:
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
            # Raw per-layer accumulators live on disk between calibration chunks.
            # Only one layer's GPU collectors and one projection's CPU merge
            # (current plus previous matrix) are resident.
            chunks = range(0, len(data["hessian"]), config.hessian_chunk_size)
            for chunk_index, chunk_start in enumerate(chunks):
                chunk = data["hessian"][chunk_start : chunk_start + config.hessian_chunk_size]
                last_chunk = chunk_start + len(chunk) == len(data["hessian"])
                state = capture_replay(
                    model,
                    chunk[rank :: distributed.world_size],
                    device=distributed.device,
                    microbatch_size=config.forward_microbatch_size,
                    placement="cuda",
                )
                for index in tqdm(
                    range(end),
                    desc=f"Hessian chunk {chunk_index + 1}/{len(chunks)}",
                    total=len(chunks) * end,
                    initial=chunk_index * end,
                    unit="layer",
                    disable=not distributed.is_primary,
                    ascii=not sys.stderr.isatty(),
                    mininterval=5,
                ):
                    layer = deepcopy(layers[index]).to(distributed.device)
                    split_work_experts(layer)
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
                                path = hessian_dir / f"{index}_{name}.pt"
                                if chunk_start:
                                    previous = torch.load(
                                        path, weights_only=True, map_location="cpu"
                                    )
                                    matrix.add_(previous["matrix"])
                                    first_moment.add_(previous["first_moment"])
                                    count += previous["ct"]
                                    del previous
                                if last_chunk:
                                    # Unvisited experts retain zero statistics; the quantizer
                                    # chooses the weight-only metric after global aggregation.
                                    if count:
                                        matrix.div_(count)
                                        if centered:
                                            first_moment.div_(count)
                                            matrix.sub_(
                                                first_moment[:, None] * first_moment[None, :]
                                            )
                                    flat = torch.cat([row[: i + 1] for i, row in enumerate(matrix)])
                                    utils.atomic_torch_save(
                                        {
                                            "flatH": flat,
                                            "n": matrix.shape[0],
                                            "ct": count,
                                            "position_mask": "all",
                                            **({"mu": first_moment} if centered else {}),
                                            **(
                                                {"total_tokens": data["hessian"].numel()}
                                                if name.startswith("expert")
                                                else {}
                                            ),
                                        },
                                        path,
                                    )
                                    del flat
                                else:
                                    utils.atomic_torch_save(
                                        {
                                            "matrix": matrix,
                                            "first_moment": first_moment,
                                            "ct": count,
                                        },
                                        path,
                                    )
                            del matrix, first_moment
                    del collectors, layer
                    utils.clear_cuda_cache()
                del state
                utils.clear_cuda_cache()
            if distributed.is_primary:
                utils.atomic_write_text(
                    hessian_dir / "complete.json",
                    json.dumps(
                        config.hessian_recipe(end) | projection_layout(model.config, end), indent=2
                    )
                    + "\n",
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
            expert_count = layer_expert_count(model.config, index)
            layer_type = layer_attention_type(model.config, index)
            specs = projections(expert_count, layer_type)
            if index >= start and not layer_complete(
                index, config.checkpoints, expert_count, layer_type
            ):
                split_work_experts(layer)
                if distributed.is_primary:
                    (config.checkpoints / f"{index}.complete").unlink(missing_ok=True)
                layer.requires_grad_(True)
                if getattr(model.config, "model_type", None) == "qwen3_5":
                    # New recurrent auxiliaries remain frozen. UniSVQ trains only
                    # the norms it persists; QTIP/QuIP# also reconstruct attention norms.
                    layer.requires_grad_(False)
                    for attribute, *_ in specs:
                        layer.get_submodule(attribute).requires_grad_(True)
                    layer.get_submodule("input_layernorm").requires_grad_(True)
                    layer.get_submodule("post_attention_layernorm").requires_grad_(True)
                    if not isinstance(config, UniSVQConfig):
                        norms = (
                            ("linear_attn.norm",)
                            if layer_type == "linear_attention"
                            else ("self_attn.q_norm", "self_attn.k_norm")
                        )
                        for attribute in norms:
                            layer.get_submodule(attribute).requires_grad_(True)
                trainer = (
                    UniSVQReconstruction
                    if isinstance(config, UniSVQConfig)
                    else DistributedReconstruction
                )
                reconstruction = trainer(
                    train_layer, validation_layer, config, distributed, config.seed + index
                )
                # Expert ownership is shared by all quantizers; attention stays replicated.
                owners = {
                    short: int(attribute.split(".")[2]) % distributed.world_size
                    for attribute, short, *_ in specs
                    if attribute.startswith("mlp.experts.") and distributed.world_size > 1
                }
                if isinstance(config, QTIPConfig):
                    finetune.quantize_finetune_decoder_layer(
                        layer,
                        quant_order(expert_count, layer_type),
                        index,
                        codebook,
                        config,
                        distributed.device,
                        None,
                        None,
                        reconstruction_callback=reconstruction,
                        write_checkpoints=distributed.is_primary,
                        # Attention remains replicated. Experts are independent
                        # within each reconstruction group; synchronize at its end.
                        projection_owners=(
                            [owners.get(short) for _, short, *_ in specs] if owners else None
                        ),
                        hessian_loader=partial(
                            load_quantization_hessian,
                            min_expert_coverage=config.min_expert_coverage,
                        ),
                        reconstruction_groups=(
                            [
                                short
                                if not attribute.startswith("mlp.experts.")
                                else attribute.rsplit(".", 1)[1]
                                for attribute, short, *_ in specs
                            ]
                            if expert_count
                            else None
                        ),
                    )
                elif isinstance(config, UniSVQConfig):
                    quantize_unisvq_layer(
                        layer,
                        index,
                        config,
                        cast(UniSVQReconstruction, reconstruction),
                        projection_owners=owners,
                    )
                else:
                    quantize_layer(layer, index, config, reconstruction, projection_owners=owners)
                if distributed.is_primary:
                    prefixes = tuple(f"{attribute}." for attribute, *_ in specs)
                    auxiliary = {
                        name: value.detach().cpu().clone()
                        for name, value in layer.state_dict().items()
                        if not name.startswith(prefixes) or name.endswith(".bias")
                    }
                    if isinstance(config, UniSVQConfig):
                        # Released quantize_llama_layer/hfize persist only these norms.
                        # q/k norm parameters are trained upstream but discarded at export;
                        # preserve that behavior rather than silently improve the method.
                        # MoE additionally persists its unquantized, trained router.
                        auxiliary = {
                            name: value
                            for name, value in auxiliary.items()
                            if name in ("input_layernorm.weight", "post_attention_layernorm.weight")
                            or (expert_count and name == "mlp.gate.weight")
                            or name.endswith(".bias")
                        }
                    utils.atomic_torch_save(auxiliary, config.checkpoints / f"{index}.auxiliary.pt")
                    if expert_count:
                        coverage = {}
                        for expert in range(expert_count):
                            saved = torch.load(
                                config.checkpoints / f"{index}_expert{expert}_gate.pt",
                                map_location="cpu",
                                weights_only=True,
                            )
                            coverage[str(expert)] = {
                                key: saved[key]
                                for key in (
                                    "observed_tokens",
                                    "total_tokens",
                                    "coverage",
                                    "fallback",
                                    "metric",
                                )
                            }
                            del saved
                        utils.atomic_write_text(
                            config.checkpoints / f"{index}.coverage.json",
                            json.dumps(coverage, indent=2) + "\n",
                        )
                    atomic_write_completion(index, config.checkpoints, expert_count, layer_type)
                distributed.barrier()
                del reconstruction
            del train_layer, validation_layer
            del layer
            utils.clear_cuda_cache()


def _run_workers(config: Config, stage: Stage) -> None:
    family, model_layers = _model_contract(config.model)
    hf_config = AutoConfig.from_pretrained(config.model, local_files_only=True)
    start, end = layer_range(config, model_layers)
    load_calibration(config)
    marker = config.hessians / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text()) != (
            config.hessian_recipe(end) | projection_layout(hf_config, end)
        ):
            raise RuntimeError("Hessian protocol differs; use a new output_dir")
        if all(
            hessian_complete(
                index,
                config.hessians,
                layer_expert_count(hf_config, index),
                layer_attention_type(hf_config, index),
            )
            for index in range(start, end)
        ):
            if stage == "hessian":
                print("Hessian complete; reusing saved statistics.", file=sys.stderr)
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
            index
            for index in range(start, end)
            if not layer_complete(
                index,
                config.checkpoints,
                layer_expert_count(hf_config, index),
                layer_attention_type(hf_config, index),
            )
        ]
        if missing:
            raise RuntimeError(f"quantize did not complete layers: {missing}")


def hessian(config: Config) -> None:
    _run_workers(config, "hessian")


def quantize(config: Config) -> None:
    _run_workers(config, "quantize")


def export(config: Config) -> None:
    family, model_layers = _model_contract(config.model)
    hf_config = AutoConfig.from_pretrained(config.model, local_files_only=True)
    start, end = layer_range(config, model_layers)
    method = {QTIPConfig: "qtip", QuIPSharpConfig: "quip_sharp", UniSVQConfig: "unisvq"}[
        type(config)
    ]
    recipe_name = f"{method}_config.json"
    recipe = json.loads((config.checkpoints / recipe_name).read_text())
    # Accept legacy recipes without rewriting the original checkpoint provenance.
    recipe.pop("gradient_checkpointing", None)
    if recipe != config.recipe(model_family=family, layer_end=end):
        raise RuntimeError("export configuration differs from checkpoint recipe")
    missing = [
        index
        for index in range(start, end)
        if not layer_complete(
            index,
            config.checkpoints,
            layer_expert_count(hf_config, index),
            layer_attention_type(hf_config, index),
        )
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
            for attribute, short, *_ in projections(
                layer_expert_count(hf_config, index), layer_attention_type(hf_config, index)
            ):
                saved = torch.load(
                    config.checkpoints / f"{index}_{short}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                target = projection_weight(layer, attribute)
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
                target.copy_(weight.to(target.dtype))
                del weight
            auxiliary = torch.load(
                config.checkpoints / f"{index}.auxiliary.pt", map_location="cpu", weights_only=True
            )
            state = layer.state_dict()
            for name, value in auxiliary.items():
                state[name].copy_(value)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=config.output_dir) as temporary:
        staging = Path(temporary) / "model"
        state_dict = model.state_dict()
        if family == "qwen3_5":
            # HF intentionally ignores MTP modules; retain their original tensors on export.
            for shard_path in config.model.glob("*.safetensors"):
                with safe_open(shard_path, framework="pt", device="cpu") as shard:
                    for key in shard.keys():  # noqa: SIM118 - safe_open is not iterable
                        if key.startswith("mtp."):
                            state_dict[key] = shard.get_tensor(key)
        model.save_pretrained(
            staging, state_dict=state_dict, safe_serialization=True, max_shard_size="5GB"
        )
        if family == "mistral3":
            AutoProcessor.from_pretrained(
                config.model, local_files_only=True, fix_mistral_regex=True
            ).save_pretrained(staging)
            shutil.copy2(config.model / "tekken.json", staging / "tekken.json")
        elif family == "qwen3_5":
            AutoProcessor.from_pretrained(config.model, local_files_only=True).save_pretrained(
                staging
            )
        else:
            AutoTokenizer.from_pretrained(config.model, local_files_only=True).save_pretrained(
                staging
            )
        shutil.copy2(config.checkpoints / recipe_name, staging / recipe_name)
        if family == "qwen3_moe":
            coverage = {
                str(index): json.loads((config.checkpoints / f"{index}.coverage.json").read_text())
                for index in range(start, end)
                if layer_expert_count(hf_config, index)
            }
            utils.atomic_write_text(
                staging / "moe_coverage.json", json.dumps(coverage, indent=2) + "\n"
            )
        os.replace(staging, config.exported_model)


def validate(config: Config) -> None:
    family, model_layers = _model_contract(config.model)
    start, end = layer_range(config, model_layers)
    prefix = "model.language_model" if family in ("mistral3", "qwen3_5") else "model"
    if family == "qwen3_moe":
        coverage = json.loads((config.exported_model / "moe_coverage.json").read_text())
        print("layer,experts,zero_coverage,fallback,coverage_min,coverage_median,coverage_max")
        for index, experts in coverage.items():
            values = torch.tensor([expert["coverage"] for expert in experts.values()])
            print(
                f"{index},{len(experts)},"
                f"{sum(expert['observed_tokens'] == 0 for expert in experts.values())},"
                f"{sum(expert['fallback'] for expert in experts.values())},"
                f"{values.min().item():.6g},{values.quantile(0.5).item():.6g},"
                f"{values.max().item():.6g}"
            )
    model = utils.validate_dense_model(
        config.exported_model,
        model_loader=model_loader(family),
        # Dense QDQ retains all 30B weights; validate on CPU instead of loading
        # the entire exported MoE onto a single GPU.
        device="cpu" if family == "qwen3_moe" else config.devices[0],
        weights=lambda model: (
            (
                f"{prefix}.layers.{index}.{attribute}",
                projection_weight(language_backbone(model).layers[index], attribute),
            )
            for index in range(start, end)
            for attribute, *_ in projections(
                layer_expert_count(model.config, index), layer_attention_type(model.config, index)
            )
        ),
    )
    if family == "qwen3_5":
        tokenizer = AutoTokenizer.from_pretrained(config.exported_model, local_files_only=True)
        inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.eval()(**inputs, use_cache=False, logits_to_keep=1)
        print(f"HF text forward: logits shape {tuple(output.logits.shape)}")


def run(config: Config) -> None:
    calibration(config)
    hessian(config)
    quantize(config)
    export(config)
    validate(config)
