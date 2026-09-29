"""Shared ParoQuant stages for SPHQuant/SAGE reconstruction and HF export."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from weight_vq import utils
from weight_vq.blockwise.artifacts import projections
from weight_vq.blockwise.models import language_backbone, layer_expert_count, projection_weight
from weight_vq.calibration import load_calibration_metadata
from weight_vq.sage.config import SAGEConfig
from weight_vq.sphquant.config import SPHQuantConfig

PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@contextmanager
def _physical_device(device: int) -> Generator[None, None, None]:
    if device < 0:
        raise ValueError("device must be a non-negative physical CUDA index")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _model_contract(model: Path) -> tuple[str, tuple[tuple[str, ...], ...]]:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    if hf_config.model_type == "qwen3":
        return "qwen3", (PROJECTIONS,) * hf_config.num_hidden_layers
    if hf_config.model_type == "qwen3_moe":
        return "qwen3_moe", tuple(
            tuple(name for name, *_ in projections(layer_expert_count(hf_config, index)))
            for index in range(hf_config.num_hidden_layers)
        )
    if hf_config.model_type == "mistral3" and hf_config.text_config.model_type == "ministral3":
        return "mistral3", (PROJECTIONS,) * hf_config.text_config.num_hidden_layers
    if hf_config.model_type == "qwen3_5":
        return "qwen3_5", tuple(
            tuple(name for name, *_ in projections(layer_type=layer_type))
            for layer_type in hf_config.text_config.layer_types
        )
    raise ValueError(
        f"SPHQuant/SAGE supports qwen3, qwen3_moe, Ministral-3 and Qwen3.5, "
        f"got {hf_config.model_type!r}"
    )


def _target_layers(config: SPHQuantConfig, model_layers: int) -> range:
    end = model_layers if config.max_layers is None else config.max_layers
    if not 1 <= end <= model_layers:
        raise ValueError(f"max_layers must be within [1, {model_layers}] or null")
    return range(end)


def checkpoint_path(config: SPHQuantConfig, layer: int, projection: str) -> Path:
    return config.checkpoints / config.model.name / f"{layer}.{projection}.pt"


def missing_checkpoints(
    config: SPHQuantConfig, layout: tuple[tuple[str, ...], ...] | None = None
) -> list[Path]:
    if layout is None:
        _, layout = _model_contract(config.model)
    return [
        checkpoint_path(config, layer, projection)
        for layer in _target_layers(config, len(layout))
        for projection in layout[layer]
        if not checkpoint_path(config, layer, projection).is_file()
    ]


def optimize(config: SPHQuantConfig) -> None:
    if not 1 <= config.codebook_bits <= 8:
        raise ValueError("codebook_bits must be within [1, 8]")
    _, layout = _model_contract(config.model)
    _target_layers(config, len(layout))
    metadata = load_calibration_metadata(config.calibration)

    # Both methods share the SPHQuant reconstruction protocol. Only SAGE adds
    # teacher-energy statistics, ESG initialization and per-matrix ABS selection.
    arguments = {
        "model": str(config.model),
        "params": [f"radius:{config.learning_rate},codebook:{config.learning_rate}"],
        "epochs": [config.epochs],
        "method": "sage" if isinstance(config, SAGEConfig) else "sphquant",
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "loss": "mse",
        "optimizer": "adam",
        "use_lr_scheduler": False,
        "group_size": 128,
        "n_bit": 2,
        "num_rotations": 0,
        "skipped_modules": [],
        "datasets": ["wikitext2", "c4", "redpajama"],
        "val_dataset": "pileval",
        "train_size": metadata.train_samples,
        "validation_size": metadata.validation_samples,
        "seqlen": metadata.sequence_length,
        "calibration_path": str(config.calibration),
        "batch_size": config.batch_size,
        "val_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "calibration_limit": config.calibration_limit,
        "max_layers": config.max_layers,
        "sph_sample_size": config.sample_size,
        "sph_kmeans_iterations": config.kmeans_iterations,
        "sph_codebook_bits": config.codebook_bits,
        "cache_shards": config.cache_shards,
        "output_dir": str(config.checkpoints),
        "resume": config.resume,
        "checkpointing": config.checkpointing,
        "seed": config.seed,
        "use_wandb": config.use_wandb,
    }

    if isinstance(config, SAGEConfig):
        arguments.update(
            sage_esg=config.esg,
            sage_adaptive_bits=config.adaptive_bits,
            sage_energy_power=config.energy_power,
            sage_allow_partial=config.allow_partial,
        )

    with _physical_device(config.device):
        from paroquant.cli import optimize as pq_opt

        # ParoQuant owns optimization semantics; this adapter only supplies Config.
        paroquant_config = pq_opt.Config(**arguments)
        reference_parse = pq_opt.simple_parsing.parse
        pq_opt.simple_parsing.parse = lambda *_args, **_kwargs: paroquant_config
        try:
            pq_opt.main()
        finally:
            pq_opt.simple_parsing.parse = reference_parse


def export(config: SPHQuantConfig) -> None:
    family, layout = _model_contract(config.model)
    if config.export_compressed and family != "qwen3":
        raise ValueError("compressed spherical export currently supports Qwen3 dense decoders only")
    missing = missing_checkpoints(config, layout)
    if missing:
        preview = [str(path) for path in missing[:8]]
        raise FileNotFoundError(
            f"missing {len(missing)} {'SAGE' if isinstance(config, SAGEConfig) else 'SPHQuant'} "
            f"checkpoints; first entries: {preview}"
        )
    if config.exported_model.exists():
        raise FileExistsError(f"refusing to overwrite {config.exported_model}")

    staging = config.exported_model.with_name(f".{config.exported_model.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    config.exported_model.parent.mkdir(parents=True, exist_ok=True)
    if config.export_compressed:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from weight_vq.integrations.transformers import WVQConfig, packed_linear
        from weight_vq.sphquant.packing import pack_spherical

        sage = isinstance(config, SAGEConfig)
        quantization = WVQConfig(method="sage" if sage else "sphquant")
        model = AutoModelForCausalLM.from_pretrained(
            config.model, dtype="auto", device_map="cpu", local_files_only=True
        ).eval()
        try:
            with torch.no_grad():
                for index in _target_layers(config, len(layout)):
                    for projection in layout[index]:
                        name = f"model.layers.{index}.{projection}"
                        linear = cast(torch.nn.Linear, model.get_submodule(name))
                        state = torch.load(
                            checkpoint_path(config, index, projection),
                            map_location="cpu",
                            weights_only=True,
                        )
                        packed = pack_spherical(state, sage=sage)
                        bias = linear.bias
                        if bias is not None:
                            bias.copy_(state["bias"])
                        model.set_submodule(
                            name,
                            packed_linear(quantization.method, packed, linear.out_features, bias),
                        )
                        quantization.modules[name] = (
                            {
                                "codebook_bits": packed["codebook"].shape[1].bit_length() - 1,
                                "permuted": bool(packed["permutation"].numel()),
                            }
                            if sage
                            else {}
                        )
                        del linear, state, packed
            model.config.quantization_config = quantization
            model.config._name_or_path = ""
            model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
            AutoTokenizer.from_pretrained(config.model, local_files_only=True).save_pretrained(
                staging
            )
            os.replace(staging, config.exported_model)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return
    arguments = [
        "--model",
        str(config.model),
        "--result-dir",
        str(config.checkpoints / config.model.name),
        "--output-path",
        str(staging),
        "--mode",
        "pseudo",
    ]

    with _physical_device(config.device):
        from paroquant.cli import convert as converter

        original_argv = sys.argv
        sys.argv = ["paroquant.cli.convert", *arguments]
        try:
            converter.main()
            os.replace(staging, config.exported_model)
        finally:
            sys.argv = original_argv
            if staging.exists():
                shutil.rmtree(staging)


def run(config: SPHQuantConfig) -> None:
    """Produce and validate a QDQ or packed model with one shared configuration."""
    optimize(config)
    export(config)
    validate(config)


def validate(config: SPHQuantConfig) -> None:
    """Load the exported HF artifact and report representation statistics."""
    if config.export_compressed:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from weight_vq.inference.linear import SAGELinear, SPHQuantLinear
        from weight_vq.integrations.transformers import WVQConfig

        model = AutoModelForCausalLM.from_pretrained(
            config.exported_model, dtype="auto", device_map="cpu", local_files_only=True
        )
        quantization = model.config.quantization_config
        method = "sage" if isinstance(config, SAGEConfig) else "sphquant"
        if not isinstance(quantization, WVQConfig) or quantization.method != method:
            raise ValueError(f"expected a compressed {method} HF artifact")
        AutoTokenizer.from_pretrained(config.exported_model, local_files_only=True)
        print(
            "layer,packed_bytes,bits_per_weight,codebook_bits,grid_min,grid_step_max,codebook_max"
        )
        for name, spec in quantization.modules.items():
            linear = cast(SAGELinear | SPHQuantLinear, model.get_submodule(name))
            tensors = linear.state_dict()
            size = sum(value.numel() * value.element_size() for value in tensors.values())
            # Inspect the stored grid/codebook without materializing dense weights.
            if method == "sage":
                minimum, step = tensors["radius_grid"].unbind(-1)
            else:
                minimum, step = tensors["radius_mins"], tensors["radius_scales"]
            print(
                f"{name},{size},{8 * size / (linear.in_features * linear.out_features):.6f},"
                f"{spec.get('codebook_bits', 6)},{minimum.min().item():.6g},"
                f"{step.max().item():.6g},{tensors['codebook'].max().item():.6g}"
            )
        return
    family, layout = _model_contract(config.model)
    target_layers = _target_layers(config, len(layout))
    prefix = "model.language_model" if family in ("mistral3", "qwen3_5") else "model"
    with _physical_device(config.device):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        utils.validate_dense_model(
            config.exported_model,
            model_loader=(
                AutoModelForImageTextToText
                if family in ("mistral3", "qwen3_5")
                else AutoModelForCausalLM
            ),
            weights=lambda model: (
                (name, projection_weight(language_backbone(model).layers[index], projection))
                for index in target_layers
                for projection in layout[index]
                for name in (f"{prefix}.layers.{index}.{projection}",)
            ),
        )
