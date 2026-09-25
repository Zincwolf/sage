"""SPHQuant stage adapters for dense Qwen3 QDQ export."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from weight_vq import utils
from weight_vq.calibration import load_calibration_metadata
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


def _model_contract(model: Path) -> int:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    if hf_config.model_type != "qwen3":
        raise ValueError(f"SPHQuant supports only qwen3, got {hf_config.model_type!r}")
    return int(hf_config.num_hidden_layers)


def _target_layers(config: SPHQuantConfig, model_layers: int) -> range:
    end = model_layers if config.max_layers is None else config.max_layers
    if not 1 <= end <= model_layers:
        raise ValueError(f"max_layers must be within [1, {model_layers}] or null")
    return range(end)


def checkpoint_path(config: SPHQuantConfig, layer: int, projection: str) -> Path:
    return config.checkpoints / config.model.name / f"{layer}.{projection}.pt"


def missing_checkpoints(config: SPHQuantConfig, model_layers: int | None = None) -> list[Path]:
    if model_layers is None:
        model_layers = _model_contract(config.model)
    return [
        checkpoint_path(config, layer, projection)
        for layer in _target_layers(config, model_layers)
        for projection in PROJECTIONS
        if not checkpoint_path(config, layer, projection).is_file()
    ]


def optimize(config: SPHQuantConfig) -> None:
    if not 1 <= config.codebook_bits <= 8:
        raise ValueError("codebook_bits must be within [1, 8]")
    model_layers = _model_contract(config.model)
    _target_layers(config, model_layers)
    metadata = load_calibration_metadata(config.calibration)

    # This is the verified Qwen3 W2 recipe.
    # Keep these constants at the ParoQuant boundary until experiments require variants.
    arguments = {
        "model": str(config.model),
        "params": [f"radius:{config.learning_rate},codebook:{config.learning_rate}"],
        "epochs": [config.epochs],
        "method": "sphquant",
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
    model_layers = _model_contract(config.model)
    missing = missing_checkpoints(config, model_layers)
    if missing:
        preview = [str(path) for path in missing[:8]]
        raise FileNotFoundError(
            f"missing {len(missing)} SPHQuant checkpoints; first entries: {preview}"
        )
    if config.exported_model.exists():
        raise FileExistsError(f"refusing to overwrite {config.exported_model}")

    staging = config.exported_model.with_name(f".{config.exported_model.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    config.exported_model.parent.mkdir(parents=True, exist_ok=True)
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


def validate(config: SPHQuantConfig) -> None:
    """Load a dense artifact through the evaluation-facing HF API."""
    model_layers = _model_contract(config.model)
    target_layers = _target_layers(config, model_layers)
    with _physical_device(config.device):
        from transformers import AutoModelForCausalLM

        utils.validate_dense_model(
            config.exported_model,
            model_loader=AutoModelForCausalLM,
            weights=lambda model: (
                (name, model.get_parameter(f"{name}.weight"))
                for index in target_layers
                for projection in PROJECTIONS
                for name in (f"model.layers.{index}.{projection}",)
            ),
        )
