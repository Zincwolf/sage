"""SPHQuant stage adapters."""

from __future__ import annotations

import importlib
from typing import Any

from weight_vq._pending import stage
from weight_vq.calibration import load_calibration_metadata
from weight_vq.config import CalibrationMetadata, PendingPipelineConfig
from weight_vq.sphquant.config import SPHQuantOptimizeConfig


def reference_arguments(
    config: SPHQuantOptimizeConfig,
    metadata: CalibrationMetadata,
) -> dict[str, Any]:
    """Resolve public configuration into the reference ParoQuant arguments."""
    initialization = config.initialization
    recovery = config.recovery

    return {
        "model": str(config.model),
        "params": [
            f"radius:{recovery.learning_rate},codebook:{recovery.learning_rate}"
        ],
        "epochs": [recovery.epochs],
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
        "batch_size": recovery.batch_size,
        "val_batch_size": recovery.batch_size,
        "gradient_accumulation_steps": recovery.gradient_accumulation_steps,
        "seqlen": metadata.sequence_length,
        "calibration_path": str(config.calibration),
        "multimodal_calibration_path": None,
        "calibration_limit": config.calibration_limit,
        "max_layers": config.max_layers,
        "sph_sample_size": initialization.sample_size,
        "sph_kmeans_iterations": initialization.kmeans_iterations,
        "model_dtype": "float16",
        "cache_shards": recovery.cache_shards,
        "output_dir": str(config.output),
        "resume": recovery.resume,
        "checkpointing": recovery.checkpointing,
        "seed": config.seed,
        "use_wandb": config.use_wandb,
    }


def _run_reference(arguments: dict[str, Any]) -> None:
    paroquant_optimize: Any = importlib.import_module("paroquant.cli.optimize")
    config_type: Any = paroquant_optimize.Config
    config = config_type(**arguments)
    reference_parse = paroquant_optimize.simple_parsing.parse
    paroquant_optimize.simple_parsing.parse = lambda *_args, **_kwargs: config
    try:
        paroquant_optimize.main()
    finally:
        paroquant_optimize.simple_parsing.parse = reference_parse


def optimize(config: SPHQuantOptimizeConfig) -> None:
    arguments = reference_arguments(config, load_calibration_metadata(config.calibration))
    _run_reference(arguments)


def export(config: PendingPipelineConfig) -> None:
    stage("sphquant", "export", config)


def validate(config: PendingPipelineConfig) -> None:
    stage("sphquant", "validate", config)
