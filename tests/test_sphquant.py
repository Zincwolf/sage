from pathlib import Path

import pytest

from weight_vq.config import CalibrationMetadata
from weight_vq.sphquant.config import (
    SPHQuantInitializationConfig,
    SPHQuantOptimizeConfig,
    SPHQuantRecoveryConfig,
)
from weight_vq.sphquant.pipeline import optimize, reference_arguments


def smoke_config() -> SPHQuantOptimizeConfig:
    return SPHQuantOptimizeConfig(
        model=Path("/models/Qwen3-4B-Instruct-2507"),
        calibration=Path("outputs/calibration.pt"),
        output=Path("outputs/sphquant/one-layer"),
        initialization=SPHQuantInitializationConfig(
            sample_size=512,
            kmeans_iterations=2,
        ),
        recovery=SPHQuantRecoveryConfig(
            epochs=0,
            cache_shards=2,
        ),
        max_layers=1,
        calibration_limit=32,
    )


def calibration_metadata() -> CalibrationMetadata:
    return CalibrationMetadata(
        train_samples=1920,
        validation_samples=128,
        sequence_length=2048,
    )


def test_resolves_reference_sphquant_arguments() -> None:
    arguments = reference_arguments(smoke_config(), calibration_metadata())

    assert arguments == {
        "model": "/models/Qwen3-4B-Instruct-2507",
        "params": ["radius:0.001,codebook:0.001"],
        "epochs": [0],
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
        "train_size": 1920,
        "validation_size": 128,
        "batch_size": 16,
        "val_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "seqlen": 2048,
        "calibration_path": "outputs/calibration.pt",
        "multimodal_calibration_path": None,
        "calibration_limit": 32,
        "max_layers": 1,
        "sph_sample_size": 512,
        "sph_kmeans_iterations": 2,
        "model_dtype": "float16",
        "cache_shards": 2,
        "output_dir": "outputs/sphquant/one-layer",
        "resume": True,
        "checkpointing": True,
        "seed": 0,
        "use_wandb": False,
    }


def test_optimize_loads_metadata_and_runs_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[dict[str, object]] = []
    monkeypatch.setattr(
        "weight_vq.sphquant.pipeline.load_calibration_metadata",
        lambda _path: calibration_metadata(),
    )
    monkeypatch.setattr("weight_vq.sphquant.pipeline._run_reference", received.append)

    optimize(smoke_config())

    assert received[0]["method"] == "sphquant"
    assert received[0]["train_size"] == 1920
    assert received[0]["seqlen"] == 2048
