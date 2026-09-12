"""Execution-only settings remain adjustable across current and legacy recipes."""

import json
from dataclasses import replace

import pytest

from weight_vq.blockwise.pipeline import _initialize_run, export
from weight_vq.qtip.config import QTIPConfig
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.unisvq.config import UniSVQConfig


@pytest.mark.parametrize(
    ("config_type", "method"),
    [(QTIPConfig, "qtip"), (QuIPSharpConfig, "quip_sharp"), (UniSVQConfig, "unisvq")],
)
@pytest.mark.parametrize("saved_checkpointing", [None, False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_execution_settings_resume_and_export_legacy_recipe(
    tmp_path, monkeypatch, config_type, method, saved_checkpointing, checkpointing
):
    config = config_type(
        model=tmp_path / "model",
        dataset=tmp_path / "dataset",
        output_dir=tmp_path / "run",
        devices=[0, 1, 2, 3],
        gradient_checkpointing=checkpointing,
    )
    recipe = config.recipe(model_family="qwen3_moe", layer_end=48)
    adjusted = replace(
        config,
        gradient_checkpointing=not checkpointing,
        devices=[3],
        hessian_chunk_size=1024,
        forward_microbatch_size=32,
        reconstruction_microbatch_size=8,
        num_cpu_threads=2,
    )
    assert adjusted.recipe(model_family="qwen3_moe", layer_end=48) == recipe
    assert adjusted.calibration_recipe() == config.calibration_recipe()
    assert adjusted.hessian_recipe(48) == config.hessian_recipe(48)
    assert "gradient_checkpointing" not in recipe
    if saved_checkpointing is not None:
        recipe["gradient_checkpointing"] = saved_checkpointing
    config.checkpoints.mkdir(parents=True)
    recipe_path = config.checkpoints / f"{method}_config.json"
    original = json.dumps(recipe)
    recipe_path.write_text(original)
    # Existing QTIP runs already have a LUT; no codebook generation is needed here.
    if isinstance(config, QTIPConfig):
        (config.checkpoints / "tlut-l16-b9-v2.pt").touch()
    _initialize_run(config, "qwen3_moe", 48)
    assert recipe_path.read_text() == original

    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline._model_contract", lambda model: ("qwen3_moe", 48)
    )
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.AutoConfig.from_pretrained", lambda *a, **k: None
    )
    monkeypatch.setattr("weight_vq.blockwise.pipeline.layer_expert_count", lambda *a: 128)
    # Export must pass recipe compatibility and reach the missing-layer check,
    # without attempting a model load or requiring any GPU artifacts.
    with pytest.raises(RuntimeError, match="checkpoints are incomplete"):
        export(config)
    assert recipe_path.read_text() == original

    for field, value in (
        ("reconstruction_epochs", config.reconstruction_epochs + 1),
        ("batch_size", config.batch_size * 2),
        ("learning_rate", config.learning_rate * 2),
        ("min_expert_coverage", 0.01),
    ):
        changed = replace(config, **{field: value})
        with pytest.raises(RuntimeError, match="configuration differs from checkpoints"):
            _initialize_run(changed, "qwen3_moe", 48)
        with pytest.raises(RuntimeError, match="export configuration differs"):
            export(changed)
