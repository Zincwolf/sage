import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from weight_vq.blockwise.artifacts import PROJECTIONS, atomic_write_completion, layer_complete
from weight_vq.blockwise.pipeline import _initialize_run, _model_contract, export, validate
from weight_vq.qtip.config import QTIPConfig


def config(tmp_path):
    return QTIPConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "rp",
        output_dir=tmp_path / "output",
        devices=[0],
    )


def test_initialize_run_uses_run_local_tlut(monkeypatch, tmp_path):
    received = []
    cfg = config(tmp_path)
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.bitshift_codebook", lambda **kwargs: received.append(kwargs)
    )
    _initialize_run(cfg, "qwen3", 2)
    assert received[0]["tlut_path"] == cfg.checkpoints / "tlut-l16-b9-v2.pt"


def test_model_contract_rejects_retired_qtip_vl(monkeypatch):
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.AutoConfig.from_pretrained",
        lambda *a, **k: SimpleNamespace(model_type="qwen3_vl"),
    )
    with pytest.raises(ValueError, match="supports qwen3"):
        _model_contract(Path("/vl"))


def test_recipe_separates_execution_and_algorithm(tmp_path):
    cfg = config(tmp_path)
    recipe = cfg.recipe(model_family="qwen3", layer_end=36)
    assert recipe["schema_version"] == 4
    assert recipe["batch_size"] == 8
    assert recipe["learning_rate"] == 5e-5
    assert recipe["susv_learning_rate"] == 5e-4
    assert recipe["train_lut"] is True
    assert recipe["reconstruction_epochs"] == 5
    assert cfg.calibration_recipe()["hessian"] == {
        "shape": [4096, 8192],
        "sampling": "replacement-full-documents-concatenate-reshape",
    }
    assert cfg.calibration_recipe()["train"] == {"shape": [256, 2048]}
    for key in ("devices", "replay_device", "forward_microbatch_size", "output_dir"):
        assert key not in recipe


def test_export_loads_auxiliary_norms_and_preserves_untouched_modules(
    tmp_path, monkeypatch, capsys
):
    cfg = config(tmp_path)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).eval()
    model.save_pretrained(cfg.model)
    original_embeddings = model.model.embed_tokens.weight.detach().clone()
    cfg.checkpoints.mkdir(parents=True)
    for attribute, short, *_ in PROJECTIONS:
        weight = model.model.layers[0].get_parameter(f"{attribute}.weight")
        torch.save({"weight": weight.detach() * 0.8}, cfg.checkpoints / f"0_{short}.pt")
        weight.data.mul_(0.8)
    auxiliary = {
        "self_attn.q_norm.weight": torch.full((8,), 1.3),
        "self_attn.k_norm.weight": torch.full((8,), 0.7),
        "input_layernorm.weight": torch.full((32,), 1.1),
        "post_attention_layernorm.weight": torch.full((32,), 0.9),
    }
    torch.save(auxiliary, cfg.checkpoints / "0.auxiliary.pt")
    for name, value in auxiliary.items():
        model.model.layers[0].get_parameter(name).data.copy_(value)
    atomic_write_completion(0, cfg.checkpoints)
    (cfg.checkpoints / "qtip_config.json").write_text(
        json.dumps(cfg.recipe(model_family="qwen3", layer_end=1))
    )
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.reconstruct_dense_weight",
        lambda saved, recipe: saved["weight"],
    )
    assets = SimpleNamespace(save_pretrained=lambda path: (path / "tokenizer.json").touch())
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.AutoTokenizer.from_pretrained", lambda *a, **k: assets
    )
    export(cfg)
    restored = Qwen3ForCausalLM.from_pretrained(cfg.exported_model).eval()
    tokens = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(restored(tokens).logits, model(tokens).logits)
    torch.testing.assert_close(restored.model.embed_tokens.weight, original_embeddings)
    for name, value in auxiliary.items():
        torch.testing.assert_close(restored.model.layers[0].get_parameter(name), value)
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: restored,
    )
    validate(cfg)
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert len(rows) == 8
    with pytest.raises(FileExistsError):
        export(cfg)


def test_completion_requires_auxiliary_state(tmp_path):
    for _, short, *_ in PROJECTIONS:
        (tmp_path / f"0_{short}.pt").touch()
    with pytest.raises(RuntimeError, match="auxiliary"):
        atomic_write_completion(0, tmp_path)
    (tmp_path / "0.auxiliary.pt").touch()
    atomic_write_completion(0, tmp_path)
    assert layer_complete(0, tmp_path)
