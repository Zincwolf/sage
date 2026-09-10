"""Native MoE routing, sparse statistics, and real method round trips."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from weight_vq.blockwise.artifacts import hessian_complete, layer_complete, projections
from weight_vq.blockwise.hessian import LayerHessianCollectors, load_quantization_hessian
from weight_vq.blockwise.models import projection_weight, split_work_experts
from weight_vq.blockwise.pipeline import export, hessian, quantize
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.artifacts import reconstruct_dense_weight as reconstruct_qtip
from weight_vq.qtip.config import QTIPConfig
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.quip_sharp.quantization import reconstruct_dense_weight as reconstruct_quip


@pytest.fixture
def moe_model():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=1,
    )
    config.update({"head_dim": 32})
    model = Qwen3MoeForCausalLM(config).eval().requires_grad_(False)
    model.set_experts_implementation("eager")
    return model


def test_private_expert_layout_matches_native_and_preserves_teacher(moe_model):
    moe_model.share_memory()
    original = moe_model.model.layers[0]
    snapshot = {name: value.clone() for name, value in original.state_dict().items()}
    work = deepcopy(original)
    split_work_experts(work)
    hidden = torch.randn(2, 7, 128)
    torch.testing.assert_close(work.mlp(hidden), original.mlp(hidden))
    for attribute, *_ in projections(4):
        torch.testing.assert_close(
            work.get_parameter(f"{attribute}.weight"), projection_weight(original, attribute)
        )
    work.requires_grad_(True)
    work.mlp(hidden).square().sum().backward()
    with torch.no_grad():
        for p in work.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-0.01)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], atol=0, rtol=0)
    assert all(parameter.is_shared() for parameter in moe_model.parameters())


def test_empty_rank_replay_contributes_zero(moe_model):
    state = capture_replay(
        moe_model,
        torch.empty((0, 8), dtype=torch.long),
        device="cpu",
        microbatch_size=1,
        placement="cpu",
    )
    replay = state.forward_original(moe_model.model.layers[0], device="cpu")
    error, count = replay.local_totals(moe_model.model.layers[0], "cpu", 1)
    assert count == 0 and error.item() == 0


def test_legacy_expert_checkpoint_loads_through_hf(moe_model, tmp_path):
    state = dict(moe_model.state_dict())
    prefix = "model.layers.0.mlp.experts"
    fused = state.pop(f"{prefix}.gate_up_proj")
    down = state.pop(f"{prefix}.down_proj")
    for expert in range(4):
        gate, up = fused[expert].chunk(2, dim=0)
        state[f"{prefix}.{expert}.gate_proj.weight"] = gate.contiguous().clone()
        state[f"{prefix}.{expert}.up_proj.weight"] = up.contiguous().clone()
        state[f"{prefix}.{expert}.down_proj.weight"] = down[expert].contiguous().clone()
    moe_model.save_pretrained(tmp_path, state_dict=state)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    for name, value in moe_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)


def test_routed_statistics_and_weight_only_metric(moe_model, tmp_path):
    layer = deepcopy(moe_model.model.layers[0])
    split_work_experts(layer)
    collectors = LayerHessianCollectors(layer, include_padding=True)
    # 8 original tokens, top-2: normal=8, low=1, zero=0, normal=7.
    hidden = torch.randn(8, 128)
    routes = torch.tensor([[0, 1]] + [[0, 3]] * 7)
    layer.mlp.experts(hidden, routes, torch.full((8, 2), 0.5))
    expected = [8, 1, 0, 7]
    for name, collector in collectors.collectors.items():
        matrix, first, count = collector.close_statistics()
        if not name.startswith("expert"):
            continue
        expert = int(name.split("_")[0].removeprefix("expert"))
        assert count == expected[expert]
        if count:
            matrix /= count
            first /= count
        flat = torch.cat([row[: i + 1] for i, row in enumerate(matrix)])
        path = tmp_path / f"{name}.pt"
        torch.save({"flatH": flat, "n": 128, "ct": count, "total_tokens": 8}, path)
        metric, metadata = load_quantization_hessian(path, 0.25)
        assert metadata["fallback"] == (expert in (1, 2))
        assert metadata["coverage"] == count / 8
        if metadata["fallback"]:
            torch.testing.assert_close(metric, torch.eye(128))
        else:
            torch.testing.assert_close(metric, matrix)
        # Threshold equality is sufficient coverage; zero remains fallback at zero.
        _, equal = load_quantization_hessian(path, count / 8)
        assert equal["fallback"] == (count == 0)
    assert sum(expected) == hidden.shape[0] * 2
    assert not hessian_complete(0, tmp_path, 4)
    assert not layer_complete(0, tmp_path, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real quantizers require CUDA")
@pytest.mark.parametrize("method", ["qtip", "quip"])
@pytest.mark.parametrize("devices", [[0], [0, 1, 2, 3]], ids=["single", "four-gpu"])
def test_moe_pipeline_sparse_restart_and_native_export(moe_model, tmp_path, method, devices):
    if torch.cuda.device_count() < len(devices):
        pytest.skip("Requested GPUs unavailable")
    # Opposite router directions win according to sign; half-strength copies
    # never win. Avoid tied logits, whose top-k order differs on CPU and CUDA.
    with torch.no_grad():
        router = moe_model.model.layers[0].mlp.gate.weight
        router[1].copy_(-router[0])
        router[2].copy_(router[0] * 0.5)
        router[3].copy_(router[0] * -0.5)
    config_class = QTIPConfig if method == "qtip" else QuIPSharpConfig
    config = config_class(
        model=tmp_path / "model",
        dataset=tmp_path / "unused",
        output_dir=tmp_path / "run",
        devices=devices,
        model_dtype="float32",
        num_cpu_threads=2,
        hessian_samples=5,
        hessian_sequence_length=8,
        hessian_chunk_size=2,
        train_samples=4,
        validation_samples=2,
        sequence_length=8,
        reconstruction_epochs=1,
        batch_size=4,
        min_expert_coverage=0.6,
        gradient_checkpointing=True,
    )
    if isinstance(config, QTIPConfig):
        config = replace(config, L=8, tlut_bits=6)
    else:
        config = replace(config, quip_tune_iters=1)
    moe_model.save_pretrained(config.model)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "test": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        config.model
    )
    config.output_dir.mkdir()
    payload = {
        "protocol": config.calibration_recipe(),
        "hessian": torch.randint(1, 32, (5, 8)),
        "train": torch.randint(1, 32, (4, 8)),
        "validation": torch.randint(1, 32, (2, 8)),
    }
    torch.save(payload, config.calibration)
    hessian(config)
    # Compare disk accumulation across chunks against one CPU collection.
    work = deepcopy(moe_model.model.layers[0])
    split_work_experts(work)
    state = capture_replay(
        moe_model, payload["hessian"], device="cpu", microbatch_size=1, placement="cpu"
    )
    collectors = LayerHessianCollectors(work, include_padding=True)
    state.forward_original(work, device="cpu", collectors=collectors)
    counts = []
    for name, collector in collectors.collectors.items():
        matrix, first, count = collector.close_statistics()
        saved = torch.load(config.hessians / f"0_{name}.pt", weights_only=True)
        assert saved["ct"] == count
        if count:
            matrix /= count
            if method == "quip":
                first /= count
                matrix -= first[:, None] * first[None, :]
        flat = torch.cat([row[: i + 1] for i, row in enumerate(matrix)])
        torch.testing.assert_close(saved["flatH"], flat, atol=2e-6, rtol=2e-5)
        if name.startswith("expert") and name.endswith("_up"):
            counts.append(count)
    assert sum(counts) == 40
    assert 0 in counts and any(0 < count < 24 for count in counts)
    config.checkpoints.mkdir()
    if isinstance(config, QTIPConfig):
        torch.save(torch.randn(64, 2), config.checkpoints / "tlut-l8-b6-v2.pt")
    quantize(config)
    assert layer_complete(0, config.checkpoints, 4)
    with pytest.raises(RuntimeError, match="configuration differs"):
        quantize(replace(config, min_expert_coverage=0.1))
    original = torch.load(config.checkpoints / "0_expert0_gate.pt", weights_only=True)
    assert original["metric"] in ("identity", "empirical")
    stamp = (config.checkpoints / "0.complete").stat().st_mtime_ns
    quantize(config)
    assert (config.checkpoints / "0.complete").stat().st_mtime_ns == stamp
    (config.checkpoints / "0_expert0_gate.pt").unlink()
    assert not layer_complete(0, config.checkpoints, 4)
    quantize(config)
    restored = torch.load(config.checkpoints / "0_expert0_gate.pt", weights_only=True)
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(restored[key], value, atol=0, rtol=0)
        else:
            assert restored[key] == value
    export(config)
    dense = AutoModelForCausalLM.from_pretrained(
        config.exported_model, local_files_only=True
    ).eval()
    assert dense.model.layers[0].mlp.experts.gate_up_proj.shape == (4, 256, 128)
    for attribute, short, *_ in projections(4):
        saved = torch.load(config.checkpoints / f"0_{short}.pt", weights_only=True)
        expected = (
            reconstruct_qtip(saved, config.recipe(model_family="qwen3_moe", layer_end=1))
            if isinstance(config, QTIPConfig)
            else reconstruct_quip(saved, torch.device("cuda:0"))
        )
        torch.testing.assert_close(projection_weight(dense.model.layers[0], attribute), expected)
        if short.startswith("expert"):
            assert saved["fallback"] == (
                saved["observed_tokens"] == 0 or saved["coverage"] < config.min_expert_coverage
            )
    assert (config.exported_model / "moe_coverage.json").is_file()
    assert torch.isfinite(dense(payload["validation"]).logits).all()
    assert cast(Any, dense).generate(payload["validation"][:1], max_new_tokens=2).shape == (1, 10)
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        torch.testing.assert_close(
            dense.get_parameter(name), moe_model.get_parameter(name), atol=0, rtol=0
        )
