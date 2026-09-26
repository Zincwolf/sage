"""Native Qwen3 MoE teacher, sparse SAGE policy and dense QDQ round trips."""

import json
import os
from dataclasses import replace
from functools import partial
from typing import Any

import pytest
import torch
from paroquant.cli import optimize as pq_opt
from paroquant.optim.moe import projection_weight, split_work_experts
from paroquant.optim.sagelinear import SAGEPseudoQuantizedLinear
from paroquant.optim.sphlinear import SPHPseudoQuantizedLinear
from paroquant.optim.util import get_spherical_linears, set_module_by_name
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from weight_vq import utils
from weight_vq.sage.config import SAGEConfig
from weight_vq.sphquant.config import SPHQuantConfig
from weight_vq.sphquant.pipeline import (
    PROJECTIONS,
    _model_contract,
    checkpoint_path,
    export,
    missing_checkpoints,
    optimize,
    validate,
)


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    hf_config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=1,
        mlp_only_layers=[1],
    )
    hf_config.update({"head_dim": 32})
    model: Any = Qwen3MoeForCausalLM(hf_config).eval()
    # Two experts are unreachable without relying on CPU/CUDA tie breaking.
    with torch.no_grad():
        for index in (0, 2):
            router = model.model.layers[index].mlp.gate.weight
            router[1].copy_(-router[0])
            router[2].copy_(router[0] * 0.5)
            router[3].copy_(router[0] * -0.5)
    config = SAGEConfig(
        model=tmp_path / "model",
        calibration=tmp_path / "calibration.pt",
        output_dir=tmp_path / "run",
        device=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
        sample_size=256,
        kmeans_iterations=1,
        codebook_bits=7,
        epochs=1,
        batch_size=2,
        cache_shards=1,
        allow_partial=True,
    )
    model.half().save_pretrained(config.model)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "test": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        config.model
    )
    samples = {"train": torch.randint(1, 32, (4, 8)), "validation": torch.randint(1, 32, (2, 8))}
    torch.save(samples, config.calibration)
    utils.manifest_path(config.calibration).write_text(
        json.dumps(
            {"tensors": {key: {"shape": list(value.shape)} for key, value in samples.items()}}
        )
    )
    return model, config, samples


def test_moe_contract_sparse_schedule_and_dense_exceptions(tmp_path):
    config = Qwen3MoeConfig(
        num_hidden_layers=4, num_experts=3, decoder_sparse_step=2, mlp_only_layers=[3]
    )
    config.save_pretrained(tmp_path)
    family, layout = _model_contract(tmp_path)
    assert family == "qwen3_moe"
    assert all(set(layout[index]) == set(PROJECTIONS) for index in (0, 2, 3))
    assert len(layout[1]) == 4 + 3 * 3
    assert "mlp.experts.2.down_proj" in layout[1]
    assert all("mlp.gate" not in names for names in layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ParoQuant uses CUDA")
@pytest.mark.parametrize("config_type", [SPHQuantConfig, SAGEConfig])
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16])
def test_moe_train_resume_export(source, config_type, source_dtype, monkeypatch, capsys):
    source_model, sage_config, samples = source
    source_model.to(source_dtype).save_pretrained(sage_config.model)
    if config_type is SAGEConfig:
        config = sage_config
    else:
        config = SPHQuantConfig(
            **{
                key: value
                for key, value in vars(sage_config).items()
                if key in SPHQuantConfig.__dataclass_fields__
            }
        )
    _, layout = _model_contract(config.model)
    original = {key: value.clone() for key, value in source_model.state_dict().items()}
    # Independent, unsplit native HF teacher. Experts hooks reconstruct only the
    # actual routed inputs, including each expert's nonlinear down-projection input.
    teacher: Any = AutoModelForCausalLM.from_pretrained(
        config.model, dtype=torch.float16, experts_implementation="eager", local_files_only=True
    )
    teacher.half().eval().cuda()
    teacher_inputs, teacher_outputs, energy_samples = {}, {}, {}

    def capture_block(index, _module, inputs, output):
        teacher_inputs.setdefault(index, []).append(inputs[0].detach().cpu())
        teacher_outputs.setdefault(index, []).append(output.detach().cpu())

    def capture_linear(key, _module, inputs, _output):
        x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
        energy_samples.setdefault(key, []).append(x.square().cpu())

    def capture_experts(index, module, inputs):
        hidden, routes, _weights = inputs
        for expert in range(module.num_experts):
            token, _slot = torch.where(routes == expert)
            if token.numel() == 0:
                continue
            x = hidden[token]
            gate, up = torch.nn.functional.linear(x, module.gate_up_proj[expert]).chunk(2, -1)
            down_input = module.act_fn(gate) * up
            for name, activation in (("gate", x), ("up", x), ("down", down_input)):
                key = index, f"mlp.experts.{expert}.{name}_proj"
                energy_samples.setdefault(key, []).append(activation.float().square().cpu())

    handles = []
    for index, layer in enumerate(teacher.model.layers):
        handles.append(layer.register_forward_hook(partial(capture_block, index)))
        for name, module in get_spherical_linears(layer).items():
            handles.append(module.register_forward_hook(partial(capture_linear, (index, name))))
        if hasattr(layer.mlp, "experts"):
            handles.append(
                layer.mlp.experts.register_forward_pre_hook(partial(capture_experts, index))
            )
    with torch.no_grad():
        for batch in samples["train"].split(config.batch_size):
            teacher(batch.cuda(), use_cache=False)
    for handle in handles:
        handle.remove()
    teacher.cpu()
    energies = {key: torch.cat(values).mean(0) for key, values in energy_samples.items()}
    for expert in (0, 1):
        assert (0, f"mlp.experts.{expert}.down_proj") in energies
    for expert in (2, 3):
        assert (0, f"mlp.experts.{expert}.down_proj") not in energies
    # Compute initializer order from actual modules, not the layout's reporting order.
    for layer in teacher.model.layers:
        split_work_experts(layer)
    expected_keys = iter(
        (index, name)
        for index, layer in enumerate(teacher.model.layers)
        for name in get_spherical_linears(layer)
    )
    if config_type is SAGEConfig:
        original_init = SAGEPseudoQuantizedLinear.__init__

        def observe_initialization(self, linear, **kwargs):
            if "channel_energy" in kwargs:
                key = next(expected_keys)
                expected = energies.get(key)
                if expected is None:
                    assert kwargs["channel_energy"] is None
                    assert kwargs["esg"] is False and kwargs["adaptive_bits"] is False
                    assert kwargs["codebook_bits"] == 7
                else:
                    torch.testing.assert_close(kwargs["channel_energy"], expected)
                    assert kwargs["esg"] and kwargs["adaptive_bits"]
            original_init(self, linear, **kwargs)

        monkeypatch.setattr(SAGEPseudoQuantizedLinear, "__init__", observe_initialization)

    original_optimize = pq_opt.optimize_module
    student_inputs = []

    def observe_training(layer, train_data, *args, **kwargs):
        index = len(student_inputs)
        if index in (0, 2):
            assert not layer.mlp.gate.weight.requires_grad
        student_inputs.append([batch[0].cpu().clone() for batch in train_data[0].batches])
        for actual, expected in zip(train_data[1].batches, teacher_outputs[index], strict=True):
            torch.testing.assert_close(actual.cpu(), expected)
        return original_optimize(layer, train_data, *args, **kwargs)

    monkeypatch.setattr(pq_opt, "optimize_module", observe_training)
    optimize(config)
    if config_type is SAGEConfig:
        assert next(expected_keys, None) is None
    assert missing_checkpoints(config) == []
    assert not checkpoint_path(config, 0, "mlp.gate").exists()
    torch.testing.assert_close(student_inputs[0], teacher_inputs[0])
    assert not torch.equal(torch.cat(student_inputs[1]), torch.cat(teacher_inputs[1]))

    pseudo_class = (
        SAGEPseudoQuantizedLinear if config_type is SAGEConfig else SPHPseudoQuantizedLinear
    )
    for name in layout[0]:
        state = torch.load(checkpoint_path(config, 0, name), weights_only=True)
        pseudo: Any = pseudo_class.from_state_dict(state)
        if name.startswith(("mlp.experts.2.", "mlp.experts.3.")):
            assert int(pseudo.radius_bits) == 3
            if config_type is SAGEConfig:
                torch.testing.assert_close(pseudo.channel_unpermutation, torch.arange(128))
        set_module_by_name(teacher.model.layers[0], name, pseudo)
    propagated = []
    handle = teacher.model.layers[1].register_forward_pre_hook(
        lambda _module, inputs: propagated.append(inputs[0].detach().cpu())
    )
    teacher.half().cuda()
    with torch.no_grad():
        for batch in samples["train"].split(config.batch_size):
            teacher(batch.cuda(), use_cache=False)
    handle.remove()
    torch.testing.assert_close(student_inputs[1], propagated)
    teacher.cpu()

    saved = {
        path: path.read_bytes()
        for pattern in ("*.pt", "*.format.json")
        for path in (config.checkpoints / config.model.name).glob(pattern)
    }
    if config_type is SAGEConfig:
        plan = json.loads((config.checkpoints / config.model.name / "0.format.json").read_text())
        unmeasured = {
            f"mlp.experts.{index}.{name}_proj"
            for index in (2, 3)
            for name in ("gate", "up", "down")
        }
        assert set(plan["unmeasured"]) == unmeasured
        assert plan["matrices_without_activation"] == 6
        assert not unmeasured.intersection(plan)
        for name in set(layout[0]) - unmeasured:
            entry = plan[name]
            assert entry["codebook_bits"] == int(
                min(entry["candidate_errors"], key=entry["candidate_errors"].get)
            )

    def reject_training(*_args, **_kwargs):
        pytest.fail("complete resume must not retrain")

    monkeypatch.setattr(pq_opt, "optimize_module", reject_training)
    # Complete checkpoints remain usable even if a later invocation disallows partial ABS.
    optimize(replace(config, allow_partial=False) if config_type is SAGEConfig else config)
    assert all(path.read_bytes() == value for path, value in saved.items())
    export(config)
    restored: Any = AutoModelForCausalLM.from_pretrained(
        config.exported_model, dtype="auto", local_files_only=True, experts_implementation="eager"
    ).eval()
    expected: Any = AutoModelForCausalLM.from_pretrained(
        config.model, dtype="auto", local_files_only=True, experts_implementation="eager"
    ).eval()
    target_keys = set()
    for index, names in enumerate(layout):
        for name in names:
            state = torch.load(
                checkpoint_path(config, index, name), weights_only=True, map_location="cuda"
            )
            weight = pseudo_class.from_state_dict(state).pseudo_weight().to(source_dtype).cpu()
            torch.testing.assert_close(
                projection_weight(restored.model.layers[index], name), weight, atol=0, rtol=0
            )
            projection_weight(expected.model.layers[index], name).data.copy_(weight)
            if name.startswith("mlp.experts."):
                suffix = "down_proj" if name.endswith("down_proj") else "gate_up_proj"
                target_keys.add(f"model.layers.{index}.mlp.experts.{suffix}")
            else:
                target_keys.add(f"model.layers.{index}.{name}.weight")
    assert restored.state_dict().keys() == original.keys()
    for name, value in restored.state_dict().items():
        if name not in target_keys:
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    assert "quantization_config" not in restored.config.to_dict()
    with torch.no_grad():
        restored.cuda()
        expected.cuda()
        actual = restored(samples["validation"].cuda(), use_cache=False).logits
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual, expected(samples["validation"].cuda(), use_cache=False).logits
        )
    restored.cpu()
    expected.cpu()
    validate(config)
    report = capsys.readouterr().out
    assert "model.layers.0.mlp.experts.3.down_proj" in report
    assert "model.layers.1.mlp.down_proj" in report

    partial_config = replace(config, output_dir=config.output_dir / "partial", max_layers=1)
    for name in layout[0]:
        path = checkpoint_path(partial_config, 0, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(checkpoint_path(config, 0, name).read_bytes())
    assert missing_checkpoints(partial_config) == []
    export(partial_config)
    partial_model: Any = AutoModelForCausalLM.from_pretrained(
        partial_config.exported_model, dtype="auto", local_files_only=True
    )
    for index in (1, 2):
        for name, value in partial_model.model.layers[index].state_dict().items():
            torch.testing.assert_close(
                value, original[f"model.layers.{index}.{name}"], atol=0, rtol=0
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ParoQuant uses CUDA")
def test_sage_requires_explicit_partial_abs(source):
    _model, config, _samples = source
    with pytest.raises(ValueError, match="6 matrices saw no tokens"):
        optimize(replace(config, allow_partial=False))
    assert not list(config.checkpoints.rglob("*.pt"))
    assert not list(config.checkpoints.rglob("*.format.json"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ParoQuant uses CUDA")
@pytest.mark.parametrize("esg,adaptive_bits", [(True, False), (False, True), (False, False)])
def test_sage_sparse_ablations(source, esg, adaptive_bits):
    _model, config, _samples = source
    config = replace(
        config,
        esg=esg,
        adaptive_bits=adaptive_bits,
        allow_partial=adaptive_bits,
        epochs=0,
        max_layers=1,
    )
    optimize(config)
    state = torch.load(checkpoint_path(config, 0, "mlp.experts.3.down_proj"), weights_only=True)
    restored: Any = SAGEPseudoQuantizedLinear.from_state_dict(state)
    assert int(restored.radius_bits) == 3
    torch.testing.assert_close(restored.channel_unpermutation, torch.arange(128))
    assert torch.isfinite(restored.pseudo_weight()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ParoQuant uses CUDA")
def test_sage_top_two_full_coverage_and_incomplete_resume(source, monkeypatch):
    model, config, _samples = source
    model.config.num_experts_per_tok = 2
    model.save_pretrained(config.model)
    config = replace(config, allow_partial=False, epochs=0, max_layers=1)
    optimize(config)
    _, layout = _model_contract(config.model)
    plan_path = config.checkpoints / config.model.name / "0.format.json"
    plan = json.loads(plan_path.read_text())
    assert set(plan) == set(layout[0])
    assert all(len(entry["candidate_errors"]) == 3 for entry in plan.values())
    checkpoint_path(config, 0, "mlp.experts.3.down_proj").unlink()
    assert len(missing_checkpoints(config)) == 1
    calls = []
    original_optimize = pq_opt.optimize_module

    def observe_training(*args, **kwargs):
        calls.append(True)
        return original_optimize(*args, **kwargs)

    monkeypatch.setattr(pq_opt, "optimize_module", observe_training)
    optimize(config)
    assert calls == [True]
    assert missing_checkpoints(config) == []
    # An incomplete layer is reinitialized, unlike byte-preserving complete
    # resume. CUDA reductions need not reproduce candidate-report bytes.
    rebuilt = json.loads(plan_path.read_text())
    assert set(rebuilt) == set(layout[0])
    for name, entry in rebuilt.items():
        state = torch.load(checkpoint_path(config, 0, name), weights_only=True)
        assert int(state["radius_bits"]) == entry["radius_bits"]
        assert entry["codebook_bits"] == int(
            min(entry["candidate_errors"], key=entry["candidate_errors"].get)
        )
