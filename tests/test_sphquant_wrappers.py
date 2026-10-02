"""SPHQuant/SAGE text reconstruction and complete HF wrapper export."""

import json
import os
from dataclasses import replace
from functools import partial
from typing import Any

import pytest
import torch
from paroquant.cli import optimize as pq_opt
from paroquant.optim.sagelinear import SAGEPseudoQuantizedLinear
from paroquant.optim.sphlinear import SPHPseudoQuantizedLinear
from paroquant.optim.util import get_spherical_linears, set_module_by_name
from safetensors import safe_open
from test_qtip_ministral import ministral as ministral
from test_qtip_ministral import save_source as save_ministral
from test_qwen35 import qwen35 as qwen35
from test_qwen35 import save_source as save_qwen35
from transformers import AutoModelForImageTextToText, AutoProcessor

from weight_vq import utils
from weight_vq.blockwise.models import language_backbone
from weight_vq.sage.config import SAGEConfig
from weight_vq.sphquant.config import SPHQuantConfig
from weight_vq.sphquant.pipeline import (
    _model_contract,
    checkpoint_path,
    export,
    missing_checkpoints,
    optimize,
    validate,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ParoQuant reconstruction uses CUDA")
@pytest.mark.parametrize("config_type", [SPHQuantConfig, SAGEConfig])
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16])
def test_ministral_train_resume_export(ministral, config_type, source_dtype, monkeypatch, capsys):
    model, source_config = ministral
    _check_wrapper_pipeline(model, source_config, config_type, source_dtype, monkeypatch, capsys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ParoQuant reconstruction uses CUDA")
@pytest.mark.parametrize("config_type", [SPHQuantConfig, SAGEConfig])
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16])
def test_qwen35_train_resume_export(qwen35, config_type, source_dtype, monkeypatch, capsys):
    model, source_config = qwen35
    _check_wrapper_pipeline(model, source_config, config_type, source_dtype, monkeypatch, capsys)


def _check_wrapper_pipeline(
    model: Any, source_config, config_type, source_dtype, monkeypatch, capsys
):
    model.to(source_dtype)
    family = model.config.model_type
    mtp = None
    if family == "qwen3_5":
        mtp = save_qwen35(model, source_config)
    else:
        save_ministral(model, source_config)
    config = config_type(
        model=source_config.model,
        calibration=source_config.calibration,
        output_dir=source_config.output_dir,
        device=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
        sample_size=256,
        kmeans_iterations=1,
        epochs=1,
        batch_size=2,
        cache_shards=1,
    )
    _, layout = _model_contract(config.model)
    config.output_dir.mkdir()
    # Cross the native YaRN original context boundary, including attention scaling.
    samples = {
        "train": torch.randint(20, 128, (4, 40)),
        "validation": torch.randint(20, 128, (2, 40)),
    }
    torch.save(samples, config.calibration)
    utils.manifest_path(config.calibration).write_text(
        json.dumps(
            {"tensors": {name: {"shape": list(value.shape)} for name, value in samples.items()}}
        )
    )
    original = {name: value.clone() for name, value in model.state_dict().items()}
    # Load an independent FP16 teacher through HF, as in ParoQuant. Casting the
    # in-memory BF16 fixture would retain rounded non-persistent RoPE buffers,
    # whereas from_pretrained regenerates them from configuration.
    model = (
        AutoModelForImageTextToText.from_pretrained(
            config.model, dtype=torch.float16, local_files_only=True
        )
        .half()
        .eval()
    )
    teacher_outputs, teacher_inputs, energies = {}, {}, {}

    def capture_layer(index, _module, inputs, output):
        teacher_inputs.setdefault(index, []).append(inputs[0].detach().cpu())
        teacher_outputs.setdefault(index, []).append(output.detach().cpu())

    def capture_energy(name, _module, inputs, _output):
        x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
        energies.setdefault(name, []).append(x.square().cpu())

    handles = []
    for index, layer in enumerate(language_backbone(model).layers):
        handles.append(layer.register_forward_hook(partial(capture_layer, index)))
        assert set(get_spherical_linears(layer)) == set(layout[index])
        for name, module in get_spherical_linears(layer).items():
            handles.append(module.register_forward_hook(partial(capture_energy, (index, name))))
    model.cuda()
    with torch.no_grad():
        for batch in samples["train"].split(config.batch_size):
            model(batch.cuda(), use_cache=False)
    for handle in handles:
        handle.remove()
    model.cpu()

    # Qwen3.5 registers out_proj before in_proj, unlike their execution order.
    expected_energies = iter(
        torch.cat(energies[index, name]).mean(0)
        for index, layer in enumerate(language_backbone(model).layers)
        for name in get_spherical_linears(layer)
    )
    if config_type is SAGEConfig:
        original_init = SAGEPseudoQuantizedLinear.__init__

        def observe_initialization(self, linear, **kwargs):
            if "channel_energy" in kwargs:
                torch.testing.assert_close(kwargs["channel_energy"], next(expected_energies))
            original_init(self, linear, **kwargs)

        monkeypatch.setattr(SAGEPseudoQuantizedLinear, "__init__", observe_initialization)

    original_optimize = pq_opt.optimize_module
    student_inputs = []

    def observe_training(layer, train_data, *args, **kwargs):
        index = len(student_inputs)
        if family == "qwen3_5" and index == 0:
            for name in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
                assert not layer.get_parameter(f"{name}.weight").requires_grad
        student_inputs.append([batch[0].cpu().clone() for batch in train_data[0].batches])
        for actual, expected in zip(train_data[1].batches, teacher_outputs[index], strict=True):
            torch.testing.assert_close(actual.cpu(), expected)
        return original_optimize(layer, train_data, *args, **kwargs)

    monkeypatch.setattr(pq_opt, "optimize_module", observe_training)
    optimize(config)
    if config_type is SAGEConfig:
        assert next(expected_energies, None) is None
    assert missing_checkpoints(config) == []
    torch.testing.assert_close(student_inputs[0], teacher_inputs[0])
    assert not torch.equal(torch.cat(student_inputs[1]), torch.cat(teacher_inputs[1]))

    pseudo_class = (
        SAGEPseudoQuantizedLinear if config_type is SAGEConfig else SPHPseudoQuantizedLinear
    )
    first = language_backbone(model).layers[0]
    for name in layout[0]:
        state = torch.load(checkpoint_path(config, 0, name), weights_only=True)
        set_module_by_name(first, name, pseudo_class.from_state_dict(state))
    propagated = []
    handle = (
        language_backbone(model)
        .layers[1]
        .register_forward_pre_hook(
            lambda _module, inputs: propagated.append(inputs[0].detach().cpu())
        )
    )
    model.half().cuda()
    with torch.no_grad():
        for batch in samples["train"].split(config.batch_size):
            model(batch.cuda(), use_cache=False)
    handle.remove()
    torch.testing.assert_close(student_inputs[1], propagated)
    model.cpu()

    saved = {
        path: path.read_bytes()
        for pattern in ("*.pt", "*.format.json")
        for path in (config.checkpoints / config.model.name).glob(pattern)
    }

    def reject_training(*_args, **_kwargs):
        pytest.fail("resume must reuse complete checkpoints")

    monkeypatch.setattr(pq_opt, "optimize_module", reject_training)
    optimize(config)
    assert all(path.read_bytes() == contents for path, contents in saved.items())
    export(config)
    restored: Any = AutoModelForImageTextToText.from_pretrained(
        config.exported_model, dtype="auto", local_files_only=True
    ).eval()
    expected: Any = AutoModelForImageTextToText.from_pretrained(
        config.model, dtype="auto", local_files_only=True
    ).eval()
    targets = set()
    for index, layer in enumerate(language_backbone(expected).layers):
        for name in layout[index]:
            # Export reconstructs on CUDA; compare on the same device to avoid
            # CPU/CUDA trigonometric rounding differences at FP16 boundaries.
            state = torch.load(
                checkpoint_path(config, index, name), map_location="cuda", weights_only=True
            )
            weight = pseudo_class.from_state_dict(state).pseudo_weight().to(source_dtype).cpu()
            full_name = f"model.language_model.layers.{index}.{name}.weight"
            targets.add(full_name)
            torch.testing.assert_close(restored.get_parameter(full_name), weight, atol=0, rtol=0)
            layer.get_parameter(f"{name}.weight").data.copy_(weight)
    for name, value in restored.state_dict().items():
        if name not in targets:
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    processor = AutoProcessor.from_pretrained(config.exported_model, local_files_only=True)
    assert processor.chat_template == "{{ messages[0]['content'] }}"
    assert processor.image_processor.patch_size == 4
    if family == "mistral3":
        assert (config.exported_model / "tekken.json").read_text() == "tekken fixture"
    else:
        with safe_open(config.exported_model / "model.safetensors", framework="pt") as shard:
            torch.testing.assert_close(shard.get_tensor("mtp.fc.weight"), mtp, atol=0, rtol=0)
        assert not any("in_proj_a" in path.name or "in_proj_b" in path.name for path in saved)
    assert "quantization_config" not in restored.config.to_dict()
    with torch.no_grad():
        restored.to("cuda")
        expected.to("cuda")
        text_inputs = {"input_ids": samples["validation"].cuda(), "use_cache": False}
        image_inputs = {
            "input_ids": torch.tensor([[11, 10, 10, 10, 10, 12]], device="cuda"),
            "pixel_values": torch.randn(1, 3, 16, 16, device="cuda", dtype=source_dtype),
            "image_sizes": torch.tensor([[16, 16]], device="cuda"),
            "use_cache": False,
        }
        if family == "qwen3_5":
            image_inputs = {
                "input_ids": torch.tensor([[20, 12, 10, 13, 21]], device="cuda"),
                "pixel_values": torch.randn(4, 48, device="cuda", dtype=source_dtype),
                "image_grid_thw": torch.tensor([[1, 2, 2]], device="cuda"),
                "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0]], device="cuda"),
                "use_cache": False,
            }
        for inputs in (text_inputs, image_inputs):
            actual = restored(**inputs).logits
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected(**inputs).logits)
    restored.cpu()
    expected.cpu()
    validate(config)
    report = capsys.readouterr().out
    assert "model.language_model.layers.1.mlp.down_proj" in report
    if family == "qwen3_5":
        assert "model.language_model.layers.0.linear_attn.in_proj_qkv" in report

    # A partial-layer run does not demand or modify later blocks at export.
    partial_config = replace(config, output_dir=config.output_dir / "partial", max_layers=1)
    for name in layout[0]:
        path = checkpoint_path(partial_config, 0, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(checkpoint_path(config, 0, name).read_bytes())
    assert missing_checkpoints(partial_config) == []
    export(partial_config)
    partial_model = AutoModelForImageTextToText.from_pretrained(
        partial_config.exported_model, dtype="auto", local_files_only=True
    )
    for name, value in partial_model.state_dict().items():
        if not name.startswith("model.language_model.layers.0."):
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
