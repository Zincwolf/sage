"""Mixed Qwen3.5 attention, method reconstruction, and complete VL dense export."""

from dataclasses import replace

import pytest
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from weight_vq.blockwise.artifacts import (
    atomic_write_completion,
    hessian_complete,
    layer_complete,
    projections,
)
from weight_vq.blockwise.hessian import LayerHessianCollectors, collector_modules
from weight_vq.blockwise.models import language_backbone
from weight_vq.blockwise.pipeline import (
    _initialize_run,
    _model_contract,
    _run_rank,
    export,
    hessian,
    quantize,
    validate,
)
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.config import QTIPConfig
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.unisvq.config import UniSVQConfig
from weight_vq.unisvq.reconstruction import UniSVQReconstruction


@pytest.fixture
def qwen35(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    text = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [4, 6, 6],
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=4,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=128,
        num_position_embeddings=16,
    )
    hf = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
    )
    hf._attn_implementation = "sdpa"
    model = Qwen3_5ForConditionalGeneration(hf).eval()
    config = QTIPConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "rp",
        output_dir=tmp_path / "run",
        devices=[0],
        model_dtype="bfloat16",
        num_cpu_threads=2,
        hessian_samples=4,
        hessian_sequence_length=8,
        hessian_chunk_size=2,
        train_samples=4,
        validation_samples=2,
        sequence_length=8,
        reconstruction_epochs=1,
        batch_size=4,
    )
    return model, config


def save_source(model, config):
    state = model.state_dict()
    # HF has no MTP module, but the real source checkpoint contains these tensors.
    mtp = torch.randn(128, 256)
    state["mtp.fc.weight"] = mtp
    model.save_pretrained(config.model, state_dict=state, max_shard_size="200KB")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {
                    "[UNK]": 0,
                    "Hello": 1,
                    **{f"token{i}": i for i in range(2, 10)},
                    "<|image_pad|>": 10,
                    "<|video_pad|>": 11,
                    "<|vision_start|>": 12,
                    "<|vision_end|>": 13,
                },
                unk_token="[UNK]",
            )
        ),
        unk_token="[UNK]",
    )
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=4, temporal_patch_size=1, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=4, temporal_patch_size=1, merge_size=2),
        chat_template="{{ messages[0]['content'] }}",
    ).save_pretrained(config.model)
    return mtp


def test_native_replay_hessians_and_sample_independence(qwen35):
    model, _ = qwen35
    backbone = language_backbone(model)
    tokens = torch.randint(20, 128, (2, 8))
    captured = []
    handles = [
        layer.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
        for layer in backbone.layers
    ]
    with torch.no_grad():
        expected = backbone(tokens, use_cache=False).last_hidden_state
    for handle in handles:
        handle.remove()
    state = capture_replay(model, tokens, device="cpu", microbatch_size=2)
    for index, layer in enumerate(backbone.layers):
        specs = projections(layer_type=layer.block_type)
        assert set(attribute for attribute, *_ in specs) == {
            name for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)
        }
        inputs = {}
        handles = []
        for attribute, *_ in specs:

            def capture(module, args, key=attribute, captured_inputs=inputs):
                captured_inputs[key] = args[0].detach().clone()

            handles.append(layer.get_submodule(attribute).register_forward_pre_hook(capture))
        collectors = LayerHessianCollectors(layer, include_padding=True)
        replay = state.forward_original(layer, device="cpu", collectors=collectors)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(replay.targets, captured[index], rtol=0, atol=0)
        representative = collector_modules(layer)
        statistics = collectors.close()
        for attribute, _, group, _ in specs:
            peer = next(
                name for name, *_ in specs if layer.get_submodule(name) is representative[group]
            )
            torch.testing.assert_close(inputs[attribute], inputs[peer], rtol=0, atol=0)
        assert all(saved["ct"] == tokens.numel() for saved in statistics.values())
        if index == 0:
            assert set(statistics) == {"in", "out", "up", "down"}
            assert layer.linear_attn.in_proj_a.weight.shape == (32, 128)
        else:
            assert layer.self_attn.q_proj.out_features == 2 * 4 * 32
        replay.loss_parts(layer, torch.arange(2), "cpu")[0].backward()
        assert not replay.targets.requires_grad
    torch.testing.assert_close(backbone.norm(state.inputs), expected)
    with torch.no_grad():
        torch.testing.assert_close(
            backbone(tokens, use_cache=False).last_hidden_state, expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            backbone(tokens[:1], use_cache=False).last_hidden_state, expected[:1]
        )


def test_layout_completion_and_recipe(qwen35):
    model, config = qwen35
    save_source(model, config)
    assert _model_contract(config.model) == ("qwen3_5", 2)
    config.checkpoints.mkdir(parents=True)
    config.hessians.mkdir()
    for _, short, group, _ in projections(layer_type="linear_attention"):
        (config.checkpoints / f"0_{short}.pt").touch()
        (config.hessians / f"0_{group}.pt").touch()
    (config.checkpoints / "0.auxiliary.pt").touch()
    atomic_write_completion(0, config.checkpoints, layer_type="linear_attention")
    assert layer_complete(0, config.checkpoints, layer_type="linear_attention")
    assert hessian_complete(0, config.hessians, layer_type="linear_attention")
    assert not layer_complete(0, config.checkpoints)
    assert not hessian_complete(0, config.hessians)
    (config.checkpoints / "0_in_a.pt").unlink()
    assert not layer_complete(0, config.checkpoints, layer_type="linear_attention")
    recipe = config.recipe(model_family="qwen3_5", layer_end=2)
    model.config.text_config.layer_types.reverse()
    model.config.save_pretrained(config.model)
    assert recipe != config.recipe(model_family="qwen3_5", layer_end=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real quantizers require CUDA")
@pytest.mark.parametrize("method", [QTIPConfig, QuIPSharpConfig, UniSVQConfig])
def test_real_mixed_reconstruction_resume_and_vl_export(qwen35, method, monkeypatch, capsys):
    model, base = qwen35
    model.bfloat16().requires_grad_(False)
    fields = {
        key: value
        for key, value in vars(base).items()
        if key in method.__dataclass_fields__ and key != "gradient_checkpointing"
    }
    config = method(**fields, gradient_checkpointing=True)
    if isinstance(config, (QuIPSharpConfig, UniSVQConfig)):
        config = replace(config, quip_tune_iters=1)
    mtp = save_source(model, config)
    config.output_dir.mkdir()
    tokens = torch.randint(20, 128, (4, 8))
    torch.save(
        {
            "protocol": config.calibration_recipe(),
            "hessian": tokens,
            "train": tokens,
            "validation": tokens[:2],
        },
        config.calibration,
    )
    hessian(config)
    hessian(config)
    config.checkpoints.mkdir()
    if isinstance(config, QTIPConfig):
        torch.save(torch.randn(512, 2), config.checkpoints / "tlut-l16-b9-v2.pt")
    _initialize_run(config, "qwen3_5", 2)
    teacher = {key: value.clone() for key, value in model.state_dict().items()}
    trainer_type = UniSVQReconstruction if method is UniSVQConfig else DistributedReconstruction
    original_call = trainer_type.__call__
    boundaries = []
    reconstructed = {}

    def observe(self, layer, name, dtype):
        index = int(name.split("_")[0])
        frozen = {
            key: value.detach().clone()
            for key, value in layer.named_parameters()
            if not value.requires_grad
        }
        if index == 0:
            assert {
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
            } <= frozen.keys()
        if method is UniSVQConfig:
            norms = (
                ("linear_attn.norm.weight",)
                if index == 0
                else ("self_attn.q_norm.weight", "self_attn.k_norm.weight")
            )
            assert all(key in frozen for key in norms)
        original_call(self, layer, name, dtype)
        for key, value in frozen.items():
            torch.testing.assert_close(layer.get_parameter(key), value, rtol=0, atol=0)
        boundaries.append(name)
        if method is UniSVQConfig or name.endswith("_down"):
            source = self.train.inputs[:1].to(self.distributed.device, dtype=torch.float32)
            with torch.no_grad():
                expected = self.train.state.forward(layer, source).cpu()
            reconstructed[index] = (source.cpu(), expected)

    monkeypatch.setattr(trainer_type, "__call__", observe)
    try:
        _run_rank(0, config, "quantize", model, 0, 2, config.output_dir / "store", config.hessians)
    finally:
        torch.set_grad_enabled(True)
    assert len(boundaries) == (2 if method is UniSVQConfig else 15)
    for key, value in teacher.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    stamps = {path: path.stat().st_mtime_ns for path in config.checkpoints.iterdir()}
    quantize(config)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())
    (config.checkpoints / "0_in_a.pt").unlink()
    resumed = replace(config, devices=[0, 1]) if torch.cuda.device_count() > 1 else config
    quantize(resumed)
    assert layer_complete(0, config.checkpoints, layer_type="linear_attention")
    assert all(
        path.stat().st_mtime_ns == stamp
        for path, stamp in stamps.items()
        if path.name.startswith("1")
    )
    export(config)
    restored = AutoModelForImageTextToText.from_pretrained(config.exported_model).eval()
    processor = AutoProcessor.from_pretrained(config.exported_model, local_files_only=True)
    assert processor.chat_template == "{{ messages[0]['content'] }}"
    with safe_open(config.exported_model / "model.safetensors", framework="pt") as shard:
        torch.testing.assert_close(shard.get_tensor("mtp.fc.weight"), mtp, rtol=0, atol=0)
    quantized_names = {
        f"model.language_model.layers.{index}.{attribute}.weight"
        for index, layer in enumerate(language_backbone(model).layers)
        for attribute, *_ in projections(layer_type=layer.block_type)
    }
    for key, value in teacher.items():
        if key in quantized_names:
            assert not torch.equal(restored.state_dict()[key], value)
        elif "norm" not in key or not key.startswith("model.language_model.layers."):
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        image = {
            "input_ids": torch.tensor([[20, 12, 10, 13, 21]]),
            "pixel_values": torch.randn(4, 48, dtype=torch.bfloat16),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0]]),
        }
        output = restored(**image, use_cache=False)
        assert torch.isfinite(output.logits).all()
        restored.float()
        replay = capture_replay(restored, tokens[:1], device="cpu", microbatch_size=1)
        for index, (source, expected) in reconstructed.items():
            actual = replay.forward(language_backbone(restored).layers[index], source)
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
    validate(config)
    assert "model.language_model.layers.0.linear_attn.in_proj_a" in capsys.readouterr().out
    # Published statistics cannot be reused after swapping attention layouts.
    model.config.text_config.layer_types.reverse()
    model.config.save_pretrained(config.model)
    with pytest.raises(RuntimeError, match="Hessian protocol differs"):
        hessian(config)
