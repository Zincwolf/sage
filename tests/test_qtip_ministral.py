"""Ministral language replay and complete multimodal QDQ serialization."""

import json
from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Ministral3Config,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PixtralImageProcessor,
    PixtralProcessor,
    PixtralVisionConfig,
    PreTrainedTokenizerFast,
)

from weight_vq.blockwise.artifacts import PROJECTIONS, atomic_write_completion
from weight_vq.blockwise.hessian import LayerHessianCollectors
from weight_vq.blockwise.models import language_backbone
from weight_vq.blockwise.pipeline import _model_contract, export, hessian, quantize, validate
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.config import QTIPConfig


@pytest.fixture(params=[128, 256], ids=["square", "rectangular"])
def ministral(tmp_path, request):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    text = Ministral3Config(
        vocab_size=128,
        hidden_size=request.param,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )
    vision = PixtralVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=16,
        patch_size=4,
    )
    hf = Mistral3Config(text_config=text, vision_config=vision, tie_word_embeddings=False)
    hf._attn_implementation = "sdpa"
    model = Mistral3ForConditionalGeneration(hf).eval()
    config = QTIPConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "rp",
        output_dir=tmp_path / "output",
        devices=[0],
        model_dtype="float32",
        num_cpu_threads=2,
        hessian_samples=8,
        hessian_sequence_length=16,
        hessian_chunk_size=4,
        train_samples=8,
        validation_samples=4,
        sequence_length=16,
        reconstruction_epochs=1,
    )
    return model, config


@pytest.mark.parametrize("sliding_window", [None, 4])
def test_native_replay_and_hessian(ministral, sliding_window):
    model, _ = ministral
    backbone = language_backbone(model)
    backbone.config.sliding_window = sliding_window
    # Cross original_max_position_embeddings to exercise native attention scaling.
    tokens = torch.randint(11, 128, (2, 48))
    with torch.no_grad():
        expected = backbone(tokens, use_cache=False).last_hidden_state
    state = capture_replay(model, tokens, device="cpu", microbatch_size=1)
    for layer in backbone.layers:
        collectors = LayerHessianCollectors(layer, include_padding=True)
        replay = state.forward_original(layer, device="cpu", collectors=collectors)
        for payload in collectors.close().values():
            assert payload["ct"] == tokens.numel()
        replay.loss_parts(layer, torch.arange(2), "cpu")[0].backward()
        assert layer.input_layernorm.weight.grad is not None
    torch.testing.assert_close(backbone.norm(state.inputs), expected)


def save_source(model, config):
    model.save_pretrained(config.model)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"[UNK]": 0, "test": 1, **{f"token{i}": i for i in range(2, 10)}, "[IMG]": 10},
                unk_token="[UNK]",
            )
        ),
        unk_token="[UNK]",
        additional_special_tokens=["[IMG]"],
        chat_template="{{ messages[0]['content'] }}",
    )
    PixtralProcessor(
        image_processor=PixtralImageProcessor(patch_size=4, size={"longest_edge": 16}),
        tokenizer=tokenizer,
    ).save_pretrained(config.model)


def test_export_preserves_multimodal_shell(ministral, monkeypatch, capsys):
    model, config = ministral
    save_source(model, config)
    assert _model_contract(config.model) == ("mistral3", 2)
    backbone = language_backbone(model)
    untouched = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not name.startswith("model.language_model.layers.")
    }
    config.checkpoints.mkdir(parents=True)
    with torch.no_grad():
        for index, layer in enumerate(backbone.layers):
            for attribute, short, *_ in PROJECTIONS:
                weight = layer.get_parameter(f"{attribute}.weight")
                weight.mul_(0.8)
                torch.save({"weight": weight.detach()}, config.checkpoints / f"{index}_{short}.pt")
            auxiliary = {}
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                layer.get_parameter(name).mul_(1.1)
                auxiliary[name] = layer.get_parameter(name).detach()
            torch.save(auxiliary, config.checkpoints / f"{index}.auxiliary.pt")
            atomic_write_completion(index, config.checkpoints)
    (config.checkpoints / "qtip_config.json").write_text(
        json.dumps(config.recipe(model_family="mistral3", layer_end=2))
    )
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.reconstruct_dense_weight",
        lambda saved, recipe: saved["weight"],
    )
    export(config)
    restored = AutoModelForImageTextToText.from_pretrained(config.exported_model).eval()
    processor = AutoProcessor.from_pretrained(config.exported_model, local_files_only=True)
    assert processor.chat_template == "{{ messages[0]['content'] }}"
    assert processor.image_processor.patch_size == 4
    for name, value in untouched.items():
        torch.testing.assert_close(restored.get_parameter(name), value, atol=0, rtol=0)
    with torch.no_grad():
        tokens = torch.tensor([[11, 12, 13]])
        torch.testing.assert_close(restored(tokens).logits, model(tokens).logits)
        image_inputs = {
            "input_ids": torch.tensor([[11, 10, 10, 10, 10, 12]]),
            "pixel_values": torch.randn(1, 3, 16, 16),
            "image_sizes": torch.tensor([[16, 16]]),
        }
        torch.testing.assert_close(restored(**image_inputs).logits, model(**image_inputs).logits)
    validate(config)
    assert "model.language_model.layers.0.self_attn.q_proj" in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real QTIP requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ministral_real_quantize_restart_export(ministral, dtype):
    model, config = ministral
    model.to(dtype)
    config = replace(config, model_dtype=str(dtype).removeprefix("torch."))
    save_source(model, config)
    config.output_dir.mkdir()
    torch.save(
        {
            "protocol": config.calibration_recipe(),
            "hessian": torch.randint(11, 128, (8, 16)),
            "train": torch.randint(11, 128, (8, 16)),
            "validation": torch.randint(11, 128, (4, 16)),
        },
        config.calibration,
    )
    hessian(config)
    config.checkpoints.mkdir()
    torch.save(torch.randn(512, 2), config.checkpoints / "tlut-l16-b9-v2.pt")
    quantize(config)
    stamps = {path: path.stat().st_mtime_ns for path in config.checkpoints.iterdir()}
    quantize(config)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())
    saved = {
        path.name: torch.load(path, weights_only=True) for path in config.checkpoints.glob("1*.pt")
    }
    (config.checkpoints / "1.complete").unlink()
    torch.save({"partial": True}, config.checkpoints / "1_q.pt")
    quantize(config)
    for name, expected in saved.items():
        torch.testing.assert_close(
            torch.load(config.checkpoints / name, weights_only=True), expected, atol=0, rtol=0
        )
    export(config)
    restored = AutoModelForImageTextToText.from_pretrained(
        config.exported_model, dtype="auto"
    ).eval()
    for name, value in model.named_parameters():
        if not name.startswith("model.language_model.layers."):
            torch.testing.assert_close(restored.get_parameter(name), value, atol=0, rtol=0)
    for index, layer in enumerate(language_backbone(restored).layers):
        auxiliary = torch.load(config.checkpoints / f"{index}.auxiliary.pt", weights_only=True)
        for name, value in auxiliary.items():
            torch.testing.assert_close(layer.get_parameter(name), value.to(dtype), atol=0, rtol=0)
    with torch.no_grad():
        assert torch.isfinite(restored(torch.tensor([[11, 12, 13]])).logits).all()
