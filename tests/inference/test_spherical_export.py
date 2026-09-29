"""Trained spherical state -> existing packed ABI; no synthetic export payloads."""

import json
import os
from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from paroquant.optim.sagelinear import SAGEPseudoQuantizedLinear
from paroquant.optim.sphlinear import (
    SPHPseudoQuantizedLinear,
    direction_codebook_from_logits,
    radius_fake_quantize,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from weight_vq import utils
from weight_vq.sage.config import SAGEConfig
from weight_vq.sphquant.config import SPHQuantConfig
from weight_vq.sphquant.packing import pack_spherical
from weight_vq.sphquant.pipeline import PROJECTIONS, checkpoint_path, export, optimize, validate

from .reference import decode_sage, decode_sphquant, unpack_sage, unpack_sphquant


@pytest.mark.parametrize("sage,bits", [(False, 6), (True, 6), (True, 7), (True, 8)])
@pytest.mark.parametrize("permuted", [False, True])
def test_pack_native_spherical_state(sage, bits, permuted):
    torch.manual_seed(17)
    n, k = 19, 256  # Exercise SAGE's padded final row tile and multiple groups.
    state = {
        "signs": torch.randint(0, 2, (n, k), dtype=torch.int8) * 2 - 1,
        "direction_indices": torch.randint(0, 1 << bits, (n, k // 8)),
        "radius_logits": torch.randn(n, k // 8),
        "angle_logits": torch.randn(1 << bits, 7),
        "radius_bits": torch.tensor(10 - bits),
        "group_size": torch.tensor(128),
        "sphquant_version": torch.tensor(1),
    }
    # A constant-radius group and an FP16-underflow group must have zero step.
    state["radius_logits"][0, :16] = 0.125
    state["radius_logits"][0, 16:] = -40
    if sage:
        order = torch.randperm(k) if permuted else torch.arange(k)
        state.update(channel_unpermutation=order.argsort(), sage_version=torch.tensor(1))
    packed = pack_spherical(state, sage=sage)
    if sage:
        signs, indices, codes = unpack_sage(packed, n)
        grid = packed["radius_grid"].permute(0, 2, 1, 3).reshape(-1, k // 128, 2)[:n]
        minimum, step = grid.unbind(-1)
        decoded = decode_sage(packed, n)
        assert bool(packed["permutation"].numel()) == permuted
        if permuted:
            assert torch.equal(
                packed["permutation"].long(), state["channel_unpermutation"].argsort()
            )
        assert torch.count_nonzero(packed["packed_codes"][-1, :, n % 16 :]) == 0
        assert torch.count_nonzero(packed["radius_grid"][-1, :, n % 16 :]) == 0
    else:
        signs, indices, codes = unpack_sphquant(packed["packed_codes"], packed["radius_extra"])
        minimum, step = packed["radius_mins"], packed["radius_scales"]
        decoded = decode_sphquant(packed)
    assert torch.equal(signs.reshape(n, k), state["signs"])
    assert torch.equal(indices.reshape(n, -1), state["direction_indices"])
    radii = radius_fake_quantize(F.softplus(state["radius_logits"]), bits=10 - bits)
    decoded_radii = minimum.float()[..., None] + codes.reshape(n, -1, 16) * step.float()[..., None]
    torch.testing.assert_close(decoded_radii.flatten(1), radii, rtol=0, atol=0)
    assert torch.count_nonzero(step[0]) == 0
    book = direction_codebook_from_logits(state["angle_logits"]).half().float()
    expected = (
        state["signs"].reshape(n, -1, 8) * book[state["direction_indices"]] * radii[..., None]
    ).flatten(1)
    if sage:
        expected = expected[:, state["channel_unpermutation"]]
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    assert all(tensor.is_contiguous() for tensor in packed.values())


@pytest.fixture
def source(tmp_path, source_dtype, request):
    torch.manual_seed(31)
    width = getattr(request, "param", 128)
    real_width = width == 2560
    hf = Qwen3Config(
        vocab_size=32,
        hidden_size=width,
        intermediate_size=9728 if real_width else 256,
        num_hidden_layers=1 if real_width else 2,
        num_attention_heads=32 if real_width else 4,
        num_key_value_heads=8 if real_width else 2,
        head_dim=128 if real_width else 32,
        attention_bias=not real_width,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(hf, dtype=source_dtype).eval()
    model.generation_config.max_new_tokens = 3
    path = tmp_path / "source"
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
        chat_template="{{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(path)
    return path, model


@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("config_type", [SPHQuantConfig, SAGEConfig])
def test_trained_spherical_hf_roundtrip(source, tmp_path, config_type, capsys):
    path, model = source
    config = config_type(path, tmp_path / "unused-data", tmp_path / "run", max_layers=1)
    assert not config.export_compressed
    config = replace(config, export_compressed=True)
    sage = config_type is SAGEConfig
    saved, expected_weights = {}, {}
    for index, name in enumerate(PROJECTIONS):
        linear = model.model.layers[0].get_submodule(name)
        kwargs: dict[str, Any] = dict(sample_size=256, kmeans_iterations=1, codebook_bits=6)
        if sage:
            kwargs.update(
                codebook_bits=6 + index % 3,
                esg=bool(index % 2),
                adaptive_bits=False,
                channel_energy=torch.arange(linear.in_features).float() + 1,
            )
        constructor = SAGEPseudoQuantizedLinear if sage else SPHPseudoQuantizedLinear
        trained = constructor(linear, **kwargs).float()
        optimizer = torch.optim.Adam(trained.parameters(), lr=1e-3)
        x = torch.randn(3, linear.in_features)
        loss = F.mse_loss(trained(x), F.linear(x, linear.weight.float(), None))
        loss.backward()
        optimizer.step()
        # Frozen bias is still part of the saved representation, not the source.
        if trained.bias is not None:
            with torch.no_grad():
                trained.bias.fill_(0.125)
        state = {key: value.detach().clone() for key, value in trained.state_dict().items()}
        saved[name] = state
        expected_weights[name] = trained.pseudo_weight()
        checkpoint = checkpoint_path(config, 0, name)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint)
    export(config)
    with pytest.raises(FileExistsError):
        export(config)
    metadata = json.loads((config.exported_model / "config.json").read_text())[
        "quantization_config"
    ]
    assert metadata["method"] == ("sage" if sage else "sphquant")
    assert metadata["synthetic_seed"] is None
    assert len(metadata["modules"]) == 7
    if sage:
        assert {spec["codebook_bits"] for spec in metadata["modules"].values()} == {6, 7, 8}
        assert {spec["permuted"] for spec in metadata["modules"].values()} == {True, False}
    # Independent artifact: no source or native state needed for loading/validate.
    path.rename(tmp_path / "hidden-source")
    config.checkpoints.rename(tmp_path / "hidden-checkpoints")
    loaded, info = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportGeneralTypeIssues]
        config.exported_model,
        dtype="auto",
        device_map="cpu",
        output_loading_info=True,
        local_files_only=True,
    )
    assert not any(info.values())
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert loaded.generation_config.max_new_tokens == 3
    assert (
        AutoTokenizer.from_pretrained(config.exported_model).chat_template
        == "{{ messages[0]['content'] }}"
    )
    for name, state in saved.items():
        linear = loaded.model.layers[0].get_submodule(name)
        tensors = dict(linear.named_buffers())
        decoded = decode_sage(tensors, linear.out_features) if sage else decode_sphquant(tensors)
        # Only the learned direction codebook is newly rounded to kernel FP16.
        torch.testing.assert_close(decoded, expected_weights[name], rtol=1e-3, atol=1e-6)
        if linear.bias is not None:
            torch.testing.assert_close(
                linear.bias, state["bias"].to(linear.bias.dtype), rtol=0, atol=0
            )
    for name, value in model.state_dict().items():
        parent, field = name.rsplit(".", 1)
        if parent in metadata["modules"] and field in ("weight", "bias"):
            if field == "weight":
                assert name not in loaded.state_dict()
            continue
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    validate(config)
    assert "bits_per_weight,codebook_bits" in capsys.readouterr().out
    resharded = tmp_path / "resharded"
    loaded.save_pretrained(resharded, max_shard_size="20KB")
    reloaded = AutoModelForCausalLM.from_pretrained(resharded, dtype="auto", device_map="cpu")
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)


def test_spherical_packing_rejects_incompatible_representations():
    state = {"angle_logits": torch.empty(256, 7)}
    with pytest.raises(ValueError, match="SPHQuant requires 64/4"):
        pack_spherical(state, sage=False)
    with pytest.raises(ValueError, match="method does not match"):
        pack_spherical(state, sage=True)
    state.update(
        sage_version=torch.tensor(1), group_size=torch.tensor(64), radius_bits=torch.tensor(2)
    )
    with pytest.raises(ValueError, match="128-group bit split"):
        pack_spherical(state, sage=True)


@pytest.mark.cuda
@pytest.mark.parametrize("source_dtype", [torch.bfloat16])
@pytest.mark.parametrize("source", [128, 2560], indirect=True, ids=["tiny", "qwen3-4b-width"])
@pytest.mark.parametrize("config_type", [SPHQuantConfig, SAGEConfig])
def test_spherical_optimize_dual_export(source, tmp_path, config_type, record_property):
    from transformers import StaticCache

    from .reference import error_metrics

    path, _ = source
    config = config_type(
        path,
        tmp_path / "data.pt",
        tmp_path / "run",
        device=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
        sample_size=256,
        kmeans_iterations=1,
        epochs=1,
        batch_size=2,
        cache_shards=1,
        checkpointing=False,
        max_layers=1,
    )
    samples = {"train": torch.randint(2, 32, (4, 8)), "validation": torch.randint(2, 32, (2, 8))}
    torch.save(samples, config.calibration)
    utils.manifest_path(config.calibration).write_text(
        json.dumps(
            {"tensors": {name: {"shape": list(value.shape)} for name, value in samples.items()}}
        )
    )
    optimize(config)
    originals = {
        file: file.read_bytes() for file in config.checkpoints.rglob("*") if file.is_file()
    }
    export(config)
    qdq_path = tmp_path / "qdq"
    config.exported_model.rename(qdq_path)
    packed_config = replace(config, export_compressed=True)
    # Changing the export policy does not change optimization/resume state.
    optimize(packed_config)
    assert all(file.read_bytes() == value for file, value in originals.items())
    export(packed_config)
    assert all(file.read_bytes() == value for file, value in originals.items())
    path.rename(tmp_path / "hidden-source")
    config.checkpoints.rename(tmp_path / "hidden-checkpoints")
    models = [
        AutoModelForCausalLM.from_pretrained(
            artifact,
            dtype="auto",
            device_map="cuda",
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        for artifact in (qdq_path, config.exported_model)
    ]
    tokens = samples["validation"][:1].cuda()
    caches = [StaticCache(config=model.config, max_cache_len=12) for model in models]
    with torch.inference_mode():
        for step in range(3):
            inputs = tokens if step == 0 else tokens[:, -1:]
            logits = [
                model(inputs, past_key_values=cache, use_cache=True).logits
                for model, cache in zip(models, caches, strict=True)
            ]
            assert all(torch.isfinite(value).all() for value in logits)
            assert all(cache.get_seq_length() == 8 + step for cache in caches)
            for key, value in error_metrics(*logits).items():
                record_property(f"step_{step}_{key}", value)
        generated = models[1].generate(  # pyright: ignore[reportAttributeAccessIssue]
            tokens,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            cache_implementation="static",
            disable_compile=True,
        )
        assert generated.shape == (1, 11)
        assert torch.equal(generated[:, :8], tokens)
    validate(packed_config)
