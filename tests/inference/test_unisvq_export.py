"""Real UniSVQ LDLQ artifacts to the existing HadaQuant inference layout."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3Config,
    StaticCache,
)

from weight_vq.blockwise.artifacts import PROJECTIONS, atomic_write_completion
from weight_vq.blockwise.pipeline import _initialize_run, export, hessian, quantize, validate
from weight_vq.checkpoint import synthetic_unisvq
from weight_vq.inference.linear import UniSVQLinear
from weight_vq.unisvq.config import UniSVQConfig
from weight_vq.unisvq.quantization import reconstruct_dense_weight

from .reference import block128_reference, decode_unisvq, error_metrics, unpack_unisvq


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layer_start", [0, 1])
def test_native_unisvq_packed_roundtrip(tmp_path, dtype, layer_start, capsys):
    from unisvq import codebook
    from unisvq.algo import quip

    torch.set_num_threads(2)
    torch.manual_seed(29)
    config = UniSVQConfig(
        tmp_path / "source",
        tmp_path / "data",
        tmp_path / "run",
        [0],
        layer_start=layer_start,
        layer_end=layer_start + 1,
        quip_tune_iters=1,
    )
    assert config.export_compressed is False
    compressed = replace(config, export_compressed=True)
    recipe = config.recipe(model_family="qwen3", layer_end=layer_start + 1)
    assert compressed.recipe(model_family="qwen3", layer_end=layer_start + 1) == recipe
    assert "export_compressed" not in recipe
    assert recipe["hadamard"] == "upstream-128-block"
    hf = Qwen3Config(
        vocab_size=16,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        tie_word_embeddings=True,
        attention_bias=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(hf, dtype=dtype).eval()
    model.generation_config.max_new_tokens = 3
    model.save_pretrained(config.model)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
        chat_template="{{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(config.model)
    _initialize_run(config, "qwen3", layer_start + 1)
    args = SimpleNamespace(
        **asdict(config),
        use_fp64=False,
        lora_rank=0,
        incoh_mode="had",
        rescale_WH=False,
        resid_scale_override=-1,
        no_use_buffered=False,
        lowmem_ldlq=False,
    )
    cb = codebook.get_codebook("identical", codebook_bit=2)
    saved_states = {}
    auxiliary: dict[str, torch.Tensor] = {"input_layernorm.weight": torch.full((128,), 0.75)}
    for attribute, short, *_ in PROJECTIONS:
        linear = model.model.layers[layer_start].get_submodule(attribute)
        k = linear.in_features
        weight = linear.weight.detach().float()
        scale = weight.square().mean().sqrt()
        # Actual released quantization, not synthetic random packed indices.
        _, saved = quip.quantize(torch.eye(k), weight / scale, 0, cb, args, device="cpu")
        # Exercise non-BF16-representable FP16 scales as reconstruction can produce.
        saved["SU"] = (saved["SU"] + torch.linspace(-0.02, 0.02, k)).half()
        saved["SV"] = (saved["SV"] * scale).half()
        assert not torch.equal(saved["SU"].float(), saved["SU"].bfloat16().float())
        torch.save(saved, config.checkpoints / f"{layer_start}_{short}.pt")
        saved_states[attribute] = saved
        if linear.bias is not None:
            auxiliary[f"{attribute}.bias"] = torch.full_like(linear.bias, 0.125)
    torch.save(auxiliary, config.checkpoints / f"{layer_start}.auxiliary.pt")
    atomic_write_completion(layer_start, config.checkpoints)
    before = {path.name: path.read_bytes() for path in config.checkpoints.iterdir()}
    _initialize_run(compressed, "qwen3", layer_start + 1)
    export(compressed)
    assert before == {path.name: path.read_bytes() for path in config.checkpoints.iterdir()}
    with pytest.raises(FileExistsError):
        export(compressed)
    quant = json.loads((config.exported_model / "config.json").read_text())["quantization_config"]
    assert quant["method"] == "unisvq"
    assert quant["synthetic_seed"] is None
    assert len(quant["modules"]) == 7
    assert all(spec == {"fp16_scales": True} for spec in quant["modules"].values())
    assert str(config.model) not in json.dumps(quant)
    config.model.rename(tmp_path / "hidden-source")
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
        == tokenizer.chat_template
    )
    for attribute, saved in saved_states.items():
        linear = loaded.model.layers[layer_start].get_submodule(attribute)
        assert isinstance(linear, UniSVQLinear)
        assert linear.SU.dtype == linear.SV.dtype == torch.float16
        assert linear.Qidxs.dtype == torch.uint8
        assert linear.cb_scale.dtype == linear.cb_zero.dtype == torch.bfloat16
        assert linear.dense is None
        for key in ("Qidxs", "SU", "SV"):
            torch.testing.assert_close(getattr(linear, key), saved[key], rtol=0, atol=0)
        packed = linear.state_dict()
        # Independent Cartesian digits plus Sylvester 128-block transforms.
        torch.testing.assert_close(
            decode_unisvq(packed), cb.by_idxs(saved["Qidxs"]).float(), rtol=0, atol=0
        )
        raw = decode_unisvq(packed)
        transformed = block128_reference(block128_reference(raw).T).T
        expected = transformed * saved["SU"].float() * saved["SV"].float()[:, None]
        torch.testing.assert_close(
            reconstruct_dense_weight(saved, torch.device("cpu")), expected, atol=1e-7, rtol=1e-5
        )
    prefix = f"model.layers.{layer_start}."
    for name, value in model.state_dict().items():
        if name.endswith(".weight") and name.removesuffix(".weight") in quant["modules"]:
            assert name not in loaded.state_dict()
            continue
        expected = (
            auxiliary.get(name.removeprefix(prefix), value) if name.startswith(prefix) else value
        )
        torch.testing.assert_close(
            loaded.state_dict()[name], cast(torch.Tensor, expected).to(value.dtype), rtol=0, atol=0
        )
    validate(compressed)
    assert "bits_per_weight" in capsys.readouterr().out
    resharded = tmp_path / "resharded"
    loaded.save_pretrained(resharded, max_shard_size="20KB")
    reloaded = AutoModelForCausalLM.from_pretrained(resharded, dtype="auto", device_map="cpu")
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor, rtol=0, atol=0)


def test_released_cartesian_grid_exhaustive():
    from unisvq import codebook

    cb = codebook.get_codebook("identical", codebook_bit=2)
    indices = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    packed = {
        "Qidxs": indices,
        "cb_scale": cb.linear_proj_weight[0, 0],
        "cb_zero": cb.linear_proj_bias[0],
    }
    expected_digits = torch.cartesian_prod(*([torch.arange(4)] * 4)).reshape(8, 128)
    torch.testing.assert_close(unpack_unisvq(packed), expected_digits, rtol=0, atol=0)
    torch.testing.assert_close(decode_unisvq(packed), cb.by_idxs(indices).float(), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("k,n", [(2560, 1024), (9728, 2560)])
@pytest.mark.parametrize("rows", [1, 9])
@torch.inference_mode()
def test_fp16_scale_runtime_boundary(k, n, rows):
    torch.set_num_threads(2)
    state = cast(dict[str, torch.Tensor], synthetic_unisvq(k, n, seed=32))
    state.pop("metadata")
    state["SU"] = torch.linspace(-1.12, 1.12, k).half()
    state["SV"] = (torch.linspace(-1.12, 1.12, n) / k**0.5).half()
    reference = {
        key: value.bfloat16() if key in ("SU", "SV") else value for key, value in state.items()
    }
    trained = UniSVQLinear(state, cached_dense_weight=False).cuda()
    bf16 = UniSVQLinear(reference, cached_dense_weight=False).cuda()
    inputs = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    actual = trained(inputs)
    # Isolate the scale dtype boundary from the already-tested kernel math:
    # FP16 checkpoint scales must be converted, never reinterpreted as BF16 bits.
    torch.testing.assert_close(actual, bf16(inputs), rtol=0, atol=0)
    for key in ("SU", "SV"):
        torch.testing.assert_close(getattr(trained, key).cpu(), state[key], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trained_unisvq_dual_export(tmp_path, dtype, record_property):
    torch.set_num_threads(2)
    torch.manual_seed(31)
    config = UniSVQConfig(
        tmp_path / "source",
        tmp_path / "data",
        tmp_path / "run",
        [0],
        model_dtype=str(dtype).removeprefix("torch."),
        layer_end=1,
        num_cpu_threads=2,
        hessian_samples=4,
        hessian_sequence_length=16,
        hessian_chunk_size=2,
        forward_microbatch_size=2,
        reconstruction_microbatch_size=2,
        train_samples=4,
        validation_samples=2,
        sequence_length=16,
        reconstruction_epochs=1,
        batch_size=2,
        quip_tune_iters=1,
    )
    hf = Qwen3Config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        tie_word_embeddings=True,
        attention_bias=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(hf, dtype=dtype)
    model.save_pretrained(config.model)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")), unk_token="[UNK]"
    ).save_pretrained(config.model)
    config.output_dir.mkdir()
    samples = {
        "protocol": config.calibration_recipe(),
        "hessian": torch.randint(1, 32, (4, 16)),
        "train": torch.randint(1, 32, (4, 16)),
        "validation": torch.randint(1, 32, (2, 16)),
    }
    torch.save(samples, config.calibration)
    hessian(config)
    quantize(config)
    before = {path.name: path.read_bytes() for path in config.checkpoints.iterdir()}
    export(config)
    qdq = config.output_dir / "qdq"
    config.exported_model.rename(qdq)
    compressed = replace(config, export_compressed=True)
    quantize(compressed)
    export(compressed)
    assert before == {path.name: path.read_bytes() for path in config.checkpoints.iterdir()}
    config.model.rename(tmp_path / "hidden-source")
    config.checkpoints.rename(tmp_path / "hidden-checkpoints")
    dense, info = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportGeneralTypeIssues]
        qdq,
        dtype="auto",
        device_map="cuda:0",
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    assert not any(info.values())
    packed, info = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportGeneralTypeIssues]
        compressed.exported_model,
        dtype="auto",
        device_map="cuda:0",
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    assert not any(info.values())
    for attribute, short, *_ in PROJECTIONS:
        saved = torch.load(tmp_path / "hidden-checkpoints" / f"0_{short}.pt", weights_only=True)
        linear = packed.model.layers[0].get_submodule(attribute)
        assert linear.SU.dtype == linear.SV.dtype == torch.float16
        for key in ("Qidxs", "SU", "SV"):
            torch.testing.assert_close(getattr(linear, key).cpu(), saved[key], rtol=0, atol=0)
        # Check serialization against the exporter's CUDA decode. CPU/GPU FP32
        # reductions can straddle a final FP16/BF16 rounding boundary; independent
        # CPU layout/transform coverage lives in the roundtrip test above.
        expected = reconstruct_dense_weight(saved, torch.device("cuda:0"))
        torch.testing.assert_close(
            dense.model.layers[0].get_parameter(f"{attribute}.weight").cpu(),
            expected.to(dtype),
            rtol=0,
            atol=0,
        )
    tokens = cast(torch.Tensor, samples["validation"])[:1].cuda()
    models = (dense, packed)
    caches = [StaticCache(config=loaded.config, max_cache_len=20) for loaded in models]
    with torch.inference_mode():
        for step in range(3):
            inputs = tokens if step == 0 else tokens[:, -1:]
            logits = [
                loaded(inputs, past_key_values=cache, use_cache=True).logits
                for loaded, cache in zip(models, caches, strict=True)
            ]
            assert all(torch.isfinite(value).all() for value in logits)
            assert all(cache.get_seq_length() == 16 + step for cache in caches)
            # Kernel BF16 scales/GEMM and affine GEMV differ from QDQ rounding.
            # Record model-level differences, without a strict logits gate.
            for key, value in error_metrics(*logits).items():
                record_property(f"step_{step}_{key}", value)
        generated = packed.generate(
            tokens,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            cache_implementation="static",
            disable_compile=True,
        )
        assert generated.shape == (1, 19)
        assert torch.equal(generated[:, :16], tokens)
    validate(compressed)
