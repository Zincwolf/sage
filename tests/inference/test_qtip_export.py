"""QTIP paper-layout compatibility and real-checkpoint HF serialization."""

import json
from dataclasses import replace
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
from weight_vq.checkpoint import synthetic_qtip
from weight_vq.inference.linear import QTIPLinear
from weight_vq.qtip.artifacts import reconstruct_dense_weight
from weight_vq.qtip.config import QTIPConfig

from .reference import decode_qtip, error_metrics, qtip_hadamard_reference


@pytest.fixture
def qtip_source(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(43)
    config = QTIPConfig(
        tmp_path / "source",
        tmp_path / "data",
        tmp_path / "run",
        [0],
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
    )
    hf = Qwen3Config(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        tie_word_embeddings=True,
        attention_bias=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(hf, dtype=torch.bfloat16).eval()
    model.generation_config.max_new_tokens = 3
    model.save_pretrained(config.model)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{{ messages[0]['content'] }}",
    ).save_pretrained(config.model)
    config.checkpoints.mkdir(parents=True)
    # Avoid the unrelated million-point LUT initializer; LDLQ and training are real.
    torch.save(torch.randn(512, 2), config.checkpoints / "tlut-l16-b9-v2.pt")
    return config, model


def test_requested_kernel_cannot_silently_fallback(qtip_source, monkeypatch):
    from lib import codebook
    from lib.algo import finetune

    cfg, _ = qtip_source
    cfg = replace(cfg, packed_inference_kernel=True)
    monkeypatch.setattr(codebook, "HAS_QTIP_KERNELS", False)
    with pytest.raises(ValueError, match="kernel support is unavailable"):
        finetune.quantize_finetune_decoder_layer(
            torch.nn.Identity(), (), 0, torch.nn.Identity(), cfg, "cpu", None, None
        )


def test_private_kernel_registration():
    from lib import codebook
    from lib.utils.kernel_check import has_kernel

    from weight_vq.inference import _qtip  # pyright: ignore[reportAttributeAccessIssue]

    assert codebook.qtip_kernels is _qtip
    assert has_kernel("quantlut_sym", 16, 2, 2, 9, 16, 16)
    for n, k in ((256, 256), (1024, 2560), (4096, 2560), (9728, 2560), (2560, 4096), (2560, 9728)):
        assert hasattr(torch.ops.quip_lib, f"decompress_matvec_qtip_{n}_1_{k}_2")
        assert hasattr(_qtip, f"decompress_matvec_16_9_2_1_{n}_1_{k}")


@pytest.mark.parametrize("kernel_layout", [False, True])
def test_config_recipe_and_legacy_resume(qtip_source, kernel_layout):
    cfg, _ = qtip_source
    assert not cfg.packed_inference_kernel and not cfg.export_compressed
    with pytest.raises(ValueError, match="requires packed_inference_kernel"):
        replace(cfg, export_compressed=True)
    cfg = replace(cfg, packed_inference_kernel=kernel_layout)
    recipe = cfg.recipe(model_family="qwen3", layer_end=1)
    assert recipe["packed_inference_kernel"] == kernel_layout
    assert "export_compressed" not in recipe
    if kernel_layout:
        assert (
            replace(cfg, export_compressed=True).recipe(model_family="qwen3", layer_end=1) == recipe
        )
        with pytest.raises(ValueError, match="2-bit"):
            replace(cfg, export_compressed=True, K=3)
    # Only the known historical recipe lacked the field; it meant forced False.
    legacy = dict(recipe)
    legacy.pop("packed_inference_kernel")
    path = cfg.checkpoints / "qtip_config.json"
    path.write_text(json.dumps(legacy))
    before = path.read_bytes()
    if kernel_layout:
        with pytest.raises(RuntimeError, match="configuration differs"):
            _initialize_run(cfg, "qwen3", 1)
        with pytest.raises(RuntimeError, match="configuration differs"):
            export(replace(cfg, export_compressed=True))
    else:
        _initialize_run(cfg, "qwen3", 1)
    assert path.read_bytes() == before
    path.write_text(json.dumps(recipe))
    _initialize_run(cfg, "qwen3", 1)
    with pytest.raises(RuntimeError, match="configuration differs"):
        _initialize_run(replace(cfg, packed_inference_kernel=not kernel_layout), "qwen3", 1)


@pytest.mark.parametrize("kernel_layout", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_trellis_dual_export_cpu(qtip_source, kernel_layout, dtype, tmp_path, capsys):
    from lib.codebook.bitshift import bitshift_codebook

    base, model = qtip_source
    model.to(dtype).save_pretrained(base.model)
    cfg = replace(base, packed_inference_kernel=kernel_layout, layer_start=1, layer_end=2)
    _initialize_run(cfg, "qwen3", 2)
    cb = bitshift_codebook(
        L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym", tlut=torch.randn(512, 2)
    )
    auxiliary = {"input_layernorm.weight": torch.full((256,), 0.75)}
    states = {}
    expected_weights = {}
    for attribute, short, *_ in PROJECTIONS:
        linear = model.model.layers[1].get_submodule(attribute)
        n, k = linear.weight.shape
        # Valid cyclic trellises through the native packer. This CPU test covers
        # serialization/layout, not quantization quality; the CUDA test runs LDLQ.
        words = torch.randint(-32768, 32768, (n * k // 256, 32), dtype=torch.int16)
        trellis = cb.unpack_trellis(words, 256)
        packed = cb.pack_trellis(trellis)
        torch.testing.assert_close(packed.view(torch.int16), words, rtol=0, atol=0)
        if kernel_layout:
            packed = (
                packed.view(torch.uint8)
                .view(-1, 2)
                .flip((-1,))
                .reshape(n // 32, 2, k // 32, 2, 32, 2)
                .permute(0, 2, 4, 3, 1, 5)
                .flip((-1,))
                .contiguous()
                .view(torch.int16)
                .reshape(words.shape)
            )
        saved = {
            "trellis": packed.view(torch.int16),
            "SU": torch.randn(k),
            "SV": torch.randn(n),
            "Wscale": torch.tensor(0.0173),
            "tlut": cb.tlut.detach().clone(),
            "rcp": 0,
        }
        torch.save(saved, cfg.checkpoints / f"1_{short}.pt")
        states[attribute] = saved
        if kernel_layout:
            raw = decode_qtip(saved)
        else:
            raw = (
                cb.recons(trellis)
                .permute(1, 2, 0)
                .reshape(n // 16, k // 16, 16, 16)
                .transpose(1, 2)
                .reshape(n, k)
            )
        expected = qtip_hadamard_reference(qtip_hadamard_reference(raw / 32).T).T
        expected = expected * saved["SU"] * (saved["SV"] * saved["Wscale"] * 32)[:, None]
        decoded = reconstruct_dense_weight(saved, cfg.recipe(model_family="qwen3", layer_end=2))
        torch.testing.assert_close(decoded, expected, rtol=1e-5, atol=1e-6)
        expected_weights[attribute] = decoded.to(dtype)
        if linear.bias is not None:
            auxiliary[f"{attribute}.bias"] = torch.full_like(linear.bias, 0.125)
    torch.save(auxiliary, cfg.checkpoints / "1.auxiliary.pt")
    atomic_write_completion(1, cfg.checkpoints)
    if not kernel_layout:
        # Exercise actual legacy export, not only recipe comparison.
        path = cfg.checkpoints / "qtip_config.json"
        legacy = json.loads(path.read_text())
        legacy.pop("packed_inference_kernel")
        path.write_text(json.dumps(legacy))
    before = {path.name: path.read_bytes() for path in cfg.checkpoints.iterdir()}
    export(cfg)
    dense = AutoModelForCausalLM.from_pretrained(cfg.exported_model, dtype="auto", device_map="cpu")
    for attribute, expected in expected_weights.items():
        torch.testing.assert_close(
            dense.model.layers[1].get_parameter(f"{attribute}.weight"), expected, rtol=0, atol=0
        )
    if not kernel_layout:
        assert before == {path.name: path.read_bytes() for path in cfg.checkpoints.iterdir()}
        return
    cfg.exported_model.rename(tmp_path / "qdq")
    compressed = replace(cfg, export_compressed=True)
    _initialize_run(compressed, "qwen3", 2)
    export(compressed)
    assert before == {path.name: path.read_bytes() for path in cfg.checkpoints.iterdir()}
    with pytest.raises(FileExistsError):
        export(compressed)
    cfg.model.rename(tmp_path / "hidden-source")
    cfg.checkpoints.rename(tmp_path / "hidden-checkpoints")
    loaded, info = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportGeneralTypeIssues]
        cfg.exported_model,
        dtype="auto",
        device_map="cpu",
        local_files_only=True,
        output_loading_info=True,
    )
    assert not any(info.values())
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert loaded.generation_config.max_new_tokens == 3
    assert (
        AutoTokenizer.from_pretrained(cfg.exported_model).chat_template
        == "{{ messages[0]['content'] }}"
    )
    quant = loaded.config.quantization_config
    assert quant.method == "qtip" and quant.synthetic_seed is None
    assert len(quant.modules) == 7
    for attribute, saved in states.items():
        linear = loaded.model.layers[1].get_submodule(attribute)
        assert isinstance(linear, QTIPLinear) and linear.dense is None
        assert quant.modules[f"model.layers.1.{attribute}"] == {"fp32_parameters": True}
        for key in ("trellis", "SU", "tlut"):
            torch.testing.assert_close(getattr(linear, key), saved[key], rtol=0, atol=0)
        torch.testing.assert_close(linear.SV, saved["SV"] * saved["Wscale"], rtol=0, atol=0)
        assert linear.SU.dtype == linear.SV.dtype == linear.tlut.dtype == torch.float32
    for name, value in model.state_dict().items():
        if name.endswith(".weight") and name.removesuffix(".weight") in quant.modules:
            assert name not in loaded.state_dict()
            continue
        expected = cast(torch.Tensor, auxiliary.get(name.removeprefix("model.layers.1."), value))
        torch.testing.assert_close(loaded.state_dict()[name], expected.to(dtype), rtol=0, atol=0)
    validate(compressed)
    assert "bits_per_weight" in capsys.readouterr().out
    loaded.save_pretrained(tmp_path / "resharded", max_shard_size="100KB")
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmp_path / "resharded", dtype="auto", device_map="cpu"
    )
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("k,n", [(2560, 1024), (9728, 2560)])
@pytest.mark.parametrize("rows", [1, 9])
@torch.inference_mode()
def test_fp32_parameter_runtime_boundary(k, n, rows):
    torch.set_num_threads(2)
    state = cast(dict[str, torch.Tensor], synthetic_qtip(k, n, seed=43))
    state["SU"] = torch.linspace(-1.1234, 1.1234, k)
    state["tlut"] = state["tlut"].float() + 0.000123
    reference = dict(state, tlut=state["tlut"].half())
    trained = QTIPLinear(state, cached_dense_weight=False).cuda()
    rounded = QTIPLinear(reference, cached_dense_weight=False).cuda()
    inputs = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(trained(inputs), rounded(inputs), rtol=0, atol=0)
    for key in ("SU", "tlut"):
        torch.testing.assert_close(getattr(trained, key).cpu(), state[key], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("kernel_layout", [False, True])
def test_trained_qtip_export(qtip_source, kernel_layout, tmp_path, record_property):
    cfg, _ = qtip_source
    cfg = replace(cfg, packed_inference_kernel=kernel_layout)
    samples = {
        "protocol": cfg.calibration_recipe(),
        "hessian": torch.randint(1, 32, (4, 16)),
        "train": torch.randint(1, 32, (4, 16)),
        "validation": torch.randint(1, 32, (2, 16)),
    }
    torch.save(samples, cfg.calibration)
    hessian(cfg)
    quantize(cfg)
    before = {path.name: path.read_bytes() for path in cfg.checkpoints.iterdir()}
    quantize(cfg)
    export(cfg)
    qdq = tmp_path / "qdq"
    cfg.exported_model.rename(qdq)
    dense = AutoModelForCausalLM.from_pretrained(
        qdq, dtype="auto", device_map="cuda", attn_implementation="sdpa"
    )
    tokens = cast(torch.Tensor, samples["validation"])[:1].cuda()
    with torch.inference_mode():
        assert torch.isfinite(dense(tokens).logits).all()
    if not kernel_layout:
        assert before == {path.name: path.read_bytes() for path in cfg.checkpoints.iterdir()}
        return
    compressed = replace(cfg, export_compressed=True)
    quantize(compressed)
    export(compressed)
    assert before == {path.name: path.read_bytes() for path in cfg.checkpoints.iterdir()}
    cfg.model.rename(tmp_path / "hidden-source")
    cfg.checkpoints.rename(tmp_path / "hidden-checkpoints")
    packed, info = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportGeneralTypeIssues]
        compressed.exported_model,
        dtype="auto",
        device_map="cuda",
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    assert not any(info.values())
    for attribute, short, *_ in PROJECTIONS:
        saved = torch.load(tmp_path / "hidden-checkpoints" / f"0_{short}.pt", weights_only=True)
        linear = packed.model.layers[0].get_submodule(attribute)
        for key in ("trellis", "SU", "tlut"):
            torch.testing.assert_close(getattr(linear, key).cpu(), saved[key], rtol=0, atol=0)
        expected = reconstruct_dense_weight(saved, cfg.recipe(model_family="qwen3", layer_end=1))
        torch.testing.assert_close(
            dense.model.layers[0].get_parameter(f"{attribute}.weight").cpu(),
            expected.bfloat16(),
            rtol=0,
            atol=0,
        )
    compiled = torch.compile(packed, fullgraph=True)
    models = (dense, packed, compiled)
    caches = [StaticCache(config=packed.config, max_cache_len=20) for _ in models]
    with torch.inference_mode():
        for step in range(3):
            inputs = tokens if step == 0 else tokens[:, -1:]
            logits = [
                model(inputs, past_key_values=cache, use_cache=True).logits
                for model, cache in zip(models, caches, strict=True)
            ]
            assert all(torch.isfinite(value).all() for value in logits)
            assert all(cache.get_seq_length() == 16 + step for cache in caches)
            for label, left, right in (
                ("qdq_packed", logits[0], logits[1]),
                ("eager_compiled", logits[1], logits[2]),
            ):
                for key, value in error_metrics(left, right).items():
                    record_property(f"{label}_{step}_{key}", value)
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
