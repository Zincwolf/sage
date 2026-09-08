"""Real E8P block reconstruction and shared-stage integration, without downloads."""

from dataclasses import fields, replace
from typing import Any, cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from weight_vq.blockwise.artifacts import PROJECTIONS
from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.hessian import LayerHessianCollectors
from weight_vq.blockwise.pipeline import export, hessian, quantize
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.blockwise.replay import capture_replay
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.quip_sharp.quantization import quantize_layer, reconstruct_dense_weight


@pytest.fixture
def tiny_quip(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    config = QuIPSharpConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "rp",
        output_dir=tmp_path / "run",
        devices=[0],
        model_dtype="float32",
        num_cpu_threads=2,
        hessian_samples=5,
        hessian_sequence_length=16,
        hessian_chunk_size=3,
        train_samples=4,
        validation_samples=2,
        sequence_length=16,
        reconstruction_epochs=1,
        quip_tune_iters=1,
        batch_size=4,
    )
    hf = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    hf._attn_implementation = "sdpa"
    return config, hf


def test_method_owned_statistics_and_paper_defaults(tiny_quip):
    config, _ = tiny_quip
    defaults = QuIPSharpConfig(config.model, config.dataset, config.output_dir, [0])
    assert defaults.hessian_samples == 6144
    assert defaults.hessian_sequence_length == 2048
    assert defaults.hessian_recipe(1)["statistics_dtype"] == "float64"
    assert "statistics_dtype" not in {field.name for field in fields(defaults)}
    assert not defaults.use_fp64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="E8P requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_projections_reconstruct_and_export_weights(tiny_quip, dtype):
    config, hf = tiny_quip
    config = replace(config, gradient_checkpointing=True)
    model = Qwen3ForCausalLM(hf).to(dtype).eval().requires_grad_(False)
    tokens = torch.randint(0, 128, (6, 16))
    train = capture_replay(model, tokens[:4], device="cuda:0", microbatch_size=1)
    valid = capture_replay(model, tokens[4:], device="cuda:0", microbatch_size=1)
    layer = model.model.layers[0].cuda()
    collectors = LayerHessianCollectors(layer, include_padding=True, dtype=torch.float64)
    train_replay = train.forward_original(layer, device="cuda:0", collectors=collectors)
    valid_replay = valid.forward_original(layer, device="cuda:0")
    config.hessians.mkdir(parents=True)
    config.checkpoints.mkdir()
    for name, collector in collectors.collectors.items():
        torch.save(collector.close(center=True), config.hessians / f"0_{name}.pt")
    completed = []

    class ObservedReconstruction(DistributedReconstruction):
        def __call__(self, block, name, orig_dtype):
            super().__call__(block, name, orig_dtype)
            completed.append(name)

    trainer = ObservedReconstruction(
        train_replay, valid_replay, config, DistributedContext(0, 1, torch.device("cuda:0")), 7
    )
    layer.requires_grad_(True)
    quantize_layer(layer, 0, config, trainer)
    assert completed == [f"0_{short}" for _, short, *_ in PROJECTIONS]
    with torch.no_grad():
        layer.cuda()
        for attribute, short, *_ in PROJECTIONS:
            saved = torch.load(config.checkpoints / f"0_{short}.pt", weights_only=True)
            weight = reconstruct_dense_weight(saved, torch.device("cuda:0"))
            linear = layer.get_submodule(attribute)
            hidden = torch.randn(2, 16, weight.shape[1], device="cuda:0") * 0.1
            actual = linear(hidden)
            expected = torch.nn.functional.linear(hidden, weight.cuda())
            # Upstream train mode caches transformed weights in FP16 and uses FP16 GEMM.
            torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.002)
            assert saved["SU"].dtype == saved["SV"].dtype == torch.float32


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="NCCL integration requires two GPUs")
def test_fp64_hessian_restart_and_hf_export(tiny_quip):
    single, hf = tiny_quip
    model = Qwen3ForCausalLM(hf)
    model.save_pretrained(single.model)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "test": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        single.model
    )
    multiple = replace(single, devices=[0, 1], output_dir=single.output_dir.parent / "dp2")
    payload = {
        "protocol": single.calibration_recipe(),
        "hessian": torch.randint(0, 128, (5, 16)),
        "train": torch.randint(0, 128, (4, 16)),
        "validation": torch.randint(0, 128, (2, 16)),
    }
    for config in (single, multiple):
        config.output_dir.mkdir()
        torch.save(payload, config.calibration)
        hessian(config)
        stamp = (config.hessians / "complete.json").stat().st_mtime_ns
        hessian(config)
        assert (config.hessians / "complete.json").stat().st_mtime_ns == stamp
    for name in ("qkv", "o", "up", "down"):
        one = torch.load(single.hessians / f"0_{name}.pt", weights_only=True)
        many = torch.load(multiple.hessians / f"0_{name}.pt", weights_only=True)
        assert one["flatH"].dtype == one["mu"].dtype == torch.float64
        assert one["ct"] == many["ct"] == 80
        torch.testing.assert_close(one["flatH"], many["flatH"], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(one["mu"], many["mu"], rtol=1e-10, atol=1e-10)
    quantize(multiple)
    original = torch.load(multiple.checkpoints / "0_v.pt", weights_only=True)
    stamp = (multiple.checkpoints / "0.complete").stat().st_mtime_ns
    quantize(multiple)
    assert (multiple.checkpoints / "0.complete").stat().st_mtime_ns == stamp
    (multiple.checkpoints / "0.complete").unlink()
    torch.save({"invalid": True}, multiple.checkpoints / "0_v.pt")
    quantize(multiple)
    restored = torch.load(multiple.checkpoints / "0_v.pt", weights_only=True)
    for key in ("Qidxs", "SU", "SV"):
        torch.testing.assert_close(restored[key], original[key], rtol=0, atol=0)
    export(multiple)
    dense = AutoModelForCausalLM.from_pretrained(multiple.exported_model, local_files_only=True)
    for attribute, short, *_ in PROJECTIONS:
        saved = torch.load(multiple.checkpoints / f"0_{short}.pt", weights_only=True)
        expected = reconstruct_dense_weight(saved, torch.device("cuda:0"))
        torch.testing.assert_close(
            dense.model.layers[0].get_parameter(f"{attribute}.weight"), expected
        )
    assert torch.isfinite(dense(payload["validation"]).logits).all()
    assert cast(Any, dense).generate(payload["validation"][:1], max_new_tokens=2).shape == (1, 18)
