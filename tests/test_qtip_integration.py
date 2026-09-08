"""Small real-trellis CUDA tests; no downloads or large calibration runs."""

import json
import shutil
from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.optim.optimizer import register_optimizer_step_post_hook
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from weight_vq.blockwise.artifacts import PROJECTIONS, quant_order
from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.hessian import LayerHessianCollectors
from weight_vq.blockwise.pipeline import export, hessian, quantize
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.artifacts import reconstruct_dense_weight
from weight_vq.qtip.config import QTIPConfig


@pytest.fixture
def tiny_qtip(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    config = QTIPConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "rp",
        output_dir=tmp_path / "single",
        devices=[0],
        model_dtype="float32",
        num_cpu_threads=2,
        hessian_samples=9,
        hessian_sequence_length=16,
        hessian_chunk_size=4,
        train_samples=9,
        validation_samples=4,
        sequence_length=16,
        reconstruction_epochs=1,
    )
    hf = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    hf._attn_implementation = "sdpa"
    return config, hf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real QTIP test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_block_updates_fixed_indices_and_dense_reconstruction(tiny_qtip, dtype):
    # Method-environment dependencies: actual QTIP, not a mocked quantizer.
    from lib.algo import finetune
    from lib.codebook.bitshift import bitshift_codebook

    config, hf = tiny_qtip
    config = replace(
        config,
        model_dtype=str(dtype).removeprefix("torch."),
        train_samples=8,
        reconstruction_epochs=5,
        gradient_checkpointing=True,
    )
    model = Qwen3ForCausalLM(hf).to(dtype).eval()
    tokens = torch.randint(0, 128, (12, 16))
    train = capture_replay(model, tokens[:8], device="cuda:0", microbatch_size=1)
    valid = capture_replay(model, tokens[8:], device="cuda:0", microbatch_size=1)
    layer = model.model.layers[0].cuda()
    collectors = LayerHessianCollectors(layer, include_padding=True)
    train_replay = train.forward_original(layer, device="cuda:0", collectors=collectors)
    valid_replay = valid.forward_original(layer, device="cuda:0")
    config.hessians.mkdir(parents=True)
    config.checkpoints.mkdir()
    for name, payload in collectors.close().items():
        torch.save(payload, config.hessians / f"0_{name}.pt")
    # Fixed LUT fixture bypasses the unrelated million-point k-means initializer.
    codebook = bitshift_codebook(
        L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym", tlut=torch.randn(512, 2)
    )
    trainer = DistributedReconstruction(
        train_replay, valid_replay, config, DistributedContext(0, 1, torch.device("cuda:0")), 7
    )
    completed = []

    def reconstruct(block, name, orig_dtype):
        block.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            train_replay.state.forward(block, train_replay.inputs[:1])
        indices = {
            key: value.cpu().clone()
            for key, value in block.state_dict().items()
            if key.endswith("trellis")
        }
        before = {key: value.detach().cpu().clone() for key, value in block.named_parameters()}
        initial_loss = trainer.validation_loss(block, orig_dtype)
        steps = 0
        changed = set()

        def observe_step(_optimizer, _args, _kwargs):
            nonlocal steps
            steps += 1
            changed.update(
                key
                for key, value in block.named_parameters()
                if not torch.equal(value.detach().cpu(), before[key])
            )
            for key, value in indices.items():
                torch.testing.assert_close(block.state_dict()[key].cpu(), value, rtol=0, atol=0)

        with register_optimizer_step_post_hook(observe_step):
            trainer(block, name, orig_dtype)
        assert steps == config.reconstruction_epochs
        assert any(key.endswith("tlut") for key in changed)
        assert any(key.endswith("SU") for key in changed)
        assert any(key.endswith("SV") for key in changed)
        for norm in ("self_attn.q_norm", "self_attn.k_norm", "input_layernorm"):
            assert f"{norm}.weight" in changed
        if name == "0_v":
            assert "self_attn.q_proj.weight" in changed
        block.cuda()
        assert trainer.validation_loss(block, orig_dtype) <= initial_loss
        completed.append(name)

    grad_enabled = torch.is_grad_enabled()
    try:
        finetune.quantize_finetune_decoder_layer(
            layer,
            quant_order(),
            0,
            codebook,
            config,
            "cuda:0",
            None,
            None,
            reconstruction_callback=reconstruct,
        )
    finally:
        torch.set_grad_enabled(grad_enabled)
    assert completed == [f"0_{short}" for _, short, *_ in PROJECTIONS]
    with torch.no_grad():
        layer.cuda()
        # Compare in FP32 to isolate serialization/decoding from BF16 GEMM rounding.
        hidden = train_replay.inputs[:1].float()
        expected = train_replay.state.forward(layer, hidden)
        dense = Qwen3ForCausalLM(hf).model.layers[0].cuda()
        prefixes = tuple(f"{name}." for name, *_ in PROJECTIONS)
        for name, value in layer.state_dict().items():
            if not name.startswith(prefixes):
                dense.state_dict()[name].copy_(value)
        for name, short, *_ in PROJECTIONS:
            saved = torch.load(config.checkpoints / f"0_{short}.pt", weights_only=True)
            assert saved["SU"].dtype == saved["SV"].dtype == saved["tlut"].dtype == torch.float32
            weight = reconstruct_dense_weight(
                saved, config.recipe(model_family="qwen3", layer_end=1)
            )
            dense.get_parameter(f"{name}.weight").copy_(weight)
        actual = train_replay.state.forward(dense, hidden)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multi_gpu_hessians_full_block_restart_and_hf_export(tiny_qtip, dtype, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"NCCL integration requires {world_size} GPUs")
    single, hf = tiny_qtip
    single = replace(single, model_dtype=str(dtype).removeprefix("torch."))
    model = Qwen3ForCausalLM(hf).to(dtype)
    model.save_pretrained(single.model)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "test": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        single.model
    )
    multiple = replace(
        single,
        devices=list(range(world_size)),
        output_dir=single.output_dir.parent / f"dp{world_size}",
    )
    payload = {
        "protocol": single.calibration_recipe(),
        "hessian": torch.randint(0, 128, (9, 16)),
        "train": torch.randint(0, 128, (9, 16)),
        "validation": torch.randint(0, 128, (4, 16)),
    }
    for config in (single, multiple):
        config.output_dir.mkdir()
        torch.save(payload, config.calibration)
        hessian(config)
        marker = config.hessians / "complete.json"
        stamp = marker.stat().st_mtime_ns
        hessian(config)
        assert marker.stat().st_mtime_ns == stamp
    for index in range(hf.num_hidden_layers):
        for name in ("qkv", "o", "up", "down"):
            one = torch.load(single.hessians / f"{index}_{name}.pt", weights_only=True)
            many = torch.load(multiple.hessians / f"{index}_{name}.pt", weights_only=True)
            assert one["ct"] == many["ct"] == 9 * 16
            torch.testing.assert_close(one["flatH"], many["flatH"], atol=1e-6, rtol=1e-5)
            # Test quantize DP with the SAME published Hessian. Different FP32
            # reduction orders can flip near-tied discrete trellis decisions;
            # Hessian agreement is tested above, independently of that boundary.
            shutil.copy2(
                multiple.hessians / f"{index}_{name}.pt", single.hessians / f"{index}_{name}.pt"
            )
    tlut = torch.randn(512, 2)
    for config in (single, multiple):
        config.checkpoints.mkdir()
        torch.save(tlut, config.checkpoints / "tlut-l16-b9-v2.pt")
        quantize(config)
    stamps = {path: path.stat().st_mtime_ns for path in multiple.checkpoints.iterdir()}
    quantize(multiple)
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())
    saved = {
        path.name: torch.load(path, weights_only=True)
        for path in multiple.checkpoints.glob("1*.pt")
    }
    # The first complete block stays published; the second is rerun from its teacher.
    (multiple.checkpoints / "1.complete").unlink()
    torch.save({"partial": True}, multiple.checkpoints / "1_q.pt")
    quantize(multiple)
    for filename, expected in saved.items():
        actual = torch.load(multiple.checkpoints / filename, weights_only=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for path, stamp in stamps.items():
        if path.name.startswith("0"):
            assert path.stat().st_mtime_ns == stamp
    export(single)
    export(multiple)
    one_model = AutoModelForCausalLM.from_pretrained(single.exported_model, dtype="auto").eval()
    many_model = AutoModelForCausalLM.from_pretrained(multiple.exported_model, dtype="auto").eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            one_model(ids).logits, many_model(ids).logits, atol=1e-5, rtol=1e-5
        )
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        torch.testing.assert_close(
            many_model.get_parameter(name), model.get_parameter(name), atol=0, rtol=0
        )
    for index in range(hf.num_hidden_layers):
        auxiliary = torch.load(multiple.checkpoints / f"{index}.auxiliary.pt", weights_only=True)
        for name, value in auxiliary.items():
            torch.testing.assert_close(
                many_model.get_parameter(f"model.layers.{index}.{name}"),
                value.to(dtype),
                atol=0,
                rtol=0,
            )
        for _, short, *_ in PROJECTIONS:
            one = torch.load(single.checkpoints / f"{index}_{short}.pt", weights_only=True)
            many = torch.load(multiple.checkpoints / f"{index}_{short}.pt", weights_only=True)
            torch.testing.assert_close(one["trellis"], many["trellis"], atol=0, rtol=0)
    recipe = json.loads((multiple.checkpoints / "qtip_config.json").read_text())
    assert recipe["format"] == "qtip-blockwise-full-state-dense-qdq"
