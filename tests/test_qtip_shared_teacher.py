"""Shared CPU teachers with private replay/work modules; no downloads or NCCL."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import spawn
from transformers import (
    Ministral3Config,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PixtralVisionConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from weight_vq.blockwise.models import language_backbone
from weight_vq.blockwise.pipeline import _run_rank, model_loader
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.config import QTIPConfig
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.unisvq.config import UniSVQConfig


def _worker(rank, model, tokens, output, device, barrier):
    torch.set_num_threads(1)
    assert all(p.device.type == "cpu" and p.is_shared() for p in model.parameters())
    assert all(b.is_shared() for b in model.buffers())
    original = {k: v.clone() for k, v in model.state_dict().items() if k != "sharing_probe"}
    backbone = language_backbone(model)
    outputs = []
    # Two captures exercise repeated Hessian chunks as well as train/valid replay.
    for chunk in tokens.split(2):
        state = capture_replay(model, chunk, device=device, microbatch_size=1, placement="cpu")
        assert state.rotary_emb is not backbone.rotary_emb
        for teacher_layer in backbone.layers:
            layer = deepcopy(teacher_layer)
            assert all(not p.is_shared() for p in layer.parameters())
            layer.to(device).requires_grad_(True)
            replay = state.forward_original(layer, device=device)
            # The same work module becomes the student, without touching teacher.
            with torch.no_grad():
                layer.self_attn.q_proj.weight.add_(0.01 * (rank + 1))
            optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
            replay.loss_parts(layer, torch.arange(len(chunk)), device)[0].backward()
            optimizer.step()
            layer.cpu()
            del layer, optimizer, replay
        with torch.no_grad():
            outputs.append(backbone.norm(state.inputs))
            state.rotary_emb.get_buffer("inv_freq").zero_()
    torch.save(torch.cat(outputs), Path(output) / f"{rank}.pt")
    # Test-owned writable buffer: cross-worker and parent visibility verifies
    # spawn uses shared backing storage, rather than independently shared copies.
    model.sharing_probe[rank] = rank + 1
    barrier.wait()
    torch.testing.assert_close(model.sharing_probe, torch.tensor([1.0, 2.0]))
    for name, value in original.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert all(p.device.type == "cpu" and p.is_shared() for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize("family", ["qwen3", "mistral3"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
        ),
    ],
)
def test_spawn_shares_teacher_and_isolates_work_modules(tmp_path, family, device):
    torch.set_num_threads(1)
    text: dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    if family == "qwen3":
        source = Qwen3ForCausalLM(Qwen3Config(**text, tie_word_embeddings=True))
    else:
        source = Mistral3ForConditionalGeneration(
            Mistral3Config(
                text_config=Ministral3Config(**text, sliding_window=None),
                vision_config=PixtralVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=16,
                    patch_size=4,
                ),
            )
        )
    source.save_pretrained(tmp_path / "model")
    model = (
        model_loader(family)
        .from_pretrained(
            tmp_path / "model",
            device_map="cpu",
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
    )
    original = {name: value.clone() for name, value in model.state_dict().items()}
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]])
    with torch.no_grad():
        expected = language_backbone(model)(tokens, use_cache=False).last_hidden_state
    model.register_buffer("sharing_probe", torch.zeros(2))
    model.share_memory()
    barrier = mp.get_context("spawn").Barrier(2)
    # GPU tests use two independent processes on GPU0, without a communicator.
    spawn(_worker, args=(model, tokens, str(tmp_path), device, barrier), nprocs=2, join=True)
    torch.testing.assert_close(model.sharing_probe, torch.tensor([1.0, 2.0]))
    for rank in range(2):
        actual = torch.load(tmp_path / f"{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    for name, value in original.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    if family == "qwen3":
        assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real quantizers require CUDA")
@pytest.mark.parametrize("config_type", [QTIPConfig, QuIPSharpConfig, UniSVQConfig])
def test_real_rank_keeps_shared_teacher_unchanged(tmp_path, config_type):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    config = config_type(
        model=tmp_path / "model",
        dataset=tmp_path / "rp",
        output_dir=tmp_path / "run",
        devices=[0],
        model_dtype="bfloat16",
        num_cpu_threads=2,
        hessian_samples=5,
        hessian_sequence_length=8,
        hessian_chunk_size=2,
        train_samples=4,
        validation_samples=2,
        sequence_length=8,
        reconstruction_epochs=1,
        batch_size=4,
    )
    if isinstance(config, (QuIPSharpConfig, UniSVQConfig)):
        config = replace(config, quip_tune_iters=1)
    if isinstance(config, UniSVQConfig):
        config = replace(config, hessian_dtype="float64")
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
            )
        )
        .bfloat16()
        .eval()
        .requires_grad_(False)
    )
    model.config._attn_implementation = "sdpa"
    model.share_memory()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    config.hessians.mkdir(parents=True)
    config.checkpoints.mkdir()
    if isinstance(config, QTIPConfig):
        # Avoid the unrelated million-point LUT initialization.
        torch.save(torch.randn(512, 2), config.checkpoints / "tlut-l16-b9-v2.pt")
    torch.save(
        {
            "protocol": config.calibration_recipe(),
            "hessian": torch.randint(0, 128, (5, 8)),
            "train": torch.randint(0, 128, (4, 8)),
            "validation": torch.randint(0, 128, (2, 8)),
        },
        config.calibration,
    )
    # Run the actual worker against the parent's shared model: private layer
    # replacement/training must leave both weights and nonpersistent buffers intact.
    statistics_before = {}
    grad_enabled = torch.is_grad_enabled()
    try:
        for stage in ("hessian", "quantize", "hessian"):
            _run_rank(0, config, stage, model, 0, 2, tmp_path / "store", config.hessians)
            if stage == "hessian":
                statistics = {
                    path.name: {
                        key: value
                        for key, value in torch.load(path, weights_only=True).items()
                        if key != "position_mask"
                    }
                    for path in config.hessians.glob("*.pt")
                }
                if statistics_before:
                    torch.testing.assert_close(statistics, statistics_before, rtol=0, atol=0)
                else:
                    statistics_before = statistics
            for name, value in original.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            for name, value in buffers.items():
                torch.testing.assert_close(model.get_buffer(name), value, rtol=0, atol=0)
            assert all(p.device.type == "cpu" and p.is_shared() for p in model.parameters())
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    finally:
        torch.set_grad_enabled(grad_enabled)
    assert (config.checkpoints / "0.complete").exists()
    assert (config.checkpoints / "1.complete").exists()
