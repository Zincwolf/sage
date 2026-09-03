from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.hessian import LayerHessianCollectors
from weight_vq.qtip.replay import BatchContext, LayerReplay, ReplayState, capture_replay


def tiny_qwen() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()


def config(tmp_path: Path, calibration: Path) -> QTIPConfig:
    return QTIPConfig(
        model=tmp_path / "model",
        calibration=calibration,
        hessians=tmp_path / "hessians",
        checkpoints=tmp_path / "checkpoints",
        exported_model=tmp_path / "export",
        devices=[0],
        train_samples=4,
        validation_samples=2,
        batch_size=2,
    )


def test_compact_replay_matches_native_qwen_decoder(tmp_path: Path) -> None:
    tokens = torch.randint(0, 64, (4, 8))
    calibration = tmp_path / "calibration.pt"
    torch.save({"train": tokens, "validation": tokens[:2]}, calibration)
    model = tiny_qwen()

    expected = [
        model.model(input_ids=batch, use_cache=False).last_hidden_state for batch in tokens.split(2)
    ]
    state = capture_replay(
        model, config(tmp_path, calibration), family="qwen3", split="train", device="cpu"
    )

    assert len(state.inputs) == 2
    assert all(context.attention_mask is None for context in state.contexts)
    for index, layer in enumerate(model.model.layers):
        collectors = LayerHessianCollectors(layer) if index == 0 else None
        state.forward_original(layer, index, device="cpu", collectors=collectors)
        if collectors is not None:
            assert all(payload["ct"] == 32 for payload in collectors.close().values())

    for actual, native in zip(state.inputs, expected, strict=True):
        torch.testing.assert_close(model.model.norm(actual), native)


def test_rank_local_capture_slices_each_global_batch(tmp_path: Path) -> None:
    tokens = torch.arange(4 * 8).reshape(4, 8) % 64
    calibration = tmp_path / "calibration.pt"
    torch.save({"train": tokens, "validation": tokens[:2]}, calibration)
    model = tiny_qwen()

    state = capture_replay(
        model,
        config(tmp_path, calibration),
        family="qwen3",
        split="train",
        device="cpu",
        rank=1,
        world_size=2,
    )

    expected = model.model.embed_tokens(tokens[[1, 3]])
    torch.testing.assert_close(torch.cat(state.inputs), expected)


def test_reconstruction_masks_padding_and_deepstack() -> None:
    model = tiny_qwen()
    layer = model.model.layers[0]
    valid_mask = torch.tensor([[True, False, True]])
    context = BatchContext(
        attention_mask=valid_mask,
        position_ids=None,
        valid_mask=valid_mask,
        visual_pos_mask=torch.tensor([[False, True, False]]),
        deepstack_visual_embeds=[torch.ones(1, 16)],
    )
    source = torch.randn(1, 3, 16)
    state = ReplayState("qwen3", model.model.config, model.model.rotary_emb, [source], [context])
    with torch.no_grad():
        target = state._forward(layer, source, 0, "cpu")
    corrupted = target.clone()
    corrupted[:, 1] = 1000
    replay = LayerReplay(state, 0, [source], [corrupted])

    assert replay.loss(layer, 0, "cpu", mask_padding=True).item() == 0
    advanced = state._apply_deepstack(target, context, 0)
    torch.testing.assert_close(advanced[:, 0], target[:, 0])
    torch.testing.assert_close(advanced[:, 1], target[:, 1] + 1)
    assert context.deepstack_visual_embeds is None


def test_reconstruction_loss_keeps_gradients() -> None:
    model = tiny_qwen()
    layer: nn.Module = model.model.layers[0]
    context = BatchContext(None, None, torch.ones(1, 3, dtype=torch.bool))
    source = torch.randn(1, 3, 16)
    state = ReplayState("qwen3", model.model.config, model.model.rotary_emb, [source], [context])
    with torch.no_grad():
        target = state._forward(layer, source, 0, "cpu")
    replay = LayerReplay(state, 0, [source], [target])

    replay.loss(layer, 0, "cpu", mask_padding=True).backward()

    assert any(parameter.grad is not None for parameter in layer.parameters())
