from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from weight_vq.qtip import replay
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.hessian import LayerHessianCollectors
from weight_vq.qtip.replay import LayerReplay, ReconstructionAdapter, ReplayState


class ReplayLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        self.last_kwargs: dict[str, Any] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.last_kwargs = {
            "position_embeddings": position_embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **kwargs,
        }
        return self.projection(hidden_states) * scale


def replay_state(layer_count: int = 1) -> ReplayState:
    mask = torch.tensor([[True, False, True]])
    return ReplayState(
        kwargs_by_layer=[
            [
                {
                    "position_embeddings": (torch.ones(1, 3, 2), torch.zeros(1, 3, 2)),
                    "attention_mask": torch.ones(1, 1, 3, 3),
                    "position_ids": torch.arange(3).unsqueeze(0),
                    "scale": 2.0,
                }
            ]
            for _ in range(layer_count)
        ],
        other_args_by_layer=[[()] for _ in range(layer_count)],
        deepstack_contexts=[
            {
                "valid_token_mask": mask,
                "visual_pos_masks": torch.tensor([[False, True, False]]),
                "deepstack_visual_embeds": [
                    torch.tensor([[3.0, 4.0]]),
                    torch.tensor([[5.0, 6.0]]),
                    torch.tensor([[7.0, 8.0]]),
                ],
            }
        ],
        valid_masks=[mask],
        inputs=[torch.ones(1, 3, 2)],
        retained_kwargs=[{}],
    )


def test_capture_replay_reads_the_monolithic_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: dict[str, Any] = {}

    class Calibration:
        def __init__(
            self, manifest: Path, *, model_path: Path, split: str, batch_size: int, limit: int
        ) -> None:
            received.update(
                manifest=manifest,
                model_path=model_path,
                split=split,
                batch_size=batch_size,
                limit=limit,
            )
            self.records = [{"id": "a"}, {"id": "b"}]

        def __iter__(self):
            yield {"attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]])}

    def capture(
        model: nn.Module, layers: nn.ModuleList, calibration: Any, *, batch_size: None, device: str
    ):
        received.update(model=model, layers=layers, capture_batch_size=batch_size, device=device)
        list(calibration)
        return (
            [torch.ones(2, 3, 2)],
            [[{"position_ids": torch.arange(3).unsqueeze(0)}]],
            [[()]],
            [{}],
            [{}],
        )

    monkeypatch.setattr(replay, "CocoCalibrationBatches", Calibration)
    monkeypatch.setattr(replay, "capture_layer_inputs_and_args", capture)
    config = QTIPConfig(
        model=tmp_path / "model",
        calibration=tmp_path / "calibration.json",
        hessians=tmp_path / "hessians",
        checkpoints=tmp_path / "checkpoints",
        exported_model=tmp_path / "export",
        devices=[0],
        train_samples=2,
        batch_size=2,
    )
    model = nn.Linear(2, 2)
    layers = nn.ModuleList([ReplayLayer()])

    state = replay.capture_replay(
        model, layers, config, family="qwen3_vl", split="train", device="cuda:3"
    )

    assert received == {
        "manifest": config.calibration,
        "model_path": config.model,
        "split": "train",
        "batch_size": 2,
        "limit": 2,
        "model": model,
        "layers": layers,
        "capture_batch_size": None,
        "device": "cuda:3",
    }
    assert state.valid_masks[0].tolist() == [[True, True, False], [True, False, False]]
    assert state.inputs[0].shape == (2, 3, 2)


def test_token_capture_replays_real_qwen3_layers(tmp_path: Path) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    calibration = tmp_path / "calibration.pt"
    torch.save(
        {"train": torch.randint(0, 64, (4, 8)), "validation": torch.randint(0, 64, (2, 8))},
        calibration,
    )
    config = QTIPConfig(
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
    model = Qwen3ForCausalLM(
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

    state = replay.capture_replay(
        model, model.model.layers, config, family="qwen3", split="train", device="cpu"
    )

    assert len(state.inputs) == 2
    assert all(mask.all() for mask in state.valid_masks)
    assert state.deepstack_contexts == [{}, {}]
    for index, layer in enumerate(model.model.layers):
        collectors = LayerHessianCollectors(layer) if index == 0 else None
        snapshot = state.forward_original(layer, index, device="cpu", collectors=collectors)
        assert len(snapshot) == 2
        if collectors is not None:
            assert all(payload["ct"] == 32 for payload in collectors.close().values())


def test_replay_state_forwards_kwargs_collects_mask_and_advances_deepstack() -> None:
    class MaskRecorder:
        def __init__(self) -> None:
            self.masks: list[torch.Tensor] = []

        def set_position_mask(self, mask: torch.Tensor) -> None:
            self.masks.append(mask)

    layer = ReplayLayer()
    with torch.no_grad():
        layer.projection.weight.copy_(torch.eye(2))
    state = replay_state()
    collectors = MaskRecorder()

    snapshot = state.forward_original(layer, 0, device="cpu", collectors=collectors)

    assert collectors.masks[0].tolist() == [[True, False, True]]
    assert snapshot.inputs[0][0, 0, 0].item() == 1
    assert snapshot.targets[0][0, 0, 0].item() == 2
    assert state.inputs[0][0, 1].tolist() == [5, 6]
    assert layer.last_kwargs is not None
    assert layer.last_kwargs["position_embeddings"][0].shape == (1, 3, 2)
    assert layer.last_kwargs["position_ids"].shape == (1, 3)


def test_layer_replay_masks_padding_in_loss() -> None:
    layer = ReplayLayer()
    with torch.no_grad():
        layer.projection.weight.copy_(torch.eye(2))
    state = replay_state()
    target = state.inputs[0] * 2
    target[:, 1, :] = -1000
    snapshot = LayerReplay(
        state=state,
        layer_index=0,
        inputs=state.inputs,
        targets=[target],
        retained_kwargs=state.retained_kwargs,
    )

    assert snapshot.loss(layer, 0, "cpu", mask_padding=True).item() == 0
    assert snapshot.mean_loss(layer, "cpu", mask_padding=True) == 0


def test_reconstruction_adapter_delegates_two_layer_snapshots() -> None:
    layer = ReplayLayer()
    with torch.no_grad():
        layer.projection.weight.copy_(torch.eye(2))
    state = replay_state()
    target = state.inputs[0] * 2
    snapshot = LayerReplay(
        state=state,
        layer_index=0,
        inputs=state.inputs,
        targets=[target],
        retained_kwargs=state.retained_kwargs,
    )
    adapter = ReconstructionAdapter(train=snapshot, validation=snapshot, seed=4, mask_padding=True)

    assert list(adapter.iter_train(0)) == [0]
    assert adapter.loss(layer, 0, "cpu").item() == 0
    assert adapter.validation_loss(layer, "cpu") == 0
    adapter.materialize(layer, "cpu")


def test_layer_replay_accepts_captured_retained_kwargs() -> None:
    state = replay_state()
    state.retained_kwargs = [{"shared_kv_states": torch.ones(1, 3, 2)}]
    snapshot = LayerReplay(
        state=state,
        layer_index=0,
        inputs=state.inputs,
        targets=state.inputs,
        retained_kwargs=state.retained_kwargs,
    )

    assert "shared_kv_states" in snapshot.retained_kwargs[0]


def test_replay_state_rejects_misaligned_parallel_state() -> None:
    state = replay_state()
    state.valid_masks = []
    with torch.no_grad(), pytest.raises(ValueError, match="replay batch mismatch"):
        state.forward_original(ReplayLayer(), 0, device="cpu")
