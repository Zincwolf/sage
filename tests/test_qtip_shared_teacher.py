from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.multiprocessing.spawn import spawn
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)


def _shared_teacher_worker(rank: int, model: Any, family: str, output: str, device: str) -> None:
    torch.set_num_threads(1)
    assert all(parameter.is_shared() for parameter in model.parameters())
    backbone = model.model if family == "qwen3" else model.model.language_model
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
        vision_output = None
        if family == "qwen3_vl":
            vision_output = model.model.visual(
                torch.ones(4, 24), grid_thw=torch.tensor([[1, 2, 2]])
            )[0]
        # CPU-only runs detach via dtype conversion; CUDA runs use the pipeline's transfer.
        layer = backbone.layers[0].to(
            device=device, dtype=torch.float64 if device == "cpu" else torch.float32
        )
        if device == "cpu":
            assert all(not parameter.is_shared() for parameter in layer.parameters())
        else:
            assert all(parameter.is_cuda for parameter in layer.parameters())
        for parameter in layer.parameters():
            parameter.add_(rank + 1)
        backbone.layers[0] = nn.Identity()
    torch.save({"logits": logits, "vision": vision_output}, Path(output, f"{rank}.pt"))


@pytest.mark.parametrize("family", ["qwen3", "qwen3_vl"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spawn_shared_teacher_preserves_parent(tmp_path: Path, family: str, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    text: dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    if family == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**text))
    else:
        model = Qwen3VLForConditionalGeneration(
            Qwen3VLConfig(
                text_config={
                    **text,
                    "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
                },
                vision_config=dict(
                    depth=2,
                    hidden_size=16,
                    intermediate_size=32,
                    num_heads=2,
                    out_hidden_size=16,
                    patch_size=2,
                    temporal_patch_size=2,
                    spatial_merge_size=2,
                    num_position_embeddings=4,
                    deepstack_visual_indexes=[0],
                ),
            )
        )
    model.eval().requires_grad_(False).share_memory()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    expected = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits

    spawn(_shared_teacher_worker, args=(model, family, str(tmp_path), device), nprocs=2, join=True)

    for rank in range(2):
        result = torch.load(tmp_path / f"{rank}.pt", weights_only=True)
        torch.testing.assert_close(result["logits"], expected)
        if family == "qwen3_vl":
            assert result["vision"].ndim == 2
            assert result["vision"].shape[-1] == 16
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
