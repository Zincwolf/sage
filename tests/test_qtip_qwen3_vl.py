from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from weight_vq.qtip.artifacts import (
    HESSIAN_NAMES,
    PROJECTIONS,
    atomic_write_completion,
    hessian_complete,
    layer_complete,
    projection_paths,
    quant_order,
)
from weight_vq.qtip.qwen3_vl import (
    audit_qwen3_vl,
    get_layers,
    inject_dense_checkpoints,
    projection_state_key,
)


class TinyAttention(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class TinyMlp(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)


class TinyLayer(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.self_attn = TinyAttention(width)
        self.mlp = TinyMlp(width)


class TinyLanguageModel(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TinyLayer() for _ in range(layer_count))


class TinyQwen3VLCore(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.language_model = TinyLanguageModel(layer_count)


class TinyQwen3VL(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.model = TinyQwen3VLCore(layer_count)
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            text_config=SimpleNamespace(num_hidden_layers=layer_count),
            vision_config=SimpleNamespace(deepstack_visual_indexes=[8, 16, 24]),
        )


def test_architecture_contract_covers_seven_language_projections() -> None:
    model = TinyQwen3VL(2)
    report = audit_qwen3_vl(model, expected_layers=2)

    assert get_layers(model) is model.model.language_model.layers
    assert tuple(short for _attribute, short, _hessian, _parallel in PROJECTIONS) == (
        "v",
        "q",
        "k",
        "o",
        "up",
        "gate",
        "down",
    )
    assert HESSIAN_NAMES == ("qkv", "o", "up", "down")
    assert report["projection_count"] == 14
    assert report["deepstack_language_injection_layers"] == [0, 1, 2]
    assert projection_state_key(3, "self_attn.q_proj") == (
        "model.language_model.layers.3.self_attn.q_proj.weight"
    )


def test_quant_order_matches_qtip_finetune_contract() -> None:
    assert quant_order() == [
        ("self_attn.v_proj", "v", "qkv", "v", "col"),
        ("self_attn.q_proj", "q", "qkv", "q", "col"),
        ("self_attn.k_proj", "k", "qkv", "k", "col"),
        ("self_attn.o_proj", "o", "o", "o", "row"),
        ("mlp.up_proj", "up", "up", "up", "col"),
        ("mlp.gate_proj", "gate", "up", "gate", "col"),
        ("mlp.down_proj", "down", "down", "down", "row"),
    ]


def test_flat_checkpoint_names_require_marker_and_all_projections(tmp_path: Path) -> None:
    expected = tuple(
        tmp_path / f"4_{short}.pt" for _attribute, short, _hessian, _parallel in PROJECTIONS
    )
    assert projection_paths(4, tmp_path) == expected
    assert not layer_complete(4, tmp_path)

    for path in expected:
        path.touch()
    assert not layer_complete(4, tmp_path)

    atomic_write_completion(4, tmp_path)
    assert layer_complete(4, tmp_path)

    expected[0].unlink()
    assert not layer_complete(4, tmp_path)


def test_completion_marker_requires_all_projection_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing QTIP checkpoints"):
        atomic_write_completion(0, tmp_path)
    assert not (tmp_path / "0.complete").exists()


def test_dense_injection_targets_all_seven_projections(tmp_path: Path) -> None:
    model = TinyQwen3VL(1)
    for _attribute, short, _hessian, _parallel in PROJECTIONS:
        torch.save({"dense": torch.full((16, 16), 0.25)}, tmp_path / f"0_{short}.pt")
    atomic_write_completion(0, tmp_path)
    modified = inject_dense_checkpoints(
        model,
        tmp_path,
        {},
        layer_start=0,
        layer_end=1,
        reconstruct=lambda saved, _config: saved["dense"],
    )

    assert len(modified) == 7
    for attribute, _short, _hessian, _parallel in PROJECTIONS:
        module: Any = model.model.language_model.layers[0]
        for component in attribute.split("."):
            module = getattr(module, component)
        torch.testing.assert_close(module.weight, torch.full_like(module.weight, 0.25))


def test_flat_hessian_names_are_layer_scoped(tmp_path: Path) -> None:
    assert not hessian_complete(5, tmp_path)
    for name in HESSIAN_NAMES:
        (tmp_path / f"5_{name}.pt").touch()
    assert hessian_complete(5, tmp_path)


def test_audit_rejects_non_linear_target_projection() -> None:
    model = TinyQwen3VL(1)
    layer = model.model.language_model.layers[0]
    assert isinstance(layer, TinyLayer)
    mlp: Any = layer.mlp
    mlp.down_proj = nn.Identity()
    with pytest.raises(TypeError, match=r"mlp\.down_proj"):
        audit_qwen3_vl(model, expected_layers=1)


def test_real_transformers_qwen3vl_layer_matches_projection_contract() -> None:
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    layer = Qwen3VLTextDecoderLayer(config, layer_idx=0)

    for attribute, _short, _hessian, _parallel in PROJECTIONS:
        module = layer
        for component in attribute.split("."):
            module = getattr(module, component)
        assert isinstance(module, torch.nn.Linear)
