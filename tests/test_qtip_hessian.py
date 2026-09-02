import pytest
import torch
from torch import nn

from weight_vq.qtip.hessian import LayerHessianCollectors, MaskedHessianCollector, collector_modules


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.k_proj = nn.Linear(2, 2, bias=False)
        self.v_proj = nn.Linear(2, 2, bias=False)
        self.o_proj = nn.Linear(2, 2, bias=False)


class TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False)
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMlp()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention = self.self_attn.q_proj(hidden_states)
        attention = self.self_attn.o_proj(attention)
        mlp = self.mlp.up_proj(hidden_states)
        return self.mlp.down_proj(mlp) + attention


def test_masked_hessian_excludes_padding_rows() -> None:
    linear = nn.Linear(2, 2, bias=False)
    collector = MaskedHessianCollector(linear)
    values = torch.tensor([[[1.0, 2.0], [100.0, 100.0], [3.0, 4.0]]])
    collector.set_position_mask(torch.tensor([[True, False, True]]))
    linear(values)
    result = collector.close()

    selected = values[:, [0, 2], :].reshape(-1, 2)
    expected = selected.T @ selected / selected.shape[0]
    rows, columns = torch.tril_indices(2, 2)
    torch.testing.assert_close(result["flatH"], expected[rows, columns])
    assert result["n"] == 2
    assert result["ct"] == 2
    assert result["position_mask"] == "attention_mask"


def test_unmasked_hessian_counts_every_position() -> None:
    linear = nn.Linear(2, 2, bias=False)
    collector = MaskedHessianCollector(linear, include_padding=True)
    collector.set_position_mask(torch.tensor([[True, False, True]]))
    linear(torch.randn(1, 3, 2))
    result = collector.close()

    assert result["ct"] == 3
    assert result["position_mask"] == "all"


def test_layer_collectors_use_qtip_hessian_groups() -> None:
    layer = TinyLayer()
    assert set(collector_modules(layer)) == {"qkv", "o", "up", "down"}

    collectors = LayerHessianCollectors(layer)
    collectors.set_position_mask(torch.tensor([[True, False, True]]))
    layer(torch.randn(1, 3, 2))
    result = collectors.close()

    assert set(result) == {"qkv", "o", "up", "down"}
    assert all(payload["ct"] == 2 for payload in result.values())
    assert all(
        isinstance(payload["flatH"], torch.Tensor) and payload["flatH"].shape == (3,)
        for payload in result.values()
    )


def test_mask_is_required_before_projection_forward() -> None:
    linear = nn.Linear(2, 2, bias=False)
    collector = MaskedHessianCollector(linear)
    with pytest.raises(RuntimeError, match="Position mask must be set"):
        linear(torch.randn(1, 2, 2))
    collector._handle.remove()
