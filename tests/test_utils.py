import csv
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from weight_vq.utils import validate_dense_model


def test_validate_loads_standard_hf_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
        )
    )
    model.save_pretrained(tmp_path)
    loaded = validate_dense_model(
        tmp_path,
        model_loader=AutoModelForCausalLM,
        device="cpu",
        weights=lambda model: [("lm_head", model.get_parameter("lm_head.weight"))],
    )
    torch.testing.assert_close(loaded.lm_head.weight, model.lm_head.weight)
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert rows[0] == ["layer", "p01", "p50", "p99", "abs_p99", "abs_p999", "abs_max"]
    assert rows[1][0] == "lm_head"


@pytest.mark.parametrize("large", [False, True])
def test_validate_statistics_and_exact_max(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], large: bool
) -> None:
    values = torch.tensor([-4.0, -1.0, 0.0, 2.0, 3.0])
    if large:
        values = torch.zeros(1_000_002)
        values[1] = -100.0  # Outside the regularly spaced quantile sample.
    model = nn.Linear(values.numel(), 1, bias=False)
    model.weight.data.copy_(values)

    class Loader:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> nn.Module:
            assert path == tmp_path
            assert kwargs["device_map"] == "cpu"
            return model

    validate_dense_model(
        tmp_path,
        model_loader=Loader,
        device="cpu",
        weights=lambda model: [("weight", model.get_parameter("weight"))],
    )
    row = list(csv.reader(capsys.readouterr().out.splitlines()))[1]
    sample = values[::2] if large else values
    expected = [
        *torch.quantile(sample, torch.tensor([0.01, 0.5, 0.99])).tolist(),
        *torch.quantile(sample.abs(), torch.tensor([0.99, 0.999])).tolist(),
        values.abs().max().item(),
    ]
    assert [float(value) for value in row[1:]] == pytest.approx(expected, rel=1e-5)


def test_validate_rejects_nonfinite_selected_weight(tmp_path: Path) -> None:
    class Loader:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> nn.Module:
            return nn.Identity()

    with pytest.raises(ValueError, match="non-finite weight: projection"):
        validate_dense_model(
            tmp_path,
            model_loader=Loader,
            device="cpu",
            weights=lambda _model: [("projection", torch.tensor([float("nan")]))],
        )
