import csv
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from weight_vq.qtip.artifacts import layer_complete
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.pipeline import _get_layers, _initialize_run, _model_contract, export, validate


def config(tmp_path: Path) -> QTIPConfig:
    return QTIPConfig(
        model=Path("/models/Qwen3-VL-8B-Instruct"),
        calibration=Path("/data/coco.manifest.json"),
        output_dir=tmp_path,
        devices=[2, 5, 7, 9],
        train_samples=16,
        validation_samples=8,
        batch_size=4,
        reconstruction_epochs=1,
        layer_start=3,
        layer_end=10,
    )


def test_initialize_run_uses_run_local_tlut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[dict[str, object]] = []
    qtip_config = config(tmp_path)
    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.bitshift_codebook", lambda **kwargs: received.append(kwargs)
    )

    _initialize_run(qtip_config, "qwen3_vl", 10)

    assert received[0]["tlut_path"] == qtip_config.checkpoints / "tlut-l16-b9-v2.pt"


def test_model_family_selects_configured_decoder_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = {
        "text": SimpleNamespace(model_type="qwen3", num_hidden_layers=36),
        "vlm": SimpleNamespace(
            model_type="qwen3_vl", text_config=SimpleNamespace(num_hidden_layers=40)
        ),
    }
    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.AutoConfig.from_pretrained",
        lambda path, **_kwargs: configs[str(path)],
    )
    text_layers = nn.ModuleList([nn.Identity()])
    vlm_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    text_model = SimpleNamespace(model=SimpleNamespace(layers=text_layers))
    vlm_model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=vlm_layers))
    )

    assert _model_contract(Path("text")) == ("qwen3", 36)
    assert _model_contract(Path("vlm")) == ("qwen3_vl", 40)
    assert _get_layers(text_model, "qwen3") is text_layers  # type: ignore[arg-type]
    assert _get_layers(vlm_model, "qwen3_vl") is vlm_layers  # type: ignore[arg-type]


def test_qtip_config_is_the_verified_native_recipe() -> None:
    qtip_config = QTIPConfig(
        model=Path("/model"),
        calibration=Path("/calibration"),
        output_dir=Path("/output"),
        devices=[0],
        seed=17,
    )

    assert (qtip_config.L, qtip_config.K, qtip_config.V, qtip_config.tlut_bits) == (16, 2, 2, 9)
    assert (qtip_config.td_x, qtip_config.td_y, qtip_config.decode_mode) == (16, 16, "quantlut_sym")
    assert qtip_config.packed_inference_kernel is False
    assert qtip_config.split_for_tp is False
    assert qtip_config.learning_rate == 3e-6
    recipe = qtip_config.recipe(model_family="qwen3_vl", layer_end=36)
    assert recipe["schema_version"] == 3
    assert recipe["seed"] == 17
    assert recipe["format"] == "qtip-bitshift-trellis-qwen3-vl-dense-qdq"
    assert "output_dir" not in recipe
    assert "checkpoints" not in recipe
    assert "devices" not in recipe
    assert "batch_size" not in recipe
    assert "replay_device" not in recipe


class ExportAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(1, 1, bias=False)
        self.k_proj = nn.Linear(1, 1, bias=False)
        self.v_proj = nn.Linear(1, 1, bias=False)
        self.o_proj = nn.Linear(1, 1, bias=False)


class ExportMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(1, 1, bias=False)
        self.up_proj = nn.Linear(1, 1, bias=False)
        self.down_proj = nn.Linear(1, 1, bias=False)


class ExportLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = ExportAttention()
        self.mlp = ExportMlp()


class FakeModel:
    def __init__(self, family: str) -> None:
        layers = nn.ModuleList(ExportLayer() for _ in range(10))
        self.model = (
            SimpleNamespace(layers=layers)
            if family == "qwen3"
            else SimpleNamespace(language_model=SimpleNamespace(layers=layers))
        )

    def eval(self) -> "FakeModel":
        return self

    def save_pretrained(self, path: Path, **_kwargs: Any) -> None:
        (path / "model.safetensors").touch()


class FakeAssets:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def save_pretrained(self, path: Path) -> None:
        (path / self.filename).touch()


@pytest.mark.parametrize(
    ("family", "model_api", "assets_api", "assets_file"),
    [
        ("qwen3", "AutoModelForCausalLM", "AutoTokenizer", "tokenizer_config.json"),
        ("qwen3_vl", "AutoModelForImageTextToText", "AutoProcessor", "processor_config.json"),
    ],
)
def test_export_uses_family_hf_apis_and_injects_dense_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    family: str,
    model_api: str,
    assets_api: str,
    assets_file: str,
) -> None:
    qtip_config = config(tmp_path)
    qtip_config.checkpoints.mkdir()
    for index in range(3, 10):
        (qtip_config.checkpoints / f"{index}.complete").touch()
        for name in ("v", "q", "k", "o", "up", "gate", "down"):
            (qtip_config.checkpoints / f"{index}_{name}.pt").touch()
    (qtip_config.checkpoints / "qtip_config.json").write_text("{}")
    model = FakeModel(family)
    model_loads: list[dict[str, Any]] = []
    assets_loads: list[dict[str, Any]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(_path: Path, **kwargs: Any) -> FakeModel:
            model_loads.append(kwargs)
            return model

    class AutoAssets:
        @staticmethod
        def from_pretrained(_path: Path, **kwargs: Any) -> FakeAssets:
            assets_loads.append(kwargs)
            return FakeAssets(assets_file)

    monkeypatch.setattr("weight_vq.qtip.pipeline._model_contract", lambda _model: (family, 10))
    monkeypatch.setattr(f"weight_vq.qtip.pipeline.{model_api}", AutoModel)
    monkeypatch.setattr(f"weight_vq.qtip.pipeline.{assets_api}", AutoAssets)
    monkeypatch.setattr("weight_vq.qtip.pipeline.torch.load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.reconstruct_dense_weight", lambda _saved, _recipe: torch.ones(1, 1)
    )

    export(qtip_config)

    assert model_loads[0]["device_map"] == "cpu"
    assert assets_loads[0]["local_files_only"] is True
    assert (qtip_config.exported_model / "model.safetensors").is_file()
    assert (qtip_config.exported_model / assets_file).is_file()
    layers = _get_layers(model, family)  # type: ignore[arg-type]
    layer = cast(ExportLayer, layers[3])
    torch.testing.assert_close(layer.self_attn.q_proj.weight, torch.ones(1, 1))


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeWeight:
    device = "cuda:0"

    def detach(self) -> "FakeWeight":
        return self

    def flatten(self) -> "FakeWeight":
        return self

    def float(self) -> "FakeWeight":
        return self

    def numel(self) -> int:
        return 4

    def __getitem__(self, _key: object) -> "FakeWeight":
        return self

    def abs(self) -> "FakeWeight":
        return self

    def max(self) -> FakeScalar:
        return FakeScalar(2.0)


class Projection:
    weight = FakeWeight()


class Layer:
    self_attn = SimpleNamespace(
        v_proj=Projection(), q_proj=Projection(), k_proj=Projection(), o_proj=Projection()
    )
    mlp = SimpleNamespace(up_proj=Projection(), gate_proj=Projection(), down_proj=Projection())


@pytest.mark.parametrize(
    ("family", "model_api", "layer_prefix"),
    [
        ("qwen3", "AutoModelForCausalLM", "model.layers"),
        ("qwen3_vl", "AutoModelForImageTextToText", "model.language_model.layers"),
    ],
)
def test_validate_uses_family_model_and_prints_target_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    family: str,
    model_api: str,
    layer_prefix: str,
) -> None:
    qtip_config = config(tmp_path)
    loads: list[dict[str, Any]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(_path: Path, **kwargs: Any) -> object:
            loads.append(kwargs)
            return object()

    fake_torch = ModuleType("torch")
    fake_torch.no_grad = nullcontext  # type: ignore[attr-defined]
    fake_torch.isfinite = lambda _value: SimpleNamespace(all=lambda: True)  # type: ignore[attr-defined]
    fake_torch.tensor = lambda values, **_kwargs: values  # type: ignore[attr-defined]
    fake_torch.quantile = lambda _sample, values: [  # type: ignore[attr-defined]
        FakeScalar(value) for value in values
    ]
    monkeypatch.setattr("weight_vq.qtip.pipeline.torch", fake_torch)
    monkeypatch.setattr("weight_vq.qtip.pipeline._model_contract", lambda _model: (family, 36))
    monkeypatch.setattr(f"weight_vq.qtip.pipeline.{model_api}", AutoModel)
    monkeypatch.setattr(
        "weight_vq.qtip.pipeline._get_layers", lambda _model, _family: [Layer() for _ in range(36)]
    )

    validate(qtip_config)

    assert loads[0]["device_map"] == 0
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert len(rows) == 1 + 7 * 7
    assert rows[1][0] == f"{layer_prefix}.3.self_attn.v_proj"


def test_layer_complete_requires_marker_and_all_seven_checkpoints(tmp_path: Path) -> None:
    for name in ("v", "q", "k", "o", "up", "gate", "down"):
        (tmp_path / f"0_{name}.pt").touch()
    assert not layer_complete(0, tmp_path)
    (tmp_path / "0.complete").touch()
    assert layer_complete(0, tmp_path)
