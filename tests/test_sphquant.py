import csv
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from weight_vq.config import CalibrationMetadata
from weight_vq.sphquant.config import SPHQuantConfig
from weight_vq.sphquant.pipeline import export, optimize, validate


def smoke_config() -> SPHQuantConfig:
    return SPHQuantConfig(
        model=Path("/models/Qwen3-4B-Instruct-2507"),
        calibration=Path("outputs/calibration.pt"),
        checkpoints=Path("outputs/sphquant/one-layer"),
        exported_model=Path("outputs/sphquant/hf-one-layer"),
        sample_size=512,
        kmeans_iterations=2,
        epochs=0,
        cache_shards=2,
        max_layers=1,
        calibration_limit=32,
    )


def test_optimize_maps_config_and_runs_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed: list[dict[str, object]] = []
    reference_parse = object()
    fake_module = ModuleType("paroquant.cli.optimize")
    fake_module.Config = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fake_module.simple_parsing = SimpleNamespace(parse=reference_parse)  # type: ignore[attr-defined]
    fake_module.main = lambda: parsed.append(fake_module.simple_parsing.parse())  # type: ignore[attr-defined]
    fake_paroquant = ModuleType("paroquant")
    fake_cli = ModuleType("paroquant.cli")
    fake_cli.optimize = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paroquant", fake_paroquant)
    monkeypatch.setitem(sys.modules, "paroquant.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "paroquant.cli.optimize", fake_module)
    monkeypatch.setattr(
        "weight_vq.sphquant.pipeline.load_calibration_metadata",
        lambda _path: CalibrationMetadata(1920, 128, 2048),
    )
    optimize(smoke_config())

    assert parsed[0]["method"] == "sphquant"
    assert parsed[0]["params"] == ["radius:0.001,codebook:0.001"]
    assert parsed[0]["train_size"] == 1920
    assert parsed[0]["validation_size"] == 128
    assert parsed[0]["seqlen"] == 2048
    assert parsed[0]["sph_sample_size"] == 512
    assert parsed[0]["sph_kmeans_iterations"] == 2
    assert fake_module.simple_parsing.parse is reference_parse


def test_export_runs_reference_pseudo_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str]] = []
    fake_converter = ModuleType("paroquant.cli.convert")
    fake_converter.main = lambda: received.append(sys.argv.copy())  # type: ignore[attr-defined]
    fake_paroquant = ModuleType("paroquant")
    fake_cli = ModuleType("paroquant.cli")
    fake_cli.convert = fake_converter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paroquant", fake_paroquant)
    monkeypatch.setitem(sys.modules, "paroquant.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "paroquant.cli.convert", fake_converter)
    original_argv = sys.argv

    config = smoke_config()
    export(config)

    assert received == [[
        "paroquant.cli.convert",
        "--model",
        "/models/Qwen3-4B-Instruct-2507",
        "--result-dir",
        "outputs/sphquant/one-layer/Qwen3-4B-Instruct-2507",
        "--output-path",
        "outputs/sphquant/hf-one-layer",
        "--mode",
        "pseudo",
    ]]
    assert sys.argv is original_argv


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

    def mean(self) -> FakeScalar:
        return FakeScalar(0.0)

    def std(self) -> FakeScalar:
        return FakeScalar(1.0)

    def min(self) -> FakeScalar:
        return FakeScalar(-2.0)

    def max(self) -> FakeScalar:
        return FakeScalar(2.0)

    def abs(self) -> "FakeWeight":
        return self


class FakeLinear:
    weight = FakeWeight()


class FakeModel:
    def named_parameters(self) -> list[tuple[str, object]]:
        return [("proj.weight", object())]

    def named_modules(self) -> list[tuple[str, object]]:
        return [("proj", FakeLinear())]


def test_validate_loads_on_gpu_and_prints_linear_statistics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loads: list[tuple[Path, dict[str, Any]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> FakeModel:
            loads.append((path, kwargs))
            return FakeModel()

    fake_torch = ModuleType("torch")
    fake_torch.nn = SimpleNamespace(Linear=FakeLinear)  # type: ignore[attr-defined]
    fake_torch.isfinite = lambda _value: SimpleNamespace(all=lambda: True)  # type: ignore[attr-defined]
    fake_torch.no_grad = nullcontext  # type: ignore[attr-defined]
    fake_torch.tensor = lambda values, **_kwargs: values  # type: ignore[attr-defined]
    fake_torch.quantile = lambda _sample, values: [  # type: ignore[attr-defined]
        FakeScalar(value) for value in values
    ]
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = AutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    validate(smoke_config())

    assert loads[0][0] == Path("outputs/sphquant/hf-one-layer")
    assert loads[0][1]["device_map"] == 0
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert rows == [
        ["layer", "p01", "p50", "p99", "abs_p99", "abs_p999", "abs_max"],
        [
            "proj",
            "1.00000e-02",
            "5.00000e-01",
            "9.90000e-01",
            "9.90000e-01",
            "9.99000e-01",
            "2.00000e+00",
        ],
    ]
