import csv
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from weight_vq.qtip.artifacts import layer_complete
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.pipeline import (
    _initialize_run,
    _run_workers,
    export,
    split_layer_ranges,
    validate,
)


def config(tmp_path: Path) -> QTIPConfig:
    return QTIPConfig(
        model=Path("/models/Qwen3-VL-8B-Instruct"),
        calibration=Path("/data/coco.manifest.json"),
        hessians=tmp_path / "hessians",
        checkpoints=tmp_path / "trellis",
        exported_model=tmp_path / "hf-dense-qdq",
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

    _initialize_run(qtip_config)

    assert received[0]["tlut_path"] == qtip_config.checkpoints / "tlut-l16-b9-v2.pt"


def test_split_layer_ranges_is_balanced_and_contiguous() -> None:
    assert split_layer_ranges(3, 10, [2, 5, 7, 9]) == [(2, 3, 5), (5, 5, 7), (7, 7, 9), (9, 9, 10)]
    assert split_layer_ranges(0, 2, [0, 1, 2, 3]) == [(0, 0, 1), (1, 1, 2)]


def test_run_workers_uses_spawn_with_direct_config_and_layer_ranges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launches: list[tuple[object, tuple[object, ...]]] = []
    qtip_config = config(tmp_path)

    class Process:
        exitcode = 0

        def __init__(self, *, target: object, args: tuple[object, ...]) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            launches.append((self.target, self.args))
            _config, _stage, _device, start, end = self.args
            assert isinstance(start, int) and isinstance(end, int)
            for index in range(start, end):
                for name in ("qkv", "o", "up", "down"):
                    (qtip_config.hessians / f"{index}_{name}.pt").touch()

        def join(self) -> None:
            pass

    context = SimpleNamespace(Process=Process)

    def initialize(_config: QTIPConfig) -> None:
        qtip_config.hessians.mkdir(parents=True)
        qtip_config.checkpoints.mkdir(parents=True)

    monkeypatch.setattr("weight_vq.qtip.pipeline._initialize_run", initialize)
    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.multiprocessing.get_context",
        lambda method: context if method == "spawn" else None,
    )
    _run_workers(qtip_config, "hessian")

    assert [args[2] for _target, args in launches] == [2, 5, 7, 9]
    assert all(args[0] is qtip_config for _target, args in launches)
    assert [args[3:] for _target, args in launches] == [(3, 5), (5, 7), (7, 9), (9, 10)]


def test_run_workers_preserves_outputs_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qtip_config = config(tmp_path)
    qtip_config.hessians.mkdir()
    qtip_config.checkpoints.mkdir()
    (qtip_config.hessians / "3_qkv.pt").touch()
    monkeypatch.setattr("weight_vq.qtip.pipeline._initialize_run", lambda _config: None)

    class Process:
        exitcode = 4

        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    context = SimpleNamespace(Process=Process)

    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.multiprocessing.get_context", lambda _method: context
    )
    with pytest.raises(RuntimeError, match="workers failed"):
        _run_workers(qtip_config, "hessian")
    assert (qtip_config.hessians / "3_qkv.pt").is_file()


def test_qtip_config_is_the_verified_native_recipe() -> None:
    qtip_config = QTIPConfig(
        model=Path("/model"),
        calibration=Path("/calibration"),
        hessians=Path("/hessians"),
        checkpoints=Path("/checkpoints"),
        exported_model=Path("/export"),
        devices=[0],
        seed=17,
    )

    assert (qtip_config.L, qtip_config.K, qtip_config.V, qtip_config.tlut_bits) == (16, 2, 2, 9)
    assert (qtip_config.td_x, qtip_config.td_y, qtip_config.decode_mode) == (16, 16, "quantlut_sym")
    assert qtip_config.packed_inference_kernel is False
    assert qtip_config.split_for_tp is False
    assert qtip_config.learning_rate == 3e-6
    recipe = qtip_config.recipe(model_family="qwen3_vl", layer_end=36)
    assert recipe["seed"] == 17
    assert recipe["format"] == "qtip-bitshift-trellis-qwen3-vl-dense-qdq"
    assert "checkpoints" not in recipe


class FakeModel:
    def __init__(self) -> None:
        self.saved: tuple[Path, dict[str, Any]] | None = None

    def eval(self) -> "FakeModel":
        return self

    def save_pretrained(self, path: Path, **kwargs: Any) -> None:
        self.saved = (path, kwargs)
        (path / "model.safetensors").touch()


class FakeProcessor:
    def save_pretrained(self, path: Path) -> None:
        (path / "processor_config.json").touch()


def test_export_uses_standard_hf_model_and_processor_apis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qtip_config = config(tmp_path)
    qtip_config.checkpoints.mkdir()
    for index in range(3, 10):
        (qtip_config.checkpoints / f"{index}.complete").touch()
        for name in ("v", "q", "k", "o", "up", "gate", "down"):
            (qtip_config.checkpoints / f"{index}_{name}.pt").touch()
    (qtip_config.checkpoints / "qtip_config.json").write_text("{}")
    model = FakeModel()
    model_loads: list[tuple[Path, dict[str, Any]]] = []
    processor_loads: list[tuple[Path, dict[str, Any]]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> FakeModel:
            model_loads.append((path, kwargs))
            return model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> FakeProcessor:
            processor_loads.append((path, kwargs))
            return FakeProcessor()

    monkeypatch.setattr("weight_vq.qtip.pipeline.AutoModelForImageTextToText", AutoModel)
    monkeypatch.setattr("weight_vq.qtip.pipeline.AutoProcessor", AutoProcessor)
    monkeypatch.setattr("weight_vq.qtip.pipeline.audit_qwen3_vl", lambda _model: {})
    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.inject_dense_checkpoints",
        lambda *_args, **_kwargs: [str(index) for index in range(49)],
    )

    export(qtip_config)

    assert model_loads[0][1]["device_map"] == "cpu"
    assert processor_loads[0][1]["local_files_only"] is True
    assert (qtip_config.exported_model / "model.safetensors").is_file()
    assert (qtip_config.exported_model / "processor_config.json").is_file()


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


def test_validate_loads_image_text_model_and_prints_target_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
    monkeypatch.setattr("weight_vq.qtip.pipeline.AutoModelForImageTextToText", AutoModel)
    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.get_layers", lambda _model: [Layer() for _ in range(36)]
    )

    validate(qtip_config)

    assert loads[0]["device_map"] == 0
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert len(rows) == 1 + 7 * 7
    assert rows[1][0] == "model.language_model.layers.3.self_attn.v_proj"


def test_layer_complete_requires_marker_and_all_seven_checkpoints(tmp_path: Path) -> None:
    for name in ("v", "q", "k", "o", "up", "gate", "down"):
        (tmp_path / f"0_{name}.pt").touch()
    assert not layer_complete(0, tmp_path)
    (tmp_path / "0.complete").touch()
    assert layer_complete(0, tmp_path)
