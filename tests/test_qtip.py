import csv
import sys
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.pipeline import (
    _run_workers,
    export,
    layer_complete,
    split_layer_ranges,
    validate,
)
from weight_vq.qtip.worker import qtip_arguments


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


def test_split_layer_ranges_is_balanced_and_contiguous() -> None:
    assert split_layer_ranges(3, 10, [2, 5, 7, 9]) == [
        (2, 3, 5),
        (5, 5, 7),
        (7, 7, 9),
        (9, 9, 10),
    ]
    assert split_layer_ranges(0, 2, [0, 1, 2, 3]) == [(0, 0, 1), (1, 1, 2)]


def test_run_workers_isolates_devices_and_checks_flat_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launches: list[tuple[list[str], str]] = []
    qtip_config = config(tmp_path)

    class Process:
        def __init__(self, command: list[str], environment: dict[str, str]) -> None:
            self.command = command
            self.environment = environment

        def wait(self) -> int:
            start = int(self.command[self.command.index("--layer-start") + 1])
            end = int(self.command[self.command.index("--layer-end") + 1])
            for index in range(start, end):
                for name in ("qkv", "o", "up", "down"):
                    (qtip_config.hessians / f"{index}_{name}.pt").touch()
            return 0

    monkeypatch.setattr("weight_vq.qtip.pipeline._initialize_run", lambda _config: (
        qtip_config.hessians.mkdir(parents=True),
        qtip_config.checkpoints.mkdir(parents=True),
    ))

    def popen(command: list[str], *, env: dict[str, str]) -> Process:
        launches.append((command, env["CUDA_VISIBLE_DEVICES"]))
        return Process(command, env)

    monkeypatch.setattr("weight_vq.qtip.pipeline.subprocess.Popen", popen)
    _run_workers(qtip_config, "hessian")

    assert [device for _command, device in launches] == ["2", "5", "7", "9"]
    assert all("weight_vq.qtip.worker" in command for command, _device in launches)


def test_run_workers_preserves_outputs_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qtip_config = config(tmp_path)
    qtip_config.hessians.mkdir()
    qtip_config.checkpoints.mkdir()
    (qtip_config.hessians / "3_qkv.pt").touch()
    monkeypatch.setattr("weight_vq.qtip.pipeline._initialize_run", lambda _config: None)

    class Process:
        def wait(self) -> int:
            return 4

    monkeypatch.setattr(
        "weight_vq.qtip.pipeline.subprocess.Popen",
        lambda *_args, **_kwargs: Process(),
    )
    with pytest.raises(RuntimeError, match="workers failed"):
        _run_workers(qtip_config, "hessian")
    assert (qtip_config.hessians / "3_qkv.pt").is_file()


def test_qtip_argument_mapping_freezes_verified_recipe() -> None:
    args = qtip_arguments(
        Namespace(
            model=Path("/model"),
            checkpoints=Path("/checkpoints"),
            hessians=Path("/hessians"),
            learning_rate=3e-6,
            reconstruction_epochs=10,
            seed=17,
        )
    )
    assert (args.L, args.K, args.V, args.tlut_bits) == (16, 2, 2, 9)
    assert (args.td_x, args.td_y, args.decode_mode) == (16, 16, "quantlut_sym")
    assert args.split_for_tp is False
    assert args.disable_kernel is True
    assert args.seed == 17
    assert args.ft_lr == 3e-6


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

    fake_torch = ModuleType("torch")
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = AutoModel  # type: ignore[attr-defined]
    fake_transformers.AutoProcessor = AutoProcessor  # type: ignore[attr-defined]
    fake_common = ModuleType("lib.qwen3_vl.common")
    fake_common.audit_qwen3_vl = lambda _model: {}  # type: ignore[attr-defined]
    fake_common.inject_dense_checkpoints = lambda *_args, **_kwargs: [  # type: ignore[attr-defined]
        str(index) for index in range(49)
    ]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "lib", ModuleType("lib"))
    monkeypatch.setitem(sys.modules, "lib.qwen3_vl", ModuleType("lib.qwen3_vl"))
    monkeypatch.setitem(sys.modules, "lib.qwen3_vl.common", fake_common)

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
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = AutoModel  # type: ignore[attr-defined]
    fake_common = ModuleType("lib.qwen3_vl.common")
    fake_common.__dict__["PROJECTIONS"] = (
        ("self_attn.v_proj", "v", "qkv", "col"),
        ("self_attn.q_proj", "q", "qkv", "col"),
        ("self_attn.k_proj", "k", "qkv", "col"),
        ("self_attn.o_proj", "o", "o", "row"),
        ("mlp.up_proj", "up", "up", "col"),
        ("mlp.gate_proj", "gate", "up", "col"),
        ("mlp.down_proj", "down", "down", "row"),
    )
    fake_common.__dict__["get_layers"] = lambda _model: [Layer() for _ in range(36)]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "lib", ModuleType("lib"))
    monkeypatch.setitem(sys.modules, "lib.qwen3_vl", ModuleType("lib.qwen3_vl"))
    monkeypatch.setitem(sys.modules, "lib.qwen3_vl.common", fake_common)

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
