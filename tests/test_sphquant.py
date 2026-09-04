import csv
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from weight_vq.config import CalibrationMetadata
from weight_vq.sphquant.config import SPHQuantConfig
from weight_vq.sphquant.pipeline import (
    PROJECTIONS,
    _model_contract,
    checkpoint_path,
    export,
    missing_checkpoints,
    optimize,
    validate,
)


def smoke_config(tmp_path: Path, family: str = "qwen3_vl") -> SPHQuantConfig:
    vlm = family == "qwen3_vl"
    return SPHQuantConfig(
        model=Path("/models/Qwen3-VL-8B-Instruct" if vlm else "/models/Qwen3-4B-Instruct-2507"),
        calibration=Path(
            "/data/coco_calibration_manifest.json" if vlm else "outputs/calibration.pt"
        ),
        checkpoints=tmp_path / "checkpoints",
        exported_model=tmp_path / ("hf-vlm" if vlm else "hf-text"),
        device=3,
        sample_size=512,
        kmeans_iterations=2,
        codebook_bits=8,
        epochs=0,
        cache_shards=2,
        max_layers=1,
        calibration_limit=32,
    )


def create_checkpoints(config: SPHQuantConfig) -> None:
    for projection in PROJECTIONS:
        path = checkpoint_path(config, 0, projection)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def install_fake_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[dict[str, object], str | None]], object]:
    parsed: list[tuple[dict[str, object], str | None]] = []
    reference_parse = object()
    fake_module = ModuleType("paroquant.cli.optimize")
    fake_module.Config = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fake_module.simple_parsing = SimpleNamespace(parse=reference_parse)  # type: ignore[attr-defined]
    fake_module.main = lambda: parsed.append(  # type: ignore[attr-defined]
        (fake_module.simple_parsing.parse(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    )
    fake_cli = ModuleType("paroquant.cli")
    fake_cli.optimize = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paroquant", ModuleType("paroquant"))
    monkeypatch.setitem(sys.modules, "paroquant.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "paroquant.cli.optimize", fake_module)
    return parsed, reference_parse


def test_model_contract_accepts_only_qwen3_and_qwen3_vl(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = {
        "text": SimpleNamespace(model_type="qwen3", num_hidden_layers=36),
        "vlm": SimpleNamespace(
            model_type="qwen3_vl", text_config=SimpleNamespace(num_hidden_layers=40)
        ),
        "other": SimpleNamespace(model_type="llama"),
    }

    class AutoConfig:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> object:
            assert kwargs == {"local_files_only": True, "trust_remote_code": True}
            return configs[path.name]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = AutoConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert _model_contract(Path("text")) == ("qwen3", 36)
    assert _model_contract(Path("vlm")) == ("qwen3_vl", 40)
    with pytest.raises(ValueError, match="supports only qwen3 and qwen3_vl"):
        _model_contract(Path("other"))


def test_optimize_preserves_qwen3_text_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parsed, reference_parse = install_fake_optimizer(monkeypatch)
    monkeypatch.setattr("weight_vq.sphquant.pipeline._model_contract", lambda _model: ("qwen3", 36))
    monkeypatch.setattr(
        "weight_vq.sphquant.pipeline.load_calibration_metadata",
        lambda _path: CalibrationMetadata(1920, 128, 2048),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8")

    optimize(smoke_config(tmp_path, "qwen3"))

    arguments, device = parsed[0]
    assert arguments["datasets"] == ["wikitext2", "c4", "redpajama"]
    assert arguments["val_dataset"] == "pileval"
    assert arguments["calibration_path"] == "outputs/calibration.pt"
    assert arguments["multimodal_calibration_path"] is None
    assert arguments["model_dtype"] == "float16"
    assert arguments["train_size"] == 1920
    assert arguments["validation_size"] == 128
    assert arguments["seqlen"] == 2048
    assert device == "3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "8"
    assert sys.modules["paroquant.cli.optimize"].simple_parsing.parse is reference_parse  # type: ignore[attr-defined]


def test_optimize_maps_qwen3_vl_recipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parsed, _ = install_fake_optimizer(monkeypatch)
    monkeypatch.setattr(
        "weight_vq.sphquant.pipeline._model_contract", lambda _model: ("qwen3_vl", 36)
    )

    optimize(smoke_config(tmp_path))

    arguments, device = parsed[0]
    assert arguments["method"] == "sphquant"
    assert arguments["params"] == ["radius:0.001,codebook:0.001"]
    assert arguments["datasets"] == ["coco"]
    assert arguments["train_size"] == 1920
    assert arguments["validation_size"] == 128
    assert arguments["calibration_path"] is None
    assert arguments["multimodal_calibration_path"] == ("/data/coco_calibration_manifest.json")
    assert arguments["model_dtype"] == "bfloat16"
    assert arguments["sph_sample_size"] == 512
    assert arguments["sph_kmeans_iterations"] == 2
    assert arguments["sph_codebook_bits"] == 8
    assert device == "3"


def test_missing_checkpoints_uses_model_layer_count(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    create_checkpoints(config)
    assert missing_checkpoints(config, model_layers=36) == []
    missing = checkpoint_path(config, 0, "mlp.down_proj")
    missing.unlink()
    assert missing_checkpoints(config, model_layers=36) == [missing]


def test_export_rejects_incomplete_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "weight_vq.sphquant.pipeline._model_contract", lambda _model: ("qwen3_vl", 36)
    )
    with pytest.raises(FileNotFoundError, match="missing 7 SPHQuant checkpoints"):
        export(smoke_config(tmp_path))


def test_export_runs_shared_converter_in_staging_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[tuple[list[str], str | None]] = []

    def converter_main() -> None:
        received.append((sys.argv.copy(), os.environ.get("CUDA_VISIBLE_DEVICES")))
        output = Path(sys.argv[sys.argv.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "model.safetensors").touch()

    fake_converter = ModuleType("paroquant.cli.convert")
    fake_converter.main = converter_main  # type: ignore[attr-defined]
    fake_cli = ModuleType("paroquant.cli")
    fake_cli.convert = fake_converter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paroquant", ModuleType("paroquant"))
    monkeypatch.setitem(sys.modules, "paroquant.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "paroquant.cli.convert", fake_converter)
    monkeypatch.setattr("weight_vq.sphquant.pipeline._model_contract", lambda _model: ("qwen3", 36))
    original_argv = sys.argv

    config = smoke_config(tmp_path, "qwen3")
    create_checkpoints(config)
    export(config)

    arguments, device = received[0]
    assert arguments[1:5] == [
        "--model",
        "/models/Qwen3-4B-Instruct-2507",
        "--result-dir",
        str(config.checkpoints / "Qwen3-4B-Instruct-2507"),
    ]
    staging = Path(arguments[arguments.index("--output-path") + 1])
    assert staging.name.startswith(".hf-text.tmp-")
    assert device == "3"
    assert (config.exported_model / "model.safetensors").is_file()
    assert not staging.exists()
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

    def max(self) -> FakeScalar:
        return FakeScalar(2.0)

    def abs(self) -> "FakeWeight":
        return self


class Projection:
    weight = FakeWeight()


class Layer:
    self_attn = SimpleNamespace(
        q_proj=Projection(), k_proj=Projection(), v_proj=Projection(), o_proj=Projection()
    )
    mlp = SimpleNamespace(gate_proj=Projection(), up_proj=Projection(), down_proj=Projection())


class TextModel:
    model = SimpleNamespace(layers=[Layer()])


class VLMModel:
    model = SimpleNamespace(language_model=SimpleNamespace(layers=[Layer()]))


@pytest.mark.parametrize(
    ("family", "model_class", "expected_prefix"),
    [("qwen3", TextModel, "model.layers"), ("qwen3_vl", VLMModel, "model.language_model.layers")],
)
def test_validate_uses_family_hf_api_and_target_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    family: str,
    model_class: type[TextModel] | type[VLMModel],
    expected_prefix: str,
) -> None:
    loads: list[tuple[str, Path, dict[str, Any], str | None]] = []

    class AutoCausal:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> object:
            loads.append(("causal", path, kwargs, os.environ.get("CUDA_VISIBLE_DEVICES")))
            return model_class()

    class AutoVLM:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> object:
            loads.append(("vlm", path, kwargs, os.environ.get("CUDA_VISIBLE_DEVICES")))
            return model_class()

    fake_torch = ModuleType("torch")
    fake_torch.isfinite = lambda _value: SimpleNamespace(all=lambda: True)  # type: ignore[attr-defined]
    fake_torch.no_grad = nullcontext  # type: ignore[attr-defined]
    fake_torch.tensor = lambda values, **_kwargs: values  # type: ignore[attr-defined]
    fake_torch.quantile = lambda _sample, values: [  # type: ignore[attr-defined]
        FakeScalar(value) for value in values
    ]
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = AutoCausal  # type: ignore[attr-defined]
    fake_transformers.AutoModelForImageTextToText = AutoVLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("weight_vq.sphquant.pipeline._model_contract", lambda _model: (family, 36))

    config = smoke_config(tmp_path, family)
    validate(config)

    expected_loader = "causal" if family == "qwen3" else "vlm"
    assert loads == [
        (
            expected_loader,
            config.exported_model,
            {
                "dtype": "auto",
                "device_map": 0,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
                "trust_remote_code": True,
            },
            "3",
        )
    ]
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert len(rows) == 1 + len(PROJECTIONS)
    assert rows[1][0] == f"{expected_prefix}.0.self_attn.q_proj"
    assert rows[-1][0] == f"{expected_prefix}.0.mlp.down_proj"
