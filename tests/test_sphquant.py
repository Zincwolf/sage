import csv
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from torch import nn

from weight_vq.config import CalibrationMetadata
from weight_vq.sage.config import SAGEConfig
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


def smoke_config(
    tmp_path: Path, config_type: type[SPHQuantConfig] = SPHQuantConfig
) -> SPHQuantConfig:
    return config_type(
        model=Path("/models/Qwen3-4B-Instruct-2507"),
        calibration=Path("outputs/calibration.pt"),
        output_dir=tmp_path,
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


def test_model_contract_accepts_only_qwen3(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = {
        "text": SimpleNamespace(model_type="qwen3", num_hidden_layers=36),
        "retired_vl": SimpleNamespace(model_type="qwen3_vl"),
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

    assert _model_contract(Path("text")) == 36
    for name in ("retired_vl", "other"):
        with pytest.raises(ValueError, match="supports only qwen3, got"):
            _model_contract(Path(name))


@pytest.mark.parametrize("config_type", [SPHQuantConfig, SAGEConfig])
def test_optimize_preserves_qwen3_text_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_type: type[SPHQuantConfig]
) -> None:
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
    monkeypatch.setattr("weight_vq.sphquant.pipeline._model_contract", lambda _model: 36)
    monkeypatch.setattr(
        "weight_vq.sphquant.pipeline.load_calibration_metadata",
        lambda _path: CalibrationMetadata(1920, 128, 2048),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8")

    optimize(smoke_config(tmp_path, config_type))

    arguments, device = parsed[0]
    assert arguments["method"] == ("sage" if config_type is SAGEConfig else "sphquant")
    if config_type is SAGEConfig:
        assert arguments["sage_esg"] is True
        assert arguments["sage_adaptive_bits"] is True
        assert arguments["sage_energy_power"] == 0.75
    else:
        assert not any(key.startswith("sage_") for key in arguments)
    assert arguments["params"] == ["radius:0.001,codebook:0.001"]
    assert arguments["datasets"] == ["wikitext2", "c4", "redpajama"]
    assert arguments["val_dataset"] == "pileval"
    assert arguments["calibration_path"] == "outputs/calibration.pt"
    assert "multimodal_calibration_path" not in arguments
    assert "model_dtype" not in arguments
    assert arguments["train_size"] == 1920
    assert arguments["validation_size"] == 128
    assert arguments["seqlen"] == 2048
    assert arguments["output_dir"] == str(tmp_path / "checkpoints")
    assert arguments["sph_sample_size"] == 512
    assert arguments["sph_kmeans_iterations"] == 2
    assert arguments["sph_codebook_bits"] == 8
    assert device == "3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "8"
    assert sys.modules["paroquant.cli.optimize"].simple_parsing.parse is reference_parse  # type: ignore[attr-defined]


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
    monkeypatch.setattr("weight_vq.sphquant.pipeline._model_contract", lambda _model: 36)
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
    monkeypatch.setattr("weight_vq.sphquant.pipeline._model_contract", lambda _model: 36)
    original_argv = sys.argv

    config = smoke_config(tmp_path)
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
    assert staging.name.startswith(".exported_model.tmp-")
    assert staging.parent == config.output_dir
    assert device == "3"
    assert (config.exported_model / "model.safetensors").is_file()
    assert not staging.exists()
    assert sys.argv is original_argv


def test_validate_uses_causal_hf_api_and_target_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.mlp = nn.Module()
    for projection in PROJECTIONS:
        parent, name = projection.split(".")
        layer.get_submodule(parent).add_module(name, nn.Linear(2, 2, bias=False))
    model.model.layers = nn.ModuleList([layer])
    loads: list[tuple[Path, dict[str, Any], str | None]] = []

    class AutoCausal:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: Any) -> object:
            loads.append((path, kwargs, os.environ.get("CUDA_VISIBLE_DEVICES")))
            return model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = AutoCausal  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("weight_vq.sphquant.pipeline._model_contract", lambda _model: 36)

    config = smoke_config(tmp_path)
    validate(config)

    assert loads == [
        (
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
    assert rows[1][0] == "model.layers.0.self_attn.q_proj"
    assert rows[-1][0] == "model.layers.0.mlp.down_proj"
