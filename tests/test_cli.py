from pathlib import Path

import pytest

from weight_vq import cli
from weight_vq.config import CalibrationConfig, TokenizerConfig
from weight_vq.qtip.config import QTIPConfig
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.sage.config import SAGEConfig
from weight_vq.sphquant.config import SPHQuantConfig


def test_cli_options_after_config_override_same_named_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "calibration.yaml"
    config.write_text(
        "tokenizer:\n"
        "  name: original\n"
        "  path: /models/original\n"
        "  revision: revision\n"
        "dataset_cache: /datasets\n"
        "output: outputs/calibration.pt\n"
        "seed: 0\n"
        "train_samples: 1920\n"
        "validation_samples: 128\n"
        "sequence_length: 2048\n"
    )
    received: list[CalibrationConfig] = []
    monkeypatch.setitem(cli.STAGES, ("data", "calibration"), (received.append, CalibrationConfig))

    cli.main(
        [
            "data",
            "calibration",
            "--config",
            str(config),
            "--tokenizer.path=/models/override",
            "--train_samples=1",
            "--force=true",
        ]
    )

    assert received == [
        CalibrationConfig(
            tokenizer=TokenizerConfig(
                name="original", path=Path("/models/override"), revision="revision"
            ),
            dataset_cache=Path("/datasets"),
            output=Path("outputs/calibration.pt"),
            seed=0,
            train_samples=1,
            validation_samples=128,
            sequence_length=2048,
            force=True,
        )
    ]


@pytest.mark.parametrize("kernel_layout,compressed", [(False, False), (True, False), (True, True)])
def test_cli_loads_and_forwards_qtip_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kernel_layout: bool, compressed: bool
) -> None:
    config = tmp_path / "qtip.yaml"
    config.write_text(
        "model: /models/qwen3\n"
        "dataset: /data/rp\n"
        "output_dir: outputs/qtip\n"
        "devices: [0, 2]\n"
        "train_samples: 16\n"
        "validation_samples: 8\n"
        "batch_size: 4\n"
        "reconstruction_epochs: 1\n"
        "layer_start: 0\n"
        "layer_end: 4\n"
        f"packed_inference_kernel: {str(kernel_layout).lower()}\n"
        f"export_compressed: {str(compressed).lower()}\n"
    )
    received: list[QTIPConfig] = []
    monkeypatch.setitem(cli.STAGES, ("qtip", "export"), (received.append, QTIPConfig))

    cli.main(["qtip", "export", "--config", str(config)])

    assert received == [
        QTIPConfig(
            model=Path("/models/qwen3"),
            dataset=Path("/data/rp"),
            output_dir=Path("outputs/qtip"),
            devices=[0, 2],
            train_samples=16,
            validation_samples=8,
            batch_size=4,
            reconstruction_epochs=1,
            layer_start=0,
            layer_end=4,
            packed_inference_kernel=kernel_layout,
            export_compressed=compressed,
        )
    ]


def test_cli_rejects_retired_vlm_calibration() -> None:
    assert ("data", "vlm-calibration") not in cli.STAGES
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["data", "vlm-calibration"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize("method,config_type", [("sphquant", SPHQuantConfig), ("sage", SAGEConfig)])
@pytest.mark.parametrize("stage", ["optimize", "export", "validate", "run"])
def test_cli_loads_shared_flat_sphquant_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str, method: str, config_type: type
) -> None:
    config = tmp_path / "sphquant.yaml"
    config.write_text(
        "model: /models/qwen3\n"
        "calibration: /data/calibration.pt\n"
        "output_dir: outputs/sphquant\n"
        "device: 2\n"
        "sample_size: 512\n"
        "kmeans_iterations: 2\n"
        "codebook_bits: 8\n"
        "epochs: 0\n"
        "cache_shards: 2\n"
        "max_layers: 1\n"
        "calibration_limit: 32\n"
    )
    received: list[SPHQuantConfig] = []
    monkeypatch.setitem(cli.STAGES, (method, stage), (received.append, config_type))

    cli.main([method, stage, "--config", str(config)])

    assert received == [
        config_type(
            model=Path("/models/qwen3"),
            calibration=Path("/data/calibration.pt"),
            output_dir=Path("outputs/sphquant"),
            device=2,
            sample_size=512,
            kmeans_iterations=2,
            codebook_bits=8,
            epochs=0,
            cache_shards=2,
            max_layers=1,
            calibration_limit=32,
        )
    ]


@pytest.mark.parametrize(
    "stage", ["calibration", "hessian", "quantize", "export", "validate", "run"]
)
def test_quip_sharp_shared_stages(monkeypatch, tmp_path, stage):
    path = tmp_path / "quip.yaml"
    path.write_text(
        "model: /models/qwen3\ndataset: /data/rp\noutput_dir: /outputs/quip\ndevices: [0, 1]\n"
    )
    received = []
    function, config_type = cli.STAGES[("quip-sharp", stage)]
    assert callable(function)
    assert config_type is QuIPSharpConfig
    monkeypatch.setitem(cli.STAGES, ("quip-sharp", stage), (received.append, config_type))
    cli.main(["quip-sharp", stage, "--config", str(path)])
    assert received[0].hessian_samples == 6144
    assert received[0].hessian_sequence_length == 2048
    assert received[0].devices == [0, 1]


def test_cli_has_nested_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["qtip", "--help"])

    assert exit_info.value.code == 0
    assert "hessian" in capsys.readouterr().out


@pytest.mark.parametrize(
    "method,stage,description",
    [
        ("sage", "run", "Fixed direction-index bits when adaptive_bits is disabled"),
        ("sage", "run", "Local Hugging Face model directory."),
        ("sphquant", "optimize", "quantized student inputs and full-precision teacher targets"),
        ("qtip", "run", "Cached Hugging Face Arrow training dataset directory"),
        ("quip-sharp", "run", "dense teacher inputs and targets"),
        ("unisvq", "run", "Precision for activation-statistics accumulation and storage"),
        ("data", "calibration", "Local directory containing the tokenizer files"),
    ],
)
def test_cli_help_uses_config_docstrings(method, stage, description, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([method, stage, "--help"])
    assert exit_info.value.code == 0
    assert description in " ".join(capsys.readouterr().out.split())


def test_cli_rejects_invalid_pair() -> None:
    with pytest.raises(SystemExit):
        cli.main(["data", "export", "--config", "data.yaml"])
