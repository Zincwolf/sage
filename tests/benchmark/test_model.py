"""Short native-HF benchmark acceptance, not full-model throughput or quality."""

import csv
import json
from dataclasses import asdict
from statistics import median

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

from benchmarks import model as benchmark_model
from benchmarks.config import BenchmarkConfig
from weight_vq import checkpoint, cli
from weight_vq.inference.linear import QTIPLinear, QuIPSharpLinear, UniSVQLinear
from weight_vq.integrations.transformers import prepare_for_inference


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    # Production projection widths are required by QTIP's frozen GEMV dispatch.
    # One layer and a tiny vocabulary keep this a short offline integration test.
    directory = tmp_path_factory.mktemp("benchmark-hf") / "source"
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    model.save_pretrained(directory)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[UNK]": 2})),
        pad_token="[PAD]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(directory)
    return directory


@pytest.mark.parametrize("method", ["qtip", "quip-sharp", "unisvq"])
def test_fp16_hf_loading_preserves_packed_abi(tmp_path, method):
    # Exercise trained-parameter ABI flags without pretending these random values
    # are a trained model. HF must preserve explicit buffer precision on load.
    source_dir = tmp_path / "source"
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
    )
    original = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    original.save_pretrained(source_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0})), unk_token="[UNK]"
    )
    tokenizer.save_pretrained(source_dir)
    destination = tmp_path / "packed"
    checkpoint.main(checkpoint.CheckpointConfig(source_dir, method, destination))
    packed = AutoModelForCausalLM.from_pretrained(destination, dtype="auto")
    for name, spec in packed.config.quantization_config.modules.items():
        layer = packed.get_submodule(name)
        if method == "qtip":
            spec["fp32_parameters"] = True
            layer.SU, layer.SV, layer.tlut = layer.SU.float(), layer.SV.float(), layer.tlut.float()
        elif method == "quip-sharp":
            spec["fp32_scales"] = True
            layer.SU, layer.SV = layer.SU.float(), layer.SV.float()
        else:
            spec["fp16_scales"] = True
            layer.SU, layer.SV = layer.SU.half(), layer.SV.half()
    packed.save_pretrained(destination)
    loaded = AutoModelForCausalLM.from_pretrained(
        destination,
        dtype=torch.float16,
        device_map="cpu",
        local_files_only=True,
        attn_implementation="sdpa",
    )
    assert loaded.model.embed_tokens.weight.dtype == torch.float16
    for name in loaded.config.quantization_config.modules:
        layer, expected = loaded.get_submodule(name), packed.get_submodule(name)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)
        assert layer.dense is None and not layer.cached_dense_weight
    assert loaded.config.quantization_config.synthetic_seed == 0


@pytest.mark.cuda
@pytest.mark.parametrize(
    "method,cached",
    [
        ("sage", False),
        ("sphquant", False),
        ("qtip", False),
        ("qtip", True),
        ("quip-sharp", False),
        ("quip-sharp", True),
        ("unisvq", False),
        ("unisvq", True),
    ],
)
def test_hf_benchmark_cli(source, tmp_path, monkeypatch, capsys, method, cached):
    destination = tmp_path / "packed"
    checkpoint.main(checkpoint.CheckpointConfig(source, method, destination, seed=44))
    before = {path.name: path.stat().st_mtime_ns for path in destination.iterdir()}
    prepared = []

    def observe_preparation(model, *, cached_dense_weight=False):
        prepare_for_inference(model, cached_dense_weight=cached_dense_weight)
        layers = [
            module
            for module in model.modules()
            if isinstance(module, (QTIPLinear, QuIPSharpLinear, UniSVQLinear))
        ]
        prepared.append(
            (
                model.model.embed_tokens.weight.dtype,
                [(layer.cached_dense_weight, layer.dense is not None) for layer in layers],
            )
        )

    monkeypatch.setattr(benchmark_model, "prepare_for_inference", observe_preparation)
    # Labels intentionally differ from method names: the HF metadata is authoritative.
    config = BenchmarkConfig(
        models={"dense": source, "candidate": destination},
        output_dir=tmp_path / "reports",
        name="cli-test",
        cached_dense_weight=cached,
        prefill=9,
        decode=3,
        warmup=0,
        samples=2,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config), default=str))
    cli.main(["benchmark", "model", "--config", str(config_path)])
    captured = capsys.readouterr()
    assert "preferred CUDNN_ATTENTION was not verified" not in captured.err
    assert prepared[0] == (torch.float16, [])
    assert prepared[1][0] == torch.float16
    if method in ("qtip", "quip-sharp", "unisvq"):
        assert len(prepared[1][1]) == 7
        assert all(policy == (cached, cached) for policy in prepared[1][1])
    assert before == {path.name: path.stat().st_mtime_ns for path in destination.iterdir()}
    (path,) = config.output_dir.glob("*.json")
    result = json.loads(path.read_text())
    with path.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert result["cudnn_version"] and result["weight_vq_version"]
    assert result["attention_backend_priority"] == [
        "CUDNN_ATTENTION",
        "FLASH_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
    ]
    assert result["execution"]["compile"] == {
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }
    assert result["methods"]["candidate"]["method"] == method
    assert result["methods"]["candidate"]["synthetic_seed"] == 44
    for label, raw in result["methods"].items():
        assert raw["checkpoint_directory_bytes"] > 0
        assert 0 < raw["model_bytes"] <= raw["peak_allocated_bytes"] <= raw["peak_reserved_bytes"]
        for phase in config.phases:
            assert len(raw["samples"][phase]) == config.samples
            reports = raw["attention_backends"][phase]
            assert set(reports) == ({"prefill", "decode"} if phase == "end_to_end" else {phase})
            assert all(report["observed_backends"] for report in reports.values())
            for sample in raw["samples"][phase]:
                assert 0 < sample["cuda_event_ms"] <= sample["synchronized_wall_ms"]
        samples = raw["samples"]["end_to_end"]
        for sample in samples:
            assert sample["actual_generated_tokens"] == config.batch_size * config.decode
            assert 0 < sample["gpu_ttft_ms"] <= sample["cuda_event_ms"]
            assert 0 < sample["first_token_cpu_ms"] <= sample["synchronized_wall_ms"]
        metrics = {
            row["metric"]: float(row["value"])
            for row in rows
            if row["method"] == label and row["phase"] == "end_to_end"
        }
        assert metrics["ttft"] == pytest.approx(
            median(sample["gpu_ttft_ms"] for sample in samples), abs=0.0005, rel=0
        )
        assert metrics["tpot"] == pytest.approx(
            median(
                (sample["synchronized_wall_ms"] - sample["first_token_cpu_ms"])
                / (config.decode - 1)
                for sample in samples
            ),
            abs=0.0005,
            rel=0,
        )
        assert metrics["generate_latency"] == pytest.approx(
            median(sample["synchronized_wall_ms"] for sample in samples), abs=0.0005, rel=0
        )
        assert metrics["output_throughput"] == pytest.approx(
            config.batch_size
            * config.decode
            * config.samples
            * 1000
            / sum(sample["synchronized_wall_ms"] for sample in samples),
            abs=0.0005,
            rel=0,
        )
        decode_samples = raw["samples"]["decode"]
        decode_metrics = {
            row["metric"]: float(row["value"])
            for row in rows
            if row["method"] == label and row["phase"] == "decode"
        }
        assert decode_metrics["forward_tpot"] == pytest.approx(
            median(sample["synchronized_wall_ms"] for sample in decode_samples)
            / (config.decode - 1),
            abs=0.0005,
            rel=0,
        )
        assert decode_metrics["output_throughput"] == pytest.approx(
            config.batch_size
            * (config.decode - 1)
            * config.samples
            * 1000
            / sum(sample["synchronized_wall_ms"] for sample in decode_samples),
            abs=0.0005,
            rel=0,
        )
    if method in ("qtip", "quip-sharp", "unisvq"):
        quantized_bytes = result["methods"]["candidate"]["model_bytes"]
        dense_bytes = result["methods"]["dense"]["model_bytes"]
        assert (quantized_bytes > dense_bytes) if cached else (quantized_bytes < dense_bytes)
