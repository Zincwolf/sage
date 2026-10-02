"""Real CLI parsing/report output, with only GPU measurement substituted."""

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks import model
from benchmarks.config import BenchmarkConfig
from weight_vq import cli


@pytest.mark.parametrize("empty", [False, True])
def test_benchmark_reports(empty, tmp_path, monkeypatch, capsys):
    output = tmp_path / "measurements"
    rows = (
        []
        if empty
        else [
            ("dense", "prefill", "forward_latency", 12345.6789, "ms"),
            ("qtip", "decode", "forward_tpot", 0.01234, "ms/token"),
            ("quip-sharp", "end_to_end", "output_throughput", 98.7654, "tokens/s"),
        ]
    )

    def measure(config):
        print("measurement diagnostic")
        return {"config": asdict(config), "methods": {}}, rows

    monkeypatch.setattr(model, "benchmark", measure)
    monkeypatch.setenv("COLUMNS", "100")
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"models:\n  dense: {tmp_path}\noutput_dir: {output}\nname: report-test\n"
    )
    cli.main(["benchmark", "model", "--config", str(config_path)])
    captured = capsys.readouterr()
    paths = list(output.glob("*.json"))
    assert len(paths) == 1
    path = paths[0]
    assert re.fullmatch(r"\d{8}-\d{6}-report-test\.json", path.name)
    csv_path = path.with_suffix(".csv")
    assert set(output.iterdir()) == {path, csv_path}
    config = json.loads(path.read_text())["config"]
    assert config["output_dir"] == str(output)
    assert config["models"] == {"dense": str(tmp_path)}
    expected = [
        [method, phase, metric, f"{value:.3f}", unit] for method, phase, metric, value, unit in rows
    ]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["method", "phase", "metric", "value", "unit"],
            *expected,
        ]
    assert str(path) in captured.err and str(csv_path) in captured.err
    assert "measurement diagnostic" in captured.err
    assert "measurement diagnostic" not in captured.out
    assert "\x1b[" not in captured.out
    lines = captured.out.splitlines()
    assert lines[0].split() == ["method", "phase", "metric", "value", "unit"]
    value_end = lines[0].index("value") + len("value")
    for row in expected:
        line = next(line for line in lines if row[2] in line)
        assert line.split() == row
        assert line.index(row[3]) + len(row[3]) == value_end


@pytest.mark.parametrize("cached", [False, True])
def test_benchmark_config_and_overrides(cached, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(
        "models:\n  original: ~/model\noutput_dir: ~/measurements\n"
        f"cached_dense_weight: {str(not cached).lower()}\n"
    )
    received = []
    monkeypatch.setitem(cli.STAGES, ("benchmark", "model"), (received.append, BenchmarkConfig))
    cli.main(
        [
            "benchmark",
            "model",
            "--config",
            str(path),
            "--models",
            '{"dense": "~/dense", "run-a": "~/packed"}',
            f"--cached_dense_weight={str(cached).lower()}",
            "--decode=1",
        ]
    )
    config = received[0]
    assert config.models == {"dense": tmp_path / "dense", "run-a": tmp_path / "packed"}
    assert list(config.models) == ["dense", "run-a"]
    assert config.output_dir == tmp_path / "measurements"
    assert config.cached_dense_weight is cached
    assert config.decode == 1
    assert config.phases == ["prefill", "decode", "end_to_end"]
    assert config.prefill == 128 and config.warmup == 5 and config.samples == 10
    assert (
        not {"mode", "dynamic", "fullgraph", "methods", "quantized_weights"} & asdict(config).keys()
    )


def test_example_and_help(monkeypatch, capsys):
    received = []
    monkeypatch.setitem(cli.STAGES, ("benchmark", "model"), (received.append, BenchmarkConfig))
    path = Path(__file__).parents[2] / "configs/benchmarks/model.example.yaml"
    cli.main(["benchmark", "model", "--config", str(path)])
    assert list(received[0].models) == ["dense", "sage"]
    assert received[0].cached_dense_weight is False
    with pytest.raises(SystemExit) as exc:
        cli.main(["benchmark", "model", "--help"])
    assert exc.value.code == 0
    assert "Retain QTIP/QuIP#/UniSVQ dense weights" in " ".join(capsys.readouterr().out.split())


def test_output_directory_required(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("models: {dense: /path/to/model}\n")
    with pytest.raises(SystemExit):
        cli.main(["benchmark", "model", "--config", str(path)])
