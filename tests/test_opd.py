import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import replace
from threading import Event, Lock

import httpx
import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from weight_vq import cli, opd
from weight_vq.calibration import load_calibration_metadata
from weight_vq.config import CalibrationMetadata


class Tokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
        text = messages[0]["content"]
        if text == "bad-template":
            raise ValueError("unrenderable tool turn")
        return [ord(character) for character in text]


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    dataset = tmp_path / "data"
    dataset.mkdir()
    for split, filename in (
        ("train", "train_clean_balanced.parquet"),
        ("held", "heldout_clean.parquet"),
    ):
        pd.DataFrame(
            [
                {
                    "ability": domain,
                    "prompt": [{"role": "user", "content": f"{split}-{domain}-{index}"}],
                    "extra_info": {"source": domain, "source_id": f"{split}-{index}"},
                }
                for domain in opd.QUOTA
                for index in range(32)
            ]
        ).to_parquet(dataset / filename, index=False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    return opd.OPDConfig(
        model=tmp_path / "model",
        dataset=dataset,
        base_url="http://service/v1",
        output_dir=tmp_path / "output",
        train_samples=2,
        validation_samples=1,
        sequence_length=32,
        train_prompts=16,
        validation_prompts=16,
        max_new_tokens=8,
        max_model_len=64,
        batch_size=8,
    )


@pytest.fixture
def service(monkeypatch, recipe):
    calls = []
    settings = {
        "fail": None,
        "wrong_root": False,
        "tokens": True,
        "prompt_mismatch": False,
        "finish_reason": "stop",
        "context": 64,
    }
    client_type = httpx.Client

    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "served-alias",
                            "root": str(recipe.model) if not settings["wrong_root"] else "/other",
                            "max_model_len": settings["context"],
                        }
                    ]
                },
            )
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        assert body["model"] == "served-alias"
        assert body["temperature"] == 0.0
        assert body["return_token_ids"] is True
        assert body["add_special_tokens"] is False
        calls.append(body["prompt"])
        if body["prompt"] == settings["fail"]:
            return httpx.Response(503, json={"error": "interrupted"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": [401, 999] if settings["tokens"] else None,
                        "prompt_token_ids": [] if settings["prompt_mismatch"] else body["prompt"],
                        "finish_reason": settings["finish_reason"],
                        "stop_reason": 999,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handle)),
    )
    return calls, settings


def test_sampling_matches_reference_and_retains_identity(recipe):
    frame = pd.read_parquet(recipe.dataset / "train_clean_balanced.parquet")
    result = opd.sample_prompts(frame, 64, 7)
    assert Counter(row["ability"] for row in result) == opd.QUOTA
    expected = []
    for domain, share in opd.QUOTA.items():
        expected.extend(frame[frame.ability == domain].sample(n=share, random_state=7).index)
    expected.sort(key=lambda index: hashlib.sha256(str(index).encode()).hexdigest())
    assert [row["row"] for row in result] == expected
    assert result == opd.sample_prompts(frame, 64, 7)
    assert all(row["source_id"].startswith("train-") for row in result)


def test_pack_eos_cut_and_domain_accounting():
    records = [
        {"prompt_ids": [1], "token_ids": [2, 9], "ability": "code"},
        {"prompt_ids": [3], "token_ids": [4], "ability": "math"},
        {"prompt_ids": [5], "token_ids": [6, 7], "ability": "tool"},
    ]
    tensor, stats = opd.pack(records, 2, 4, 9)
    assert tensor.dtype == torch.int64
    assert tensor.tolist() == [[1, 2, 9, 3], [4, 9, 5, 6]]
    assert stats == {
        "conversations": {"code": 1, "math": 1, "tool": 1},
        "tokens": {"code": 3, "math": 3, "tool": 2},
    }
    with pytest.raises(RuntimeError, match="needed"):
        opd.pack(records, 100, 4, 9)


def test_real_parquet_to_calibration_contract(recipe, service):
    calls, _ = service
    opd.main(recipe)
    tensors = torch.load(recipe.output_dir / "calibration.pt", weights_only=True)
    assert load_calibration_metadata(recipe.output_dir / "calibration.pt") == CalibrationMetadata(
        2, 1, 32
    )
    manifest = json.loads((recipe.output_dir / "calibration.manifest.json").read_text())
    with sqlite3.connect(recipe.output_dir / "rollout.sqlite3") as cache:
        selection = json.loads(dict(cache.execute("SELECT key, value FROM metadata"))["selection"])
        responses = dict(cache.execute("SELECT position, value FROM responses"))
    assert len(calls) == len(responses) == len(selection)
    for split, count in (("train", 2), ("validation", 1)):
        ordered = [
            token
            for row in selection
            if row["split"] == split
            for token in row["prompt_ids"] + [401, 999]
        ]
        assert tensors[split].flatten().tolist() == ordered[: count * 32]
        assert manifest["provenance"]["rollout"][split]["empty_completions"] == 0
    with pytest.raises(FileExistsError, match="completed"):
        opd.main(recipe)


def test_failed_request_drains_successes_and_resume_requests_only_gaps(recipe, service):
    calls, settings = service
    frame = pd.read_parquet(recipe.dataset / "train_clean_balanced.parquet")
    first = opd.sample_prompts(frame, recipe.train_prompts, recipe.seed)[0]
    failed_prompt = [ord(c) for c in first["messages"][0]["content"]]
    settings["fail"] = failed_prompt
    with pytest.raises(RuntimeError, match="rerun to resume"):
        opd.main(recipe)
    with sqlite3.connect(recipe.output_dir / "rollout.sqlite3") as cache:
        assert cache.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == len(calls) - 1
        assert len(calls) >= recipe.batch_size
    successful = [prompt for prompt in calls if prompt != failed_prompt]
    calls.clear()
    settings["fail"] = None
    # Endpoint/batch-size are operational, not token recipe fields.
    opd.main(replace(recipe, batch_size=3, base_url="http://restarted/v1"))
    assert failed_prompt in calls
    assert all(prompt not in calls for prompt in successful)


@pytest.mark.parametrize("change", ["recipe", "tokenizer", "dataset"])
def test_resume_rejects_changed_inputs(recipe, service, monkeypatch, change):
    _, settings = service
    settings["tokens"] = False
    with pytest.raises(RuntimeError, match="rerun to resume"):
        opd.main(recipe)
    if change == "recipe":
        recipe = replace(recipe, seed=1)
    elif change == "tokenizer":
        monkeypatch.setattr(Tokenizer, "apply_chat_template", lambda *args, **kwargs: [77])
    else:
        path = recipe.dataset / "train_clean_balanced.parquet"
        frame = pd.read_parquet(path)
        frame["prompt"] = [[{"role": "user", "content": "changed"}]] * len(frame)
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="differ"):
        opd.main(recipe)


def test_publication_failure_resumes_without_service(recipe, service, monkeypatch):
    original = opd.utils.atomic_write_text

    def interrupt(path, content):
        if path.name == "calibration.manifest.json":
            raise OSError("interrupted publication")
        original(path, content)

    monkeypatch.setattr(opd.utils, "atomic_write_text", interrupt)
    with pytest.raises(OSError, match="publication"):
        opd.main(recipe)
    assert (recipe.output_dir / "calibration.pt").is_file()
    monkeypatch.setattr(opd.utils, "atomic_write_text", original)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("cache must suffice"))
    opd.main(recipe)
    assert (recipe.output_dir / "calibration.manifest.json").is_file()


@pytest.mark.parametrize(
    "setting,value,error",
    [
        ("wrong_root", True, "matching local model"),
        ("context", 32, "server context"),
        ("tokens", False, "request failed"),
        ("prompt_mismatch", True, "request failed"),
        ("finish_reason", "error", "request failed"),
    ],
)
def test_rejects_incompatible_server(recipe, service, setting, value, error):
    _, settings = service
    settings[setting] = value
    with pytest.raises((RuntimeError, ValueError), match=error):
        opd.main(recipe)
    assert not (recipe.output_dir / "calibration.pt").exists()


def test_length_stopped_completions_are_kept(recipe, service):
    _, settings = service
    settings["finish_reason"] = "length"
    opd.main(recipe)
    manifest = json.loads((recipe.output_dir / "calibration.manifest.json").read_text())
    for stats in manifest["provenance"]["rollout"].values():
        assert stats["finish_reasons"] == {"length": stats["rollouts"]}


def test_filtering_is_reported_per_split_and_domain(recipe, service):
    for filename in ("train_clean_balanced.parquet", "heldout_clean.parquet"):
        path = recipe.dataset / filename
        frame = pd.read_parquet(path)
        frame.loc[frame.ability == "tool", "prompt"] = pd.Series(
            [[{"role": "assistant", "content": "bad-template"}]] * len(frame), index=frame.index
        )
        frame.loc[frame.ability == "general", "prompt"] = pd.Series(
            [[{"role": "user", "content": "x" * 80}]] * len(frame), index=frame.index
        )
        frame.to_parquet(path, index=False)
    opd.main(recipe)
    preparation = json.loads((recipe.output_dir / "preparation.json").read_text())
    for split in preparation.values():
        assert split["template_errors"] == {"tool": split["sampled"]["tool"]}
        assert split["too_long"] == {"general": split["sampled"]["general"]}
        assert "tool" not in split["kept"]
        assert "general" not in split["kept"]


@pytest.mark.parametrize("capacity", [2, 5])
def test_queue_refills_before_slowest_request_finishes(recipe, monkeypatch, capacity):
    cohort_started = Event()
    replacement_started = Event()
    lock = Lock()
    started = active = peak = 0
    client_type = httpx.Client

    def handle(request):
        nonlocal started, active, peak
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "model", "root": str(recipe.model), "max_model_len": 64}]},
            )
        body = json.loads(request.content)
        with lock:
            started += 1
            position = started
            active += 1
            peak = max(peak, active)
            if position == capacity:
                cohort_started.set()
            if position == capacity + 1:
                replacement_started.set()
        try:
            if position <= capacity:
                assert cohort_started.wait(timeout=10), "initial concurrency not reached"
            if position == 1:
                assert replacement_started.wait(timeout=10), "whole-batch barrier prevented refill"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "prompt_token_ids": body["prompt"],
                            "token_ids": [401, 999],
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handle)),
    )
    opd.main(replace(recipe, batch_size=capacity))
    assert peak == capacity
    assert active == 0
    assert started > capacity


def test_cli_opd_registration_and_overrides(recipe, monkeypatch):
    config = recipe.output_dir.parent / "config.yaml"
    config.write_text(
        f"model: {recipe.model}\ndataset: {recipe.dataset}\n"
        f"base_url: {recipe.base_url}\noutput_dir: {recipe.output_dir}\n"
    )
    received = []
    monkeypatch.setitem(cli.STAGES, ("data", "opd"), (received.append, opd.OPDConfig))
    cli.main(["data", "opd", "--config", str(config), "--batch_size=64"])
    assert received[0].batch_size == 64
    assert received[0].train_samples == 1920
    assert received[0].validation_samples == 128
