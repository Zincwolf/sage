import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from weight_vq import calibration, utils
from weight_vq.config import CalibrationConfig, TokenizerConfig


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def test_text_blocks_filter_concatenate_and_chunk() -> None:
    rows = [
        {"text": ""},
        {"text": " too-long "},
        {"text": " ab "},
        {"text": "cd"},
        {"text": "ef"},
    ]

    assert calibration._text_blocks(rows, CharacterTokenizer(), 2, 3) == [
        [ord("a"), ord("b"), ord("c")],
        [ord("d"), ord("e"), ord("f")],
    ]


def test_allocation_gives_remainder_to_last_dataset() -> None:
    assert calibration._allocation(1920) == {
        "wikitext2": 640,
        "c4": 640,
        "redpajama": 640,
    }
    assert calibration._allocation(128) == {
        "wikitext2": 42,
        "c4": 42,
        "redpajama": 44,
    }


def test_redpajama_partition() -> None:
    class Rows(list[int]):
        def select(self, indices: range) -> "Rows":
            return Rows(self[index] for index in indices)

    rows = Rows(range(10))
    assert calibration._redpajama_split(rows, "train") == rows[:7]
    assert calibration._redpajama_split(rows, "validation") == rows[7:8]


def test_load_rows_uses_standard_datasets_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: dict[str, object] = {}

    class FakeDataset:
        def shuffle(self, *, seed: int) -> "FakeDataset":
            received["seed"] = seed
            return self

    def load_dataset(*args: object, **kwargs: object) -> FakeDataset:
        received["args"] = args
        received["kwargs"] = kwargs
        return FakeDataset()

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )

    result = calibration._load_rows(tmp_path, "c4", "train", 9)

    recipe = calibration.DATASETS["c4"]
    assert isinstance(result, FakeDataset)
    assert received["args"] == (recipe["name"], recipe["subset"])
    assert received["kwargs"] == {
        "split": "train",
        "revision": recipe["revision"],
        "data_files": {"train": recipe["data_files"]["train"]},
        "cache_dir": str(tmp_path),
        "trust_remote_code": False,
    }
    assert received["seed"] == 9


def test_manifest_path_is_derived_from_output() -> None:
    assert utils.manifest_path(Path("calibration.pt")) == Path(
        "calibration.manifest.json"
    )


def test_tokenizer_manifest_has_identity_without_local_path(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    config = CalibrationConfig(
        tokenizer=TokenizerConfig(
            name="tests/tokenizer", path=tmp_path, revision="b" * 40
        ),
        dataset_cache=tmp_path,
        output=tmp_path / "calibration.pt",
        seed=0,
        train_samples=3,
        validation_samples=3,
        sequence_length=2,
    )
    tokenizer = SimpleNamespace(
        is_fast=True,
        vocab_size=128,
        bos_token_id=None,
        eos_token_id=2,
        pad_token_id=0,
        unk_token_id=None,
    )

    identity = calibration._tokenizer_manifest(config, tokenizer)

    assert identity["name"] == "tests/tokenizer"
    assert str(tmp_path) not in str(identity)
    assert identity["files"] == {
        "tokenizer.json": hashlib.sha256(b"tokenizer").hexdigest()
    }


def test_expected_hash_is_optional_but_enforced_when_present() -> None:
    utils.check_hash("train", "abc", None)
    utils.check_hash("train", "abc", "abc")
    with pytest.raises(RuntimeError, match="train SHA-256 mismatch"):
        utils.check_hash("train", "abc", "def")
