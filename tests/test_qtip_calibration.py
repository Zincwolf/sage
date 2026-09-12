from dataclasses import replace
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from weight_vq.blockwise.calibration import calibration, load_calibration, sample_documents
from weight_vq.qtip.config import QTIPConfig


class Tokenizer:
    name_or_path = "/model"

    def __call__(self, text, *, truncation, padding, return_attention_mask, max_length=None):
        tokens = [int(value) for value in text.split()]
        return {"input_ids": tokens[:max_length] if truncation else tokens}


def test_hessian_concatenates_full_documents_and_keeps_long_ones():
    dataset = Dataset.from_dict({"text": ["1 2", "3 4 5 6 7 8 9"]})
    generator = torch.Generator().manual_seed(3)
    reference = torch.Generator().manual_seed(3)
    expected = []
    while len(expected) < 12:
        index = int(torch.randint(2, (1,), generator=reference).item())
        expected.extend([int(v) for v in dataset[index]["text"].split()])
    actual = sample_documents(
        dataset, Tokenizer(), samples=3, length=4, concatenate=True, generator=generator
    )
    torch.testing.assert_close(actual, torch.tensor(expected[:12]).reshape(3, 4))


def test_reconstruction_rejects_short_documents_and_keeps_prefixes_with_replacement():
    dataset = Dataset.from_dict({"text": ["1 2", "3 4 5 6 7 8 9"]})
    actual = sample_documents(
        dataset,
        Tokenizer(),
        samples=5,
        length=4,
        concatenate=False,
        generator=torch.Generator().manual_seed(0),
    )
    torch.testing.assert_close(actual, torch.tensor([[3, 4, 5, 6]]).expand(5, -1))


def test_calibration_artifact_protocol_and_stage_splits(tmp_path, monkeypatch, capsys):
    from datasets import DatasetInfo

    config = QTIPConfig(
        model=Path("/model"),
        dataset=tmp_path / "rp",
        output_dir=tmp_path,
        devices=[0],
        hessian_samples=3,
        hessian_sequence_length=8,
        train_samples=2,
        validation_samples=1,
        sequence_length=4,
    )
    dataset = Dataset.from_dict(
        {"text": ["1 2 3 4 5 6 7 8 9"]}, info=DatasetInfo(dataset_name="fixture")
    )
    dataset.save_to_disk(str(config.dataset))
    (config.dataset / "data-00000-of-00001.arrow").rename(
        config.dataset / "fixture-train-00000-of-00001.arrow"
    )
    tokenizer = Tokenizer()
    monkeypatch.setattr(
        "weight_vq.blockwise.calibration.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer
    )
    capsys.readouterr()  # Exclude the fixture's Dataset.save_to_disk progress.
    calibration(config)
    progress = capsys.readouterr()
    assert not progress.out
    assert progress.err.isascii()
    assert "Calibration: Hessian tokens" in progress.err
    assert "Calibration: reconstruction tokens" in progress.err
    assert progress.err.count("100%") == 2
    payload = load_calibration(config)
    assert payload["hessian"].shape == (3, 8)
    assert payload["train"].shape == (2, 4)
    assert payload["validation"].shape == (1, 4)
    assert not payload["source"]["upstream_sample_identity_verified"]
    calibration(config)
    assert "reusing saved samples" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="protocol differs"):
        load_calibration(replace(config, sequence_length=5))
