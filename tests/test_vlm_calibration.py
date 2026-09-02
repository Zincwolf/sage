import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from weight_vq import vlm_calibration
from weight_vq.config import VLMCalibrationConfig


class FakeImage:
    def __enter__(self) -> "FakeImage":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def verify(self) -> None:
        return None


def install_fake_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    image_module = ModuleType("PIL.Image")
    image_module.open = lambda _path: FakeImage()  # type: ignore[attr-defined]
    pil_module = ModuleType("PIL")
    pil_module.Image = image_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)


def annotations(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "image": f"coco/train2017/{index:012d}.jpg",
            "conversations": [
                {"from": "human", "value": "Describe. <image>"},
                {"from": "gpt", "value": f"Image {index}."},
            ],
        }
        for index in range(count)
    ]


def test_select_annotations_filters_and_is_deterministic() -> None:
    source = [*annotations(5), {"id": "other", "image": "vg/other.jpg", "conversations": []}]

    first = vlm_calibration._select_annotations(source, samples=3, seed=7)
    second = vlm_calibration._select_annotations(source, samples=3, seed=7)

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert all(str(item["image"]).startswith("coco/train2017/") for item in first)


def test_manifest_uses_reader_compatible_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_calibration, "VALIDATION_SAMPLES", 1)
    records = [
        vlm_calibration._record(item, vlm_calibration._image_id(item)) for item in annotations(3)
    ]

    manifest = vlm_calibration._manifest(records)

    assert [record["split"] for record in manifest["records"]] == ["train", "train", "validation"]
    assert [record["sample_index"] for record in manifest["records"]] == [0, 1, 2]
    assert manifest["records"][0]["image"] == "coco/train2017/000000000000.jpg"
    assert set(manifest["records"][0]) == {"id", "image", "conversations", "split", "sample_index"}


def test_extract_images_reads_only_selected_archive_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_pillow(monkeypatch)
    archive = tmp_path / "train2017.zip"
    image = b"fake-jpeg"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("train2017/000000000001.jpg", image)
        output.writestr("train2017/000000000002.jpg", image)
        output.writestr("train2017/unselected.jpg", b"unused")
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    records = vlm_calibration._extract_images(annotations(3)[1:], archive, image_dir)

    assert [record["id"] for record in records] == ["000000000001", "000000000002"]
    assert (image_dir / "000000000001.jpg").read_bytes() == image
    assert not (image_dir / "unselected.jpg").exists()
    assert not list(image_dir.glob("*.part"))


def test_download_reuses_valid_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_fake_pillow(monkeypatch)
    destination = tmp_path / "000000000001.jpg"
    destination.write_bytes(b"existing")
    monkeypatch.setattr(
        vlm_calibration.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("valid image should be reused"),
    )

    record = vlm_calibration._download_image(annotations(2)[1], tmp_path)

    assert record["id"] == "000000000001"
    assert destination.read_bytes() == b"existing"


def test_download_replaces_invalid_image_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def verify(path: Path) -> bool:
        return path.read_bytes() == b"valid" if path.is_file() else False

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(vlm_calibration, "_verify_image", verify)
    monkeypatch.setattr(
        vlm_calibration.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b"valid")
    )
    destination = tmp_path / "000000000001.jpg"
    destination.write_bytes(b"broken")

    record = vlm_calibration._download_image(annotations(2)[1], tmp_path)

    assert record["id"] == "000000000001"
    assert destination.read_bytes() == b"valid"
    assert not (tmp_path / "000000000001.jpg.part").exists()


def test_main_uses_pinned_hf_revision_and_writes_manifest_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotations(3)))
    received: dict[str, object] = {}

    class Api:
        def dataset_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            received["dataset_info"] = (repo_id, revision)
            return SimpleNamespace(sha=revision)

    def download(**kwargs: object) -> str:
        received["download"] = kwargs
        return str(annotation_path)

    hub = ModuleType("huggingface_hub")
    hub.HfApi = Api  # type: ignore[attr-defined]
    hub.hf_hub_download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(vlm_calibration, "SAMPLES", 3)
    monkeypatch.setattr(vlm_calibration, "VALIDATION_SAMPLES", 1)
    monkeypatch.setattr(
        vlm_calibration,
        "_download_image",
        lambda item, _directory: vlm_calibration._record(item, vlm_calibration._image_id(item)),
    )
    output_dir = tmp_path / "output"

    vlm_calibration.main(VLMCalibrationConfig(output_dir=output_dir, workers=2))

    manifest = json.loads((output_dir / vlm_calibration.MANIFEST_NAME).read_text())
    assert received["dataset_info"] == (vlm_calibration.REPO_ID, vlm_calibration.REVISION)
    assert received["download"] == {
        "repo_id": vlm_calibration.REPO_ID,
        "filename": vlm_calibration.ANNOTATION_FILE,
        "repo_type": "dataset",
        "revision": vlm_calibration.REVISION,
        "local_dir": output_dir,
    }
    assert manifest["train_samples"] == 2
    assert manifest["validation_samples"] == 1
    assert not list(output_dir.glob("*.tmp-*"))


def test_main_refuses_existing_manifest_before_network(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output = output_dir / vlm_calibration.MANIFEST_NAME
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        vlm_calibration.main(VLMCalibrationConfig(output_dir=output_dir))

    assert output.read_text() == "existing"
