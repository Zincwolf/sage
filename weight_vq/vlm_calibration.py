"""Build the fixed ShareGPT4V/COCO calibration manifest for VLM pipelines."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from weight_vq.config import VLMCalibrationConfig

REPO_ID = "Lin-Chen/ShareGPT4V"
REVISION = "55d02b0bc53a2754095a14110dda6daedd95671d"
ANNOTATION_FILE = "sharegpt4v_instruct_gpt4-vision_cap100k.json"
COCO_URL = "http://images.cocodataset.org/train2017/{image_id}.jpg"
SAMPLES = 2048
VALIDATION_SAMPLES = 128
SEED = 0
MANIFEST_NAME = "coco_calibration_manifest.json"


def _image_id(item: dict[str, Any]) -> str:
    return str(item["id"]).zfill(12)


def _verify_image(path: Path) -> bool:
    if not path.is_file():
        return False
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def _record(item: dict[str, Any], image_id: str) -> dict[str, Any]:
    return {
        "id": image_id,
        "image": str(Path("coco") / "train2017" / f"{image_id}.jpg"),
        "conversations": item["conversations"],
    }


def _download_image(item: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    image_id = _image_id(item)
    destination = image_dir / f"{image_id}.jpg"
    if _verify_image(destination):
        return _record(item, image_id)
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(".jpg.part")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                COCO_URL.format(image_id=image_id),
                headers={"User-Agent": "weight-vq/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            if not _verify_image(temporary):
                raise RuntimeError(f"downloaded invalid COCO image {image_id}")
            os.replace(temporary, destination)
            return _record(item, image_id)
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download COCO image {image_id}") from last_error


def _extract_images(
    selected: list[dict[str, Any]],
    archive: Path,
    image_dir: Path,
) -> list[dict[str, Any]]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    records = []
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        for item in selected:
            image_id = _image_id(item)
            destination = image_dir / f"{image_id}.jpg"
            if not _verify_image(destination):
                member = f"train2017/{image_id}.jpg"
                if member not in names:
                    raise FileNotFoundError(f"{member} is absent from {archive}")
                temporary = destination.with_suffix(".jpg.part")
                try:
                    temporary.write_bytes(source.read(member))
                    if not _verify_image(temporary):
                        raise RuntimeError(f"archive contains invalid COCO image {image_id}")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            records.append(_record(item, image_id))
    return records


def _select_annotations(
    annotations: list[dict[str, Any]],
    *,
    samples: int = SAMPLES,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in annotations
        if str(item.get("image", "")).startswith("coco/train2017/")
    ]
    if len(candidates) < samples:
        raise ValueError(f"need {samples} COCO annotations, found {len(candidates)}")
    random.Random(seed).shuffle(candidates)
    return candidates[:samples]


def _manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_at = len(records) - VALIDATION_SAMPLES
    if split_at <= 0:
        raise ValueError("validation sample count must be smaller than total samples")
    completed = []
    for index, source in enumerate(records):
        record = dict(source)
        record["split"] = "train" if index < split_at else "validation"
        record["sample_index"] = index
        completed.append(record)
    return {
        "format": "sharegpt4v-coco-caption-calibration-v1",
        "source_repo": REPO_ID,
        "resolved_revision": REVISION,
        "source_file": ANNOTATION_FILE,
        "seed": SEED,
        "samples": len(completed),
        "train_samples": split_at,
        "validation_samples": VALIDATION_SAMPLES,
        "records": completed,
    }


def main(config: VLMCalibrationConfig) -> None:
    if config.workers <= 0:
        raise ValueError("workers must be positive")
    output = config.output_dir / MANIFEST_NAME
    if output.exists() and not config.force:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    from huggingface_hub import HfApi, hf_hub_download

    resolved_revision = HfApi().dataset_info(REPO_ID, revision=REVISION).sha
    if resolved_revision != REVISION:
        raise RuntimeError(
            f"ShareGPT4V revision resolved to {resolved_revision}, expected {REVISION}"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=ANNOTATION_FILE,
            repo_type="dataset",
            revision=resolved_revision,
            local_dir=config.output_dir,
        )
    )
    annotations = json.loads(annotation_path.read_text())
    selected = _select_annotations(annotations, samples=SAMPLES, seed=SEED)
    image_dir = config.output_dir / "coco" / "train2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    if config.image_archive is None:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            records = list(pool.map(lambda item: _download_image(item, image_dir), selected))
    else:
        records = _extract_images(selected, config.image_archive, image_dir)

    manifest = _manifest(records)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "format",
                    "samples",
                    "train_samples",
                    "validation_samples",
                )
            },
            indent=2,
        )
    )
