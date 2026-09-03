"""Qwen3-VL calibration batches backed by the frozen COCO manifest."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


def _conversation(record: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
    prompt = record["conversations"][0]["value"].replace("<image>", "").strip()
    caption = record["conversations"][1]["value"].strip()
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": caption}]},
    ]


def local_batch_slice(size: int, rank: int, world_size: int) -> slice:
    local_size, remainder = divmod(size, world_size)
    if remainder:
        raise ValueError(f"global batch of {size} samples is not divisible by {world_size} ranks")
    start = rank * local_size
    return slice(start, start + local_size)


class CocoCalibrationBatches:
    def __init__(
        self,
        manifest_path: Path,
        *,
        model_path: Path,
        split: str,
        batch_size: int,
        limit: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        manifest = json.loads(manifest_path.read_text())
        self.records = []
        for source in manifest["records"]:
            if source["split"] != split:
                continue
            record = dict(source)
            image = Path(record["image"])
            record["image"] = str(image if image.is_absolute() else manifest_path.parent / image)
            self.records.append(record)
        self.records = self.records[:limit]
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        template = getattr(self.processor, "chat_template", None)
        if template is None and tokenizer is not None:
            template = getattr(tokenizer, "chat_template", None)
        self.template_kwargs = (
            {"enable_thinking": False} if template and "enable_thinking" in template else {}
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for start in range(0, len(self.records), self.batch_size):
            global_records = self.records[start : start + self.batch_size]
            records = global_records[
                local_batch_slice(len(global_records), self.rank, self.world_size)
            ]
            images = [Image.open(record["image"]).convert("RGB") for record in records]
            try:
                conversations = [
                    _conversation(record, image)
                    for record, image in zip(records, images, strict=True)
                ]
                batch = self.processor.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=False,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt",
                    **self.template_kwargs,
                )
            finally:
                for image in images:
                    image.close()
            yield {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
