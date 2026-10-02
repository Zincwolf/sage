"""Build resumable, chat-templated OPD self-rollout calibration via a vLLM server.

Uses /v1/models and /v1/completions with vLLM's return_token_ids extension.
No model weights or inference engine are loaded by this client. Completed requests
are committed individually to SQLite; rerunning the same recipe requests only gaps.
Greedy sampling fixes the recipe, not bitwise reproducibility across serving stacks.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from weight_vq import utils

QUOTA = {"code": 13, "math": 13, "reasoning": 13, "instruction": 10, "tool": 9, "general": 6}


@dataclass(frozen=True)
class OPDConfig:
    """Build shared SAGE/SPHQuant calibration from target-model OPD self-rollouts.

    Args:
        model: Local HF model directory, used only for its tokenizer. Must match
            the server's advertised model root; the served name is discovered.
        dataset: Directory containing train_clean_balanced.parquet and heldout_clean.parquet.
        base_url: vLLM OpenAI-compatible base URL, including /v1. Server managed separately.
        output_dir: Directory for rollout.sqlite3, preparation.json and calibration artifacts.
            Resume with the same recipe; completed calibration artifacts are not overwritten.
        train_samples: Number of packed training sequences.
        validation_samples: Number of packed held-out sequences.
        sequence_length: Tokens per packed sequence.
        train_prompts: Training prompt draw budget before template and length filtering.
        validation_prompts: Held-out prompt draw budget before filtering.
        max_new_tokens: Maximum generated tokens per prompt. Length-stopped answers are kept.
        max_model_len: Prompt plus completion context budget, also checked against the server.
        seed: Seed for domain-stratified sampling and generation.
        batch_size: Maximum number of in-flight prompt requests, not vLLM's internal
            scheduling batch. Completed requests are cached and replaced continuously.
        request_timeout: HTTP timeout in seconds, including waiting for generated responses.
    """

    model: Path
    dataset: Path
    base_url: str
    output_dir: Path
    train_samples: int = 1920
    validation_samples: int = 128
    sequence_length: int = 2048
    train_prompts: int = 16000
    validation_prompts: int = 1200
    max_new_tokens: int = 1024
    max_model_len: int = 4096
    seed: int = 0
    batch_size: int = 8
    request_timeout: float = 600.0


def sample_prompts(frame: Any, count: int, seed: int) -> list[dict[str, Any]]:
    """Preserve the reference pandas draw and index-hash interleaving, including rounding."""
    picked = []
    for domain, share in QUOTA.items():
        pool = frame[frame["ability"] == domain]
        taken = pool.sample(
            n=min(len(pool), round(count * share / sum(QUOTA.values()))), random_state=seed
        )
        for index, row in taken.iterrows():
            picked.append(
                {
                    "row": int(index),
                    "ability": domain,
                    "source": row["extra_info"]["source"],
                    "source_id": row["extra_info"]["source_id"],
                    "messages": [
                        {"role": str(turn["role"]), "content": str(turn["content"])}
                        for turn in row["prompt"]
                    ],
                }
            )
    # This hash is the reference ordering rule, not an integrity check.
    picked.sort(key=lambda item: hashlib.sha256(str(item["row"]).encode()).hexdigest())
    return picked


def pack(
    records: list[dict[str, Any]], blocks: int, sequence_length: int, eos: int
) -> tuple[Any, dict[str, Any]]:
    """Pack an ordered split, counting actual per-domain token contributions at the cut."""
    import torch

    needed = blocks * sequence_length
    flat: list[int] = []
    conversations: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for record in records:
        sequence = record["prompt_ids"] + record["token_ids"]
        if sequence[-1] != eos:
            sequence = [*sequence, eos]
        take = min(len(sequence), needed - len(flat))
        flat.extend(sequence[:take])
        conversations[record["ability"]] += 1
        tokens[record["ability"]] += take
        if len(flat) == needed:
            break
    if len(flat) < needed:
        raise RuntimeError(
            f"rolled out {len(flat)} tokens, {needed} needed; "
            "use a larger prompt budget in a new output directory"
        )
    return torch.tensor(flat, dtype=torch.int64).reshape(blocks, sequence_length), {
        "conversations": dict(conversations),
        "tokens": dict(tokens),
    }


def main(config: OPDConfig) -> None:
    # These are budget constraints, not a second parquet/tensor schema validator.
    if (
        min(
            config.train_samples,
            config.validation_samples,
            config.sequence_length,
            config.train_prompts,
            config.validation_prompts,
            config.batch_size,
            config.max_new_tokens,
        )
        < 1
        or config.max_new_tokens >= config.max_model_len
    ):
        raise ValueError("positive budgets required, with max_new_tokens < max_model_len")

    import httpx
    import pandas as pd
    from transformers import AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = config.output_dir / "calibration.pt"
    manifest_path = utils.manifest_path(artifact)
    # A single writer owns selection, rollout and publication. SQLite provides crash
    # recovery for committed responses; flock prevents two CLI runs sharing a recipe.
    with (
        (config.output_dir / ".lock").open("a") as lock,
        closing(sqlite3.connect(config.output_dir / "rollout.sqlite3")) as cache,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists():
            raise FileExistsError(f"completed calibration already exists: {manifest_path}")
        cache.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        cache.execute(
            "CREATE TABLE IF NOT EXISTS responses (position INTEGER PRIMARY KEY, value TEXT)"
        )
        recipe = asdict(config)
        for key in ("base_url", "batch_size", "request_timeout", "output_dir"):
            del recipe[key]
        recipe.update(
            model=str(config.model.resolve()),
            dataset=str(config.dataset.resolve()),
            version=1,
            domain_quota=QUOTA,
            temperature=0.0,
            thinking="auto",
        )
        saved = dict(cache.execute("SELECT key, value FROM metadata"))
        if "recipe" in saved and json.loads(saved["recipe"]) != recipe:
            raise ValueError("rollout recipe differs from cache; use a new output directory")

        tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
        selection = []
        preparation = {}
        for split, filename, count in (
            ("train", "train_clean_balanced.parquet", config.train_prompts),
            ("validation", "heldout_clean.parquet", config.validation_prompts),
        ):
            frame = pd.read_parquet(config.dataset / filename)
            prompts = sample_prompts(frame, count, config.seed)
            sampled: Counter[str] = Counter()
            kept: Counter[str] = Counter()
            too_long: Counter[str] = Counter()
            template_errors: Counter[str] = Counter()
            error_examples = []
            for prompt in prompts:
                domain = prompt["ability"]
                sampled[domain] += 1
                try:
                    ids = tokenizer.apply_chat_template(
                        prompt["messages"],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                except Exception as error:  # Reference filters template failures.
                    template_errors[domain] += 1
                    if len(error_examples) < 6:
                        error_examples.append({"row": prompt["row"], "error": str(error)})
                    continue
                if len(ids) + config.max_new_tokens > config.max_model_len:
                    too_long[domain] += 1
                    continue
                kept[domain] += 1
                selection.append({**prompt, "split": split, "prompt_ids": ids})
            preparation[split] = {
                "rows": len(frame),
                "domains": frame["ability"].value_counts().to_dict(),
                "sampled": dict(sampled),
                "kept": dict(kept),
                "too_long": dict(too_long),
                "template_errors": dict(template_errors),
                "error_examples": error_examples,
            }
        # Compare the actual selected conversations and token IDs, not filesystem hashes.
        # Batch size and endpoint may change on resume; the token recipe may not.
        if "selection" in saved and json.loads(saved["selection"]) != selection:
            raise ValueError(
                "sampled prompts/tokenizer differ from cache; use a new output directory"
            )
        with cache:
            cache.executemany(
                "INSERT OR IGNORE INTO metadata VALUES (?, ?)",
                [("recipe", json.dumps(recipe)), ("selection", json.dumps(selection))],
            )
        utils.atomic_write_text(
            config.output_dir / "preparation.json", json.dumps(preparation, indent=2) + "\n"
        )
        completed = dict(cache.execute("SELECT position, value FROM responses"))
        missing = [index for index in range(len(selection)) if index not in completed]
        print(
            f"prompts={len(selection)} cached={len(completed)} pending={len(missing)}", flush=True
        )

        # HTTPX's public synchronous client is thread-safe. Keep a bounded set of
        # futures and refill as requests finish, without waiting for a whole batch.
        # On failure, stop submitting and drain/cache the other in-flight responses.
        if missing:
            with (
                httpx.Client(
                    base_url=config.base_url.rstrip("/") + "/",
                    timeout=config.request_timeout,
                    limits=httpx.Limits(
                        max_connections=config.batch_size,
                        max_keepalive_connections=config.batch_size,
                    ),
                ) as client,
                ThreadPoolExecutor(max_workers=config.batch_size) as executor,
            ):
                response = client.get("models")
                response.raise_for_status()
                models = response.json()["data"]
                matches = [m for m in models if Path(m["root"]).resolve() == config.model.resolve()]
                if len(matches) != 1:
                    raise ValueError("server must advertise exactly one matching local model root")
                service = matches[0]
                if service["max_model_len"] < config.max_model_len:
                    raise ValueError("server context is smaller than max_model_len")
                with cache:
                    cache.execute(
                        "INSERT OR REPLACE INTO metadata VALUES ('service', ?)",
                        (json.dumps(service),),
                    )
                remaining = iter(missing)
                futures = {}
                errors = []
                exhausted = False
                while futures or not exhausted:
                    while not errors and not exhausted and len(futures) < config.batch_size:
                        index = next(remaining, None)
                        if index is None:
                            exhausted = True
                            break
                        future = executor.submit(
                            client.post,
                            "completions",
                            json={
                                "model": service["id"],
                                "prompt": selection[index]["prompt_ids"],
                                "temperature": 0.0,
                                "max_tokens": config.max_new_tokens,
                                "seed": config.seed,
                                "return_token_ids": True,
                                "add_special_tokens": False,
                            },
                        )
                        futures[future] = index
                    if not futures:
                        break
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    # Consume one completion, then refill its slot before handling
                    # the next; simultaneous completions must not form another barrier.
                    future = next(iter(done))
                    index = futures.pop(future)
                    try:
                        response = future.result()
                        response.raise_for_status()
                        choice = response.json()["choices"][0]
                        if (
                            choice.get("token_ids") is None
                            or choice.get("prompt_token_ids") != selection[index]["prompt_ids"]
                        ):
                            raise RuntimeError("server did not preserve/return token IDs")
                        if choice["finish_reason"] not in ("stop", "length"):
                            raise RuntimeError(
                                f"unexpected finish reason: {choice['finish_reason']}"
                            )
                        value = json.dumps(
                            {
                                "token_ids": choice["token_ids"],
                                "finish_reason": choice["finish_reason"],
                                "stop_reason": choice.get("stop_reason"),
                            }
                        )
                        with cache:
                            cache.execute("INSERT INTO responses VALUES (?, ?)", (index, value))
                        completed[index] = value
                    except Exception as error:  # Drain successful siblings before raising.
                        errors.append(error)
                        print(f"request failed at position={index}: {error}", flush=True)
                    if len(completed) % config.batch_size == 0 or not futures:
                        print(f"cached {len(completed)}/{len(selection)}", flush=True)
                if errors:
                    raise RuntimeError(
                        "rollout request failed; successful responses cached, rerun to resume"
                    ) from errors[0]

        tensors = {}
        statistics = {}
        for split, blocks in (
            ("train", config.train_samples),
            ("validation", config.validation_samples),
        ):
            records = [
                {**prompt, **json.loads(completed[index])}
                for index, prompt in enumerate(selection)
                if prompt["split"] == split
            ]
            tensor, packed = pack(records, blocks, config.sequence_length, tokenizer.eos_token_id)
            tensors[split] = tensor
            lengths = pd.Series([len(record["token_ids"]) for record in records])
            statistics[split] = {
                "rollouts": len(records),
                "finish_reasons": dict(Counter(record["finish_reason"] for record in records)),
                "empty_completions": sum(not record["token_ids"] for record in records),
                "completion_tokens": int(lengths.sum()),
                "completion_length_quantiles": lengths.quantile([0, 0.5, 0.9, 1]).to_dict(),
                "packed": packed,
            }
        manifest = {
            "tensors": {
                name: {"dtype": str(t.dtype), "shape": list(t.shape)} for name, t in tensors.items()
            },
            "provenance": {
                "recipe": recipe,
                "preparation": preparation,
                "service": json.loads(
                    dict(cache.execute("SELECT key, value FROM metadata"))["service"]
                ),
                "rollout": statistics,
            },
        }
        # Manifest is the completion marker. If publication is interrupted after the
        # tensor rename, the next run rebuilds from the committed cache without requests.
        utils.atomic_torch_save(tensors, artifact)
        utils.atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps(
                {"output": str(artifact), "tensors": manifest["tensors"], "rollout": statistics},
                indent=2,
            ),
            flush=True,
        )
