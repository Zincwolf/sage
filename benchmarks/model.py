"""Full-model FP16 / SDPA / StaticCache / torch.compile measurements.

Fixed-length greedy generation keeps token feedback on the GPU. TTFT measures
GPU readiness; TPOT uses CPU first-token arrival and whole-window completion.
Independent forward phases use fixed token IDs. No per-step host waits are added.
Compilation, cache allocation, and timing-buffer preparation are untimed.
As in sage-kernels, run with TORCHINDUCTOR_AUTOGRAD_CACHE=0 to avoid reusing
AOT graphs traced under an older SDPA policy. Shared cache files are not deleted.
"""

import csv
import gc
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast
from zoneinfo import ZoneInfo

import torch
from rich import box
from rich.console import Console
from rich.table import Table
from torch._functorch import config as aot_config
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache

from benchmarks.attention import profile_attention
from benchmarks.config import BenchmarkConfig
from weight_vq.inference.attention import SDPA_PRIORITY, preferred_sdpa
from weight_vq.integrations.awq import LLMAWQConfig
from weight_vq.integrations.transformers import WVQConfig, prepare_for_inference


@torch.inference_mode()
def measure_window(
    model,
    cache: StaticCache,
    prompt: torch.Tensor,
    tokens: list[torch.Tensor],
    phase: str,
    decode: int,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """Measure one window; return generated CPU IDs only for end_to_end.

    Decode-only is teacher-forced forward timing, not autoregressive generation.
    The first-token observer waits on a copy event in a background thread, never
    in the generation loop. CPU timestamps include observation/scheduling delay.
    """
    cache.reset()
    if phase == "decode":
        model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if phase != "end_to_end":
        start.record()
        end.record()
        torch.cuda.synchronize(prompt.device)
        wall_start = perf_counter()
        start.record()
        for input_ids in tokens if phase == "decode" else [prompt]:
            model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        end.record()
        end.synchronize()
        wall_ms = (perf_counter() - wall_start) * 1000
        return {"cuda_event_ms": start.elapsed_time(end), "synchronized_wall_ms": wall_ms}, None

    generated = torch.empty((prompt.shape[0], decode), dtype=prompt.dtype, device=prompt.device)
    generated_cpu = torch.empty(generated.shape, dtype=prompt.dtype, pin_memory=True)
    first_cpu = torch.empty((prompt.shape[0], 1), dtype=prompt.dtype, pin_memory=True)
    copy_stream = torch.cuda.Stream(device=prompt.device)
    first_ready = torch.cuda.Event(enable_timing=True)
    first_copied = torch.cuda.Event()

    def observe_first_token() -> float:
        first_copied.synchronize()
        return perf_counter()

    with ThreadPoolExecutor(max_workers=1) as observer:
        # Start the worker and materialize CUDA events outside the timer.
        observer.submit(perf_counter).result()
        first_ready.record()
        first_copied.record()
        start.record()
        end.record()
        torch.cuda.synchronize(prompt.device)

        wall_start = perf_counter()
        start.record()
        output = model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        del output
        first_ready.record()
        # Hold the source alive until both streams finish; later tokens are
        # fresh argmax outputs, independent of compiled forward output storage.
        first_token = next_token
        with torch.cuda.stream(copy_stream):
            first_ready.wait()
            first_cpu.copy_(first_token, non_blocking=True)
            first_copied.record()
        first_arrival = observer.submit(observe_first_token)
        generated[:, :1].copy_(next_token)
        for position in range(1, decode):
            output = model(next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            del output
            generated[:, position : position + 1].copy_(next_token)
        # Deliver the complete output once, not once per token.
        generated_cpu.copy_(generated, non_blocking=True)
        end.record()

        # All main-thread waits are outside the forward/generation loop. Observe
        # first arrival before final completion, including the one-token case.
        first_stamp = first_arrival.result()
        end.synchronize()
        wall_ms = (perf_counter() - wall_start) * 1000
        measurement = {
            "cuda_event_ms": start.elapsed_time(end),
            "synchronized_wall_ms": wall_ms,
            "gpu_ttft_ms": start.elapsed_time(first_ready),
            "first_token_cpu_ms": (first_stamp - wall_start) * 1000,
            "actual_generated_tokens": generated_cpu.numel(),
        }
    return measurement, generated_cpu


def inspect_model_attention(
    model, cache, prompt, tokens, phase: str, decode: int
) -> dict[str, Any]:
    """Inspect warmed prefill/decode specializations without timing the inspection."""
    cache.reset()
    reports = {}

    def inspect(input_ids):
        return profile_attention(
            lambda: model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1),
            expected_layers=model.config.num_hidden_layers,
            device=prompt.device,
        )

    if phase == "decode":
        model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        if tokens:
            output, reports["decode"] = inspect(tokens[0])
            del output
    else:
        output, reports["prefill"] = inspect(prompt)
        if phase == "end_to_end" and decode > 1:
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            del output
            output, reports["decode"] = inspect(next_token)
        del output
    return reports


@preferred_sdpa()
@torch.inference_mode()
def benchmark(
    config: BenchmarkConfig,
) -> tuple[dict[str, Any], list[tuple[str, str, str, float, str]]]:
    torch.cuda.set_device(config.device)
    properties = torch.cuda.get_device_properties(config.device)
    # Source checkouts have a Git revision; installed wheels retain package version.
    root = Path(__file__).resolve().parents[1]
    revision = None
    if (root / ".git").exists():
        revision = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    result: dict[str, Any] = {
        "config": asdict(config),
        "python_version": platform.python_version(),
        "weight_vq_version": version("weight-vq"),
        "source_revision": revision,
        "execution": {
            "activation_dtype": "float16",
            "kv_cache_dtype": "float16",
            "attention_implementation": "sdpa",
            "cache": "StaticCache",
            "compile": {"fullgraph": True, "dynamic": False, "mode": "reduce-overhead"},
            "autograd_cache_enabled": aot_config.enable_autograd_cache,
        },
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "attention_backend_priority": [backend.name for backend in SDPA_PRIORITY],
        "hardware": {
            "name": properties.name,
            "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
            "sm_count": properties.multi_processor_count,
            "l2_bytes": properties.L2_cache_size,
            "memory_bytes": properties.total_memory,
        },
        "methods": {},
    }
    rows = []
    for method, model_dir in config.models.items():
        torch.cuda.reset_peak_memory_stats(config.device)
        model = cache = None
        try:
            hf_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
            saved_quantization = getattr(hf_config, "quantization_config", {})
            if saved_quantization.get("quant_method") == "awq":
                # Select our registered runtime without modifying the standard AWQ
                # checkpoint or invoking GPTQModel's non-fullgraph inference path.
                hf_config.quantization_config = LLMAWQConfig(**saved_quantization).to_dict()
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                config=hf_config,
                local_files_only=True,
                dtype=torch.float16,
                device_map=config.device,
                attn_implementation="sdpa",
            ).eval()
            # HF replaces packed skeletons before loading. Explicit ABI buffer
            # dtypes survive dtype=float16; do not cast the whole loaded model.
            # Cache preparation is untimed, including for GEMV-only prompts.
            prepare_for_inference(model, cached_dense_weight=config.cached_dense_weight)
            quantization = getattr(model.config, "quantization_config", None)
            ids = torch.randint(
                3,
                model.config.vocab_size,
                (config.batch_size, config.prefill + config.decode - 1),
                generator=torch.Generator().manual_seed(config.seed),
            ).to(config.device)
            prompt = ids[:, : config.prefill].contiguous()
            tokens = [
                ids[:, pos : pos + 1].contiguous() for pos in range(config.prefill, ids.shape[1])
            ]
            cache = StaticCache(config=model.config, max_cache_len=ids.shape[1])
            # Initialize native cache storage outside compilation, including its static addresses.
            model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache.reset()
            # Experiment shapes are fixed; prefill and decode get separate specializations.
            cast(torch.nn.Module, model).compile(
                fullgraph=True, dynamic=False, mode="reduce-overhead"
            )
            method_result: dict[str, Any] = {
                "method": (
                    quantization.method
                    if isinstance(quantization, WVQConfig)
                    else "awq"
                    if isinstance(quantization, LLMAWQConfig)
                    else "dense"
                ),
                "synthetic_seed": (
                    quantization.synthetic_seed if isinstance(quantization, WVQConfig) else None
                ),
                # Logical file bytes of the self-contained HF directory, not VRAM.
                "checkpoint_directory_bytes": sum(
                    path.stat().st_size for path in model_dir.rglob("*") if path.is_file()
                ),
                "model_bytes": model.get_memory_footprint(),
                "samples": {phase: [] for phase in config.phases},
                "attention_backends": {},
            }
            if isinstance(quantization, LLMAWQConfig):
                method_result["quantization"] = {
                    "bits": quantization.bits,
                    "group_size": quantization.group_size,
                    "format": quantization.format,
                    "backend": "wvq::awq",
                }
            for phase in config.phases:
                # First two passes prepare compilation and CUDA Graphs, before requested warmup.
                for sample in tqdm(
                    range(-config.warmup - 2, config.samples),
                    desc=f"{method}/{phase}",
                    file=sys.stderr,
                    unit="run",
                ):
                    with torch.compiler.set_stance(
                        "fail_on_recompile" if sample >= 0 else "default"
                    ):
                        measurement, _ = measure_window(
                            model, cache, prompt, tokens, phase, config.decode
                        )
                    if sample == -config.warmup - 1:
                        method_result["attention_backends"][phase] = inspect_model_attention(
                            model, cache, prompt, tokens, phase, config.decode
                        )
                        # Re-prime after profiling even when the requested warmup is zero.
                        measure_window(model, cache, prompt, tokens, phase, config.decode)
                    if sample >= 0:
                        method_result["samples"][phase].append(measurement)
                samples = method_result["samples"][phase]
                metrics = []
                times = [sample["synchronized_wall_ms"] for sample in samples]
                if phase == "decode":
                    steps = len(tokens)
                    if steps:
                        metrics.extend(
                            [
                                ("forward_tpot", median(times) / steps, "ms/token"),
                                (
                                    "output_throughput",
                                    config.batch_size * steps * len(times) * 1000 / sum(times),
                                    "tokens/s",
                                ),
                            ]
                        )
                else:
                    metrics.append(
                        (
                            "forward_latency" if phase == "prefill" else "generate_latency",
                            median(times),
                            "ms",
                        )
                    )
                    metrics.append(
                        (
                            "input_throughput" if phase == "prefill" else "output_throughput",
                            config.batch_size
                            * (config.prefill if phase == "prefill" else config.decode)
                            * len(times)
                            * 1000
                            / sum(times),
                            "tokens/s",
                        )
                    )
                    if phase == "end_to_end":
                        metrics.append(
                            ("ttft", median(sample["gpu_ttft_ms"] for sample in samples), "ms")
                        )
                        if config.decode > 1:
                            metrics.append(
                                (
                                    "tpot",
                                    median(
                                        (
                                            sample["synchronized_wall_ms"]
                                            - sample["first_token_cpu_ms"]
                                        )
                                        / (config.decode - 1)
                                        for sample in samples
                                    ),
                                    "ms/token",
                                )
                            )
                rows.extend((method, phase, metric, value, unit) for metric, value, unit in metrics)
            method_result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(config.device)
            method_result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(config.device)
            result["methods"][method] = method_result
        finally:
            del model, cache
            torch.compiler.reset()
            gc.collect()
            torch.cuda.empty_cache()

    return result, rows


def main(config: BenchmarkConfig) -> None:
    """Write raw JSON and summary CSV; keep diagnostics separate from the table."""
    with redirect_stdout(sys.stderr):
        result, rows = benchmark(config)
    directory = config.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d-%H%M%S}-{config.name}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    csv_path = path.with_suffix(".csv")
    formatted = [
        (method, phase, metric, f"{value:.3f}", unit) for method, phase, metric, value, unit in rows
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "phase", "metric", "value", "unit"))
        writer.writerows(formatted)
    print(path, file=sys.stderr)
    print(csv_path, file=sys.stderr)

    table = Table("method", "phase", "metric", box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("value", justify="right", no_wrap=True)
    table.add_column("unit")
    for row in formatted:
        table.add_row(*row)
    Console().print(table)
