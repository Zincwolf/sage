"""SDPA policy and untimed verification of the kernels actually executed."""

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch._C._autograd import DeviceType


def summarize_attention_kernels(names: Iterable[str], expected_layers: int) -> dict[str, Any]:
    """Recognize pinned CUDA implementations conservatively, including graph replay.

    Compiled math attention can be indistinguishable from ordinary model GEMMs.
    Missing or incomplete evidence is unknown, never proof of cuDNN selection.
    """
    kernels: Counter[str] = Counter()
    backends: Counter[str] = Counter()
    for name in names:
        lowered = name.lower()
        if "cudnn" in lowered and ("sdpa" in lowered or "attention" in lowered):
            backend = "CUDNN_ATTENTION"
        elif "fmha_cutlass" in lowered:
            backend = "EFFICIENT_ATTENTION"
        elif "flash" in lowered and ("flash_fwd" in lowered or "flash::" in lowered):
            backend = "FLASH_ATTENTION"
        else:
            continue
        # Triton fusion names may mention an SDPA op without executing attention.
        if lowered.startswith("triton"):
            continue
        kernels[name] += 1
        backends[backend] += 1
    verified = expected_layers > 0 and dict(backends) == {"CUDNN_ATTENTION": expected_layers}
    return {
        "preferred_backend": "CUDNN_ATTENTION",
        "observed_backends": sorted(backends) or ["unknown"],
        "kernel_counts": dict(backends),
        "kernel_names": dict(kernels),
        "expected_attention_layers": expected_layers,
        "preferred_backend_verified": verified,
    }


def profile_attention(
    forward: Callable[[], Any], *, expected_layers: int, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    """Inspect one warmed forward outside timing; do not infer selection from flags."""
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        output = forward()
        torch.cuda.synchronize(device)
    report = summarize_attention_kernels(
        (event.name for event in profile.events() or [] if event.device_type == DeviceType.CUDA),
        expected_layers,
    )
    return output, report
