"""Shared SDPA policy for native HF inference and benchmarks.

Use preferred_sdpa() around warmup/compilation and subsequent forward or
model.generate calls. It restores the caller's policy on exit; importing this
module does not change global settings. PyTorch handles backend eligibility and
fallback, without a project-specific dispatcher or mandatory profiler.
For compiled inference, the upstream runtime convention is
TORCHINDUCTOR_AUTOGRAD_CACHE=0: previously cached AOT graphs can retain an older
SDPA selection. This context does not change compiler settings or clear caches.
"""

from torch.nn.attention import SDPBackend, sdpa_kernel

SDPA_PRIORITY = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def preferred_sdpa():
    """Prefer cuDNN, retaining native Flash/Efficient/Math fallbacks."""
    return sdpa_kernel(list(SDPA_PRIORITY), set_priority=True)
