"""QTIP packed decoding and the Hadamard transforms used by its Linear ABI."""

import torch

from weight_vq.inference.ops import hadamard

L = 16
R = 2
V = 2
LOG2_V = 1
TLUT_BITS = 9


HADAMARD_20 = torch.tensor(
    [
        [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1],
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1],
        [1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1],
        [1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1],
        [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1],
        [1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1],
        [1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1],
        [1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1],
        [1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1],
        [1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
        [1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
        [1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1],
        [1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1],
        [1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1],
    ],
    dtype=torch.float32,
)


def hadamard_transform(
    x: torch.Tensor, matrix: torch.Tensor | None = None, *, transpose: bool = False
) -> torch.Tensor:
    """Upstream QTIP transform; the caller owns the device-resident H20 buffer.

    Do not pass 2560/9728 directly to FHT: its implicit zero padding is not
    QTIP's H20 x H128 / block512 transform. H20 is not symmetric.
    """
    size = x.shape[-1]
    if size == 2560:
        assert matrix is not None
        blocks = hadamard(x.float().reshape(-1, 20, 128).contiguous(), size**-0.5)
        if transpose:
            matrix = matrix.T.contiguous()
        return (matrix @ blocks).to(x.dtype).reshape_as(x)
    if size == 9728:
        blocks = x.reshape(*x.shape[:-1], -1, 512).contiguous()
        return hadamard(blocks, 512**-0.5).reshape_as(x)
    if size > 0 and size & (size - 1) == 0:
        return hadamard(x.contiguous(), size**-0.5)
    raise ValueError(f"Unsupported QTIP Hadamard dimension: {size}")


def expanded_lut(tlut: torch.Tensor) -> torch.Tensor:
    """Upstream ``quantlut_sym``: map every 16-bit state to one LUT row and sign."""
    states = torch.arange(1 << L, device=tlut.device)
    squared = (states + 1) * states
    sign = 1 - ((squared >> 15) & 1) * 2
    lut = tlut[(squared >> (16 - TLUT_BITS - 1)) & ((1 << TLUT_BITS) - 1)].clone()
    lut[:, 0] = lut[:, 0] * sign
    return lut


def decode_dense(
    trellis: torch.Tensor, tlut: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    """Decode packed QTIP trellis into raw FP16 weights ``[N, K]``.

    The diagonal scales and the two Hadamard transforms belong to QTIP Linear,
    so callers that need the actual QDQ weight must apply those there.
    """
    m, k = out_features, in_features
    torch._check(m % 32 == 0 and k % 32 == 0, lambda: "QTIP dimensions must be divisible by 32")
    torch._check(tlut.shape == (1 << TLUT_BITS, V), lambda: "QTIP LUT shape mismatch")
    torch._check(trellis.numel() * 16 == R * m * k, lambda: "QTIP trellis shape mismatch")
    compressed = trellis.reshape(-1).view(torch.uint16)
    bits_per_block = R * 16 * 16
    compressed = (
        compressed.view(torch.uint8)
        .reshape(m // 16 // 2, k // 16 // 2, 16 * 16 // 8, 2, 2, R)
        .permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(m // 16, k // 16, bits_per_block // 16, 2)
        .flip((-1,))
        .view(torch.uint16)
        .reshape(m // 16, k // 16, bits_per_block // 16)
    )
    blocked = compressed.reshape(R * m * k // bits_per_block, bits_per_block // 16, 1)
    rolled = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    paired = (
        torch.cat((rolled, blocked), dim=-1)
        .reshape(blocked.shape[0], -1)
        .contiguous()
        .view(torch.uint32)
    )
    expanded = paired.reshape(*paired.shape, 1).expand(*paired.shape, 16).view(torch.int32)
    shifts = torch.arange(16, dtype=torch.int32, device=blocked.device).reshape(1, 1, 16)
    indices = torch.bitwise_and(
        (expanded >> (16 - shifts)).reshape(blocked.shape[0], -1)[:, 16 - L :: R << LOG2_V],
        (1 << L) - 1,
    )
    return (
        expanded_lut(tlut)[indices]
        .reshape(m // 16, k // 16, 16, 16)
        .reshape(m // 16, k // 16, 8, 4, 2, 2, 2)
        .permute(0, -2, 2, 1, -3, 3, -1)
        .reshape(m, k)
    )
