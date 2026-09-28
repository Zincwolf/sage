"""Independent arithmetic decoding of the checkpoint, never imported by inference."""

import itertools

import torch


def unpack_sphquant(codes: torch.Tensor, extra: torch.Tensor) -> tuple[torch.Tensor, ...]:
    words = codes.long().remainder(65536)
    high_words = extra.long().remainder(4294967296)
    signs = torch.stack([1 - 2 * ((words // (2**i)) % 2) for i in range(8)], dim=-1)
    high = torch.stack([(high_words // (4**i)) % 4 for i in range(16)], dim=-1)
    return signs, (words // 256) % 64, words // 16384 + 4 * high.flatten(1)


def decode_sphquant(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    signs, indices, codes = unpack_sphquant(packed["packed_codes"], packed["radius_extra"])
    directions = packed["codebook"].float()[indices]
    radii = packed["radius_mins"].float().repeat_interleave(16, dim=1) + codes.float() * packed[
        "radius_scales"
    ].float().repeat_interleave(16, dim=1)
    return (signs.float() * directions * radii.unsqueeze(-1)).flatten(1)


def unpack_sage(packed: dict[str, torch.Tensor], n: int) -> tuple[torch.Tensor, ...]:
    tiled = packed["packed_codes"].long().remainder(2**32)
    words = tiled.permute(0, 2, 1, 3).reshape(-1, tiled.shape[1], 8)[:n]
    words = torch.stack((words.remainder(65536), words // 65536), dim=-1).flatten(2)
    signs = torch.stack([1 - 2 * ((words // 2**i) % 2) for i in range(8)], dim=-1)
    radius_words = packed["radius_codes"].long().remainder(2**32)
    radius_words = radius_words.permute(0, 2, 1).reshape(-1, tiled.shape[1])[:n]
    entries = packed["codebook"].shape[1]
    radii = words // (256 * entries) + (256 // entries) * torch.stack(
        [(radius_words // 4**i) % 4 for i in range(16)], dim=-1
    )
    return signs, (words // 256) % entries, radii


def decode_sage(packed: dict[str, torch.Tensor], n: int) -> torch.Tensor:
    signs, indices, codes = unpack_sage(packed, n)
    book = packed["codebook"].permute(1, 0, 2).reshape(-1, 8).float()
    grid = packed["radius_grid"].permute(0, 2, 1, 3).reshape(-1, indices.shape[1], 2)[:n]
    radii = grid[..., 0, None].float() + codes.float() * grid[..., 1, None].float()
    weight = (signs * book[indices] * radii[..., None]).flatten(1)
    if packed["permutation"].numel():
        original = torch.empty_like(weight)
        original[:, packed["permutation"].long()] = weight
        return original
    return weight


def unpack_e8p(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Read each absolute index and interleaved sign bit by its tile coordinates."""
    words = packed["Qidxs"]
    n, k = words.shape[0], words.shape[1] * 32
    tiles = words.reshape(n // 16, k // 64, 8, 4)
    indices = torch.empty(n // 16, 16, k // 64, 8, dtype=torch.int64)
    for row_pair, column_pair, lane in itertools.product(range(8), range(4), range(4)):
        word = tiles[:, :, row_pair, column_pair]
        absolute = (word >> (8 * lane)) & 255
        signs = sum(((word >> (32 + 4 * bit + lane)) & 1) * 2**bit for bit in range(8))
        indices[:, 2 * row_pair + lane % 2, :, 2 * column_pair + lane // 2] = absolute * 256 + signs
    return indices.reshape(n, k // 8)


def decode_e8p(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Independent lattice arithmetic, without the production packed codebook."""
    absolute = torch.tensor(list(itertools.product((0.5, 1.5, 2.5, 3.5), repeat=8)))
    absolute = absolute[absolute.square().sum(-1) <= 10]
    # The 29 norm-12 entries are part of the E8P12 specification, in this order.
    norm12 = (
        "31113333",
        "13113333",
        "11313333",
        "11133333",
        "33313311",
        "33313131",
        "33311331",
        "33313113",
        "33311313",
        "33311133",
        "33133311",
        "33133131",
        "33131331",
        "33133113",
        "33131313",
        "33131133",
        "31333311",
        "31333131",
        "31331331",
        "31333113",
        "31331313",
        "13331133",
        "13333311",
        "13333131",
        "13331331",
        "13333113",
        "13331313",
        "11331333",
        "33113331",
    )
    absolute = torch.cat((absolute, torch.tensor([[int(c) / 2 for c in s] for s in norm12])))
    assert absolute.shape == (256, 8)
    absolute[:, 7] *= 1 - 2 * absolute.sum(-1).remainder(2)
    indices = unpack_e8p(packed)
    sign_byte = indices.remainder(256)
    parity = torch.stack([(sign_byte // 2**bit) % 2 for bit in range(8)]).sum(0) % 2
    sign_byte = sign_byte ^ parity
    signs = torch.stack(
        [1 - 2 * ((sign_byte // 2**bit) % 2) for bit in (0, 4, 1, 5, 2, 6, 3, 7)], -1
    )
    return (absolute[indices // 256] * signs + (0.25 - 0.5 * parity)[..., None]).flatten(1)


def unpack_qtip(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Read circular 16-bit states from the R=2 kernel-interleaved byte layout."""
    n, k = packed["SV"].numel(), packed["SU"].numel()
    data = packed["trellis"].view(torch.uint8).flatten().long()
    row = torch.arange(n // 16)[:, None, None]
    col = torch.arange(k // 16)[None, :, None]
    byte = torch.arange(64)[None, None, :]
    address = (
        (((row // 2 * (k // 32) + col // 2) * 32 + byte // 2) * 2 + col % 2) * 2 + row % 2
    ) * 2 + (1 - byte % 2)
    stream = data[address]
    starts = torch.arange(128) * 4
    states = torch.zeros(n // 16, k // 16, 128, dtype=torch.int64)
    for bit in range(16):
        position = (starts + bit) % 512
        states = states * 2 + ((stream[..., position // 8] >> (7 - position % 8)) & 1)
    return states


def decode_qtip(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Independent state hash and MMA coordinate mapping; returns raw FP32 W."""
    states = unpack_qtip(packed)
    hashed = states * (states + 1)
    values = packed["tlut"].float()[(hashed // 64) % 512].clone()
    values[..., 0] *= 1 - 2 * ((hashed // 32768) % 2)
    n, k = packed["SV"].numel(), packed["SU"].numel()
    raw = torch.empty(n // 16, 16, k // 16, 16)
    for scalar in range(256):
        row = (scalar // 2) % 2 * 8 + scalar // 32
        col = (scalar // 4) % 2 * 8 + (scalar // 8) % 4 * 2 + scalar % 2
        raw[:, row, :, col] = values[..., scalar // 2, scalar % 2]
    return raw.reshape(n, k)


def qtip_hadamard_reference(x: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    """Independent FP32 transform, with Paley-generated (not copied) H20.

    No production transforms or CUDA extension calls. The Sylvester factors
    are applied by a descending-axis butterfly rather than FHT's CUDA schedule.
    """
    size = x.shape[-1]
    if size == 2560:
        factor, width = 20, 128
    elif size == 9728:
        factor, width = 19, 512
    elif size > 0 and size & (size - 1) == 0:
        factor, width = 1, size
    else:
        raise ValueError(f"Unsupported reference dimension: {size}")
    output = x.float().reshape(-1, factor, width, 1)
    while output.shape[-2] > 1:
        pairs = output.reshape(*output.shape[:-2], output.shape[-2] // 2, 2, -1)
        output = torch.cat(
            (pairs[..., 0, :] + pairs[..., 1, :], pairs[..., 0, :] - pairs[..., 1, :]), dim=-1
        )
    output = output.reshape(-1, factor, width)
    if size == 2560:
        # Paley type I, q=19: diagonal +1, first row -1 except corner,
        # first column +1, interior off-diagonal Legendre symbol (row-col).
        residues = {i * i % 19 for i in range(1, 19)}
        matrix = torch.tensor(
            [[1] + [-1] * 19]
            + [
                [1] + [1 if i == j or (i - j) % 19 in residues else -1 for j in range(19)]
                for i in range(19)
            ],
            dtype=torch.float32,
            device=x.device,
        )
        if transpose:
            matrix = matrix.T
        output = matrix @ output
    normalizer = 512 if size == 9728 else size
    return (output / normalizer**0.5).reshape_as(x)


def unpack_unisvq(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Row-major Cartesian-product digits, most significant scalar first."""
    words = packed["Qidxs"].long()
    return torch.stack([(words // divisor) % 4 for divisor in (64, 16, 4, 1)], -1).flatten(1)


def decode_unisvq(packed: dict[str, torch.Tensor], *, rounded: bool = True) -> torch.Tensor:
    """BF16 GEMM weights or the unrounded FP32 affine weights used by GEMV."""
    raw = unpack_unisvq(packed).float() * packed["cb_scale"].float() + packed["cb_zero"].float()
    return raw.bfloat16().float() if rounded else raw


def block128_reference(x: torch.Tensor) -> torch.Tensor:
    """Independent block transform, reusing only the test's Sylvester butterfly."""
    return qtip_hadamard_reference(x.reshape(-1, 128)).reshape_as(x)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    error = actual.float() - expected.float()
    return {
        "relative_l2": (error.norm() / expected.float().norm().clamp_min(1e-30)).item(),
        "max_absolute": error.abs().max().item(),
    }
