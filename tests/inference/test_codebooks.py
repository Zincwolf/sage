"""Frozen E8P12 / identical-2bit ABI tests, independent of the native decoders."""

import itertools
from typing import cast

import pytest
import torch

from weight_vq.checkpoint import synthetic_quip_sharp, synthetic_unisvq
from weight_vq.inference.linear import QuIPSharpLinear, UniSVQLinear
from weight_vq.inference.ops import quip_sharp_decompress, unisvq_decompress
from weight_vq.inference.quip_sharp import E8P_PACKED_ABS

from .reference import (
    block128_reference,
    decode_e8p,
    decode_unisvq,
    error_metrics,
    qtip_hadamard_reference,
    unpack_e8p,
    unpack_unisvq,
)


def pack_e8p(indices: torch.Tensor) -> torch.Tensor:
    n, vectors = indices.shape
    lanes = indices.reshape(n // 2, 2, vectors // 2, 2).permute(0, 2, 3, 1).reshape(-1, 4)
    words = torch.zeros(lanes.shape[0], dtype=torch.int64)
    for lane in range(4):
        words |= (lanes[:, lane] // 256) << (8 * lane)
        for bit in range(8):
            words |= ((lanes[:, lane] >> bit) & 1) << (32 + 4 * bit + lane)
    return words.reshape(n // 16, 8, vectors // 8, 4).transpose(1, 2).contiguous().reshape(n, -1)


@pytest.mark.parametrize("k,n", [(64, 16), (2560, 1024)])
def test_e8p_roundtrip(k, n):
    packed = cast(dict[str, torch.Tensor], synthetic_quip_sharp(k, n, seed=42))
    torch.testing.assert_close(pack_e8p(unpack_e8p(packed)), packed["Qidxs"], rtol=0, atol=0)


def test_e8p_coordinates():
    packed = cast(dict[str, torch.Tensor], synthetic_quip_sharp(64, 16))
    packed["Qidxs"].zero_()
    packed["Qidxs"].view(1, 1, 8, 4)[0, 0, 3, 2] = (173 << 24) | (1 << 59)
    expected = torch.zeros(16, 8, dtype=torch.int64)
    expected[7, 5] = 173 * 256 + 64
    torch.testing.assert_close(unpack_e8p(packed), expected, rtol=0, atol=0)


def test_unisvq_roundtrip_and_affine():
    packed = cast(dict[str, torch.Tensor], synthetic_unisvq(128, 128, seed=42))
    packed["Qidxs"] = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    expected = torch.tensor(list(itertools.product(range(4), repeat=4))).reshape(8, 128)
    torch.testing.assert_close(unpack_unisvq(packed), expected, rtol=0, atol=0)
    digits = unpack_unisvq(packed).reshape(8, 32, 4)
    torch.testing.assert_close(
        (digits << torch.tensor([6, 4, 2, 0])).sum(-1).byte(), packed["Qidxs"], rtol=0, atol=0
    )
    affine = expected.float() * torch.tensor(1 / 1.25**0.5).bfloat16().float()
    affine += torch.tensor(-1.5 / 1.25**0.5).bfloat16().float()
    torch.testing.assert_close(decode_unisvq(packed), affine.bfloat16().float(), rtol=0, atol=0)
    assert (decode_unisvq(packed, rounded=False) - decode_unisvq(packed)).abs().max() == 0.00390625


@pytest.mark.cuda
def test_e8p_exhaustive_native_codebook():
    indices = torch.arange(65536).reshape(256, 256)
    packed = {"Qidxs": pack_e8p(indices)}
    actual = quip_sharp_decompress(packed["Qidxs"].view(16, 32, 8, 4).cuda(), E8P_PACKED_ABS.cuda())
    torch.testing.assert_close(actual.cpu().float(), decode_e8p(packed), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("method", ["quip-sharp", "unisvq"])
@pytest.mark.parametrize("k,n", [(2560, 1024), (9728, 2560)])
@pytest.mark.parametrize("rows", [1, 4, 8, 9, 128])
@pytest.mark.parametrize("cached", [False, True])
@torch.inference_mode()
def test_native_linear_reference(method, k, n, rows, cached, record_property):
    generator = synthetic_quip_sharp if method == "quip-sharp" else synthetic_unisvq
    packed = cast(dict[str, torch.Tensor], generator(k, n, seed=42))
    constructor = QuIPSharpLinear if method == "quip-sharp" else UniSVQLinear
    layer = constructor(packed, cached_dense_weight=cached).cuda()
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(43)).half()
    actual = layer(x.cuda()).cpu()
    if method == "quip-sharp":
        raw = decode_e8p(packed)
        transformed = qtip_hadamard_reference(x.float() * packed["SU"], transpose=True) / 32
        expected = qtip_hadamard_reference(transformed @ raw.T) * packed["SV"] * 32
        limit = 5e-3
    else:
        scaled = block128_reference(x.bfloat16().float() * packed["SU"].float()).bfloat16()
        raw = decode_unisvq(packed, rounded=rows > 8)
        mid = (scaled.float() @ raw.T).bfloat16()
        expected = (block128_reference(mid.float()) * packed["SV"].float()).bfloat16().half()
        limit = 6e-3
        native = unisvq_decompress(layer.Qidxs, packed["cb_scale"].item(), packed["cb_zero"].item())
        torch.testing.assert_close(native.cpu().float(), decode_unisvq(packed), rtol=0, atol=0)
    metrics = error_metrics(actual, expected)
    for key, value in metrics.items():
        record_property(key, value)
    assert metrics["relative_l2"] < limit, metrics
    assert actual.dtype == torch.float16
    layer.warm_weight_cache()
    assert (layer.dense is not None) == cached
    assert "dense" not in layer.state_dict()


@pytest.mark.cuda
@pytest.mark.parametrize("method", ["quip-sharp", "unisvq"])
@pytest.mark.parametrize("rows", [1, 9])
@pytest.mark.parametrize("cached", [False, True])
@torch.inference_mode()
def test_native_compile_and_graph(method, rows, cached):
    generator = synthetic_quip_sharp if method == "quip-sharp" else synthetic_unisvq
    packed = cast(dict[str, torch.Tensor], generator(2560, 1024, seed=42))
    constructor = QuIPSharpLinear if method == "quip-sharp" else UniSVQLinear
    layer = constructor(packed, cached_dense_weight=cached).cuda()
    layer.warm_weight_cache()
    x = torch.randn(rows, 2560, generator=torch.Generator().manual_seed(43)).half().cuda()
    expected = layer(x)
    compiled = torch.compile(layer, fullgraph=True, dynamic=False, mode="reduce-overhead")
    tolerance = 2e-3 if method == "quip-sharp" else 0.0
    try:
        torch.testing.assert_close(compiled(x), expected, rtol=tolerance, atol=tolerance)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x)
        # This is graph liveness, not support for hot-swapping cached weights.
        if not cached or rows == 1:
            layer.Qidxs.bitwise_not_()
        else:
            assert layer.dense is not None
            layer.dense.mul_(0.5)
        x.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(captured, layer(x), rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(compiled(x), layer(x), rtol=tolerance, atol=tolerance)
    finally:
        torch.compiler.reset()
