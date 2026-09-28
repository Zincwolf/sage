"""Frozen FHT and QTIP transform conventions, with an independent reference."""

from typing import cast

import pytest
import torch

from weight_vq.inference.ops import hadamard
from weight_vq.inference.qtip import HADAMARD_20, hadamard_transform

from .reference import qtip_hadamard_reference


@pytest.mark.parametrize("size", [32, 128, 512])
def test_reference_sylvester_matrix(size):
    # Matrix entries from binary dot-product parity, not a butterfly.
    indices = torch.arange(size)
    shared_bits = indices[:, None] & indices[None, :]
    parity = torch.zeros_like(shared_bits)
    for bit in range(size.bit_length() - 1):
        parity ^= (shared_bits >> bit) & 1
    matrix = (1 - 2 * parity).float()
    x = torch.randn(3, size, generator=torch.Generator().manual_seed(43))
    torch.testing.assert_close(
        qtip_hadamard_reference(x), x @ matrix / size**0.5, rtol=2e-5, atol=2e-6
    )


def test_reference_h20_direction_and_block512():
    # Constant H128 channels isolate the non-symmetric H20 factor.
    x = torch.eye(20).repeat_interleave(128, dim=1)
    for transpose in (False, True):
        result = qtip_hadamard_reference(x, transpose=transpose).reshape(20, 20, 128)
        matrix = HADAMARD_20.T if transpose else HADAMARD_20
        torch.testing.assert_close(result[..., 0], matrix.T * (128 / 2560**0.5))
        assert torch.count_nonzero(result[..., 1:]) == 0
    # The 9728 transform must not mix different 512-element blocks.
    x = torch.zeros(1, 9728)
    x[0, 512] = 1
    expected = torch.zeros_like(x)
    expected[:, 512:1024] = 512**-0.5
    torch.testing.assert_close(qtip_hadamard_reference(x), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("size", [1024, 2560, 4096, 9728])
@pytest.mark.parametrize("transpose", [False, True])
@torch.inference_mode()
def test_qtip_hadamard_cuda(size, transpose):
    # Exercise real FP32 QTIP transform inputs and non-contiguous leading strides.
    source = torch.randn(2, 3, size, generator=torch.Generator().manual_seed(43)).cuda()
    x = source.transpose(0, 1)
    matrix = HADAMARD_20.cuda() if size == 2560 else None
    actual = hadamard_transform(x, matrix, transpose=transpose)
    expected = qtip_hadamard_reference(x, transpose=transpose)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert actual.dtype == torch.float32


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fht_opcheck(dtype):
    x = torch.randn(2, 512, device="cuda", dtype=dtype)
    torch.library.opcheck(hadamard, (x, 512**-0.5))
    expected = qtip_hadamard_reference(x).to(dtype)
    torch.testing.assert_close(hadamard(x, 512**-0.5), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("cached_dense_weight", [False, True])
@torch.inference_mode()
def test_qtip_linear_compile_and_graph(rows, cached_dense_weight):
    from weight_vq.checkpoint import synthetic_qtip
    from weight_vq.inference.linear import QTIPLinear

    packed = cast(dict[str, torch.Tensor], synthetic_qtip(2560, 1024, seed=42))
    layer = QTIPLinear(packed, cached_dense_weight=cached_dense_weight).cuda()
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    layer.warm_weight_cache()
    dense = layer.dense
    assert (dense is not None) == cached_dense_weight
    expected = layer(x)
    compiled = torch.compile(layer, fullgraph=True, dynamic=False, mode="reduce-overhead")
    try:
        torch.testing.assert_close(compiled(x), expected, rtol=2e-3, atol=2e-3)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x)
        if rows == 1 or not cached_dense_weight:
            layer.trellis.bitwise_not_()
        else:
            assert dense is not None
            dense.mul_(0.5)
        x.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(captured, layer(x), rtol=0, atol=0)
        torch.testing.assert_close(compiled(x), layer(x), rtol=2e-3, atol=2e-3)
        assert layer.dense is dense
        assert "dense" not in layer.state_dict()
    finally:
        torch.compiler.reset()
