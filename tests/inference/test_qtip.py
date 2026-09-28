"""QTIP R=2 packed ABI and the frozen native single-vector entry points."""

from typing import cast

import pytest
import torch

from weight_vq.checkpoint import synthetic_qtip

from .reference import decode_qtip, error_metrics, qtip_hadamard_reference, unpack_qtip


@pytest.mark.parametrize("k,n", [(32, 32), (128, 64)])
def test_qtip_packed_roundtrip(k, n):
    packed = cast(dict[str, torch.Tensor], synthetic_qtip(k, n, seed=42))
    states = unpack_qtip(packed)
    # Adjacent windows share twelve bits, including the final -> initial window.
    torch.testing.assert_close(states % 4096, torch.roll(states, -1, -1) // 16, rtol=0, atol=0)
    # Each pair of windows contributes its high nibble to one canonical byte.
    canonical = (states[..., ::2] // 4096 * 16 + states[..., 1::2] // 4096).to(torch.uint8)
    # Inverse layout expressed as tensor axes, independently of the address gather.
    restored = (
        canonical.reshape(n // 32, 2, k // 32, 2, 32, 2)
        .permute(0, 2, 4, 3, 1, 5)
        .flip(-1)
        .contiguous()
        .view(torch.int16)
        .reshape_as(packed["trellis"])
    )
    torch.testing.assert_close(restored, packed["trellis"], rtol=0, atol=0)


@pytest.mark.parametrize("k,n", [(32, 32), (128, 64), (2560, 1024)])
def test_qtip_dense_decode(k, n):
    from weight_vq.inference.qtip import decode_dense

    packed = cast(dict[str, torch.Tensor], synthetic_qtip(k, n, seed=42))
    # Two independent implementations of the same packed layout: the byte-address
    # gather in the reference and the upstream unswizzle plus shift extraction.
    dense = decode_dense(packed["trellis"], packed["tlut"], n, k)
    torch.testing.assert_close(dense, decode_qtip(packed).half(), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("rows", [1, 4, 128])
@pytest.mark.parametrize("k,n", [(2560, 1024), (9728, 2560)])
@pytest.mark.parametrize("cached_dense_weight", [False, True])
@torch.inference_mode()
def test_qtip_linear(rows, k, n, cached_dense_weight):
    from weight_vq.inference.linear import QTIPLinear

    packed = cast(dict[str, torch.Tensor], synthetic_qtip(k, n, seed=42))
    layer = QTIPLinear(packed, cached_dense_weight=cached_dense_weight).cuda()
    x = torch.randn(rows, k, generator=torch.Generator().manual_seed(43)).half().cuda()
    actual = layer(x)
    # Independent FP32 decode and transforms, without the production FHT wrapper.
    raw = decode_qtip(packed).cuda()
    transformed = qtip_hadamard_reference(x.float() * packed["SU"].cuda(), transpose=True) / 32
    expected = qtip_hadamard_reference(transformed @ raw.T)
    expected = expected * (packed["SV"].cuda() * 32)
    metrics = error_metrics(actual, expected)
    print({"rows": rows, "k": k, "n": n, **metrics})
    assert actual.shape == (rows, n)
    assert metrics["relative_l2"] < 5e-3
    dense = layer.dense
    if cached_dense_weight and rows > 1:
        assert dense is not None
        # Decode must retain, not rebuild, the prefill weight.
        layer(x[:1])
        assert layer.dense is dense
    if not cached_dense_weight:
        layer.warm_weight_cache()
        torch.testing.assert_close(layer(x), actual, rtol=0, atol=0)
        assert layer.dense is None
    assert "dense" not in layer.state_dict()
    assert "had_left" not in layer.state_dict()
    assert "had_right" not in layer.state_dict()


def test_qtip_known_bits_and_coordinates():
    packed = cast(dict[str, torch.Tensor], synthetic_qtip(32, 32))
    packed["trellis"].zero_()
    # Byte 1 starts the top-left tile's circular big-endian stream.
    packed["trellis"].view(torch.uint8).flatten()[1] = 128
    states = unpack_qtip(packed)
    assert states[0, 0, 0] == 32768
    assert states[0, 0, -1] == 2048
    assert states[0, 0, -2] == 128
    assert states[0, 0, -3] == 8
    assert torch.count_nonzero(states) == 4

    packed["tlut"].zero_()
    packed["tlut"][0] = torch.tensor([1, 2])
    packed["tlut"][1] = torch.tensor([5, 6])
    expected = torch.tensor([1.0, 2.0]).repeat(32, 16)
    # State 32768 flips the first component; state 128 selects LUT entry 258.
    # State 2048 selects entry 32. Both entries are zero in this test.
    expected[0, :2] = torch.tensor([-1.0, 2.0])
    expected[15, 14:16] = 0
    expected[7, 14:16] = 0
    # State 8 selects entry 1, at columns 6 and 7 of row 15.
    expected[15, 6:8] = torch.tensor([5.0, 6.0])
    torch.testing.assert_close(decode_qtip(packed), expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "k,n", [(2560, 1024), (2560, 4096), (2560, 9728), (4096, 2560), (9728, 2560), (4096, 1024)]
)
@torch.inference_mode()
def test_qtip_native_decode_shapes(k, n, record_property):
    from weight_vq.inference import (
        _qtip as qtip_kernels,  # pyright: ignore[reportAttributeAccessIssue]
    )

    packed = cast(dict[str, torch.Tensor], synthetic_qtip(k, n, seed=42))
    weight = decode_qtip(packed)
    compressed = packed["trellis"].view(torch.int32).flatten().cuda()
    codebook = packed["tlut"].flatten().cuda()
    symbol = f"decompress_matvec_16_9_2_1_{n}_1_{k}"
    native = getattr(qtip_kernels, symbol)
    record_property("native_symbol", symbol)
    # Native ABI is [K, 1] FP16 -> [N, 1] FP32. Multiple decode rows must
    # be dispatched separately; its wrapper does not enforce that column count.
    for leading_shape in ((), (1,), (4,), (2, 3)):
        shape = (*leading_shape, k)
        x = torch.randn(shape, generator=torch.Generator().manual_seed(43)).half()
        expected = x.float() @ weight.T
        device_x = x.cuda().reshape(-1, k)
        output = torch.full((device_x.shape[0], n), float("nan"), device="cuda")
        for row in range(device_x.shape[0]):
            native(output[row].view(n, 1), compressed, device_x[row].view(k, 1), codebook)
        actual = output.reshape(*leading_shape, n).cpu()
        metrics = error_metrics(actual, expected)
        print({"symbol": symbol, "shape": shape, **metrics})
        for key, value in metrics.items():
            record_property(f"rows_{device_x.shape[0]}_rank_{x.ndim}_{key}", value)
        assert actual.shape == (*leading_shape, n)
        # FP16 operands and FP32 MMA/reduction; independent reference uses FP32 GEMM.
        assert metrics["relative_l2"] < 2e-5
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-3)
