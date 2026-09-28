import pytest
import torch

from weight_vq.checkpoint import synthetic_sage
from weight_vq.inference.linear import SAGELinear
from weight_vq.inference.ops import sage

from .reference import decode_sage, error_metrics, unpack_sage


def assert_fp16_output(actual, expected):
    metrics = error_metrics(actual, expected)
    rms = expected.float().square().mean().sqrt().clamp_min(1e-30).item()
    metrics["max_absolute_over_rms"] = metrics["max_absolute"] / rms
    # Global and worst-element gates allow low-precision reduction differences,
    # without making relative error at a cancellation near zero the criterion.
    assert metrics["relative_l2"] < 5e-3, metrics
    assert metrics["max_absolute_over_rms"] < 5e-3, metrics
    return metrics


@pytest.mark.parametrize("k,n", [(128, 1), (256, 17), (2560, 9), (9728, 32), (16384, 1)])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("seed,entries", [(0, 256), (1, 128), (2, 64)])
def test_sage_layout(k, n, permuted, seed, entries):
    packed = synthetic_sage(k, n, seed=seed, permuted=permuted)
    other = synthetic_sage(k, n, seed=seed, permuted=permuted)
    assert packed["codebook"].shape == (4, entries, 2)
    assert packed["packed_codes"].shape == ((n + 15) // 16, k // 128, 16, 8)
    assert packed["packed_codes"].dtype == packed["radius_codes"].dtype == torch.int32
    assert packed["radius_grid"].dtype == packed["codebook"].dtype == torch.float16
    assert sum(t.numel() * t.element_size() for t in packed.values()) == (
        (n + 15) // 16 * 16 * k * 5 // 16 + entries * 16 + (4 * k if permuted else 0)
    )
    for field in packed:
        torch.testing.assert_close(packed[field], other[field], rtol=0, atol=0)
    if permuted:
        torch.testing.assert_close(packed["permutation"].sort().values, torch.arange(k).int())
    book = packed["codebook"].permute(1, 0, 2).flatten(1).float()
    torch.testing.assert_close(book.norm(dim=-1), torch.ones(entries), rtol=5e-4, atol=0)
    assert torch.all(book >= 0)
    dense = decode_sage(packed, (n + 15) // 16 * 16)
    assert torch.count_nonzero(dense[n:]) == 0


@pytest.mark.parametrize("seed,bits", [(0, 8), (1, 7), (2, 6)])
def test_sage_discrete_fields(seed, bits):
    packed = synthetic_sage(128, 4096, seed=seed, permuted=False)
    words = torch.arange(65536).reshape(4096, 16)
    pairs = (words[:, ::2] | (words[:, 1::2] << 16)).int()
    packed["packed_codes"] = pairs.reshape(256, 1, 16, 8)
    packed["radius_codes"].fill_(int(torch.tensor(0xE4E4E4E4).int()))
    signs, indices, radii = unpack_sage(packed, 4096)
    torch.testing.assert_close(
        signs.flatten(1, 2), 1 - 2 * ((words[..., None] >> torch.arange(8)) & 1), rtol=0, atol=0
    )
    torch.testing.assert_close(indices.flatten(1), (words >> 8) & ((1 << bits) - 1), rtol=0, atol=0)
    torch.testing.assert_close(
        radii.flatten(1),
        (words >> (8 + bits)) | ((torch.arange(16) % 4) << (8 - bits)),
        rtol=0,
        atol=0,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "shape,n",
    [
        ((128,), 1),
        ((4, 256), 17),
        ((2, 3, 2560), 9),
        ((1, 9728), 33),
        ((1, 16384), 9),
        ((128, 256), 32),
        ((16, 128), 1),
        ((17, 256), 17),
        ((65, 2560), 33),
        ((256, 128), 16),
        # Multiple output tiles, row tails and long K reductions.
        ((16, 512), 129),
        ((31, 2560), 257),
        ((32, 4096), 256),
        ((129, 9728), 65),
        # Both sides of the fused-gather dispatch boundary.
        ((1, 4096), 1024),
        ((1, 4096), 1025),
        # Reused-activation GEMV: multiple rows, partial K groups and large N.
        ((2, 3, 4352), 33),
        ((15, 4096), 1025),
        ((1, 2560), 4096),
        # Remaining Qwen3-4B projection shapes exercised by the fixed-K path.
        ((1, 2560), 1024),
        ((1, 2560), 9728),
        ((1, 4096), 2560),
        ((1, 9728), 2560),
        ((0, 128), 17),
    ],
)
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("input_scale", [0.03125, 1.0, 32.0])
def test_sage_reference(shape, n, permuted, seed, input_scale, record_property):
    cpu_packed = synthetic_sage(shape[-1], n, seed=seed, permuted=permuted)
    layer = SAGELinear({key: value.cuda() for key, value in cpu_packed.items()}, n)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(43), dtype=torch.float16).cuda()
    x *= input_scale
    # A contiguous tensor can have an unaligned storage offset.
    storage = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)
    offset_x = storage[1:].view(shape).copy_(x)
    actual = layer(offset_x)
    torch.testing.assert_close(actual, layer(x), rtol=0, atol=0)
    weight = decode_sage(cpu_packed, n).cuda()
    # The execution contract is native FP16 dense QDQ, not unrounded weights.
    expected = torch.nn.functional.linear(x, weight.half())
    assert actual.shape == (*shape[:-1], n)
    if actual.numel():
        metrics = assert_fp16_output(actual, expected)
        for key, value in metrics.items():
            record_property(key, value)
        for key, value in error_metrics(actual, x.float() @ weight.T).items():
            record_property(f"unrounded_{key}", value)
        print(
            {
                "shape": shape,
                "n": n,
                "permuted": permuted,
                "seed": seed,
                "input_scale": input_scale,
                **metrics,
            }
        )


@pytest.mark.cuda
@pytest.mark.parametrize("k,n", [(2560, 4096), (2560, 9728), (4096, 2560), (9728, 2560)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sage_compiled_preordered_reference(k, n, seed, record_property):
    torch.compiler.reset()
    cpu = synthetic_sage(k, n, seed=seed)
    layer = SAGELinear({key: value.cuda() for key, value in cpu.items()}, n)
    reference = decode_sage(cpu, n).half().cuda()
    compiled = torch.compile(layer, fullgraph=True, dynamic=False, mode="reduce-overhead")
    x = torch.randn(1, k, generator=torch.Generator().manual_seed(43), dtype=torch.float16).cuda()
    for scale in (0.03125, 1.0, 32.0):
        scaled = x * scale
        expected = torch.nn.functional.linear(scaled, reference)
        metrics = assert_fp16_output(compiled(scaled), expected)
        for key, value in metrics.items():
            record_property(f"scale_{scale}_{key}", value)
    torch.compiler.reset()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "rows,k", [(2, 256), (65, 256), (32, 4096), (1, 2560), (1, 4096), (1, 9728)]
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sage_grid_graph_state_and_compile(rows, k, seed):
    torch.compiler.reset()
    cpu = synthetic_sage(k, 17, seed=seed)
    cpu["radius_grid"][..., 1].zero_()
    cpu["radius_grid"][..., 0].fill_(0.25)
    cpu["radius_grid"][0, :, 0].zero_()
    layer = SAGELinear({key: value.cuda() for key, value in cpu.items()}, 17)
    restored = SAGELinear({key: torch.zeros_like(value) for key, value in cpu.items()}, 17)
    restored.load_state_dict(layer.state_dict())
    restored.cuda()
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    expected = layer(x)
    assert torch.count_nonzero(expected[:, 0]) == 0
    reference = torch.nn.functional.linear(x, decode_sage(cpu, 17).half().cuda())
    assert_fp16_output(expected, reference)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(layer(x.T.contiguous().T), expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(x)
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    compiled = torch.compile(layer, fullgraph=True)
    # Compile may select a different reduction layout for preordered inputs.
    assert_fp16_output(compiled(x), reference)
    layer.bias = torch.ones(17, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(layer(x), expected + 1, rtol=0, atol=0)
    torch.library.opcheck(
        sage,
        (
            x,
            layer.packed_codes,
            layer.radius_codes,
            layer.radius_grid,
            layer.codebook,
            layer.permutation,
            17,
        ),
    )
