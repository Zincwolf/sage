import pytest
import torch

from weight_vq.checkpoint import synthetic_sphquant
from weight_vq.inference.linear import SPHQuantLinear

from .reference import decode_sphquant, error_metrics


@pytest.mark.cuda
@pytest.mark.parametrize(
    "shape,n", [((128,), 1), ((4, 256), 17), ((2, 3, 2560), 9), ((1, 9728), 17), ((1, 16384), 9)]
)
def test_cuda_reference(shape, n, record_property):
    cpu_packed = synthetic_sphquant(shape[-1], n, seed=42)
    packed = {key: tensor.cuda() for key, tensor in cpu_packed.items()}
    layer = SPHQuantLinear(packed)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(43), dtype=torch.float16).cuda()
    with torch.inference_mode():
        actual = layer(x)
        reference_weight = decode_sphquant(cpu_packed).cuda()
        expected = x.float() @ reference_weight.T
        dense_fp16 = x @ reference_weight.half().T
    metrics = error_metrics(actual, expected)
    dense_metrics = error_metrics(dense_fp16, expected)
    for key, value in metrics.items():
        record_property(key, value)
    for key, value in dense_metrics.items():
        record_property(f"dense_fp16_{key}", value)
    print({"shape": shape, "n": n, "sphquant": metrics, "dense_fp16": dense_metrics})
    # Screening bound for frozen FP16 local accumulation; this synthetic coverage
    # does not establish a universal tolerance for real checkpoints.
    assert metrics["relative_l2"] < 5e-3


@pytest.mark.cuda
def test_cuda_zero_and_constant_grid():
    packed = synthetic_sphquant(128, 9)
    packed["radius_scales"].zero_()
    packed["radius_mins"].zero_()
    packed["radius_mins"][1:].fill_(0.25)
    layer = SPHQuantLinear({k: v.cuda() for k, v in packed.items()})
    x = torch.ones(4, 128, device="cuda", dtype=torch.float16)
    actual = layer(x)
    assert torch.count_nonzero(actual[:, 0]) == 0
    expected = x.float() @ decode_sphquant(packed).cuda().T
    assert error_metrics(actual, expected)["relative_l2"] < 5e-3


@pytest.mark.cuda
def test_cuda_graph_and_state_dict():
    cpu_packed = synthetic_sphquant(256, 17)
    packed = {k: v.cuda() for k, v in cpu_packed.items()}
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    layer = SPHQuantLinear(packed, torch.ones(17, device="cuda", dtype=torch.float16))
    restored = SPHQuantLinear(cpu_packed, torch.zeros(17).half())
    restored.load_state_dict(layer.state_dict())
    restored = restored.cuda()
    with torch.inference_mode():
        expected = layer(x)
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
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
        # The Linear wrapper must include any required activation layout conversion.
        torch.testing.assert_close(layer(x.T.contiguous().T), expected, rtol=0, atol=0)
