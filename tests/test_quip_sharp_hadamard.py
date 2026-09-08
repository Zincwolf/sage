"""QTIP-compatible block RHT, retaining QuIP#'s reference FP64 path."""

import importlib.util
from types import ModuleType

import pytest
import torch
from scipy.linalg import hadamard


@pytest.fixture(scope="module")
def had() -> ModuleType:
    missing = [
        name for name in ("quiptools_cuda", "primefac") if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip(f"QuIP# upstream imports still require: {', '.join(missing)}")
    from quip_sharp.utils import matmul_had

    return matmul_had


@pytest.mark.parametrize("width,factor", [(2560, 20), (5120, 20), (4096, 1), (9728, -512)])
def test_factor_selection(had: ModuleType, width: int, factor: int) -> None:
    matrix, actual = had.get_hadK(width)
    assert actual == factor
    if factor <= 1:
        assert matrix is None
    else:
        assert matrix.shape == (factor, factor)


def test_unsupported_width_still_fails(had: ModuleType) -> None:
    with pytest.raises(ValueError, match="Unsupported Hadamard dimension: 13"):
        had.get_hadK(13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_block_transform(had: ModuleType, dtype: torch.dtype, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # A strided projection-shaped input, without allocating a 9728-square Hessian.
    x = torch.randn(9728, 3, dtype=dtype, device=device).T
    original = x.clone()
    h512 = torch.tensor(hadamard(512), dtype=dtype, device=device) * (512**-0.5)
    expected = (x.reshape(3, 19, 512) @ h512).reshape_as(x)
    actual = had.matmul_hadU(x)
    assert actual.dtype == dtype
    assert actual.device == x.device
    # The upstream reference normalizes with torch.tensor(n).sqrt() (FP32 scalar).
    tolerance = 2e-6 if dtype == torch.float32 else 1e-7
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(had.matmul_hadUt(actual), x, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("width", [2560, 5120, 9728])
def test_cuda_transform_matches_reference_and_qtip(had: ModuleType, width: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from lib.utils import matmul_had as qtip_had

    x = torch.randn(width, 3, device="cuda").T
    matrix, factor = had.get_hadK(width)
    actual = had.matmul_hadU_cuda(x, matrix, factor)
    torch.testing.assert_close(actual, had.matmul_hadU(x), rtol=1e-5, atol=1e-5)
    inverse = had.matmul_hadUt_cuda(actual, matrix, factor)
    torch.testing.assert_close(inverse, x, rtol=1e-5, atol=1e-5)
    if width == 9728:
        torch.testing.assert_close(actual, qtip_had.matmul_hadU(x), rtol=0, atol=0)


@pytest.mark.parametrize("in_features,out_features", [(9728, 16), (64, 9728)])
def test_training_cache_with_implicit_factor(
    had: ModuleType, in_features: int, out_features: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from quip_sharp.codebook import get_codebook, get_id
    from quip_sharp.linear.quantized_linear import QuantizedLinear

    cb = get_codebook("E8P12").cuda()
    layer = QuantizedLinear(
        in_features,
        out_features,
        codesz=cb.codesz,
        packsz=cb.packsz,
        pack_out=cb.pack_out,
        idx_dtype=str(cb.idx_dtype),
        codebook_version=cb.version,
        rank=0,
        train_mode=True,
    ).cuda()
    layer.get_buffer("codebook_id").fill_(get_id("E8P12"))
    layer.Qidxs.copy_(
        torch.randint(-(2**62), 2**62, layer.Qidxs.shape, dtype=torch.int64, device="cuda")
    )
    # Independently reconstruct the packed codebook weights via the reference RHT.
    decoded = cb.by_idxs(layer.Qidxs).float()
    dense = had.matmul_hadU(had.matmul_hadU(decoded).T).T
    x = torch.randn(2, in_features, device="cuda", requires_grad=True)
    # Upstream finetuning first runs validation under no_grad to build the cache.
    with torch.no_grad():
        layer(x)
    layer.get_buffer("SU").requires_grad_(True)
    layer.get_buffer("SV").requires_grad_(True)
    actual = layer(x)
    cached = layer.codebook_class.W
    torch.testing.assert_close(cached.float(), dense.T / 32, rtol=1e-3, atol=1e-4)
    # Isolate FP16 GEMM from reference/CUDA transform rounding near FP16 midpoints.
    expected = (x.half() @ cached).float() * 32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    for tensor in (x, layer.get_buffer("SU"), layer.get_buffer("SV")):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
