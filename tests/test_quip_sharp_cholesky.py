"""Bounded Cholesky recovery; requires the real QuIP# dependencies, even on CPU."""

import importlib.util
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def quip() -> ModuleType:
    missing = [
        name for name in ("quiptools_cuda", "primefac") if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip(f"QuIP# upstream imports still require: {', '.join(missing)}")
    # QuIP# environment boundary: do not replace missing dependencies with stubs.
    from quip_sharp.algo import quip

    return quip


@pytest.fixture
def args() -> SimpleNamespace:
    return SimpleNamespace(
        use_fp64=False,
        rescale_WH=False,
        incoh_mode="had",
        sigma_reg=0.01,
        lora_rank=0,
        scale_override=0,
        no_use_buffered=True,
        quip_tune_iters=0,
        resid_scale_override=-1,
    )


@pytest.mark.parametrize("use_fp64", [False, True])
def test_success_does_not_change_transformed_hessian(
    quip: ModuleType, args: SimpleNamespace, use_fp64: bool
) -> None:
    args.use_fp64 = use_fp64
    dtype = torch.float64 if use_fp64 else torch.float32
    h = torch.diag(torch.arange(1, 9, dtype=dtype))
    w = torch.arange(64, dtype=dtype).reshape(8, 8)
    out = quip.incoherence_preprocess(h, w, args)
    assert out is not None
    factor, transformed, _, signs, _, _ = out
    torch.testing.assert_close(transformed, quip.RHT_H(h, signs), rtol=0, atol=0)
    torch.testing.assert_close(factor @ factor.T, transformed)


@pytest.mark.parametrize("use_fp64", [False, True])
def test_real_indefinite_hessian_returns_none_or_repairs_once(
    quip: ModuleType, args: SimpleNamespace, use_fp64: bool
) -> None:
    args.use_fp64 = use_fp64
    dtype = torch.float64 if use_fp64 else torch.float32
    h = torch.eye(8, dtype=dtype)
    h[0, 0] = -0.001
    original = h.clone()
    out = quip.incoherence_preprocess(h, torch.ones_like(h), args)
    torch.testing.assert_close(h, original, rtol=0, atol=0)
    if not use_fp64:
        assert out is None
        return
    assert out is not None
    factor, transformed, _, signs, _, _ = out
    expected = quip.RHT_H(h, signs)
    expected = (expected + expected.T) * 0.5
    expected.diagonal().add_(expected.diagonal().mean() * args.sigma_reg)
    torch.testing.assert_close(transformed, expected, rtol=0, atol=0)
    torch.testing.assert_close(factor @ factor.T, transformed)


def test_failed_fp64_repair_raises_without_more_attempts(
    quip: ModuleType, args: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    args.use_fp64 = True
    h = torch.eye(8, dtype=torch.float64)
    h[0, 0] = -1
    cholesky = torch.linalg.cholesky
    calls = []

    def traced(matrix: torch.Tensor) -> torch.Tensor:
        calls.append(matrix.clone())
        return cholesky(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky", traced)
    with pytest.raises(RuntimeError, match="positive-definite"):
        quip.incoherence_preprocess(h, torch.ones_like(h), args)
    assert len(calls) == 2
    expected = (calls[0] + calls[0].T) * 0.5
    expected.diagonal().add_(expected.diagonal().mean() * args.sigma_reg)
    torch.testing.assert_close(calls[1], expected, rtol=0, atol=0)


def test_unrelated_runtime_error_is_not_retried(
    quip: ModuleType, args: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(matrix: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("unrelated device failure")

    monkeypatch.setattr(torch.linalg, "cholesky", fail)
    with pytest.raises(RuntimeError, match="unrelated device failure"):
        quip.incoherence_preprocess(torch.eye(8), torch.ones(8, 8), args)


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_quantize_retries_fp64_before_damping(
    quip: ModuleType, args: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    from quip_sharp.codebook import get_codebook

    h = torch.diag(torch.linspace(1, 2, 64, dtype=torch.float64))
    w = torch.randn(16, 64, generator=torch.Generator().manual_seed(19))
    original_h, original_w = h.clone(), w.clone()
    cholesky = torch.linalg.cholesky
    calls = []

    def traced(matrix: torch.Tensor) -> torch.Tensor:
        calls.append(matrix.clone())
        if len(calls) <= failures:
            # Use the native factorization error, not a generic RuntimeError.
            return cholesky(-torch.eye(matrix.shape[0], dtype=matrix.dtype))
        return cholesky(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky", traced)
    quantized, attr = quip.quantize(h, w, 0, get_codebook("E8P12"), args)
    expected_dtypes = [torch.float32] + [torch.float64] * failures
    # The last call is block_LDL, using the same effective Hr as preprocessing.
    expected_dtypes.append(expected_dtypes[-1])
    assert [matrix.dtype for matrix in calls] == expected_dtypes
    torch.testing.assert_close(calls[-1], calls[-2], rtol=0, atol=0)
    if failures == 2:
        expected = (calls[1] + calls[1].T) * 0.5
        expected.diagonal().add_(expected.diagonal().mean() * args.sigma_reg)
        torch.testing.assert_close(calls[2], expected, rtol=0, atol=0)
    assert quantized.shape == w.shape
    assert quantized.dtype == w.dtype
    assert torch.isfinite(quantized).all()
    assert set(attr) == {"Qidxs", "A", "B", "SU", "SV", "scaleWH"}
    assert args.use_fp64 is False
    torch.testing.assert_close(h, original_h, rtol=0, atol=0)
    torch.testing.assert_close(w, original_w, rtol=0, atol=0)
