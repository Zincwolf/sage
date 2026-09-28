"""Dispatcher registration; native extensions are loaded only on CUDA execution."""

from typing import cast

import torch


@torch.library.custom_op("wvq::hadamard", mutates_args=(), device_types="cuda")
def hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Frozen upstream FHT, kept opaque to Inductor like QTIP's upstream wrapper."""
    import fast_hadamard_transform  # pyright: ignore[reportMissingImports]

    return cast(torch.Tensor, fast_hadamard_transform.hadamard_transform(x, scale))


@hadamard.register_fake
def _hadamard_fake(x, scale):
    return x.new_empty(x.shape)


@torch.library.custom_op("wvq::sphquant", mutates_args=(), device_types="cuda")
def sphquant(
    x: torch.Tensor,
    packed_codes: torch.Tensor,
    radius_extra: torch.Tensor,
    radius_scales: torch.Tensor,
    radius_mins: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    from weight_vq.inference import _C  # pyright: ignore[reportMissingModuleSource]

    return _C.sph_gemv_forward_cuda(
        x, packed_codes, radius_extra, radius_scales, radius_mins, codebook, 128
    )


@sphquant.register_fake
def _sphquant_fake(x, packed_codes, radius_extra, radius_scales, radius_mins, codebook):
    return x.new_empty((*x.shape[:-1], packed_codes.shape[0]))


@torch.library.custom_op("wvq::sage", mutates_args=(), device_types="cuda")
def sage(
    x: torch.Tensor,
    packed_codes: torch.Tensor,
    radius_codes: torch.Tensor,
    radius_grid: torch.Tensor,
    codebook: torch.Tensor,
    permutation: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    from weight_vq.inference import _C  # pyright: ignore[reportMissingModuleSource]

    return _C.sage_gemv_forward_cuda(
        x, packed_codes, radius_codes, radius_grid, codebook, permutation, out_features
    )


@sage.register_fake
def _sage_fake(x, packed_codes, radius_codes, radius_grid, codebook, permutation, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


@torch.library.custom_op("wvq::qtip", mutates_args=(), device_types="cuda")
def qtip(
    x: torch.Tensor, trellis: torch.Tensor, tlut: torch.Tensor, out_features: int
) -> torch.Tensor:
    """Single-token QTIP decode; the frozen kernels only accept one column."""
    from weight_vq.inference import (
        _qtip as qtip_kernels,  # pyright: ignore[reportAttributeAccessIssue]
    )

    in_features = x.shape[-1]
    rows = x.numel() // in_features
    kernel = getattr(qtip_kernels, f"decompress_matvec_16_9_2_1_{out_features}_1_{in_features}")
    compressed = trellis.view(torch.int32).reshape(-1).contiguous()
    flat_x = x.reshape(rows, in_features)
    output = torch.empty((rows, out_features), dtype=torch.float32, device=x.device)
    for row in range(rows):
        kernel(
            output[row].view(out_features, 1),
            compressed,
            flat_x[row].contiguous().view(in_features, 1),
            tlut.reshape(-1),
        )
    return output.reshape(*x.shape[:-1], out_features)


@qtip.register_fake
def _qtip_fake(x, trellis, tlut, out_features):
    return x.new_empty((*x.shape[:-1], out_features), dtype=torch.float32)


@torch.library.custom_op("wvq::quip_sharp", mutates_args=(), device_types="cuda")
def quip_sharp(x: torch.Tensor, indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Frozen E8P12 single-vector decode; indices use the native 4-D tile view."""
    import quiptools_cuda  # pyright: ignore[reportMissingImports]

    return cast(torch.Tensor, quiptools_cuda.decode_matvec_e8p(x, indices, codebook))


@quip_sharp.register_fake
def _quip_sharp_fake(x, indices, codebook):
    return x.new_empty((indices.shape[0] * 16,), dtype=torch.float32)


@torch.library.custom_op("wvq::quip_sharp_decompress", mutates_args=(), device_types="cuda")
def quip_sharp_decompress(indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Materialize raw FP16 weights for the untimed, retained Linear cache."""
    import quiptools_cuda  # pyright: ignore[reportMissingImports]

    return cast(torch.Tensor, quiptools_cuda.decompress_packed_e8p(indices, codebook))


@quip_sharp_decompress.register_fake
def _quip_sharp_decompress_fake(indices, codebook):
    return indices.new_empty((indices.shape[0] * 16, indices.shape[1] * 64), dtype=torch.float16)


@torch.library.custom_op("wvq::unisvq_su_had", mutates_args=(), device_types="cuda")
def unisvq_su_had(x: torch.Tensor, su: torch.Tensor) -> torch.Tensor:
    """Frozen fused BF16 input scaling and block-128 Hadamard, out of place."""
    from weight_vq.inference import _unisvq  # pyright: ignore[reportAttributeAccessIssue]

    output = torch.empty_like(x)
    _unisvq.fused_su_had128(x, su, output)
    return output


@unisvq_su_had.register_fake
def _unisvq_su_had_fake(x, su):
    return torch.empty_like(x)


@torch.library.custom_op("wvq::unisvq_had_sv", mutates_args=(), device_types="cuda")
def unisvq_had_sv(x: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
    """Frozen block-128 Hadamard and output scaling; the input is not mutated."""
    from weight_vq.inference import _unisvq  # pyright: ignore[reportAttributeAccessIssue]

    output = torch.empty_like(x)
    _unisvq.fused_had128_sv(x, sv, output)
    return output


@unisvq_had_sv.register_fake
def _unisvq_had_sv_fake(x, sv):
    return torch.empty_like(x)


@torch.library.custom_op("wvq::unisvq", mutates_args=(), device_types="cuda")
def unisvq(x: torch.Tensor, indices: torch.Tensor, cb_scale: float, cb_zero: float) -> torch.Tensor:
    """Frozen packed 2-bit GEMV: unrounded FP32 affine weights, BF16 output."""
    from weight_vq.inference import _unisvq  # pyright: ignore[reportAttributeAccessIssue]

    return cast(torch.Tensor, _unisvq.fused_dequant_gemv_packed(x, indices, cb_scale, cb_zero))


@unisvq.register_fake
def _unisvq_fake(x, indices, cb_scale, cb_zero):
    return x.new_empty((x.shape[0], indices.shape[0]))


@torch.library.custom_op("wvq::unisvq_decompress", mutates_args=(), device_types="cuda")
def unisvq_decompress(indices: torch.Tensor, cb_scale: float, cb_zero: float) -> torch.Tensor:
    """Build the retained raw BF16 weight cache with the frozen dequantizer."""
    from weight_vq.inference import _unisvq  # pyright: ignore[reportAttributeAccessIssue]

    return cast(torch.Tensor, _unisvq.dequant_weight_packed(indices, cb_scale, cb_zero))


@unisvq_decompress.register_fake
def _unisvq_decompress_fake(indices, cb_scale, cb_zero):
    return indices.new_empty((indices.shape[0], indices.shape[1] * 4), dtype=torch.bfloat16)
