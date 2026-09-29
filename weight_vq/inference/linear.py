"""Inference-only Linear modules consuming packed quantized weights.

Ported from sage-kernels 4d791bf. CUDA arithmetic/layouts are unchanged.
HF loading defaults to no dense retention and prepares derived buffers before
compilation/capture. Spherical kernels compute in FP16 at the module boundary.
"""

import torch
from torch import nn

from weight_vq.inference.ops import (
    qtip,
    quip_sharp,
    quip_sharp_decompress,
    sage,
    sphquant,
    unisvq,
    unisvq_decompress,
    unisvq_had_sv,
    unisvq_su_had,
)
from weight_vq.inference.qtip import HADAMARD_20, decode_dense, hadamard_transform
from weight_vq.inference.quip_sharp import E8P_PACKED_ABS


class QTIPLinear(nn.Module):
    """QTIP 2-bit quantlut_sym with optional raw FP16 weight retention.

    Retention is a deployment policy, not upstream default eval. Without it,
    every GEMM call decodes a temporary weight. Single-row decode stays packed.
    Real checkpoints preserve FP32 SU/TLUT; the inference kernels consume an
    FP16 LUT. This execution-time rounding is not dense-QDQ equivalence.
    """

    trellis: torch.Tensor
    SU: torch.Tensor
    SV: torch.Tensor
    tlut: torch.Tensor
    bias: torch.Tensor | None
    dense: torch.Tensor | None
    had_left: torch.Tensor | None
    had_right: torch.Tensor | None

    def __init__(
        self,
        packed: dict[str, torch.Tensor],
        bias: torch.Tensor | None = None,
        *,
        cached_dense_weight: bool = True,
    ):
        super().__init__()
        self.cached_dense_weight = cached_dense_weight
        self.in_features = packed["SU"].numel()
        self.out_features = packed["SV"].numel()
        for name in ("trellis", "SU", "SV", "tlut"):
            self.register_buffer(name, packed[name])
        self.register_buffer("bias", bias)
        # Counted by get_memory_footprint, but not checkpoint state.
        self.register_buffer("dense", None, persistent=False)
        self.register_buffer(
            "had_left",
            HADAMARD_20.to(self.SU.device).clone() if self.in_features == 2560 else None,
            persistent=False,
        )
        self.register_buffer(
            "had_right",
            HADAMARD_20.to(self.SV.device).clone() if self.out_features == 2560 else None,
            persistent=False,
        )

    def prepare_for_inference(self) -> None:
        self.had_left = HADAMARD_20.to(self.SU.device).clone() if self.in_features == 2560 else None
        self.had_right = (
            HADAMARD_20.to(self.SV.device).clone() if self.out_features == 2560 else None
        )

    def warm_weight_cache(self) -> None:
        if self.cached_dense_weight and self.dense is None:
            self.dense = decode_dense(
                self.trellis, self.tlut.half(), self.out_features, self.in_features
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = hadamard_transform(x.float() * self.SU, self.had_left, transpose=True) / 32
        scaled = scaled.to(torch.float16).reshape(-1, self.in_features)
        if scaled.shape[0] == 1:
            output = qtip(scaled, self.trellis, self.tlut.half(), self.out_features).float()
        else:
            if self.cached_dense_weight:
                self.warm_weight_cache()
                weight = self.dense
                assert weight is not None
            else:
                weight = decode_dense(
                    self.trellis, self.tlut.half(), self.out_features, self.in_features
                )
            output = (scaled @ weight.T).float()
        output = hadamard_transform(output, self.had_right, transpose=False) * (self.SV * 32)
        output = output.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
        return output if self.bias is None else output + self.bias


class QuIPSharpLinear(nn.Module):
    """E8P12 inference: packed GEMV or GEMM with optional raw FP16 retention.

    Without retention, every GEMM call decompresses a temporary weight. When
    enabled, prepare the cache before compilation/capture and keep weights fixed.

    Qwen3 transforms match wvq's QTIP adaptation exactly (including block512),
    so both methods use the same FHT implementation. Wscale is folded into SV.
    """

    dense: torch.Tensor | None
    Qidxs: torch.Tensor
    SU: torch.Tensor
    SV: torch.Tensor
    grid_packed_abs: torch.Tensor
    had_left: torch.Tensor | None
    had_right: torch.Tensor | None
    bias: torch.Tensor | None

    def __init__(
        self,
        packed: dict[str, torch.Tensor],
        bias: torch.Tensor | None = None,
        *,
        cached_dense_weight: bool = True,
    ):
        super().__init__()
        self.cached_dense_weight = cached_dense_weight
        self.in_features = packed["SU"].numel()
        self.out_features = packed["SV"].numel()
        for name in ("Qidxs", "SU", "SV"):
            self.register_buffer(name, packed[name])
        self.register_buffer("bias", bias)
        self.register_buffer("dense", None, persistent=False)
        self.register_buffer(
            "grid_packed_abs", E8P_PACKED_ABS.to(self.SU.device).clone(), persistent=False
        )
        self.register_buffer(
            "had_left",
            HADAMARD_20.to(self.SU.device).clone() if self.in_features == 2560 else None,
            persistent=False,
        )
        self.register_buffer(
            "had_right",
            HADAMARD_20.to(self.SV.device).clone() if self.out_features == 2560 else None,
            persistent=False,
        )

    def prepare_for_inference(self) -> None:
        self.grid_packed_abs = E8P_PACKED_ABS.to(self.SU.device).clone()
        self.had_left = HADAMARD_20.to(self.SU.device).clone() if self.in_features == 2560 else None
        self.had_right = (
            HADAMARD_20.to(self.SV.device).clone() if self.out_features == 2560 else None
        )

    def warm_weight_cache(self) -> None:
        if self.cached_dense_weight and self.dense is None:
            indices = self.Qidxs.view(self.out_features // 16, self.in_features // 64, 8, 4)
            self.dense = quip_sharp_decompress(indices, self.grid_packed_abs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = hadamard_transform(x.float() * self.SU, self.had_left, transpose=True) / 32
        scaled = scaled.to(torch.float16).reshape(-1, self.in_features)
        indices = self.Qidxs.view(self.out_features // 16, self.in_features // 64, 8, 4)
        if scaled.shape[0] == 1:
            output = quip_sharp(scaled[0], indices, self.grid_packed_abs)
        else:
            if self.cached_dense_weight:
                self.warm_weight_cache()
                weight = self.dense
                assert weight is not None
            else:
                weight = quip_sharp_decompress(indices, self.grid_packed_abs)
            output = (scaled @ weight.T).float()
        # Keep upstream's two FP32 multiplications, not a rounded FP16 SV*32.
        output = hadamard_transform(output, self.had_right) * self.SV * 32
        output = output.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
        return output if self.bias is None else output + self.bias


class UniSVQLinear(nn.Module):
    """Identical 2-bit HadaQuant, BF16 internally with the caller's output dtype.

    M<=8 reads packed weights; M>8 uses a temporary or retained BF16 raw matrix.
    Prepare host scalars (and, if enabled, the cache) before compilation/capture.
    The timed boundary casts preserve the external model/attention/KV dtype.
    Trained checkpoints retain FP16 SU/SV on disk; the frozen HadaQuant kernels
    consume BF16 scales. That execution-time rounding is not QDQ equivalence.
    """

    Qidxs: torch.Tensor
    SU: torch.Tensor
    SV: torch.Tensor
    cb_scale: torch.Tensor
    cb_zero: torch.Tensor
    bias: torch.Tensor | None
    dense: torch.Tensor | None

    def __init__(
        self,
        packed: dict[str, torch.Tensor],
        bias: torch.Tensor | None = None,
        *,
        cached_dense_weight: bool = True,
    ):
        super().__init__()
        self.cached_dense_weight = cached_dense_weight
        self.in_features = packed["SU"].numel()
        self.out_features = packed["SV"].numel()
        for name in ("Qidxs", "SU", "SV", "cb_scale", "cb_zero"):
            self.register_buffer(name, packed[name])
        self.register_buffer("bias", bias)
        self.register_buffer("dense", None, persistent=False)
        self._cb_s: float | None = None
        self._cb_z: float | None = None

    def prepare_for_inference(self) -> None:
        # The frozen ABI accepts host floats. Extract once outside captured work,
        # after state_dict loading, as in HadaQuant's prepare_for_inference.
        if self._cb_s is None:
            self._cb_s = float(self.cb_scale.item())
            self._cb_z = float(self.cb_zero.item())

    def warm_weight_cache(self) -> None:
        self.prepare_for_inference()
        assert self._cb_s is not None and self._cb_z is not None
        if self.cached_dense_weight and self.dense is None:
            self.dense = unisvq_decompress(self.Qidxs, self._cb_s, self._cb_z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.prepare_for_inference()
        assert self._cb_s is not None and self._cb_z is not None
        scaled = x.to(torch.bfloat16).reshape(-1, self.in_features).contiguous()
        scaled = unisvq_su_had(scaled, self.SU.to(torch.bfloat16))
        if scaled.shape[0] <= 8:
            output = unisvq(scaled, self.Qidxs, self._cb_s, self._cb_z)
        else:
            if self.cached_dense_weight:
                self.warm_weight_cache()
                weight = self.dense
                assert weight is not None
            else:
                weight = unisvq_decompress(self.Qidxs, self._cb_s, self._cb_z)
            output = scaled @ weight.T
        output = unisvq_had_sv(output.contiguous(), self.SV.to(torch.bfloat16))
        output = output.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
        return output if self.bias is None else output + self.bias


class SPHQuantLinear(nn.Module):
    def __init__(self, packed: dict[str, torch.Tensor], bias: torch.Tensor | None = None):
        super().__init__()
        self.out_features = packed["packed_codes"].shape[0]
        self.in_features = packed["packed_codes"].shape[1] * 8
        for name in ("packed_codes", "radius_extra", "radius_scales", "radius_mins", "codebook"):
            self.register_buffer(name, packed[name])
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layout conversion is real inference work and remains inside forward/timing.
        output = sphquant(
            x.to(torch.float16).contiguous(),
            self.packed_codes,
            self.radius_extra,
            self.radius_scales,
            self.radius_mins,
            self.codebook,
        ).to(x.dtype)
        return output if self.bias is None else output + self.bias


class SAGELinear(nn.Module):
    """SAGE 64/4, 128/3 or 256/2; out_features excludes physical tile padding."""

    permutation: torch.Tensor

    def __init__(
        self, packed: dict[str, torch.Tensor], out_features: int, bias: torch.Tensor | None = None
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = packed["packed_codes"].shape[1] * 128
        for name in ("packed_codes", "radius_codes", "radius_grid", "codebook", "permutation"):
            self.register_buffer(name, packed[name])
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep decode gathers visible to Inductor so they can fuse with producers.
        # Eager execution still uses the native fused-gather dispatch.
        permutation = self.permutation
        if (
            torch.compiler.is_compiling()
            and x.numel() // self.in_features < 16
            and permutation.numel()
        ):
            x = x.index_select(-1, permutation)
            permutation = permutation[:0]
        output = sage(
            x.to(torch.float16).contiguous(),
            self.packed_codes,
            self.radius_codes,
            self.radius_grid,
            self.codebook,
            permutation,
            self.out_features,
        ).to(x.dtype)
        return output if self.bias is None else output + self.bias
