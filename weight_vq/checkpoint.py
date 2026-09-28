"""Synthetic packed HF artifacts; no calibration or model-quality claim.

Generators preserve the sage-kernels 4d791bf packed inference ABI. Export keeps
all non-target tensors from the source model and never saves dense QDQ targets.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weight_vq.integrations.transformers import Method, WVQConfig, packed_linear


@dataclass(frozen=True)
class CheckpointConfig:
    """Generate a standalone synthetic compressed Qwen3 HF model.

    Args:
        model: Local source HF model; supplies architecture and non-quantized weights.
        method: Packed inference method, not a training recipe.
        output_dir: Destination HF model directory; existing paths are never overwritten.
        seed: Seed for reproducible synthetic packed weights.
    """

    model: Path
    method: Method
    output_dir: Path
    seed: int = 0


def synthetic_sphquant(
    in_features: int, out_features: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    codebook = torch.nn.functional.normalize(torch.rand(64, 8, generator=rng), dim=-1).half()
    shape = (out_features, in_features // 128)
    radius_unit = (8 / in_features) ** 0.5
    # Every bit pattern is legal: sign/direction/low radius in int16, high radius in int32.
    return {
        "packed_codes": torch.randint(
            -32768, 32768, (out_features, in_features // 8), generator=rng, dtype=torch.int16
        ),
        "radius_extra": torch.randint(-(2**31), 2**31, shape, generator=rng, dtype=torch.int32),
        "radius_scales": (torch.rand(shape, generator=rng) * radius_unit / 15).half(),
        "radius_mins": (torch.rand(shape, generator=rng) * radius_unit).half(),
        "codebook": codebook,
    }


def synthetic_sage(
    in_features: int, out_features: int, seed: int = 0, *, permuted: bool = True
) -> dict[str, torch.Tensor]:
    """Choose one of 64/4, 128/3, 256/2 for this Linear using the data seed.

    Main words hold two vectors: 8 sign bits, 6/7/8 direction bits, then 2/1/0
    low radius bits. Each radius word holds sixteen 2-bit high radius fields.
    Grid stores FP16 (minimum, step); codebook shape determines the bit split.
    The permutation is synthetic, not fitted to activation statistics.
    """
    if in_features <= 0 or in_features % 128 or out_features <= 0:
        raise ValueError("SAGE requires positive N and K divisible by 128")
    rng = torch.Generator().manual_seed(seed)
    direction_bits = int(torch.randint(6, 9, (), generator=rng))
    entries = 1 << direction_bits
    tiles, groups = (out_features + 15) // 16, in_features // 128
    codebook = torch.nn.functional.normalize(torch.rand(entries, 8, generator=rng), dim=-1)
    codes = torch.randint(-(2**31), 2**31, (tiles, groups, 16, 8), generator=rng, dtype=torch.int32)
    radii = torch.randint(-(2**31), 2**31, (tiles, groups, 16), generator=rng, dtype=torch.int32)
    grid = torch.rand(tiles, groups, 16, 2, generator=rng) * (8 / in_features) ** 0.5
    grid[..., 1] /= (1 << (10 - direction_bits)) - 1
    if out_features % 16:
        codes[-1, :, out_features % 16 :] = 0
        radii[-1, :, out_features % 16 :] = 0
        grid[-1, :, out_features % 16 :] = 0
    return {
        "packed_codes": codes,
        "radius_codes": radii,
        "radius_grid": grid.half(),
        "codebook": codebook.half().reshape(entries, 4, 2).permute(1, 0, 2).contiguous(),
        "permutation": (
            torch.randperm(in_features, generator=rng, dtype=torch.int32)
            if permuted
            else torch.empty(0, dtype=torch.int32)
        ),
    }


def _diagonal_scales(
    in_features: int, out_features: int, rng: torch.Generator, *, sv_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    # Wscale is already folded into SV, as in the upstream inference modules.
    su = 2 * torch.randint(0, 2, (in_features,), generator=rng) - 1
    sv = 2 * torch.randint(0, 2, (out_features,), generator=rng) - 1
    sv = sv * (0.5 + torch.rand(out_features, generator=rng)) / in_features**0.5
    return {"SU": su.half(), "SV": sv.to(sv_dtype)}


def synthetic_quip_sharp(in_features: int, out_features: int, seed: int = 0) -> dict[str, object]:
    """E8P12 inference ABI from wvq/third_party/quip-sharp.

    Qidxs is the final 16-row/64-column tiled layout, not row-major E8 indices.
    Each int64 contains four absolute indices and bit-interleaved signs. All
    bit patterns are legal; no grid or Hadamard workspace belongs in this file.
    """
    if in_features <= 0 or in_features % 64 or out_features <= 0 or out_features % 16:
        raise ValueError("E8P12 requires K divisible by 64 and N divisible by 16")
    rng = torch.Generator().manual_seed(seed)
    return {
        "Qidxs": torch.randint(
            -(2**63), 2**63 - 1, (out_features, in_features // 32), generator=rng, dtype=torch.int64
        ),
        **_diagonal_scales(in_features, out_features, rng, sv_dtype=torch.float16),
        "metadata": {"codebook": "E8P12", "codebook_version": 1},
    }


def synthetic_qtip(in_features: int, out_features: int, seed: int = 0) -> dict[str, object]:
    """QTIP's 2-bit quantlut_sym CUDA ABI from wvq/third_party/qtip.

    Random words are already in the kernel's byte-interleaved 32x32 layout
    (finetune.py's packed_inference_kernel branch), including the MMA ordering
    within each 16x16 tile. Every circular bitstream is a legal trellis. This
    must NOT be loaded into the sequential-trellis fallback Linear. The small
    tunable LUT is checkpoint state, not the expanded 65536-entry decode LUT.
    """
    if in_features <= 0 or in_features % 32 or out_features <= 0 or out_features % 32:
        raise ValueError("QTIP's packed kernel layout requires K and N divisible by 32")
    rng = torch.Generator().manual_seed(seed)
    return {
        "trellis": torch.randint(
            -32768,
            32768,
            ((out_features // 16) * (in_features // 16), 32),
            generator=rng,
            dtype=torch.int16,
        ),
        **_diagonal_scales(in_features, out_features, rng, sv_dtype=torch.float32),
        "tlut": torch.randn(512, 2, generator=rng).half(),
        "metadata": {
            "td_x": 16,
            "td_y": 16,
            "L": 16,
            "K": 2,
            "V": 2,
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
            "packed_inference_kernel": True,
        },
    }


def synthetic_unisvq(in_features: int, out_features: int, seed: int = 0) -> dict[str, object]:
    """UniSVQ restricted identical 2-bit checkpoint for the HadaQuant ABI.

    Each uint8 holds four scalars, high bits first. BF16 affine constants
    reproduce the identical grid after BF16 rounding; GEMV uses the unrounded
    affine weights. SU/SV are BF16 here, unlike the original PTQ checkpoint.
    """
    if in_features <= 0 or in_features % 128 or out_features <= 0 or out_features % 128:
        raise ValueError("UniSVQ's block Hadamard requires K and N divisible by 128")
    rng = torch.Generator().manual_seed(seed)
    scales = _diagonal_scales(in_features, out_features, rng, sv_dtype=torch.bfloat16)
    return {
        "Qidxs": torch.randint(
            0, 256, (out_features, in_features // 4), generator=rng, dtype=torch.uint8
        ),
        "SU": scales["SU"].bfloat16(),
        "SV": scales["SV"],
        "cb_scale": torch.tensor(1 / 1.25**0.5, dtype=torch.bfloat16),
        "cb_zero": torch.tensor(-1.5 / 1.25**0.5, dtype=torch.bfloat16),
        "metadata": {"codebook": "identical", "codebook_bit": 2, "codebook_version": 0},
    }


def main(config: CheckpointConfig) -> None:
    source = config.model.expanduser()
    destination = config.output_dir.expanduser()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    hf_config = AutoConfig.from_pretrained(source, local_files_only=True)
    if hf_config.model_type != "qwen3" or getattr(hf_config, "quantization_config", None):
        raise ValueError("synthetic packed export requires an unquantized Qwen3 dense source")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype="auto", device_map="cpu", local_files_only=True
    ).eval()
    synthetic = {
        "sage": synthetic_sage,
        "sphquant": synthetic_sphquant,
        "quip-sharp": synthetic_quip_sharp,
        "qtip": synthetic_qtip,
        "unisvq": synthetic_unisvq,
    }[config.method]
    quantization = WVQConfig(method=config.method, synthetic_seed=config.seed)
    # Enumeration matches sage-kernels' deterministic seed convention. Replace
    # one projection at a time so the exporter does not keep two whole models.
    for index, (name, module) in enumerate(model.named_modules()):
        if not name.startswith("model.layers.") or not isinstance(module, torch.nn.Linear):
            continue
        generated = synthetic(module.in_features, module.out_features, config.seed + index)
        packed = {key: value for key, value in generated.items() if isinstance(value, torch.Tensor)}
        spec: dict[str, int | bool] = {}
        if config.method == "sage":
            spec = {
                "codebook_bits": packed["codebook"].shape[1].bit_length() - 1,
                "permuted": bool(packed["permutation"].numel()),
            }
        quantization.modules[name] = spec
        model.set_submodule(
            name, packed_linear(config.method, packed, module.out_features, module.bias)
        )
    model.config.quantization_config = quantization
    # The source path is neither a dependency nor useful published metadata.
    model.config._name_or_path = ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ckpt-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "model"
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(staging)
        os.replace(staging, destination)
    print(destination)
