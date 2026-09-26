"""SAGE's method controls on the shared spherical-quantization pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from weight_vq.sphquant.config import SPHQuantConfig


@dataclass(frozen=True)
class SAGEConfig(SPHQuantConfig):
    # Disable either contribution to reproduce the paper's component ablations.
    esg: bool = True
    adaptive_bits: bool = True
    # Tempering exponent used only for fitting, never for ABS scoring.
    energy_power: float = 0.75
    # Permit ABS to leave unrouted matrices unmeasured, using codebook_bits.
    allow_partial: bool = False
    # Inherited codebook_bits applies without ABS or to allowed unmeasured matrices.
