"""SAGE's method controls on the shared spherical-quantization pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from weight_vq.sphquant.config import SPHQuantConfig


@dataclass(frozen=True)
class SAGEConfig(SPHQuantConfig):
    """SAGE: run optimizes with ESG/ABS, exports dense QDQ, and validates the result.

    Args:
        esg: Enable energy-stratified grouping and weighted spherical fitting.
        adaptive_bits: Select each matrix's direction/radius split from 64/4,
            128/3, and 256/2, retaining the winning initialization.
        energy_power: Teacher-energy exponent for ESG fitting. ABS scores use raw energy.
        allow_partial: Allow ABS to skip matrices with zero calibration tokens and
            use codebook_bits for them. The default requires complete coverage.
        codebook_bits: Fixed direction-index bits when adaptive_bits is disabled,
            or for unmeasured matrices when allow_partial is enabled.
    """

    # Disable either contribution to reproduce the paper's component ablations.
    esg: bool = True
    adaptive_bits: bool = True
    # Tempering exponent used only for fitting, never for ABS scoring.
    energy_power: float = 0.75
    # Permit ABS to leave unrouted matrices unmeasured, using codebook_bits.
    allow_partial: bool = False
    # Inherited codebook_bits applies without ABS or to allowed unmeasured matrices.
