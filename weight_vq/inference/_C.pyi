"""Public signature of the compiled pybind11 extension."""

from torch import Tensor

def sph_gemv_forward_cuda(
    input: Tensor,
    packed_codes: Tensor,
    radius_extra: Tensor,
    radius_scales: Tensor,
    radius_mins: Tensor,
    codebook: Tensor,
    group_size: int,
) -> Tensor: ...
def sage_gemv_forward_cuda(
    input: Tensor,
    packed_codes: Tensor,
    radius_codes: Tensor,
    radius_grid: Tensor,
    codebook: Tensor,
    permutation: Tensor,
    out_features: int,
) -> Tensor: ...
