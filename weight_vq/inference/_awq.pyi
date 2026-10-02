from torch import Tensor


def forward(x: Tensor, qweight: Tensor, scales: Tensor, scaled_zeros: Tensor) -> Tensor: ...
