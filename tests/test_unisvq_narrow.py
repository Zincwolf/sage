"""Narrow-axis RHT must preserve token independence and its exact derivative."""

import pytest
import torch
from scipy.linalg import hadamard


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native Hadamard requires CUDA")
@pytest.mark.parametrize("width", [32, 128, 2560])
def test_ptq_transform_matches_channelwise_matrix_and_gradient(width):
    from unisvq.utils.matmul_had import get_hadK, matmul_hadU, matmul_hadU_cuda

    torch.manual_seed(31)
    size = min(width, 128)
    matrix = torch.from_numpy(hadamard(size)).to(device="cuda", dtype=torch.float32) / size**0.5
    x = torch.randn(2, 3, width, device="cuda", requires_grad=True)
    direction = torch.randn_like(x)
    had, k = get_hadK(width)
    actual = matmul_hadU_cuda(x, had, k)
    expected = (x.reshape(-1, size) @ matrix).reshape_as(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(matmul_hadU(x.detach().cpu()).cuda(), expected)
    (actual * direction).sum().backward()
    torch.testing.assert_close(x.grad, (direction.reshape(-1, size) @ matrix.T).reshape_as(x))
    single = matmul_hadU_cuda(x[:1, :1], had, k)
    torch.testing.assert_close(single, actual[:1, :1], rtol=0, atol=0)
