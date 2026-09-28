import pytest
import torch

from weight_vq.checkpoint import synthetic_sphquant

from .reference import decode_sphquant, unpack_sphquant


@pytest.mark.parametrize("k,n", [(128, 1), (256, 17), (2560, 9), (9728, 3), (16384, 1)])
def test_synthetic_layout(k, n):
    packed = synthetic_sphquant(k, n, seed=42)
    assert packed["packed_codes"].shape == (n, k // 8)
    assert packed["radius_extra"].shape == (n, k // 128)
    assert packed["packed_codes"].dtype == torch.int16
    assert packed["radius_extra"].dtype == torch.int32
    # 2.5 bits/weight plus one FP16 64x8 codebook.
    assert sum(t.numel() * t.element_size() for t in packed.values()) == n * k * 5 // 16 + 1024


def test_all_discrete_codes():
    words = torch.arange(65536).reshape(4096, 16)
    extra = torch.full((4096, 1), 0xE4E4E4E4).int()
    signs, indices, radii = unpack_sphquant(words.short(), extra)
    # Known packed fields check the arithmetic reference independently, including signed storage.
    expected_signs = 1 - 2 * ((words.unsqueeze(-1) >> torch.arange(8)) & 1)
    torch.testing.assert_close(signs, expected_signs, rtol=0, atol=0)
    torch.testing.assert_close(indices, (words >> 8) & 63, rtol=0, atol=0)
    torch.testing.assert_close(radii, (words >> 14) | ((torch.arange(16) % 4) << 2), rtol=0, atol=0)


def test_constant_and_zero_grids():
    packed = synthetic_sphquant(128, 2)
    packed["radius_scales"].zero_()
    packed["radius_mins"][0].zero_()
    packed["radius_mins"][1].fill_(0.25)
    dense = decode_sphquant(packed)
    assert torch.count_nonzero(dense[0]) == 0
    torch.testing.assert_close(
        dense[1].reshape(-1, 8).norm(dim=1), torch.full((16,), 0.25), rtol=5e-4, atol=0
    )


def test_seed_reproducible():
    first = synthetic_sphquant(128, 2, 10)
    second = synthetic_sphquant(128, 2, 10)
    for field in first:
        torch.testing.assert_close(first[field], second[field], rtol=0, atol=0)
    torch.testing.assert_close(
        first["codebook"].float().norm(dim=-1), torch.ones(64), rtol=5e-4, atol=0
    )
    assert torch.all(first["codebook"] >= 0)
