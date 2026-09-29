"""Serialize trained spherical state into the existing SPHQuant/SAGE CUDA ABI."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def pack_spherical(state: dict[str, torch.Tensor], *, sage: bool) -> dict[str, torch.Tensor]:
    """Pack frozen signs/indices and trained radii/angles, without refitting.

    SAGE's checkpoint stores an inverse channel permutation; inference instead
    gathers inputs with the forward permutation. The kernel codebook is FP16,
    while radius codes must be selected against the training-time FP16 grid.
    """
    # Only export needs the training package; HF inference does not import it.
    from paroquant.optim.sphlinear import direction_codebook_from_logits

    if sage != ("sage_version" in state):
        raise ValueError("spherical checkpoint method does not match the export configuration")
    entries = state["angle_logits"].shape[0]
    bits = entries.bit_length() - 1
    if entries not in ((64, 128, 256) if sage else (64,)):
        raise ValueError("packed SPHQuant requires 64/4; packed SAGE requires 64/4, 128/3 or 256/2")
    if int(state["group_size"]) != 128 or int(state["radius_bits"]) != 10 - bits:
        raise ValueError("checkpoint does not use the spherical kernel's 128-group bit split")
    n, k = state["signs"].shape
    if k % 128:
        raise ValueError("packed spherical inference requires input width divisible by 128")
    groups = k // 128
    codebook = direction_codebook_from_logits(state["angle_logits"]).half()

    # Match radius_fake_quantize: round offset/step BEFORE selecting the codes.
    # Serialization needs these integer codes and grid, not its dense STE result.
    radius = F.softplus(state["radius_logits"].float()).reshape(n, groups, 16)
    minimum = radius.amin(dim=-1)
    step = ((radius.amax(dim=-1) - minimum) / ((1 << (10 - bits)) - 1)).half()
    minimum = minimum.half()
    safe_step = torch.where(step > 0, step.float(), torch.ones_like(step, dtype=torch.float32))
    codes = ((radius - minimum.float().unsqueeze(-1)) / safe_step.unsqueeze(-1)).round()
    codes = codes.clamp(0, (1 << (10 - bits)) - 1).long()
    codes = torch.where((step > 0).unsqueeze(-1), codes, 0)

    signs = (state["signs"].reshape(n, groups, 16, 8) < 0).long()
    sign_words = (signs << torch.arange(8, device=signs.device)).sum(-1)
    low_bits = 8 - bits
    words = (
        sign_words
        | (state["direction_indices"].long().reshape(n, groups, 16) << 8)
        | ((codes & ((1 << low_bits) - 1)) << (8 + bits))
    )
    high = ((codes >> low_bits) << (2 * torch.arange(16, device=codes.device))).sum(-1)
    if not sage:
        return {
            "packed_codes": words.reshape(n, k // 8).short().contiguous(),
            "radius_extra": high.int().contiguous(),
            "radius_scales": step.contiguous(),
            "radius_mins": minimum.contiguous(),
            "codebook": codebook.contiguous(),
        }

    # A main uint32 holds two adjacent subvectors; tile as [N/16, K/128, 16, 8].
    pairs = words[..., 0::2] | (words[..., 1::2] << 16)
    grid = torch.stack((minimum, step), dim=-1)
    padding = (-n) % 16
    pairs = F.pad(pairs, (0, 0, 0, 0, 0, padding))
    high = F.pad(high, (0, 0, 0, padding))
    grid = F.pad(grid, (0, 0, 0, 0, 0, padding))
    permutation = state["channel_unpermutation"].argsort().int()
    if torch.equal(permutation, torch.arange(k, device=permutation.device, dtype=torch.int32)):
        permutation = torch.empty(0, device=permutation.device, dtype=torch.int32)
    return {
        "packed_codes": pairs.reshape(-1, 16, groups, 8).permute(0, 2, 1, 3).int().contiguous(),
        "radius_codes": high.reshape(-1, 16, groups).permute(0, 2, 1).int().contiguous(),
        "radius_grid": grid.reshape(-1, 16, groups, 2).permute(0, 2, 1, 3).contiguous(),
        "codebook": codebook.reshape(entries, 4, 2).permute(1, 0, 2).contiguous(),
        "permutation": permutation,
    }
