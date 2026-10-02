"""Frozen llm-awq quantization_new W4G128 runtime; no GPTQModel dependency."""

import torch
from torch import nn


@torch.library.custom_op("wvq::awq", mutates_args=(), device_types="cuda")
def awq(
    x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, scaled_zeros: torch.Tensor
) -> torch.Tensor:
    from weight_vq.inference import _awq  # pyright: ignore[reportMissingModuleSource]

    return _awq.forward(x, qweight, scales, scaled_zeros)


@awq.register_fake
def _awq_fake(x, qweight, scales, scaled_zeros):
    return x.new_empty((*x.shape[:-1], qweight.shape[0] * 4))


class AWQLinear(nn.Module):
    """Consume GPTQModel FORMAT.LLM_AWQ buffers without repacking or dense caches."""

    qweight: torch.Tensor
    scales: torch.Tensor
    scaled_zeros: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, in_features: int, out_features: int, bias: torch.Tensor | None = None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        groups = ((in_features // 128 + 7) // 8) * 8
        self.register_buffer(
            "qweight", torch.empty(out_features // 4, in_features, dtype=torch.int16)
        )
        self.register_buffer("scales", torch.empty(groups, out_features, dtype=torch.float16))
        self.register_buffer("scaled_zeros", torch.empty(groups, out_features, dtype=torch.float16))
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = awq(x.contiguous(), self.qweight, self.scales, self.scaled_zeros)
        return output if self.bias is None else output + self.bias
