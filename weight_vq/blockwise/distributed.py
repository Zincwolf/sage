"""Single-node data parallel primitives for blockwise quantization."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def sum_(self, value: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def global_count(self, local_count: int) -> float:
        count = torch.tensor(float(local_count), device=self.device, dtype=torch.float64)
        return self.sum_(count).item()

    def sync_gradients(self, module: nn.Module) -> None:
        if self.world_size == 1:
            return
        for parameter in module.parameters():
            if parameter.requires_grad:
                # A shuffled global batch can contain no samples owned by a rank.
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()


@contextmanager
def process_group(
    rank: int, devices: list[int], rendezvous: Path
) -> Generator[DistributedContext, None, None]:
    device = torch.device(f"cuda:{devices[rank]}")
    torch.cuda.set_device(device)
    if len(devices) > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=rendezvous.as_uri(),
            rank=rank,
            world_size=len(devices),
            device_id=device,
        )
    try:
        yield DistributedContext(rank, len(devices), device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
