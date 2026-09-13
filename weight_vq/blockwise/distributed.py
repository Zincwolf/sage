"""Single-node data parallel primitives for blockwise quantization."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import glog
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
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        present = torch.tensor(
            [parameter.grad is not None for parameter in parameters],
            device=self.device,
            dtype=torch.int32,
        )
        self.sum_(present)
        for parameter, used in zip(parameters, present.tolist(), strict=True):
            # Globally unused experts must keep grad=None: Adam should skip them,
            # including momentum updates, just as in the single-rank global batch.
            if used:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()


def quantization_attempts(
    distributed: DistributedContext, sharded: bool, retries: list[Any]
) -> Generator[bool, None, None]:
    """Retry a group replicated if FP64 recovery changed its random draw sequence."""
    if not sharded:
        yield False
        return
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(distributed.device)
    yield True
    recovered = torch.tensor(bool(retries), device=distributed.device, dtype=torch.int32)
    if distributed.sum_(recovered).item():
        glog.info("FP64 recovery: repeating quantization group with replicated ownership")
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, distributed.device)
        yield False


def broadcast_quantized(
    payloads: dict[str, dict[str, Any]], owners: dict[str, int], distributed: DistributedContext
) -> None:
    """Exchange QuIP#/UniSVQ CPU checkpoints after independent expert work."""
    # Standard collectives carry scalar metadata and tensor bytes separately.
    # NCCL cannot broadcast int16 directly (UniSVQ also supports packed indices).
    tensor_fields = ("Qidxs", "SU", "SV", "A", "B", "scaleWH", "bias")
    for owner in sorted(set(owners.values())):
        names = [name for name, rank in owners.items() if rank == owner]
        metadata: list[Any] = [None]
        if distributed.rank == owner:
            metadata[0] = {
                name: (
                    {
                        key: value
                        for key, value in payloads[name].items()
                        if key not in tensor_fields
                    },
                    {
                        key: None if value is None else (value.shape, value.dtype)
                        for key, value in payloads[name].items()
                        if key in tensor_fields
                    },
                )
                for name in names
            }
        dist.broadcast_object_list(metadata, src=owner, device=distributed.device)
        for name in names:
            scalars, tensors = metadata[0][name]
            data = payloads[name] if distributed.rank == owner else dict(scalars)
            for key, spec in tensors.items():
                if spec is None:
                    data[key] = None
                    continue
                shape, dtype = spec
                value = (
                    data[key].to(distributed.device).contiguous()
                    if distributed.rank == owner
                    else torch.empty(shape, dtype=dtype, device=distributed.device)
                )
                dist.broadcast(value.reshape(-1).view(torch.uint8), src=owner)
                if distributed.rank != owner:
                    data[key] = value.cpu()
                del value
            payloads[name] = data


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
