from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing.spawn import spawn

from weight_vq.qtip.distributed import DistributedContext
from weight_vq.qtip.hessian import MaskedHessianCollector


def _collective_worker(rank: int, rendezvous: str, output: str) -> None:
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
    try:
        distributed = DistributedContext(rank, 2, torch.device("cpu"))
        linear = nn.Linear(1, 1, bias=False)
        value = torch.tensor([[[float(rank + 1)]]])
        collector = MaskedHessianCollector(linear)
        collector.set_position_mask(torch.ones(1, 1, dtype=torch.bool))
        linear(value)
        hessian = collector.close(distributed)

        linear.weight.grad = torch.tensor([[float(rank + 1)]])
        distributed.sync_gradients(linear)
        torch.save(
            {"hessian": hessian, "gradient": linear.weight.grad}, Path(output, f"rank-{rank}.pt")
        )
    finally:
        dist.destroy_process_group()


def test_collectives_aggregate_raw_hessian_and_gradients(tmp_path: Path) -> None:
    rendezvous = tmp_path / "store"
    spawn(_collective_worker, args=(str(rendezvous), str(tmp_path)), nprocs=2, join=True)

    for rank in range(2):
        result = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert result["hessian"]["ct"] == 2
        torch.testing.assert_close(result["hessian"]["flatH"], torch.tensor([2.5]))
        torch.testing.assert_close(result["gradient"], torch.tensor([[3.0]]))
