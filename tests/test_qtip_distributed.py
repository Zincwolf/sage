from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing.spawn import spawn

from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.hessian import MaskedHessianCollector


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
        experts = nn.ParameterList([nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))])
        optimizer = torch.optim.Adam(experts.parameters(), lr=0.01)
        # Seed Adam momentum for both experts, then leave the second globally unused.
        for parameter in experts:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        untouched = experts[1].detach().clone()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            experts[0].grad = torch.ones_like(experts[0])
        distributed.sync_gradients(experts)
        assert experts[1].grad is None
        torch.testing.assert_close(experts[0].grad, torch.ones(1))
        optimizer.step()
        torch.testing.assert_close(experts[1], untouched, atol=0, rtol=0)
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
