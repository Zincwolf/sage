"""Shared moments: masking, covariance convention, and Gloo/NCCL reduction."""

from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing.spawn import spawn

from weight_vq.hessian import LinearHessianCollectors, MaskedHessianCollector
from weight_vq.qtip.distributed import DistributedContext


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("include_padding", [False, True])
def test_moments_across_microbatches(center: bool, include_padding: bool) -> None:
    linear = nn.Linear(2, 2, bias=False)
    collector = MaskedHessianCollector(linear, include_padding=include_padding)
    values = torch.tensor([[[1.0, 3.0], [9.0, 7.0]], [[-1.0, 5.0], [3.0, 1.0]]])
    mask = torch.tensor([[True, False], [True, True]])
    for batch, positions in zip(values.split(1), mask.split(1), strict=True):
        collector.set_position_mask(positions)
        linear(batch)
    result = collector.close(center=center)

    selected = values.reshape(-1, 2) if include_padding else values[mask]
    mean = selected.mean(0)
    deviations = selected - mean if center else selected
    expected = deviations.T @ deviations / len(selected)
    rows, columns = torch.tril_indices(2, 2)
    torch.testing.assert_close(result["flatH"], expected[rows, columns])
    assert result["ct"] == len(selected)
    assert result["n"] == 2
    assert result["position_mask"] == ("all" if include_padding else "attention_mask")
    if center:
        torch.testing.assert_close(result["mu"], mean)
        # QuIP# adds mu mu.T back before quantization; recover QTIP's raw second moment.
        recovered = cast(torch.Tensor, result["flatH"]) + torch.outer(mean, mean)[rows, columns]
        torch.testing.assert_close(
            recovered, (selected.T @ selected / len(selected))[rows, columns]
        )
    else:
        assert "mu" not in result
    # Closing must detach the hook, not leave a collector with emptied storage active.
    linear(values)
    assert collector.matrix.numel() == collector.first_moment.numel() == 0


def test_raw_chunk_statistics_are_summed_before_normalization() -> None:
    linear = nn.Linear(2, 2, bias=False)
    values = torch.tensor([[1.0, 3.0], [9.0, 7.0], [-1.0, 5.0], [3.0, 1.0]])
    total = torch.zeros(2, 2)
    first_moment = torch.zeros(2)
    count = 0
    for chunk in (values[:1], values[1:]):
        collector = MaskedHessianCollector(linear, include_padding=True)
        linear(chunk)
        matrix, first, tokens = collector.close_statistics()
        total.add_(matrix)
        first_moment.add_(first)
        count += tokens
    assert count == len(values)
    torch.testing.assert_close(total, values.T @ values)
    torch.testing.assert_close(first_moment, values.sum(0))
    mean = first_moment / count
    centered = total / count - torch.outer(mean, mean)
    torch.testing.assert_close(centered, torch.cov(values.T, correction=0))


def test_linear_mapping_collects_each_projection_input() -> None:
    first = nn.Linear(2, 3, bias=False)
    second = nn.Linear(3, 1, bias=False)
    collectors = LinearHessianCollectors({"first": first, "second": second})
    values = torch.tensor([[[1.0, 3.0], [9.0, 7.0], [-1.0, 5.0]]])
    mask = torch.tensor([[True, False, True]])
    collectors.set_position_mask(mask)
    hidden = first(values)
    second(hidden)
    results = collectors.close()
    for name, inputs in (("first", values), ("second", hidden)):
        selected = inputs[mask].detach()
        rows, columns = torch.tril_indices(selected.shape[1], selected.shape[1])
        expected = selected.T @ selected / len(selected)
        torch.testing.assert_close(results[name]["flatH"], expected[rows, columns])
        assert results[name]["ct"] == 2


def _unequal_rank_worker(rank: int, directory: str, empty_rank: bool, backend: str) -> None:
    torch.set_num_threads(1)
    root = Path(directory)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=(root / "store").as_uri(),
        rank=rank,
        world_size=2,
        device_id=device if backend == "nccl" else None,
    )
    try:
        distributed = DistributedContext(rank, 2, device)
        linear = nn.Linear(2, 2, bias=False).to(device)
        raw = MaskedHessianCollector(linear)
        centered = MaskedHessianCollector(linear)
        uncentered = MaskedHessianCollector(linear)
        values = torch.tensor([[[1.0, 3.0], [9.0, 7.0], [-1.0, 5.0], [3.0, 1.0]]])
        local = values[:, :1] if rank == 0 else values[:, 1:]
        mask = torch.full(local.shape[:2], not (rank == 0 and empty_rank), dtype=torch.bool)
        for collector in (raw, centered, uncentered):
            collector.set_position_mask(mask)
        linear(local.to(device))
        torch.save(
            {
                "raw": raw.close_statistics(distributed),
                "centered": centered.close(distributed, center=True),
                "uncentered": uncentered.close(distributed),
            },
            root / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty_rank", [False, True])
@pytest.mark.parametrize(
    "backend",
    [
        "gloo",
        pytest.param(
            "nccl",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="NCCL moments require two GPUs"
            ),
        ),
    ],
)
def test_distributed_moments_use_global_token_count(
    tmp_path: Path, empty_rank: bool, backend: str
) -> None:
    spawn(_unequal_rank_worker, args=(str(tmp_path), empty_rank, backend), nprocs=2, join=True)
    values = torch.tensor([[1.0, 3.0], [9.0, 7.0], [-1.0, 5.0], [3.0, 1.0]])
    if empty_rank:
        values = values[1:]
    rows, columns = torch.tril_indices(2, 2)
    for rank in range(2):
        result = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        matrix, first_moment, count = result["raw"]
        assert count == len(values)
        torch.testing.assert_close(matrix, values.T @ values)
        torch.testing.assert_close(first_moment, values.sum(0))
        torch.testing.assert_close(result["centered"]["mu"], values.mean(0))
        torch.testing.assert_close(
            result["centered"]["flatH"], torch.cov(values.T, correction=0)[rows, columns]
        )
        torch.testing.assert_close(
            result["uncentered"]["flatH"], (values.T @ values / len(values))[rows, columns]
        )
        assert result["centered"]["ct"] == result["uncentered"]["ct"] == len(values)
