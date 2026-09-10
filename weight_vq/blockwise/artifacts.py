"""Shared projection order and complete-block artifact contracts."""

from __future__ import annotations

from pathlib import Path

from weight_vq import utils

PROJECTIONS = (
    ("self_attn.v_proj", "v", "qkv", "col"),
    ("self_attn.q_proj", "q", "qkv", "col"),
    ("self_attn.k_proj", "k", "qkv", "col"),
    ("self_attn.o_proj", "o", "o", "row"),
    ("mlp.up_proj", "up", "up", "col"),
    ("mlp.gate_proj", "gate", "up", "col"),
    ("mlp.down_proj", "down", "down", "row"),
)
HESSIAN_NAMES = ("qkv", "o", "up", "down")


def projections(num_experts: int = 0) -> tuple[tuple[str, str, str, str], ...]:
    if not num_experts:
        return PROJECTIONS
    return PROJECTIONS[:4] + tuple(
        (
            f"mlp.experts.{expert}.{projection}_proj",
            f"expert{expert}_{projection}",
            f"expert{expert}_{'down' if projection == 'down' else 'up'}",
            "row" if projection == "down" else "col",
        )
        for projection in ("gate", "up", "down")
        for expert in range(num_experts)
    )


def quant_order(num_experts: int = 0) -> list[tuple[str, str, str, str, str]]:
    return [
        (attribute, short, hessian, short, parallel)
        for attribute, short, hessian, parallel in projections(num_experts)
    ]


def projection_paths(
    layer_index: int, checkpoint_dir: str | Path, num_experts: int = 0
) -> tuple[Path, ...]:
    checkpoint_dir = Path(checkpoint_dir)
    return tuple(
        checkpoint_dir / f"{layer_index}_{short}.pt"
        for _attribute, short, _hessian, _parallel in projections(num_experts)
    )


def layer_complete(layer_index: int, checkpoint_dir: str | Path, num_experts: int = 0) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    return (
        (checkpoint_dir / f"{layer_index}.complete").is_file()
        and (checkpoint_dir / f"{layer_index}.auxiliary.pt").is_file()
        and (not num_experts or (checkpoint_dir / f"{layer_index}.coverage.json").is_file())
        and all(
            path.is_file() for path in projection_paths(layer_index, checkpoint_dir, num_experts)
        )
    )


def hessian_complete(layer_index: int, hessian_dir: str | Path, num_experts: int = 0) -> bool:
    hessian_dir = Path(hessian_dir)
    names = dict.fromkeys(spec[2] for spec in projections(num_experts))
    return all((hessian_dir / f"{layer_index}_{name}.pt").is_file() for name in names)


def atomic_write_completion(
    layer_index: int, checkpoint_dir: str | Path, num_experts: int = 0
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    missing = [
        path
        for path in projection_paths(layer_index, checkpoint_dir, num_experts)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"layer {layer_index} is missing projection checkpoints: {missing}")
    marker = checkpoint_dir / f"{layer_index}.complete"
    if not (checkpoint_dir / f"{layer_index}.auxiliary.pt").is_file():
        raise RuntimeError(f"layer {layer_index} is missing auxiliary state")
    if num_experts and not (checkpoint_dir / f"{layer_index}.coverage.json").is_file():
        raise RuntimeError(f"layer {layer_index} is missing expert coverage")
    utils.atomic_write_text(marker, "blockwise-full-state-v4\n")
