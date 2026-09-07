from pathlib import Path

from weight_vq.qtip.artifacts import (
    PROJECTIONS,
    atomic_write_completion,
    layer_complete,
    projection_paths,
    quant_order,
)


def test_quant_order_matches_qtip_finetune_contract() -> None:
    assert quant_order() == [
        ("self_attn.v_proj", "v", "qkv", "v", "col"),
        ("self_attn.q_proj", "q", "qkv", "q", "col"),
        ("self_attn.k_proj", "k", "qkv", "k", "col"),
        ("self_attn.o_proj", "o", "o", "o", "row"),
        ("mlp.up_proj", "up", "up", "up", "col"),
        ("mlp.gate_proj", "gate", "up", "gate", "col"),
        ("mlp.down_proj", "down", "down", "down", "row"),
    ]


def test_completion_marker_covers_all_projection_checkpoints(tmp_path: Path) -> None:
    expected = tuple(
        tmp_path / f"4_{short}.pt" for _attribute, short, _hessian, _parallel in PROJECTIONS
    )
    assert projection_paths(4, tmp_path) == expected
    assert not layer_complete(4, tmp_path)

    for path in expected:
        path.touch()
    assert not layer_complete(4, tmp_path)

    (tmp_path / "4.auxiliary.pt").touch()
    atomic_write_completion(4, tmp_path)
    assert layer_complete(4, tmp_path)
