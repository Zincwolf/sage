from contextlib import nullcontext

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.hessian import LayerHessianCollectors, load_quantization_hessian
from weight_vq.blockwise.pipeline import _run_rank
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.config import QTIPConfig


def tiny_qwen():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval()


def test_replay_matches_native_qwen_sdpa_and_teacher_remains_original():
    model = tiny_qwen()
    tokens = torch.randint(0, 64, (4, 8))
    with torch.no_grad():
        expected = model.model(input_ids=tokens, use_cache=False).last_hidden_state
    state = capture_replay(model, tokens, device="cpu", microbatch_size=1)
    for layer in model.model.layers:
        collector = LayerHessianCollectors(layer, include_padding=True)
        replay = state.forward_original(layer, device="cpu", collectors=collector)
        for payload in collector.close().values():
            assert payload["ct"] == 32
        replay.loss_parts(layer, torch.arange(4), "cpu")[0].backward()
        assert layer.get_parameter("self_attn.q_norm.weight").grad is not None
        assert not state.inputs.requires_grad
    torch.testing.assert_close(model.model.norm(state.inputs), expected)


@pytest.mark.parametrize("terminal", [False, True])
def test_hessian_chunking_preserves_raw_moments_with_unequal_chunks(
    tmp_path, monkeypatch, capsys, terminal
):
    model = tiny_qwen()
    tokens = torch.randint(0, 64, (5, 8))

    def collect(chunks):
        result = {}
        for chunk in chunks:
            state = capture_replay(model, chunk, device="cpu", microbatch_size=1)
            for index, layer in enumerate(model.model.layers):
                collectors = LayerHessianCollectors(layer, include_padding=True)
                state.forward_original(layer, device="cpu", collectors=collectors)
                for name, collector in collectors.collectors.items():
                    matrix, _first_moment, count = collector.close_statistics()
                    previous, tokens_seen = result.get((index, name), (torch.zeros_like(matrix), 0))
                    result[index, name] = previous + matrix, tokens_seen + count
        return result

    expected, actual = collect([tokens]), collect(tokens.split(2))
    for key, (matrix, count) in expected.items():
        assert count == actual[key][1] == 40
        torch.testing.assert_close(matrix, actual[key][0])

    # Run the production chunk/merge loop on CPU and compare its published statistics.
    config = QTIPConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "dataset",
        output_dir=tmp_path,
        devices=[0],
        num_cpu_threads=1,
        hessian_samples=5,
        hessian_sequence_length=8,
        hessian_chunk_size=2,
    )
    torch.save({"protocol": config.calibration_recipe(), "hessian": tokens}, config.calibration)
    config.hessians.mkdir()
    monkeypatch.setattr(
        "weight_vq.blockwise.pipeline.process_group",
        lambda *args: nullcontext(DistributedContext(0, 1, torch.device("cpu"))),
    )
    monkeypatch.setattr("sys.stderr.isatty", lambda: terminal)
    _run_rank(0, config, "hessian", model, 0, 2, tmp_path / "store", config.hessians)
    progress = capsys.readouterr()
    assert not progress.out
    assert "Hessian chunk 3/3" in progress.err
    assert "6/6" in progress.err
    assert progress.err.isascii() is not terminal
    assert (config.hessians / "complete.json").exists()
    for (index, name), (matrix, count) in expected.items():
        saved, _metadata = load_quantization_hessian(
            config.hessians / f"{index}_{name}.pt", min_expert_coverage=0.0
        )
        torch.testing.assert_close(saved, matrix / count)
