import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from weight_vq.qtip.hessian import LayerHessianCollectors
from weight_vq.qtip.replay import capture_replay


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


def test_hessian_chunking_preserves_raw_moments_with_unequal_chunks():
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
                    matrix, count = collector.close_raw()
                    previous, tokens_seen = result.get((index, name), (torch.zeros_like(matrix), 0))
                    result[index, name] = previous + matrix, tokens_seen + count
        return result

    expected, actual = collect([tokens]), collect(tokens.split(2))
    for key, (matrix, count) in expected.items():
        assert count == actual[key][1] == 40
        torch.testing.assert_close(matrix, actual[key][0])
