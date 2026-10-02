"""Small native HF generation-loop checks, not full-model throughput acceptance."""

import threading
from typing import cast

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config, StaticCache

from benchmarks.model import measure_window

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("batch,prefill,decode", [(1, 1, 1), (1, 4, 8), (2, 4, 4)])
@torch.inference_mode()
def test_generation_windows(batch, prefill, decode, monkeypatch):
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.float16, attn_implementation="sdpa"
        ).eval()
    cast(torch.nn.Module, model).cuda()
    ids = torch.arange(3, 3 + batch * (prefill + decode - 1), device="cuda").reshape(batch, -1)
    prompt = ids[:, :prefill].contiguous()
    tokens = [ids[:, pos : pos + 1].contiguous() for pos in range(prefill, ids.shape[1])]
    cache = StaticCache(config=config, max_cache_len=ids.shape[1])
    model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache.reset()
    expected = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
        prompt,
        attention_mask=torch.ones_like(prompt),
        past_key_values=cache,
        max_new_tokens=decode,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        use_cache=True,
        disable_compile=True,
    )[:, prefill:].cpu()
    main_thread = threading.get_ident()
    device_sync = torch.cuda.synchronize
    event_sync = torch.cuda.Event.synchronize
    seen = []
    waits = []

    def counted_model(input_ids, **kwargs):
        seen.append(input_ids.clone())
        return cast(torch.nn.Module, model)(input_ids, **kwargs)

    def counted_device_sync(*args, **kwargs):
        waits.append((threading.get_ident(), len(seen)))
        return device_sync(*args, **kwargs)

    def counted_event_sync(event):
        waits.append((threading.get_ident(), len(seen)))
        return event_sync(event)

    try:
        for compiled in (False, True):
            if compiled:
                cast(torch.nn.Module, model).compile(
                    fullgraph=True, dynamic=False, mode="reduce-overhead"
                )
            for phase in ("prefill", "decode", "end_to_end"):
                # Cover both compilation and CUDA Graph preparation, then enforce
                # the same no-recompile policy as the actual benchmark.
                for _ in range(2):
                    measure_window(model, cache, prompt, tokens, phase, decode)
                seen.clear()
                waits.clear()
                with monkeypatch.context() as patch:
                    patch.setattr(torch.cuda, "synchronize", counted_device_sync)
                    patch.setattr(torch.cuda.Event, "synchronize", counted_event_sync)
                    with (
                        torch.compiler.set_stance("fail_on_recompile"),
                        torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU]
                        ) as profile,
                    ):
                        sample, generated = measure_window(
                            counted_model, cache, prompt, tokens, phase, decode
                        )
                if compiled:
                    # Eager HF prefill checks the GPU cache offset in its SDPA
                    # mask heuristic. The production benchmark is fullgraph:
                    # that heuristic is disabled and must not extract scalars.
                    assert not {"aten::item", "aten::_local_scalar_dense"} & {
                        event.key for event in profile.key_averages()
                    }
                expected_calls = 1 if phase == "prefill" else decode
                assert len(seen) == expected_calls
                assert [(tid, count) for tid, count in waits if tid == main_thread] == [
                    (main_thread, 1 if phase == "decode" else 0),
                    (main_thread, expected_calls),
                ]
                assert cache.get_seq_length() == (prefill if phase == "prefill" else ids.shape[1])
                assert 0 < sample["cuda_event_ms"] <= sample["synchronized_wall_ms"]
                torch.testing.assert_close(seen[0], prompt, rtol=0, atol=0)
                if phase == "end_to_end":
                    assert generated is not None and generated.device.type == "cpu"
                    torch.testing.assert_close(generated, expected, rtol=0, atol=0)
                    for position, input_ids in enumerate(seen[1:]):
                        torch.testing.assert_close(
                            input_ids.cpu(), generated[:, position : position + 1], rtol=0, atol=0
                        )
                    assert 0 < sample["gpu_ttft_ms"] <= sample["cuda_event_ms"]
                    assert 0 < sample["first_token_cpu_ms"] <= sample["synchronized_wall_ms"]
                    assert len([tid for tid, _ in waits if tid != main_thread]) == 1
                else:
                    assert generated is None
                    assert "gpu_ttft_ms" not in sample and "first_token_cpu_ms" not in sample
                    if phase == "decode":
                        for input_ids, fixed_token in zip(seen[1:], tokens, strict=True):
                            torch.testing.assert_close(input_ids, fixed_token, rtol=0, atol=0)
    finally:
        model = cache = None
        torch.compiler.reset()
        torch.cuda.empty_cache()
