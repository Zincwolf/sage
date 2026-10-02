"""Backend preference and quiet, conservative untimed observation."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch._C._autograd import DeviceType
from torch._functorch import config as aot_config
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, Qwen3Config, StaticCache

from benchmarks.attention import profile_attention, summarize_attention_kernels
from benchmarks.model import inspect_model_attention, measure_window
from weight_vq.inference.attention import SDPA_PRIORITY, preferred_sdpa

CUDNN = "cudnn_generated_fort_native_sdpa_sm80_flash_fprop_kernel"
EFFICIENT = "fmha_cutlassF_f16_aligned_64x128_rf_sm80"
FLASH = "void pytorch_flash::flash_fwd_kernel()"


@pytest.mark.parametrize(
    "names,expected,verified,observed",
    [
        ([CUDNN] * 2, 2, True, ["CUDNN_ATTENTION"]),
        ([CUDNN], 2, False, ["CUDNN_ATTENTION"]),
        ([CUDNN] * 3, 2, False, ["CUDNN_ATTENTION"]),
        ([CUDNN, EFFICIENT], 2, False, ["CUDNN_ATTENTION", "EFFICIENT_ATTENTION"]),
        ([EFFICIENT] * 2, 2, False, ["EFFICIENT_ATTENTION"]),
        ([FLASH] * 2, 2, False, ["FLASH_ATTENTION"]),
        (["gemm", "softmax"], 2, False, ["unknown"]),
        (["triton_poi_fused__scaled_dot_product_cudnn_attention_clone"], 1, False, ["unknown"]),
        ([], 0, False, ["unknown"]),
    ],
)
def test_attention_kernel_classification(names, expected, verified, observed):
    report = summarize_attention_kernels(names, expected)
    assert report["preferred_backend_verified"] is verified
    assert report["observed_backends"] == observed
    assert report["expected_attention_layers"] == expected


def test_sdpa_priority_and_restoration():
    original = torch.nn.attention._cur_sdpa_kernel_backends(with_priority=True)
    with preferred_sdpa():
        assert torch.nn.attention._cur_sdpa_kernel_backends(with_priority=True) == list(
            SDPA_PRIORITY
        )
    assert torch.nn.attention._cur_sdpa_kernel_backends(with_priority=True) == original
    with pytest.raises(RuntimeError, match="restore"), preferred_sdpa():
        raise RuntimeError("restore")
    assert torch.nn.attention._cur_sdpa_kernel_backends(with_priority=True) == original


@pytest.mark.parametrize("names,verified", [([CUDNN], True), ([EFFICIENT], False), ([], False)])
def test_profile_quietly_records_observed_kernels(names, verified, monkeypatch, capsys):
    events = [SimpleNamespace(name=name, device_type=DeviceType.CUDA) for name in names]
    # A CPU event alone cannot prove which kernel ran inside a captured graph.
    events.append(SimpleNamespace(name=CUDNN, device_type=DeviceType.CPU))
    calls = []

    class Profile:
        def __enter__(self):
            calls.append("profile-enter")
            return self

        def __exit__(self, *args):
            calls.append("profile-exit")

        def events(self):
            return events

    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: Profile())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("synchronize"))
    marker = object()

    def forward():
        calls.append("forward")
        return marker

    output, report = profile_attention(forward, expected_layers=1, device=torch.device("cuda"))
    assert output is marker
    assert calls == ["synchronize", "profile-enter", "forward", "synchronize", "profile-exit"]
    assert report["preferred_backend_verified"] is verified
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.cuda
@pytest.mark.parametrize("backend", [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
@torch.inference_mode()
def test_profile_compiled_sdpa_backend(backend, capsys):
    q = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 32, 255, 128, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    mask = (torch.arange(255, device="cuda") < 129).view(1, 1, 1, 255)
    with aot_config.patch(enable_autograd_cache=False), sdpa_kernel(backend):
        forward = torch.compile(
            torch.nn.functional.scaled_dot_product_attention,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )

        def run():
            return forward(q, k, v, attn_mask=mask)

        try:
            for _ in range(3):
                run()
            output, report = profile_attention(run, expected_layers=1, device=q.device)
            assert report["observed_backends"] == [backend.name]
            assert report["preferred_backend_verified"] is (backend == SDPBackend.CUDNN_ATTENTION)
            with sdpa_kernel(SDPBackend.MATH):
                expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            torch.testing.assert_close(output, expected, rtol=0.01, atol=0.001)
        finally:
            torch.compiler.reset()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "WARNING: probe:" not in captured.err


@pytest.mark.cuda
@torch.inference_mode()
def test_model_backend_probe_rewarms_without_changing_generation(capsys):
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
    torch.manual_seed(44)
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.float16, attn_implementation="sdpa"
    ).eval()
    cast(torch.nn.Module, model).cuda()
    ids = torch.arange(3, 14, device="cuda").view(1, -1)
    prompt = ids[:, :4].contiguous()
    tokens = [ids[:, i : i + 1].contiguous() for i in range(4, 11)]
    cache = StaticCache(config=config, max_cache_len=11)
    model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache.reset()
    cast(torch.nn.Module, model).compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
    try:
        before = None
        for _ in range(3):
            _, before = measure_window(model, cache, prompt, tokens, "end_to_end", 8)
        assert before is not None
        reports = inspect_model_attention(model, cache, prompt, tokens, "end_to_end", 8)
        assert set(reports) == {"prefill", "decode"}
        for report in reports.values():
            assert report["observed_backends"] == ["CUDNN_ATTENTION"]
            assert report["preferred_backend_verified"]
        measure_window(model, cache, prompt, tokens, "end_to_end", 8)
        with torch.compiler.set_stance("fail_on_recompile"):
            sample, after = measure_window(model, cache, prompt, tokens, "end_to_end", 8)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert sample["synchronized_wall_ms"] > 0
        assert "WARNING: dense/end_to_end/" not in capsys.readouterr().err
    finally:
        model = cache = None
        torch.compiler.reset()
        torch.cuda.empty_cache()
