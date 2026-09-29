"""Native HF execution after standard checkpoint loading; requires a local build.

This validates loading/device placement, cache and generation, not model quality.
Independent per-Linear numerical references live in the other inference tests.
"""

import gc

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, StaticCache

from weight_vq import checkpoint
from weight_vq.inference.linear import SAGELinear, SPHQuantLinear
from weight_vq.integrations.transformers import prepare_for_inference

from .reference import decode_sage, decode_sphquant, error_metrics

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def real_width_source(tmp_path_factory):
    # One random native Qwen3 layer at the production projection shapes. QTIP's
    # frozen GEMV exports only these shapes; a tiny-width toy would not test it.
    directory = tmp_path_factory.mktemp("qwen3-packed") / "source"
    torch.manual_seed(9)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16).eval()
    model.save_pretrained(directory)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[UNK]": 2})),
        pad_token="[PAD]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(directory)
    return directory


@pytest.mark.parametrize("method", ["sage", "sphquant", "qtip", "quip-sharp", "unisvq"])
@torch.inference_mode()
def test_loaded_model_prefill_decode_generate(real_width_source, tmp_path, method, record_property):
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 60 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    )
    destination = tmp_path / "packed"
    checkpoint.main(checkpoint.CheckpointConfig(real_width_source, method, destination, seed=44))
    # Cover direct CUDA loading and CPU inspection followed by a device move.
    direct = AutoModelForCausalLM.from_pretrained(
        destination, device_map="cuda", dtype="auto", attn_implementation="sdpa"
    ).eval()
    moved = AutoModelForCausalLM.from_pretrained(
        destination, device_map="cpu", dtype="auto", attn_implementation="sdpa"
    ).eval()
    moved.to(device="cuda")  # pyright: ignore[reportCallIssue] - HF's inferred base model type.
    prepare_for_inference(moved)
    tokens = torch.tensor([[1, 3, 5, 7]], device="cuda")
    caches = [StaticCache(config=model.config, max_cache_len=8) for model in (direct, moved)]
    try:
        for step in range(3):
            inputs = tokens if step == 0 else tokens[:, -1:]
            results = [
                model(inputs, past_key_values=cache, use_cache=True).logits
                for model, cache in zip((direct, moved), caches, strict=True)
            ]
            assert all(torch.isfinite(result).all() for result in results)
            metrics = error_metrics(*results)
            for key, value in metrics.items():
                record_property(f"step_{step}_{key}", value)
            # QuIP#'s upstream atomic GEMV is nondeterministic even on identical
            # inputs; low-precision layers amplify it.
            if method == "quip-sharp" and step > 0:
                record_property(
                    f"step_{step}_consistency_skipped", "upstream QuIP# atomic GEMV nondeterminism"
                )
            else:
                torch.testing.assert_close(results[0], results[1], rtol=2e-3, atol=2e-3)
            assert all(cache.get_seq_length() == 4 + step for cache in caches)
        # HF's generated-model protocol annotation omits runtime model fields.
        generated = direct.generate(  # pyright: ignore[reportAttributeAccessIssue]
            tokens, max_new_tokens=3, do_sample=False, eos_token_id=None
        )
        assert generated.shape == (1, 7)
        assert torch.equal(generated[:, :4], tokens)
        assert not any(name.endswith(".dense") for name in direct.state_dict())
    finally:
        del direct, moved, caches
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("method", ["sage", "sphquant"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_spherical_external_dtype_boundary(method, dtype):
    factory = checkpoint.synthetic_sage if method == "sage" else checkpoint.synthetic_sphquant
    packed = factory(256, 32, seed=2)
    if method == "sage":
        layer = SAGELinear(packed, 32).cuda()
        weight = decode_sage(packed, 32)
    else:
        layer = SPHQuantLinear(packed).cuda()
        weight = decode_sphquant(packed)
    for rows in (1, 17):
        x = torch.randn(rows, 256, generator=torch.Generator().manual_seed(43)).to(dtype).cuda()
        actual = layer(x)
        expected = torch.nn.functional.linear(x.half(), weight.half().cuda()).to(dtype)
        assert actual.dtype == dtype
        assert error_metrics(actual, expected)["relative_l2"] < 5e-3
