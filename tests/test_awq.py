"""Standard GPTQModel-packed weights -> native HF loading -> frozen AWQ runtime.

Synthetic fixtures establish layout/loading, not activation-aware quantization quality.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AwqConfig, Qwen3Config
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
from transformers.quantizers.quantizer_awq import AwqQuantizer

from benchmarks.config import BenchmarkConfig
from benchmarks.model import benchmark
from weight_vq.inference.awq import AWQLinear
from weight_vq.integrations.awq import LLMAWQConfig


def unpack_awq(qweight):
    # Independent logical address reference, migrated from sage-kernels tests.
    n, k = qweight.shape[0] * 4, qweight.shape[1]
    row = torch.arange(n, device=qweight.device)[:, None]
    col = torch.arange(k, device=qweight.device)[None, :]
    shuffled = (col % 64 // 32) * 32 + (col // 2 % 4) * 8 + (col % 2) * 4 + col % 32 // 8
    address = (col // 64) * 64 + (row % 4) * 16 + shuffled // 4
    words = qweight[row // 4, address].int()
    return (words >> (4 * (shuffled % 4))) & 15


@pytest.fixture(scope="module")
def packed_checkpoint(tmp_path_factory):
    from gptqmodel.nn_modules.qlinear.gemv_fast_awq import LLMAwqLinear

    directory = tmp_path_factory.mktemp("awq-hf")
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        attention_bias=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
        for name, linear in list(model.named_modules()):
            if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
                continue
            n, k = linear.weight.shape
            codes = torch.randint(0, 16, (n, k))
            scales = torch.full((n, k // 128), 0.01, dtype=torch.float16)
            zeros = torch.full_like(scales, 8)
            linear.weight.data.copy_((codes - 8) * scales.repeat_interleave(128, dim=1))
            packed = LLMAwqLinear(
                bits=4, group_size=128, sym=False, desc_act=False, in_features=k, out_features=n
            )
            packed.pack(linear, scales, zeros)
            # Exact discrete comparison is appropriate for the packed ABI, not logits.
            torch.testing.assert_close(unpack_awq(packed.qweight), codes, rtol=0, atol=0)
            model.set_submodule(name, packed)
    model.config.quantization_config = AwqConfig(
        bits=4, group_size=128, format="llm-awq", sym=False, desc_act=False
    ).to_dict()
    model.save_pretrained(directory)
    return directory, model.state_dict()


def test_standard_hf_loading(packed_checkpoint):
    directory, state = packed_checkpoint
    before = (directory / "config.json").read_bytes()
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    config.quantization_config = LLMAWQConfig(**config.quantization_config).to_dict()
    loaded = AutoModelForCausalLM.from_pretrained(
        directory, config=config, dtype=torch.float16, device_map="cpu", local_files_only=True
    )
    layers = [module for module in loaded.modules() if isinstance(module, AWQLinear)]
    assert len(layers) == 7
    assert isinstance(loaded.lm_head, torch.nn.Linear)
    assert all(layer.qweight.dtype == torch.int16 for layer in layers)
    assert all(layer.scales.dtype == layer.scaled_zeros.dtype == torch.float16 for layer in layers)
    assert all(not hasattr(layer, "dense") for layer in layers)
    for name, expected in state.items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)
    assert (directory / "config.json").read_bytes() == before
    assert AUTO_QUANTIZER_MAPPING["awq"] is AwqQuantizer
    assert json.loads(before)["quantization_config"]["quant_method"] == "awq"


@pytest.mark.parametrize("override", [{"format": "gemm"}, {"bits": 8}, {"group_size": 64}])
def test_reject_other_layouts(override):
    config = {"bits": 4, "group_size": 128, "format": "llm-awq", **override}
    with pytest.raises(ValueError, match="requires"):
        LLMAWQConfig(**config)


def test_fake_fullgraph():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        layer = AWQLinear(256, 128).to("cuda")
        inputs = torch.empty(1, 9, 256, device="cuda", dtype=torch.float16)
        graph = torch.export.export(layer, (inputs,))
        assert "wvq.awq" in graph.graph_module.code


@pytest.mark.cuda
@pytest.mark.parametrize("m", [1, 7, 8, 33])
def test_awq_reference_and_compile(packed_checkpoint, m):
    directory, _ = packed_checkpoint
    config = AutoConfig.from_pretrained(directory)
    config.quantization_config = LLMAWQConfig(**config.quantization_config).to_dict()
    loaded = AutoModelForCausalLM.from_pretrained(
        directory, config=config, dtype=torch.float16, device_map="cuda:0"
    )
    layer = loaded.model.layers[0].self_attn.q_proj
    codes = unpack_awq(layer.qweight).float()
    group = torch.arange(layer.in_features, device="cuda") // 128
    weight = (codes * layer.scales[group].T.float() + layer.scaled_zeros[group].T.float()).half()
    x = torch.randn(1, m, layer.in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = torch.nn.functional.linear(x, weight, layer.bias)
        actual = layer(x)
        # Preserve the FP16 operator tolerances already used in sage-kernels.
        delta = actual.float() - expected.float()
        assert delta.norm() / expected.float().norm() < 0.008
        assert delta.abs().max() / expected.float().square().mean().sqrt() < 0.04
        compiled = torch.compile(layer, fullgraph=True, dynamic=False, mode="reduce-overhead")
        for _ in range(3):
            torch.testing.assert_close(compiled(x), actual, rtol=0, atol=0)
    torch.compiler.reset()


@pytest.mark.cuda
def test_awq_benchmark(packed_checkpoint, tmp_path):
    directory, _ = packed_checkpoint
    result, _ = benchmark(
        BenchmarkConfig(
            models={"candidate": directory},
            output_dir=tmp_path,
            prefill=9,
            decode=3,
            warmup=0,
            samples=1,
        )
    )
    method = result["methods"]["candidate"]
    assert method["method"] == "awq"
    assert method["quantization"] == {
        "bits": 4,
        "group_size": 128,
        "format": "llm-awq",
        "backend": "wvq::awq",
    }
    assert all(method["samples"][phase] for phase in ("prefill", "decode", "end_to_end"))


@pytest.mark.cuda
def test_real_awq_production(tmp_path):
    # One small real AWQ optimization, not a synthetic-packed quality claim.
    # Requires the WVQ extension build; GPTQModel may also JIT its own extensions.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from scripts.quantize_awq import main

    source, destination = tmp_path / "source", tmp_path / "packed"
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        pad_token_id=0,
        eos_token_id=None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        original = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    original.save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="2")),
        pad_token="0",
        bos_token="1",
        unk_token="2",
    )
    tokenizer.save_pretrained(source)
    calibration = tmp_path / "calibration.pt"
    train = torch.randint(3, 32, (8, 32), generator=torch.Generator().manual_seed(8))
    torch.save({"train": train, "validation": train[:1]}, calibration)
    main(source, calibration, destination, samples=8)
    assert len(AutoTokenizer.from_pretrained(destination, local_files_only=True)) == 32
    saved = AutoConfig.from_pretrained(destination, local_files_only=True)
    assert saved.quantization_config["quant_method"] == "awq"
    assert saved.quantization_config["format"] == "llm-awq"
    saved.quantization_config = LLMAWQConfig(**saved.quantization_config).to_dict()
    loaded = AutoModelForCausalLM.from_pretrained(
        destination, config=saved, dtype=torch.float16, device_map="cuda:0", local_files_only=True
    ).eval()
    with torch.inference_mode():
        logits = loaded(train[:1].cuda()).logits
        assert torch.isfinite(logits).all()
    del loaded, logits, original
    result, _ = benchmark(
        BenchmarkConfig(
            models={"real_awq": destination},
            output_dir=tmp_path / "report",
            prefill=9,
            decode=3,
            samples=1,
            warmup=0,
        )
    )
    assert result["methods"]["real_awq"]["method"] == "awq"


@pytest.mark.parametrize("option", ["--samples", "--batch_size"])
def test_production_cli_rejects_zero(option):
    from jsonargparse import auto_parser

    from scripts.quantize_awq import main

    with pytest.raises(SystemExit):
        auto_parser(main).parse_args(["source", "calibration.pt", "packed", option, "0"])


def test_production_token_contract(tmp_path, monkeypatch):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import FORMAT
    from gptqmodel.quantization.config import GcMode

    from scripts.quantize_awq import main

    train = torch.arange(64).reshape(4, 16)
    calibration = tmp_path / "calibration.pt"
    torch.save({"train": train, "validation": train + 100}, calibration)
    observed = {}

    class Model:
        def quantize(self, **kwargs):
            observed.update(kwargs)

        def save(self, path):
            observed["output"] = path

    def load(path, **kwargs):
        observed["source"] = path
        observed["config"] = kwargs["quantize_config"]
        return Model()

    monkeypatch.setattr(GPTQModel, "load", load)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    output = tmp_path / "packed"
    main(Path("source"), calibration, output, samples=2)
    qconfig = observed["config"]
    assert (qconfig.bits, qconfig.group_size, qconfig.format) == (4, 128, FORMAT.LLM_AWQ)
    assert qconfig.pack_dtype == torch.int16 and not qconfig.sym
    assert qconfig.fallback is None
    assert qconfig.gc_mode is GcMode.ON_STAGE_END
    assert observed["calibration_sort"] is None
    assert len(observed["calibration"]) == 2
    for index, row in enumerate(observed["calibration"]):
        torch.testing.assert_close(row["input_ids"], train[index])
        torch.testing.assert_close(row["attention_mask"], torch.ones(16, dtype=torch.long))
    assert observed["output"] == str(output)
