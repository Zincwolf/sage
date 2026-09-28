"""Real HF save/load roundtrips without CUDA, source weights, or mock loaders."""

import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from weight_vq import checkpoint, cli
from weight_vq.inference.linear import QTIPLinear, QuIPSharpLinear, UniSVQLinear
from weight_vq.inference.qtip import HADAMARD_20
from weight_vq.inference.quip_sharp import E8P_PACKED_ABS
from weight_vq.integrations.transformers import WVQConfig

METHODS = ["sage", "sphquant", "qtip", "quip-sharp", "unisvq"]


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    torch.manual_seed(9)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        tie_word_embeddings=True,
        attention_bias=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = AutoModelForCausalLM.from_config(config).half().eval()
    model.generation_config.max_new_tokens = 7
    model.save_pretrained(directory)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3})),
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        chat_template="{{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(directory)
    return directory


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_standalone_roundtrip(source, tmp_path, method, dtype):
    original = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    original.save_pretrained(source)
    destination = tmp_path / "packed"
    checkpoint.main(checkpoint.CheckpointConfig(source, method, destination, seed=5))
    config = json.loads((destination / "config.json").read_text())
    quant = config["quantization_config"]
    assert quant["quant_method"] == "wvq"
    assert quant["method"] == method
    assert quant["synthetic_seed"] == 5
    assert len(quant["modules"]) == 7
    assert str(source) not in json.dumps(config)
    source.rename(tmp_path / "unavailable-source")

    # HF's annotation omits the output_loading_info tuple return.
    loaded, info = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportGeneralTypeIssues]
        destination, dtype="auto", device_map="cpu", local_files_only=True, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert loaded.generation_config.max_new_tokens == 7
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    assert tokenizer.chat_template == "{{ messages[0]['content'] }}"
    assert loaded.model.embed_tokens.weight.dtype == dtype
    for name, tensor in original.state_dict().items():
        if name.removesuffix(".weight") in quant["modules"] and name.endswith(".weight"):
            assert name not in loaded.state_dict()
        else:
            torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
    for index, (name, module) in enumerate(original.named_modules()):
        if name not in quant["modules"]:
            continue
        generator = {
            "sage": checkpoint.synthetic_sage,
            "sphquant": checkpoint.synthetic_sphquant,
            "qtip": checkpoint.synthetic_qtip,
            "quip-sharp": checkpoint.synthetic_quip_sharp,
            "unisvq": checkpoint.synthetic_unisvq,
        }[method]
        expected = generator(module.in_features, module.out_features, 5 + index)
        actual = loaded.get_submodule(name)
        for field, tensor in expected.items():
            if isinstance(tensor, torch.Tensor):
                torch.testing.assert_close(getattr(actual, field), tensor, rtol=0, atol=0)
        if isinstance(actual, (QTIPLinear, QuIPSharpLinear, UniSVQLinear)):
            assert actual.dense is None and not actual.cached_dense_weight
        if isinstance(actual, QuIPSharpLinear):
            torch.testing.assert_close(actual.grid_packed_abs, E8P_PACKED_ABS, rtol=0, atol=0)
    assert all(not tensor.is_meta for tensor in loaded.buffers())
    assert all(not tensor.is_meta for tensor in loaded.parameters())
    # Standard re-export and HF sharding; no project-specific checkpoint writer.
    second = tmp_path / "resharded"
    loaded.save_pretrained(second, max_shard_size="20KB")
    assert (second / "model.safetensors.index.json").is_file()
    reloaded = AutoModelForCausalLM.from_pretrained(second, dtype="auto", device_map="cpu")
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor, rtol=0, atol=0)
    for shard_path in second.glob("*.safetensors"):
        with safe_open(shard_path, framework="pt") as shard:
            assert not any(
                key.endswith((".dense", ".grid_packed_abs", ".had_left", ".had_right"))
                for key in shard.keys()  # noqa: SIM118 - safe_open is not iterable
            )


def test_cli_ckpt_and_reproducibility(source, tmp_path):
    path = tmp_path / "ckpt.yaml"
    path.write_text(f"model: {source}\nmethod: sage\noutput_dir: {tmp_path / 'first'}\nseed: 7\n")
    cli.main(["data", "ckpt", "--config", str(path)])
    cli.main(["data", "ckpt", "--config", str(path), f"--output_dir={tmp_path / 'second'}"])
    first = AutoModelForCausalLM.from_pretrained(tmp_path / "first")
    second = AutoModelForCausalLM.from_pretrained(tmp_path / "second")
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert {
        spec["codebook_bits"] for spec in first.config.quantization_config.modules.values()
    } == {6, 7, 8}
    assert not (tmp_path / "calibration.pt").exists()
    with pytest.raises(FileExistsError):
        cli.main(["data", "ckpt", "--config", str(path)])
    with pytest.raises(SystemExit):
        cli.main(["data", "ckpt", "--config", str(path), "--compressed=false"])


def test_fresh_process_load_without_source(source, tmp_path):
    destination = tmp_path / "packed"
    checkpoint.main(checkpoint.CheckpointConfig(source, "unisvq", destination))
    source.rename(tmp_path / "hidden-source")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import weight_vq.integrations.transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
model, info = AutoModelForCausalLM.from_pretrained(
    sys.argv[1], local_files_only=True, output_loading_info=True
)
assert not info['missing_keys'] and not info['unexpected_keys']
assert model.model.layers[0].self_attn.q_proj.Qidxs.element_size() == 1
assert AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True).pad_token_id == 0
assert not any(name.startswith(('paroquant', 'lib.', 'sage_kernels')) for name in sys.modules)
""",
            str(destination),
        ],
        cwd=tmp_path,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0


def test_reject_other_architecture_and_format_version(source, tmp_path):
    config = json.loads((source / "config.json").read_text())
    config["model_type"] = "qwen2"
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Qwen3"):
        checkpoint.main(checkpoint.CheckpointConfig(source, "sage", tmp_path / "packed"))
    assert not (tmp_path / "packed").exists()
    with pytest.raises(ValueError, match="format version"):
        WVQConfig(format_version=2)


@pytest.mark.parametrize("method", ["qtip", "quip-sharp"])
def test_hf_rebuilds_nonpersistent_h20(tmp_path, method):
    # A single real-width layer to exercise the asymmetric H20 constant. No CUDA
    # execution or full pretrained model; only a metadata skeleton is required.
    from weight_vq.integrations.transformers import WVQQuantizer

    config = Qwen3Config(
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    quantizer = WVQQuantizer(
        WVQConfig(method=method, modules={"model.layers.0.self_attn.q_proj": {}})
    )
    quantizer.preprocess_model(model)
    layer = model.model.layers[0].self_attn.q_proj
    layer.to_empty(device="cpu")
    layer.prepare_for_inference()
    torch.testing.assert_close(layer.had_left, HADAMARD_20, rtol=0, atol=0)
    if method == "quip-sharp":
        torch.testing.assert_close(layer.grid_packed_abs, E8P_PACKED_ABS, rtol=0, atol=0)
