"""Native trained-checkpoint export contract, without synthetic export helpers."""

import json
from dataclasses import replace
from typing import cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from weight_vq.blockwise.artifacts import PROJECTIONS, atomic_write_completion
from weight_vq.blockwise.pipeline import export, validate
from weight_vq.quip_sharp.config import QuIPSharpConfig


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quip_checkpoint_packed_roundtrip(tmp_path, dtype, capsys):
    torch.manual_seed(5)
    config = QuIPSharpConfig(tmp_path / "source", tmp_path / "data", tmp_path / "run", [0])
    assert config.export_compressed is False
    compressed = replace(config, export_compressed=True, layer_end=1)
    # Export policy does not alter the quantization recipe, including old recipes.
    recipe = config.recipe(model_family="qwen3", layer_end=1)
    assert compressed.recipe(model_family="qwen3", layer_end=1) == recipe
    assert "export_compressed" not in recipe
    hf = Qwen3Config(
        vocab_size=16,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=True,
        attention_bias=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(hf, dtype=dtype).eval()
    model.generation_config.max_new_tokens = 3
    model.save_pretrained(config.model)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
        chat_template="{{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(config.model)
    config.checkpoints.mkdir(parents=True)
    (config.checkpoints / "quip_sharp_config.json").write_text(json.dumps(recipe))
    saved_states = {}
    auxiliary = {"input_layernorm.weight": torch.full((128,), 0.75)}
    for attribute, short, *_ in PROJECTIONS:
        linear = model.model.layers[0].get_submodule(attribute)
        n, k = linear.weight.shape
        saved = {
            "Qidxs": torch.randint(-(2**63), 2**63 - 1, (n, k // 32), dtype=torch.int64),
            # Deliberately not representable in FP16; export must not lose FT updates.
            "SU": torch.linspace(0.912345, 1.112345, k),
            "SV": torch.linspace(-0.012345, 0.012345, n),
        }
        torch.save(saved, config.checkpoints / f"0_{short}.pt")
        saved_states[attribute] = saved
        if linear.bias is not None:
            auxiliary[f"{attribute}.bias"] = torch.full_like(linear.bias, 0.125)
    torch.save(auxiliary, config.checkpoints / "0.auxiliary.pt")
    atomic_write_completion(0, config.checkpoints)
    export(compressed)
    with pytest.raises(FileExistsError):
        export(compressed)
    quant = json.loads((config.exported_model / "config.json").read_text())["quantization_config"]
    assert quant["synthetic_seed"] is None
    assert quant["method"] == "quip-sharp"
    assert len(quant["modules"]) == 7
    assert str(config.model) not in json.dumps(quant)
    # Neither original weights nor native checkpoint files are runtime dependencies.
    config.model.rename(tmp_path / "hidden-source")
    config.checkpoints.rename(tmp_path / "hidden-checkpoints")
    # HF's annotation omits the output_loading_info tuple return.
    loaded, info = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportGeneralTypeIssues]
        config.exported_model,
        dtype="auto",
        device_map="cpu",
        output_loading_info=True,
        local_files_only=True,
    )
    assert not any(info.values())
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert loaded.generation_config.max_new_tokens == 3
    assert (
        AutoTokenizer.from_pretrained(config.exported_model).chat_template
        == tokenizer.chat_template
    )
    for attribute, saved in saved_states.items():
        linear = loaded.model.layers[0].get_submodule(attribute)
        assert linear.SU.dtype == linear.SV.dtype == torch.float32
        assert linear.dense is None
        for key, tensor in saved.items():
            torch.testing.assert_close(getattr(linear, key), tensor, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        if name.endswith(".weight") and name.removesuffix(".weight") in quant["modules"]:
            assert name not in loaded.state_dict()
            continue
        auxiliary_name = name.removeprefix("model.layers.0.")
        expected = cast(torch.Tensor, auxiliary.get(auxiliary_name, value)).to(value.dtype)
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)
    validate(compressed)
    assert "bits_per_weight" in capsys.readouterr().out
    resharded = tmp_path / "resharded"
    loaded.save_pretrained(resharded, max_shard_size="20KB")
    reloaded = AutoModelForCausalLM.from_pretrained(resharded, dtype="auto", device_map="cpu")
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor, rtol=0, atol=0)
