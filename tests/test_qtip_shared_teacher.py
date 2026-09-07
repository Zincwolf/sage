from pathlib import Path

import torch
from torch.multiprocessing.spawn import spawn
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM


def _worker(rank: int, source: str, output: str):
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        source, device_map="cpu", attn_implementation="sdpa"
    ).eval()
    assert all(not parameter.is_shared() for parameter in model.parameters())
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
        model.model.layers[0].self_attn.q_norm.weight.add_(rank + 1)
    torch.save(logits, Path(output, f"{rank}.pt"))


def test_workers_load_independent_cpu_teachers(tmp_path):
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).eval()
    source = tmp_path / "model"
    model.save_pretrained(source)
    expected = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    spawn(_worker, args=(str(source), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        torch.testing.assert_close(torch.load(tmp_path / f"{rank}.pt", weights_only=True), expected)
