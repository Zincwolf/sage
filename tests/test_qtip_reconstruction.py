from pathlib import Path

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from transformers import Qwen3Config, Qwen3ForCausalLM

from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.config import QTIPConfig


def _worker(rank, world_size, path):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            init_method=Path(path, f"store-{world_size}").as_uri(),
            rank=rank,
            world_size=world_size,
        )
    try:
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval()
        tokens = torch.randint(0, 64, (13, 8))
        cfg = QTIPConfig(
            model=Path("/model"),
            dataset=Path("/rp"),
            output_dir=Path(path),
            devices=list(range(world_size)),
            model_dtype="float32",
            train_samples=9,
            validation_samples=4,
            reconstruction_epochs=3,
            learning_rate=5e-5,
            batch_size=8,
            reconstruction_microbatch_size=1,
        )
        layer = model.model.layers[0]
        train = capture_replay(model, tokens[:9][rank::world_size], device="cpu", microbatch_size=1)
        valid = capture_replay(model, tokens[9:][rank::world_size], device="cpu", microbatch_size=1)
        train_replay = train.forward_original(layer, device="cpu")
        valid_replay = valid.forward_original(layer, device="cpu")
        with torch.no_grad():
            layer.get_parameter("self_attn.o_proj.weight").mul_(0.5)
            layer.get_parameter("self_attn.q_norm.weight").mul_(0.8)
        before = {name: value.clone() for name, value in layer.state_dict().items()}
        trainer = DistributedReconstruction(
            train_replay,
            valid_replay,
            cfg,
            DistributedContext(rank, world_size, torch.device("cpu")),
            0,
        )
        loss_before = trainer.validation_loss(layer, torch.float32)
        trainer(layer, "test", torch.float32)
        loss_after = trainer.validation_loss(layer, torch.float32)
        assert loss_after < loss_before
        for name in (
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "mlp.down_proj.weight",
        ):
            assert not torch.equal(before[name], layer.state_dict()[name]), name
        torch.save(layer.state_dict(), Path(path, f"{world_size}-{rank}.pt"))
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def test_global_batch_training_matches_one_two_four_ranks(tmp_path):
    _worker(0, 1, str(tmp_path))
    expected = torch.load(tmp_path / "1-0.pt", weights_only=True)
    for world_size in (2, 4):
        spawn(_worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)
        for rank in range(world_size):
            actual = torch.load(tmp_path / f"{world_size}-{rank}.pt", weights_only=True)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=2e-5)
