"""Real QuIP#/UniSVQ expert ownership, RNG and reconstruction over NCCL."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from weight_vq.blockwise.artifacts import projections
from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.models import split_work_experts
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.blockwise.replay import capture_replay
from weight_vq.quip_sharp.config import QuIPSharpConfig
from weight_vq.unisvq.config import UniSVQConfig
from weight_vq.unisvq.reconstruction import UniSVQReconstruction


def _worker(rank, world_size, root, method, epochs, recovery):
    # Actual method dependencies and quantizers, including their CUDA codebooks.
    trainer_type: Any
    if method == "quip":
        from weight_vq.quip_sharp import quantization

        config_type = QuIPSharpConfig
        trainer_type = DistributedReconstruction
    else:
        from weight_vq.unisvq import quantization

        config_type = UniSVQConfig
        trainer_type = UniSVQReconstruction
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=(Path(root) / "store").as_uri(),
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        specs = projections(3)
        owners = {
            short: int(attribute.split(".")[2]) % world_size
            for attribute, short, *_ in specs
            if attribute.startswith("mlp.experts.")
        }
        reference_before = {}
        reference_final = {}
        reference_rng = None
        original_loader = quantization.load_quantization_hessian
        for mode in ("replicated", "sharded"):
            config: Any = config_type(
                model=Path(root) / "unused-model",
                dataset=Path(root) / "unused-data",
                output_dir=Path(root) / mode,
                devices=list(range(world_size)),
                num_cpu_threads=2,
                model_dtype="bfloat16" if epochs else "float32",
                train_samples=3,
                validation_samples=2,
                sequence_length=4,
                reconstruction_epochs=epochs,
                reconstruction_microbatch_size=1,
                batch_size=2,
                quip_tune_iters=1,
                min_expert_coverage=0.5,
                gradient_checkpointing=bool(epochs),
            )
            if rank == 0:
                config.hessians.mkdir(parents=True)
                config.checkpoints.mkdir()
                for _, short, name, *_ in specs:
                    n = 256 if short.endswith("_down") else 128
                    matrix = torch.eye(n)
                    payload = dict(
                        flatH=torch.cat([row[: i + 1] for i, row in enumerate(matrix)]),
                        mu=torch.zeros(n),
                        n=n,
                        ct=8,
                    )
                    if short.startswith("expert"):
                        payload.update(ct=(8, 0, 2)[int(short[6])], total_tokens=8)
                    torch.save(payload, config.hessians / f"0_{name}.pt")
            dist.barrier()
            torch.manual_seed(23)
            hf = Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=128,
                moe_intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_experts=3,
                num_experts_per_tok=1,
            )
            hf._attn_implementation = "sdpa"
            dtype = torch.bfloat16 if epochs else torch.float32
            model: Any = Qwen3MoeForCausalLM(hf)
            model = model.to(dtype).eval()
            model.set_experts_implementation("eager")
            layer = deepcopy(model.model.layers[0]).to(device)
            tokens = torch.randint(0, 32, (5, 4))
            train = capture_replay(
                model, tokens[:3][rank::world_size], device=device, microbatch_size=1
            )
            valid = capture_replay(
                model, tokens[3:][rank::world_size], device=device, microbatch_size=1
            )
            train = train.forward_original(layer, device=device)
            valid = valid.forward_original(layer, device=device)
            split_work_experts(layer)
            layer.requires_grad_(True)
            calls = []
            loaded = []

            class Observed(trainer_type):
                def __call__(self, block, name, dtype, mode=mode, calls=calls):
                    self.materialize_layer(block, torch.float32 if method == "unisvq" else dtype)
                    state = {
                        key: value.detach().cpu().clone()
                        for key, value in block.state_dict().items()
                    }
                    if mode == "replicated":
                        reference_before[name] = state
                    else:
                        torch.testing.assert_close(state, reference_before[name], rtol=0, atol=0)
                    super().__call__(block, name, dtype)
                    assert all(parameter.is_cuda for parameter in block.parameters())
                    calls.append(name)

            retry_pending = []
            cholesky = torch.linalg.cholesky

            def load_hessian(path, coverage, loaded=loaded, retry_pending=retry_pending):
                loaded.append(Path(path).name)
                if recovery and "expert0" in Path(path).name:
                    retry_pending.append(True)
                return original_loader(path, coverage)

            def factor(matrix, retry_pending=retry_pending, cholesky=cholesky):
                if retry_pending:
                    retry_pending.pop()
                    # Native factorization failure, then the real FP64 LDLQ path.
                    return cholesky(
                        -torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
                    )
                return cholesky(matrix)

            trainer = Observed(
                train, valid, config, DistributedContext(rank, world_size, device), 7
            )
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(quantization, "load_quantization_hessian", load_hessian)
                if recovery:
                    patch.setattr(torch.linalg, "cholesky", factor)
                quantization.quantize_layer(
                    layer,
                    0,
                    config,
                    trainer,
                    projection_owners=owners if mode == "sharded" else None,
                )
            expected_calls = (
                ["0"]
                if method == "unisvq"
                else [
                    f"0_{short}"
                    for _, short, *_ in specs
                    if not short.startswith("expert") or short.startswith("expert2")
                ]
            )
            assert calls == expected_calls
            expected_loads = [
                f"0_{name}.pt"
                for _, short, name, *_ in specs
                if mode == "replicated" or owners.get(short, rank) == rank
            ]
            if recovery and mode == "sharded":
                # All three expert groups recover, hence one replicated retry per group.
                for _, _short, name, *_ in specs:
                    assert loaded.count(f"0_{name}.pt") >= expected_loads.count(f"0_{name}.pt")
                assert len(loaded) > len(expected_loads)
            else:
                assert loaded == expected_loads
            assert all(parameter.device.type == "cpu" for parameter in layer.parameters())
            assert all(buffer.device.type == "cpu" for buffer in layer.buffers())
            state = {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
            if mode == "replicated":
                reference_final = state
                reference_rng = rng
            else:
                torch.testing.assert_close(state, reference_final, rtol=0, atol=0)
                torch.testing.assert_close(rng, reference_rng, rtol=0, atol=0)
            torch.save(state, Path(root) / f"{mode}-{rank}.pt")
            dist.barrier()
            torch.testing.assert_close(
                state, torch.load(Path(root) / f"{mode}-0.pt", weights_only=True), rtol=0, atol=0
            )
            if rank == 0 and mode == "sharded":
                for _, short, *_ in specs:
                    original = torch.load(
                        Path(root) / "replicated" / "checkpoints" / f"0_{short}.pt",
                        weights_only=True,
                    )
                    saved = torch.load(config.checkpoints / f"0_{short}.pt", weights_only=True)
                    assert saved.keys() == original.keys()
                    for key, value in original.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
                        else:
                            assert saved[key] == value
            del layer, model, trainer, state
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method", ["quip", "unisvq"])
@pytest.mark.parametrize("world_size,epochs", [(2, 0), (2, 1), (4, 1)])
def test_expert_ownership_matches_replicated(tmp_path, method, world_size, epochs):
    if torch.cuda.device_count() < world_size:
        pytest.skip("NCCL test requires the requested GPUs")
    spawn(
        _worker,
        args=(world_size, str(tmp_path), method, epochs, False),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="NCCL test requires two GPUs")
def test_fp64_recovery_preserves_group_rng(tmp_path):
    spawn(_worker, args=(2, str(tmp_path), "quip", 0, True), nprocs=2, join=True)
