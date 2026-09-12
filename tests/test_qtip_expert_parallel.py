"""Real QTIP/NCCL ownership, group reconstruction and payload round trips."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing.spawn import spawn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from weight_vq.blockwise.artifacts import quant_order
from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.hessian import load_quantization_hessian
from weight_vq.blockwise.models import split_work_experts
from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.blockwise.replay import capture_replay
from weight_vq.qtip.config import QTIPConfig


def _parallel_worker(rank, world_size, path, dtype, epochs):
    # QTIP and its CUDA dependencies belong to the method environment.
    from lib.algo import finetune
    from lib.codebook.bitshift import bitshift_codebook
    from lib.linear import QuantizedLinear

    torch.set_num_threads(2)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=Path(path, "store").as_uri(),
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        config = QTIPConfig(
            model=Path(path, "unused-model"),
            dataset=Path(path, "unused-data"),
            output_dir=Path(path),
            devices=list(range(world_size)),
            num_cpu_threads=2,
            model_dtype=str(dtype).removeprefix("torch."),
            train_samples=3,
            validation_samples=2,
            reconstruction_epochs=epochs,
            reconstruction_microbatch_size=1,
            batch_size=3,
            min_expert_coverage=0.5,
            L=8,
            tlut_bits=6,
        )
        order = quant_order(3)
        groups = [
            short if i < 4 else attribute.rsplit(".", 1)[1]
            for i, (attribute, short, *_rest) in enumerate(order)
        ]
        owners = [None] * 4 + list(range(3)) * 3
        owners = [None if owner is None else owner % world_size for owner in owners]
        reference = {}
        reference_before = {}
        reference_rng = torch.empty(0, dtype=torch.uint8)
        modes = ("replicated", "sharded", "resume") if not epochs else ("replicated", "sharded")
        for mode in modes:
            torch.manual_seed(23)
            hf = Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=128,
                moe_intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts=3,
                num_experts_per_tok=1,
            )
            hf._attn_implementation = "sdpa"
            model = Qwen3MoeForCausalLM(hf).to(dtype).eval()
            model.set_experts_implementation("eager")
            layer = deepcopy(model.model.layers[0]).to(device)
            trainer = None
            if epochs:
                tokens = torch.randint(0, 32, (5, 8))
                train = capture_replay(
                    model, tokens[:3][rank::world_size], device=device, microbatch_size=1
                )
                valid = capture_replay(
                    model, tokens[3:][rank::world_size], device=device, microbatch_size=1
                )
                trainer = DistributedReconstruction(
                    train.forward_original(layer, device=device),
                    valid.forward_original(layer, device=device),
                    config,
                    DistributedContext(rank, world_size, device),
                    config.seed,
                )
            split_work_experts(layer)
            layer.requires_grad_(True)
            codebook = bitshift_codebook(
                L=8,
                K=2,
                V=2,
                tlut_bits=6,
                decode_mode="quantlut_sym",
                tlut_path=Path(path, "tlut.pt"),
            )
            loaded = []
            reconstructed = []
            result = {}

            def load_hessian(filename, loaded=loaded):
                loaded.append(Path(filename).name)
                return load_quantization_hessian(filename, config.min_expert_coverage)

            def reconstruct(
                block, name, orig_dtype, trainer=trainer, mode=mode, reconstructed=reconstructed
            ):
                assert trainer is not None
                end = next(i for i, spec in enumerate(order) if name == f"0_{spec[1]}") + 1
                for i, (attribute, *_rest) in enumerate(order):
                    assert isinstance(
                        block.get_submodule(attribute), QuantizedLinear if i < end else nn.Linear
                    )
                trainer.materialize_layer(block, orig_dtype)
                # Compare to replicated DP on the SAME rank and sample shard,
                # avoiding single-vs-DP floating reduction order as a confounder.
                state = {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}
                if mode == "replicated":
                    reference_before[name] = state
                else:
                    torch.testing.assert_close(state, reference_before[name], rtol=0, atol=0)
                trainer(block, name, orig_dtype)
                reconstructed.append(name)

            def progress(completed, payloads, result=result):
                assert set(payloads) == {spec[1] for spec in order[:completed]}
                result.clear()
                result.update(deepcopy(payloads))

            completed = 7 if mode == "resume" else 0
            resume = None
            if completed:
                resume = {
                    "completed": completed,
                    "projections": {spec[1]: reference[spec[1]] for spec in order[:completed]},
                }
            grad_enabled = torch.is_grad_enabled()
            try:
                finetune.quantize_finetune_decoder_layer(
                    layer,
                    order,
                    0,
                    codebook,
                    config,
                    device,
                    None,
                    None,
                    hessian_loader=load_hessian,
                    reconstruction_groups=groups,
                    projection_owners=None if mode == "replicated" else owners,
                    reconstruction_callback=reconstruct,
                    progress_callback=progress,
                    resume_state=resume,
                    write_checkpoints=rank == 0,
                )
            finally:
                torch.set_grad_enabled(grad_enabled)
            expected_loads = [
                f"0_{spec[2]}.pt"
                for i, spec in enumerate(order)
                if i >= completed and (mode == "replicated" or owners[i] in (None, rank))
            ]
            assert loaded == expected_loads
            assert reconstructed == (
                [f"0_{order[i][1]}" for i in (0, 1, 2, 3, 6, 9, 12)] if epochs else []
            )
            assert all(p.device.type == "cpu" for p in layer.parameters())
            assert codebook.lut.device.type == "cpu"
            rng = torch.cuda.get_rng_state(device)
            if mode == "replicated":
                reference = deepcopy(result)
                reference_rng = rng
            else:
                torch.testing.assert_close(rng, reference_rng, rtol=0, atol=0)
                assert result.keys() == reference.keys()
                for name, payload in result.items():
                    for key, value in payload.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, reference[name][key], rtol=0, atol=0)
                        else:
                            assert value == reference[name][key]
            # Every rank must receive the same complete initial/trained payload,
            # including coverage metadata, BF16 signs, scalar scale and int16 indices.
            if rank == 0:
                torch.save(result, Path(path, f"{mode}.pt"))
            dist.barrier()
            published = torch.load(Path(path, f"{mode}.pt"), weights_only=True)
            for name, payload in result.items():
                for key, value in payload.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, published[name][key], rtol=0, atol=0)
                    else:
                        assert value == published[name][key]
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype,epochs", [(torch.float32, 0), (torch.bfloat16, 1)])
def test_real_expert_assignment_matches_replicated_quantization(
    tmp_path, world_size, dtype, epochs
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"NCCL expert parallel test requires {world_size} GPUs")
    # Three experts exercise uneven assignment on two GPUs and an entirely
    # non-owning rank on four GPUs. Gate/up and down have different input sizes.
    config = QTIPConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "data",
        output_dir=tmp_path,
        devices=list(range(world_size)),
    )
    config.hessians.mkdir()
    config.checkpoints.mkdir()
    for _, short, hessian, *_rest in quant_order(3):
        n = 256 if short.endswith("_down") else 128
        matrix = torch.eye(n)
        payload = {
            "flatH": torch.cat([row[: i + 1] for i, row in enumerate(matrix)]),
            "n": n,
            "ct": 8,
        }
        if short.startswith("expert"):
            # Empirical, zero-coverage and below-threshold experts respectively.
            payload.update(ct=(8, 0, 2)[int(short[6])], total_tokens=8)
        torch.save(payload, config.hessians / f"0_{hessian}.pt")
    torch.manual_seed(17)
    torch.save(torch.randn(64, 2), tmp_path / "tlut.pt")
    spawn(
        _parallel_worker,
        args=(world_size, str(tmp_path), dtype, epochs),
        nprocs=world_size,
        join=True,
    )
