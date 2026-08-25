"""SPHQuant stage adapters."""

from __future__ import annotations

import sys

from weight_vq.calibration import load_calibration_metadata
from weight_vq.sphquant.config import SPHQuantConfig


def optimize(config: SPHQuantConfig) -> None:
    metadata = load_calibration_metadata(config.calibration)

    # NOTE: still some fixed hyperparams inside, only make them configurable in the
    # future when necessary
    arguments = {
        "model": str(config.model),
        "params": [f"radius:{config.learning_rate},codebook:{config.learning_rate}"],
        "epochs": [config.epochs],
        "method": "sphquant",
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "loss": "mse",
        "optimizer": "adam",
        "use_lr_scheduler": False,
        "group_size": 128,
        "n_bit": 2,
        "num_rotations": 0,
        "skipped_modules": [],
        "datasets": ["wikitext2", "c4", "redpajama"],
        "val_dataset": "pileval",
        "train_size": metadata.train_samples,
        "validation_size": metadata.validation_samples,
        "batch_size": config.batch_size,
        "val_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "seqlen": metadata.sequence_length,
        "calibration_path": str(config.calibration),
        "multimodal_calibration_path": None,
        "calibration_limit": config.calibration_limit,
        "max_layers": config.max_layers,
        "sph_sample_size": config.sample_size,
        "sph_kmeans_iterations": config.kmeans_iterations,
        "model_dtype": "float16",
        "cache_shards": config.cache_shards,
        "output_dir": str(config.checkpoints),
        "resume": config.resume,
        "checkpointing": config.checkpointing,
        "seed": config.seed,
        "use_wandb": config.use_wandb,
    }

    # Imported here because the root dispatcher environment does not install method deps.
    from paroquant.cli import optimize as pq_opt  # pyright: ignore[reportMissingImports]

    # ParoQuant owns the optimization semantics; this adapter only supplies its Config.
    paroquant_config = pq_opt.Config(**arguments)
    reference_parse = pq_opt.simple_parsing.parse
    pq_opt.simple_parsing.parse = lambda *_args, **_kwargs: paroquant_config
    try:
        pq_opt.main()
    finally:
        pq_opt.simple_parsing.parse = reference_parse


def export(config: SPHQuantConfig) -> None:
    # ParoQuant appends model.name when writing optimization checkpoints.
    arguments = [
        "--model",
        str(config.model),
        "--result-dir",
        str(config.checkpoints / config.model.name),
        "--output-path",
        str(config.exported_model),
        "--mode",
        "pseudo",
    ]

    # Imported here because the root dispatcher environment does not install method deps.
    from paroquant.cli import convert as converter  # pyright: ignore[reportMissingImports]

    # The reference main() parses sys.argv directly, so replace it temporarily.
    original_argv = sys.argv
    sys.argv = ["paroquant.cli.convert", *arguments]
    try:
        converter.main()
    finally:
        sys.argv = original_argv


def validate(config: SPHQuantConfig) -> None:
    """Load an exported HF model and print **sampled** linear layer statistics."""
    # Imported here because the root dispatcher environment does not install method deps.
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import AutoModelForCausalLM  # pyright: ignore[reportMissingImports]

    # Loading on one GPU validates the HF config, artifacts, tensor shapes,
    # standard dense architecture, and the prerequisite used by evaluation.
    model = AutoModelForCausalLM.from_pretrained(
        config.exported_model,
        torch_dtype="auto",
        device_map=0,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Non-finite parameter in exported model: {name}")

    print(
        "name\tnumel\tsampled\tmean\tstd\tmin\tp01\tp05\tp50\tp95\tp99\tmax"
        "\tabs_p95\tabs_p99\tabs_p999\tabs_max"
    )
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight.detach().flatten().float()
            stride = max(1, (weight.numel() + 999_999) // 1_000_000)
            sample = weight[::stride]
            quantiles = torch.quantile(
                sample,
                torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=sample.device),
            )
            abs_quantiles = torch.quantile(
                sample.abs(),
                torch.tensor([0.95, 0.99, 0.999], device=sample.device),
            )
            print(
                name,
                weight.numel(),
                sample.numel(),
                weight.mean().item(),
                weight.std().item(),
                weight.min().item(),
                *(value.item() for value in quantiles),
                weight.max().item(),
                *(value.item() for value in abs_quantiles),
                weight.abs().max().item(),
                sep="\t",
            )
