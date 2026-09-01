"""Single-GPU QTIP Qwen3-VL stage worker."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from copy import deepcopy
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("hessian", "quantize"))
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--calibration", type=Path, required=True)
    cli.add_argument("--hessians", type=Path, required=True)
    cli.add_argument("--checkpoints", type=Path, required=True)
    cli.add_argument("--train-samples", type=int, required=True)
    cli.add_argument("--validation-samples", type=int, required=True)
    cli.add_argument("--batch-size", type=int, required=True)
    cli.add_argument("--reconstruction-epochs", type=int, required=True)
    cli.add_argument("--learning-rate", type=float, required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--layer-start", type=int, required=True)
    cli.add_argument("--layer-end", type=int, required=True)
    return cli


def qtip_arguments(args: argparse.Namespace) -> Namespace:
    """Map the frozen Qwen3-VL W2 recipe to QTIP's native LDLQ API."""
    return Namespace(
        base_model=str(args.model),
        save_path=str(args.checkpoints),
        in_hess_path=str(args.hessians),
        num_cpu_threads=8,
        sigma_reg=1e-2,
        scale_override=0.9,
        ft_lr=args.learning_rate,
        ft_update_freq=1,
        ft_epochs=args.reconstruction_epochs,
        ft_valid_freq=1,
        ft_early_stop=1,
        ft_grad_ckpt=False,
        ft_train_lut=False,
        td_x=16,
        td_y=16,
        L=16,
        K=2,
        V=2,
        tlut_bits=9,
        decode_mode="quantlut_sym",
        use_fp64=False,
        seed=args.seed,
        disable_kernel=True,
        split_for_tp=False,
        tp_rank=8,
    )


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)

    import torch  # pyright: ignore[reportMissingImports]
    from lib import utils  # pyright: ignore[reportMissingImports]
    from lib.algo import finetune  # pyright: ignore[reportMissingImports]
    from lib.codebook.bitshift import bitshift_codebook  # pyright: ignore[reportMissingImports]
    from lib.qwen3_vl.common import (  # pyright: ignore[reportMissingImports]
        HESSIAN_NAMES,
        atomic_write_completion,
        audit_qwen3_vl,
        freeze_non_qtip_parameters,
        get_layers,
        hessian_complete,
        layer_complete,
        projection_paths,
        quant_order,
    )
    from lib.qwen3_vl.hessian import LayerHessianCollectors  # pyright: ignore[reportMissingImports]
    from lib.qwen3_vl.replay import (  # pyright: ignore[reportMissingImports]
        MultimodalReconstructionAdapter,
        apply_post_layer_context,
        build_reconstruction_batches,
        capture_split,
        forward_original_layer,
    )
    from paroquant.optim.util import (  # pyright: ignore[reportMissingImports]
        empty_cache,
        load_model,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QTIP hessian and quantize stages require CUDA")
    if not args.calibration.is_file():
        raise FileNotFoundError(args.calibration)
    if args.stage == "quantize":
        missing = [
            index
            for index in range(args.layer_start, args.layer_end)
            if not hessian_complete(index, args.hessians)
        ]
        if missing:
            raise FileNotFoundError(f"missing QTIP Hessians for layers: {missing}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    native_args = qtip_arguments(args)
    tlut_path = args.checkpoints / "tlut-l16-b9-v2.pt"
    if not tlut_path.is_file():
        raise FileNotFoundError(tlut_path)
    tlut = torch.load(tlut_path, map_location="cpu", weights_only=True)
    codebook = bitshift_codebook(
        L=16,
        K=2,
        V=2,
        tlut_bits=9,
        decode_mode="quantlut_sym",
        tlut=tlut,
    )

    model = load_model(
        str(args.model),
        device_map="cpu",
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    audit_qwen3_vl(model)
    layers = get_layers(model)
    model.to(device)
    train = capture_split(
        model,
        layers,
        manifest_path=args.calibration,
        model_path=args.model,
        split="train",
        sample_count=args.train_samples,
        batch_size=args.batch_size,
        device=device,
    )
    validation = None
    if args.stage == "quantize":
        validation = capture_split(
            model,
            layers,
            manifest_path=args.calibration,
            model_path=args.model,
            split="validation",
            sample_count=args.validation_samples,
            batch_size=args.batch_size,
            device=device,
        )
    model.cpu()
    empty_cache()

    train_inputs = train.first_layer_inputs
    validation_inputs = (
        validation.first_layer_inputs if validation is not None else None
    )
    train_retained = deepcopy(train.retained_kwargs)
    validation_retained = (
        deepcopy(validation.retained_kwargs) if validation is not None else None
    )
    for index in range(args.layer_end):
        layer = layers[index].to(device).eval()
        selected = args.layer_start <= index < args.layer_end
        train_retained_before = deepcopy(train_retained)
        validation_retained_before = (
            deepcopy(validation_retained)
            if validation_retained is not None
            else None
        )
        collectors = (
            LayerHessianCollectors(layer, include_padding=False)
            if args.stage == "hessian"
            and selected
            and not hessian_complete(index, args.hessians)
            else None
        )
        train_outputs, train_retained = forward_original_layer(
            layer,
            train_inputs,
            train,
            index,
            device=device,
            collectors=collectors,
            autocast_dtype=dtype,
            retained_kwargs=train_retained_before,
        )
        if collectors is not None:
            hessians = collectors.close()
            for name in HESSIAN_NAMES:
                finetune.atomic_torch_save(
                    hessians[name],
                    args.hessians / f"{index}_{name}.pt",
                )
            del hessians, collectors
            empty_cache()

        validation_outputs = None
        if validation is not None:
            if validation_inputs is None or validation_retained_before is None:
                raise RuntimeError("validation replay state was not initialized")
            validation_outputs, validation_retained = forward_original_layer(
                layer,
                validation_inputs,
                validation,
                index,
                device=device,
                autocast_dtype=dtype,
                retained_kwargs=validation_retained_before,
            )

        if (
            args.stage == "quantize"
            and selected
            and not layer_complete(index, args.checkpoints)
        ):
            (args.checkpoints / f"{index}.complete").unlink(missing_ok=True)
            freeze_non_qtip_parameters(layer)
            if validation is None or validation_outputs is None:
                raise RuntimeError("quantize requires validation replay batches")
            reconstruction = None
            if args.reconstruction_epochs:
                reconstruction = MultimodalReconstructionAdapter(
                    build_reconstruction_batches(
                        train_inputs,
                        train_outputs,
                        train,
                        index,
                        retained_kwargs=train_retained_before,
                    ),
                    build_reconstruction_batches(
                        validation_inputs,
                        validation_outputs,
                        validation,
                        index,
                        retained_kwargs=validation_retained_before,
                    ),
                    seed=args.seed + index,
                    mask_padding=True,
                )
            finetune.quantize_finetune_decoder_layer(
                layer,
                quant_order(),
                index,
                codebook,
                native_args,
                device,
                None,
                None,
                reconstruction_adapter=reconstruction,
            )
            missing = [
                path
                for path in projection_paths(index, args.checkpoints)
                if not path.is_file()
            ]
            if missing:
                raise RuntimeError(f"layer {index} missing checkpoints: {missing}")
            atomic_write_completion(index, args.checkpoints)

        train_inputs = apply_post_layer_context(train_outputs, train, index)
        if validation is not None and validation_outputs is not None:
            validation_inputs = apply_post_layer_context(
                validation_outputs,
                validation,
                index,
            )
        layers[index] = torch.nn.Identity()
        del layer, train_outputs, validation_outputs
        utils.clean()

    print(
        json.dumps(
            {
                "stage": args.stage,
                "physical_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "layer_range": [args.layer_start, args.layer_end],
                "train_sample_ids": train.sample_ids,
                "validation_sample_ids": (
                    validation.sample_ids if validation is not None else []
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
