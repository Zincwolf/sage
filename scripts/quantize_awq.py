"""Produce standard HF LLM_AWQ W4G128 checkpoints with GPTQModel, outside WVQ PTQ."""

from pathlib import Path
from typing import cast

import torch
from jsonargparse import auto_cli
from jsonargparse.typing import PositiveInt


def main(
    model: Path,
    calibration: Path,
    output_dir: Path,
    # jsonargparse creates these validating int subclasses dynamically.
    samples: PositiveInt = 128,  # pyright: ignore[reportInvalidTypeForm]
    batch_size: PositiveInt = 1,  # pyright: ignore[reportInvalidTypeForm]
    device: str = "cuda:0",
) -> None:
    """Quantize a local causal LM; export weights and tokenizer with GPTQModel.save.

    Calibration must have been encoded with this model's tokenizer. Use the first
    ``samples`` train rows unchanged, without retokenization, sorting or concatenation.
    Validation rows are not used. This script does not score model quality.
    CUDA caches are reclaimed at stage boundaries, not per calibration microbatch.
    First use may build GPTQModel's own extensions; inference benchmarks instead
    use WVQ's migrated llm-awq runtime. No source-model download is requested.

    Args:
        model: Local source HF model directory.
        calibration: WVQ calibration.pt containing packed train token IDs.
        output_dir: New directory for the self-contained standard AWQ checkpoint.
        samples: Calibration-row budget (all rows if fewer are available).
        batch_size: Calibration forwarding batch size; controls memory use.
        device: Single CUDA device used for quantization.
    """
    # Production-only dependency: inference uses neither this package nor its JIT loader.
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import FORMAT, AWQConfig, QuantizeConfig
    from gptqmodel.quantization.config import GcMode

    target = torch.device(device)
    torch.cuda.set_device(target)
    output_dir.mkdir(parents=True, exist_ok=False)
    train = torch.load(calibration, map_location="cpu", weights_only=True)["train"][:samples]
    quantization = AWQConfig(
        bits=4,
        group_size=128,
        sym=False,
        format=FORMAT.LLM_AWQ,
        pack_dtype=torch.int16,
        device=device,
        fallback=None,
        offload_to_disk=False,
        # Stage boundaries bound idle-cache growth without flushing each AWQ search step.
        gc_mode=GcMode.ON_STAGE_END,
        auto_forward_data_parallel=False,
        dense_vram_strategy_devices=[device],
        compute_device_filter=lambda devices: [item for item in devices if item == target],
        meta={"calibration": str(calibration), "calibration_rows": len(train)},
    )
    quantized = GPTQModel.load(
        str(model),
        # QuantizeConfig's metaclass accepts AWQConfig; its load annotation is narrower.
        quantize_config=cast(QuantizeConfig, quantization),
        dtype=torch.float16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    quantized.quantize(
        # torch.load/ones_like return Tensor in stubs; this token artifact is int64.
        calibration=[
            {
                "input_ids": cast(torch.LongTensor, row),
                "attention_mask": cast(torch.LongTensor, torch.ones_like(row)),
            }
            for row in train
        ],
        calibration_sort=None,
        batch_size=batch_size,
    )
    quantized.save(str(output_dir))


if __name__ == "__main__":
    auto_cli(main)
