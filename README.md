# WVQ

For paper *SAGE: PUSHING 2-BIT VECTOR-QUANTIZED LLMS TOWARD REAL-WORLD USABILITY*.

Weight vector quantization for Hugging Face models with QTIP, QuIP#, UniSVQ,
SPHQuant, and SAGE. Outputs are standard HF models with quantized-and-dequantized
floating-point weights (dense QDQ). Evaluation is handled separately.

## Setup

Requires Linux, an NVIDIA GPU, the CUDA 12.8 toolkit, Python 3.11, and
[uv](https://docs.astral.sh/uv/).

Initialize the submodules and apply their patches following
[THIRD_PARTY.md](THIRD_PARTY.md#initial-checkout), then create the environment:

`CUDA_HOME=/usr/local/cuda-12.8 uv sync --locked`

The initial sync downloads dependencies and builds CUDA extensions.

## Usage

[Example configurations](configs/) cover the supported methods and models.
Prepare a local HF model and a calibration artifact; the
[calibration template](configs/data/calibration.example.yaml) and
`uv run wvq data calibration --help` describe calibration setup for
SPHQuant and SAGE. Our OPD datasets are provided as a [training set](https://huggingface.co/datasets/chizhongw/opd-data/blob/main/text/train_clean_balanced.parquet) and an optional [validation set](https://huggingface.co/datasets/chizhongw/opd-data/blob/main/text/heldout_clean.parquet).

Copy a method template and set its model, calibration, and output paths:

```bash
cp configs/sage/qwen3.example.yaml configs/sage/qwen3.yaml
# Edit configs/sage/qwen3.yaml before running.
uv run wvq sage run --config configs/sage/qwen3.yaml
```

This produces `exported_model/` under the configured `output_dir`.
Local configuration files are Git-ignored.

We evaluate our models on [MMLU-Redux](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/mmlu_redux.html), [GSM8K](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/gsm8k.html), [MATH-500](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/math_500.html), [IFEval](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/ifeval.html), [HumanEval+](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/humaneval_plus.html) and [BFCL-v3](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/bfcl_v3.html) with [EvalScope](https://github.com/modelscope/evalscope) 1.10.0. Serving and inference details are provided in our preprint.

## Help

```bash
uv run wvq --help
uv run wvq sage --help
uv run wvq sage run --help
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run basedpyright
```

## License

Project code is licensed under [GPL-3.0-only](LICENSE).
Third-party licenses and the missing upstream UniSVQ license are recorded in
[THIRD_PARTY.md](THIRD_PARTY.md).
