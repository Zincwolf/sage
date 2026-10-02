"""Full-model benchmark settings; inputs are standalone local HF directories."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BenchmarkConfig:
    """Compare local dense and packed HF models under one measurement protocol.

    Args:
        models: Ordered mapping of report labels to HF model directories. Method and
            synthetic provenance are read from each checkpoint, not from its label.
        output_dir: Destination for paired raw JSON and summary CSV reports.
        name: Experiment label used in report filenames.
        phases: Measurement windows and their execution order. Decode-only uses
            teacher-forced inputs; end_to_end performs fixed-length greedy generation.
        cached_dense_weight: Retain QTIP/QuIP#/UniSVQ dense weights before compilation.
            Otherwise per-GEMM dequantization remains inside timed forwards.
        seed: Seed for shared synthetic input token IDs, not for model weights.
        device: CUDA device used for this run.
        batch_size: Number of sequences per window.
        prefill: Number of prompt tokens per sequence.
        decode: Number of generated tokens per sequence, including the first token
            from prefill. Decode-only measures decode minus one forward calls.
        warmup: Extra warmup windows after compilation, graph preparation and profiling.
        samples: Independent measured windows per model and phase.
    """

    models: dict[str, Path]
    output_dir: Path
    name: str = "benchmark"
    phases: list[Literal["prefill", "decode", "end_to_end"]] = field(
        default_factory=lambda: ["prefill", "decode", "end_to_end"]
    )
    cached_dense_weight: bool = False
    seed: int = 0
    device: str = "cuda:0"
    batch_size: int = 1
    prefill: int = 128
    decode: int = 32
    warmup: int = 5
    samples: int = 10

    def __post_init__(self) -> None:
        self.models = {label: path.expanduser() for label, path in self.models.items()}
        self.output_dir = self.output_dir.expanduser()
