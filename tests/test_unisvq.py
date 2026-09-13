"""Small released-UniSVQ numerical and pipeline tests; no model downloads."""

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from datasets import Dataset
from scipy.linalg import hadamard
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    Ministral3Config,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PixtralImageProcessor,
    PixtralProcessor,
    PixtralVisionConfig,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from weight_vq import cli
from weight_vq.blockwise.artifacts import projections
from weight_vq.blockwise.calibration import sample_documents
from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.hessian import LayerHessianCollectors
from weight_vq.blockwise.models import language_backbone, split_work_experts
from weight_vq.blockwise.pipeline import export, hessian, quantize
from weight_vq.blockwise.replay import capture_replay
from weight_vq.unisvq.config import UniSVQConfig
from weight_vq.unisvq.quantization import quantize_layer, reconstruct_dense_weight
from weight_vq.unisvq.reconstruction import UniSVQReconstruction


@pytest.fixture
def tiny_unisvq(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    config = UniSVQConfig(
        model=tmp_path / "model",
        dataset=tmp_path / "rp",
        output_dir=tmp_path / "run",
        devices=[0],
        model_dtype="bfloat16",
        num_cpu_threads=2,
        hessian_samples=4,
        hessian_sequence_length=8,
        hessian_chunk_size=2,
        train_samples=4,
        validation_samples=2,
        sequence_length=8,
        quip_tune_iters=1,
        reconstruction_epochs=1,
        batch_size=2,
    )
    hf = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
    )
    hf._attn_implementation = "sdpa"
    return config, hf


def test_released_defaults_and_hessian_precision(tiny_unisvq):
    config, _ = tiny_unisvq
    defaults = UniSVQConfig(config.model, config.dataset, config.output_dir, [0])
    assert defaults.hessian_samples == defaults.train_samples + defaults.validation_samples == 1024
    assert defaults.sequence_length == defaults.hessian_sequence_length == 4096
    assert defaults.reconstruction_epochs == 1
    assert (defaults.batch_size, defaults.gradient_accumulation_steps) == (4, 2)
    assert defaults.codebook == "identical"
    assert defaults.scale_override == 0.83
    assert defaults.hessian_recipe(1)["statistics_dtype"] == "float32"
    assert "use_fp64" not in {field.name for field in fields(defaults)}
    assert defaults.hessian_dtype == "float32"
    fp64 = replace(defaults, hessian_dtype="float64")
    assert fp64.hessian_recipe(1)["statistics_dtype"] == "float64"
    assert fp64.hessian_recipe(1) != defaults.hessian_recipe(1)
    assert fp64.calibration_recipe() == defaults.calibration_recipe()
    assert fp64.recipe(model_family="qwen3", layer_end=1)["hessian_dtype"] == "float64"
    dense_recipe = defaults.recipe(model_family="qwen3", layer_end=1)
    assert "min_expert_coverage" not in dense_recipe and "moe_protocol" not in dense_recipe
    moe_recipe = defaults.recipe(model_family="qwen3_moe", layer_end=1)
    assert moe_recipe["min_expert_coverage"] == 0
    moe_protocol = cast(dict[str, object], moe_recipe["moe_protocol"])
    assert moe_protocol["reconstruction"] == "whole-block-after-all-projections"
    assert moe_protocol["router"] == "unquantized-trained-and-saved"


@pytest.mark.parametrize(
    "stage", ["calibration", "hessian", "quantize", "export", "validate", "run"]
)
@pytest.mark.parametrize("hessian_dtype", ["float32", "float64"])
def test_cli(tiny_unisvq, monkeypatch, stage, hessian_dtype):
    config, _ = tiny_unisvq
    path = config.output_dir.parent / "unisvq.yaml"
    path.write_text(
        "model: /model\ndataset: /rp\noutput_dir: /run\ndevices: [0]\n"
        f"hessian_dtype: {hessian_dtype}\n"
    )
    received = []
    assert cli.STAGES[("unisvq", stage)][1] is UniSVQConfig
    monkeypatch.setitem(cli.STAGES, ("unisvq", stage), (received.append, UniSVQConfig))
    cli.main(["unisvq", stage, "--config", str(path)])
    assert received == [
        UniSVQConfig(Path("/model"), Path("/rp"), Path("/run"), [0], hessian_dtype=hessian_dtype)
    ]


def test_truncated_concatenation_matches_upstream_sampling():
    from tempfile import TemporaryDirectory

    from unisvq.utils.data_utils import sample_jsonl_concat

    # Real HF tokenizer, not a fake loader: long documents are truncated, short
    # documents are retained and concatenated, unlike QTIP's reconstruction samples.
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    from tokenizers.pre_tokenizers import Whitespace

    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    dataset = Dataset.from_dict({"text": ["a b", "b a a b a a b b a a"]})
    with TemporaryDirectory() as directory:
        path = Path(directory) / "data.jsonl"
        dataset.to_json(path)
        torch.manual_seed(4)
        expected = sample_jsonl_concat(str(path), tokenizer, size=5, ctx_size=4)
    actual = sample_documents(
        dataset,
        tokenizer,
        samples=5,
        length=4,
        concatenate=True,
        truncate_documents=True,
        generator=torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(actual, expected)


def test_package_and_fixed_grid():
    import lib.algo.finetune
    import quip_sharp.algo.quip
    from unisvq import codebook
    from unisvq.algo import quip

    assert quip is not quip_sharp.algo.quip
    assert lib.algo.finetune.__name__ == "lib.algo.finetune"
    cb = codebook.get_codebook("identical", codebook_bit=2)
    assert cb.grid.shape == (256, 4)
    assert not list(cb.parameters())
    torch.testing.assert_close(
        cb.linear_proj_weight, torch.eye(4, dtype=torch.bfloat16) * cb.linear_proj_weight[0, 0]
    )
    torch.testing.assert_close(cb.by_idxs(torch.arange(256).reshape(1, -1)), cb.grid.reshape(1, -1))


@pytest.mark.parametrize("width", [128, 256, 768, 2048, 2560, 5120, 9728])
def test_upstream_128_block_hadamard(width):
    from unisvq.utils.matmul_had import matmul_hadU

    torch.manual_seed(8)
    weight = torch.randn(2, width)
    reference = torch.from_numpy(hadamard(128)).float() / 128**0.5
    expected = (weight.reshape(-1, 128) @ reference).reshape_as(weight)
    actual = matmul_hadU(weight)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(matmul_hadU(actual), weight, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tune_iters", [0, 1])
def test_real_ldlq_and_saved_dense_decode(tune_iters):
    from unisvq import codebook
    from unisvq.algo import quip
    from unisvq.utils.matmul_had import matmul_hadU

    torch.set_num_threads(2)
    torch.manual_seed(17)
    weight = torch.randn(128, 128)
    source = torch.randn(256, 128)
    hessian = source.T @ source / 256 + torch.eye(128) * 0.1
    args = SimpleNamespace(
        use_fp64=False,
        lora_rank=0,
        incoh_mode="had",
        rescale_WH=False,
        sigma_reg=0.01,
        scale_override=0.83,
        resid_scale_override=-1,
        no_use_buffered=False,
        lowmem_ldlq=False,
        quip_tune_iters=tune_iters,
    )
    cb = codebook.get_codebook("identical", codebook_bit=2)
    decoded, saved = quip.quantize(hessian, weight, 0, cb, args, device="cpu")
    assert saved["Qidxs"].shape == (128, 32)
    assert saved["Qidxs"].dtype == torch.uint8
    actual = reconstruct_dense_weight(saved, torch.device("cpu"))
    # quantize() returns before FP16 checkpoint rounding; exported weights use it.
    torch.testing.assert_close(actual, decoded, atol=0.002, rtol=0.002)
    raw = cb.by_idxs(saved["Qidxs"]).float()
    reference = (
        matmul_hadU(matmul_hadU(raw).T).T * saved["SU"].float() * saved["SV"].float()[:, None]
    )
    torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Released Linear requires CUDA")
@pytest.mark.parametrize("shape", [(768, 2048), (2048, 768), (32, 2560)])
def test_real_projection_ldlq_and_backward(shape):
    from unisvq import codebook
    from unisvq import utils as upstream_utils
    from unisvq.algo import quip
    from unisvq.linear import QuantizedLinear

    torch.set_num_threads(2)
    torch.manual_seed(19)
    out_features, in_features = shape
    weight = torch.randn(shape) * 0.02
    source = torch.randn(32, in_features)
    metric = source.T @ source / len(source) + torch.eye(in_features) * 0.1
    args = SimpleNamespace(
        use_fp64=False,
        lora_rank=0,
        incoh_mode="had",
        rescale_WH=False,
        sigma_reg=0.01,
        scale_override=0.83,
        resid_scale_override=-1,
        no_use_buffered=False,
        lowmem_ldlq=False,
        quip_tune_iters=10,
    )
    cb = codebook.get_codebook("identical", codebook_bit=2)
    scale = weight.square().mean().sqrt()
    decoded, saved = quip.quantize(metric, weight / scale, 0, cb, args, device="cuda:0")
    saved["SV"] *= scale
    saved.update(fused=False, shapes=[shape], scales=[1.0], bias=None)
    actual = reconstruct_dense_weight(saved, torch.device("cuda:0"))
    torch.testing.assert_close(actual, decoded.cpu() * scale, atol=1e-4, rtol=0.003)
    linear = QuantizedLinear(
        in_features,
        out_features,
        cb.codesz,
        cb.packsz,
        cb.pack_out,
        cb.version,
        rank=0,
        train_mode=False,
        grad_ckpt=True,
        codebook_id=codebook.get_id("identical"),
        codebook_bit=2,
        idx_dtype=str(cb.idx_dtype),
    )
    upstream_utils.unpack_quip(linear, saved, codebook.get_id("identical"), cb.codesz)
    linear.SU = torch.nn.Parameter(linear.get_buffer("SU").float())
    linear.SV = torch.nn.Parameter(linear.get_buffer("SV").float())
    linear.cuda()
    inputs = (torch.randn(2, 4, in_features, device="cuda") * 0.1).requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        output = linear(inputs)
        expected = torch.nn.functional.linear(inputs, actual.cuda())
    torch.testing.assert_close(output, expected.float(), atol=0.002, rtol=0.03)
    output.square().mean().backward()
    for parameter in (inputs, linear.SU, linear.SV):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Released Linear requires CUDA")
@pytest.mark.parametrize("family", ["dense", "moe"])
def test_whole_block_reconstruction(tiny_unisvq, family):
    config, hf = tiny_unisvq
    if family == "moe":
        hf = Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=1,
        )
        hf.update({"head_dim": 64})
        hf._attn_implementation = "sdpa"
        model = Qwen3MoeForCausalLM(hf).bfloat16().eval().requires_grad_(False)
        with torch.no_grad():
            router = model.get_parameter("model.layers.0.mlp.gate.weight")
            router[1].copy_(-router[0])
            router[2].copy_(router[0] * 0.5)
            router[3].copy_(router[0] * -0.5)
    else:
        model = Qwen3ForCausalLM(hf).bfloat16().eval().requires_grad_(False)
    specs = projections(4 if family == "moe" else 0)
    tokens = torch.randint(0, 128, (6, 8))
    train = capture_replay(model, tokens[:4], device="cuda:0", microbatch_size=1)
    valid = capture_replay(model, tokens[4:], device="cuda:0", microbatch_size=1)
    layer = model.model.layers[0].cuda()
    split_work_experts(layer)
    collectors = LayerHessianCollectors(layer, include_padding=True, dtype=torch.float32)
    train_replay = train.forward_original(layer, device="cuda:0", collectors=collectors)
    valid_replay = valid.forward_original(layer, device="cuda:0")
    config.hessians.mkdir(parents=True)
    config.checkpoints.mkdir()
    for name, collector in collectors.collectors.items():
        # Zero-coverage experts cannot be normalized by the collector's dense close().
        matrix, first, count = collector.close_statistics()
        if count:
            matrix /= count
            first /= count
            matrix -= first[:, None] * first[None, :]
        torch.save(
            {
                "flatH": torch.cat([row[: i + 1] for i, row in enumerate(matrix)]),
                "mu": first,
                "n": matrix.shape[0],
                "ct": count,
                **({"total_tokens": tokens[:4].numel()} if name.startswith("expert") else {}),
            },
            config.hessians / f"0_{name}.pt",
        )
    calls = []

    class Observed(UniSVQReconstruction):
        def __call__(self, block, name, orig_dtype):
            from unisvq.linear import QuantizedLinear

            assert all(
                isinstance(block.get_submodule(attribute), QuantizedLinear)
                for attribute, *_ in specs
            )
            super().__call__(block, name, orig_dtype)
            assert all(
                block.get_submodule(attribute).built_codebook_class for attribute, *_ in specs
            )
            calls.append(name)

    trainer = Observed(
        train_replay, valid_replay, config, DistributedContext(0, 1, torch.device("cuda:0")), 0
    )
    layer.requires_grad_(True)
    quantize_layer(layer, 0, config, trainer)
    assert calls == ["0"]
    with torch.no_grad():
        layer.cuda()
        for attribute, short, *_ in specs:
            saved = torch.load(config.checkpoints / f"0_{short}.pt", weights_only=True)
            assert saved["SU"].dtype == saved["SV"].dtype == torch.float16
            linear = layer.get_submodule(attribute)
            # Compare the exported checkpoint, including the released FP16 rounding.
            linear.get_parameter("SU").copy_(saved["SU"].cuda())
            linear.get_parameter("SV").copy_(saved["SV"].cuda())
            weight = reconstruct_dense_weight(saved, torch.device("cuda:0"))
            hidden = torch.randn(1, 8, weight.shape[1], device="cuda:0") * 0.1
            torch.testing.assert_close(
                linear(hidden),
                torch.nn.functional.linear(hidden, weight.cuda()),
                atol=0.005,
                rtol=0.03,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pipeline requires CUDA")
@pytest.mark.parametrize(
    ("family", "hessian_dtype"),
    [
        ("qwen3", "float32"),
        ("qwen3", "float64"),
        ("ministral-square", "float32"),
        ("ministral-rectangular", "float32"),
    ],
)
def test_single_gpu_hessian_restart_export(tiny_unisvq, family, hessian_dtype):
    config, hf = tiny_unisvq
    config = replace(config, hessian_dtype=hessian_dtype)
    if family == "qwen3":
        model = Qwen3ForCausalLM(hf).bfloat16()
        loader = AutoModelForCausalLM
    else:
        text = Ministral3Config(
            vocab_size=128,
            hidden_size=128 if family == "ministral-square" else 256,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=512,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 4,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1,
            },
        )
        vision = PixtralVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=16,
            patch_size=4,
        )
        hf = Mistral3Config(text_config=text, vision_config=vision, tie_word_embeddings=False)
        hf._attn_implementation = "sdpa"
        model = Mistral3ForConditionalGeneration(hf).bfloat16()
        loader = AutoModelForImageTextToText
    model.save_pretrained(config.model)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "test": 1, "[IMG]": 10}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=["[IMG]"],
        chat_template="{{ messages[0]['content'] }}",
    )
    if family == "qwen3":
        tokenizer.save_pretrained(config.model)
    else:
        PixtralProcessor(
            image_processor=PixtralImageProcessor(patch_size=4, size={"longest_edge": 16}),
            tokenizer=tokenizer,
        ).save_pretrained(config.model)
        (config.model / "tekken.json").write_text("tekken fixture")
    payload = {
        "protocol": config.calibration_recipe(),
        "hessian": torch.randint(0, 128, (4, 8)),
        "train": torch.randint(0, 128, (4, 8)),
        "validation": torch.randint(0, 128, (2, 8)),
    }
    config.output_dir.mkdir()
    torch.save(payload, config.calibration)
    hessian(config)
    statistics = torch.load(config.hessians / "0_qkv.pt", weights_only=True)
    assert statistics["flatH"].dtype == statistics["mu"].dtype == getattr(torch, hessian_dtype)
    other_dtype = "float64" if hessian_dtype == "float32" else "float32"
    with pytest.raises(RuntimeError, match="Hessian protocol differs"):
        hessian(replace(config, hessian_dtype=other_dtype))
    stamp = (config.hessians / "complete.json").stat().st_mtime_ns
    hessian(config)
    assert (config.hessians / "complete.json").stat().st_mtime_ns == stamp
    quantize(config)
    saved = torch.load(config.checkpoints / "0_v.pt", weights_only=True)
    stamp = (config.checkpoints / "0.complete").stat().st_mtime_ns
    quantize(config)
    assert (config.checkpoints / "0.complete").stat().st_mtime_ns == stamp
    (config.checkpoints / "0.complete").unlink()
    torch.save({"invalid": True}, config.checkpoints / "0_v.pt")
    quantize(config)
    restored = torch.load(config.checkpoints / "0_v.pt", weights_only=True)
    for key in ("Qidxs", "SU", "SV"):
        torch.testing.assert_close(restored[key], saved[key], rtol=0, atol=0)
    export(config)
    dense = loader.from_pretrained(config.exported_model, local_files_only=True)
    if family != "qwen3":
        processor = AutoProcessor.from_pretrained(config.exported_model, local_files_only=True)
        assert processor.chat_template == tokenizer.chat_template
        assert (config.exported_model / "tekken.json").read_text() == "tekken fixture"
        for name, value in model.named_parameters():
            if not name.startswith("model.language_model.layers."):
                torch.testing.assert_close(dense.get_parameter(name), value, rtol=0, atol=0)
    auxiliary = torch.load(config.checkpoints / "0.auxiliary.pt", weights_only=True)
    assert set(auxiliary) == {"input_layernorm.weight", "post_attention_layernorm.weight"}
    if family == "qwen3":
        for name in ("q_norm", "k_norm"):
            torch.testing.assert_close(
                language_backbone(dense).layers[0].get_parameter(f"self_attn.{name}.weight"),
                language_backbone(model).layers[0].get_parameter(f"self_attn.{name}.weight"),
                rtol=0,
                atol=0,
            )
    assert torch.isfinite(dense(payload["validation"]).logits).all()
    assert cast(Any, dense).generate(payload["validation"][:1], max_new_tokens=2).shape == (1, 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Hadamard extension requires CUDA")
def test_upstream_hadamard_backward_matches_explicit_matrix():
    from unisvq.utils.matmul_had import get_hadK, matmul_hadU_cuda

    torch.manual_seed(12)
    inputs = torch.randn(2, 256, device="cuda", requires_grad=True)
    direction = torch.randn_like(inputs)
    had_k, k = get_hadK(256)
    actual = matmul_hadU_cuda(inputs, had_k, k)
    gradient = torch.autograd.grad((actual * direction).sum(), inputs)[0]
    matrix = torch.from_numpy(hadamard(128)).to(device="cuda", dtype=torch.float32) / 128**0.5
    expected = (inputs.reshape(-1, 128) @ matrix).reshape_as(inputs)
    expected_gradient = (direction.reshape(-1, 128) @ matrix.T).reshape_as(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-6, rtol=1e-5)


def test_real_cpu_reconstruction_early_stop_and_best_state(tiny_unisvq):
    config, hf = tiny_unisvq
    config = replace(
        config, reconstruction_epochs=5, early_stop=2, learning_rate=0, susv_learning_rate=0
    )
    model = Qwen3ForCausalLM(hf).eval()
    tokens = torch.randint(0, 128, (6, 8))
    train = capture_replay(model, tokens[:4], device="cpu", microbatch_size=1, placement="cpu")
    valid = capture_replay(model, tokens[4:], device="cpu", microbatch_size=1, placement="cpu")
    layer = model.model.layers[0]
    train_replay = train.forward_original(layer, device="cpu")
    valid_replay = valid.forward_original(layer, device="cpu")
    before = {key: value.clone() for key, value in layer.state_dict().items()}
    losses = []

    class Observed(UniSVQReconstruction):
        def validation_loss(self, layer, dtype):
            loss = super().validation_loss(layer, dtype)
            losses.append(loss)
            return loss

    trainer = Observed(
        train_replay, valid_replay, config, DistributedContext(0, 1, torch.device("cpu")), 0
    )
    trainer(layer, "cpu", torch.float32)
    assert len(losses) == 3  # initial validation plus two non-improving epochs
    assert losses == [0.0, 0.0, 0.0]
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
