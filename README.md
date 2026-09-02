# Weight-VQ

从原始 Hugging Face 模型生成 dense QDQ 量化模型的统一流水线。评测不属于本仓库范围。

## 环境

仓库只使用根目录的 Python 3.11 uv 环境。Torch、Transformers、datasets、QTIP、ParoQuant、Fast Hadamard Transform 和开发工具都安装到 `.venv/`；QTIP 与 ParoQuant 从 `third_party/` editable install。根 `pyproject.toml` 和 `uv.lock` 是唯一的依赖声明与锁文件。

环境提供互斥的 CUDA profile：

- `cu124`：Torch 2.6.0、torchvision 0.21.0 与 CUDA 12.4 wheels
- `cu128`：Torch 2.9.0、torchvision 0.24.0 与 CUDA 12.8 wheels

不要同时指定两个 profile，也不要在切换 profile 时复用旧 `.venv` 或扩展产物。例如 CUDA 12.8 环境使用：

```bash
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8 uv sync --extra cu128 --locked
```

Fast Hadamard Transform 从本仓库 submodule 构建，并由 uv 注入与运行时匹配的 Torch 构建依赖。构建产物进入 uv cache，后续同步可以复用。

## 第三方补丁

第三方仓库保持在 `.gitmodules` 和 [THIRD_PARTY.md](THIRD_PARTY.md) 记录的上游 revision；本地适配以 `patches/<upstream>/` 下的完整 `git format-patch` 系列维护，不更新父仓库中的 submodule gitlink。首次检出后，按文件名顺序应用所需补丁。例如 SPHQuant：

```bash
git submodule update --init third_party/paroquant
(cd third_party/paroquant && git am ../../patches/paroquant/*.patch)
```

应用后，父仓库中 `third_party/paroquant` 显示 modified 是正常现象，不要执行 `git add third_party/paroquant`。补丁的维护、重建和校验规范见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 配置与运行

只提交 `configs/**/*.example.yaml`。复制模板后填写模型、calibration 和输出路径；普通 `.yaml` 已被 Git 忽略：

```bash
cp configs/sphquant/qwen3.example.yaml configs/sphquant/qwen3.yaml
uv run --no-sync wvq sphquant optimize --config configs/sphquant/qwen3.yaml
```

统一接口为 `wvq <method> <stage> --config <yaml>`。当前命令布局：

- `wvq data calibration`
- `wvq data vlm-calibration`
- `wvq qtip {hessian,quantize,export,validate}`
- `wvq quip-sharp {hessian,quantize,export,validate}`
- `wvq sphquant {optimize,export,validate}`

Qwen3-VL 使用 `data vlm-calibration` 生成的 COCO manifest；Qwen3 使用 tokenized calibration artifact。QuIP-Sharp 命令目前是保留的占位实现。

## 开发检查

```bash
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync basedpyright
```

更新依赖声明后先执行 `uv lock`，同步时使用对应 CUDA extra 和 `--locked`，避免隐式修改 lockfile。
