# Weight-VQ

从原始 Hugging Face 模型生成 dense QDQ 量化模型的统一流水线。

## 支持列表

| Model                              | QuIP# | QTIP | UniSVQ | SPHQuant |
| ---------------------------------- | ----- | ---- | ------ | -------- |
| Qwen3-4B-Instruct-2507             | ✅     | ✅    | ✅      | ✅        |
| Qwen3-30B-A3B-Instruct-2507        | ✅     | ✅    |         |          |
| Ministral-3-8B-Instruct-2512-BF16  | ✅     | ✅    | ✅      |          |
| Ministral-3-14B-Instruct-2512-BF16 | ✅     | ✅    | ✅      |          |

## 环境

仓库只使用根目录的 Python 3.11 uv 环境。环境提供互斥的 CUDA profile `cu124`（停止维护）和 `cu128`。
例如配置 CUDA 12.8 环境可使用：

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

只提交 `configs/**/*.example.yaml`。复制模板后填写实际路径；普通 `.yaml` 已被 Git 忽略。

```bash
cp configs/sphquant/qwen3.example.yaml configs/sphquant/qwen3.yaml
uv run wvq sphquant optimize --config configs/sphquant/qwen3.yaml
```

统一接口为 `wvq <method> <stage> --config <yaml>`。支持 `wvq --help`, `wvq <method> --help` 和 `wvq <method> <stage> --help` 获取帮助。

## 开发检查

```bash
uv run pytest
uv run ruff check .
uv run basedpyright
uv run ruff format .
```

- 更新依赖声明后先执行 `uv lock`，同步时使用对应 CUDA extra 和 `--locked`，避免隐式修改 lockfile。
- 不要写 shell 胶水，为了实验方便写的 shell 脚本一律留存于本地。
- 做新 idea 时开新分支。
