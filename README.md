# Weight-VQ

从原始 Hugging Face 模型生成 dense QDQ 量化模型的统一流水线。评测不属于本仓库范围。

## 环境布局

仓库使用 uv 管理五个相互隔离的 Python 3.11 环境：

| 环境 | 位置 | 用途 |
|---|---|---|
| 根环境 | `.venv/` | `wvq` dispatcher、pytest、ruff、basedpyright；不依赖 Torch |
| data | `projects/data/.venv/` | tokenizer、datasets、Torch 和 calibration 生成 |
| qtip | `projects/qtip/.venv/` | QTIP 与 CUDA 扩展 |
| quip-sharp | `projects/quip-sharp/.venv/` | QuIP-Sharp 与 CUDA 扩展 |
| sphquant | `projects/sphquant/.venv/` | SPHQuant/ParoQuant |

每个环境都有独立的 `pyproject.toml` 和 `uv.lock`，不过内容大致相同，可以复用本地的 uv 缓存。data 与三个方法环境均提供互斥的 `cu124`、`cu128` extra：

- `cu124`：Torch 2.6.0 + CUDA 12.4 wheels
- `cu128`：Torch 2.9.0 + CUDA 12.8 wheels

`wvq` 只执行目标环境中的 worker，并固定使用 `uv run --no-sync`；它不会创建、更新或修复环境。

## 准备环境

要求 uv 0.12.x 和 Python 3.11。先安装根环境，再按机器选择 data 的 CUDA profile，比如在 CUDA 12.4 环境下：

```sh
uv sync --locked
uv sync --project projects/data --extra cu124 --locked
```

按需安装方法环境。CUDA 12.4 + QTIP 示例：

```sh
CUDA_HOME=/usr/local/cuda-12.4 MAX_JOBS=8 uv sync --project projects/qtip --extra cu124 --locked
```

CUDA 12.8 主机将 `cu124` 改为 `cu128`，并令 `CUDA_HOME` 指向该主机的 CUDA 12.8 toolkit。QTIP/QuIP-Sharp 可能编译 CUDA 扩展，安装时间和磁盘占用会明显增加。

切换 CUDA profile 时不要复用旧环境或扩展产物。例如将 QTIP 从 cu124 切换到 cu128：

```sh
rm -rf projects/qtip/.venv
CUDA_HOME=/path/to/cuda-12.8 MAX_JOBS=8 uv sync --project projects/qtip --extra cu128 --locked
```

其他方法及 data 同理。不要同时指定 `--extra cu124` 和 `--extra cu128`。同一 profile 在各环境中使用相同的 Torch 版本和 wheel index，使 uv 能通过全局 cache 复用已下载的 wheel；CPU 与 CUDA wheel 是不同 artifact，不能互相复用。

## 配置与运行

只提交 `configs/**/*.example.yaml`。复制模板后填写模型、calibration 和输出路径；普通 `.yaml` 已被 Git 忽略：

```sh
cp configs/sphquant/qwen3.example.yaml configs/sphquant/qwen3.yaml
uv run wvq sphquant optimize --config configs/sphquant/qwen3.yaml
```

统一接口为 `wvq <method> <stage> --config <yaml>`。当前命令布局：

- `wvq data calibration`
- `wvq qtip {hessian,quantize,export,validate}`
- `wvq quip-sharp {hessian,quantize,export,validate}`
- `wvq sphquant {optimize,export,validate}`

目前 package、环境和命令框架已经就绪，算法 stage 尚在迁移；占位 stage 会明确抛出 `NotImplementedError`。

## 开发检查

根环境同步完成后运行：

```bash
uv run pytest
uv run ruff check .
uv run basedpyright
```

更新依赖声明后，在对应项目重新执行 `uv lock --project projects/<name>`；同步时继续使用 `--locked`，避免隐式修改 lockfile。
