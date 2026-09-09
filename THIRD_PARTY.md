# Third-party components

Third-party code remains in clean submodules pinned by the parent repository. Local adaptations will be distributed as patch series under `patches/<upstream>/`; the submodule gitlinks remain at the revisions below.

| Component | Upstream | Pinned revision | License | Purpose |
|---|---|---|---|---|
| Fast Hadamard Transform | <https://github.com/Dao-AILab/fast-hadamard-transform> | `4ea722e434e3d4f2a14522341959ebdbe62be2de` | BSD-3-Clause | CUDA Hadamard transform used by QTIP incoherence processing |
| QTIP | <https://github.com/Cornell-RelaxML/qtip> | `e90c6688c8dfae326a3a81b5eb032db7c6680ec0` | GPL-3.0-only | Bitshift-trellis quantization |
| QuIP-Sharp | <https://github.com/Cornell-RelaxML/quip-sharp> | `1d8f873e9a2a8b86b12bb1064c312c5689b77d98` | GPL-3.0-only | E8P codebooks and LDLQ |
| UniSVQ | <https://github.com/AI9Stars/UniSVQ> | `449e8e4f317d2c539d7135c39f66817adce3cf4c` | Not specified upstream; includes QuIP-derived code | Author's released PTQ: identical codebook and 128-block Hadamard |
| ParoQuant | <https://github.com/z-lab/paroquant> | `f74a96c306f7db3938e758d7dbc945d450f9f23f` | MIT | SPHQuant optimization and dense conversion |

The original license texts, where provided, remain in each submodule. UniSVQ's pinned snapshot contains no LICENSE file; its redistribution terms and QuIP-derived code provenance need clarification before distributing that component. Model checkpoints and calibration datasets are not distributed and remain subject to their own terms.

## Submodule patch maintenance

父仓库中的 gitlink 始终固定在上表的干净上游 revision。适配提交不依赖额外 fork；其规范来源是父仓库中的 `patches/<upstream>/*.patch`，而不是某台机器上 submodule 内的本地分支。因此应用补丁后，父仓库显示 submodule modified 是预期状态，禁止暂存或提交该 gitlink。

以 ParoQuant 为例，从固定 base 恢复补丁：

```bash
git submodule update --init third_party/paroquant
(cd third_party/paroquant && git am ../../patches/paroquant/*.patch)
```

维护补丁时，直接在 submodule 的 detached HEAD 上从固定 base 叠加本地提交；不要建立仅用于保存补丁提交的本地分支。每项逻辑更改仍分别提交：

```bash
cd third_party/paroquant
git switch --detach f74a96c306f7db3938e758d7dbc945d450f9f23f
# edit, test, and create one or more commits directly on detached HEAD
git format-patch --no-numbered --zero-commit --no-signature \
  --output-directory ../../patches/paroquant \
  f74a96c306f7db3938e758d7dbc945d450f9f23f..HEAD
```

这些 submodule 内的提交只是生成 patch series 的临时来源，不由分支负责持久化；父仓库中的 `patches/<upstream>/` 才是规范产物。需要重做时，从固定 base 重新进入 detached HEAD 并应用或重建补丁。

导出任何 patch series 时一律使用 `--no-numbered --zero-commit --no-signature`。`--no-numbered` 固定 `Subject:` 为 `[PATCH]`，避免 series 中提交数量变化时将所有既有补丁改成新的 `[PATCH n/m]`；它不会移除输出文件名中的 `0001-` 等序号。`--zero-commit` 将首行 `From` 中的本地 commit ID 统一写成全零，避免补丁经 `git am` 重建后因 committer 信息和 parent commit 改变而产生无关差异。`--no-signature` 省略末尾的 Git 版本，避免开发机升级 Git 时改动所有补丁。

重建 patch series 前先删除该目录中的旧 `.patch`，避免已删除或重命名的提交残留。生成后必须在另一个从固定 base 创建的临时 detached worktree 中顺序执行 `git am patches/*.patch`，确认可以无冲突恢复；同时检查最终 tree，而非要求 commit ID 相同，因为 `git am` 会记录新的 committer 信息。父仓库只提交更新后的 patch 文件。

若应用失败，在 submodule 内执行 `git am --abort` 后修复 patch series；不要手工改到“看似应用完成”的半成品工作树。`git submodule update` 会把 submodule 恢复到父仓库固定的 base，之后需要重新 `git am`。
