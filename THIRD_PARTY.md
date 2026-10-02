# Third-party components

Submodules are pinned to the upstream revisions below. Adaptations are distributed
as patch series under `patches/<component>/`. The parent repository keeps its
submodule gitlinks at the upstream revisions.

| Component | Upstream | Pinned revision | License | Purpose |
|---|---|---|---|---|
| Fast Hadamard Transform | <https://github.com/Dao-AILab/fast-hadamard-transform> | `4ea722e434e3d4f2a14522341959ebdbe62be2de` | BSD-3-Clause | CUDA Hadamard transform |
| QTIP | <https://github.com/Cornell-RelaxML/qtip> | `e90c6688c8dfae326a3a81b5eb032db7c6680ec0` | GPL-3.0-only | Bitshift-trellis quantization |
| QuIP-Sharp | <https://github.com/Cornell-RelaxML/quip-sharp> | `1d8f873e9a2a8b86b12bb1064c312c5689b77d98` | GPL-3.0-only | E8P codebooks and LDLQ |
| UniSVQ | <https://github.com/AI9Stars/UniSVQ> | `449e8e4f317d2c539d7135c39f66817adce3cf4c` | Unspecified upstream | UniSVQ PTQ; includes QuIP-derived code |
| ParoQuant | <https://github.com/z-lab/paroquant> | `f74a96c306f7db3938e758d7dbc945d450f9f23f` | MIT | SPHQuant and SAGE optimization and dense conversion |

The frozen `csrc/awq/` runtime is llm-awq `quantization_new`
(<https://github.com/mit-han-lab/llm-awq>, `d6e797a`, MIT), migrated unchanged from
sage-kernels `a5764f15d1adb189f4f1b717dd81585104643908`, including its current-stream
launch adaptation and checked binding. GPTQModel supplies quantization and standard
HF `LLM_AWQ` packing; it is not used by this inference runtime.

Third-party components retain their upstream licenses and copyright notices.
UniSVQ is referenced as a submodule; its pinned snapshot contains no LICENSE file.
The project's GPL-3.0-only declaration applies to project-owned code. Model
checkpoints and datasets remain subject to their respective terms.

## Initial checkout

Run these commands from a fresh checkout, before syncing the uv environment:

```bash
git submodule update --init
for patch_dir in patches/*; do
    (cd "third_party/${patch_dir##*/}" && git am "../../$patch_dir"/*.patch) || exit
done
```

Git applies each series in filename order. `git am` requires a configured Git
committer identity. These commands are for clean upstream checkouts; applying a
series twice will fail.

Patched submodules appear as modified in the parent repository. Keep their
upstream gitlinks unchanged when staging parent-repository changes.
`git submodule update` checks out the pinned upstream revisions again, requiring
the patches to be reapplied.

## Patch maintenance

Work on a detached submodule HEAD with the existing series applied. Commit each
logical change within that submodule. The patch files in the parent repository
are the durable record of these adaptations.

For example, after committing changes in ParoQuant, regenerate its complete series:

```bash
rm patches/paroquant/*.patch
git -C third_party/paroquant format-patch \
    --no-numbered --zero-commit --no-signature \
    --output-directory ../../patches/paroquant \
    f74a96c306f7db3938e758d7dbc945d450f9f23f..HEAD
```

Always use these three format options. `--no-numbered` keeps subjects stable as
the series grows, `--zero-commit` removes local commit IDs from patch headers,
and `--no-signature` omits the local Git version. Removing the previous files
prevents obsolete patches from remaining in the series. Generate patches from
commits; edit the source code rather than patch files.

Before delivery, apply the entire series with `git am` in a separate temporary
detached worktree starting at the pinned upstream revision. Compare the resulting
Git tree with the source tree, then remove the temporary worktree. Commit IDs
can differ because replay creates new committer metadata.

If an application fails, use `git am --abort`, correct the source commits, and
regenerate the series. Commit the resulting patch files in the parent repository
while leaving the upstream gitlinks unchanged.
