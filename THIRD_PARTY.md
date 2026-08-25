# Third-party components

Third-party code remains in clean submodules pinned by the parent repository. Local adaptations will be distributed as patch series under `patches/<upstream>/`; the submodule gitlinks remain at the revisions below.

| Component | Upstream | Pinned revision | License | Purpose |
|---|---|---|---|---|
| QTIP | <https://github.com/Cornell-RelaxML/qtip> | `e90c6688c8dfae326a3a81b5eb032db7c6680ec0` | GPL-3.0-only | Bitshift-trellis quantization |
| QuIP# | <https://github.com/Cornell-RelaxML/quip-sharp> | `1d8f873e9a2a8b86b12bb1064c312c5689b77d98` | GPL-3.0-only | E8P codebooks and LDLQ |
| ParoQuant | <https://github.com/z-lab/paroquant> | `f74a96c306f7db3938e758d7dbc945d450f9f23f` | MIT | SPHQuant optimization and dense conversion |

The original license texts remain in each submodule. Model checkpoints and calibration datasets are not distributed and remain subject to their own terms.
