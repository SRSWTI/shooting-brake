# Vendored llama.cpp/ggml snapshot

This directory contains the ggml-only subset used by Lucebox Hub.

- Source repository: https://github.com/Luce-Org/lucebox-ggml
- Source base branch: `luce-dflash`
- Source base commit: `6fbe72d67069136bbd370be703e1d4f441b5e942`
- Included merged PR: `#35` (`0fe65d9354b7c5da52a7741d2e37ba85f0d0c925`)
- Included test PR: `#37` (`0699be81480428f01b9b7ac49a09a2d51c77f8df`)
- Reconstruction: `luce-dflash@6fbe72d67069136bbd370be703e1d4f441b5e942` plus cherry-picked PRs `#35` and `#37`
- Vendored paths: `LICENSE`, `common/jinja`, `common/log.h`, `common/unicode.*`, `ggml`, `gguf-py`

Open ggml feature PRs are intentionally not included until they are merged, except for explicitly listed hub test PRs.

## Hub-local heterogeneous execution patches

PR `Luce-Org/lucebox#505` carries a temporary, reviewable patch set on top of
the snapshot above for AMD heterogeneous MoE execution:

- scheduler support for late cross-backend joins and deferred peer copies;
- ROCmFP quantized MMVQ/MMID kernels and grouped expert execution;
- fused MoE owner/combine operations and DeepSeek V4 HC/indexer kernels; and
- thread-local CUDA/HIP graph policy overrides used by fixed-topology graphs.

These changes are limited to `ggml/`. Keep their public declarations in
`ggml/include`, avoid DeepSeek-specific policy in generic kernels, and update
this provenance when the patch set is moved to `lucebox-ggml`.
