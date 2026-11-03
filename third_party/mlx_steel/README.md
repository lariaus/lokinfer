# Vendored: MLX "steel" Metal kernels

Source: MLX v0.32.4 (https://github.com/ml-explore/mlx/archive/refs/tags/v0.32.4.tar.gz), files under
`mlx/backend/metal/kernels/` (`steel/gemm`, `steel/conv`, `steel/utils*`, `steel/defines.h`, and the shared
`utils.h`, `defines.h`, `logging.h`, `bf16*.h`, `complex.h`). MIT License, © 2023 Apple Inc. (see `LICENSE`).
Unmodified; the paths are kept so the original `#include`s resolve with `-I third_party/mlx_steel`.

Why: lokinfer's M2 conv proof (`docs/impl-notes/m2-conv-proof.md`) measured MLX's steel GEMM at 8.7 TFLOP/s through
lokinfer's runtime on the M2 Max, vs ≤ 1.1 for the hand-written kernel. lokinfer's GPU GEMM/conv kernels build on
steel's loaders and MMA, adding lokinfer's fused prologues/epilogues.
