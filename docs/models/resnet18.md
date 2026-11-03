# ResNet-18 on lokinfer

ImageNet ResNet-18 (`torchvision.models.resnet18`, IMAGENET1K_V1 weights), forward only: normalized image
`f32[N, 3, H, W]` → logits `f32[N, 1000]`. Plan and op inventory: `impl-notes/research-models.md` §2.

Status (2026-11-01): **built and tested on CPU and GPU** (`models/resnet18.py`, `tests/models/test_resnet18.py`).

## Using it

```python
from models import resnet18 as M
w = M.fold_weights(M.torch_reference())   # BatchNorm folded into the convs (float64, then f32)
cpu = M.build(w)                           # CPU graph
gpu = M.build_gpu(w)                       # GPU graph, weights on the GPU
logits = cpu(x)                            # x f32[N, 3, H, W], N / H / W symbolic
```

## Graph

- NHWC internally. The input arrives NCHW (like PyTorch) and is converted once by `nchw_to_nhwc_f32_cpu` on the CPU.
  On the GPU the 7×7 stem reads NCHW directly: `im2col_nchw_f32_gpu` (K = 147, padded to 160) + one steel GEMM with
  bias + ReLU, whose `[N·Ho·Wo, 64]` output is already NHWC. The im2col is tiled (`lk_im2col_nchw_tile`: the input
  patch of 32 output pixels staged in threadgroup memory, a per-column offset table): N=8 1.22 → 0.37 ms on the Mac.
- Every BatchNorm is folded offline (`w·γ/√(σ²+ε)`, `β − μ·γ/√(σ²+ε)`), so each conv is one kernel with a fused
  epilogue: bias, optional residual (the BasicBlock tail), ReLU. 20 conv kernels, 1 maxpool, 1 global avgpool, 1 GEMM.
- CPU conv (`kernels/cpu/conv2d.cpp`) **[built]**, as `[pixels, KH·KW·Ci] × [K, Co]` GEMMs:
  - small convs (< 4 M MACs): one thread, ~3 MB im2col tiles;
  - big feature maps: a grid of ~3 tasks per pool thread (pixel tiles × Co chunks, the shape that reads the fewest
    bytes), each im2col'ing its tile in cache, one single-threaded Accelerate sgemm, epilogue (bias / residual /
    activation) while the tile is in cache. Before 2026-11-01 the tile was sized by the 3 MB budget only, so a
    112×112 stem or a 14×14 layer was a single tile on one core;
  - one-tile maps with long K (K ≥ 1024: the 14×14 and 7×7 stages, DCGAN D): one sgemm with Accelerate's own threading
    (faster than our grid there, iPhone + M2 Max); weights > 8 MiB (the 3×3×512×512 layers, DRAM-bound) as two K
    halves on two threads, partials summed (iPhone 430 → 345 µs per layer);
  - im2col copies a whole kernel row (KW·Ci contiguous floats in NHWC) at once (7×7 stem: M2 Max 500 → 340 µs).
- CPU maxpool: NEON, output rows on the pool (iPhone 228 → 47 µs); NCHW→NHWC of the 3-channel input with `vst3q`.
- CPU channel-planar build `build_cnhw()` **[built, tested on the Mac (Accelerate fallback) and the iPhone (SME2);
  the iPhone dashboard row "lokinfer planar": N=1 2.88 ms, N=8 18.96 ms vs NHWC 4.17 / 23.34, 2026-11-03]**:
  activations as `[C, N, H+2, W+2]` (zero border; the stem output has one extra zero row / column). Each direct conv is
  one SME2 GEMM `W^T[Co, K] · B[K, columns]` over the flattened padded planes: B row (tap, c) is plane c shifted by a
  constant (`rowOff`; 1x1 convs: plain row stride), read in place — no im2col, no per-call packing; weights packed
  offline (`ops.pack_conv_weights_cnhw`). Border columns are computed and zeroed by a column mask (cached per
  geometry). Stride 2 reads 2×2 phase planes (`nchw_to_phases`, `cnhw_to_phases`; small ones written on the calling P
  core: data written by the E cores is slower for the SME GEMM to read). Input planes whose stride is a multiple of 2 KB
  (14×14 at N = 8) are copied to padded planes first (they ran ~5× slow). The 3×3 stride-1 convs are
  `conv2d_wino_cnhw`: Winograd F(2×2, 3×3) — input transform (pool) → 16 SME2 GEMMs (one P-core stream) → output
  transform + bias / residual / ReLU (pool), pipelined over tile chunks; chosen at run time (≥ 512 tiles, or ≥ 64 with
  ≥ 4 images; else the direct GEMM: at N = 1 the 1.8× larger Winograd weights stream from DRAM). Tried and dropped:
  F(4×4, 3×3), an Accelerate grid instead of SME2, a gathered-B pipeline for the stem (kernels/cpu/conv2d.cpp notes).
- GPU conv: the vendored steel implicit-GEMM conv (`kernels/gpu/kokoro_lc.metal`, the general NHWC conv2d) with a
  bias / residual / ReLU epilogue.

## Correctness

vs torchvision on random normalized inputs, N ∈ {1, 2, 3}, H×W ∈ {224², 225×223, 160², 64×96, 33²}: max relative
error < 1e-4 and identical top-1, CPU and GPU.

## Performance

`bench/models/bench_vision.py` records lokinfer; `bench/models/bench_other_baselines.py` the baselines (same
inputs). See the dashboard (`scripts/perf_dashboard.sh`).
