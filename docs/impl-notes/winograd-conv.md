# Winograd convolutions on the GPU (2026-11-01)

Status: **built** — Kokoro's generator resblock convs (1-D, K = 3 / 7 / 11, dilation 1 / 3 / 5) run as Winograd
convs on the GPU (`models/kokoro/model.py` `_snake_conv_gpu`, default on; `LOKINFER_KOKORO_WINO=0` selects the old
snake + steel conv pair for A/B). The 2-D F(2x2, 3x3) kernel for ResNet-18 is **built and tested but not
integrated** (it only wins on one layer shape, see the end).

Files: `kernels/gpu/winograd.metal`, `kernels/gpu/winograd.cpp`, `python/lokinfer/ops/winograd.py`,
`tests/kernels/test_winograd_gpu.py`.

## Why

The steel implicit-GEMM conv (`conv1d_lc_f32_gpu`) was ~62 % of Kokoro's Mac GPU time and ~74 % of the iPhone's and
already ran at the steel f32 matrix ceiling (iPhone 2.6 TF/s, Mac ~5-6 TF/s): fewer multiplications is the only lever.

## Math

Toom-Cook / Winograd correlation F(m, r): y = AT [(G w) ⊙ (BT d)], A = m + r - 1 points. Built exactly over rationals
(`winograd.transforms`, Vandermonde + inverse), BT rows normalized to max |.| = 1 (G absorbs the scale).

- **K = 3**: F(4, 3), points {0, ±1, ±2, ∞}: 6 multiplications per 4 outputs instead of 12 (0.5x).
- **K = 7, 11 — nested F(4, 4)**: split the taps into groups of 4 at offsets 0, 4, 8 (K padded to 8 / 12 with zero
  taps). Group g of tile t is an F(4, 4) on the transformed tile t + g, so all groups share one transformed input:
  `Y[t] = AT Σ_ξ e_ξ Σ_{g, ci} V_ξ[t + g, ci] U[ξ, g ci, co]`. 7 points {0, ±1, ±2, -1/2, ∞}: 14 / 21 multiplications
  per 4 outputs instead of 28 / 44 (0.50x / 0.48x). This is what keeps K = 11 well conditioned: a plain F(m, 11) needs
  ≥ 12 points.
- **Dilation d**: the conv decimates — residue class ρ of the output only reads x[ρ + d q] — so each class is an
  undilated conv (padding % d == 0, true for every Kokoro conv).

### Accuracy (float64 reference, real Kokoro activations)

Inputs/weights captured from the float64 golden forward of `real117` (all 48 generator resblock / noise-res convs,
first 4096 rows), numpy f32 simulation of each candidate:

| scheme | max-abs / max \|y\| | relative RMS |
|---|---|---|
| direct f32 (im2col sgemm) | 0.4 - 3.0e-6 | 2 - 9e-7 |
| F(4,3), K = 3 | 0.5 - 2.7e-6 | 2.3 - 8.4e-7 |
| nested F(4,4) {0,±1,±2,-1/2,∞}, K = 7 / 11 | 0.8 - 4.3e-6 | 2 - 11e-7 |
| F(6,3) (8 points) | 0.6 - 4e-6 | 3 - 12e-7 |
| F(4,7) / nested F(5,6) / F(6,6) / F(6,7) (10-12 points) | 4e-6 - 4.4e-5 | 1 - 9.5e-6 (~10x direct) |

The 6-7-point schemes are within 1-2x of the direct conv's own f32 error; ≥ 10 points costs ~10x, so they were not
built. Measured on the GPU kernels (same 48 convs, full length): steel max-abs/max|y| median 1.2e-6 (max 2.8e-6),
rel-RMS median 5.7e-7 (max 9.4e-7); Winograd median 1.5e-6 (max 3.1e-6), rel-RMS median 6.2e-7 (max 1.0e-6).
End to end (`tests/models/test_kokoro.py::test_gpu_free_running_snr`, audio SNR vs the f64 golden): **identical to
the steel path to 0.1 dB on all 14 cases** (e.g. real117 32.9 dB, rand510 26.9 dB, app59f 35.3 dB; torch f32 33.5 /
26.8 / 35.3) — the audio error is dominated by the f32 harmonic-source phase, not the convs.

## Kernels (what was tried, in order)

1. **Fully fused** (transform in the GEMM's A loader, ξ-outer loop, output transform folded into M = 4 output-row
   accumulators in registers): 2.3-3 TF/s-equivalent on the Mac, half of steel. GPU counters (`prof-lokinfer trace -t
   gpu-counters`) showed 42 % "buffer store" utilization: register spills from arrays of `simdgroup_float8x8`.
   Rewriting the accumulators as plain `float2` thread elements (MLX steel's BaseMMAFrag style) with every loop
   unrolled fixed the spills: ~4.2 TF-eq, still slower than steel. With the transform replaced by a single load
   (a timing-only probe) the same loop ran 6.4-6.8 TF-eq: the in-loop transform (A loads per element, redone for
   every point ξ) was the bound.
2. **Reduction-block outer, all A points' accumulators live** (transform once per channel block): 8 % occupancy,
   ~4.6-5.1 TF-eq: register-bound.
3. **Two kernels (shipped)**: `lk_wino_tf_lc` writes V [d, A, rmax, Ci] (A/4 = 1.5-1.75x the input), optionally
   applying the AdaIN-affine + Snake activation that precedes every generator conv (so the separate snake kernel and
   its output disappear); `lk_wino_mm_lc` = loop 1's GEMM with a plain float4 A loader. Tiles: 32 tiles (128 output
   rows) x 64 channels, BK 16, 2x2 simdgroups; 16-tile threadgroups when the 32-tile grid has < 128 threadgroups
   (short sequences: 5-15 % on L = 1100). A/B'd and rejected: 64x32, BK 32, 32x32, 16x64 tiles for all grids, B read
   straight from device memory, 4 tiles per transform thread (2x slower transform: occupancy).

## Results

Microbench (scratch `bench_wino.py`: chains of 6 convs, `bench_kokoro.timeit` / `timed_run` protocol), the pair
snake + steel conv vs the fused-snake Winograd conv, per conv:

| shape (L, C) | M2 Max K3 / K7 / K11 (d 1) | iPhone K3 / K7 / K11 (d 1) |
|---|---|---|
| 44161, 128 (real117 stage 2) | 1.04x / 1.29x / 1.34x | 1.24x / 1.47x / 1.54x |
| 7360, 256 (real117 stage 1) | 1.33x / 1.38x / 1.41x | 1.52x / 1.52x / 1.64x |
| 19321, 128 (app59f stage 2) | 1.07x / 1.34x / 1.24x | 1.39x / 1.34x / 1.60x |
| 3220, 256 (app59f stage 1) | 1.07x / 1.26x / 1.31x | 1.32x / 1.39x / 1.49x |
| 6601, 128 (word stage 2) | 0.91x / 1.07x / 1.35x | 1.23x / 1.44x / 1.48x |
| 1100, 256 (word stage 1) | 1.00x / 1.04x / 1.21x | 0.87x / 1.43x / 1.51x |

(d = 3 / 5 within ±10 % of d = 1.) The only losses are K = 3 at the shortest lengths (Mac L = 6601: -6 %; iPhone
L = 1100: -13 %); `F` is a runtime dim, so the graph can't pick per length, and those convs are a small slice of
`word`'s time — Winograd is used for every resblock conv.

Kokoro E2E, Mac GPU (`prof-lokinfer run kokoro -d mac-gpu`, steel → Winograd): word 29.1 → 26.8 ms, app47f
48.5 → 46.4, app59f 55.6 → 52.5, app71m 66.4 → 60.6, real117 115.7 → 101.2, rand510 330.8 → 288.8 ms. In the rand510
profile the GPU conv time (snake + convs) went 235.7 → 179.4 ms (mm 154.8 ms + transform 24.6 ms).

iPhone GPU E2E (`prof-lokinfer run kokoro -d iphone-gpu`, same session, steel → Winograd): app59f 98.6 → 70.0 ms
(-29 %), real117 229.5 → 159.6 ms (-30 %), rand510 691.7 → 487.8 ms (-29 %; a first rand510 pair was throttled and
was re-run). The iPhone gains more than the Mac because its steel conv was further from the f32 peak per FLOP of
work saved and the fused snake removes a full read + write of the activation, which costs more at iPhone bandwidth.
Mac dashboard after the change (`bench/models/bench_kokoro_gpu.py`, recorded): real28 40.0,
app47f 43.9, app59f 52.5, app71m 66.9, real117 113.6, rand510 303.2 ms.

## Round 2 (2026-11-01, later): stats epilogue, decoder, conv_post, GEMM tuning

- **Channel stats in the GEMM epilogue** (`stats=True` → `wino_mm[_res]_stats_lc_f32_gpu` + `wino_adain_affine_lc_f32_gpu`):
  each 32-tile threadgroup reduces its 128 output rows per column (two register passes, lane shuffles over the row
  bits, threadgroup memory across the 2 simdgroup rows) into (mean, M2) chunks; for d > 1 a chunk is 128 rows of one
  residue class, so the affine kernel derives chunk counts from (rho, y). Removes most `channel_stats_lc` passes in the
  generator (rand510 Mac: 10.9 → 2.2 ms). Per-kernel cost on the M2 Max: K = 11 GEMM -8 %, K = 3 +3 %, **K = 7 +8-19 %**
  (its 2-group kernel crosses a register/occupancy cliff), so K = 7 keeps the separate stats pass. E2E Mac: app59f
  53.2 → 52.2 ms, s9f ~98 → 96.3, rand510 ~297 → 290.7.
  Re-checked after the coordinator's threadgroup-parallel `lk_stats_lc_tg` (2026-11-02, Mac, 2 runs each, separate
  stats pass → epilogue stats): app59f 48.5 / 48.1 → 48.1 / 48.8 ms, s9f 93.8 / 94.2 → 93.9 / 88.4 ms: neutral to
  slightly positive, kept. (rand510 not re-measured: macbench lock queue.)
- **Decoder / F0-N AdainResBlk1d K = 3 convs** (Ci 256-1104, Co 256-1024, L = F or 2F): F(4,3) with the AdaIN +
  leaky(0.2) activation fused into the transform (`wino_tf_leaky_lc_f32_gpu`; `LOKINFER_KOKORO_WINO_DEC=0` for A/B).
  GPU time on rand510 (Mac) 16.4 → 12.2 ms, but E2E unchanged on the Mac (the decoder overlaps the CPU front end):
  app59f 53.3 / 53.3, s9f 97.3 / 99.4, rand510 295.0 / 298.2 (steel / Winograd, 2 runs each). iPhone: see below.
- **conv_post** (K = 7, Co 22 padded to 64): Winograd with leaky(0.01) fused (identity affine). Neutral on the Mac.
- **iPhone** (same session, `LOKINFER_KOKORO_WINO_DEC` 0 → 1, stats + conv_post on in both): app59f 67.0 → 65.9 ms,
  s9f 146.1 → 142.0 ms. Dashboard (`bench_remote_models.py kokoro --gpu-only`, nominal) before → after this round:
  word 33.2 → 31.6, app47f 57.1 → 54.1, app47m 65.2 → 61.7, app59f 70.4 → 66.4, app59m 76.4 → 71.9 ms. The sweep's
  long cases (s9f 206.6, real117 232.7, rand510 680.1 ms) ran after the A/B runs on a phone reported "nominal" but
  clearly clock-limited (prof-lokinfer minutes earlier: s9f 142, real117 159.6, rand510 487.8): re-record them cool.
- **The input transform can't be fused away**: its activation needs the AdaIN affine, i.e. the stats of the whole
  previous conv output, so no producer epilogue can write V; only the stats pass could be folded (above).
- **GEMM loader**: register prefetch of the next (point, channel block) and BK = 32 were both slower (prefetch:
  -10..-28 %, register pressure). GPU counters (M2 Max, L = 19321, K = 11): Winograd GEMM occupancy 8.4 % vs steel
  16.7 %, F32 utilization 35 % vs 58 %: the M = 4 output-row accumulators (64 floats/thread) + the point accumulator
  bound occupancy. Smaller tiles (32x32, 16x64) raise occupancy but lose more on fragment reuse (A/B'd). This is the
  structural limit of the fused output transform; the next idea would be F(2,3) (2 output rows, 32x32 per simdgroup
  like steel) for K = 3 at 0.67x the multiplications.
- **Per-length K = 3 choice**: not done — `F` is a runtime dim and the only losses are K = 3 convs at word-sized
  inputs (-6 % Mac, -13 % iPhone at L = 1100); a runtime switch would need the steel weights and the activation pass
  on the Winograd op too.

## Round 3 (2026-11-02): F(2,3), K = 7 stats, Metal 4 tensor ops, ResNet point-split

- **F(2,3) for K = 3** (`wino_weights(w, tile=2)`: 4 points, 2 output rows per tile, 64-tile threadgroups = 32x32
  per simdgroup like steel, 0.67x the multiplications): **rejected**. iPhone (winosnake F(4,3) → F(2,3), ms):
  L44161/C128 1.65 → 1.97, L19321 0.74 → 0.90, L7360/C256 0.92 → 1.07, L3220 0.50 → 0.51, L1100 0.19 → 0.19;
  Mac within -12..+9 %. The halved register cost does not buy back the extra 1/3 of multiplications. The path stays
  built and tested (kernels `*_m2a4*`), unused.
- **K = 7 GEMM with fused stats** (8-24 % slower than without): re-reading the outputs from cache instead of keeping
  the accumulators live made it worse (L44161: 1.94 → 2.05 ms). Not fixed: a register / occupancy cliff of the
  2-group kernel; K = 7 keeps the separate stats pass (now the coordinator's threadgroup-parallel `lk_stats_lc_tg`).
  With that faster pass the fused stats are still neutral-to-better for K = 3 / 11 (rand510 Mac 293.2 vs 296.4 ms
  average of 2 pairs, app59f / s9f within noise) → kept.
- **Metal 4 tensor ops, f32 (`mpp::tensor_ops::matmul2d`)**, experiment code removed: plain f32 GEMM [M, K] x [K, K]
  vs steel / MPS — Mac M=11041 K=384: steel 610 µs, MPS 580, matmul2d 695-955 (tile 64x64 best); iPhone: steel 1593,
  MPS 1316, matmul2d 1295 (64x64; other tiles ≈ steel). As the Winograd point-GEMMs (all points batched, V's
  overlapping rows as one strided matrix, exact vs float64): Mac 0.85-1.46x and iPhone 1.06-1.27x the *whole* fused
  Winograd GEMM (which also does the output transform + epilogue) — before the extra output-transform pass the MPP
  split would need. Not worth it under the f32 rule.
- **ResNet-18 2-D: point-split for small grids** — the 16 points are split over grid z (2..16 ways, until the grid has
  ≥ 100 threadgroups); each threadgroup atomically adds its partial Y (`atomic_float`) into an output zeroed by
  `lk_wino_fill_zero`, then `lk_wino2d_finalize` applies bias / residual / ReLU. Sweep (Mac, splits 1-16, best):
  N=8 14²x256 1 (1.36x steel), 7²x512 8 (1.20x); N=1 14²x256 8 (1.55x), 7²x512 16 (2.79x), 28²x128 4 (1.06x);
  the target-100 rule picks these except N=8 7²x512 (2 splits: 1.11x). 56² and 28² layers stay on steel (≤ 1.0x
  at N=8). A static per-layer choice (Co ≥ 256) covers it, no runtime switch: **ResNet-18 Mac GPU N=1 2.18 → 1.43 ms,
  N=8 7.07 → 6.63-6.90 ms**, logits within 3e-6 of the steel build, same top-1. (Co ≥ 128 too: N=1 1.40, N=8 6.87 —
  not taken.) Needs iPhone validation (6 GPU cores: the split target was tuned on 30).

## 2-D F(2x2, 3x3) for ResNet-18 (built; integration patch with the coordinator)

Same GEMM kernel (`D2` template flag: 16 points, 2x2 outputs per tile, AT ⊗ AT folded in registers, bias / residual /
ReLU epilogue) + `lk_wino_tf2d_nhwc` (V [16, tiles, C] = 4x the input). Ops `conv2d_wino_nhwc_f32_gpu` +
`wino2d_weights`; tests pass at 1e-4 (5 shapes x 3 epilogues). Mac microbench, 3x3 s1 conv + ReLU, steel → Winograd:

| N | 56²x64 | 28²x128 | 14²x256 | 7²x512 |
|---|---|---|---|---|
| 8 | 1.04x | 0.93x | 1.36x | 0.78x |
| 1 | 0.85x | 0.71x | 0.62x | 0.49x |

Small grids (7² x 512: 32-64 threadgroups) and the 4x V traffic eat the 2.25x FLOP cut. `build_gpu` compiles with
symbolic N, H, W, so a per-shape choice needs a runtime switch. Superseded by round 3's point-split (above), which
wins for every Co ≥ 256 layer at N = 1 and N = 8.
