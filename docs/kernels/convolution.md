# Convolution kernels

Sibling kernel docs: [gemm.md](gemm.md) · [elementwise.md](elementwise.md) · [reductions.md](reductions.md) ·
[attention.md](attention.md).

Every convolution kernel in lokinfer (CPU and Metal GPU, f32 only), how each family works, what was measured, what was
tried and dropped, and what is still open. Measurements are quoted from the docs and code comments named in each row;
"not measured" means no number exists in the repo. Devices: **M2 Max** (Mac, 8 P + 4 E cores, pool of 8 workers) and
**iPhone 18 Pro Max** (2 P + 4 E cores, pool of 6 workers, SME2; [todo.md](../todo.md) M6 "Thread pool size").

Code: [`kernels/cpu/kokoro_conv.cpp`](../../kernels/cpu/kokoro_conv.cpp),
[`kokoro_conv_fused.cpp`](../../kernels/cpu/kokoro_conv_fused.cpp), [`kokoro_common.h`](../../kernels/cpu/kokoro_common.h),
[`sme2_conv.{h,cpp}`](../../kernels/cpu/sme2_conv.cpp), [`conv2d.cpp`](../../kernels/cpu/conv2d.cpp);
[`kernels/gpu/kokoro_lc.{metal,cpp}`](../../kernels/gpu/kokoro_lc.metal), [`vision.metal`](../../kernels/gpu/vision.metal),
[`winograd.{metal,cpp}`](../../kernels/gpu/winograd.metal). Op definitions (Python): `python/lokinfer/ops/kokoro.py`,
`kokoro_gpu.py`, `vision.py`, `winograd.py`. Benchmarks: `bench/kernels/` (below). Background notes:
[m2-conv-proof.md](../impl-notes/m2-conv-proof.md), [winograd-conv.md](../impl-notes/winograd-conv.md),
[iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md).

## 1. Kernel inventory

Names are the registered kernel names (`registerKernel` in the `.cpp` files). "cf" = channel-first `[C, L]`, "cl" =
channel-last `[L, C]`.

### CPU, 1-D (Kokoro)

| kernel | layout | prologue / epilogue | used by |
|---|---|---|---|
| `conv1d_f32_cpu`, `conv1d_nobias_f32_cpu` | cf, w `[Co, Ci, K]` | bias (as the GEMM's C, β = 1) | Kokoro: basic graph (`fused=False`), text encoder k5, decoder F0/N stride-2 convs, `asr_res` and other 1×1s, the generator's noise convs (stride 6; CPU in both builds), CPU `conv_post`; `_nobias` = AdainResBlk1d 1×1 shortcut |
| `conv1d_{snake,leaky,affine}_f32_cpu`, `conv1d_{snake,leaky}_res_f32_cpu` | cf | **pro:** per-input-channel AdaIN affine `(x − mean)·scale + beta`, then Snake(α) / leaky(slope) / none; **epi:** bias, residual, `oscale` (leaky_res only), per-(channel, 128-sample chunk) `(mean, M2)` stats | Kokoro CPU fused graph: F0/N predictor AdainResBlk1d K = 3 convs (always direct, see §2.3); generator resblocks / decoder only when `LOKINFER_KOKORO_WINO_CPU=0` |
| `conv1d_{snake,leaky}_wino[_res]_f32_cpu`, `conv1d_affine_wino_f32_cpu` | cf; U `[A, ⌈Co/32⌉, Ci·G, 32]` | same as above (same `ConvFusedArgs`) | Kokoro CPU default: generator resblocks + noise resblocks (snake), decoder AdainResBlk1d K = 3 (leaky / affine) |
| `conv_transpose1d_f32_cpu` | cf, w `[Ci, Co, K]` | bias | Kokoro CPU generator upsamplers (k20 s10, k12 s6) |
| `conv_transpose1d_dw_f32_cpu` | cf, w `[C, 1, K]` | bias | AdainResBlk1d upsample "pool" (CPU) |
| `channel_stats_f32_cpu`, `adain_affine_f32_cpu` | cf | — | stats / affine feeding the fused convs' prologue (details in [reductions.md](reductions.md)) |

### CPU, 2-D (vision, NHWC)

| kernel | layout | prologue / epilogue | used by |
|---|---|---|---|
| `conv2d_nhwc_f32_cpu`, `_relu`, `_relu6`, `_leaky`, `_res`, `_res_relu` | NHWC, w `[KH, KW, Ci, Co]` | bias, [residual], act ∈ {none, ReLU, ReLU6, LeakyReLU(0.2)} | ResNet-18, MobileNetV2 (stem, head, unfused blocks), DCGAN D |
| `dwconv2d_nhwc_f32_cpu`, `_relu6` | NHWC, w `[KH, KW, C]` | bias, act | MobileNetV2 (`fused=False`, and inside `mbv2_block` for small maps) |
| `mbv2_block_nhwc_f32_cpu`, `_res` | NHWC | 1×1 expand + ReLU6 → dw 3×3 + ReLU6 → 1×1 project [+ x], all in one kernel | MobileNetV2 CPU default (`fused=True`) |
| `convt2d_gather_nhwc_{relu,tanh}_f32_cpu` | NHWC; input y = x @ W `[B, H, W, K·K·Co]` | bias, ReLU / tanh | DCGAN G layers 2–5 (after a GEMM, see [gemm.md](gemm.md)) |

### GPU, 1-D (Kokoro, channel-last)

| kernel (Metal function) | layout | prologue / epilogue | used by |
|---|---|---|---|
| `conv1d_lc_f32_gpu`, `conv1d_lc_res_f32_gpu` (`lk_conv1d_lc_f32`) | cl, w `[Co, K, Ci]`; Ci % 16 == 0, K ≤ 16, stride 1 | bias, [residual], `oscale` | Kokoro GPU: 1×1 shortcuts; every generator / decoder conv when Winograd is switched off (`LOKINFER_KOKORO_WINO=0`, `_WINO_DEC=0`) |
| `wino_tf_lc_f32_gpu`, `wino_tf_snake_lc_f32_gpu`, `wino_tf_leaky_lc_f32_gpu` (`lk_wino_tf_lc_m4a{6,7}`) | cl → V `[d, A, rmax, Ci]` | input transform, optionally of snake(AdaIN affine) or leaky(affine) | first half of every Winograd conv |
| `wino_mm[_res]_lc_f32_gpu`, `wino_mm[_res]_stats_lc_f32_gpu` (`lk_wino_mm_lc_m4a6n1`, `m4a7n2`, `m4a7n3`, `_bt16`, `_stats`); Co % 64 == 0, Ci % 16 == 0 | V, U `[A, G·Ci, Co]` → cl | per-point GEMMs + output transform, bias, [residual], `oscale`, [channel stats] | Kokoro GPU default: generator resblocks + noise resblocks (K 3/7/11), decoder and F0/N AdainResBlk1d K = 3, `conv_post` (K 7, leaky 0.01) |
| `wino_adain_affine_lc_f32_gpu` | stats from `_stats` | — | AdaIN affine from the Winograd epilogue stats (K = 3 / 11 generator convs) |
| `convt_gather_lc_f32_gpu`, `convt_gather_add_lc_f32_gpu` (`lk_convt_gather_lc`) | cl; y = x @ W `[Lin, K·Co]` | bias, [+ add], optional 1-row reflect pad on the left | Kokoro generator upsamplers (after `gemm_f32_gpu_mps`; `add` = the noise branch, reflect = `ups[1]`) |
| `dwconvt_lc_f32_gpu` (`lk_dwconvt_lc`) | cl, w `[C, K]` | bias | AdainResBlk1d upsample "pool" (GPU) |

### GPU, 2-D (vision, NHWC)

| kernel (Metal function) | layout | prologue / epilogue | used by |
|---|---|---|---|
| `conv2d_nhwc_f32_gpu`, `_relu`, `_relu6`, `_leaky`, `_res`, `_res_relu` (`lk_conv1d_lc_f32`, `lk_conv1d_lc_f32_bm32`) | NHWC, w `[Co, KH, KW, Ci]`; Ci % 16 == 0, KH, KW ≤ 16 | bias, [residual], act ∈ {none, ReLU, ReLU6, leaky 0.2} | ResNet-18, MobileNetV2 (channels padded to 16 / 32), DCGAN D (incl. the 1×1 conv over im2col patches) |
| `dwconv2d_nhwc_f32_gpu`, `_relu6` (`lk_dwconv2d_nhwc_k3s1c4`, `_k3s2c4`, generic `lk_dwconv2d_nhwc`) | NHWC | bias, act | MobileNetV2 |
| `im2col_nchw_f32_gpu` (`lk_im2col_nchw_tile`, `lk_im2col_nchw`) | NCHW → patches `[N·Ho·Wo, Kpad]` | zero padding to Kpad | ResNet-18 stem (K 147 → 160, then `gemm_steel_f32_gpu` bias + ReLU), DCGAN D first conv (K = 48) |
| `convt2d_gather_nhwc_{relu,tanh}_f32_gpu` (`lk_convt2d_gather_nhwc`) | NHWC | bias, ReLU / tanh | DCGAN G |
| `wino_tf2d_nhwc_f32_gpu`, `wino_mm2d_nhwc[_res][_relu]_f32_gpu` (`lk_wino_tf2d_nhwc`, `lk_wino_mm2d_nhwc[_bt16]`) | NHWC, 3×3 s1 p1, Ci % 16, Co % 64 | bias, [residual], [ReLU] | **none** — built and tested, not wired into any model (§2.8) |

Related layout kernels (not convs, but part of the conv paths): `nchw_to_nhwc_pad_f32_gpu` (zero-pads channels to the
steel loader's multiple of 16), `nchw_to_nhwc_f32_cpu`, `nhwc_to_nchw_f32_{cpu,gpu}`, `lc_to_cf_f32_gpu`,
`concat_pad_lc_f32_gpu` / `pad_cols_lc_f32_gpu` (Kokoro's 514 / 1090 channels → 528 / 1104), `upsample_nearest1d_f32_cpu`,
`upsample2_lc_f32_gpu`, `pad_reflect1d_f32_cpu` — see [elementwise.md](elementwise.md).

## 2. How each family works

### 2.1 Shared CPU machinery (`kokoro_common.h`)

- `detail::sgemm`: row-major `cblas_sgemm`. Below `kSmallGemmFlops` = 5e7 FLOP it forces single-threaded BLAS for the
  call (Accelerate's own threading costs more than it saves there); above, it first `park()`s the runtime pool's
  spinning workers so they don't compete with Accelerate's threads.
- `detail::parallelForSt`: `ThreadPool::parallelFor` with single-threaded BLAS inside every task — one Accelerate
  (AMX / SME) stream per core, the M2 conv-proof recipe. The caller's BLAS threading is restored afterwards (a bug where
  it was left single-threaded was fixed in M3, [todo.md](../todo.md) "Profile-driven fixes").
- `detail::scratch(n, slot)`: grow-only thread-local buffers; steady-state calls don't allocate.
- `detail::sinSq4`: sin² for the Snake prologue (π reduction, degree-11 odd polynomial), replacing a `vvsinf` pass.

### 2.2 Basic 1-D conv (`conv1d_f32_cpu`, `kokoro_conv.cpp`)

- 1×1 stride-1 unpadded: one sgemm, no im2col.
- Otherwise time-tiled im2col (`[Ci·K, tile]` column block, tile width from an 8 MB budget, min 64 samples) and one
  `detail::sgemm` per tile, sequential on the calling thread (Accelerate's threading above 5e7 FLOP). Stride 1 rows
  are `memcpy`s with zero edges; strided rows gather per sample. Dilation is folded into the row offsets.
- `conv_transpose1d_f32_cpu`: per input tile one sgemm `y[Co·K, n] = Wᵀ x` (W viewed as `[Ci, Co·K]`) followed by a
  scalar col2im overlap-add into the bias-filled output. Not polyphase (open, §5).
- `conv_transpose1d_dw_f32_cpu`: scalar scatter per channel.

### 2.3 Fused direct 1-D conv (`conv1d_fused_f32`, `kokoro_conv_fused.cpp`)

`out = conv1d(act((x − mean)·scale + beta), w) + bias [+ residual]` (·`oscale` for leaky_res), stride 1, zero padding
applied *after* the activation (as torch does).

- Output processed in time tiles of `tile` samples (a multiple of 128). Per tile and input channel the transformed
  input window `[t0 − pad, t0 + n − pad + (K−1)·d)` is built once (centered affine: mean subtracted first, for
  precision when the variance is tiny next to the mean), then im2col copies K shifted rows from it, then one
  single-threaded sgemm `[Co, Ci·K] × [Ci·K, n]`.
- Epilogue on the tile while it is in L2: bias / residual / scale, then the next AdaIN's stats as `(mean, M2)` per
  (output channel, 128-sample chunk). The chunk is fixed (not the conv tile) so the stats don't depend on tiling;
  `adain_affine_f32_cpu` combines the partials exactly in double.
- Default tile: `ops.conv_tile(ci, k)` = largest multiple of 128 whose im2col block fits 3 MiB, clamped to [128, 768]
  (512 when Ci or K is symbolic).
- Threading: the whole conv runs inline on the calling thread below **4 M MACs** (`kParallelMacs`; pool hand-off
  ~5–20 µs) or when there is one tile; otherwise tiles go to `parallelForSt`.
- Kokoro model use: the AdaINResBlock1 chain is `channel_stats → adain_affine → conv(snake) → adain_affine →
  conv(snake, +x)`, each conv emitting the stats the next AdaIN needs; the three resblocks of a stage share their input
  stats. The F0/N predictor's K = 3 AdainResBlk1d convs always use this direct kernel (CPU Winograd there costs ~1 dB of
  audio SNR, §4).

### 2.4 CPU Winograd 1-D conv (`conv1d_wino_f32`, `kokoro_conv_fused.cpp`)

Same contract and epilogue as §2.3, ~half the multiplications. The decomposition is the GPU one (§2.7):

- **Transforms**: F(4,3) (A = 6 points {0, ±1, ±2, ∞}) for K = 3; nested F(4,4) (A = 7, points {0, ±1, ±2, −1/2, ∞})
  for K = 4..12, taps split into G = 2 or 3 groups of 4 at offsets 0/4/8 that share one transformed input (group g of
  tile q reads transformed tile q + g). Tables are compile-time `constexpr` (`WinoTables<6|7>`) so the unrolled loops
  drop zero and ±1 terms; the kernel checks the op's `tr` input against them (≤ 1e-6 relative) and refuses on mismatch.
- **Dilation**: output residue class ρ only reads `x[ρ + d·j]`, i.e. an undilated conv. The window row is
  de-interleaved into d class rows (NEON `vld2`/`vld3` for d = 2/3, a `vqtbl4q`/`vqtbx1q` byte-table shuffle for d = 5)
  and re-interleaved after the output transform.
- Per tile: window + input transform → V `[A][Ci][cols]`; per point ξ one GEMM `M[ξ] = U[ξ] · V[ξ]` with the group
  shift as a column offset; output transform (+ epilogue). U is pre-packed by `ops.wino_weights_cpu`
  (`[A, ⌈Co/32⌉, Ci·G, 32]`, 32 output channels innermost).
- **GEMM engine**: SME2 (`sme2::convTile`, §2.5) when `sme2::available()`, reading the B rows in place shifted by g;
  otherwise Accelerate `sgemm` on a `[Ci·G, cols]` copy (the M2 Max path).
- **Scheduling** (MACs counted ×0.5; parallel above 4 M and with > 1 pool thread):
  1. fewer than 2 tiles per thread → one tile in **two parallel phases** (V over channel blocks; then per 32 output
     channels the A GEMMs + output transform + epilogue);
  2. SME2 + a P/E core split + > 1 tile → **P/E pipeline**: GEMMs on P cores, transforms + epilogue mostly on E cores
     (each thread classifies itself per step with `pthread_cpu_number_np`), through a ring of R tile buffers
     (R = 16 MB / slot size, clamped to [4, 16]); an E core takes a GEMM only after it sat unclaimed for 2 ms;
  3. otherwise tiles on `parallelForSt` (or inline).
- Default tile: `ops.wino_tile(ci)` = 512 up to Ci = 256, shrinking for wider inputs (min 128); 1024 when Ci is symbolic.

### 2.5 SME2 tile kernel (`sme2_conv.cpp`)

`out[co, t] = Σ_{c,kk} W[co, c, kk] · xw[c, kk·d + t]` as FMOPA outer products into the four 16×16 f32 ZA tiles
(2×2 blocking = 32 output channels × 32 samples per block). Weights pre-packed `[⌈Co/32⌉][Ci·K][32]` so one
multi-vector load fetches the A column pair; B rows are read straight from the window (tap kk = the same row shifted by
kk·d, no im2col). Full 32-sample blocks use `svld1_f32_x2`; tails use predicates. Compiled with
`-march=armv8.6-a+sme2` (CMakeLists.txt); `available()` checks `hw.optional.arm.FEAT_SME2` at runtime and the
non-SME2 build has empty stubs. The same file holds `gemvT` (LSTM step GEMV with y held in ZA), not a conv.

In the shipped graph `convTile` is reached only through the Winograd conv (with K = groups, d = 1). The direct
SME2 conv with per-call weight packing was dropped (§4).

### 2.6 CPU 2-D convs (`conv2d.cpp`)

**`conv2d_nhwc`** = `[P, KH·KW·Ci] × [KH·KW·Ci, Co]` GEMMs (P = output pixels); weights are already im2col row order.
- Pointwise (1×1, s1, p0): the input is the A matrix, no im2col.
- im2col copies a whole kernel row (KW·Ci contiguous floats in NHWC) at once when it's fully inside; per-tap copies at
  edges; a plain loop instead of `memcpy` when Ci < 16 (3-channel stems).
- < 4 M MACs: one thread, ~3 MB im2col tiles (`kColBudget`), each tile im2col'd then multiplied.
- One-tile map with long K (K ≥ 1024, pixels ≤ one tile, ≥ 5e7 FLOP; ResNet-18's 14×14 / 7×7 stages, DCGAN D): one
  sgemm with Accelerate's own threading; if the weights exceed 8 MiB (DRAM-bound, ResNet-18's 3×3×512×512), two K
  halves on two threads with the partials summed, then the epilogue.
- Otherwise a grid of pixel tiles × Co chunks, about `kTilesPerThread` = 3 tasks per pool thread of ≥ 0.5 M MACs
  each; the grid shape minimizes bytes read (every pixel tile streams its weight chunk, every Co chunk its im2col rows).
  With one Co chunk each task im2cols its tile in cache; with several, one shared im2col pass first. Single-threaded
  sgemm per task, then the bias / residual / act epilogue on the tile.

- **SME2 path [built, opt-in `LOKINFER_SME2=1`, needs iPhone validation]**: on SME2 CPUs, `smeWins()` convs (not
  pointwise, Ci % 16 = 0, ≤ 1024 pixels, weights ≤ 8 MiB, K ≥ 1024 or 1×1 strided) run on the shared engine
  ([gemm.md](gemm.md)) as one stream on the calling thread: the pixel tile's taps are gathered straight into packed A
  panels (`packPanelsGather`, no im2col buffer), B = the weights in place. All-convs-on-SME A/B on the iPhone: ResNet-18
  N=1 4.18 → 3.88 ms, N=8 23.3 → 23.6, MobileNetV2 N=1 1.65 → 2.28; per op it wins on 28×28 / 14×14 maps (14×14×256
  266-294 → 204-210 µs, strided 1×1 70 → 26 µs) and loses on 56×56 (215 → 240-300: the per-call packing), the stem
  (378 → 733: single-threaded im2col) and the 9.4 MB 7×7×512 layers (330-380 → 420). The rule keeps the winners; the
  combination is untested. **Planned**: a channel-planar padded layout (`[C, N, H+2, W+2]`) where B = the activation
  rows read in place with a per-k row offset and A = weights pre-packed at init (no per-call packing; stride 2 via
  2×2 phase planes), like the Kokoro conv tile.
- Inline threshold `kParallelMacs` = 4 M MACs on every device: M2 Max A/B 2026-11-02, 1e6 → ResNet-18 N=1 4.3 → 4.8 ms,
  1.6e7 → MobileNetV2 N=1 1.5 → 2.6 ms; the iPhone was tuned around it (re-check 2e6 / 8e6 there).

**`dwconv2d_nhwc`**: tasks = (output row, channel chunk), chunk count chosen so small maps still give ~4 tasks per
thread; serial below 2e5 multiply-adds. 3×3 stride 1/2 is fully specialized (`dwRowK<3, S, Act>`): an 8-channel block's
18 weight vectors stay in registers, two output pixels per iteration (independent FMA chains), padded rows read a zero
row, only the first / last columns check bounds. Other kernel sizes take a scalar generic path (no model uses it).

**`mbv2_block_nhwc`**: the whole inverted-residual block. One task per (image, band of R output rows); the band's
expanded input rows (with the 3×3 halo) and its depthwise output live in per-thread scratch (band budget 384 KiB) so the
expanded tensor never round-trips through memory. Expand = one sgemm over the band's contiguous rows + ReLU6
epilogue, then `dwRowK<3, s, Relu6>` per row, then the project sgemm with bias [+ x]. When the halo would cost > 1.5×
the band's rows (small maps) or the expand is < 4 M MACs, it runs expand → dw → project unfused inside the kernel.

**`convt2d_gather_nhwc`** (DCGAN G): ConvTranspose2d as one GEMM `y = x @ W[Ci, K·K·Co]` (a separate op) plus this
gather: each output pixel sums its ≤ ⌈K/s⌉² contributions (no atomics), adds bias, applies ReLU, or tanh through
`vvtanhf` over the whole row. Parallel over output rows above a per-pixel work estimate (rows × Wo × (16 + 4·Co) ≥ 4 M).

### 2.7 GPU steel conv (`lk_conv1d_lc`, `kokoro_lc.metal`)

A lokinfer-owned copy of MLX's steel implicit-GEMM conv body (MLX v0.32.4, MIT, vendored in `third_party/mlx_steel`)
with a fused epilogue. Despite the name it is the general NHWC conv2d; a 1-D conv is H = 1.
- GEMM view: M = output pixels (N·Ho·Wo, or Lo), N = Co, K = KH·KW·Ci. A loader = `Conv2DInputBlockLoaderSmallFilter`
  (gathers shifted / dilated input rows, needs Ci % 16 == 0 and K ≤ 16), B loader = `Conv2DWeightBlockLoader`
  (weights `[Co, KH, KW, Ci]`), `BlockMMA` on `simdgroup_matrix` f32.
- Tiles: 64×64×16, 2×2 simdgroups (128 threads). A 32×32×16 instantiation (`_bm32`) is picked by the **2-D** dispatch
  when the 64×64 grid has fewer than 128 threadgroups (`kSmallGrid`); the 1-D Kokoro dispatch always uses 64×64.
- Epilogue (on the MMA fragments): + bias, + residual, × scale, then ReLU / ReLU6 / LeakyReLU(0.2).
- 1-D dispatch: stride 1 only (strided Kokoro convs stay on the CPU), dilation via `kdil`.

### 2.8 GPU Winograd (`winograd.metal`, `winograd.cpp`)

Exact Toom-Cook transforms are built over rationals on the host (`ops.winograd.transforms`, Vandermonde + inverse,
BT rows normalized to max |·| = 1); `wino_weights` precomputes `U = G w` (`[A, G·Ci, Co]`) and the table
`tr = [BT | AT]` in float64. Variants as in §2.4: F(4,3) (A = 6, G = 1) for K = 3, nested F(4,4) (A = 7, G = 2 for
K = 4..8, G = 3 for K = 9..12). Dilation by residue classes (requires padding % d == 0).

Two kernels:
1. **`lk_wino_tf_lc`** writes V `[d, A, rmax, Ci]` (A/4 = 1.5–1.75× the input). One thread = 4 channels (float4) ×
   RT = 1 tile; grid (C/4, rmax, d), threadgroup 32×4. Optionally applies AdaIN affine + Snake (same math as
   `lk_act_lc`, `precise::sin`) or affine + leaky on load, which removes the separate activation kernel and its output.
2. **`lk_wino_mm_lc`**: per point ξ a GEMM over (group, Ci) on V (`simdgroup_float8x8`, accumulators kept as plain
   `float2` thread elements, every loop unrolled), folded straight into M = 4 output-row accumulators by `AT` in
   registers; then bias, [residual], × scale. Threadgroup = 128 threads (2×2 simdgroups), 32 tiles (128 output rows) ×
   64 output channels, BK = 16. `_bt16` (16-tile) variants when the 32-tile grid has < 128 threadgroups. The `_stats`
   variants (32-tile only) also reduce each threadgroup's 128 rows per column into `(mean, M2)` (two register passes,
   lane shuffles, threadgroup memory across the two simdgroup rows); for d > 1 a chunk is 128 rows of one residue class,
   and `lk_wino_adain_affine_lc` derives each chunk's count from (ρ, y).

Model use: generator resblocks (+ noise resblocks) with snake fused, stats fused for K = 3 / 11 only (K = 7 keeps the
separate `channel_stats_lc` pass; `_snake_conv_gpu` in `models/kokoro/model.py`); decoder / F0-N K = 3 convs with
leaky(0.2) fused (Ci zero-padded to a multiple of 16); `conv_post` K = 7 with Co 22 → 64 and leaky(0.01) fused via an
identity affine. The 1×1 shortcuts stay on steel. Switches: `LOKINFER_KOKORO_WINO=0` (generator), `_WINO_DEC=0`
(decoder / F0-N), `LOKINFER_KOKORO_WINO_STATS=0` (epilogue stats; marked "TEMP A/B" in the model).

**2-D F(2×2, 3×3)** (built, not integrated): `lk_wino_tf2d_nhwc` (separable BT ⊗ BT, V `[16, tiles, C]` = 4× the
input) + the same GEMM kernel with the `D2` flag (16 points, 2×2 outputs per tile, AT ⊗ AT in registers, bias /
residual / ReLU). Ops `conv2d_wino_nhwc_f32_gpu` + `wino2d_weights`.

### 2.9 Other GPU conv kernels

- **ConvTranspose1d** = `gemm_f32_gpu_mps(x, W[Ci, K·Co])` + `lk_convt_gather_lc`: one thread per output element sums
  the taps `kk ≡ t + pad (mod s)`, adds bias and optionally the noise branch (`add`), and can emit Kokoro's
  `F.pad(..., (1, 0), "reflect")` (`reflect_left`) without a separate pass.
- **`lk_dwconvt_lc`**: depthwise ConvTranspose1d, one thread per output element.
- **Depthwise 2-D**: `lk_dwconv2d_nhwc_k3s{1,2}c4` when K = 3×3, C % 4 == 0, stride 1/2 (all of MobileNetV2): one
  thread = 4 channels (float4) × 4 consecutive output columns, the input row segment loaded once for the 4 outputs;
  stride is a template parameter so all register-array indices are compile-time. Otherwise the generic
  one-thread-per-element `lk_dwconv2d_nhwc`.
- **Stem im2col** from NCHW: `lk_im2col_nchw_tile` (one threadgroup of 256 threads = 32 output pixels of one row;
  the C × KH × ((32−1)·s + KW) input patch staged in threadgroup memory with coalesced reads, a per-column offset table,
  no division per element) when the patch fits 4096 floats and Kpad ≤ 256; else the per-element `lk_im2col_nchw`.
  Columns are `(y·KW + x)·C + c`, zero-padded to Kpad, so the patches feed a GEMM (ResNet-18) or a 1×1 steel conv
  (DCGAN D) directly.
- **`lk_convt2d_gather_nhwc`**: DCGAN G's gather, one thread per output element, ReLU or `precise::tanh`.

## 3. Measured performance

FLOP counts are 2·MACs. "E2E" = whole-model latency.

### 3.1 Kokoro 1-D, kernel level (M2 proof, rand510 shapes)

Stage 1 = 128 → 128 ch, L = 132 121; stage 0 = 256 → 256, L = 22 020; "same" padding, stride 1. Mac: M2 Max, Low Power
Mode off, 2026-11-01. iPhone: thermal / LPM not recorded. Source: [m2-conv-proof.md](../impl-notes/m2-conv-proof.md).

| candidate (CPU, Mac) | stage 1 k11 d5 | stage 0 k7 d3 |
|---|---|---|
| im2col + one Accelerate sgemm (Accelerate threading) = `conv1d_f32_cpu`'s scheme | 97 ms (0.49 TF/s) | 22 ms (0.93) |
| shifted GEMMs (one sgemm per tap, no im2col) | 54 ms (0.88) | 14 ms (1.46) |
| shifted, 8 single-threaded streams | 66 ms (0.73) | 12 ms (1.69) |
| **time-tiled im2col, 512-sample tiles, 8 single-threaded streams** | **23 ms (2.04)** | **11.7 ms (1.73)** |
| time-tiled im2col, 2048-sample tiles | 149 ms | 65 ms |

The im2col / shifted / streams rows are also in `bench/kernels/results_conv1d_cpu.json` (Google Benchmark, medians);
the tiled rows are only in the note.

| shape | iPhone CPU tiled im2col (best streams) | iPhone GPU steel | Mac GPU steel |
|---|---|---|---|
| stage 1 k3 d1 | 9.03 ms, 1.44 TF/s (6) | 4.92 ms, 2.64 | 2.08 ms, 6.24 |
| stage 1 k7 d3 | 17.10 ms, 1.77 (6) | 11.33 ms, 2.68 | 4.57 ms, 6.64 |
| stage 1 k11 d5 | 24.57 ms, 1.94 (6) | 17.73 ms, 2.69 | 7.06 ms, 6.75 |
| stage 0 k3 d1 | 4.37 ms, 1.98 (6) | 3.29 ms, 2.63 | 1.36 ms, 6.39 |
| stage 0 k7 d3 | 9.70 ms, 2.08 (6) | 7.60 ms, 2.66 | 3.41 ms, 5.92 |
| stage 0 k11 d5 | 16.36 ms, 1.94 (4) | 11.92 ms, 2.66 | 5.09 ms, 6.24 |

Mac GPU references (same day): MLX `conv1d` 6.4–6.5 TF/s; MLX's steel GEMM through lokinfer's runtime 8.7 TF/s;
PyTorch MPS conv1d 4.75–5.2 TF/s (LPM). Pool vs raw threads (Mac, `lokinfer-bench-conv tune`, partial run): the
runtime pool with dynamic tiles beat raw threads at every tile, e.g. s1 k3 tile 768: pool 2.26 TF/s vs ≤ 1.37 raw.

### 3.2 Kokoro 1-D, fused direct CPU conv

| measurement | result | device / state | source |
|---|---|---|---|
| tile sweep: best tile puts `[Ci·K, tile]` at ≈ 3 MB (6 of 7 shapes); 256 k11 tile 768 vs 256 | 0.97 vs 2.42 TF/s | M2 Max | m2-conv-proof.md "Fused-conv tile sweep" |
| fused resblock1 iteration vs basic kernel chain, stage 1 k3 / k7 / k11, stage 0 k11 | 4.20× / 3.73× / 3.47× / 2.56× | M2 Max | m2-conv-proof.md |
| fused `adain_resblk` (decoder, predictor F0/N) | −22 ms at rand510 (interleaved A/B) | not stated | [todo.md](../todo.md) M3 |
| generator snake convs in the app59f profile | 96 of 150 ms at 1.81 TF/s | M2 Max | todo.md M6 |
| generator snake convs in the app59f profile | 125 of 164 ms at 1.40 TF/s | iPhone | todo.md M6, [iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md) |
| stage-1 conv Ci = Co = 128, L = 17 641, K 11, d 5: baseline 6 streams / no GEMM / no snake / no im2col | 3.96 / 0.90 / 3.48 / 3.03 ms | iPhone | iphone-cpu-sme2.md |
| Kokoro E2E, fused CPU graph: real28 / real117 / rand510 | 103 / 344 / 920 ms (torch.compile 237 / 705 / 2129) | M2 Max, LPM off, battery | todo.md M3 |

### 3.3 SME2 and CPU Winograd (iPhone; Mac via Accelerate)

Source: [iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md) (2026-11-01/02).

| measurement | result |
|---|---|
| same stage-1 conv, custom SME2 tile kernel (no im2col): 6 / 2 / 1 streams | 3.39 / 3.72 / 4.22 ms |
| SME2 FMOPA-only, 1 stream, no loads, no prologue | 3.14 ms (≈ 2.0 TF/s = the P-cluster SME unit) |
| tile GEMM on a P core (both P cores in SME) vs on an E core | ~190 µs vs ~1.75 ms (~9×) |
| P/E pipeline vs no split: stage-1 K 11 d 1 / K 3 / K 7 d 3; rand510 K 11 d 5 | 2.20 → 1.73 / 1.12 → 0.90 / 2.01 → 1.47 ms; 18.3 → 14.0 ms |
| two-phase short-sequence mode, stage 0 at app59f (Ci 256, L 2940, K 11) | 1.36 ms vs 1.51 pipelined, 2.40 direct |
| iPhone CPU app59f E2E (nominal) | 153 → 96.8 ms (ExecuTorch 288.9) |
| iPhone CPU dashboard after the change: s20f / s28f / rand510 | 718.4 (was 1169.8) / 951.8 (was ~1546) / 1001.8 (was 1230) ms |
| Mac CPU (Winograd on Accelerate, no SME2): app59f / s9f | 152.0 → 122.3 / 291.7 → 256.9 ms |

### 3.4 GPU steel conv in Kokoro

| measurement | result | source |
|---|---|---|
| Kokoro GPU convs at the steel f32 ceiling (prof-lokinfer FLOP count) | iPhone 2.62 TF/s (real117), Mac 6.3 TF/s (rand510) | todo.md M4 |
| same, earlier statement | ≈ 6 TF/s in LPM | todo.md M4 |
| Metal System Trace, real117, Mac: `lk_conv1d_lc_f32` share / occupancy | ~90 % of sampled shader time; ALU limiter 67 %, F32 util 53 %, occupancy 22 % | [profiling.md](../profiling.md) |
| steel share of GPU time before Winograd | ~62 % (Mac), ~74 % (iPhone) | winograd-conv.md |

### 3.5 GPU Winograd 1-D

Source: [winograd-conv.md](../impl-notes/winograd-conv.md). Per-conv speedup of the fused-snake Winograd conv over the
snake + steel pair (chains of 6 convs), d = 1 (d = 3 / 5 within ±10 %):

| shape (L, C) | M2 Max K3 / K7 / K11 | iPhone K3 / K7 / K11 |
|---|---|---|
| 44161, 128 | 1.04× / 1.29× / 1.34× | 1.24× / 1.47× / 1.54× |
| 7360, 256 | 1.33× / 1.38× / 1.41× | 1.52× / 1.52× / 1.64× |
| 19321, 128 | 1.07× / 1.34× / 1.24× | 1.39× / 1.34× / 1.60× |
| 3220, 256 | 1.07× / 1.26× / 1.31× | 1.32× / 1.39× / 1.49× |
| 6601, 128 | 0.91× / 1.07× / 1.35× | 1.23× / 1.44× / 1.48× |
| 1100, 256 | 1.00× / 1.04× / 1.21× | 0.87× / 1.43× / 1.51× |

| measurement | result |
|---|---|
| E2E Mac GPU, steel → Winograd: word / app59f / real117 / rand510 | 29.1 → 26.8 / 55.6 → 52.5 / 115.7 → 101.2 / 330.8 → 288.8 ms |
| rand510 Mac GPU conv time (snake + convs) | 235.7 → 179.4 ms (mm 154.8 + transform 24.6) |
| E2E iPhone GPU: app59f / real117 / rand510 | 98.6 → 70.0 / 229.5 → 159.6 / 691.7 → 487.8 ms |
| 16-tile threadgroups for small grids (L = 1100, Mac) | 5–15 % faster, neutral above |
| stats in the GEMM epilogue: per-kernel cost, Mac | K 11 −8 %, K 3 +3 %, K 7 +8–19 % (hence K 7 keeps the stats pass) |
| stats in the epilogue: rand510 Mac stats pass | 10.9 → 2.2 ms; E2E app59f 53.2 → 52.2, rand510 ~297 → 290.7 ms |
| decoder / F0-N K 3 on Winograd: GPU time rand510 Mac | 16.4 → 12.2 ms (E2E unchanged on the Mac: overlaps the CPU) |
| decoder Winograd, iPhone: app59f / s9f | 67.0 → 65.9 / 146.1 → 142.0 ms |
| GPU counters, L = 19321 K 11, Mac: Winograd GEMM vs steel | occupancy 8.4 % vs 16.7 %, F32 util 35 % vs 58 % |
| accuracy on 48 real Kokoro convs: rel-RMS median (max) | steel 5.7e-7 (9.4e-7), Winograd 6.2e-7 (1.0e-6); E2E SNR identical to steel within 0.1 dB on 14 cases |

### 3.6 Vision convs (2-D)

| measurement | result | device | source |
|---|---|---|---|
| specialized 3×3 depthwise vs runtime-tap version, single core | 15 → 60 GFLOP/s | M2 Max | [mobilenet_v2.md](../models/mobilenet_v2.md), conv2d.cpp comment |
| register-blocked NEON depthwise: MobileNetV2 CPU | Mac 6.4 → 4.1 ms; iPhone 4.2 → 3.8 ms; 112×112×32 dw layer 0.61 → 0.12 ms (listed with the iPhone figures) | M2 Max / iPhone | todo.md M6 |
| fused `mbv2_block` vs three kernels: N=1 / N=8 | Mac 1.62 → 1.53 / 8.4 → 6.9 ms; iPhone N=8 10.1 → 9.0 ms (N=1 within noise) | both | mobilenet_v2.md |
| ResNet-18 3×3×512×512 layer, 2-way K split | 430 → 345 µs (6-way: 365 µs) | iPhone | conv2d.cpp comment, [resnet18.md](../models/resnet18.md) |
| im2col whole-row copy, 7×7 stem | 500 → 340 µs | M2 Max | resnet18.md |
| CPU convnet round (tiling, K split, dw, mbv2 fusion, pool hand-off): MobileNetV2 N=1 / N=8 | iPhone 3.54 → 1.61–1.69 / 20.1 → 8.5 ms (ExecuTorch 1.49 / 7.92); Mac 4.03 → 1.52 / 18.7 → 6.85 ms | both | todo.md M6 |
| same round, ResNet-18 N=1 / N=8 | iPhone 6.58 → 4.16 / 36.4 → 23.4 ms (ExecuTorch 2.93 / 18.67); Mac 6.41 → 4.32 / 25.1 → 20.1 ms | both | todo.md M6 |
| same round, DCGAN D N=64 | iPhone 10.0 ms (ExecuTorch 10.66) | iPhone | todo.md M6 |
| Accelerate throughput on ResNet-18's layers | 1.0–1.2 TF/s (56×56 / 28×28), ~0.7 TF/s (512-ch layers) | iPhone | todo.md M6 |
| DCGAN G convT gather parallel by pixels + `vvtanhf`, CPU N=64 | 19.7 → 11–12.6 ms | not stated | todo.md M6 |
| GPU steel 32×32 variant for small grids, ResNet-18 N=1 | 3.8 → 2.4 ms | not stated | todo.md M4 |
| GPU depthwise `k3s{1,2}c4`: depthwise N=8 | 2.4 → 0.55 ms (4.4× the per-element kernel) | M2 Max | todo.md M6, [profiling.md](../profiling.md) |
| GPU MobileNetV2 N=1 / N=8 after dw + NHWC changes | Mac 1.42 → 1.10 / 5.30 → 3.15 ms (MLX 1.76 / 4.50); iPhone 1.61 → 1.04 / 11.05 → 6.19 ms | both | todo.md M6 |
| GPU stem im2col tiled: N=8 / N=1 | 1.22 → 0.37 / 0.157 → 0.048 ms; ResNet-18 N=1 2.30 → 2.19, N=8 7.73 → 7.06 ms | M2 Max | todo.md M6, resnet18.md |
| DCGAN D GPU first conv as im2col (K 48) + 1×1 steel instead of padding to 16 ch (K 256), N=64 | Mac 3.07 → 2.84 ms; iPhone 6.64 → 6.01 ms | both | todo.md M6 |
| GPU 2-D Winograd vs steel, 3×3 s1 + ReLU, N=8: 56²×64 / 28²×128 / 14²×256 / 7²×512 | 1.04× / 0.93× / 1.36× / 0.78× | M2 Max | winograd-conv.md |
| same, N=1 | 0.85× / 0.71× / 0.62× / 0.49× | M2 Max | winograd-conv.md |

Not measured as standalone kernels: `conv_transpose1d_f32_cpu`, `conv_transpose1d_dw_f32_cpu`, `lk_convt_gather_lc`,
`lk_dwconvt_lc`, `lk_convt2d_gather_nhwc`, the generic depthwise paths. CPU `conv_post` (`conv1d_f32_cpu`, Co = 22):
0.42 TF/s, 1.7 ms at app59f (iphone-cpu-sme2.md "Open"; device not stated there).

## 4. Tried and rejected

| idea | result | source |
|---|---|---|
| one big im2col + Accelerate-threaded sgemm (CPU 1-D) | 0.49–0.93 TF/s; kept only as the basic kernel | m2-conv-proof.md |
| shifted GEMMs (one sgemm per tap), alone or over 8 streams | 0.73–1.69 TF/s, per-tap GEMMs too thin | m2-conv-proof.md |
| CPU time tiles ≥ 1024 (2048: overflows L2) | 2048: 0.3 TF/s; 1024–2048 never helped in the fused sweep | m2-conv-proof.md, `ops.conv_tile` |
| raw threads with static strides instead of the pool | slower at every tile size | m2-conv-proof.md |
| hand-written GPU implicit-GEMM conv (v1, v2, BK 8/16/32, padded strides, 128-thread cap, no-threadgroup "direct") | 0.4–1.1 TF/s vs steel 6.2–6.75 → vendored steel | m2-conv-proof.md |
| steel conv tiles 64×64×32 / 128×64 / 64×128 (Kokoro rand510 Mac E2E) | 390 / 376 / 378 ms vs 361 for 64×64×16 | todo.md M4, kokoro_lc.cpp comment |
| small-grid threshold 32 for the steel 32×32 variant (iPhone) | no gain on MNIST / ResNet-18 / MobileNetV2 | kokoro_lc.cpp comment |
| AdaIN + Snake fused into the steel conv's A loader | not built: the loader reloads each element 2·K times, so `sin` would be recomputed 6–22× (≈ +20 % ALU) vs ≈ 0.3 ms for a separate pass; superseded by the Winograd input transform | todo.md M4 |
| GPU Winograd, transform fused into the GEMM loader | 2.3–3 TF/s-eq (spills), ~4.2 after the float2 rewrite, still < steel | winograd-conv.md |
| GPU Winograd, all A points' accumulators live (transform once per channel block) | 8 % occupancy, ~4.6–5.1 TF/s-eq | winograd-conv.md |
| GPU Winograd tiles 64×32, BK 32, 32×32, 16×64 for all grids; B straight from device memory; 4 tiles per transform thread | all slower (transform with RT = 4: 2× slower) | winograd-conv.md, winograd.cpp comment |
| GPU Winograd GEMM register prefetch of the next block | −10 to −28 % | winograd-conv.md |
| Winograd schemes with ≥ 10 points (F(4,7), nested F(5,6), F(6,6), F(6,7)) | ~10× the direct conv's f32 error; not built | winograd-conv.md |
| fusing the Winograd input transform into the previous conv's epilogue | impossible: the activation needs the AdaIN affine of the whole previous output | winograd-conv.md |
| per-length K = 3 choice (steel vs Winograd) | dropped: F is a runtime dim, losses only on word-length K = 3 convs (Mac −6 %, iPhone −13 % at L = 1100) | winograd-conv.md |
| CPU Winograd on the F0/N predictor's K = 3 convs | −1 dB audio SNR (app47m 37.8 → 36.9 vs 37.5 limit) → decoder only (`LOKINFER_KOKORO_WINO_CPU_DEC=2`) | iphone-cpu-sme2.md |
| direct SME2 conv with per-call weight packing | stage 1 3.39 vs 3.96 ms, but the decoder conv 3× slower (12 MB packed per call); superseded by Winograd with pre-packed U | iphone-cpu-sme2.md |
| CPU Winograd tiles 1536–2048; ring depths 2–4 | slower (depth 4 slower than no split; 8–24 within 5 %) | iphone-cpu-sme2.md, kokoro_conv_fused.cpp |
| research recommendation "skip Winograd on CPU" ([research-kernels.md](../impl-notes/research-kernels.md) §1.7) | superseded by measurement: CPU Winograd built (§3.3) | iphone-cpu-sme2.md |
| conv2d tasks per thread 1 / 8 (vs 3) | within noise on the M2 Max; 3 best on MobileNetV2 N=8 | conv2d.cpp comment |
| serial pointwise conv2d up to 2e7 / 6e7 MACs | MobileNetV2 N=1 2.2 → 3.3 / 3.6 ms (M2 Max) | conv2d.cpp comment |
| whole-conv Accelerate threading for all conv2d; task caps 2 / 6; 6-way K split; pre-transposed `[Co, K]` weights for the 512-ch layers (iPhone 320 → 310 µs per layer, E2E within noise) | reverted / not kept | todo.md M6, conv2d.cpp comment |
| 1 M multiply-add inline threshold for depthwise; longer pool spin (250 µs) | slower on the iPhone | todo.md M6, conv2d.cpp comment |
| GPU depthwise with runtime stride (register arrays indexed at runtime) | spilled, slower than the scalar kernel | profiling.md, vision.metal comment |
| GPU per-element NCHW → NHWC pad (MobileNetV2 N=8 stem) / per-element im2col (ResNet-18 N=8 stem, ~50 GB/s) | 0.26 ms / 1.22 ms; replaced by the per-pixel / tiled kernels | vision.metal comments |
| GPU 2-D Winograd F(2×2, 3×3) in ResNet-18 | wins only on 14²×256 at N=8 (~5 % of 6.9 ms), N=1 regresses → not wired | winograd-conv.md |

## 5. Open

From [todo.md](../todo.md) and the notes; nothing here is built unless stated.
- **GPU Winograd F(2,3) for K = 3** (2 output rows, steel-like register use, 0.67× multiplies), and why K = 7 slows
  8–19 % with fused stats (todo.md M4; winograd-conv.md: the M = 4 accumulators bound occupancy).
- **ResNet 2-D Winograd**: split the 16 points across threadgroups for small grids, runtime per-shape choice,
  F(4×4, 3×3) for the 56² layer (accuracy to be measured); integrate only if N=1 doesn't regress.
- **Polyphase ConvTranspose1d** (ups[0] k20 s10, ups[1] k12 s6) with a leaky prologue and `+ x_source` epilogue on
  the CPU (todo.md M3 `[~]`); today: sgemm + col2im (CPU), MPS GEMM + gather (GPU).
- Remaining fused CPU-conv input transforms: gather by `idx`, nearest ×2, depthwise convT, virtual concat, 3-input
  mean (todo.md M3); implicit alignment gather in the decoder conv prologue.
- CPU Winograd: decoder K = 3 at L = F (147) uses only 37 Winograd columns → 2 SME blocks of 32 (42 % waste); a 64×16
  tail block would recover ~0.1 ms per conv. `conv_post` and the convTs on the CPU are not optimized
  (iphone-cpu-sme2.md).
- iPhone CPU ResNet-18 gap = GEMM throughput (Accelerate ~0.7–1.2 TF/s vs ExecuTorch ~1.24): a custom SME2 kernel with
  pre-packed weights for conv2d (todo.md M6).
- MobileNetV2: direct NEON stem conv (~100 µs), single-threaded fused block for 7×7 maps; the remaining iPhone CPU gap
  is spread over ~35 kernels of 50–100 µs — profile per-kernel time vs FLOPs before changing anything (todo.md M6).
- M2 sweep leftovers: CPU tile × stream sweep on the iPhone, BNNS conv as a library candidate (todo.md M2 `[~]`).
- iPhone GPU: f32 Metal 4 tensor-op (`matmul2d`) experiment for conv / GEMM (todo.md M2 / M5).
- `prof-lokinfer`'s TF/s column counts Winograd ops by their U size and overstates them ~2× (iphone-cpu-sme2.md).

## 6. Tests and benchmarks

- Tests: `tests/kernels/test_kokoro_kernels.py` (basic 1-D), `test_kokoro_conv_fused.py` (fused direct + CPU
  Winograd), `test_kokoro_gpu.py` (steel 1-D, convT gather, dw-convT), `test_winograd_gpu.py` (1-D and 2-D Winograd),
  `test_dwconv_gpu.py`, `tests/unit/test_sme2_conv.cpp` (runs on the phone in the deploy suite); model tests in
  `tests/models/` (vision models vs torchvision / PyTorch < 1e-4).
- `bench/kernels/conv1d_cpu.cpp` (Google Benchmark, M2 proof CPU candidates → `results_conv1d_cpu.json`),
  `conv1d_gpu.{cpp,metal}` (hand-written GPU implicit GEMM, M2), `conv1d_steel.metal` (vendored steel driven as a 1-D
  conv), `conv_bench.{cpp,h}` (the same shapes through the runtime on Mac and iPhone, `lokinfer-bench-conv`,
  `LokiHost --bench`), `bench_conv_fused.py` (fused vs basic AdaINResBlock1 iteration; `--sweep` = the tile sweep).
