# GEMM and GEMV kernels

Kernel docs: [convolution](convolution.md) · **gemm** · [elementwise](elementwise.md) · [reductions](reductions.md) ·
[attention](attention.md)

Every matrix multiply in lokinfer: CPU (hand-written NEON, Accelerate with lokinfer's blocking and fused epilogues,
the LSTM's per-step GEMV) and GPU (hand-written simdgroup-matrix tiles, steel GEMM with fused epilogues, split-K, MPS
on lokinfer's encoder, decode GEMVs). All f32, row-major. Conv-as-GEMM (im2col, implicit GEMM, Winograd, ConvTranspose
as GEMM + gather) is in [convolution.md](convolution.md); attention's internal GEMMs are in
[attention.md](attention.md).

Devices: **Mac** = M2 Max (30-core GPU), **iPhone** = iPhone 18 Pro Max ([baselines.md](../baselines.md)). LPM =
Low Power Mode on (the early research numbers; re-measure before relying on them).

Sources: `kernels/cpu/gemm.cpp`, `kernels/cpu/decode.cpp`, `kernels/cpu/kokoro_lstm.cpp`, `kernels/cpu/sme2_conv.cpp`
(`gemvT`), `kernels/cpu/sme2_gemm.{h,cpp}` (shared SME2 GEMM engine), `kernels/cpu/kokoro_common.h` (`detail::sgemm`, `parallelForSt`), `kernels/gpu/gemm.metal`,
`kernels/gpu/gemm_steel.metal`, `kernels/gpu/kokoro_tok.metal` (GEMVs), `kernels/gpu/vision.metal` (`lk_rowdot`),
`kernels/gpu/mps.mm`, `kernels/gpu/dispatch.cpp`, `kernels/gpu/kokoro_lc.cpp` (steel / bias / GEMV dispatch),
`kernels/gpu/transformer_gpu.cpp`. Python defs: `python/lokinfer/ops/{gemm,transformer,kokoro,kokoro_gpu,vision}.py`.

## 1. Overview

Layouts: `a [M, K]`, `b [K, N]` unless marked NT (`b [N, K]`, read transposed: GPT-2's tied LM head reads `wte`
directly, no transposed copy). Epilogues are applied before the store (GPU) or per output block while it is in
cache (CPU).

### CPU

| registered name | layout | prologue / epilogue | used by | selection |
|---|---|---|---|---|
| `gemm_f32_cpu`, `gemm_bias_f32_cpu`, `gemm_bias_relu_f32_cpu` | NN | bias, bias+ReLU (in registers) | MNIST `cpu` variant | explicit variant; NEON 4×16 tiles, single-threaded |
| `gemm_f32_cpu_accelerate` | NN | – | DCGAN G (convT GEMM) | iPhone, M ≥ 16, N ≥ 512, ≥ 50 MFLOP: 2 row blocks; else `sgemm()` (below) |
| `gemm_bias_f32_cpu_accelerate`, `gemm_bias_relu_f32_cpu_accelerate` | NN | bias / bias+ReLU as a separate NEON pass (iPhone row blocks: per block) | Kokoro (ALBERT fused QKV + all Linears, style bank GEMV), MNIST, ResNet-18 / MobileNetV2 FC, DCGAN, GPT-2 `cpu="basic"`, `build_generate` CPU | same as above |
| `gemm_bias_act_f32_cpu_accelerate`, `gemm_split_f32_cpu_accelerate` (Python: `gemm_accel_f32_cpu`) | NN | bias, gelu_tanh (NEON exp, or `vvtanhf` = `act="gelu_vv"`), ReLU, fused per block | GPT-2 CPU prefill (`cpu="fused"`, default), GPT-2 CPU decode step | `split` attr: 0 = `autoSplit` (below), 6000 = one multithreaded call, > 0 column blocks, < 0 row blocks, 5000 + t = K split over t tasks |
| `gemm_nt_f32_cpu_accelerate` | NT | – | GPT-2 LM head (all CPU builds) | M ≤ 8 and N·K·4 ≥ 4 MB: row blocks of B on the pool; else one `CblasTrans` call |
| `lstm_bidir_f32_cpu`, `lstm_bidir_t_f32_cpu` | x `[T, I]`, w_ih `[8H, I]`, w_hh `[8H, H]` or w_hhᵀ `[H, 8H]` | input projection + bias (b_ih + b_hh folded) as one GEMM; per step a GEMV + fused gates | Kokoro's 6 BiLSTMs (both builds) | `_t` only with `LOKINFER_KOKORO_LSTM_T=1` (off by default, see §4) |

CPU blocking rules (`gemm.cpp`):

- `sgemm()` (the `*_accelerate` NN kernels): **M ≤ 8, N·K·4 ≥ 4 MB, N ≥ 256** → `min(4·threads, N/64)` column blocks,
  each a single-threaded Accelerate call on a pool task (`splitN`); otherwise `ThreadPool::park()` then one
  multithreaded `cblas_sgemm`.
- `autoSplit()` (the `gemm_accel_f32_cpu` kernels, split = 0):
  - M ≤ 8 and N·K·4 ≥ 1 MB → **K split** into `3·threads` (Mac) / `2·threads` (iPhone) contiguous row ranges of B;
    partial `[M, N]` sums reduced by 256-element column chunks on the pool, with the epilogue;
  - iPhone, M > 8: `iphoneRowSplit` (M ≥ 16, N ≥ 512, 2·M·N·K ≥ 5e7) → **4 row blocks**, else 0;
  - else 0: `park()`, one multithreaded call, then the epilogue — on the pool only for gelu with M·N ≥ 65 536.
- `LOKINFER_GEMM_SPLIT=<split>` (experiment knob) routes the generic `*_accelerate` kernels through the same blocked
  path with that split, to A/B blockings in a model.
- Inside other CPU kernels (attention, LSTM input projection, convs), `detail::sgemm` runs GEMMs below 50 MFLOP
  single-threaded and parks the pool before multithreaded ones; `detail::parallelForSt` gives every pool task
  single-threaded BLAS (one Accelerate stream per core).

### GPU

| registered name | Metal function(s) | layout | epilogue | used by | selection |
|---|---|---|---|---|---|
| `gemm_f32_gpu` | `gemm_f32` (`gemm.metal`) | NN | – | runtime unit tests and the basic E2E / lit graphs | explicit; 32×32 tiles, 4 simdgroups |
| `gemm_bias_f32_gpu`, `gemm_bias_relu_f32_gpu` | `gemm_bias[_relu]_smallm_f32`, `lk_gemm_f32_{64x64,32x32}`, `gemm_bias[_relu]_f32` | NN | bias, bias+ReLU | MNIST GPU, ResNet-18 / MobileNetV2 FC, DCGAN G layer 1 | M ≤ 8 → split-K small-M kernel; Mac M > 8 → steel; iPhone 8 < M ≤ 256 → steel, M > 256 → 32×32 tiled |
| `gemm_steel_f32_gpu`, `gemm_steel_bias_f32_gpu`, `gemm_steel_bias_gelu_f32_gpu`, `gemm_steel_bias_relu_f32_gpu`, `gemm_steel_bias_res_f32_gpu` | `lk_gemm_f32_{64x64,32x32}` (`gemm_steel.metal`) | NN | bias, gelu_tanh, ReLU, + residual | ResNet-18 GPU stem (bias+ReLU); GPT-2 `gemm="steel"` | 32×32 tiles when the 64×64 grid has < 128 threadgroups (`kSmallGrid`), else 64×64 |
| `gemm_nt_steel_f32_gpu` | `lk_gemm_nt_f32_{64x64,32x32}` | NT | – | GPT-2 `gemm="steel"` LM head | same |
| `gemm_splitk_f32_gpu` | `lk_gemm_splitk_f32_{64x64,32x32}` | NN → partials `[S, M, N]` | none (consumer sums) | GPT-2 GPU prefill attn / MLP output projections (`build(sk=(2, 4), sk_tile=32)`) | Mac: split when ⌈M/64⌉·⌈N/64⌉ < 48, else MPS into plane 0; iPhone: always MPS |
| `add_bias_layernorm2_sk_f32_gpu` | `lk_add_bias_ln2_sk` | parts `[S, R, C]` | sums S partials + bias + residual, LayerNorm | consumer of the split-K partials | same tile rule as the producer ([reductions.md](reductions.md)) |
| `gemm_f32_gpu_mps` | MPS `MPSNDArrayMatrixMultiplication` | NN | – | GPT-2 GPU prefill (default `gemm="mps"`), `build_generate` T > 1, Kokoro ALBERT GPU (4 GEMMs/layer), Kokoro GPU ConvTranspose GEMMs, DCGAN G GPU | explicit |
| `gemm_nt_f32_gpu_mps` | same, B through a transposed descriptor | NT | – | GPT-2 GPU LM head (prefill, decode, generate) | explicit |
| `gemm_auto_f32_gpu`, `gemm_auto_bias_gelu_f32_gpu` | delegates to `gemv4_*` or MPS (+ `bias_gelu_f32_gpu`) | NN | bias + gelu (fused in `gemv4`, separate pass after MPS) | GPT-2 `build_generate` GPU (`kernels="auto"`) | per call from runtime M: M = 1 → `gemv4` (auto cw), else MPS |
| `gemv4_f32_gpu`, `gemv4_bias_f32_gpu`, `gemv4_bias_gelu_f32_gpu`, `gemv4_bias_res_f32_gpu` | `lk_gemv4_c{2,4,8,16}` (`kokoro_tok.metal`) | x `[1, K]`, W `[K, N]`, N % 4 = 0 | bias, gelu_tanh, + residual | GPT-2 GPU decode (default `gemm="gemv4"`) | `cw` attr; 0 = auto: Mac 4, iPhone 8 |
| `gemv_f32_gpu`, `gemv_bias_f32_gpu`, `gemv_bias_gelu_f32_gpu` | `lk_gemv` | x `[1, K]`, W `[K, N]` | bias, gelu_tanh (residual slot unused by registrations) | GPT-2 decode `gemm="gemv"` (slower, kept for A/B) | explicit |
| `gemv_rows_f32_gpu` | `lk_gemv_rows` | x `[1, K]`, W `[N, K]` | – | GPT-2 decode `gemm="gemv"` LM head | explicit |
| `gemv1_bias_f32_gpu`, `gemv1_bias_sigmoid_f32_gpu` | `lk_rowdot` (`vision.metal`) | x `[R, K]`, w `[K, 1]` | + b, sigmoid | DCGAN D last layer (sigmoid), Kokoro GPU F0 / N projections | explicit |

Model-level selection: GPT-2 prefill defaults to MPS (`build(gemm="mps")`), decode to `gemv4` with the LM head on MPS
(`build_decode(gemm="gemv4")`), generation to the per-call `gemm_auto_*` ops ([gpt2.md](../models/gpt2.md)).
Kokoro: CPU GEMMs are `gemm_bias_f32_cpu_accelerate`; the GPU build runs ALBERT's GEMMs on MPS
([kokoro.md](../models/kokoro.md)).

## 2. How each works

### CPU NEON GEMM (`gemm_f32_cpu` family)

Register-blocked micro-kernel: a 4×16 output tile (16 `float32x4_t` accumulators) streams one row of B (4 vectors) per
k and broadcasts one A element per row; the bias / ReLU epilogue is applied on the accumulators. Row tails use the
same tile with R = 1..3 rows (so small-M / GEMV shapes stay vectorized); column tails (N % 16) fall back to 4-wide
vectors then scalars. Single-threaded, no packing. It exists as the hand-written baseline: for M ≥ ~8 Accelerate
(AMX) is 10–30× faster than the best single-core NEON GEMM on the M2 (LPM,
[research-kernels.md §1.1](../impl-notes/research-kernels.md)).

### Accelerate GEMMs with lokinfer blocking (`*_cpu_accelerate`, `gemm_accel_f32_cpu`)

The math is `cblas_sgemm`; lokinfer owns the blocking, the threading and the epilogue:

- **One multithreaded call** (Mac default for M > 8): Accelerate spreads it over both P-clusters' AMX units. The pool
  is `park()`ed first: workers still spinning from the previous `parallelFor` otherwise compete with Accelerate's
  threads. The epilogue (`epilogueBlock`: bias, gelu_tanh via NEON `exp`, or via `vvtanhf`, ReLU) is a pass over the
  output while it is in cache, split over the pool only when it is heavy (gelu, M·N ≥ 65 536).
- **Row blocks** (iPhone, M ≥ 16, N ≥ 512, ≥ 50 MFLOP): each block is a single-threaded Accelerate stream on a pool
  task with its own epilogue. `gemm_accel_f32_cpu` (GPT-2) uses 4 blocks, the generic `*_accelerate` kernels (Kokoro's
  ALBERT, weights shared across layers and cache-hot) use 2.
- **Column blocks** (small M on big weights in the generic kernels; `split > 0` in `gemm_accel_f32_cpu`): blocks of
  ≥ 16 columns, single-threaded Accelerate each, so every core streams its share of B.
- **K split** (`split = 5000 + t`, the small-M auto choice of `gemm_accel_f32_cpu`): each task multiplies a contiguous
  range of B's rows (streams whole rows, unlike column blocks); the partial `[M, N]` vectors land in scratch and are
  summed by 256-element chunks on the pool, which also apply the epilogue.
- **NT** (`gemm_nt_f32_cpu_accelerate`): `CblasTrans`; for GEMV-like calls (M ≤ 8, ≥ 4 MB, the decode LM head over
  50 257 vocab rows) B's row blocks are contiguous, one block per pool task.

The Kokoro **style bank** is a GEMV through this path: every `<root>.*.fc` style projection (AdaIN1d /
AdaLayerNorm) of a model root is concatenated into one `[128, total]` weight + bias (`W.style_bank` in
`models/kokoro/model.py`; decoder 37 396 + predictor 12 288 outputs), multiplied once per style vector with
`gemm_bias_f32_cpu_accelerate` (M = 1, so the column-block `splitN` path), and each AdaIN reads a zero-copy view
of its slice. The GPU build copies the CPU bank to the GPU once per bank (`style_vec_gpu`).

The ALBERT **fused QKV** GEMM (CPU and GPU): `[T, 768] × [768, 2304]` with `[Wqᵀ | Wkᵀ | Wvᵀ]` built once
(`_qkv_w`); attention reads q / k / v from the fused output ([attention.md](attention.md)). GPT-2's `c_attn` is
already fused in the checkpoint.

### LSTM: the recurrent GEMV (`lstm_bidir[_t]_f32_cpu`)

One GEMM projects every timestep of both directions: `gx[T, 8H] = x[T, I] · w_ihᵀ + b` (`detail::sgemm`, the
`b_ih + b_hh` sum folded offline). Then each direction runs its recurrence on its own pool task (two concurrent
single-threaded streams, each keeping its 4H×H `w_hh` block in its core's cache; sequential when T < 4). Per step:
`gates[0:4H] = gx[t] + w_hh · h` (`cblas_sgemv`, beta = 1), then sigmoid (vvexpf + NEON divide) on i/f/o, tanh
(`vvtanhf`) on g and c, `c = f·c + i·g`, `h = o·tanh(c)` written straight into the output row. The `_t` variant
stores `w_hhᵀ [H, 8H]`: on SME2 (A19) with 4H % 64 = 0 and 4H ≤ 1024 the step GEMV is `sme2::gemvT` (all 4H gates
accumulate in ZA, one multi-vector FMLA per 64 gates and hidden unit); elsewhere `cblas_sgemv` on the transposed
matrix. LSTMs run on the CPU in both builds (decided by measurement, §3).

### GPU 32×32 tiled kernel (`gemm.metal`: `gemm_f32`, `gemm_bias[_relu]_f32`)

One threadgroup (128 threads = 4 simdgroups) per 32×32 output tile; each simdgroup owns a 16×16 block as 2×2
`simdgroup_float8x8` accumulators. A 32×16 slab of A and 16×32 of B are staged through threadgroup memory with zero
fill at the edges (BK = 16); results go through threadgroup memory for the bounds-checked store and epilogue.

### GPU split-K small-M kernel (`gemm_bias[_relu]_smallm_f32`)

For M ≤ 8 a 32-column tile grid gives ⌈N/32⌉ threadgroups that each walk all of K. Here a threadgroup (32 × 16
threads) owns 32 columns of an 8-row block and splits K into 16 strided slices: each thread accumulates all ≤ 8 rows
of one column over its slice (B reads coalesced across the simdgroup, A reads broadcast), then thread (x, y) sums
the 16 slices of row y through threadgroup memory and applies the epilogue.

### Steel GEMM (`gemm_steel.metal`: `lk_gemm`, `lk_gemm_splitk`)

Built on MLX's steel `BlockLoader` + `BlockMMA` (MIT, vendored in `third_party/mlx_steel`): the tile loop mirrors
steel's `GEMMKernel::run` (aligned fast path per tile, safe loads at the edges) without its function constants.
Configs: 64×64×16 and 32×32×16 tiles, 2×2 simdgroups (128 threads); `TB = true` instantiates the transposed-B
loader (`lk_gemm_nt_*`). Epilogue on the accumulators before the store: + bias, then gelu_tanh or ReLU, then
+ residual (`D = act(A·op(B) + bias) [+ R]`). The dispatcher (`runGemmSteel`) picks 32×32 tiles when the 64×64
grid would have fewer than 128 threadgroups; `swizzle_log` is 0.

**Split-K** (`lk_gemm_splitk`): grid z = split s computes `A[:, s·kc:(s+1)·kc] · B[s·kc:(s+1)·kc, :]` into plane s
of `[S, M, N]` with no epilogue (kc % 16 = 0, checked by the Python def). `add_bias_layernorm2_sk_f32_gpu` sums the
S planes while it loads them, adds bias + residual and produces both the new residual stream and its LayerNorm, so
the reduction costs no extra pass. Both kernels share `splitKEffective` (the consumer only knows M and N): when the
64×64 tiles already fill the GPU (≥ 48 tiles on the Mac; always on the iPhone, `kSplitKMaxTiles = 0`) the producer
runs MPS into plane 0 and the consumer reads one plane.

### MPS on lokinfer's encoder (`mps.mm`)

`MPSNDArrayMatrixMultiplication` (α = 1, β = 0), prepared once per op, encoded with
`encodeWithMTL4CommandEncoder` into lokinfer's own Metal 4 compute encoder (no separate command buffer). Operands
are `MPSNDArray`s wrapping the existing buffers (`preferPackedRows`, no copy); NT uses a `[N, K]` descriptor with
dimensions 0 and 1 transposed. Metal 4 command buffers don't retain resources, so the arrays are kept alive by the
GPU context until the segment completes.

### Decode GEMVs (`kokoro_tok.metal`, `vision.metal`)

- `lk_gemv4_c{CW}` (`gemv4_*`): one 256-thread threadgroup per `4·CW` output columns; thread = (float4 column
  `tid % CW`, K slice `tid / CW`), so 256/CW K slices each read CW consecutive float4 per W row, unrolled 4 rows deep
  with 4 independent accumulators. Partials reduce by `simd_shuffle_xor` within a simdgroup, then through
  threadgroup memory; bias / gelu_tanh / residual fused. Every CW variant's pipeline is prepared; the run picks one.
- `lk_gemv` (`gemv_*`): one 256-thread threadgroup per 32 columns, lane = column (coalesced 128-byte reads of each W
  row), the 8 simdgroups split K, partials summed in threadgroup memory.
- `lk_gemv_rows` (`gemv_rows_f32_gpu`): W `[N, K]`, one simdgroup per row (8 rows per threadgroup), float4 dots,
  `simd_sum`.
- `lk_rowdot` (`gemv1_*`): a GEMV with one output column: one 256-thread threadgroup per row of x, SIMD reduction,
  `+ b[0]`, optional sigmoid.

### Runtime-dispatched GEMMs (`transformer_gpu.cpp`)

`build_generate` compiles one graph with symbolic T for both prefill and decode. `gemm_auto_f32_gpu` /
`gemm_auto_bias_gelu_f32_gpu` prepare the kernels they delegate to and choose per call from the runtime M: M = 1 →
`gemv4` (auto cw, fused bias + gelu), else MPS, followed for the gelu variant by `bias_gelu_f32_gpu` in place.

## 3. Measured performance

All numbers as recorded; where a source gives an end-to-end model time rather than a kernel time it says so.

### CPU

| what | device | case / shape | result | source |
|---|---|---|---|---|
| Accelerate `cblas_sgemm`, 1 thread | Mac (LPM) | M ≥ 64 | ~1.0–1.1 TF/s | [research-kernels.md §1.1](../impl-notes/research-kernels.md) |
| Accelerate, its own threading / 3 concurrent 1-thread streams | Mac (LPM) | – | ~2.0 / 3.4 TF/s aggregate | same |
| Accelerate default threads vs 1 thread vs NEON 8×12 (1T) | Mac (LPM) | 512×768×768 | 858–1985 / 973–1108 / 66 GF/s | same |
| same | Mac (LPM) | 64×768×768 | 1093 / 907 / 60 GF/s | same |
| same | Mac (LPM) | 200×1024×512 (LSTM W_ih over T) | 812 / 374 / – GF/s | same |
| same | Mac (LPM) | 1×768×768 (GEMV via sgemm) | 137 / 145 / – GF/s | same |
| `cblas_sgemv` | Mac (LPM) | 768×768 / 1024×256 / 50256×768 | 11.5 µs (206 GB/s) / 7.9 µs / 3.6–4.2 ms (37–43 GB/s) | [research-kernels.md §1.5](../impl-notes/research-kernels.md) |
| GPT-2 CPU GEMMs | iPhone | prefill T = 128, before row blocks | 20.5 ms, 1.06 TF/s; LM head 1.6 ms (154 MB at ~100 GB/s) | [gpt2.md](../models/gpt2.md) |
| single SME stream | iPhone | GPT-2 shapes | ~1.5–1.9 TF/s | [gpt2.md](../models/gpt2.md) |
| P-cluster SME unit | iPhone | FMOPA-only microbench | ≈ 2.0 TF/s | [iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md) |
| `gemm_accel_f32_cpu` 4 row blocks (e2e) | iPhone | GPT-2 prefill T = 128 / 512 | 26.7 → 23.0 ms / 80.1 → 77.4 ms | [gpt2.md](../models/gpt2.md), `gemm.cpp` |
| K-split decode GEMVs (e2e decode step) | Mac / iPhone | GPT-2 decode | 10.9 → 7.9 ms / 6.2 vs 6.0 ms (wash) | [gpt2.md](../models/gpt2.md), `gemm.cpp` |
| GPT-2 CPU dashboard (e2e) | Mac / iPhone | T = 128 / 512 / decode | 26.3 / 69.5 / 7.92 ms; 23.16 / 75.06 / 5.46 ms | [gpt2.md](../models/gpt2.md) |
| generic iPhone row blocks (2 vs 4 vs MT Accelerate) | iPhone | ALBERT-like encoder T = 59 / T = 141 | 8.9 / 9.8 / 11.0 ms; 18.4 / 20.7 / 18.7 ms | `gemm.cpp` (A/B 2026-11-01) |
| ALBERT QKV `[59,768]×[768,2304]` | Mac | in model vs alone | 224 µs vs 112 µs | `gemm.cpp` comment, [todo.md](../todo.md) M6 |
| ALBERT fused QKV vs 3 GEMMs | Mac | T ≈ 60 | ~112 µs vs 3 × 117 µs | `models/kokoro/model.py` |
| ALBERT GEMMs (prof) | Mac | Kokoro app59f | 0.6–0.93 TF/s | [todo.md](../todo.md) M6 |
| MNIST `cpu-accelerate` | iPhone | B = 1 / 64 / 1024 | 1.5 / 11.2 / 161 µs (ExecuTorch 1.7 / 12.1 / 173); ~1.4 TF/s at B = 1024 | [mnist-mlp.md](../models/mnist-mlp.md) |
| MNIST Accelerate vs PyTorch CPU | Mac (LPM) | B = 1 / 1024 | 0.004 vs 0.027 ms / 0.43 vs 0.77 ms | [mnist-mlp.md](../models/mnist-mlp.md) |
| style projections batched | Kokoro | – | 151 → 77 GEMM calls (time was small) | [todo.md](../todo.md) M3 |
| LSTM basic (sequential directions) | Mac | 130 / 512 / 1101 steps × 640 | 4.1 / 14.6 / 31.9 ms (torch 5.0 / 18.4 / 40.2); ~29–31.5 µs per step | [m2-conv-proof.md](../impl-notes/m2-conv-proof.md) |
| LSTM fused (directions concurrent, vectorized gates) | Mac (LPM) | Kokoro rand510, all LSTMs | 150 → 51 ms | [todo.md](../todo.md) M3 |
| LSTM step (CPU) | – | 1 MB w_hh from L2 + gates | ≈ 8–12 µs | [todo.md](../todo.md) M2 |
| `lstm_bidir_t` (SME2 / transposed sgemv) | iPhone / Mac | T = 59; T = 161 | 0.51 → 0.39 / 0.74 → 0.52 ms; 1.22 → 0.90 / 1.66 → 1.05 ms | [iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md) |

### GPU

| what | device | case / shape | result | source |
|---|---|---|---|---|
| MPS f32 | Mac (LPM) | 512×768×768 / 64×768×768 / 512×3072×768 / 2048³ | 5.7 / 0.46 / 2.4 / 8.0 TF/s | [research-kernels.md §2.1](../impl-notes/research-kernels.md) |
| steel GEMM through lokinfer's runtime | Mac | 128 × 132 096 × 1 408 | 5.45 ms, 8.7 TF/s (MLX matmul 7.3) | [m2-conv-proof.md](../impl-notes/m2-conv-proof.md) |
| old simple kernel (code comment) | Mac | – | ~1.5 TF/s | `gemm_steel.metal` header |
| steel vs 32×32 tiled (`gemm_bias_f32_gpu`) | Mac | M = 64..1024 | steel 1.2–1.9× faster | `kokoro_lc.cpp`, [todo.md](../todo.md) M6 |
| small-M kernel | Mac | MNIST 784→128 at M = 1 | 192 µs (32×32 tiles, 4 threadgroups) → small-M kernel | `gemm.metal`, [profiling.md](../profiling.md) |
| MNIST GPU (e2e) | Mac | B = 1 / 64 / 1024 | 0.205 / 0.257 / 0.152 ms (torch MPS 0.306 / 0.430 / 0.390) | [todo.md](../todo.md) M6 |
| MNIST GPU (e2e) | iPhone | B = 1 / 64 / 1024 | 0.27 → 0.13–0.14 ms; 0.37 → 0.09–0.28 ms (noisy); steel 0.49–0.52 vs tiled 0.43 ms at B = 1024 (MLX Swift 0.26) | `kokoro_lc.cpp`, [todo.md](../todo.md) M6 |
| MPS on GPT-2's N = 768 projections | Mac | T = 128 (24 tiles of 64×64) | 4–4.5 TF/s | [gpt2.md](../models/gpt2.md) |
| split-K N = 768 projections | Mac | GPT-2 T = 128, 24 GEMMs | 2.0 → 1.55 ms; e2e 5.45 → 4.95 ms (dashboard 4.99) | [gpt2.md](../models/gpt2.md), `transformer_gpu.cpp` |
| split-K vs MPS | iPhone | GPT-2 T = 128 (e2e) | 11.89 vs 11.43 ms (MPS kept) | same |
| MPS vs steel (e2e) | Mac | GPT-2 prefill T = 512 | 23.5 vs 26.1 ms | `models/gpt2.py` |
| `gemv4` vs MPS (e2e decode step) | Mac / iPhone | GPT-2 decode | 2.05 vs 2.7 ms / 5.6 vs 6.9 ms | `models/gpt2.py` |
| `gemv4` decode GEMVs | Mac | GPT-2 decode, all GEMVs | MPS 1.34 → 1.17 ms | [gpt2.md](../models/gpt2.md) |
| `gemv4` cw (e2e decode, P = 512) | Mac / iPhone | cw 4 vs 8/16, MPS / cw 8 vs 4, MPS | 2.2 vs 2.6, 2.8 ms / 5.60 vs 6.01, 6.91 ms | `transformer_gpu.cpp` |
| `lk_gemv` vs MPS (e2e decode) | Mac | GPT-2 decode | 5.2 vs 3.6 ms | `models/gpt2.py` |
| MPS LM head (`gemm_nt`) | Mac | GPT-2 decode | ~385 GB/s | [gpt2.md](../models/gpt2.md) |
| `gemm_auto_*` in `build_generate` (e2e decode) | Mac | P = 512 | 6.0 → 2.05 ms (= `build_decode` 2.07) | [gpt2.md](../models/gpt2.md) |
| naive GPU GEMV (research, not lokinfer's kernels) | Mac (LPM) | 768×768 / 1024×256 / 50256×768 | 22 / 15 / 437 µs (MPS 23 / 18 / 435) | [research-kernels.md §2.3](../impl-notes/research-kernels.md) |
| `gemv1_bias_sigmoid_f32_gpu` | – | DCGAN D last layer | 1.5 ms → µs (D GPU ≈ MLX) | [todo.md](../todo.md) M6 |

Per-kernel TF/s for any GEMM / GEMV in a model: `tools/prof-lokinfer` ([profiling.md](../profiling.md)).

## 4. Tried and rejected

CPU:
- **Hand-written NEON for M ≥ 8**: 10–30× slower than Accelerate on one core (LPM); kept only as the baseline variant
  ([research-kernels.md §1.1](../impl-notes/research-kernels.md)).
- **iPhone GPT-2 prefill blockings** (T = 128, against 4 row blocks at 23.0 ms): multithreaded Accelerate + pool
  epilogue 26.7; column blocks 2 / 6 / 12 / 24: 24.3 / 25.4 / 24.6 / 26.5; 2 row blocks 23.7, 3 row blocks 25.8 (T = 512:
  2 blocks 87.5 vs 77.4); one single-threaded stream 28.1; a pipelined GEMM (one thread streams column blocks, the
  others apply the epilogues) 30.0 / 26.5; a `QOS_CLASS_UTILITY` helper thread to reach the E-cluster's SME (no gain);
  prefetching the next GEMM's weights from the other cores 26.8; `vvtanhf` gelu instead of NEON exp (same speed, kept
  as `act="gelu_vv"`) ([gpt2.md](../models/gpt2.md), `gemm.cpp`).
- **Decode**: column-block GEMVs vs K split on the iPhone 6.2 vs 6.0 ms (wash; K split kept for the Mac's 10.9 → 7.9)
  ([gpt2.md](../models/gpt2.md)).
- **Mac in-model blockings**: none beat one multithreaded Accelerate call; best was 16 column blocks at ≈ 1 % e2e.
  `park()` before multithreaded BLAS helps standalone (168 → 120 µs after a `parallelFor`) but did not change Kokoro
  (148.4 ms) ([todo.md](../todo.md) M6).
- **MNIST iPhone CPU** (vs 1.5 / 11.2 / 161 µs at B = 1 / 64 / 1024): single-threaded Accelerate 1.8 / 11.8 / 164
  (wash); 2 row or column blocks (B = 64: 53 µs; B = 1024: 178–208 µs); a pool epilogue after the multithreaded GEMM
  (B = 1024: 292 µs; now only gelu epilogues go to the pool); a fused 3-layer kernel 1.7 / 12.8 / 173 µs
  ([mnist-mlp.md](../models/mnist-mlp.md)). Two row blocks on the generic kernel: B = 64 / 1024 53 / 208 µs vs
  12 / 162 µs (`gemm.cpp`), hence the 50 MFLOP floor of `iphoneRowSplit`.
- **SME2 GEMM for ALBERT** (59 × 768 × 2304 etc.): 0.19 vs Accelerate 0.12 ms standalone
  ([iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md)).
- **LSTM**: one direction split over 2 cores by hidden unit with a per-step spin barrier — 8.1 vs 10.4 µs/step in a
  microbench, but slower in the kernel (T = 147: 1.52 → 1.88 ms; 4 sgemv calls per step + pool hand-off)
  (`kokoro_lstm.cpp`, [todo.md](../todo.md) M6). LSTMs on the GPU: a per-step dispatch alone costs more than the CPU
  step and a persistent kernel can't sync threadgroups per step → CPU in both builds ([todo.md](../todo.md) M2).
  `lstm_bidir_t` (SME2 / transposed sgemv) is faster but **off by default**: the reordered sums move app47m's audio
  SNR 37.8 → 36.1 dB vs the f64 golden (test limit 37.5) ([iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md)).

GPU:
- **Hand-written 32×32 kernel as the general GEMM**: replaced by steel for M > 8 (Mac); stays on the iPhone above
  M = 256 where steel was slower (B = 1024: 0.49–0.52 vs 0.43 ms) (`kokoro_lc.cpp`).
- **Small-M kernel for any M** (row blocks of 8): loses from M = 16 on the Mac and at M = 64 on the iPhone (0.19 vs
  0.34 ms for MNIST B = 64), so the threshold stays 8 (`kokoro_lc.cpp`, [todo.md](../todo.md) M6).
- **iOS small-grid threshold 32** (instead of 128 threadgroups): no gain on MNIST / ResNet-18 / MobileNetV2
  (`kokoro_lc.cpp`, [todo.md](../todo.md) M6).
- **Steel as GPT-2's default GEMM**: slower than MPS on GPT-2's shapes (T = 512 prefill 26.1 vs 23.5 ms); kept
  behind `gemm="steel"` ([todo.md](../todo.md) M4).
- **Split-K at large M / on the iPhone**: at T = 512 (96 tiles) split-K steel is slower than MPS (not measured as a
  separate number); iPhone T = 128 11.89 vs 11.43 ms → MPS above 48 tiles and always on the iPhone
  (`transformer_gpu.cpp`).
- **First decode GEMV `lk_gemv`**: slower than MPS on the M2 Max (5.2 vs 3.6 ms per step); replaced by `gemv4`
  (`models/gpt2.py`). gemv4 tile widths: Mac cw 8 / 16 2.6 ms vs cw 4 2.2; iPhone cw 4 6.01 vs cw 8 5.60
  (`transformer_gpu.cpp`).
- **Naive f32 simdgroup GEMM without threadgroup staging** (research): 0.2–0.75 TF/s, register pressure
  ([research-kernels.md §2.1](../impl-notes/research-kernels.md)).

## 5. Open ideas

From [todo.md](../todo.md) and the "Open" sections of the impl notes; none of these is built.
- Split-K for the other underfilled N = 768 GEMMs on the Mac GPU (todo M6).
- Re-check the old "slower than MPS" verdicts on the scalar decode kernels and the steel GEMM: they may only lack the
  full-unroll fix that made the attention kernels fast (todo M6).
- iPhone GPU: f32 Metal 4 tensor ops (`matmul2d`) experiment, on the Kokoro GPU agent's list (todo M2).
- In-model Accelerate GEMMs cost ~2× their standalone time (ALBERT QKV 224 vs 112 µs on the Mac); open on the
  gemm.cpp threading / `park()` side ([iphone-cpu-sme2.md](../impl-notes/iphone-cpu-sme2.md), todo M6).
- iPhone CPU GEMM throughput: the shared SME2 engine below exists; wiring it where it wins needs the phone (todo M6).
  MNIST iPhone CPU sits at Accelerate's limit; the engine's microbench has its fc layer at 7.7 vs 9.3 µs (B = 64) and
  123 vs 145 µs (B = 1024) with A pre-packed — not wired (needs an in-model A/B on the phone).
- MNIST: NEON GEMM for large M; re-bench out of LPM (todo M6).
- LSTMs (the remaining ≈ 8 ms GPU-idle window at app59f, latency-bound at ~10–12 µs/step): overlap the TextEncoder
  LSTM with a duration LSTM (independent), pack `w_hh` per core at prepare time (todo M6).
- Re-verify the multi-stream AMX finding (3 concurrent single-thread `sgemm` = 3.4 TF/s) out of LPM (todo M0).

## Shared SME2 GEMM engine (`kernels/cpu/sme2_gemm.{h,cpp}`) **[built; opt-in, needs iPhone validation]**

One engine for every SME2 GEMM (vision convs, and the Kokoro agent's conv tile is to move onto it). Compiled with
`-march=armv8.6-a+sme2` on every Apple target (CMakeLists; no `+sve`), used only when `sme2::available()`
(`hw.optional.arm.FEAT_SME2`: the iPhone 18 Pro Max yes, the M2 Max no — callers fall back to Accelerate; on CPUs
without SME2 the engine's own entry points run a scalar reference, which is what the Mac unit tests exercise).

- `C[m, n] = epi(A · B)`: FMOPA outer products into the four 16×16 f32 ZA tiles, one 32 × 32 block of C per pass over
  K (per k: one 2-vector load of A, one of B, four FMOPAs; K unrolled by 2).
- A (C's rows) is always packed k-major in 32-row panels `[⌈m/32⌉][k][32]`: `packPanels` (ZA transposes: rows loaded
  as horizontal slices, stored as vertical slices), `packPanelsGather` (implicit im2col: per-row segment pointers),
  or pre-packed weights.
- B is read in place through `BOperand{b, ldb, panel, taps, dil}`: plain row-major `[k, n]`, k-major panels from
  `packPanelsKMajor`, or Kokoro-style channel-first conv windows (tap kk of channel c at `c·ldb + kk·dil`).
- Epilogue: ZA starts from 0 / C (`accumulate`) / the residual; bias (per column or per row) is one more outer
  product `ones ⊗ bias`; tiles are stored straight from ZA (horizontal slices); the activation (ReLU / ReLU6 / leaky)
  is a NEON pass per ≤ 256 KiB chunk of C. (Reading ZA back into Z registers for the epilogue cost ~3× on K = 24.)
- Loop order: A panels in ≤ 4 MiB chunks, B 32-column blocks outside the chunk's panels (weights stream once).

iPhone microbench (`[sme2bench]`, 2026-11-02, µs; one stream, A pre-packed vs single-threaded Accelerate):

| shape (m × n × k) | Accelerate 1 thread | Accelerate MT | SME2 (pack incl.) | SME2 A pre-packed | 2 / 6 pool streams |
|---|---|---|---|---|---|
| ResNet 56×56 3×3 (3136 × 64 × 576) | 157 | 157 | 169 | 128 (1.8 TF/s) | 176 / 211 |
| ResNet 28×28 (784 × 128 × 1152) | 151 | 150 | 149 | 129 | 161 / 195 |
| ResNet 14×14 (196 × 256 × 2304) | 143 | 171 | 147 | 140 | 151 / 193 |
| ResNet 7×7 (49 × 512 × 4608) | 183 | 269 | 179 | 168 (B packed 157) | 170 / 233 |
| MobileNetV2 pw 56×56 (3136 × 144 × 24) | 17.5 | 17.5 | 57 | 54 (old in-ZA epilogue) | 37 / 52 |
| MNIST fc (64 × 128 × 784) | 9.3 | 9.2 | 8.8 | 7.7 | 39 / 42 |

More streams lose: the SME unit is per cluster (two P-core threads share one). Weights pre-packed into panels gain
3-6 % over reading them in place.
