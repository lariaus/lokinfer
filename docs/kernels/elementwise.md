# Elementwise, activation, layout and signal kernels

Kernel docs: [convolution](convolution.md) · [gemm](gemm.md) · [elementwise](elementwise.md) ·
[reductions](reductions.md) · [attention](attention.md)

Everything here is f32 and memory- or latency-bound ([research-kernels.md §2.6](../impl-notes/research-kernels.md): the GPU ridge point is ~30 FLOP/byte,
the CPU's ~6-10), so the work is about the number of passes and dispatches more than per-kernel tuning. Kernels that
reduce (stats, norms, pooling, durations) are in [reductions.md](reductions.md); im2col, convT gathers and depthwise
convs are in [convolution.md](convolution.md); KV-cache writes and softmax in [attention.md](attention.md).

Status: everything listed is **built** and registered. Code: `kernels/cpu/{elementwise,kokoro_misc,kokoro_signal,
kokoro_conv,conv2d,decode,gemm}.cpp`, `kernels/cpu/{neon_math,kokoro_common}.h`, `kernels/gpu/{elementwise,transpose,
kokoro_lc,kokoro_tok,vision}.metal` with dispatch in `kernels/gpu/{dispatch,kokoro_lc}.cpp`. Python definitions (the
only place op signatures and shape rules live): `python/lokinfer/ops/{elementwise,kokoro,kokoro_gpu,vision,
transformer,gemm,view}.py`.

## Kernels

### Binary / unary elementwise and activations

| kernel | dev | layout | what (fused) | used by | Python def |
|---|---|---|---|---|---|
| `add_f32_cpu` | CPU | any, same shape | a + b | GPT-2 CPU residuals and wte + wpe; Kokoro ALBERT embedding sums, basic-graph residuals, generator `x + xs` | `elementwise.py` |
| `mul_f32_cpu` | CPU | any | a · b | no model graph | `elementwise.py` |
| `add_f32_gpu`, `mul_f32_gpu` | GPU | any | a + b, a · b | no model graph | `elementwise.py` |
| `scale_f32_cpu` | CPU | any | x · s | Kokoro basic AdainResBlk1d (× 1/√2) | `kokoro.py` |
| `add3_scale_f32_cpu` | CPU | any | (a + b + c) · s | Kokoro CPU generator (mean of 3 resblocks) | `kokoro.py` |
| `add3_scale_f32_gpu` | GPU | any | (a + b + c) · s | Kokoro GPU generator | `kokoro_gpu.py` |
| `leaky_relu_f32_cpu` | CPU | any | leaky(x, slope) | Kokoro CPU (TextEncoder, basic resblocks, generator 0.1 / 0.01) | `kokoro.py` |
| `gelu_tanh_f32_cpu` | CPU | any | torch's tanh-approx GELU | Kokoro ALBERT on the CPU build | `kokoro.py` |
| `bias_gelu_f32_gpu` | GPU | `[R, C]` | gelu_tanh(x + bias[col]) | Kokoro ALBERT (GPU), GPT-2 GPU prefill (after MPS) | `kokoro_gpu.py` |
| `snake_f32_cpu` | CPU | `[C, L]` | x + (1/α)·sin²(αx) | Kokoro `fused=False` graph only | `kokoro.py` |
| `snake_affine_lc_f32_gpu` | GPU | `[L, C]` | snake((x − mean)·scale + beta, α) per channel | Kokoro GPU only with `LOKINFER_KOKORO_WINO=0` | `kokoro_gpu.py` |
| `leaky_affine_lc_f32_gpu`, `leaky_lc_f32_gpu` | GPU | `[L, C]` | leaky([(x − mean)·scale + beta], slope) | Kokoro GPU (upsample AdainResBlk1d, generator leaky 0.1; decoder convs when `LOKINFER_KOKORO_WINO_DEC=0`) | `kokoro_gpu.py` (`leaky_lc_f32_gpu(x, slope, affine=None)` picks one) |
| `sigmoid_f32_cpu` | CPU | any | 1 / (1 + e^−x) | DCGAN D on the CPU | `vision.py` |
| `sigmoid_f32_gpu` | GPU | any | same | no model graph (DCGAN D GPU uses `gemv1_bias_sigmoid`, [gemm.md](gemm.md)) | `vision.py` |

Activations fused into other kernels (described in their own docs): bias / ReLU / ReLU6 / leaky epilogues of the
convs and GEMMs, the GEMM gelu epilogue (`detail::geluTanh4`, below), tanh in `convt2d_gather_nhwc_tanh_*` (DCGAN G),
AdaIN + Snake / leaky in the CPU fused conv prologue and the GPU Winograd input transform
([convolution.md](convolution.md)).

### Layout, indexing, copies

| kernel | dev | layout | what | used by | Python def |
|---|---|---|---|---|---|
| `transpose_f32_cpu` | CPU | `[R, C] → [C, R]` | 8×8 blocked scalar transpose | Kokoro CPU (TextEncoder, `asr`) | `gemm.py` |
| `transpose_f32_gpu` | GPU | `[R, C] → [C, R]` | 32×32 threadgroup tiles | MNIST (weights, hoisted to init), Kokoro GPU (decoder inputs, noise branch to channel-last) | `gemm.py` |
| `nchw_to_nhwc_f32_cpu` | CPU | NCHW → NHWC | graph-boundary conversion | ResNet-18, MobileNetV2, DCGAN D (CPU) | `vision.py` |
| `nchw_to_nhwc_pad_f32_gpu` | GPU | NCHW → NHWC, C zero-padded | conversion + channel pad for the steel loader (Ci % 16) | MobileNetV2 GPU (3 → 16) | `vision.py` |
| `nhwc_to_nchw_f32_cpu` / `_gpu` | CPU / GPU | NHWC → NCHW | output conversion | DCGAN G | `vision.py` |
| `lc_to_cf_f32_gpu` | GPU | `[L, Cpad] → [cout, L]` | channel-last → channel-first, drops pad channels | Kokoro GPU stage outputs `dec_x`, `post` (tests / named outputs only) | `kokoro_gpu.py` |
| `concat_pad_lc_f32_gpu` / `pad_cols_lc_f32_gpu` | GPU | `[L, Ca] (+ [L, Cb]) → [L, Cout]` | channel concat + zero pad to a multiple of 16 | Kokoro GPU decoder (514 → 528, 1090 → 1104 channels) | `kokoro_gpu.py` (`concat_pad_lc_f32_gpu(a, b=None)`) |
| `upsample2_lc_f32_gpu` | GPU | `[L, C] → [2L, C]` | nearest ×2 | Kokoro GPU AdainResBlk1d shortcut | `kokoro_gpu.py` |
| `upsample_nearest1d_f32_cpu` | CPU | `[C, L] → [C, sL]` | nearest ×s | Kokoro CPU AdainResBlk1d shortcut | `kokoro.py` |
| `pad_reflect1d_f32_cpu` | CPU | `[C, L]` | reflect pad | Kokoro CPU generator ups[1] | `kokoro.py` |
| `broadcast_rows_f32_cpu` | CPU | `[C] → [R, C]` | row broadcast | Kokoro ALBERT token-type row | `kokoro.py` |
| `concat_rows_f32_cpu`, `concat_cols_f32_cpu` | CPU | 2..8 inputs | `torch.cat` dim 0 / 1 | Kokoro (decoder input, DurationEncoder style concat) | `kokoro.py` |
| `embedding_f32_cpu` | CPU | table `[V, D]`, ids `i64[T]` | row gather, ids range-checked | Kokoro (ALBERT, TextEncoder), GPT-2 (wte, wpe) | `kokoro.py` |
| `repeat_interleave_f32_cpu` | CPU | `[T, C] → [F, C]` | alignment as an exact gather; errors if Σdur ≠ F | Kokoro | `kokoro.py` |
| `arange_offset_i64_cpu` | CPU | `i64[T]` | pos[0] + i (position ids) | GPT-2 `build_generate` | `transformer.py` |
| `reshape`, `flatten`, `dim_read` | — | — | zero-cost IR `view` / `dim.read`, no kernel | all | `view.py` |

`kv_write[_rows]_f32_{cpu,gpu}` (cache row copies) are documented in [attention.md](attention.md);
`im2col_nchw_f32_gpu` in [convolution.md](convolution.md).

### Signal (Kokoro vocoder)

| kernel | dev | layout | what | Python def |
|---|---|---|---|---|
| `harmonic_source_rng_f32_cpu` | CPU | f0 `[2F]` → `[600F]` | SineGen + Linear 9→1 + tanh, noise from in-kernel Philox (`nohoist`); shipped variant | `kokoro.py` |
| `harmonic_source_f32_cpu` | CPU | + noise `[600F, 9]` | same with explicit noise (parity tests) | `kokoro.py` |
| `stft_magphase_f32_cpu` | CPU | `[S] → [22, S/5 + 1]` | `torch.stft` (center, reflect, periodic hann) → (|X|, angle X) | `kokoro.py` |
| `istft_logits_f32_cpu` | CPU | `[22, L] → [5(L − 1)]` | `torch.istft(exp(x[:11]) · e^{i·sin(x[11:])})` | `kokoro.py` |
| `istft_logits_lc_f32_gpu` | GPU | `[L, Cpad] → [5(L − 1)]` | same math straight from conv_post's channel-last output | `kokoro_gpu.py` |

Both Kokoro builds run the harmonic source and the STFT on the CPU (the GPU build overlaps them with the GPU decoder,
[models/kokoro.md](../models/kokoro.md)); the iSTFT runs on the GPU in the GPU build.

## How they work

### CPU vectorization

- **Plain NEON loops** (`add`, `mul`, `scale`, `add3_scale`, `leaky_relu`): 4-lane `float32x4_t` with a scalar tail;
  `binary_f32` (elementwise.cpp) unrolls 16 elements per iteration (four independent load / op / store groups).
  `leaky_relu` is a compare + `vbslq` select. None of them use the thread pool; `sigmoid_f32_cpu` is a scalar
  `std::exp` loop (conv2d.cpp).
- **Accelerate vForce for transcendentals**, applied over L1-sized stack chunks of 2048 floats:
  - `snake_f32_cpu`: `buf = α·x` → `vvsinf` → `x + (1/α)·buf²` per channel — the exact torch formula and op order.
  - `gelu_tanh_f32_cpu`: `buf = √(2/π)(x + 0.044715 x³)` → `vvtanhf` → `0.5 x (1 + buf)` (torch's op order). Chunks
    run on the pool when there are ≥ 32 of them (n > 31 · 2048).
  - `convt2d_gather_nhwc_tanh_f32_cpu` applies `vvtanhf` to a whole output row at once (scalar tanh dominated
    small-Co rows; conv2d.cpp).
- **NEON polynomials** (the hand-written transcendentals, inlined into the kernels that need them):
  - `detail::exp4` (neon_math.h): 2ⁿ·p(r), r = x − n·ln2 split in two constants (Cody-Waite), degree-6 Taylor, input
    clamped to [−87, 88]; relative error ~1.2e-7 (code comment). Used by the attention softmaxes and `geluTanh4`.
  - `detail::geluTanh4` (neon_math.h): gelu_tanh(x) = x / (1 + exp(−2u)) on `exp4`; the GEMM bias + gelu epilogue
    ([gemm.md](gemm.md)). `geluTanh1` is the scalar torch form for tails.
  - `detail::sinSq4` (**kokoro_common.h**, not neon_math.h): sin²(x) with a 3-part Cody-Waite reduction by π (the sign
    of sin drops out of the square), then a degree-11 odd polynomial on [−π/2, π/2] (least-squares fit on Chebyshev
    nodes, fit error 2e-11 in double). Max abs error of sin² vs double: **2.4e-7 for |x| < 1e4** (kokoro_common.h
    comment; [todo.md](../todo.md) records 2.6e-7). Used by the CPU fused conv prologue's Snake (`transformRow` in
    kokoro_conv_fused.cpp, tail through a zero-padded vector), which therefore needs no `vvsinf` pass. The basic
    `snake_f32_cpu` keeps `vvsinf`.
- **Copies**: `embedding`, `repeat_interleave`, `broadcast_rows`, `concat_rows/cols`, `pad_reflect1d` are `memcpy`
  per row (plus scalar reflect edges); `upsample_nearest1d` uses `vst2q_f32` for ×2, `std::fill` otherwise.
  `embedding` checks every id before copying and `repeat_interleave` checks dur ≥ 0 and Σdur = frames, both reporting
  through `ExecCtx::error` (never truncate or pad silently).
- **Layout**: `nchw_to_nhwc_f32_cpu` special-cases C = 3 (image inputs): `vst3q_f32` interleaves 4 pixels per store,
  4096-pixel chunks on the pool above N·HW = 32768; other C and `nhwc_to_nchw_f32_cpu` are scalar strided loops.
  `transpose_f32_cpu` is an 8×8-blocked scalar loop (gemm.cpp).

### GPU kernels

- One thread per output element, 256-thread threadgroups (`dispatch1d`), for `lk_act_lc`, `lk_add3_scale`,
  `lk_concat_pad_lc`, `lk_upsample2_lc`, `lk_bias_gelu`, `lk_sigmoid`, `lk_nhwc_to_nchw`, `lk_nchw_to_nhwc_pad`.
  `add_f32` / `mul_f32` (elementwise.metal) process a `float4` per thread with a scalar tail in the last thread.
- Transcendentals use `precise::` (`precise::sin` in Snake, `precise::tanh` in GELU / convT tanh, `precise::exp` in
  sigmoid) so results stay close to the CPU kernels; snake in `lk_act_lc` and in the Winograd transform use the same
  expression.
- `lk_act_lc` applies the AdaIN affine `[3, C]` = (mean, scale, beta) centered first, `(x − mean)·scale + beta`, then
  snake (per-channel α) or leaky; `has_affine = 0` gives the plain leaky.
- `transpose_f32` (transpose.metal): 32×32 tile in threadgroup memory with one column of padding against bank
  conflicts, 32×8 threads, each thread moves 4 elements in and 4 out.
- `lk_nchw_to_nhwc_pad_pix` (vision.metal): one thread per pixel when Cpad % 4 = 0 and C ≤ 16: reads the C planes
  coalesced across the simdgroup, keeps 16 values in registers (compile-time indices) and writes `float4`s; the
  per-element `lk_nchw_to_nhwc_pad` remains for other shapes.
- `lk_lc_to_cf`: 2-D grid with consecutive threads on consecutive t, so writes are coalesced and the strided reads
  are left uncoalesced (comment: tiny next to the conv).
- `lk_concat_pad_lc`: the padded input channels are zeros and meet zero weight columns (`_w_lc_pad` in
  models/kokoro/model.py), so they add nothing to the next conv.

### Signal kernels (kokoro_signal.cpp, `lk_istft_lc`)

- The file is compiled with `#pragma clang fp contract(off)`: every FMA is an explicit `std::fma` that mirrors
  PyTorch's (bit-exact against torch 2.14 CPU for the phase interpolation, file comment).
- **Harmonic source**: per frame and harmonic h (9 harmonics), `rad = (f0·(h+1)/24000) mod 1` (torch.remainder
  semantics), cumulative phase summed in **double** (as torch accumulates float cumsum), ×2π×300, then torch's
  `upsample_linear1d` index math (`fma(scale, i + 0.5, −0.5)`) per output sample, `sin` (libm `std::sin`), voiced /
  unvoiced amplitude mix with the noise, the 9→1 linear and `tanh`. Output samples are independent given the phase
  table, so blocks of 8192 samples run on the pool; results are bit-identical to the serial loop.
- **In-kernel noise** (`harmonic_source_rng`): counter-based **Philox4x32-10** keyed by the 64-bit seed, counter =
  (sample index lo, hi, block, constant); 3 Philox calls per sample give 12 uniforms → Box-Muller → 12 normals, of
  which the 9 harmonics use 9. Stateless, so the noise depends only on (seed, sample) and the blocking can't change
  it. Deterministic per seed, not PyTorch's bit stream ([models/kokoro.md](../models/kokoro.md)). The op is `nohoist` so it is never moved into `init`.
- **STFT**: a direct windowed DFT (no FFT; n_fft = 20): a table `[2·bins, n_fft]` of w[m]·cos / −w[m]·sin built in
  double, reflect padding, per frame a dot product per bin, `hypot` / `atan2`, the imaginary part forced to +0 at DC
  and Nyquist (real-input FFT behaviour, keeps `angle` on the right side of the branch cut). Blocks of 2048 frames on
  the pool.
- **iSTFT (CPU)**: the inverse real DFT with the synthesis window and the 1/N, 2× (non-edge bins) factors folded into
  one table; per frame `exp(mag)`, `sin(phase logit)`, `cos/sin`, then the table → `fr[t, m]` (frames on the pool in
  blocks of 2048); the overlap-add and window-square envelope run serially in frame order, so the sum order equals a
  serial loop. Bins are bounded to 64 on the stack (the Python rule enforces n_fft ≤ 126).
- **iSTFT (GPU, `lk_istft_lc`)**: one thread per output sample; it walks its ≤ ⌈n_fft/hop⌉ frames in frame order and
  each frame's bins in bin order (the CPU's order), recomputing the window and the frame's spectrum
  (`precise::exp/sin/cos`) per sample; reads the logits straight from conv_post's channel-last output (no
  `lc_to_cf`, no CPU pass at the end of the graph).

## Measured performance

Mac = M2 Max (30-core GPU, [research-kernels.md §0](../impl-notes/research-kernels.md)). "iPhone" is as named by each source.

| what | device | case | result | source |
|---|---|---|---|---|
| Snake: scalar `std::sin` loop | CPU (device not stated) | one Kokoro stage-1 tensor | 113 ms per call | kokoro_misc.cpp:199-200, [kokoro-port-plan.md:71](../impl-notes/kokoro-port-plan.md) |
| Snake via `vvsinf` (E2E, basic graph) | Mac CPU, LPM off | rand510 | 4374 → 3208 ms | [todo.md:64](../todo.md) |
| GELU via `vvtanhf` (profile-driven fix) | CPU (device not stated) | Kokoro CPU profile | 71 → 12 ms | [todo.md:108](../todo.md) |
| GELU single-threaded | Mac CPU | ALBERT FFN `[T, 2048]`, app59f | 1.6 ms | kokoro_misc.cpp:162-163 |
| fused QKV GEMM on the CPU + GELU on the pool (E2E) | Mac CPU | app59f | 150 → 148 ms | [todo.md:204](../todo.md) |
| harmonic source parallel over sample blocks (bit-identical) | CPU (device not stated) | Kokoro CPU profile | 84 → 12 ms | [todo.md:108](../todo.md) |
| STFT / iSTFT share of the GPU build's profile (CPU kernels; LPM, battery) | Mac | rand510 | STFT 35 ms, iSTFT 26 ms | [kokoro-port-plan.md:45-46](../impl-notes/kokoro-port-plan.md) |
| iSTFT on the GPU + F0/N predictor on the GPU (E2E, one change) | Mac GPU build | app59f / word / real117 / rand510 | 62.4 → 54.8 / 31.6 → 29.0 / 126.7 → 110.3 / 379.8 → 302.8 ms | [todo.md:202](../todo.md) |
| CPU iSTFT at the end of the GPU graph (before the move) | Mac | Kokoro GPU build | ~1.1 ms | models/kokoro/model.py:698 comment |
| GPU snake + stats passes (before Winograd) | Mac GPU | rand510 | snake 14 ms, stats 10 ms | [todo.md:122](../todo.md) |
| GPU elementwise + stats total (M4 first cut, LPM) | Mac GPU | Kokoro | ≈ 15 ms | [todo.md:117](../todo.md) |
| Snake fused into the Winograd input transform: GPU snake + conv time | Mac GPU | rand510 | 235.7 → 179.4 ms (mm 154.8 + transform 24.6) | [winograd-conv.md:87-88](../impl-notes/winograd-conv.md) |
| `nchw_to_nhwc_pad` per-pixel kernel | GPU (device not stated; listed with the Mac GPU MobileNetV2 results) | MobileNetV2 N=8 stem | 0.263 → 0.080 ms | [todo.md:192](../todo.md) (vision.metal:20: 0.26 ms before) |
| DCGAN G convT gather by pixels + `vvtanhf` | CPU (device not stated) | N=64 | 19.7 → 11–12.6 ms | [todo.md:173](../todo.md) |
| `vvtanhf` vs NEON-exp gelu in the GEMM epilogue | iPhone CPU | GPT-2 | same speed | [models/gpt2.md:138-139](../models/gpt2.md) |
| `sinSq4` accuracy | — | \|x\| < 1e4 | max abs err 2.4e-7 (code) / 2.6e-7 (todo) | kokoro_common.h:104-105, [todo.md:101](../todo.md) |
| `exp4` accuracy | — | — | rel. error ~1.2e-7 | neon_math.h:10 |
| GPU dispatch fixed cost (background for fusion) | Mac GPU, LPM | tiny dispatch | 3.25 µs per command buffer; 1.16 µs per dependent dispatch | [research-kernels.md:500-501](../impl-notes/research-kernels.md) |

Not measured: GB/s of any elementwise or layout kernel in isolation (CPU or GPU), `transpose_f32_{cpu,gpu}`,
`lc_to_cf`, `concat_pad_lc`, `upsample2_lc`, `upsample_nearest1d`, `pad_reflect1d`, `embedding`,
`repeat_interleave`, `nhwc_to_nchw`, `sigmoid`, `add3_scale`, `leaky_relu`, the GPU iSTFT alone, and the CPU
`nchw_to_nhwc` `vst3q` path alone (it shipped in a batch of iPhone CPU convnet changes, [todo.md:177](../todo.md)).

## Tried and rejected

- **Scalar `std::sin` Snake** (113 ms per stage-1 tensor) → `vvsinf` in the basic kernel; the fused CPU path then
  dropped the `vvsinf` pass for the `sinSq4` polynomial in the conv prologue (kokoro_misc.cpp, kokoro_conv_fused.cpp).
- **Scalar `tanh` GELU** → `vvtanhf` (71 → 12 ms, [todo.md:108](../todo.md)); scalar tanh in the DCGAN convT gather → `vvtanhf`
  per row (conv2d.cpp:475).
- **AdaIN + Snake fused into the steel conv loader** (GPU): decided against ([todo.md:136](../todo.md)) — the implicit-GEMM loader
  reloads each input element once per tap and per 64-column tile (2·K times at Co = 128), so a fused snake recomputes
  its `sin` 6–22× per element (≈ +20 % conv ALU) vs a separate pass costing ≈ 0.3 ms of bandwidth per conv.
  Superseded by the Winograd input transform, which applies the activation once per input element.
- **Fusing the Winograd input transform (with its Snake) into the previous conv's epilogue**: impossible — the
  activation needs the AdaIN affine, i.e. the stats of the whole previous output ([winograd-conv.md:119-120](../impl-notes/winograd-conv.md)).
- **Per-element `lk_nchw_to_nhwc_pad`** for the 3-channel stem: scattered plane reads per simdgroup (0.26 ms at
  MobileNetV2 N=8) → the per-pixel kernel (vision.metal:20).
- **`vvtanhf` instead of the NEON-exp gelu** in the CPU GEMM epilogue: same speed, not adopted ([models/gpt2.md](../models/gpt2.md)).

## Open ideas (from [todo.md](../todo.md) open items)

- NEON polynomial transcendentals (sin / cos / exp / tanh / sigmoid) shared by all CPU kernels — only `sinSq4` and
  `exp4` exist ([todo.md:100-101](../todo.md)).
- Fused harmonic source + STFT with a polynomial sin, and a fused `post_istft` ([todo.md:99](../todo.md)); the in-kernel Philox
  part is done.
- Remaining fused-conv input transforms on the CPU: gather by `idx`, nearest ×2, depthwise convT, virtual concat,
  3-input mean; bias / residual / ×1/√2 epilogues ([todo.md:90](../todo.md)). Would remove `upsample_nearest1d`, `concat_rows`,
  `add3_scale` and `scale` passes.
- Alignment as a gather pushed through the shared-LSTM input projection and `asr_res`; implicit gather in the decoder
  conv prologue ([todo.md:98](../todo.md)) — would replace `repeat_interleave` + transposes.
- Fusing the elementwise passes around the GPU convs ([todo.md:122](../todo.md)); done for Snake / leaky via the Winograd input
  transform, the remaining GPU elementwise kernels (`leaky_lc` before the convT GEMMs, `add3_scale`, `upsample2_lc`,
  `concat_pad_lc`) are still separate passes.
- [research-kernels.md §2.6](../impl-notes/research-kernels.md) (not built): generating fused elementwise chains from a small template set instead of one
  hand-written kernel per op.
