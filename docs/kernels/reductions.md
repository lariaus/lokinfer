# Reduction kernels: statistics, normalization, pooling

Kernel docs: [convolution](convolution.md) · [gemm](gemm.md) · [elementwise](elementwise.md) ·
[reductions](reductions.md) · [attention](attention.md)

Everything here reduces: per-channel statistics for Kokoro's AdaIN / instance norm, LayerNorm (plain, adaptive,
fused with residual + bias, fused with split-K partial sums), pooling, and the duration head's sigmoid sum. The
attention softmaxes are in [attention.md](attention.md); GEMV reductions (including the DCGAN sigmoid GEMV
`gemv1_bias_sigmoid_f32_gpu`) are in [gemm.md](gemm.md); the convolutions whose epilogues emit statistics are in
[convolution.md](convolution.md) (this doc covers only their reduction side).

Status: everything listed is **built** and registered. Code: `kernels/cpu/{kokoro_norm,kokoro_conv_fused,decode,
conv2d,kokoro_misc}.cpp`, `kernels/cpu/kokoro_common.h`, `kernels/gpu/{kokoro_lc,kokoro_tok,vision,winograd}.metal`,
dispatch in `kernels/gpu/{kokoro_lc,transformer_gpu,winograd}.cpp`. Python definitions:
`python/lokinfer/ops/{kokoro,kokoro_gpu,transformer,vision,winograd}.py`.

## Kernels

| kernel | dev | layout | what (fused) | used by | Python def |
|---|---|---|---|---|---|
| `channel_stats_f32_cpu` | CPU | `[C, L] → [C, ⌈L/128⌉, 2]` | per-(channel, 128-sample chunk) (mean, M2) | Kokoro CPU fused graph (block inputs, `xst` shared by a stage's 3 resblocks) | `kokoro.py` |
| conv epilogue stats (`conv1d_{snake,leaky,affine}[_wino][_res]_f32_cpu`) | CPU | same `[Co, ⌈Lo/128⌉, 2]` | the same chunk stats of the conv output, written while the tile is in cache | Kokoro CPU fused graph | `kokoro.py` |
| `adain_affine_f32_cpu` | CPU | stats + gb `[2C]` → `[3, C]` | combine the chunks → AdaIN affine (mean, (1 + g)/std, b) | Kokoro CPU fused graph | `kokoro.py` |
| `channel_stats_lc_f32_gpu` (`lk_stats_lc_tg`) | GPU | `[L, C] → [⌈L/128⌉, C, 2]` | per-(chunk, channel) (mean, M2) | Kokoro GPU (decoder, F0/N, generator K = 7 convs, stage inputs) | `kokoro_gpu.py` |
| `adain_affine_lc_f32_gpu` (`lk_adain_affine_lc`) | GPU | stats + gb `[2 Creal]` → `[3, C]` | same combine; padded channels get (0, 0, 0) | Kokoro GPU | `kokoro_gpu.py` |
| `wino_mm_stats_lc_f32_gpu`, `wino_mm_res_stats_lc_f32_gpu` | GPU | stats `[d·chunks, Co, 2]` | chunk stats in the Winograd GEMM epilogue | Kokoro GPU generator K = 3 / 11 convs | `winograd.py` |
| `wino_adain_affine_lc_f32_gpu` (`lk_wino_adain_affine_lc`) | GPU | → `[3, C]` | combine of the epilogue's per-residue-class chunks | Kokoro GPU generator | `winograd.py` |
| `adain1d_f32_cpu` | CPU | `[C, L]` | instance norm + (1 + γ)x̂ + β in one kernel | Kokoro `fused=False` graph (and the fused graph's upsample block front) | `kokoro.py` |
| `layernorm_f32_cpu` | CPU | `[R, C]` | LN with γ, β | Kokoro (ALBERT embedding LN in both builds, ALBERT layers on the CPU build, TextEncoder channel LN), GPT-2 CPU graphs | `kokoro.py` |
| `adalayernorm_f32_cpu` | CPU | `[R, C]`, gb `[2C]` | (1 + gb[:C])·LN(x) + gb[C:] | Kokoro DurationEncoder | `kokoro.py` |
| `add_bias_layernorm2_f32_cpu` | CPU | `[R, C]` | s = o + bias + r; outputs (s, LN(s)) | GPT-2 CPU fused graph | `transformer.py` |
| `add_bias_layernorm_f32_gpu` (`lk_add_bias_ln`) | GPU | `[R, C]`, C ≤ 2048 | LN(a + bias + r) | Kokoro ALBERT (GPU) | `kokoro_gpu.py` |
| `add_bias_layernorm2_f32_gpu` (`lk_add_bias_ln2`) | GPU | `[R, C]`, C ≤ 2048 | (s, LN(s)), s = a + bias + r | GPT-2 GPU | `transformer.py` |
| `add_bias_layernorm2_sk_f32_gpu` (`lk_add_bias_ln2_sk`) | GPU | parts `[S, R, C]` | same, a = Σ of S split-K partial GEMM planes | GPT-2 GPU (Mac, small T) | `transformer.py` |
| `layernorm_f32_gpu` (`lk_layernorm_rows`) | GPU | `[R, C]`, C ≤ 2048 | LN | GPT-2 GPU (first ln_1) | `transformer.py` |
| `maxpool2d_nhwc_f32_cpu` / `_gpu` | CPU / GPU | NHWC | k×k, stride, padding as −inf | ResNet-18 | `vision.py` |
| `global_avgpool_nhwc_f32_cpu` / `_gpu` | CPU / GPU | `[N, H, W, C] → [N, C]` | mean over H·W | ResNet-18, MobileNetV2 | `vision.py` |
| `durations_f32_cpu` | CPU | `[T, 50]` → `i64[T]`, `i64[1]` | round(Σ sigmoid / speed), clamp ≥ 1, and the total F | Kokoro (both builds) | `kokoro.py` |

The C ≤ 2048 limit of the GPU LayerNorms (8 values per thread × 256 threads, kept in registers) is enforced by the
Python rules (`C must be static and <= 2048`); the kernels themselves don't check it.

## How they work

### Shared CPU primitives (kokoro_common.h)

- `sumBlocked`: 4×4-lane NEON accumulators (16 independent f32 sums) within 1024-element blocks; each block's f32 sum
  is added into a **double** total.
- `sumSqDevBlocked`: Σ(x − mean)² with the same blocking (`vfmaq` of the centered values).
- `meanVar` = **two passes** (mean, then centered sum of squares), biased variance. Every CPU norm uses it:
  `layernorm`, `adalayernorm`, `adain1d`, `add_bias_layernorm2`.
- `affineCentered`: `(x − mean)·a + b` with the mean subtracted first, so a large mean next to a tiny variance
  keeps full precision (the same order as the fused conv prologue and `lk_act_lc`).

### AdaIN / instance-norm statistics (Kokoro)

Kokoro normalizes each channel over time (up to 600F samples), then applies (1 + γ)x̂ + β with γ, β from the style
vector. The fused graphs split this into three steps so no standalone norm pass touches the frame-rate tensor:

1. **Chunk statistics.** For every (channel, 128-sample chunk) the producer writes (mean, M2 = Σ(x − mean)²), two
   passes over the chunk while it is in cache. The chunk size is fixed (`kStatChunk` = `STAT_CHUNK` = 128) so the
   stats don't depend on the conv's tile size (tiles are multiples of 128).
   - CPU: in the fused conv epilogue (`writeStats` after bias / residual / out-scale, per tile), or standalone with
     `channel_stats_f32_cpu` (channels on the pool above 4 M elements, inline below).
   - GPU, standalone `lk_stats_lc_tg`: a threadgroup of 32 × 8 threads owns 32 channels of one 128-row chunk; thread
     (x, y) walks rows 16y .. 16y + 15 of channel x (adjacent threads read adjacent channels: coalesced). Pass 1:
     partial sums → threadgroup `[8][32]` → mean; pass 2: partial M2 around that mean → reduce. The older
     `lk_stats_lc` (one thread per (chunk, channel) walking 128 dependent rows) is still in kokoro_lc.metal but no
     longer dispatched.
   - GPU, Winograd epilogue (`lk_wino_mm_lc_*_stats`): each 32-tile threadgroup reduces its 128 output rows per
     column in two register passes; partials are summed across lanes with `simd_shuffle_xor` (lane bits 2, 4, 16)
     and across the 2 simdgroup rows through threadgroup memory. For dilation d > 1 a chunk is 128 rows of one
     residue class, so the stats are `[d·chunks, Co, 2]` and the combine derives each chunk's count from (ρ, y).
     Built for K = 3 and K = 11; K = 7 keeps the standalone pass (below).
   - Shared stats: the three resblocks of a generator stage read the same input, so its stats (`xst`) are computed
     once (CPU and GPU, [todo.md:93](../todo.md)).
2. **Combine** into the AdaIN affine `[3, C]` = (mean, (1 + g)/√(var + eps), b), the exact parallel (Chan-style)
   merge of the chunk partials: mean = Σ nᵢmᵢ / N, M2 = Σ M2ᵢ + Σ nᵢ(mᵢ − mean)², biased var = M2 / N. Every chunk
   but the last has nᵢ = 128.
   - CPU `adain_affine_f32_cpu`: in **double**, two passes over the chunks (no per-chunk division), result cast to
     f32. Serial over channels.
   - GPU `lk_adain_affine_lc` / `lk_wino_adain_affine_lc` (no double on the GPU): **one simdgroup per channel**, 4 per
     128-thread threadgroup; lane i takes chunks i, i + 32, …, with **Kahan-compensated** f32 sums for both passes,
     then `simd_sum`. Channels ≥ Creal (zero padding to a multiple of 16) get affine (0, 0, 0), which keeps them
     exactly 0 through the next conv.
3. **Apply** in the next kernel's prologue: `(x − mean)·scale + beta` then Snake / leaky, in the CPU fused conv
   (`transformRow`), the GPU Winograd input transform, or `lk_act_lc` before a steel conv
   ([convolution.md](convolution.md), [elementwise.md](elementwise.md)).

The basic kernel `adain1d_f32_cpu` does all of it per channel in one call (`meanVar` + `affineCentered`), serially
over channels. [research-kokoro.md §2.5](../impl-notes/research-kokoro.md) item 4 motivates the design: instance norm over up to 135 k samples needs a
two-level (tile partial + Chan/Welford combine) scheme, not Σx² − (Σx)².

### LayerNorm

- **CPU** `layernorm_f32_cpu` / `adalayernorm_f32_cpu` (`normRows`): per row `meanVar`, then a NEON
  `(x − mean)·rstd·g + b` pass (`g = 1 + gb[c]`, `b = gb[C + c]` for the adaptive form). Rows are serial (no pool).
- **CPU** `add_bias_layernorm2_f32_cpu` (decode.cpp, the CPU twin of the GPU kernel): per row s = o + bias + r
  (NEON, written out), `meanVar(s)`, then LN(s). Row blocks on the pool (min(R, 4 × threads) tasks) when R·C ≥ 32768.
- **GPU** `lk_add_bias_ln`, `lk_add_bias_ln2`, `lk_add_bias_ln2_sk`, `lk_layernorm_rows` (kokoro_tok.metal): one
  256-thread threadgroup per row; each thread keeps its ≤ 8 elements in registers (the row is read once); sum →
  `simd_sum` → threadgroup `partial[8]` → every thread adds the 8 partials → mean; the same for Σ(v − mean)²
  (two-pass from registers); `rsqrt(var / C + eps)`; write. `_ln2` also writes the pre-norm sum (GPT-2's pre-LN
  blocks need the residual stream and the next sub-block's input). `_ln2_sk` adds the S split-K partial planes
  `a[k]` while loading, so the split-K reduction costs no extra pass; the dispatch recomputes the effective S with
  the GEMM's rule (`splitKEffective`: MPS into plane 0 once the 64×64 tiles reach 48, always on the iPhone, where
  `kSplitKMaxTiles = 0`).

### Pooling

- `maxpool2d_nhwc_f32_cpu`: one output row per task; per pixel, 16 channels in four NEON registers while the k×k
  window streams through (`vmaxq`), scalar tail channels; padding = −inf (window clipped). Inline below 2e5
  (rows·Wo·C·k²), on the pool above.
- `global_avgpool_nhwc_f32_cpu`: scalar f32 accumulation per channel over H·W, then × 1/HW; serial.
- GPU `lk_maxpool_nhwc`: one thread per output element (window clipped, −inf start). `lk_global_avgpool_nhwc`: one
  thread per (n, c), serial f32 sum over H·W.

### Durations (Kokoro)

`durations_f32_cpu`: per token, a sequential f32 sum of the 50 sigmoids (`1 / (1 + exp(−x))`, j = 0..49), divided by
speed, rounded with `nearbyint` (round half to even, like `torch.round`), clamped to ≥ 1; `total` = Σdur becomes the
data-dependent dim F (`dim.read`). Serial.

## Measured performance

Mac = M2 Max (30-core GPU, [research-kernels.md §0](../impl-notes/research-kernels.md)). "iPhone" is as named by each source.

| what | device | case | result | source |
|---|---|---|---|---|
| `adain_affine` two-pass combine (profile-driven fix) | CPU (device not stated) | Kokoro CPU profile | 26 → 8 ms | [todo.md:108](../todo.md) |
| `channel_stats` on the pool | CPU | — | not measured separately | [todo.md:108](../todo.md) |
| GPU AdaIN affine: one simdgroup per channel + Kahan (was one thread per channel) | Mac GPU | rand510 | E2E 361 → 351 ms; AdaIN affine was 12.4 ms (58 calls, 12 ms in the kernel comment) | [todo.md:188](../todo.md), kokoro_lc.metal:237 |
| old `lk_stats_lc` (one thread per (chunk, channel)) | Mac GPU | — | ~125 GB/s, latency-bound | kokoro_lc.metal:193 |
| `lk_stats_lc_tg` (E2E) | Mac GPU | app59f / rand510 | 51.8 → 48.4 / 295 → 270 ms | [todo.md:125](../todo.md) |
| `lk_stats_lc_tg` (E2E) | iPhone GPU | app59f | 66.4 → 64.9 ms | [todo.md:125](../todo.md) |
| GPU stats passes before Winograd | Mac GPU | rand510 | stats 10 ms (snake 14 ms) | [todo.md:122](../todo.md) |
| Winograd epilogue stats: standalone stats pass time | Mac GPU | rand510 | 10.9 → 2.2 ms | [winograd-conv.md:103](../impl-notes/winograd-conv.md), [todo.md:124](../todo.md) |
| Winograd epilogue stats: GEMM cost per K | Mac GPU | generator shapes | K = 11 −8 %, K = 3 +3 %, K = 7 +8–19 % | [winograd-conv.md:103-104](../impl-notes/winograd-conv.md) |
| Winograd epilogue stats (E2E, vs the old stats kernel) | Mac GPU | app59f / s9f / rand510 | 53.2 → 52.2 / ~98 → 96.3 / ~297 → 290.7 ms | [winograd-conv.md:104-105](../impl-notes/winograd-conv.md) |
| Winograd epilogue stats vs `lk_stats_lc_tg` (2 runs each) | Mac GPU | app59f / s9f | 48.5, 48.1 → 48.1, 48.8 / 93.8, 94.2 → 93.9, 88.4 ms (neutral to slightly better, kept) | [winograd-conv.md:106-108](../impl-notes/winograd-conv.md) |
| GPU elementwise + stats total (M4 first cut, LPM) | Mac GPU | Kokoro | ≈ 15 ms | [todo.md:117](../todo.md) |
| split-K partials folded into add + LN: the 24 N = 768 projections | Mac GPU | GPT-2 prefill T = 128 | 2.0 → 1.55 ms; E2E 5.45 → 4.95 ms (dashboard 4.99) | [models/gpt2.md:82](../models/gpt2.md), :111 |
| fused GEMM epilogues + add / bias / LN pass (one step, not separable) | Mac CPU / iPhone CPU | GPT-2 prefill T = 128 (iPhone also T = 512) | Mac 34.2 → 28.5 ms; iPhone 27.4 → 26.7, 85.3 → 80.1 ms | [models/gpt2.md:118-119](../models/gpt2.md) |
| LN + add share before fusion | iPhone CPU | GPT-2 prefill T = 128 | 0.7 ms | [models/gpt2.md:126-127](../models/gpt2.md) |
| CPU maxpool NEON, rows on the pool | iPhone CPU | ResNet-18 | 228 → 47 µs | [models/resnet18.md:36](../models/resnet18.md) |
| durations correctness | CPU | 9 reference cases, T = 3 … 512 | exact (no reduction-order fix needed) | [todo.md:60](../todo.md) |

Not measured: GB/s of `lk_stats_lc_tg`, the CPU stats / combine kernels in isolation, any LayerNorm kernel alone
(CPU or GPU), `adalayernorm`, `adain1d`, the GPU maxpool, both global avgpools, `durations` time.

## Tried and rejected

- **GPU stats with one thread per (chunk, channel)**: 128 dependent loads per thread, latency-bound at ~125 GB/s on
  the Mac → `lk_stats_lc_tg` (kokoro_lc.metal:190-193).
- **GPU AdaIN combine with one thread per channel** walking every chunk serially: latency-bound (58 calls, 12 ms on
  Mac rand510) → one simdgroup per channel (kokoro_lc.metal:234-237).
- **Epilogue stats for the K = 7 Winograd conv**: its 2-tap-group kernel crosses a register / occupancy cliff, +8–19 %
  on the GEMM, so K = 7 keeps the separate stats pass ([winograd-conv.md:103-104](../impl-notes/winograd-conv.md); model.py `_snake_conv_gpu`).
- **Fusing the Winograd input transform (which applies the AdaIN affine) into the previous conv**: impossible, the
  affine needs the whole previous output's stats; only the stats pass could be folded ([winograd-conv.md:119-120](../impl-notes/winograd-conv.md)).
- **Split-K + LN fold above 48 output tiles** (Mac GPT-2 T = 512: split-K steel slower than MPS) and **on the iPhone**
  (T = 128: 11.89 vs 11.43 ms with MPS): the op falls back to MPS into plane 0 and the LN reads one plane
  (transformer_gpu.cpp:130-139, [models/gpt2.md:82-85](../models/gpt2.md)).

## Open ideas (from [todo.md](../todo.md) open items)

- Why the K = 7 Winograd conv slows 8–19 % with fused stats ([todo.md:130](../todo.md)).
- `lk_stats_lc_tg` vs the Winograd epilogue stats: the rand510 pair is still to measure ([todo.md:125](../todo.md),
  [winograd-conv.md:108](../impl-notes/winograd-conv.md)).
- Split-K (with the partial sums folded into the next add + LN) for other underfilled N = 768 GEMMs on the Mac GPU
  ([todo.md:185](../todo.md)).
- Durations: the near-.5 rounding kernel test remains a known weak spot ([todo.md:60](../todo.md)).
- [research-models.md §2](../impl-notes/research-models.md) (planned there, not built): global avgpool + FC as one kernel or as the last conv's epilogue,
  maxpool fused into the ResNet-18 stem.
