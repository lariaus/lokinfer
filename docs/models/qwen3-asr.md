# Qwen3-ASR-1.7B on lokinfer

Status (2026-11-07): **phase 1 complete** (plan `impl-notes/qwen3-asr-port-plan.md`, research
`impl-notes/research-qwen3-asr.md`): golden data, basic + optimized kernels, model tests on the optimized variant (all
tiers pass on all 4 devices), the 8 Mac and 3 iPhone baselines and lokinfer on the dashboard, per-kernel profiles,
`lokinfer basic` timings, memory, docs. Optimization passes on both devices (Mac CPU / GPU kernels; iPhone GPU Metal 4
tensor-op GEMM; CPU decode GEMV streaming). lokinfer vs the best baseline, geomean over the dashboard cases: **Mac CPU
2.39×, Mac GPU 1.07×, iPhone CPU 2.02×, iPhone GPU 1.14×** (enc / prefill / decode / e2e: Mac CPU 3.99 / 3.82 / 1.01
/ 1.88, Mac GPU 1.23 / 1.00 / 1.01 / 1.06, iPhone CPU 4.85 / 0.93 / 1.96 / 1.79, iPhone GPU 1.67 / 1.01 / 0.97 /
0.98). Next: speech-to-text (impl-notes/speech-to-text-design.md).

HF `Qwen/Qwen3-ASR-1.7B-hf` (pinned revision `bcd2b5b7`), **f16 on every device**. lokinfer runs only the neural
model: the audio encoder (mel features → audio embeddings) and the LM (prefill + one-token decode with a KV cache);
feature extraction, prompt ids, the greedy loop, detokenization and the splitting of long audio stay in the caller
(the plan's "caller contract"; reference implementation `models/qwen3_asr/reference.py`).

## Test audio

`models/qwen3_asr/clips.py`: LibriSpeech test-clean (CC BY 4.0, downloaded once, sha256-pinned, never committed);
the clip definitions are committed in `models/qwen3_asr/clips.json`. Each clip is consecutive utterances of one
speaker cut at a pause near the nominal length: s2 2.45 s, s5 5.53, s10 10.09, s30 29.48, s60 59.01, s111 111.95,
s300 300.16 (all different speakers below 300 s). test-clean has ~8 min per speaker, so s1200 (1199.80 s, 3 speakers)
and s1800 (1800.88 s, 4 speakers) chain speakers back to back. Edge clips (synthetic, deterministic): `e_short`
(0.3 s: the processor's padding path), `e_1s` (one chunk), `e_window` (8.00 s: exactly one 104-row encoder window),
`e_window1` (8.08 s: a 1-row last window), `e_silence`, `e_noise`, `e_tone`.

## Precision gate (Q0, 2026-11-05)

`bench/research/qwen3asr/precision_gate.py` (results in `~/.cache/lokinfer/qwen3asr/gate/`): the HF model on every
clip ≤ 300 s and the edge clips, in f32 CPU, **f16 MPS (the app's path)**, bf16 MPS and f16 CPU (short clips: torch's
CPU f16 is a slow fallback), against two float64 goldens: **G-true** (the bf16 checkpoint in f64) and **G-f16w** (the
f16-rounded weights in f64). Transcripts are compared by teacher forcing: one f64 forward over each precision's
transcript; f64 greedy decoding gives that transcript iff the f64 argmax equals it at every position.

| | f32 CPU | **f16 MPS** | bf16 MPS | f16 CPU |
|---|---|---|---|---|
| last-row prefill logits, rel. error vs G-true | 1–3e-6 | **0.9–3e-3** | 6e-3 – 2.6e-2 | 0.7–3.7e-3 |
| audio embeddings, rel. error vs G-true (s2 → s300) | 1e-6 – 3e-5 | **6e-4 – 1.6e-2** | 6e-3 – 1.1e-1 | 8e-4 – 9e-3 |
| transcript = G-true's (14 clips) | 14 / 14 | **13 / 14** | 12 / 14 | 7 / 7 |
| non-finite values | 0 | **0** | 0 | 0 |
| max \|LM residual\| / SwiGLU product / down_proj out | 10876 / 14894 / 10710 | 10872 / 14896 / 10712 | 10752 / 14912 / 10752 | same as f16 MPS |
| max \|encoder layer output\| | 68 | 68 | 68 | 62 |

- **The one f16 difference is a near-tie**: s300, generated token 48, where G-true's two best logits are " and"
  42.0922 vs "," 42.0880 (gap 0.004). f16 writes "…so much imagination, and had cost…" instead of "…imagination and
  had cost…"; everything after is identical. bf16's two differences are near-ties too (s300 token 786; e_short token
  0, gap 0.015).
- **Activations**: the LM's "massive activations" (residual ≈ 10,900, the SwiGLU product feeding down_proj ≈ 14,900)
  are the same in every precision and stay 4.4× under f16's 65504: no overflow on any clip, including silence, noise
  and a pure tone. The encoder stays below 70.
- f16 is ≈ 8× more accurate than bf16 everywhere (logits ≈ 1e-3 vs ≈ 1e-2), as expected from its 3 extra mantissa
  bits.
- **Verdict: f16 is safe for this model (gate passed).** Correctness rule adopted from it (plan decision 3): a
  transcript difference vs G-true is accepted only at a position where G-true's top-2 logit gap is < 0.05.

## Reference timings (app path, for orientation)

HF `generate()`-equivalent loop, M2 Max, Mac otherwise busy with a baseline build (indicative only):

| clip | f32 CPU (prefill / generate) | f16 MPS (prefill / generate) |
|---|---|---|
| s10 (T = 149, 22 tokens) | 1.0 s / 4.1 s | 0.55 s / 0.68 s |
| s30 (T = 401, 80 tokens) | 3.4 s / 16.0 s | 0.60 s / 2.25 s |
| s111 (T = 1473, 410 tokens) | 18.8 s / 74.5 s | 1.77 s / 13.8 s |
| s300 (T = 3920, 941 tokens) | 31.7 s / 222 s | 4.95 s / 35.7 s |

## Q5 GPU kernels [built 2026-11-05]

The optimized f16 GPU ops of `impl-notes/qwen3-asr-f16-opt-kernels.md` (`kernels/gpu/f16_fused.metal`, dispatch in
`kernels/gpu/f16_gpu.cpp`, ops `<name>_opt_f16_gpu` in `python/lokinfer/ops/f16_gpu.py`). Weights stay in the plain
`[K, N]` layout (no packing step).

| op | kernel(s) |
|---|---|
| `gemm*_opt` (M > 1) | MLX steel GEMM, half in, float simdgroup-matrix accumulators, 64x64x32 tiles (32x32x32 when the 64-tile grid has < 128 tiles on the Mac / 24 on the iPhone); epilogue (+ bias, GELU(erf), + residual, clamp) on the float accumulators, one rounding. GPUs with neural accelerators (Apple10+: the iPhone 18 Pro Max; built 2026-11-07, **opt-in** with `LOKINFER_F16_GEMM=tensor` until validated on the phone, see docs/todo.md): Metal 4 tensor ops (`f16_tensor.metal`, `matmul2d`, f32 accumulation), 128x128 / 64x128 / 64x64 tiles, same epilogues (docs/kernels/gemm.md) |
| `gemm_swiglu_opt` (M > 1) | the gate and up tiles of `[gate \| up]` in one threadgroup (two steel K loops), `silu(g) * u` epilogue |
| `gemm*_opt` (M = 1) | GEMV: half4 columns, K split over a 256-thread threadgroup (8 rows in flight), 4·cw columns per threadgroup (cw 16 for N >= 2048 on the Mac), fused epilogue / SwiGLU |
| `lm_head_last_opt` | 4 table rows per simdgroup, 16-byte loads, f32 logits |
| `qknorm_rope_kv_opt` | one threadgroup per row: the row's 64 cos / sin once in threadgroup memory, one simdgroup per head (q, k, v copy) |
| `attention_window_qkv_opt` | simdgroup-matrix flash attention over packed qkv: Q / O float fragments, K / V half in threadgroup memory multiplied as half fragments (mixed-precision MMA, float accumulation), float online softmax; one threadgroup (13 simdgroups) per 104-row window |
| `attention_gqa_opt` | T > 1: the same flash body, causal over the caches, 32-row query blocks (64-row: 60.4 vs 45.7 ms at T = 2048, pos 13,570). T = 1: key-split decode (256 keys per split, <= 32 splits, both q heads of a kv head per threadgroup sharing the K / V reads) + a combine dispatch |
| `audio_cnn_opt` | 8 chunks per batch on the Mac, 4 on the iPhone (scratch ~11 MB per chunk): conv1 direct, conv2 / conv3 as im2col + steel `A @ wᵀ` with the bias + GELU epilogue, conv_out as one GEMM over the batch |

**MPS vs own kernel** (M2 Max): torch MPS f16 matmul (MPSGraph; it accumulates in f32: its error equals one f16
rounding) `[2048, 2048] @ [2048, 4096]` 3.93 ms (8.75 TFLOP/s) vs steel 64x64x32 4.53 ms (BK = 16: 5.14 ms; 64x32,
32x64, WM = 1, swizzle: not better). MPS is ~13 % faster on a plain large GEMM, but every other GEMM here has an
epilogue (an extra pass over an f32 output to keep one rounding; SwiGLU would need an f32 `[M, 2N]` intermediate) and
MPS disables GPU record / replay, so the steel kernel is used everywhere. M = 1: MPS 0.18 ms for `[1, 2048] @
[2048, 4096]` vs the GEMV 0.04-0.06 ms.

**Speedups vs the basic composition** (`bench/kernels/bench_f16_fused_gpu.py`: one graph per side, the op repeated
up to 8x per call so a call lasts >= ~5 ms, median; LPM off). Mac (M2 Max, 2026-11-05, background load ~3; repeated
weights fit the 48 MB SLC, so M = 1 GEMV times are optimistic: with 8 distinct weights the GEMV reads 270-330 GB/s for
N >= 2048), iPhone 18 Pro Max (remote executor; first rows 2026-11-05, the rest after the phone came back, 2026-11-06
22:20, cool):

| case | Mac basic → opt (ms) | Mac | iPhone basic → opt (ms) | iPhone |
|---|---|---|---|---|
| gemm qkv M=1 [2048x4096] | 0.320 → 0.039 | 8.1x | 0.684 → 0.092 | 7.5x |
| gemm qkv M=44 | 0.967 → 0.160 | 6.0x | 2.17 → 0.432 | 5.0x |
| gemm qkv M=401 | 8.38 → 1.02 | 8.2x | 17.8 → 3.21 | 5.5x |
| gemm qkv M=2048 | 43.9 → 4.76 | 9.2x | 88.5 → 13.1 | 6.8x |
| gemm_add o M=1 [2048x2048] | 0.204 → 0.032 | 6.4x | 0.370 → 0.054 | 6.8x |
| gemm_swiglu M=1 [2048x2x6144] | 1.05 → 0.144 | 7.2x | 2.04 → 0.492 | 4.1x |
| gemm_add down M=1 [6144x2048] | 0.612 → 0.040 | 15x | 1.11 → 0.136 | 8.2x |
| gemm_add o M=2048 | 21.3 → 2.49 | 8.6x | 44.3 → 7.38 | 6.0x |
| gemm_swiglu M=2048 | 109 → 13.9 | 7.8x | 263 → 39.8 | 6.6x |
| gemm_add down M=2048 | 62.6 → 6.65 | 9.4x | 131 → 19.6 | 6.7x |
| gemm_bias enc qkv M=3900 [1024x3072] | 31.2 → 3.70 | 8.4x | 64.6 → 9.73 | 6.6x |
| gemm_bias_add enc o M=3900 [1024x1024] | 10.5 → 1.26 | 8.3x | 21.3 → 3.72 | 5.7x |
| gemm_bias_gelu enc fc1 M=3900 [1024x4096] | 36.5 → 5.17 | 7.1x | 86.9 → 13.3 | 6.6x |
| gemm_bias_add_clamp enc fc2 M=3900 [4096x1024] | 38.5 → 4.76 | 8.1x | 84.1 → 12.5 | 6.7x |
| lm_head_last V=151936 | 2.31 → 1.73 (360 GB/s) | 1.3x | 8.94 → 5.92 (105 GB/s) | 1.5x |
| qknorm_rope_kv T=1 / T=2048 | 0.041 → 0.019 / 1.09 → 0.403 | 2.2x / 2.7x | 0.026 → 0.018 / 0.957 → 0.452 | 1.5x / 2.1x |
| attention_gqa decode pos 52 / 1850 / 4900 / 20000 | 0.043 → 0.038 / 1.91 → 0.042 / 10.7 → 0.064 / 52-107 → 0.286 | 1.1x / 46x / 168x / 180-370x | 0.048 → 0.026 / 0.664 → 0.064 / 5.62 → 0.123 / - | 1.9x / 10x / 46x / - |
| attention_gqa prefill T=2048 pos 0 / T=401 pos 3517 / T=2048 pos 13570 | 28-46 → 3.4-6.0 / 31.3 → 3.04 / 699 → 47.9 | ~8x / 10x / 15x | 70.7 → 7.49 / 54.2 → 5.82 / - | 9.4x / 9.3x / - |
| attention_window S=3900 / 15600 | 5.07 → 1.06 / 19.3 → 3.25 | 4.8x / 5.9x | 8.35 → 1.12 / - | 7.4x / - |
| audio_cnn n=5 / 30 | 42.5 → 4.25 / 294 → 26.4 | 10x / 11x | 76.4 → 11.6 / 486 → 66.2 | 6.6x / 7.3x |

Ranges = run-to-run spread on the loaded Mac. Tests: `tests/kernels/test_f16_fused_gpu.py` (152 cases: 134 default +
18 slow), each op vs the basic GPU composition and the float64 torch composition (tolerances in its docstring).
Mac 152/152; iPhone 152/152 (default + slow) on the final kernels.

Notes: the basic `attention_gqa` over T = 2048 at pos ~20k (one > 1 s dispatch) and the basic `audio_cnn` over 300
chunks after another op in the same command buffer (iPhone) intermittently came back as **zeros with no error** (a
GPU watchdog kill the runtime does not report as `GPU command buffer failed`); the tests run those oracles in
chunks.

## Q5 CPU kernels [built 2026-11-05; Mac + iPhone tested (slow tier both), timings both]

`kernels/cpu/f16_fused.cpp` (+ `f16_sme2.cpp`), ops `lk.ops.f16.<name>_opt`, contract
`impl-notes/qwen3-asr-f16-opt-kernels.md`, tests `tests/kernels/test_f16_fused_cpu.py` (each op vs the basic
composition and float64; 98 default + 9 slow cases).

- **GEMMs** (`gemm*_opt`, one engine + fused epilogues: bias, GELU(erf), SwiGLU over `w = [gate | up]`, residual,
  clamp; f32 epilogue, one rounding): M ≤ 4 → K-split GEMV (GPT-2's scheme: each pool task streams a contiguous range
  of W rows, 8 at a time, FMLAL by element into an f32 partial, partials summed by column chunks with the epilogue;
  M = 1: 2 rows × 64 columns per step, see "CPU decode GEMV streaming" below);
  larger M → 4 × 16 FMLAL / FMLAL2 register tiles, the W 16-column strip **packed contiguously per task** (read in
  place, the 8 KB row stride made the tiles run at ~1/4 speed: 194 → 740 GFLOP/s on [401, 2048]·[2048, 4096]);
  M ≥ 16 on SME2 CPUs → widening `FMOPA` (f16 → f32, two k per outer product) 32 × 32 panels, x packed in k pairs
  per 8-panel chunk, B zipped from two W rows in registers, P cores only. Weights stay in the basic `[K, N]` layout
  (no pack helper needed).
- **Attention**: flash-style tiles of 64 keys, scores by 4 × 4 FMLAL dot blocks, online f32 softmax, V tile
  converted to f32 once per tile; GQA task = (kv head, 16 query rows × the group's q heads, key split) — the key
  split (decode, short blocks over long caches) merges (max, sum, accumulator) partials. Window attention reads the
  fused QKV rows directly, task = (window, head).
- `qknorm_rope_kv_opt`: one pass per row (shared cos / sin), NEON; ≥ 99.9 % bit-exact to the basic composition
  (measured: 100 %). `audio_cnn_opt`: the basic signature, im2col + the optimized GEMM with the bias + GELU epilogue.
  `lm_head_last_opt`: 2-row FMLAL dot GEMV over the table on all P cores (4 rows until 2026-11-07).

Speedups vs the basic composition (2026-11-06, after the night perf pass of 2026-11-05/06: BNNS f16 GEMMs, vector
GELU, batched audio CNN; `bench/kernels/bench_f16_fused_gpu.py --cpu [--iphone]`: one graph per side, the op repeated
up to 8× per call; Mac: ≥ 1 s warm-up, ≥ 30 runs / ≥ 2 s, median; iPhone: RUN2 on the phone, cool ("nominal") at the
start, 4.3 min for the whole table). The iPhone run skips the longest cases (decode pos 20000, prefill pos 13570,
window S = 15600):

| case | Mac basic → opt (ms) | Mac | iPhone basic → opt (ms) | iPhone |
|---|---|---|---|---|
| gemm qkv M=1 [2048x4096] | 0.217 → 0.06 | 3.6x | 0.277 → 0.077 | 3.6x |
| gemm qkv M=44 [2048x4096] | 3.9 → 0.576 | 6.8x | 4.4 → 0.762 | 5.8x |
| gemm qkv M=401 [2048x4096] | 36.9 → 2.73 | 13.5x | 38.7 → 5.4 | 7.2x |
| gemm qkv M=2048 [2048x4096] | 193 → 13.3 | 14.5x | 206 → 29.2 | 7.1x |
| gemm_add o M=1 [2048x2048] | 0.072 → 0.026 | 2.8x | 0.092 → 0.039 | 2.4x |
| gemm_swiglu M=1 [2048x2x6144] | 1.5 → 0.485 | 3.1x | 1.5 → 0.341 | 4.4x |
| gemm_add down M=1 [6144x2048] | 0.454 → 0.095 | 4.8x | 0.618 → 0.16 | 3.9x |
| gemm_add o M=2048 [2048x2048] | 34.5 → 6.74 | 5.1x | 75.0 → 9.95 | 7.5x |
| gemm_swiglu M=2048 [2048x2x6144] | 227 → 41.4 | 5.5x | 487 → 109 | 4.5x |
| gemm_add down M=2048 [6144x2048] | 309 → 24.2 | 12.7x | 422 → 65.2 | 6.5x |
| gemm_bias enc qkv M=3900 [1024x3072] | 48.6 → 10.1 | 4.8x | 93.7 → 15.3 | 6.1x |
| gemm_bias_add enc o M=3900 [1024x1024] | 15.7 → 3.62 | 4.3x | 33.0 → 5.28 | 6.3x |
| gemm_bias_gelu enc fc1 M=3900 [1024x4096] | 96.3 → 16.8 | 5.7x | 199 → 35.9 | 5.5x |
| gemm_bias_add_clamp enc fc2 M=3900 [4096x1024] | 63.3 → 13.7 | 4.6x | 153 → 21.2 | 7.2x |
| lm_head_last V=151936 D=2048 | 5.12 → 5.53 | 0.9x | 7.78 → 6.14 | 1.3x |
| qknorm_rope_kv T=1 pos=1850 | 0.004 → 0.003 | 1.6x | 0.01 → 0.006 | 1.6x |
| qknorm_rope_kv T=2048 pos=0 | 1 → 0.465 | 2.2x | 1.54 → 0.697 | 2.2x |
| attention_gqa decode pos=52 | 0.005 → 0.004 | 1.3x | 0.01 → 0.007 | 1.4x |
| attention_gqa decode pos=1850 | 0.092 → 0.069 | 1.3x | 0.159 → 0.094 | 1.7x |
| attention_gqa decode pos=4900 | 0.259 → 0.165 | 1.6x | 0.453 → 0.272 | 1.7x |
| attention_gqa decode pos=20000 | 2.16 → 0.948 | 2.3x | - | - |
| attention_gqa prefill T=2048 pos=0 | 103 → 16.7 | 6.2x | 181 → 52.0 | 3.5x |
| attention_gqa prefill T=401 pos=3517 | 72.9 → 12.4 | 5.9x | 135 → 37.0 | 3.6x |
| attention_gqa prefill T=2048 pos=13570 | 2450 → 244 | 10.0x | - | - |
| attention_window S=3900 | 12.2 → 4.67 | 2.6x | 21.7 → 5.88 | 3.7x |
| attention_window S=15600 | 47.8 → 19.3 | 2.5x | - | - |
| audio_cnn n=5 | 74.2 → 14.8 | 5.0x | 113 → 23.9 | 4.8x |
| audio_cnn n=30 | 437 → 80.3 | 5.5x | 874 → 144 | 6.1x |

Notes: `lm_head_last` is bandwidth-bound for both (622 MB of table: ~110 GB/s on the Mac's 8 P cores, ~100 GB/s on
the iPhone), so opt ≈ basic there. The earlier 2026-11-05 table (before BNNS / vector GELU) is superseded. The
2026-11-05 iPhone hang in `test_audio_cnn_opt[3]` never came back (slow tier 106 passed on 2026-11-06 and 2026-11-07).

**`kSmeMinM` (iPhone, 2026-11-07): SME2 for every M above the GEMV range (5; was 16).** One deploy per setting, the
op timed on the phone (opt GEMM alone, 4 calls per graph, median), cool phone. ms:

| shape | path | M = 5 | 8 | 13 | 16 | 32 | 44 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| qkv [2048x4096] | NEON tiles | 0.52–0.59 | 0.56–0.59 | 0.81–0.85 | 0.86 | 1.43 | 1.85 | 2.57 | 4.83 |
| | SME2 | 0.31 | 0.28 | 0.28 | 0.28–0.37 | 0.31–0.38 | 0.84–0.91 | 0.89–0.92 | 1.77–2.09 |
| enc o [1024x1024] | NEON tiles | 0.049–0.058 | 0.054 | 0.083 | 0.088 | 0.157 | 0.212 | 0.305 | 0.593 |
| | SME2 | 0.035 | 0.034 | 0.035 | 0.035–0.039 | 0.038–0.043 | 0.070 | 0.071–0.075 | 0.142 |
| conv_out [7680x1024] | NEON tiles | 0.63 | 0.59–0.67 | 0.84–0.90 | 0.93 | 1.44 | 1.86 | 2.51 | 4.88 |
| | SME2 | 0.26 | 0.27 | 0.29 | 0.29 | 0.29–0.33 | 0.64–0.71 | 0.66–0.74 | 1.37–1.99 |

SME2 wins 1.6–3× from M = 5 and 2.5–4× above 16 (one 32-row FMOPA panel costs about the same for any M ≤ 32). The
whole graph on the tiles (`kSmeMinM` off): prefill 2 s / 5 s 381 / 624 ms vs 222–227 / 317–351 with SME2. In the
model, M = 5–15 GEMMs are rare (a prefill chunk's remainder, small conv_out batches); prefill / encoder rows ≥ 16 were
already on SME2. M ≤ 4 stays on the bandwidth-bound K-split GEMV.

**Side measurement — `BFMMLA` (bf16) vs `FMLAL` (f16) tile throughput, M2 Max** (no code decision; operands in L1,
K = 512, 1 s per measurement, two runs agree within 0.5 %): FMLAL 4 × 16 tile (the kernel above) **103.6 GFLOP/s on
one P core, 793 GFLOP/s on 8**; BFMMLA 8 × 8 tile (16 independent 2 × 2 accumulators) **51.8 GFLOP/s on one core,
396 on 8**. On this core BFMMLA's 16 MACs per instruction issue at a quarter of FMLAL's rate (4 MACs each): bf16
would not speed up the CPU GEMMs here.

## Performance on the Mac (2026-11-06)

Dashboard numbers (`bench/models/bench_qwen3asr.py` / `bench_qwen3asr_baselines.py`, median ms, M2 Max, AC; the
display asleep for the GPU rows: the aerial wallpaper's GPU use made GPU timings noisy). Best baseline per row.

| case | lokinfer GPU | best GPU baseline | lokinfer CPU | best CPU baseline |
|---|---|---|---|---|
| enc 10 s / 30 s | 23.8 / 54.7 | MLX 29.6 / 59.5 | 92.6 / 211 | llama.cpp 428 / 1129 |
| prefill 2 s / 5 s | 32.3 / 44.2 | llama.cpp Metal 32.6 / 43.3 | 140 / 174 | llama.cpp 362 / 629 |
| prefill 10 s / 30 s | 69.6 / 162 | llama.cpp Metal 68.2 / 167 | 241 / 545 | llama.cpp 1058 / 2758 |
| decode 2 s / 30 s | 12.0 / 11.7 | llama.cpp Metal 11.6, MLX 12.9 | 29.9 / 30.4 (2026-11-07; was 33.3 / 34.0) | llama.cpp 29.8 / 31.7 |
| e2e 5 s / 30 s | 320 / 1211 | MLX 337 / 1288 | 963 / 3990 | llama.cpp 1508 / 6522 |

Remaining gaps: short-prompt GPU prefill (within ~2 % of llama.cpp Metal either way). Mac CPU decode was ~10 % behind
llama.cpp until 2026-11-07; the cause was the GEMV's access pattern, not llama.cpp's f16 accumulation: now even
(see "CPU decode GEMV streaming" below).

First optimization pass (2026-11-05, details and A/B numbers in docs/todo.md, "Perf (Mac)"):
- CPU large-M GEMMs on BNNS (f16 × f16 → f32 on the AMX, ~2.4 TF/s): prefill 30 s 1772 → 545 ms; the vectorized GELU
  (`erfc4`) and a batched audio CNN: enc 30 s 732 → 211 ms.
- CPU decode GEMV: one K-split task per thread, two column blocks for gate|up: 35 → 33.3 ms.
- GPU: decode GEMVs with ≥ 128 threadgroups and rmsnorm folded into the next GEMV (new ops `rmsnorm_gemm_opt` /
  `rmsnorm_gemm_swiglu_opt`): 12.74 → ~11.9 ms; 64×64 GEMM tiles only from M ≥ 1024: prefill 10 s 77.5 → 69.6 ms.
- Runtime: GPU errors were silent after the first commit (a reused `MTL4::CommitOptions`); now reported by the call
  that caused them, via a per-command-buffer sentinel (docs/runtime.md, "GPU failures"), +0.01 ms per call.

## CPU decode GEMV streaming (Mac, 2026-11-07)

A decode step streams 3.44 GB of f16 weights (28 layers × 100.7 MB of GEMVs + the 622 MB LM head). Measured with a
standalone microbenchmark (10 layers of distinct decode weights, 1 GB ≫ the 48 MB SLC, the repo's thread pool,
alternating rounds, medians): the **8 P cores read at most 124 GB/s** (plain streaming loads; 4 threads 108, 7
threads 116–119; adding the E-cores adds nothing). The old K-split GEMV reached 104 GB/s, llama.cpp's decode
(29.8 ms) implies ~115.

| M = 1 GEMV variant (decode layer weights) | GB/s |
|---|---|
| K-split, 8 W rows per pass × 8 columns (old) | 102–104.7 |
| rows per pass 1 / 2 / 4 (16-column steps) | 103 / 110–111 / 100–104 |
| **2 rows × 64 columns per step (16 f32 accumulators), new** | **115.1–115.8** |
| 2 rows × 32 / × 16 columns | 112.7 / 112.0 |
| 1 row × 64 / 4 rows × 64 | 111 / 109 |
| one parallelFor with an in-task barrier instead of the separate reduction pass | ±0 |
| 2 / 3 K-split tasks per thread (2-row kernel) | 112–113.5 / 107 |
| N-split or 2-D column blocks over `[K, N]` (strided rows) | 63–90 |
| software prefetch 32 rows ahead | 75 |
| non-temporal loads | ±0 |
| column strips (one 128 B line per row, accumulators in registers) | 49 |
| dot-product layout `[N, K]` (llama.cpp's), 2 rows / 4 rows / 1 row | 116.7 / 102–110 / 111 |

Two concurrent streams per core with few L1 partial loads / stores per weight stream best. The new kernel
(`gemvRange1`) keeps the same per-lane FMLAL order (rows in k order into one f32 accumulator), so its outputs are
**bit-identical** to the old kernel (checked on 7 shapes; the s300 e2e transcript is token-identical old vs new), and
no accuracy rule is touched (f32 accumulation as before). The dot layout would gain another ~1–2 % but needs `[N, K]`
weights in every prefill path (BNNS, SME2, NEON tiles): not done. The LM head (already the dot layout) went from 4 to
2 rows per pass: 5.6 → 4.9 ms in-model.

In-model A/B (Mac CPU, `bench_qwen3asr.py` back to back, 3 alternations, load 7–12 from other sessions): decode 2 s
33.2 / 34.7 / 33.9 → 31.7 / 30.2 / 31.9 ms, decode 30 s 37.3 / 38.5 / 34.4 → 31.5 / 31.0 / 30.8 ms (≈ −9 %).
Profile at C = 480: gate/up 14.2 → 12.0 ms, o + down 9.3 → 8.4, qkv 4.3 → 4.2, LM head 5.6 → 4.9. Mac CPU decode is
now at llama.cpp CPU's level without changing the accumulation precision: dashboard (recorded at a 1-min load
average of 1.9) decode 2 s 29.9 ms (llama.cpp 29.8), 30 s 30.4 ms (llama.cpp 31.7).

**iPhone CPU (2026-11-07)**: the new kernels pass the fused CPU kernel tests and the Qwen CPU model tests (slow
tiers) on the phone. Decode step C = 480 (prof-lokinfer `iphone-cpu`, ~50 timed calls each, alternating deploys,
cool phone): K-split tasks per thread 1 / **2** / 3 = 48.7–48.8 / **40.8–42.5** / 49.4–50.1 ms (6 pool threads on
2 P + 4 E cores: 2 tasks per thread balance the P / E speed gap; 3 add partials and hand-offs). With 2: gate/up 17.3
ms (82 GB/s; 3 tasks: 22.5), o + down 10.9 (86; 15.3), qkv 5.7 (82; 7.7), LM head 6.2 (100 GB/s). LM head 2 vs 4
rows: 6.8 vs 7.0 ms (2 kept). Chosen: 2 tasks per thread, 2-row LM head, SME2 from M = 5 ("Q5 CPU kernels").

**s300 transcript, Mac CPU fused**: 942 tokens, first difference at token 48 (", and" instead of " and"). Not
caused by this change: the same process transcribes s300 with the old and the new kernels to identical token lists,
and the bench history (`~/.cache/lokinfer/qwen3asr/bench/lokinfer.jsonl`) has Mac CPU fused e2e 300 s
`matches_golden: false` already on 2026-11-06 (01:36 and 01:54), before any of this; only the Mac GPU rows were
golden. Accepted under the model-level rule (docs/correctness-testing.md: a first difference only at a near-tie):
G-true's top-2 margin at that position is 0.004 (" and" 42.0922 vs "," 42.0880; precision gate run log "min top-2
margin 0.004"), far under 0.05 — the same flip as f16 MPS and the basic kernels.

**Timing noise** (8 pool threads on 8 P cores): 7 pool threads lose ~10 % always (103 vs 115 GB/s) — rejected.
Background spinners at user-interactive QoS (1 ms busy / 1 ms idle): up to 4 of them change nothing (they land on the
E-cores); 6–8 of them: 1 / 2 / 3 tasks per thread read 86–105 / 98–113 / 93–98 GB/s (quiet 115 / 113 / 107).
In-model: 2 tasks per thread cost +1–2 % quiet (decode 30 s 30.5 → 31.1 ms) and save 7–9 % with 8 spinners (40.7 →
37.9 ms). Kept 1 task per thread (the dashboard and the llama.cpp comparison are quiet-machine numbers); a loaded
machine (other agents' builds, macOS `mediaanalysisd`) still shows in the CPU medians — record CPU rows only at a
1-min load average < 2.

## On the iPhone (2026-11-06)

iPhone 18 Pro Max, iOS 27; lokinfer via the remote executor (on-device times, `bench_qwen3asr_iphone.py`), baselines
via their harness apps (`record_qwen3asr_iphone_baselines.py`). Median ms; thermal state at the end of the case:
f = fair, s = serious, else nominal. The phone heats within minutes and stays hot while charging: the lokinfer rows
were re-measured on 2026-11-06 at night with a cool-down to "nominal" before every case (`--cool-to nominal`); the
baselines' long cases ran hot (their harness apps can't wait on the state). An earlier set was taken in Low Power Mode
by mistake: the bench now refuses to time in it, and perfdb never lets such a row replace a faster one.

| case | lokinfer CPU | llama.cpp CPU | lokinfer GPU | MLX Swift | llama.cpp Metal |
|---|---|---|---|---|---|
| enc 2s | 42.2 | 176 (f) | 10.0 | 17.0 | 21.4 |
| enc 5s | 85.3 | 351 (f) | 16.0 | 22.7 | 30.5 |
| enc 10s | 141 | 650 (f) | 24.1 | 41.6 | 58.5 |
| enc 30s | 349 | 1,815 (f) | 59.4 | 98.2 | 146 |
| enc 60s | 827 | 4,132 (s) | 113 | 179 (f) | 294 |
| enc 111s | 1,565 | 8,102 (s) | 211 | 346 (f) | 537 |
| enc 300s | 4,158 | 24,508 (s) | 561 | 1,107 (s) | 1,507 (f) |
| prefill 2s | 226 | 291 (f) | 40.2 | 35.9 | 56.6 |
| prefill 5s | 366 | 354 (f) | 49.0 | 51.9 | 59.7 |
| prefill 10s | 599 | 482 (f) | 67.7 | 71.2 | 106 |
| prefill 30s | 1,561 | 1,218 (f) | 152 | 146 | 219 (f) |
| prefill 60s | 3,048 | 2,555 (s) | 284 | 273 (f) | 392 |
| prefill 111s | 6,183 | 5,411 (s) | 566 | 635 (f) | 710 |
| prefill 300s | 21,430 | 22,724 (s) | 1,955 (f) | 1,993 (s) | 2,809 (s) |
| decode 2s | 37.2 (was 40.0) | 64.9 (f) | 32.4 | 32.8 | 32.3 (f) |
| decode 5s | 39.8 (was 45.6) | 78.3 (f) | 33.3 | 32.8 | 32.4 (f) |
| decode 10s | 41.9 (was 49.8) | 78.1 (f) | 33.4 | 32.9 | 32.4 (f) |
| decode 30s | 44.4 (was 51.6) | 83.1 (f) | 33.9 | 33.3 | 32.8 (f) |
| decode 60s | 45.7 (was 57.6) | 74.1 (s) | 34.4 | 33.6 (f) | 33.3 |
| decode 111s | 52.0 (was 60.4) | 106 (s) | 34.9 | 34.7 (s) | 34.1 |
| decode 300s | 64.0 f (was 71.7) | 178 (s) | 38.2 | 37.7 (s) | 37.2 (s) |
| e2e 5s | 1,950 | 2,816 (s) | 729 | 750 | 768 (f) |
| e2e 30s | 7,044 (f) | 12,406 (s) | 2,933 | 2,941 | 3,010 (f) |
| e2e 111s | 33,255 (s) | 61,771 (s) | 18,434 (f) | 17,416 (s) | 15,325 (f) |
| e2e 300s | 102,257 (s) | 221,474 (s) | 40,628 (f) | 46,847 (s) | 44,110 (s) |

- iPhone CPU: lokinfer leads on the encoder (4–6×), decode (1.3–2.5×) and end to end (1.4–2.2×); llama.cpp is ahead
  on prefill from 5 s to 111 s (up to 1.3×); lokinfer is ahead at 2 s and 300 s.
- iPhone GPU: decode is even with MLX / llama.cpp (32–38 ms). Since 2026-11-07 the M > 1 GEMMs run on Metal 4 tensor
  ops (the GPU's neural accelerators, `f16_tensor.metal`): enc and prefill rows re-recorded that night (from nominal
  per case). The encoder is now 1.4–2× faster than MLX Swift; prefill is ahead of MLX from 5 s to 300 s except 30 s /
  60 s (+4 %) and 2 s (40.2 vs 35.9 ms); e2e is now the fastest at 5 s, 30 s and 300 s (40.6 vs llama.cpp Metal's
  44.1 s) and behind only at 111 s (18.4 vs 15.3 s; decode-bound). Before (steel GEMMs, 2.4–2.8 TF/s): enc 1.4–2.1× and prefill 2.4–3.4× behind
  MLX (prefill 30 s 489 ms, 300 s 5,267 ms).
- e2e transcripts equal the golden tokens on both devices for every clip incl. 300 s (one 5 min pass, the iPhone's
  cap). Before the GPU memory fix the GPU e2e 300 s row was 69,960 ms (measured while the phone was short of memory).
- llama.cpp CPU e2e 300 s: 221 s median of 3 (one call took 565 s: throttling; a first attempt on 2026-11-06 hung
  once mid-encode). Measured alone after a 10 min idle cool-down; it still ended at "serious".
- Correctness on the phone: default tier 20/20 (CPU + GPU), GPU slow tier 28/28 after the GPU work budget fix
  (docs/runtime.md).
- Memory: the CPU e2e 300 s case ran into iOS `fc-thrashing` jetsams when the phone was short of RAM (the 4 GB of
  mmapped weights re-read from flash every step): docs/todo.md. Each time it followed a GPU session whose 4.3 GB of
  GPU allocations had stayed after the model was freed (residency-set removals not committed; fixed with
  `GpuContext::trim()`, docs/runtime.md).

## Memory (2026-11-06 / 07)

- Weights: 3.4 GB of f16, memory-mapped and file-backed (not in the process footprint; on the GPU, no-copy Metal
  buffers over the same pages). KV cache: 112 KiB per token (28 layers × K/V × 1024 × f16), sized to the pass:
  `MAX_CAP` = 25,600 tokens on the Mac (one 20 min pass, 2.9 GB), 6,656 on the iPhone (one 5 min pass, 0.73 GB).
- iPhone, e2e 300 s on a fresh LokiHost (one bench session): CPU resident peak 4.69 GB (incl. the mapped weights),
  `phys_footprint` peak 0.86 GB; GPU resident peak 0.28 GB, footprint peak 0.26 GB (the Metal no-copy weight buffers
  count in neither) — within the ~5.5 GB budget of the plan (the iOS per-app limit is ~6.1 GB).
- Freed models used to keep their GPU memory until the next GPU call (4.3 GB left with 0 models on the phone):
  fixed by the residency trim (docs/runtime.md). Under memory pressure from other apps iOS evicts the mapped weights
  and re-reads them every decode step (`fc-thrashing`, ~3 s / step, then a jetsam): docs/todo.md.
- Test sessions through the phone: each test's models are freed when it ends (FREE_MODEL); LokiHost's footprint
  peaks ~1.1 GB over the fused-kernel tests (was 3.1 GB before, until iOS killed it).

### Memory per case (iPhone + Mac, 2026-11-07)

`bench/models/measure_qwen3asr_memory.py [--where mac]`: a fresh executor per (graph, kind) (LokiHost restarted
through SHUTDOWN + the deploy watchdog; a Release `lokinfer-serve` on the Mac), clips ascending, one pass of each case,
MEMORY read after the calls with the KV cache still allocated. GB of `phys_footprint` (what iOS's per-app limit
counts); KV = the cache's size (prompt + max_new rounded up to 1024), of which only the written positions are physical
pages (anonymous zero-fill). Peaks equal the values within 0.02 GB except where noted.

**iPhone 18 Pro Max** (21:22-21:40, nominal; CPU e2e 300 s golden, 941 tokens; no page-ins after each kind's first
case, which reads the weights once):

| clip | CPU enc | CPU prefill | CPU decode | CPU e2e | GPU enc | GPU prefill | GPU decode | GPU e2e (peak) | KV size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 s | 0.03 | 0.02 | 0.03 | – | 0.03 | 0.04 | 0.04 | – | 0.11 |
| 5 s | 0.04 | 0.03 | 0.03 | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.11 |
| 10 s | 0.04 | 0.04 | 0.04 | 0.06 | 0.03 | 0.04 | 0.04 | 0.05 | 0.11 |
| 30 s | 0.05 | 0.08 | 0.09 | 0.09 (0.11) | 0.04 | 0.04 | 0.05 | 0.04 (0.05) | 0.11 |
| 60 s | 0.06 | 0.13 | 0.16 | 0.15 (0.17) | 0.06 | 0.05 | 0.05 | 0.04 (0.06) | 0.22 |
| 111 s | 0.07 | 0.25 | 0.30 | 0.28 | 0.09 | 0.06 | 0.06 | 0.05 (0.08) | 0.33 |
| 300 s | 0.15 | 0.51 | 0.60 | 0.64 | 0.15 (0.17) | 0.07 | 0.06 (0.08) | 0.07 (0.15) | 0.71 |

iPhone resident (the mapped weights once touched): CPU enc 0.64-0.77 GB, prefill / decode 3.24-3.83, e2e 3.87-4.47;
GPU allocated 0.81 (enc) to 4.83 GB (e2e 300 s, the no-copy weight wraps included). `os_proc_available_memory` at the
CPU e2e 300 s point: 5.36 GB (limit ≈ 6.0 GB). With the bench's pin (both CPU graphs, 3.8 GB) the CPU e2e 300 s
footprint would be ≈ 4.4 GB.

**Mac** (M2 Max, `--where mac`):

| clip | CPU enc | CPU prefill | CPU decode | CPU e2e (peak) | GPU enc | GPU prefill | GPU decode | GPU e2e (peak) | KV size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 s | 0.05 | 0.03 | 0.03 | – | 0.01 | 0.01 | 0.02 | – | 0.11 |
| 5 s | 0.06 | 0.05 | 0.06 | 0.08 | 0.01 | 0.02 | 0.02 | 0.02 | 0.11 |
| 10 s | 0.06 | 0.07 | 0.07 | 0.10 | 0.02 | 0.02 | 0.02 | 0.02 | 0.11 |
| 30 s | 0.08 | 0.13 | 0.14 | 0.19 | 0.02 | 0.02 | 0.02 | 0.03 | 0.11 |
| 60 s | 0.14 | 0.23 | 0.27 | 0.33 | 0.04 | 0.02 | 0.03 | 0.05 | 0.22 |
| 111 s | 0.20 | 0.35 | 0.44 | 0.46 | 0.07 | 0.04 | 0.04 | 0.08 | 0.33 |
| 300 s | 0.39 | 0.65 | 0.76 | 0.84 | 0.16 | 0.06 | 0.06 | 0.18 | 0.77 |

Mac resident (incl. the mapped weights once touched): CPU prefill / decode 3.2-4.0 GB, e2e up to 4.66 GB (3.8 GB of
weights for both graphs); GPU allocated (`currentAllocatedSize`, the no-copy weight wraps included) 3.3-4.9 GB.

Reading it:
- **CPU**: the footprint is the written part of the KV cache (112 KiB per token: ~0.53 GB at the 300 s clip's
  ~4.8k positions) plus ~0.1 GB (iPhone) to ~0.25 GB (Mac) of activation arenas and kernel scratch; the 2048-row prefill chunks add
  ~0.2 GB at most. Shrinking them (smaller chunks, or freeing the arena after prefill) would save ≤ 0.2 GB of a
  ~4.6 GB working set, against a prefill speed risk: not done. The 3.2 GB of weights are the problem under pressure,
  not the footprint (next section).
- **GPU**: almost nothing is charged to the footprint (its arenas and the KV cache writes go through Metal; e2e
  300 s peak 0.18 GB), and the weights are held resident by Metal while it runs.
- **KV cache sized to the pass**: already done everywhere (bench, tests, `transcribe`); untouched capacity costs no
  physical memory anyway, only address space and GPU-allocated size.

## Memory under pressure (iPhone, 2026-11-07)

`bench/models/pressure_qwen3asr_iphone.py`: the generate graph, a 30 s clip's prompt in the KV cache, 20 decode steps
timed per point while a "balloon" of random (incompressible) device tensors grows by 0.5 GB up to 3.5 GB; MEMORY after
each point (page-ins during the steps, weights resident, pressure level, system memory). iPhone 18 Pro Max, phone in
normal use (other apps open):

| graph, pin | balloon 0 | 0.5-2.0 GB | 2.5-3.5 GB | footprint at balloon 0 |
|---|---|---|---|---|
| GPU, none | 33.1 ms, 0 page-ins | 33.0 ms, 0 | 33.0-34.9 ms, 0 (pressure critical at 3.5) | 0.20 GB |
| CPU, none | 40.4 ms, 0 | median 55-73 ms, **max 2.1-2.7 s, 3.4-10.6 GB page-ins** per 20 steps | median 44-49, max 1.4-2.3 s, 2.8-8.4 GB | 0.36 GB |
| CPU, residency in the queue's set (first mode 3) | 50.5 ms, 0 | 39-41 ms, ~0 | max ~1.1 s, 1.7-3.5 GB at 3.0 / 3.5 GB | 0.36 GB |
| CPU, mlock | 45.4 ms, 0 | 39.5 ms, 0 (stopped: available < 0.6 GB) | – | **3.57 GB** |
| CPU, own residency set + requestResidency (final "gpu", 18:47 run) | 41.2 ms, 0 | 38.4 ms, 0 (stopped at 0.5 GB: critical) | – | **3.44 GB** |

- The CPU graph thrashes exactly as in the `fc-thrashing` jetsams: its weights are plain file-backed pages, the
  cheapest memory for iOS to reclaim, so they go before other apps' memory and come back from flash during the next
  steps (each step reads all 3.2 GB).
- The GPU graph doesn't: its weights are no-copy Metal buffers in the queue's residency set, which Metal keeps
  resident (system wired +3.2 GB) while the GPU works, without charging the app's `phys_footprint`. iOS reclaimed
  other apps instead (file cache 2.42 → 0.88 GB, compressor 0.46 → 3.5 GB).
- `mlock` (PIN_MODEL "lock") works on iOS (3.44 GB wired, no error) and stops the page-ins, but wired pages count
  in `phys_footprint` (0.36 → 3.57 GB; `os_proc_available_memory` 5.64 → 2.43 GB), so the weights then compete
  with the activations and the KV cache under the ~6 GB per-app limit (CPU e2e 300 s would sit at ~4.4 GB).
- Adding the CPU graph's weights to the queue's shared residency set helped only partly (that set is made resident
  while GPU work runs). PIN_MODEL "gpu" now uses a residency set of its own with `requestResidency()`: on the Mac it
  wires without charging the footprint (256 MiB: footprint +2 MiB), **but on iOS it is charged like mlock** (0.24 →
  3.44 GB) while it stops the page-ins just as well. So on the iPhone there is no free pin: keeping 3.2 GB of weights
  in RAM costs 3.2 GB of the ~6 GB per-app budget either way (CPU e2e 300 s would peak at ~4.3 GB: it fits). The
  GPU graph's weights get it for free only through the queue's own set while GPU work runs.
- Speed with the pin (iPhone, CPU, `--quick`, alternating none / gpu / none / gpu, 2026-11-07 21:18): prefill 30 s
  1673 / 1373 / 1413 / 1416 ms, decode 30 s 53.1 / 46.9 / 42.7 / 41.5 ms, e2e 30 s 7424 / 6001 / 6397 / 5467 ms, all
  golden: no cost. So `bench_qwen3asr_iphone.py` pins the CPU graphs ("gpu") by default (`--pin auto`; the GPU
  graphs stay unpinned). Footprint with both CPU graphs pinned: ~3.8 GB of weights + ≤ 0.9 GB (e2e 300 s) of the
  ~6 GB limit. The test harness doesn't pin (many models per session: headroom matters more there).
- Pressure is now visible to the client (MEMORY: `available`, `pressure`, `pageins`, `weights_resident`, ...;
  docs/runtime.md "Memory pressure"): the iPhone bench refuses to record a case under pressure or one that paged in
  more than 1.5x the weights (a case after a fresh load reads them once: 3.86 GB for 3.80).

## Fused vs basic (Mac, 2026-11-06)

`lokinfer basic` (the oracle kernels) on the dashboard for every case ≤ 300 s and for enc / prefill / decode 1200 s on
the CPU (`bench_qwen3asr.py --variant basic`; e2e > 300 s is never timed on basic). Mac GPU basic has `enc 1200s` only:
decode / prefill at 1200 s stop with a GPU timeout (one basic attention dispatch over ~15.6k keys outlasts the macOS
GPU watchdog; the optimized kernels split keys, so not fixed in the oracle). Speedup of `lokinfer fused` over basic,
geomean over the clips 2 s … 300 s (e2e: 5 s, 30 s, 111 s, 300 s):

| kind | Mac CPU | Mac GPU | iPhone CPU (≤ 30 s) | iPhone GPU (≤ 30 s) |
|---|---|---|---|---|
| enc | 4.6× (3.5–5.5) | 8.2× (5.4–10.1) | 4.4× (3.8–5.3) | 5.2× (4.5–6.6) |
| prefill | 7.3× (5.6–8.3) | 8.2× (6.9–9.6) | 4.6× (4.2–5.3) | 5.3× (4.6–6.2) |
| decode | 3.2× (3.0–3.4) | 9.9× (5.7–30; basic's GPU attention is not key-split, 417 ms at 300 s) | 2.1× (2.0–2.4) | 3.9× (3.8–4.2) |
| e2e | 3.7× (3.6–3.8) | 10.9× (6.0–24) | 2.4× (2.0–3.0) | 4.2× (4.0–4.5) |

iPhone: `bench_qwen3asr_iphone.py --variant basic`, clips ≤ 30 s (2026-11-06 night, from "nominal" per case; longer
clips risk the iOS GPU time limit with basic's single-dispatch attention), e2e transcripts golden.

Basic e2e transcripts equal the golden tokens up to 111 s; at 300 s both differ by a comma after "imagination"
(token 48; CPU checked, 942 vs 941 tokens), the same f16 near-tie MLX Swift and llama.cpp hit on the iPhone. Basic is
not held to the model-level rules (no model tests on basic kernels, user 2026-11-05).

## Per-kernel profile, 30 s clip (2026-11-06)

`tools/prof-lokinfer run qwen3asr_{enc,prefill,decode} -d <device> -c "enc 30s" | "prefill 30s" | "decode C=480"`
(optimized graphs, golden inputs; share of the profiled call). TF/s = the kernel's FLOPs over its own time.

| stage (Mac) | GPU | CPU |
|---|---|---|
| enc 30 s | 54.7 ms: audio_cnn 41 %, encoder GEMMs 55 % (6.6–8.2 TF/s), window attention 2 %, layernorm 2 % | 217 ms: audio_cnn 38 %, GEMMs 54 % (1.7–2.3 TF/s), window attention 5 %, layernorm 2 % |
| prefill 30 s (T = 401) | 144 ms: rmsnorm + gate/up GEMM + SwiGLU 46 %, o / down GEMMs + residual 32 % (8.1 TF/s), rmsnorm + QKV 16 %, attention 3 %, LM head (last row) 1 % | 538 ms: 45 / 31 / 18 % (2.3 TF/s), attention 5 %, LM head 1 % |
| decode C = 480 | 12.1 ms: gate/up 34 %, o / down 26 %, QKV 14 %, LM head 14 %, attention 5 %, qknorm + RoPE + KV 2 % | 37 ms: 38 / 26 / 12 %, LM head 15 %, attention 3 % |

- Prefill and encoder are GEMM-bound (≥ 85 % in GEMMs at 7–8 TF/s on the GPU); the encoder's first stage (audio_cnn:
  3 convs + conv_out) is the largest single kernel, 40 % of the encoder on both devices.
- Decode is weight-streaming: the 28 layers' GEMVs ≈ 80 %, the 151,936-row LM head ≈ 14 %.

| stage (iPhone) | GPU | CPU |
|---|---|---|
| enc 30 s | 161 ms: audio_cnn 41 %, GEMMs 55 % (2.5–2.7 TF/s), window attention 2 % | 425 ms: audio_cnn 50 %, GEMMs 69 % (0.6–1.0 TF/s; the CPU kernels overlap), attention 4 % |
| prefill 30 s | 489 ms: gate/up 48 %, o / down 28 %, QKV 16 % (2.4–2.8 TF/s), attention 2 %, LM head 1 % | 2,003 ms: 50 / 31 / 16 % (0.6 TF/s), attention 3 % |
| decode C = 480 | 47 ms (phone warm; 34 ms cool on the dashboard): gate/up 39 %, o / down 26 %, LM head 16 %, QKV 14 %, attention 2 % | 58 ms: 44 / 30 / 16 %, LM head 13 %, attention 3 % |

- iPhone GPU GEMMs run at 2.4–2.8 TF/s (Mac 7–8): the gap to MLX Swift on prefill (2.4–3.4×) is the GEMMs; MLX uses the
  A-series GPU's f16 matrix units (docs/todo.md; perf work paused).
- 2026-11-07: the M > 1 GEMMs now run on Metal 4 tensor ops on the iPhone (`f16_tensor.metal`, docs/kernels/gemm.md
  "f16 tensor-op GEMM"; same fused epilogues): 7–8 TF/s. `prof-lokinfer run qwen3asr_prefill -d iphone-gpu -c "prefill
  30s"` (128x128 tiles, before the 64x128 rule): 165.4 ms: rmsnorm + gate/up + SwiGLU 45.5 % (7.5 TF/s), o / down +
  residual 26.1 % (8.7 TF/s), rmsnorm + QKV 15.0 % (7.6 TF/s), attention 5.5 %, LM head 3.4 %. With 64x128 tiles the
  recorded prefill 30 s is 151.6 ms. The table above is the steel-era profile.
- Profiled 2026-11-06 through LokiHost with `prof-lokinfer -d iphone-{gpu,cpu}`: inputs >= 1 MB (the KV cache) are held
  on the phone as device tensors and the calls go through RUN2.

## Long-context accuracy and RoPE's inv_freq (2026-11-05)

The long teacher-forced checks (300 s, 20 min) first failed the "≤ 2× torch f16's error" bound on **both** the CPU and
the GPU optimized kernels with near-identical numbers — two independent implementations, so not a kernel bug. A
per-layer comparison at the 20 min clip's token 1323 (17k context) against a fresh f32 reference showed the opposite
of the test's verdict: **lokinfer is closer to f32 than torch f16 at every layer** (residual-stream RMS error 0.2–0.7×
torch's; logits 1.5e-3 vs 3.2e-3).

Cause: `module.to(torch.float16)` also rounds the model's float **buffers**, including the LM's RoPE `inv_freq`
(computed in f32 at init). Its small values are f16 subnormals: relative error up to 8.7e-3, i.e. RoPE angles off by
up to **3 rad at position 17,000** (0.001 rad with the exact table). The f16-weights golden (G-f16w) and the torch f16
calibration reference were both built with `.to(float16)`, so they shared this rounding — torch looked accurate against
G-f16w, lokinfer (exact `inv_freq`) did not. At short positions the angle error is tiny, so only long contexts showed it.

Fix (references, not lokinfer): `reference.load_model` rounds / casts only the **parameters**; buffers stay exact (f32)
by default; `app_buffers=True` reproduces the app's rounding. G-f16w and the torch f16 references were recaptured.

**For english-practice:** its Python path (`AutoModelForMultimodalLM.from_pretrained(dtype=float16).to("mps")`) has the
rounded `inv_freq` — RoPE angle error ≈ 0.3 rad at the ~1,800-token context of the 1:51 reading, ≈ 3 rad at 17k. A
lokinfer port does not have it.
