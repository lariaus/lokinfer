# Kokoro on the iPhone CPU: SME2 + Winograd (2026-11-01/02)

Status: **built** (kernels, model integration, tests). Device: iPhone 18 Pro Max (iPhone19,3; 2 P + 4 E cores
per `hw.perflevel*`, FEAT_SME2 per `hw.optional.arm`, one SME unit per cluster as measured below). Files:
`kernels/cpu/sme2_conv.{h,cpp}` (runtime check + LSTM `gemvT`; the conv GEMMs moved onto the shared engine
`kernels/cpu/sme2_gemm.{h,cpp}` on 2026-11-02), `kernels/cpu/kokoro_conv_fused.cpp`
(`conv1d_wino_f32`), `kernels/cpu/kokoro_lstm.cpp` (`lstm_bidir_t`), `python/lokinfer/ops/kokoro.py`
(`wino_weights_cpu`, `conv1d_wino_f32_cpu`, `lstm_bidir_t_f32_cpu`), `models/kokoro/model.py` (CPU path),
tests `tests/unit/test_sme2_conv.cpp` (runs on the phone in the deploy suite), `tests/kernels/test_kokoro_conv_fused.py`.

## What was measured first (prof-lokinfer + an A/B interleaved microbench on the phone)

app59f before: 164 ms (thermal fair; 153 ms nominal the next session), snake convs 125 ms at 1.40 TF/s.

Stage-1 conv (Ci = Co = 128, L = 17641, K = 11, d = 5), fused direct conv (im2col + single-threaded Accelerate):

| variant | ms |
|---|---|
| baseline, 6 streams | 3.96 |
| no GEMM (prologue + im2col + epilogue only) | 0.90 |
| no snake / no im2col | 3.48 / 3.03 |
| custom SME2 tile kernel (2x2 ZA tiles, B rows read in place, no im2col), 6 streams | 3.39 |
| SME2, 1 stream / 2 streams | 4.22 / 3.72 |
| SME2 FMOPA-only (no loads), 1 stream, no prologue | 3.14 (≈ 2.0 TF/s) |

Findings: one P-core stream already saturates the **P-cluster SME unit (≈ 2.0 TF/s f32)**; loads are free with 2x2
blocking; the second P core adds nothing to the GEMM; the E-cluster unit adds ~10 %. Per-tile probe
(`pthread_cpu_number_np`): cpus 4-5 are the P cores (tile GEMM ~190 µs with both P cores in SME), cpus 0-3 the
E cores, whose SME unit runs the same GEMM **~9x slower (~1.75 ms)**. The non-GEMM work (snake, transforms,
epilogue) runs on the cores and does not overlap well with the shared unit.

## What was built

1. **SME2 tile kernel** (`sme2::convTile`): out[co, t] = Σ W[co, c, kk] · xw[c, kk·d + t] as FMOPA outer products
   into the four 16x16 ZA tiles (32 co x 32 t blocks), weights pre-packed `[co/32][Ci·K][32]`, B rows read straight
   from the window (no im2col). Runtime-checked (`hw.optional.arm.FEAT_SME2`); the file is compiled with
   `-march=armv8.6-a+sme2` (no `+sve`: Apple CPUs have no non-streaming SVE).
2. **Winograd fused conv on the CPU** (`conv1d_{snake,leaky,affine}_wino[_res]_f32_cpu`), the GPU agent's
   decomposition (docs/impl-notes/winograd-conv.md): F(4,3) for K = 3, nested F(4,4) (7 points, tap groups sharing one
   transformed input) for K = 7/11, dilation via residue classes. Per time tile: AdaIN affine + act window → input
   transform V (compile-time tables, zero / ±1 terms folded; NEON deinterleave for d = 3 via vld3 and d = 5 via
   table lookups) → per point one GEMM (SME2 with the tap-group shift as a B-row offset; Accelerate on a copy
   elsewhere) → output transform + the usual bias / residual / stats epilogue.
3. **P/E pipeline** (SME2 + two clusters): the GEMM stage runs on the P cores, the transforms + epilogue mostly on the
   E cores, through a ring of ≤ 16 tile buffers (≤ 16 MB); an E core takes a GEMM only if one sat unclaimed for 2 ms.
   Stage-1 K = 11 d = 1: 2.20 → 1.73 ms; K = 3: 1.12 → 0.90; K = 7 d = 3: 2.01 → 1.47; rand510 K = 11 d = 5:
   18.3 → 14.0 ms. Ring depth 4 was slower than no split; 8-24 within 5 %.
4. **Short sequences** (fewer than 2 tiles per thread): one tile in two parallel phases (V over channel blocks, then
   per 32 output channels GEMM + output transform + epilogue). Stage 0 at app59f (Ci 256, L 2940, K 11): 1.36 ms
   vs 1.51 pipelined, 2.40 direct. Used for the decoder too (L = F).
5. **Decoder** AdainResBlk1d K = 3 convs: leaky / affine Winograd, **decoder only**: the F0/N predictor blocks stay
   direct because Winograd there costs ~1 dB of audio SNR (the F0 curve feeds the harmonic-source phase; app47m 37.8
   → 36.9 dB vs the 37.5 dB test limit). Decoder-only: SNR identical to the direct path to 0.01 dB.
   `LOKINFER_KOKORO_WINO_CPU_DEC` = 0 off / 1 all / 2 decoder only (default).
6. **LSTM with w_hh^T** (`lstm_bidir_t_f32_cpu`): the step GEMV runs on SME2 with all 4H = 1024 gates in ZA
   (multi-vector FMLA per hidden unit); elsewhere sgemv on the transposed matrix, which is also faster on the Mac.
   T = 59: 0.51 → 0.39 ms iPhone (0.74 → 0.52 Mac); T = 161: 1.22 → 0.90 (1.66 → 1.05). **Built but off by default**
   (`LOKINFER_KOKORO_LSTM_T=1` opts in): the reordered sums move app47m's audio SNR 37.8 → 36.1 dB vs the f64 golden
   (test limit 37.5; word 32.5 → 32.6) — an ulp-level chaotic effect through F0 / the source phase, not a bug.

A/B switches: `LOKINFER_KOKORO_WINO_CPU=0` (direct fused convs), `LOKINFER_KOKORO_WINO_CPU_DEC`, `LOKINFER_KOKORO_LSTM_T`.

## Results

iPhone CPU app59f (prof-lokinfer, nominal, same day): **153 → 96.8 ms** (164 ms in the original profile; ExecuTorch
288.9). Dashboard sweep 2026-11-02 (`bench_remote_models.py kokoro --cpu-only`, nominal at start, one sweep; it ran
with LSTM_T still on, ~1 ms): word 47.1, real28 68.4, app47f/m 85.4/98.6, app59f/m 106.1/117.2, app71f/m 128.5/139.9,
s6f/m 174.6/182.5 (was 259.1), s9f/m 261.6/302.2 (was 381.0), real117 333.8, s20f 718.4 (was 1169.8), s28f 951.8
(was ~1546 hot; ExecuTorch 2619), rand510 1001.8 ms (was 1230).

Mac CPU (prof-lokinfer, Winograd via Accelerate, no SME2 on the M2 Max): app59f 152.0 → 122.3 ms, s9f 291.7 → 256.9 ms.

## Tried and dropped

- SME2 GEMM for ALBERT (59 x 768 x 2304 etc.): 0.19 vs Accelerate 0.12 ms standalone — Accelerate is already at the
  SME peak there. In-model the same GEMMs cost ~2x their standalone time (gemm.cpp threading / park; not this note's
  code) — open item.
- Direct SME2 conv with per-call weight packing: fine for stage 1 (3.39 vs 3.96 ms) but packing 12 MB of decoder
  weights per call made the decoder conv 3x slower; superseded by Winograd with pre-packed U.
- Larger Winograd tiles (1536-2048) and smaller ring depths (2-4).

## Round 3 (2026-11-02, Kokoro CPU agent; Mac only — the phone was unavailable)

Status: **built**, tested on the Mac (M2 Max, no SME2) and on the phone once it came back (deploy suite: all 70
on-device test cases pass, incl. the `[sme2]` engine conv test with the new 64 × 16 tail and the gemvT bit-exactness
test). Mac CPU prof-lokinfer: **app59f 122.3 → 106.6 ms, s9f 256.9 → 230.1 ms** (with the ALBERT change below, since
reverted: +0.4 ms). iPhone CPU: the phone ran ~15 % slower on untouched kernels than in the morning session, so
cross-session totals are not comparable (app59f 100.4 before vs 98-101 after); per kernel, same session conditions or
better: ConvTranspose 4.49 → 3.14 ms (with the resblock mean folded in), harmonic source 2.71 → 0.78, STFT + iSTFT
1.89 → 0.58, conv_post 1.68 → 0.80 (with the mean), `add3_scale` 0.49 → 0.

Dashboard (2026-11-02, end of round): **Mac CPU** (`bench_kokoro.py`, LPM off) word 64, app59f 109, s9f 230, real117 253,
s28f 645, rand510 674 ms (torch eager CPU 167 / 358 / 728 / 814 / 2179 / 2390); **iPhone CPU** (`bench_remote_models.py
kokoro --cpu-only`, one sweep, nominal → serious) word 42.8, app59f 107.9, s28f 911.9 (was 951.8), rand510 972.5 (was
1001.8) ms. `tests/perf/test_dashboard_goals.py` passes.

| change | where | Mac measurement |
|---|---|---|
| Polyphase ConvTranspose1d `conv_transpose1d_poly_f32_cpu`: rows (phase r, co) grouped by q_r = (r + p) div s, each group one GEMM over the shared leaky window (taps = K / s = 2 shifted reads of B, no im2col / col2im), interleave + bias + `x_source` (+ ups[1]'s reflect pad) in the epilogue; SME2: `sme2::gemm` with the conv B operand on the P cores only | `kokoro_conv.cpp`, `ops.convt_poly_weights_cpu` | ups[0] + ups[1] incl. the leaky / pad / add passes: 1.47 + 3.98 → 0.65 + 1.24 ms |
| conv_post as a nested-F(4,4) Winograd conv with the leaky prologue (identity affine) | `models/kokoro/model.py` | 2.56 → 0.50 ms (direct fused conv: 0.67) |
| The generator's 3-resblock mean fused into its two consumers (`conv_transpose1d_poly_mean3`, `conv1d_leaky_wino_mean3`): `add3_scale` gone, bit-identical | both | −0.67 ms (`add3_scale` at app59f) |
| Winograd GEMMs without SME2 (Accelerate): B read in place with one accumulating sgemm per tap group when G ≤ 2 (K = 3, 7) instead of copying V | `kokoro_conv_fused.cpp` | K = 3 stage 1 0.89 → 0.77 ms, decoder 1024 × 147 0.52 → 0.34, K = 7 1.16 → 0.93; K = 11 (G = 3) slower that way (1.69 → 2.05), keeps the copy |
| Harmonic source vectorized: NEON Philox4x32-10 (4 samples per vector), Box-Muller with `log4` / `sin4` / `cos4`, sine by `sin4Wide` (π reduction in double: phases reach ~1e6 rad), `tanh4` | `kokoro_signal.cpp` | 0.63 ms at app59f (iPhone before: 2.71) |
| STFT: 4 frames per NEON step from hop-residue rows of the reflect-padded signal, `atan2_4`; iSTFT: 4 frames per step (`exp4` / `sin4Wide` / `cos4`), overlap-add as a per-sample gather in frame order on the pool (same summation order as the serial loop) | `kokoro_signal.cpp` | 0.36 / 0.54 ms at app59f (iPhone before 0.98 / 0.91) |
| Depthwise ConvTranspose1d as a per-output gather (same accumulation order, bit-identical SNR), channels on the pool | `kokoro_conv.cpp` | 0.75 ms at app59f before |
| sme2_gemm N-tail block: a last column chunk with nt ≤ 16 runs as 64 × 16 blocks (four ZA tiles stacked along M) | `sme2_gemm.cpp` | iPhone interleaved A/B, nominal: app59f 98.0 / 99.6 vs 100.3 / 101.2 ms without; s9f 208.4 / 213.3 vs 213.5 / 213.2 |
| ~~ALBERT on GPT-2's fused CPU ops~~ (`gemm_accel_f32_cpu` bias + gelu_vv epilogue, `add_bias_layernorm2_f32_cpu`, bit-identical) — **reverted** | | Mac ALBERT 8.85 → 8.41 ms, but iPhone +1.4-2.8 ms (interleaved A/B, 3 rounds: every GEMM shape slower under gemm_accel's auto split) |
| TextEncoder BiLSTM concurrent with the first duration-encoder BiLSTM (`lstm_bidir2_f32_cpu`, 4 recurrences on 4 cores; CPU build only, sequential when the CPU has < 4 P cores) | `kokoro_lstm.cpp`, `models/kokoro/model.py` | Mac: two T = 59 BiLSTMs 1.50 → 1.39 ms (microbench), in-model −0.7 ms; iPhone (sequential fallback): neutral (LSTM total 4.26-4.75 vs 4.52-5.23 ms unpaired, interleaved) |
| Accurate transposed LSTM GEMV: 4 partial sums per gate (j mod 4), `(p0 + p1) + (p2 + p3)`, added to the input projection; `sme2::gemvT` computes the same operations (bit-identical, unit-tested on SME2 CPUs) | `kokoro_lstm.cpp`, `sme2_conv.cpp` | with `LOKINFER_KOKORO_LSTM_T=1` all 26 free-running cases pass (app47m 37.9 dB, limit 37.5; one accumulator: 36.1, 2 partials: 36.0). Mac: the NEON GEMV is slower than the default path (T = 59 1.46 vs 0.74 ms). iPhone: bit-exactness test passes on the device; app59f 98.4 / 101.0 ms vs 98.0 / 99.6 default (interleaved) — no gain, so it stays opt-in on both devices |
| `kernels/cpu/neon_math.h`: `exp4`, `log4`, `sin4` / `cos4` / `sinSq4`, `sin4Wide`, `tanh4`, `atan2_4` with measured error bounds (unit test "neon_math") | | |
| sme2_conv's conv GEMM merged into the shared engine: `sme2::gemm(wp, co, ci·k, {xw, win, 32, k, d}, n, o, ldo)`; `sme2_conv` keeps `available()` and the LSTM `gemvT` | `sme2_conv.*`, `sme2_gemm.*` | iPhone interleaved A/B vs the old tile kernel: app59f 98.0 / 99.6 vs 98.7 / 102.7 ms, s9f 208.4 / 213.3 vs 211.4 / 218.7 (nominal) |

ALBERT in-model slowdown, measured (Mac, per-kernel in-graph profile): the four GEMMs cost 182 / 97 / 194 / 137 µs
in the graph vs 143 / 66 / 146 / 118 µs alone (≈ 1.3×, not 2×). Causes: (1) a pool kernel between two multithreaded
Accelerate GEMMs (GELU, attention) costs the next GEMM ~10 % (an isolated FFN chain: 138 → 151 µs with a pool GELU in
between, unchanged with a single-threaded kernel in between); (2) the standalone numbers keep one 6-9 MB weight hot,
the model cycles 21 MB of ALBERT weights per layer. 8 column blocks per GEMM (single-threaded Accelerate on the pool)
gave 8.04 ms vs 8.41 (auto) on the Mac but that choice belongs to gemm.cpp's per-platform auto split (iPhone: 2 row
blocks) — reported to its owner, not hard-coded here. The SME2-for-ALBERT verdict needs the phone.

Mac Winograd conv (stage 1 K = 11, `sample`): ~52 % of samples in Accelerate, ~40 % in the transforms, ~6 % in the
V copy, ~7 % idle; 8 streams in flight (35 tiles over the 8-thread pool). Full-M sgemm (repacked plain U) and tiles
256 / 1024 / 2048 were not faster; the remaining lever on the Mac is the transform cost, not the stream count.

Not done (measured too small): fusing harmonic source + STFT into one kernel (saves one 350 KB round trip and one
pool hand-off, < 0.05 ms after vectorization) and iSTFT into conv_post's epilogue (≈ 0.1 ms of a 1.5 MB round trip);
gather-by-index / nearest ×2 / virtual concat input transforms (`repeat_interleave` 0.01 ms, `upsample_nearest1d`
0.07 ms, `concat_rows` 0.08 ms at app59f).

## Open

- `prof-lokinfer`'s TF/s column counts Winograd ops by their U size (overstates them ~2x).
- ALBERT GEMMs: gemm.cpp's auto split (Mac: 8 column blocks measured 8.04 vs 8.41 ms) — gemm.cpp owner.
