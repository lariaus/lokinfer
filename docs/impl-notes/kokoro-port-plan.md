# Kokoro-82M port plan

Status: **v2** (2026-11-01, after `plan-review-2026-11-01.md`). Facts about the model are in
`research-kokoro.md` (architecture §1, ops §2, numerical traps §2.5, profile §4, baselines §5, fusion §6).
This doc fixes *how* lokinfer implements it, step by step, with exit criteria. The milestone numbers (M0–M5)
match `docs/todo.md`.

## Targets

All re-measured with Low Power Mode off (2026-11-01). f32, M2 Max:

| case | torch eager CPU (must beat) | best GPU baseline (must beat) | estimated floor CPU / GPU |
|---|---|---|---|
| real117 (T=126 → F=368) | **799 ms** eager (LPM off; was 850) | **193 ms MLX** (LPM off); 351 ms torch MPS | — |
| rand510 (T=512 → F=1101) | **2067 ms** torch.compile CPU, 2280 ms eager (LPM off) | **858 ms MLX** (LPM off); 877 torch.compile MPS; 1062 torch MPS | CPU convs ≈0.4–0.5 s measured (M2) / ≈0.35 s |

iPhone 18 Pro Max (user decision): an **artificial target** = the Mac "beat PyTorch" result scaled by the ratio of
measured device peaks (GEMM/conv TFLOP/s, bandwidth, dispatch cost; measured in M2), per device class; the goal
is to beat it. The published CoreML/MLX-Swift numbers are a sanity check only.
**Measured 2026-11-01 (m2-conv-proof.md): conv ratio CPU 0.86, GPU 0.41 → iPhone targets CPU ≤ 2400 ms (rand510) /
≤ 930 ms (real117), GPU ≤ 2090 ms / ≤ 470 ms.**

**M3 status (2026-11-01, LPM off):** generator resblocks fused (`conv1d_snake[_res]` + chunk stats +
`adain_affine`) plus profile-driven fixes (attention, GELU, harmonic source, BLAS threading):

| | rand510 | real117 | real28 |
|---|---|---|---|
| M2 Max CPU, lokinfer | **1064 ms** | **352 ms** | 112 ms |
| M2 Max CPU, torch eager (same run) | 2275 ms | 791 ms | 245 ms |
| iPhone 18 Pro Max CPU, lokinfer | **1230 ms** | **397 ms** | 110 ms |
| iPhone target | ≤ 2400 ms | ≤ 930 ms | — |

**Both CPU bars are beaten** (Mac: 2067 ms torch.compile / 799 ms eager).

**Apples-to-apples baselines (2026-11-01, `bench/models/bench_kokoro_baselines.py`):** every baseline now runs on
exactly the reference cases (same ids incl. padding, voice row, speed; v1.0 weights, f32) and its output length is
checked against the reference (same durations ⇒ same work). The earlier MLX bars (193 / 858 ms) came from the
research input file, whose "real117" is T = 119 (not 126): on the exact cases MLX takes **real28 74 ms, real117
263 ms, rand510 839 ms** (rand510 with F = 1100 vs 1101 from MLX's own durations, −0.09 % work; noted on the entry).
The dashboard holds the current numbers.

**M4 status (2026-11-01, measured in LPM on battery — re-measure on AC):** `build(gpu=True)` runs the decoder +
generator on the GPU (channel-last steel convs at ≈ 6 TF), the token-rate front end on the CPU, with async commits
so CPU and GPU overlap: **real117 178 ms (MLX 193), rand510 484 ms (MLX 858), real28 67 ms** — both GPU bars beaten.
Results are on the dashboard (`scripts/perf_dashboard.sh`, docs/perf-dashboard.md). Remaining rand510 profile: fused convs
≈ 600 ms, LSTM 111, basic conv1d 98 (decoder AdainResBlk), GEMM 62, convT 41, STFT 35, iSTFT 26.

## Where the time is (why the plan looks like this)

92 % of MACs and ~75 % of CPU time are the generator's 1-D convolutions (k3/7/11, dilations 1/3/5) at 256 ch × 20F
and 128 ch × 120F. The plan therefore de-risks that one kernel family before anything else is fused (M2), and
treats everything else as second order. The two levers the research found and the plan now centers on:
- **CPU (measured in M2, `m2-conv-proof.md`):** time-tiled im2col (512-sample, L2-resident column blocks, inner
  dimension Ci·K) over 8 single-threaded Accelerate streams reaches 2.0 TFLOP/s on the heavy stage-1 convs, 4.2× the
  basic kernel and ~3× what PyTorch gets. The fused prologue is applied while building each column block.
- **GPU:** fusion to ~2 memory passes per conv and ~170 dispatches per forward; on the iPhone, f16 tensor ops on the
  A20's neural accelerators (on the M2 Max f16 brings no compute gain, so the Mac path stays f32).

## Steps (the original plan's 11, mapped to milestones)

1. [x] Reproduce: `kokoro==0.7.16` in lokinfer's venv, the same HF snapshot as english-practice.
2. [x] Reference data: `models/kokoro/reference.py` replicates `infer()` stage by stage, bit-exact vs
   english-practice (max diff 0.0 on two cases), captures every intermediate plus the SineGen noise. 9 cases in
   `~/.cache/lokinfer/kokoro/ref/`: real28, real117, real_speed125, real_speed08, rand1, rand14, rand128, rand256, rand510.
3. [x] Study: `research-kokoro.md`.
4. [x] Shapes: T ∈ {3, 16, 30, 45, 126, 130, 258, 512}, speeds 0.8/1.0/1.25, voices af_heart/am_michael/bf_emma
   (the reference cases). Edge cases to add in tests: T=512 exactly (position table bound), F forced to 1 per token.
5. [x] Basic kernels (M1): 26 CPU f32 kernels (incl. the in-kernel-RNG harmonic source) written and unit-tested (`kernels/cpu/kokoro_*.cpp`,
   `tests/kernels/test_kokoro_kernels.py`, 151 tests). They are the **reference oracle** for the fused kernels and
   the on-device reference; keep them simple. Known gaps: `durations` reduction order, `repeat_interleave`
   silent truncation, scalar `snake` (113 ms per stage-1 call; needs the NEON cos form even as a basic kernel).
6. [x] E2E basic graph (M1, done 2026-11-01): `tests/models/test_kokoro.py` green on all 9 cases; stages ≤ 5e-5
   rel (token rate, decoder); durations exact; audio as close to a float64 golden as PyTorch f32 itself; shipped
   seed variant correlates 0.992–0.997 with the reference. Unfused CPU time: real117 1015 ms, rand510 3208 ms
   (PyTorch CPU 799 / 2280 ms, LPM off). See `docs/models/kokoro.md`.
7. [ ] Fusion design (M2 → M3): `research-kokoro.md` §6 is the design; M2 fixes the numbers (layout, algorithm,
   thread strategy, library candidates) before the fused layer is written.
8. [ ] Fused kernels, basic versions + tests (M3 CPU, M4 GPU).
9. [ ] Optimize (M3/M4), guided by the per-kernel profiler.
10. [ ] E2E bench vs baselines (CPU and GPU, Mac and iPhone).
11. [ ] Profile and iterate until the targets are beaten.

## Decisions (v2)

- **Noise / RNG (user: "do like PyTorch").** The shipped model generates the SineGen noise inside the
  harmonic-source kernel (counter-based Philox, `nohoist`) from a seed the caller sets, like `torch.manual_seed`.
  The explicit `noise f32[600·F, 9]` input exists only in a parity-test variant (the reference run fixes F).
  Bit-matching torch's RNG stream is not a goal (seed-to-seed correlation 0.995).
- **F.** One data-dependent symbol, defined by `dim.read` of the duration total. On the GPU: one readback,
  overlapped with the F-independent work (TextEncoder, style GEMV) using several in-flight command buffers.
  GPU-driven dispatch with upper-bound allocation is not viable (F ≤ 25 600 ⇒ 1.6 GB activations); it stays an
  experiment for M6. Inputs typed with linear dims (`[600*F, 9]`) must bind `F` at call time (M0 IR fix).
- **Layouts.** Token-rate `[T, C]`; frame-rate channel-first `[C, L]` **for the basic graph**. The fused generator
  layout (`[C, L]` vs time-major `[L, C]`) is decided by M2's measurements, not assumed.
- **Precision.** f32 on the Mac (compute-bound; f16 gives nothing on M2). On the iPhone GPU, f16 inputs/weights
  with f32 accumulation and f32 instance-norm statistics, through Metal 4 tensor ops.
- **Numerics / tests.** Durations must be exact (replicate torch's reduction order; ±1 frame changes the output
  length). The harmonic source and iSTFT are judged against a **float64 numpy golden**, with torch's own f32
  error as the tolerance; audio is compared by SNR to the reference. Forced-durations inputs let the decoder be
  tested independently of duration flips.
- **Weights at load (numpy, once).** Weight-norm `g·v/‖v‖` (ConvTranspose: per *input* channel), LSTM
  `b_ih + b_hh`, Linear → `[in, out]`, `+1` folded into the γ halves of the style projections, `/√2` and `/3`
  folded into conv weights where the consumer is positively homogeneous, Snake `2α` and `1/(2α)`, STFT/iSTFT
  bases and the `1/Σw²` envelope as constants.
- **Threading (user: depends on the graph).** Small kernels run inline, one by one. Big kernels (the frame-rate
  convs, LSTMs over F) use the multithreaded runtime: time tiles across streams, both LSTM directions, and big
  independent kernels (the noise branch, the harmonic source) run async. A per-kernel work estimate and a
  per-device threshold decide which path is taken.

## Graph (basic, M1)

Inputs `input_ids i64[T]` (= `[0, *ids, 0]`), `ref_s f32[256]`, `speed f32` (or `dur i64[T]` when
`forced_durations=True`), `noise f32[600*F, 9]` (tests only). Output `audio f32[600*F]`. Optional named
intermediates for stage tests: `bert_dur, d_en, d, lstm_x, dur_logits, pred_dur, en, shared, F0_pred, N_pred, t_en,
asr, dec_x, har_source, har, post`.

`s_p = ref_s[128:]`, `s_d = ref_s[:128]` are views. `durations_f32_cpu` → `dur i64[T]`, `total i64[1]`;
`dim.read total → F`; `repeat_interleave_f32_cpu(x, dur, frames=F)` replaces the one-hot matmuls (exact).

## Basic CPU kernel set (step 5) — contracts

All f32, contiguous row-major; `[C, L]` is channel-first. ABI: ptrs = operands then outs; ints = every operand's
shape dims, then every out's shape dims, then int/bool/dim attrs in attribute-name order; floats = float attrs in
attribute-name order. (Structural attrs will be separated from this list in M0.)

| kernel | operands → results | attrs | semantics (PyTorch equivalent) |
|---|---|---|---|
| `embedding_f32_cpu` | `table[V,D]`, `ids i64[T]` → `[T,D]` | | `table[ids]` (ids must be < V; check planned) |
| `layernorm_f32_cpu` | `x[R,C]`, `gamma[C]`, `beta[C]` → `[R,C]` | `eps` | `F.layer_norm(x,(C,),gamma,beta,eps)` |
| `adalayernorm_f32_cpu` | `x[R,C]`, `gb[2C]` → `[R,C]` | `eps` | `(1+gb[:C])*LN_noaffine(x)+gb[C:]` |
| `attention_f32_cpu` | `q,k,v [T,H*64]` → `[T,H*64]` | `heads` | non-causal SDPA per head, no mask, scale 1/√64 |
| `gelu_tanh_f32_cpu` | `x` → same | | `F.gelu(x, approximate='tanh')` |
| `leaky_relu_f32_cpu` | `x` → same | `slope` | `F.leaky_relu` |
| `scale_f32_cpu` | `x` → same | `scale` | `x*scale` |
| `add3_scale_f32_cpu` | `a,b,c` → same | `scale` | `(a+b+c)*scale` |
| `concat_rows_f32_cpu` / `concat_cols_f32_cpu` | 2..8 inputs | | `torch.cat(dim=0 / 1)` |
| `broadcast_rows_f32_cpu` | `v[C]` → `[R,C]` | `rows = dim<R>` | `v.expand(R,C)` |
| `lstm_bidir_f32_cpu` | `x[T,I]`, `w_ih[8H,I]`, `w_hh[8H,H]`, `b[8H]` → `[T,2H]` | | 1-layer bidirectional `nn.LSTM`, gates i,f,g,o; rows 0..4H forward, 4H..8H backward; `b = b_ih+b_hh` |
| `durations_f32_cpu` | `x[T,50]`, `speed f32` → `dur i64[T]`, `total i64[1]` | | `clamp(round(sigmoid(x).sum(-1)/speed), min=1)`, torch's reduction order, round half to even |
| `repeat_interleave_f32_cpu` | `x[T,C]`, `dur i64[T]` → `[F,C]` | `frames = dim<F>` | `x.repeat_interleave(dur, 0)`; must fail if `frames ≠ Σdur` |
| `conv1d_f32_cpu` / `conv1d_nobias_f32_cpu` | `x[Ci,L]`, `w[Co,Ci,K]`, (`b[Co]`) → `[Co,Lo]` | `stride, padding, dilation` | `F.conv1d`, Lo = (L+2p−d(K−1)−1)//s+1 |
| `conv_transpose1d_f32_cpu` | `x[Ci,L]`, `w[Ci,Co,K]`, `b[Co]` → `[Co,Lo]` | `stride, padding, output_padding` | `F.conv_transpose1d`, Lo = (L−1)s−2p+K+op |
| `conv_transpose1d_dw_f32_cpu` | `x[C,L]`, `w[C,1,K]`, `b[C]` → `[C,Lo]` | same | `groups=C` |
| `adain1d_f32_cpu` | `x[C,L]`, `gb[2C]` → `[C,L]` | `eps` | `(1+γ)·instance_norm(x)+β`, biased variance |
| `snake_f32_cpu` | `x[C,L]`, `alpha[C]` → same | | `x + (1/α)·sin²(αx)` (vvsinf) |
| `upsample_nearest1d_f32_cpu` | `x[C,L]` → `[C,L·s]` | `scale` | `F.interpolate(mode='nearest')` |
| `pad_reflect1d_f32_cpu` | `x[C,L]` → `[C,L+l+r]` | `left, right` | `F.pad(mode='reflect')` |
| `harmonic_source_f32_cpu` | `f0[2F]`, `noise[600F,9]`, `w[9]`, `b[1]` → `[600F]` | | `SourceModuleHnNSF` + `SineGen` in torch's op order (dead `rand_ini` ignored) |
| `stft_magphase_f32_cpu` | `x[S]` → `[22, S/5+1]` | `n_fft, hop` | `torch.stft(center, reflect, hann periodic)` → `cat(abs, angle)` |
| `istft_logits_f32_cpu` | `x[22, L]` → `[5(L−1)]` | `n_fft, hop` | `exp(x[:11])·e^{i·sin(x[11:])}` → `torch.istft` |

Planned additions: `harmonic_source_rng_f32_cpu` (in-kernel Philox, `nohoist`, `seed` attr) for the
free-running model.

## Fused kernel families (M3/M4)

See `research-kokoro.md` §6.2 for the full list (13 families). The order of work is by measured share of time
(generator chain, then LSTMs, then harmonic source/STFT, then the token-rate front end). Each fused kernel is
tested against the composition of the basic kernels above (shape sweeps) and against the stage references.
