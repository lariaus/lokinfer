# Kokoro-82M on lokinfer

Kokoro-82M is a text-to-speech model (StyleTTS2-style prosody + an iSTFTNet vocoder). lokinfer runs its forward:
phoneme ids + a voice style row + speed → 24 kHz audio, exactly the part english-practice still runs in PyTorch
(`kokoro_model.infer`). The text frontend (G2P, chunking) is not part of the model and stays in english-practice's
C++ frontend.

Status (2026-11-01): correct end to end on three paths, all judged against the same references:
- `build(fused=False)`: CPU, basic kernels (the M1 reference graph);
- `build()`: CPU with the M3 fused generator/decoder kernels (default) — beats PyTorch CPU ≈ 2×, Mac and iPhone;
- `build(gpu=True)`: M4 — ALBERT, decoder and generator on the GPU (channel-last steel convs, MPS GEMMs), the rest
  of the token-rate front end on the CPU, explicit `to_gpu` / `to_cpu` boundaries, CPU/GPU overlap via async commits
  — beats MLX. Numbers and the full architecture study: `impl-notes/research-kokoro.md`;
plan and decisions: `impl-notes/kokoro-port-plan.md`.

## Using it

```python
from models.kokoro import model as K, reference as R
weights = K.fold_weights(R.load_model())   # weight norm folded with torch's own _weight_norm
model = K.build(weights)                     # shipped variant
audio = model(input_ids, ref_s, speed, seed) # input_ids i64[T] = [0, *phoneme_ids, 0], ref_s f32[256], speed f32, seed i64
```
- `ref_s` is the voice pack row `pack[len(ids) - 1]` (as kokoro's `KPipeline` does).
- The SineGen noise is generated inside the model from `seed` (Philox4x32-10 + Box-Muller), like PyTorch's
  `randn_like` after `manual_seed`: deterministic per seed, not PyTorch's bit stream (bit-matching it isn't needed:
  two PyTorch seeds give audio with ~0.995 correlation, and lokinfer's audio correlates 0.992–0.997 with PyTorch's).
- Variants for testing: `explicit_noise=True` takes the reference noise `f32[600·F, 9]` instead of a seed (parity
  with PyTorch sample by sample); `forced_durations=True` also replaces `speed` with the reference durations.
- `fused=False` builds the basic-kernel graph; `gpu=True` the M4 GPU graph (`fused` then only affects the CPU
  parts that stay on the CPU).
- `outputs=(...)` returns named intermediates: `bert_dur, d_en, d, lstm_x, dur_logits, pred_dur, en, shared,
  F0_pred, N_pred, t_en, asr, dec_x, har_source, har, post, audio`.

## Architecture → graph

| part | PyTorch | lokinfer graph (basic kernels, `models/kokoro/model.py`) |
|---|---|---|
| PL-BERT (ALBERT, 12 layers sharing weights) | `model.bert` | embedding + position view + token-type row, LayerNorm, per layer: one fused QKV GEMM, attention (12×64), GEMM, residual LayerNorm, GEMM + gelu_tanh, GEMM, residual LayerNorm |
| bert_encoder | Linear 768→512 | GEMM |
| DurationEncoder | 3 × (BiLSTM 640→2×256, AdaLayerNorm, concat style) | `lstm_bidir`, style GEMV, `adalayernorm`, `concat_cols` |
| duration head | BiLSTM, Linear 512→50, sigmoid-sum-round | `lstm_bidir`, GEMM, `durations` → `dur i64[T]`, `total` |
| alignment | one-hot matmul `[T, F]` | `dim.read total → F`, `repeat_interleave` (an exact gather) |
| F0 / N predictor | shared BiLSTM over F, 3 AdainResBlk1d each, 1×1 conv | `lstm_bidir`, `adain1d`, `leaky_relu`, `conv1d`, depthwise `conv_transpose1d_dw`, `upsample_nearest1d` |
| TextEncoder | embedding, 3 × (conv k5, channel LN, leaky), BiLSTM | `embedding`, `conv1d`, transposes + `layernorm`, `lstm_bidir` |
| decoder | F0/N stride-2 convs, encode + 4 decode AdainResBlk1d | `conv1d` (stride 2), `concat_rows`, the same AdainResBlk1d chain |
| harmonic source | SineGen (×300 upsample, 9 harmonics, cumsum phase, noise) + Linear 9→1 + tanh | `harmonic_source_rng` (or `harmonic_source` with explicit noise) |
| STFT | `torch.stft` n_fft 20 hop 5 | `stft_magphase` |
| generator | 2 upsample stages: ConvTranspose, noise conv + AdaINResBlock1, 3 resblocks (k 3/7/11, dilations 1/3/5) | `conv_transpose1d`, `conv1d`, `adain1d` + `snake` + `conv1d` chains, `add3_scale` |
| post | leaky, conv_post k7, exp/sin, iSTFT | `leaky_relu`, `conv1d`, `istft_logits` |

Fused CPU path (`build()` default, details in impl-notes/iphone-cpu-sme2.md): Winograd convs with AdaIN + Snake / leaky
prologues; the TextEncoder BiLSTM runs
concurrently with the first duration BiLSTM (`lstm_bidir2`); polyphase ConvTranspose1d with the leaky prologue, the
previous stage's resblock mean and the `x_source` epilogue; conv_post as a Winograd conv on the resblock mean + leaky.

Layouts: token-rate tensors are `[T, C]`; frame-rate tensors channel-first `[C, L]` (PyTorch NCL with batch 1).
The one data-dependent dim is `F` (frames, 40 Hz): every frame-rate shape is `F`, `2F`, `20F`, `120F + 1` or `600F`.
The compiler proves conv output lengths such as `(2F − 1)//2 + 1` equal to `F`.

## Representative shapes

Kokoro's cost is set by two dims: `T` (phoneme tokens incl. the two pads, ≤ 512) and the data-dependent `F` (frames
at 40 Hz = audio seconds × 40), which depends on the text, the voice and the speed. Measured on the real workload
(2026-11-01): the app's 1000-sentence TTS pack (sentences of 5-10 words):

| | min | p5 | median | p95 | max |
|---|---|---|---|---|---|
| T | 34 | 47 | 59 | 71 | 83 |
| F | 98 | 128 | 161 | 191 | 224 |
| audio | 2.5 s | 3.2 s | 4.0 s | 4.8 s | 5.6 s |

The app speaks with two voices, `af_heart` and `am_echo` (random per sentence); `am_echo` yields 8-16 % more frames
for the same text. Over all 28 English voices F varies ~1.6× for one sentence (`af_nicole` the slowest).

Cases (`models/kokoro/reference.py`), used by the tests, the dashboard and `prof-lokinfer` (default `app59f`):
- **workload**: `app47{f,m}`, `app59{f,m}`, `app71{f,m}` — real sentences at the p5 / median / p95 token length, in
  both app voices (f = af_heart, m = am_echo); F = 118/137, 147/163, 174/188;
- **longer utterances** (the user's shadowing is moving toward 5-10 s, plus occasional long-text TTS of 20-60 s+):
  `s6{f,m}` (T = 92, F ≈ 238, 6 s), `s9{f,m}` (T = 141, F ≈ 325, 8 s), `s20f` (T = 334, F = 794, 20 s; `s20m` for
  correctness), `s28f` (T = 487, F = 1048, 26 s: a real-text near-max chunk). Texts longer than one forward's
  510-token context are chunked by KPipeline into several forwards, so 40-60 s of TTS = 2-3 forwards of the s20-s28
  kind back to back;
- **edges**: `word` (one word, T = 9, F = 55: fixed costs), `real28` (T = 31), `real117` (T = 126), `rand510`
  (T = 512, F = 1101: max context with random ids, convolution-bound);
- **correctness only**: the median sentence in `af_nicole` (F = 209), `bf_emma` (British G2P), `am_fenrir`; speeds
  0.8 / 1.25; random-id cases `rand1` … `rand256`.

Optimization priority follows the workload: the `app*` cases first. GPU build layout (2026-11-01): ALBERT on the
GPU; DurationEncoder / duration / shared LSTMs and the TextEncoder on the CPU; F0/N predictor on the GPU,
overlapped with the CPU TextEncoder; decoder + generator + iSTFT on the GPU (harmonic source, STFT and the noise
convs on the CPU, overlapped with the decoder). At these sizes fixed costs (front end, LSTMs,
GPU launch/sync latency, small grids) weigh far more than at `rand510`, where the convolutions dominate.

Generator resblock convs on the GPU (built 2026-11-01): **Winograd** — F(4,3) for K = 3, a nested F(4,4) (tap groups
at offsets 0/4/8 sharing one transformed input) for K = 7 / 11, dilation by decimation, with the AdaIN + Snake
activation fused into the input transform (`ops.conv1d_wino_lc_f32_gpu`, `_snake_conv_gpu` in the model;
`LOKINFER_KOKORO_WINO=0` = the old snake + steel conv pair). ~0.5x the multiplications; per conv 0.87-1.64x on the
iPhone and 0.91-1.41x on the M2 Max vs the snake + steel conv pair (the losses: K = 3 at word-length inputs); error on real activations ≈ the steel conv's (rel-RMS 6e-7) and audio SNR unchanged
to 0.1 dB on every case. Also Winograd: the decoder / F0-N AdainResBlk1d K = 3 convs (AdaIN + leaky fused into the
transform; `LOKINFER_KOKORO_WINO_DEC=0` = steel) and conv_post (leaky 0.01 fused); the K = 3 / 11 generator convs
compute their output's channel stats in the GEMM epilogue (`wino_adain_affine_lc_f32_gpu`), replacing the separate
stats pass. Details, what was tried and the E2E numbers: `impl-notes/winograd-conv.md`. The 1x1 shortcuts stay on
steel.

## Correctness (`tests/models/test_kokoro.py`)

Reference: `models/kokoro/reference.py` replays english-practice's `infer()` stage by stage — bit-exact with it
(max diff 0.0) — on 26 cases (T = 3 … 512, seven voices incl. both app voices, speeds 0.8/1.0/1.25; the 10 workload /
voice cases above were added 2026-11-01: exact durations and audio SNR within 3 dB of PyTorch f32 on CPU and GPU,
per-stage checks on `app59f` / `app59m`), capturing every intermediate and the
noise. Results with the basic kernels:
- token-rate stages and the decoder: relative error ≤ 5e-5;
- durations: **exact** on all 9 cases (so the output length matches);
- harmonic source / spectrum: compared to a **float64 golden** of the same forward (same durations, same noise),
  because f32 phase accumulation reaches ~5·10⁵ rad: lokinfer's f32 audio is as close to the golden as PyTorch's own
  f32 audio (e.g. real117 34.2 dB vs 33.5 dB; rand510 26.8 vs 26.8);
- the spectrum's phase is compared circularly (the `angle()` branch cut flips ±π where the imaginary part is ~0).

## Performance

Baselines (M2 Max, Low Power Mode off, 2026-11-01, `bench/models/bench_kokoro.py`), forward only:

| case | T → F | PyTorch CPU | PyTorch MPS | lokinfer CPU, basic kernels |
|---|---|---|---|---|
| real28 | 31 → 95 | 252 ms | 130 ms | 271 ms |
| real117 | 126 → 368 | 799 ms | 351 ms | 1 015 ms |
| rand510 | 512 → 1101 | 2 280 ms | 1 062 ms | 3 208 ms |

The basic kernels are the reference layer, not the fast path. 92 % of the work is the generator's 1-D convolutions;
the fused path (one conv kernel with the AdaIN/Snake prologue and normalization statistics in the epilogue,
multi-stream Accelerate GEMMs, gather-before-linear alignment, fused LSTMs) is milestone M3, after the conv proof
(M2) measures what each device can reach. MLX (GPU) is the other baseline to beat on short inputs (re-measure).
