# Qwen3-ASR-1.7B port plan (phase 1: measure the gap)

Status: **v3, approved by the user (2026-11-05)**; implementation started. v2 added the audio-length limits and the dashboard
totals decision; v3 adds the caller contract, the chunked-prefill splice rule, streaming attention (no materialized
score matrices), the long-case test strategy, the memory and disk budget, and the file map. Facts are in
`research-qwen3-asr.md` (§ numbers below refer to it). This doc fixes what lokinfer builds, in which order, file by
file, with exit criteria. The milestones (Q0–Q8) match `docs/todo.md`.

Provenance: **[user]** = the user's decision or requirement; **[derived]** = my inference from the research (loses
to anything the user says).

## Goal and non-goals

- **[user]** Port Qwen3-ASR-1.7B like Kokoro: **only the neural model** (audio encoder, LM prefill, LM one-token
  decode with a KV cache). Feature extraction, prompt / token building, the greedy loop, detokenization and the
  splitting of very long audio stay in the caller (as in english-practice's `qwen_asr_model.py`).
- **[user]** Phase 1 goal is **not to beat anything**: golden reference data, a **basic** lokinfer version and a
  **slightly optimized** one (basic + obvious fusions), baselines on all **4 devices** (Mac CPU, Mac GPU, iPhone CPU,
  iPhone GPU), all on the perf dashboard **as the first model at the top**, so we know how far lokinfer is.
- **[user]** Precision: **f16 on all four devices**, for lokinfer and every baseline. f32 only where the reference
  path itself uses it (norm math, softmax, RoPE tables, matrix accumulations). No f32 model, no bf16 for now; a
  device may get separate bf16 kernels later (no type-generic kernels: lokinfer writes one kernel per dtype).
- **[user]** Benchmarks: encoder per clip length, prefill per clip length, decode step per context length, and
  end-to-end transcription, on real speech of different sizes; clip lengths are a generic spread (no app usage data).
- **[user]** Test audio: public-domain speech (never committed; lokinfer is public). **[user]** Baselines: PyTorch
  eager + torch.compile (Mac CPU), PyTorch MPS + MLX Python (Mac GPU), MLX Swift (iPhone GPU), llama.cpp (all 4);
  ExecuTorch deferred. **[user]** Installs and downloads outside lokinfer (baselines checkout, GGUF, LibriSpeech,
  model uploads to the phone) are approved.
- **[user]** Audio length: up to **30 min on the Mac, 5 min on the iPhone**. **[derived, from Qwen's own `qwen-asr`
  0.0.6]** one model pass takes at most 20 min (`MAX_ASR_INPUT_SECONDS = 1200`); longer audio is split by the caller
  at the quietest point within ±5 s of each boundary (`split_audio_into_chunks`). So lokinfer supports **one pass of
  up to 20 min on the Mac and 5 min on the iPhone**; 30 min = the caller's split into passes.
- **[user]** The dashboard counts Qwen3-ASR's configurations in the top totals.
- Non-goals (phase 1): beating baselines, quantization, streaming, bf16 kernels, speculative decoding, batching,
  integrating into english-practice, splitting long audio inside the model, the Kokoro / GPT-2 "#1 everywhere" guard
  (Qwen3-ASR is **not** a goal model: `GOAL_MODELS` stays Kokoro + GPT-2).

## The caller contract (what stays outside the model)

Written here so the model boundary is exact and so english-practice's agent can use it later. Everything in this
section is reproduced by `models/qwen3_asr/reference.py` (Q0) and checked against the HF processor.

1. **Features** (§2): 16 kHz mono f32 audio → the processor's `input_features f32[1, 128, 100·n]` (Whisper-style
   128-mel log spectrogram, the clip-global max normalization, zero-padded to a multiple of 100 frames; clips < 0.5 s
   are padded to 0.5 s first). The model takes them reshaped as **`features f32[n, 128, 100]`** (n = 1 s chunks).
2. **Audio-token count**: `F = floor(n_samples / 160)` (after the 0.5 s padding), `N_a = 13·floor(F/100) + f(F mod
   100)`, `f(r) = c(c(c(r)))`, `c(x) = (x−1)//2 + 1`, `f(0) = 0`. Passed as **`n_audio i64[1]`**.
3. **Prompt ids**: `[9 prefix ids] + [151676] × N_a + [9 suffix ids]` (§2; `language English<asr_text>` forced as
   the app does), so T = N_a + 18 and the audio slots are positions 9 .. 9 + N_a − 1.
4. **One transcription**: `audio = encoder(features, n_audio)` (once), then prefill in chunks of ≤ 2048 ids: for
   each chunk, `logits = generate(ids_chunk, audio_slice, pos)` where `audio_slice` is the rows of `audio` that the
   chunk's `<|audio_pad|>` ids take, in order (the j-th pad id in the chunk takes row j of the slice; the caller
   slices `audio[k0:k1]` by counting pads before the chunk — trivial since the pads are one contiguous run), and
   `pos` = the number of ids already in the cache. Only the last chunk's logits matter. Then greedy decode:
   `tok = argmax(logits)`, stop on eos `151643` or `151645` or after `max_new_tokens = 128 + 8·seconds` (the app's
   rule), else `logits = generate([tok], audio_dummy, pos)` with `pos += 1` and `audio_dummy` any 1-row f16 tensor
   (no pad id in a decode call, so it is never read). Detokenize with the processor (`<asr_text>` parsing and
   repetition cleanup are the processor's, §4).
5. **Long audio**: > 20 min (Mac) or > 5 min (iPhone) is split by the caller into passes with `qwen-asr`'s rule
   (lowest-energy sample within ±5 s of each boundary), each pass transcribed as above and the texts joined. The
   model has no state across passes except the KV caches, which the caller resets by starting at `pos = 0`.
6. **No batching**: one clip at a time (B = 1 everywhere, like Kokoro and GPT-2).

## Decisions

1. **Model boundary [derived, from the user's "only the model"].** Two compiled lokinfer models per device:
   - **Encoder**: `(features f32[n, 128, 100], n_audio i64[1]) → audio f16[N_a, 2048]`. n is an input symbol; N_a
     becomes a symbol through `dim.read` of `n_audio` (Kokoro's F mechanism) because the attention windows and the
     valid-row prefix depend on it (§3: keep the first N_a rows right after `conv_out` + positions, *before* the 24
     layers). The f32 → f16 cast is the graph's first op (the reference's `.to(dtype)`). The CNN runs **chunk by
     chunk** (a loop of n iterations over a `[1, 128, 100]` view: no `[n, 480, 64, 50]` peak, which would be 7.4 GB
     f16 at n = 1200). Capacity: n ≤ 1200 (Mac) / 300 (iPhone), a build parameter only used for sanity checks (the
     graph is symbolic in n).
   - **Generate** (prefill and decode in one model sharing persistent KV caches, GPT-2's `build_generate` pattern):
     `(ids i64[T], audio f16[A, 2048], pos i64[1]) → logits f32[1, 151936]` of the last row. `embed(ids)`, then the
     rows whose id is `151676` are replaced in order by the rows of `audio` (`scatter_audio_f16`, the chunk rule of
     the contract). T symbolic, T ≤ 2048 per call (chunked prefill, like GPT-2's `attention_cache` path), T = 1 for
     decode. **KV capacity `cap`** (persistent f16 caches `[cap, 1024]` for K and V per layer, allocated at load):
     **25,600 on the Mac** (20 min pass: 15,618 prompt + 9,728 max new = 25,346; 2.94 GB) and **6,656 on the
     iPhone** (5 min: 3,918 + 2,528 = 6,446; 0.76 GB). Logits in f32 (the LM head accumulates in f32; argmax is the
     caller's).
   - Why two models: the encoder runs once per clip and has no state; GPT-2 already proves prefill + decode sharing
     one model. Why the embedding stays inside: the 311 M-param table is also the LM head (tied) — one copy.
2. **Precision layout [user + derived].** Weights f16 (converted once from the bf16 checkpoint at build time; §8:
   exact for all but 0.017 % of values, none overflow), activations and the residual stream f16, KV caches f16. f32
   inside kernels: LayerNorm / RMSNorm / QK-norm statistics (HF's cast order: f16 → f32 → normalize → f16, then the
   weight multiply), softmax, RoPE cos/sin tables (f32 constants built from positions), every GEMM / GEMV /
   attention accumulator. The encoder's f16 clamp (±64504 after each layer, `MOD`:236-238) is reproduced. Inputs:
   features f32 (cast in-graph); outputs: logits f32.
3. **Correctness standard [derived].** The judge is a **float64 golden** run of the HF model on CPU. Two golden
   variants: **G-true** (bf16 checkpoint upcast to f64: the real model) and **G-f16w** (the f16-rounded weights
   upcast: isolates compute error from weight rounding). Stage and logit errors of lokinfer f16 are measured against
   G-f16w and must be **within 2× of the torch f16 reference path's own error** (torch MPS fp16, measured on the
   same stages: this is the calibration, not a fixed epsilon). Transcripts must be **identical to G-true's** greedy
   transcript on every test clip, except at near-ties: a difference is accepted only at a position where G-true's
   top-2 logit gap is < 0.05 (the Q0 gate found exactly such a flip in torch f16: s300 token 48, gap 0.004, "imagination
   and" vs "imagination, and"); the test reports it. Exact token-by-token equality with torch f16 is **not** required. Every model
   test asserts no NaN / inf anywhere.
4. **Cases [user + derived].** Real public-domain speech: **LibriSpeech test-clean** (CC BY 4.0, read audiobooks;
   downloaded at first use into `~/.cache/lokinfer/qwen3asr/librispeech/`, never committed). Clips of **2, 5, 10, 30,
   60, 111, 300 s**, plus **1200 s** and a **30 min** clip (Mac only), built by concatenating consecutive utterances
   of one speaker (one chapter when it is long enough, else consecutive chapters of the same speaker) and trimming
   to the length at a pause; the utterance lists are fixed and committed (`clips.json`). Dashboard cases per device:
   - `enc 2s … enc 300s` (+ `enc 1200s`, Mac): the encoder forward;
   - `prefill 2s … prefill 300s` (+ `prefill 1200s`, Mac): the whole prompt (T = 44 / 83 / 148 / 408 / 798 / 1461 /
     3918 / 15618) through `generate` in ≤ 2048-id chunks, at pos 0;
   - `decode @52 … @1850, @4900` (+ `@20000`, Mac): one token at each clip's final context (exact values from the
     reference run's generated length);
   - `e2e 5s / 30s / 111s / 300s` (+ `e2e 1200s` and `e2e 30min`, Mac; 30 min = the caller's 2 passes): encoder +
     prefill + the full greedy loop, **model time only** (feature extraction and detokenization excluded for every
     framework alike; the loop's own overhead included).
5. **"Slightly optimized" [user: basic + obvious fusions].** Same graphs as basic with: fused QKV and gate/up GEMMs
   (weights concatenated at build time), fused residual-add + RMSNorm, fused QK-norm + RoPE + KV write, SwiGLU in
   the gate/up epilogue, bias + GELU epilogues, a library or simple-tiled f16 GEMM per device (MPS f16 on the GPUs,
   as GPT-2's prefill already uses MPS; NEON `FMLAL` tiles on the Mac CPU since Accelerate has no f16 BLAS; SME2
   `FMOPA` f16→f32 on the iPhone CPU), K-split multithreaded GEMV for decode, flash-style attention with f32 softmax,
   key-split GQA decode attention. No profiling-driven tuning beyond that in phase 1.
6. **Kernels are f16-only and new [user].** lokinfer has 0 16-bit kernels today (183 f32 / int kernels). Every
   kernel is written for f16 specifically (`*_f16_cpu` / `*_f16_gpu`). The IR already has `F16`; numpy interop is
   zero-copy (`np.float16`); the remote protocol is dtype-generic (no protocol work).
7. **Attention never materializes a score matrix [derived].** Basic and opt alike compute per (head, query row)
   with a streaming (online) f32 softmax over the keys; the largest transient is one f32 row of ≤ `cap` scores (100
   KB). Materializing `[2048, 15618]` f32 per head would be 2 GB per layer on the Mac's 1200 s case.
8. **Memory and disk budget [derived].**

   | | Mac (32 GB) | iPhone (6136 MB per app) |
   |---|---|---|
   | f16 weights (both models) | 4.08 GB | 4.08 GB |
   | KV caches (`cap` × 114,688 B) | 2.94 GB (25,600) | 0.76 GB (6,656) |
   | transient activations (prefill chunk ≤ 2048: gate/up 50 MB, qkv 16 MB; encoder Te ≤ 15,600: fc1 128 MB) | < 0.3 GB | < 0.1 GB |
   | total | ≈ 7.3 GB | **≈ 5.0 GB (margin ≈ 1.1 GB)** |

   The embedding table is stored once per model (gather and LM head share it). The `.lkm` stores weights f16; a
   load-time f32 → f16 conversion is forbidden. Disk: 2 models × 2 devices (CPU / GPU graphs are different models)
   ≈ 8.2 GB in `~/.cache/lokinfer/qwen3asr/models/`, the same again uploaded to the phone once (cached by sha256).
   The executor has **no blob-cache clearing today** (`put_blob` / `load_blob` only): Q3 adds a `clear_blobs` message
   (LokiHost deletes its cached blobs; `RemoteDevice.clear_blobs()`), so old model versions do not pile up on the phone. The iPhone high-water mark is **measured in Q4
   before any iPhone benchmark**; if it exceeds 5.5 GB the iPhone `cap` drops (the 5 min case still needs 6,446).
9. **Baseline precision [user].** Every baseline runs f16 (weights and compute) on every device; each row records
   how f16 was forced (torch `dtype=torch.float16`, MLX `astype(float16)`, GGUF `--outtype f16`) and anything the
   framework keeps in f32 that we know of. torch CPU f16 is PyTorch's slow generic fallback (§8): recorded, labelled.
   A baseline that cannot run a case (out of memory on the 1200 s prefill, no f16 on a device) gets "n/a: reason"
   in the notes, not a number.
10. **Dashboard [user].** `Qwen3-ASR-1.7B` is the **first model** (first in `perfdb.DEFAULT_META["models"]`), its
    description says "f16 on every device", impl names `lokinfer basic` and `lokinfer` (both count as lokinfer in
    `wins()`), one-column layout (many cases). Its configurations count in the page's top totals (overall and per
    device), so the totals show the honest overall picture.
11. **Benchmark run rule [derived from docs/benchmarking.md].** Warm-up, then ≥ 30 runs or ≥ 2 s, median; when one
    run exceeds 10 s, ≥ 3 runs. Thermal / Low Power Mode recorded as always; iPhone: one cool-down per model, not
    per case.

12. **KV caches are caller-owned mutable inputs [user, 2026-11-05].** Not model-owned `persistent_buffer` state:
    the generate model takes one mutable input `kv f16[28, 2, P, 1024]` (layers × K/V × positions × kv width) that
    the `inout` kernels write in place and that is never an output. P is a symbol: the caller sizes the cache for the
    transcription at hand (a short clip needs MBs, not the 0.76 GB / 2.9 GB fixed `cap`), can keep several
    sequences with one compiled model, and can read / reset / reuse the cache directly. Same speed (in place either
    way); the gain is memory and flexibility, and it is decided before english-practice consumes the API. IR: an
    exec argument flagged mutable (verifier: only `inout` kernels write it, never returned, no aliasing between
    mutable arguments; bufferize / plan-memory / hoist-init leave it alone; the runtime binds it without a copy or
    refuses).

## Milestones

### Q0 — Precision gate and reproduction (no lokinfer kernel code)

1. `models/qwen3_asr/__init__.py`, `models/qwen3_asr/reference.py`: pinned HF id + revision, `load_model(dtype,
   device)`, `processor()`, `prepare(audio) → (features[n, 128, 100], n_audio, ids)` (the caller contract, checked
   against the processor's own outputs), `manual_generate(model, prepared, max_new)` (encoder → splice → chunked
   prefill → T=1 loop → argmax; asserted equal to `generate()` on CPU f32 and MPS f16), `split_long(audio)` (the
   `qwen-asr` rule, for the 30 min case only).
2. `models/qwen3_asr/clips.py` + committed `models/qwen3_asr/clips.json`: LibriSpeech test-clean download (openslr
   URL, sha256) into the cache, FLAC decode (`soundfile` if in the venv, else `afconvert`), clip assembly from the
   committed utterance lists, `load_clip(name) → f32[16 kHz]`, LibriSpeech's text per clip (for info; the model's
   transcript is judged against the golden, not against LibriSpeech). First task: confirm test-clean has one
   speaker with ≥ 30 min of consecutive material (else use test-other / dev-clean for the two long clips).
3. Edge clips, synthetic and deterministic, built in code: 0.3 s (< 0.5 s padding, N_a = 7), exactly 1.00 s (chunk
   boundary), 8.00 s (window boundary, N_a = 104), 8.08 s (N_a = 105: a 1-row last window), 1 s of silence, 3 s of
   white noise, 2 s of a 440 Hz tone.
4. **Precision gate** (`bench/research/qwen3asr/precision_gate.py`; results tabulated in `docs/models/qwen3-asr.md`):
   on the speech clips ≤ 300 s and the edge clips, run HF in f64 CPU (G-true and G-f16w), f32 CPU, f16 MPS (the
   app path), bf16 MPS, and f16 CPU if torch supports every op: per-stage relative error vs G-true (encoder output,
   projector output, last-row prefill logits), max |activation| per LM layer (residual, attention scores before
   softmax, SwiGLU product, down_proj output) and per encoder layer, greedy transcripts, NaN / inf anywhere.
   **Exit**: f16 transcripts identical to G-true's on every speech clip (or every difference also in bf16 / f32 and
   explained as a near-tie), no NaN / inf, max activation < 32768 everywhere (2× margin to 65504) — **else stop and
   bring the numbers to the user before any kernel is written**.
5. Reference timings of the app path (MPS f16 `generate()`) per clip, for orientation only.

### Q1 — Golden reference data

`models/qwen3_asr/reference.py` captures per case into `~/.cache/lokinfer/qwen3asr/ref/<case>.npz` (not committed):
- inputs: `features`, `n_audio`, `ids`;
- **G-f16w** stages: conv1 / conv2 / conv3 outputs (first and last chunk), conv_out + positions, encoder layers 0 /
  11 / 23, ln_post, projector output (= `audio`), LM hidden after layers 0 / 13 / 27, final norm, last-row prefill
  logits; **G-true**: greedy tokens and **teacher-forced logits at every generated position** (one non-autoregressive
  f64 forward over prompt + generated tokens: golden logits for each decode step without an f64 autoregressive loop);
- the **torch f16 (MPS) reference** values of the same stages and logits (the calibration for decision 3).
Long cases: ≤ 300 s in f64 (the 300 s teacher-forced forward over ≈ 4,900 tokens: minutes, once, cached); the
**1200 s case against an f32 CPU reference** (f64 over ≈ 20k tokens would take hours; f32 vs f64 logits differ by
≈ 1e-6 relative, §8), stated as such in the tests. **Exit**: all cases captured, greedy transcripts recorded.

### Q2 — f16 basic kernels (CPU and GPU), the oracle

Simple, readable kernels; f32 accumulation inside; one Python op each with its type rule (CPU
`python/lokinfer/ops/f16.py`, GPU `python/lokinfer/ops/f16_gpu.py`; the exact signatures are fixed in
`qwen3-asr-f16-kernels.md` so both devices match — the encoder's conv front end is one `audio_cnn` kernel that
processes one chunk at a time, since a graph cannot loop over chunks) and a test through a one-op graph against the **float64 torch composition**
(`tests/kernels/test_f16_basic.py`, new), over the real shapes plus odd sizes, length 1 and the window / chunk
boundaries. Tolerances relative to f16 output rounding (≈ 2⁻¹¹·√K for reductions, stated per kernel). The same
tests run on the iPhone with `pytest --device=iphone` (compiled one-op graphs through the remote executor).

CPU (`kernels/cpu/f16_basic.cpp`, new; NEON f16 loads / stores, f32 math) and GPU (`kernels/gpu/f16_basic.metal` +
`kernels/gpu/f16_gpu.cpp`, new; `half` storage, `float` accumulators): the same list on both.

| op | shapes (§3, §4) | notes |
|---|---|---|
| `cast_f32_f16`, `cast_f16_f32` | features in; logits out | |
| `conv2d3x3s2_bias_gelu_f16` | per chunk: `[1, 128×100] → [480, 64×50] → [480, 32×25] → [480, 16×13]` | NHWC with H = time, W = freq; weights re-laid out at build time; called once per chunk |
| `gemm_f16` (+ optional bias) | conv_out 7680→1024 (its input columns in (freq, channel) order: `conv_out`'s weight columns permuted at build time so the NHWC flatten is a view), encoder 1024-wide, LM 2048 / 1024 / 6144 | naive tiled, f32 accumulate |
| `gemv_f16` | decode (M = 1) | |
| `gemm_nt_last_f16` | LM head: last row `[1, 2048]·[151936, 2048]ᵀ` → f32 | output f32 |
| `gelu_erf_f16` | encoder FFN / convs / projector | exact erf |
| `layernorm_f16` | encoder, eps 1e-5, with bias | f32 stats |
| `rmsnorm_f16` | LM, eps 1e-6 | f32 stats, HF's cast order |
| `rmsnorm_heads_f16` | QK-norm over 128-wide heads | |
| `rope_f16` | rotate_half, θ 1e6; takes `pos i64[1]`, positions `pos + i` computed inside; f32 cos/sin | |
| `add_f16`, `add_clamp_f16` | residuals; the encoder's ±64504 clamp | |
| `silu_mul_f16` | SwiGLU (separate gate / up GEMMs in basic) | |
| `add_pos_tiled_f16` | + the sinusoid `[13, 1024]`, restarting every chunk | |
| `embedding_f16` | gather from the tied `[151936, 2048]` table | |
| `scatter_audio_f16` | the j-th `151676` id takes row j of `audio` | |
| `attention_window_f16` | encoder: separate q / k / v `[Te, 1024]`, 16 heads, hd 64, non-causal, block-diagonal windows of 104 rows | streaming f32 softmax |
| `attention_causal_gqa_f16` | LM prefill: q `[T, 2048]` over cache rows `0 .. pos + i`, 16 q / 8 kv heads, hd 128 | streaming; reads the caches |
| `kv_write_f16` | K (after QK-norm + RoPE) and V rows `[T, 1024]` into the `[cap, 1024]` caches at `pos` | `inout` persistent buffers |
| `attention_decode_gqa_f16` | LM decode: one query over `0 .. pos` | streaming; any `cap` |

**Exit**: every kernel's test green on Mac CPU, Mac GPU, and on the iPhone (`--device=iphone`) for both devices.

### Q3 — Basic graphs

`models/qwen3_asr/model.py` (new): `fold_weights(hf_model) → dict[str, np.float16]` (bf16 → f16 once; conv weights
re-laid out; `conv_out` columns permuted; QKV / gate-up **not** concatenated in basic), `build_encoder(sd, device,
variant="basic", max_chunks=...)`, `build_generate(sd, device, variant="basic", cap=...)` (defaults: the iPhone's
values 300 / 6,656; the Mac benches pass 1200 / 25,600), `transcribe(enc, gen, prepared, max_new)` = the caller
contract's loop (used by the tests and the e2e benches), `save_models(...)` into the model cache. Also in Q3: the
remote executor's `clear_blobs` message (`src/remote/server.cpp`, `python/lokinfer/remote.py`; decision 8). Devices `"cpu"`
and `"gpu"` (GPU graphs GPU end to end; only the scalar inputs and the logits readback cross). **Exit**: both graphs
compile, save / load, and run every case on Mac CPU and Mac GPU.

### Q4 — Model correctness (on the optimized variant)

**[user, 2026-11-05]** For this model the end-to-end model tests never run on the basic kernels: the basic kernels are
proven by their independent per-op tests (Q2, both devices) and stay the oracle for the fused kernels. The model tests
below run on the optimized variant (Q5), on 4 devices. (A first full run on the basic graphs, stopped when this was
decided: all 26 transcripts and the 20 min encoder + prefill passed on the Mac CPU and GPU; the per-position "2×
torch f16" bounds of the long teacher-forced checks and one GPU encoder case (e_window1) failed — to be re-examined on
the optimized variant, with the failure details, before any conclusion.)

Former Q4 text (now applied to `variant="opt"`):

`tests/models/test_qwen3_asr.py` (new; `slow` marker on the long clips):
- stage tests vs G-f16w with decision 3's calibrated bounds (encoder stages, LM stages, last-row logits);
- transcripts vs G-true on the speech clips ≤ 111 s by the full greedy loop (basic kernels are slow: the 111 s
  clip's ≈ 400 decode steps on naive kernels is minutes, acceptable once per device);
- long clips (300 s, 1200 s) **teacher-forced**: prefill the prompt, then feed the golden tokens and compare logits
  at 16 sampled positions (early / middle / last) — this checks decode at long contexts without a long greedy run;
- edge clips: no NaN / inf, transcript = G-true;
- save / load round trip of both `.lkm` models; the `.lkm` weights are f16 on disk (file size ≈ parameter bytes);
- iPhone CPU and GPU through `--device=iphone` (models uploaded once, cached); **memory high-water on the iPhone
  recorded** (LokiHost's `memory` feature) on the 300 s case: must stay < 5.5 GB.
**Exit**: all green on 4 devices; the memory number written in `docs/models/qwen3-asr.md`.

### Q5 — Slightly optimized version

`variant="opt"` in `build_encoder` / `build_generate`; new kernels in `kernels/cpu/f16_fused.cpp`,
`kernels/gpu/f16_fused.metal` (+ dispatch in `f16_gpu.cpp`), ops in `python/lokinfer/ops/f16.py`, tests in
`tests/kernels/test_f16_fused.py` against **the composition of the basic f16 kernels** (the oracle) and the f64
torch composition:
- weights: QKV concatenated (LM 2048 + 1024 + 1024; encoder 3 × 1024), gate / up concatenated (12288);
- fused `add_rmsnorm_f16` (residual + next norm, HF's cast order kept), `qknorm_rope_kvwrite_f16` (QK-norm + RoPE +
  cache write of K and V in one pass), `silu_mul` as the gate/up GEMM epilogue, `bias_gelu_erf` epilogue in the
  encoder GEMMs and convs, `add_pos` folded into `conv_out`'s epilogue;
- GEMM per device: GPU `gemm_f16_gpu_mps` (MPS f16, f32 accumulate; GPT-2's prefill already runs on MPS) and a
  `gemv4`-style f16 GEMV for decode; Mac CPU NEON `FMLAL` register-blocked tiles (f16 × f16 → f32) + GPT-2's K-split
  multithreaded GEMV for decode; iPhone CPU SME2 `FMOPA` widening (an f16 input path in `sme2_gemm`) + the same GEMV;
- attention: simdgroup-matrix flash attention in half with f32 online softmax (encoder windows; LM prefill GQA hd
  128), key-split GQA decode attention (GPT-2's split scheme, any `cap`); CPU: per-head NEON with f32 softmax;
- encoder conv2d: im2col + the f16 GEMM;
- side measurement (feeds a later bf16 decision, no change now): Mac CPU microbenchmark `BFMMLA` (bf16) vs `FMLAL`
  (f16) tile throughput, recorded in `docs/models/qwen3-asr.md`.
**Exit**: fused-kernel tests green on 4 devices; `variant="opt"` passes Q4's model tests on 4 devices; speedup vs
basic recorded per case kind.

### Q6 — Baselines (all f16; scripts in lokinfer, third-party checkouts in `<baselines>`)

- `bench/models/bench_qwen3asr_baselines.py` (new, Mac): torch eager CPU f16, torch.compile CPU f16, torch MPS f16
  (eager; torch.compile MPS if it compiles), the same cases (encoder, prefill, decode at the case contexts, e2e),
  each **verified before timing** (encoder output rel error vs G-f16w within the torch f16 calibration, transcript
  = G-true's). The 1200 s prefill may be out of memory for frameworks that materialize attention: recorded as
  "n/a: OOM".
- MLX Python (Mac GPU): `bench/models/runners/qwen3asr_mlx.py` run with `mlx_python` (lokinfer.local.json); model
  from mlx-audio's `qwen3_asr` with weights cast to f16; same cases and verification. `mlx` + `mlx-audio` installed
  in the baselines venv (not in lokinfer's).
- llama.cpp (4 devices): `<baselines>/llama.cpp` pinned at a commit, built with Metal; F16 GGUF + the audio mmproj
  from the pinned HF snapshot (`convert_hf_to_gguf.py --outtype f16`); verified once (transcript of the 10 s clip =
  G-true's); timings via `llama-bench` (prefill at the case T, decode at the case contexts, CPU and Metal) and
  `llama-mtmd-cli` with timing output (encoder per clip, e2e); iOS via `build-xcframework.sh` and a small harness
  app in `<baselines>/llamacpp-ios/`, launched like the existing iPhone baselines (`iphone-baselines.md`), CPU and
  Metal builds, 5 min cases only.
- MLX Swift (iPhone GPU): `<baselines>/mlx-audio-swift` pinned at a commit, a harness timing encoder / prefill /
  decode separately, MLX cache capped (`Memory.cacheLimit`) as for Kokoro, weights cast to f16 at load (recorded).
- Every row records Low Power Mode / thermal state and the precision forcing (decision 9).
**Exit**: each device has ≥ 1 baseline row per case kind on the dashboard, each verified; every "n/a" has its reason.

### Q7 — lokinfer benchmarks and the dashboard

- `bench/models/bench_qwen3asr.py` (new): `lokinfer basic` and `lokinfer` (opt) on Mac CPU / GPU (local) and
  iPhone CPU / GPU (remote executor), decision 11's run rule. `lokinfer basic`: every enc / prefill / decode case and
  e2e ≤ 300 s (its 1200 s / 30 min e2e would take hours on naive kernels: skipped, noted); `lokinfer`: every case.
  Decode cases: the per-step median over a real decode run at the case context.
- `bench/perfdb.py`: `"Qwen3-ASR-1.7B"` **first** in `DEFAULT_META["models"]` (cases from decision 4, description
  with "f16 on every device"); `perf/results.json`'s meta updated the same way; `GOAL_MODELS` unchanged.
- `perf/index.html`: Qwen3-ASR first (meta order drives the page), impl names `lokinfer basic` / `lokinfer`, one
  column layout, counted in the top totals.
- A **gap summary** in `docs/models/qwen3-asr.md`: per device and case kind, lokinfer basic / opt vs the best
  baseline (ratio), and where the time goes (prof-lokinfer per-kernel profile of the 30 s case on each device).
**Exit**: http://127.0.0.1:8765 shows Qwen3-ASR-1.7B at the top with all four devices filled.

### Q8 — Docs and wrap-up

`docs/models/qwen3-asr.md` (new: the caller contract, precision gate results, correctness standard and results,
memory numbers, gap summary, the bf16 side measurement), `docs/correctness-testing.md` (the f16 calibrated-bound
rule), `docs/perf-dashboard.md` and `docs/baselines.md` (the model, its cases, the f16 note, the new baselines),
`docs/runtime.md` if the persistent-cache or chunk-loop machinery changed, `README.md` (models list: Qwen3-ASR as
phase-1 "measured, not yet optimized", its 4-device f16 numbers), `docs/todo.md`.

## File map

New: `models/qwen3_asr/{__init__,reference,clips,model}.py`, `models/qwen3_asr/clips.json`,
`bench/research/qwen3asr/precision_gate.py`, `docs/impl-notes/qwen3-asr-f16-kernels.md` (the Q2 op contract),
`python/lokinfer/ops/f16.py` (CPU ops), `python/lokinfer/ops/f16_gpu.py` (GPU ops), `kernels/cpu/f16_basic.cpp`,
`kernels/cpu/f16_fused.cpp`, `kernels/gpu/f16_basic.metal`, `kernels/gpu/f16_fused.metal`, `kernels/gpu/f16_gpu.cpp`,
`tests/kernels/test_f16_basic_cpu.py`, `tests/kernels/test_f16_basic_gpu.py`, `tests/kernels/test_f16_fused.py`, `tests/models/test_qwen3_asr.py`,
`bench/models/bench_qwen3asr.py`, `bench/models/bench_qwen3asr_baselines.py`, `bench/models/runners/qwen3asr_mlx.py`,
`docs/models/qwen3-asr.md`, `<baselines>/llamacpp-ios/`, `<baselines>/mlx-audio-swift` harness.
Modified: `CMakeLists.txt` (the new kernel files), `src/runtime/registry.cpp` (registration), `python/lokinfer/ops/__init__.py`,
`src/remote/server.cpp` + `python/lokinfer/remote.py` (`clear_blobs`), `include/lokinfer/remote/server.h` (the message id),
`bench/perfdb.py`, `perf/results.json` (meta), `perf/index.html`, `tests/perf/test_dashboard_goals.py` only if its
model list is explicit, the docs listed in Q8.

## Order, testing cadence, risks

- Order: Q0 → Q1 → Q2 (CPU and GPU kernels in parallel, agents) → Q3 → Q4 → Q5 and Q6 in parallel (agents) → Q7 →
  Q8. Q0's gate stops the plan if f16 is not safe.
- Testing cadence: kernel tests per kernel; model tests per milestone; full suites (unit, lit, pytest, iPhone deploy)
  at the end of Q4 and Q7 only; dashboard sweeps only at Q7 (no long sweeps per change). Kokoro / GPT-2 goal guard
  run at Q7 (shared runtime changes, if any).
- Risks: (1) f16 overflow on an input the gate missed — the edge clips plus the NaN / inf assertion in every model
  test; (2) iPhone memory — measured in Q4 before any iPhone benchmark, `cap` adjustable; (3) baselines that cannot
  run a case (OOM on 1200 s, no f16 path) — recorded as "n/a: reason"; (4) llama.cpp / mlx-audio-swift upstream churn
  — pinned commits; (5) 8 GB of model files locally and on the phone — cached by sha256, old versions cleared; (6)
  torch CPU f16 very slow — recorded as is, labelled; (7) naive basic kernels make the long cases slow — long-case
  correctness is teacher-forced, long-case e2e benchmarks are opt-only; (8) LibriSpeech test-clean may lack a 30 min
  single-speaker run — checked first in Q0, with test-other / dev-clean as the fallback.
