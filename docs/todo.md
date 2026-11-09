# lokinfer — TODO

Legend: `[ ]` todo, `[~]` in progress, `[x]` done. A `[x]` means tested, not just written. Finished work is
summarized; the full item-by-item history of v0.0.1 is in git (tag `v0.0.1`, `docs/todo.md`).

## Status

- **v0.0.1 complete (2026-11-03)**, published at https://github.com/lariaus/lokinfer (tag `v0.0.1`, which includes the
  dependency-build fix: dev targets skipped when lokinfer is a dependency, Metal `-O0/-O3` chosen at configure time).
- **Qwen3-ASR-1.7B phase 1 complete (2026-11-07)** — plan `impl-notes/qwen3-asr-port-plan.md` (v3), research
  `impl-notes/research-qwen3-asr.md`, milestones Q0–Q8 below. lokinfer vs the best baseline (geomean over the
  dashboard cases): Mac CPU 2.39×, Mac GPU 1.07×, iPhone CPU 2.02×, iPhone GPU 1.14× (docs/models/qwen3-asr.md).
- **Next: speech-to-text in lokinfer** (designed, not started; the user starts it) — section below.

## Goals

1. **(user, 2026-11-02) Kokoro-82M and GPT-2 124M: lokinfer #1 in every configuration** (every device × case on
   the dashboard), never regress. Kokoro is the most important model. Guarded by `tests/perf/test_dashboard_goals.py`;
   any change touching their kernels / runtime is checked on the affected cases before landing.
2. **(user, 2026-11-02)** Other v0.0.1 models (ResNet-18, MobileNetV2, DCGAN G/D, MNIST): within 5-10 % of the best
   baseline, ideally #1.
3. **(user)** Strictly apples to apples (same model, inputs, work and precision as the baselines). f32 for every model
   except Qwen3-ASR-1.7B, which is **f16 on all four devices** (user, 2026-11-05), for lokinfer and every baseline.
4. **(user, 2026-11-05) Qwen3-ASR-1.7B phase 1: not to beat anything** — golden reference data, a basic and a
   slightly optimized lokinfer version, baselines on all four devices, on the dashboard as the first model, to measure
   the gap. Not a goal model for the "#1 everywhere" guard.

## Done in v0.0.1 (summary)

- Core stack: MLIR-like IR, verifier at load and after every pass, compiler pipeline (dce, bufferize, plan-memory,
  hoist-init), runtime (CPU thread pool, Metal 4 GPU context with in-flight segments, thread-safe), Python + C++ APIs,
  save / load (`.lkm`), remote executor (LokiHost on the iPhone), profiling tools (prof-lokinfer, kbench), perf
  dashboard with locks and perfdb.
- Models (f32, Mac CPU / Mac GPU / iPhone CPU / iPhone GPU): Kokoro-82M (fused CPU, Winograd / SME2; GPU end to end),
  GPT-2 124M (prefill, decode, chunked generation), ResNet-18, MobileNetV2, DCGAN G / D, MNIST MLP. Kokoro and GPT-2
  #1 in every dashboard configuration (113 / 120 configurations overall).
- Kokoro end-to-end quality validated against english-practice's Python path on 225 texts (Mac GPU and iOS GPU:
  difference test PASS, the user's listening approval); `tests/e2e/kokoro/`.
- Public release scrub: machine-local values in the git-ignored `lokinfer.local.json` (`docs/benchmarking.md` "Local
  configuration").

## Deferred (user, 2026-11-03)

- [~] iPhone gaps vs ExecuTorch / MLX Swift (CPU small models: MobileNetV2, ResNet-18, DCGAN D N=1, MNIST; GPU fixed per-call latency)
  - [~] (deferred (user, 2026-11-03): MobileNetV2 N=1 1.61 / N=8 8.58 ms vs 1.49 / 7.92, DCGAN D N=1 0.18-0.28 vs 0.19, ResNet-18 N=8 18.96 vs 18.67) The remaining iPhone CPU gap on small convnets is spread over ~35 kernels of 50-100 µs. NOT the GEMM math: single-threaded Accelerate (AMX) runs MobileNetV2's pointwise shapes (K = 16-144) at 500-1000 GFLOP/s on the M2 Max (microbench 2026-11-01), so a NEON micro-kernel would be slower. Suspects: per-kernel fixed costs (pool hand-off, im2col/epilogue passes), memory traffic between unfused dw/pw layers; profile per-kernel time vs FLOPs on the phone before changing anything

## iPhone unavailable (user, 2026-11-06 18:05: "I'm taking back the iphone") — `/tmp/lokinfer-phone-disabled`; remove it only when the user says
Left for the phone (2026-11-06): **all done the same night** (21:39–23:02, docs/worklog.md): every lokinfer row
re-measured from "nominal" with `--cool-to`, llama.cpp CPU e2e 300 s (221 s), the iPhone kernel bench rows, the
Kokoro / other baseline data pushed back (kokoro.pte, kokoro-v1_0.safetensors, cases/*.kkc, other/*), and (23:11) the
iPhone CPU kernel timing table. The `kSmeMinM` A/B was done 2026-11-07 (16 -> 5).

### Earlier (user, 2026-11-05 16:00)
Needs iPhone validation once it's back (focused runs, not sweeps): Q5 fused-kernel tests on the phone
(`test_f16_fused_{cpu,gpu}.py --device=iphone`, after `scripts/ios_deploy.sh`); then Q4 model tests on the optimized
variant on the phone (device tensors for the KV cache); the llama.cpp iPhone timings (Q6); re-push of the old iPhone
baseline data.

## Qwen3-ASR-1.7B — phase 1: measure the gap (plan: `impl-notes/qwen3-asr-port-plan.md`, draft v3)

### Q0 — Precision gate and reproduction (no kernel code)
- [x] (2026-11-05: `manual_generate` == HF `generate()` token for token on CPU f32 and MPS f16, chunked prefill
  (chunks of 37) == unchunked; `split_long` is a copy of qwen-asr 0.0.6's 100 ms-window rule) `models/qwen3_asr/reference.py`: pinned HF revision; `prepare(audio)` = the caller contract (chunked features
  `[n, 128, 100]`, `n_audio`, prompt ids) checked against the processor; `manual_generate` (chunked prefill ≤ 2048 +
  T=1 loop) asserted equal to `generate()` on CPU f32 and MPS f16; `split_long` (qwen-asr's rule, 30 min case)
- [x] (2026-11-05: test-clean sha256 pinned; s2 2.45 s, s5 5.53, s10 10.09, s30 29.48, s60 59.01, s111 111.95, s300
  300.16 (one speaker each, all different below 300 s); test-clean has only ~8 min per speaker, so s1200 (1199.80 s,
  3 speakers) and s1800 (1800.88 s, 4 speakers) chain speakers back to back) `models/qwen3_asr/clips.py` + committed `clips.json`: LibriSpeech test-clean (sha256, cache), clips 2 / 5 / 10 /
  30 / 60 / 111 / 300 s + 1200 s + 30 min (Mac) from consecutive utterances of one speaker — first confirm a speaker
  with ≥ 30 min exists (else test-other / dev-clean); edge clips 0.3 s, 1.00 s, 8.00 s, 8.08 s, silence, noise, tone
- [x] (2026-11-05, **passed**, `docs/models/qwen3-asr.md`: f16 logit error ≈ 1e-3 (bf16 ≈ 1e-2), no non-finite
  value, max activation 14,896 (4.4× under 65504), transcripts = G-true on 13 / 14 clips, the 14th a comma at a
  0.004 near-tie) Precision gate (`bench/research/qwen3asr/precision_gate.py`): f64 G-true + G-f16w, f32, f16 MPS, bf16 MPS, f16
  CPU if supported — per-stage error, max |activation| per layer, transcripts, NaN / inf. Exit: f16 transcripts =
  G-true, max activation < 32768, no NaN; else stop and report to the user
- [x] App-path reference timings (MPS f16) per clip: `docs/models/qwen3-asr.md` (indicative, Mac busy)

### Q1 — Golden reference data
- [x] (verified 2026-11-06: `~/.cache/lokinfer/qwen3asr/ref/` holds g_true / g_f16w / torch_f16 / inputs .npz for all
  15 cases s2..s1200 + 7 edge clips, capture2.log CAPTURE_RC=0) (code written: `reference.capture` + `python -m models.qwen3_asr.reference`; one teacher-forced forward per
  case and variant, tokens = the gate-verified f32 transcript) Capture per case into `~/.cache/lokinfer/qwen3asr/ref/`: G-f16w stages (convs, conv_out + pos, layers 0 / 11 /
  23, ln_post, projector; LM layers 0 / 13 / 27, final norm, last-row logits), G-true greedy tokens + teacher-forced
  logits at every generated position, torch f16 (MPS) values of the same stages (the calibration); ≤ 300 s in f64,
  1200 s vs an f32 reference

### Q2 — f16 basic kernels (the oracle), CPU and GPU
- [x] Op contract `docs/impl-notes/qwen3-asr-f16-kernels.md` (same signature on both devices; `audio_cnn` = the conv
  front end chunk by chunk; `add_pos_prefix` keeps the first N_a rows; one `attention_gqa` for prefill and decode)
  and the scaffold (files wired into CMake / registry / ops, builds)
- [x] CPU ops `python/lokinfer/ops/f16.py`, GPU ops `ops/f16_gpu.py`; `tests/kernels/test_f16_basic_{cpu,gpu}.py` vs f64 torch (real + odd
  shapes, chunk / window boundaries, f16-rounding tolerances stated per kernel) (verified 2026-11-06: all four files
  exist; 97 CPU / 145 GPU tests green per the two items below)
- [x] (2026-11-05: the 18 contract ops as `<name>_f16_cpu`, ops `lk.ops.f16.<name>`; `tests/kernels/test_f16_basic_cpu.py` 97 passed on the Mac and 97 on the iPhone (`--device=iphone`), whole `tests/kernels` 877 passed on the Mac; casts / GEMV not separate ops per the contract: `audio_cnn` takes f32 features, `lm_head_last` returns f32, `gemm` covers M = 1) CPU `kernels/cpu/f16_basic.cpp`: casts, conv2d 3×3 s2 + bias + GELU(erf) (per chunk, NHWC), GEMM (+bias),
  GEMV, LM-head last row → f32, GELU(erf), LayerNorm, RMSNorm, per-head RMSNorm, RoPE (takes `pos`), add /
  add-clamp, SiLU·mul, per-chunk positions, embedding, audio scatter (j-th pad ← row j), windowed attention,
  causal GQA attention over the cache, KV write, GQA decode attention — attention always streaming (no `[T, P]`
  score matrix)
- [x] GPU `kernels/gpu/f16_basic.metal` + `f16_gpu.cpp`: the same list (all 18 contract ops as `<name>_f16_gpu`, ops in
  `ops/f16_gpu.py`; `tests/kernels/test_f16_basic_gpu.py` 145 tests green on the Mac GPU and the iPhone GPU,
  2026-11-05; tolerances = f16 rounding + a stated, measured float32 term per op). RoPE: float angle
  `float(pos + i) * float(inv_freq)` like HF's own f32 rotary path (also inside the f64 golden), cos / sin of it
  accurate (Cody-Waite). Speed (Mac, indicative): 16x16-tiled GEMM ~0.9 TFLOP/s, audio_cnn ~8.5 ms per chunk
- [x] Kernel tests green on Mac CPU / GPU and on the iPhone (`pytest --device=iphone`) (verified 2026-11-06:
  test_f16_basic_cpu 97/97 Mac + iPhone, test_f16_basic_gpu 145 Mac GPU + iPhone GPU, recorded above)

- [x] **Compiler bug (found by the Q2 CPU agent, fixed 2026-11-05)**: returning a persistent buffer from exec made
  bufferize replace the state by a fresh output alloc each call, so that cache lost its contents every call (the
  "first 16 bytes zeroed" = row 0 of a 4-wide f32 cache). Now a compile error ("persistent buffers can't be graph
  outputs: return a copy"); lit `tests/lit/bufferize/persistent_return_rejected.lkir`; lit 24/24, Python API +
  f16 CPU kernel tests 211 passed

### Q3 — Basic graphs
- [x] `fold_weights` (2026-11-05): layouts verified against HF — a torch emulation of the `audio_cnn` +
  `add_pos_prefix` contract with the folded weights matches HF's encoder layer-0 input to 3.8e-5 (s10, e_window1,
  e_short)
- [x] (2026-11-05) Folded f16 weights straight from the safetensors, one tensor at a time (`fold_checkpoint`,
  14 s, 708 tensors, 2.038 B params, 4.08 GB in `~/.cache/lokinfer/qwen3asr/weights_f16/`), memory-mapped by
  `load_weights` (file-backed: no copy for CPU graphs)
- [x] (2026-11-05: basic + opt graphs, CPU and GPU; caller-owned mutable KV cache; `debug_layers` per-layer outputs)
  `models/qwen3_asr/model.py`:
  `build_encoder` (features f32[n, 128, 100] + n_audio i64[1] → audio f16[N_a, 2048]; `dim.read`; CNN chunk loop),
  `build_generate` (ids + audio + pos → last-row logits f32; persistent f16 KV caches, `cap` 6,656 default / 25,600
  Mac; T ≤ 2048 per call), `transcribe` (the contract's loop), `save_models`; CPU and GPU graphs run every case on
  the Mac, save / load round trip
- [x] Remote executor split transport (user, 2026-11-06: "the original weights should be stored on the iPhone, without
  losing perf"): HELLO feature `split_models`, message LOAD_MODEL_SPLIT (16). The client splits a .lkm into its
  skeleton (header + metadata) and one blob per constant (sha256 each, `_C.lkm_constant_ranges`); only missing blobs
  go up; the server maps each constant's blob file (`readModelSplit`: page-aligned, zero-copy, as readModelFile).
  A weight on the device is never sent again, whatever graph uses it (Qwen's CPU and GPU graphs share theirs; a
  graph / kernel change re-sends only the skeleton). Also fixed: a whole-model upload could never exceed the 2 GiB
  frame limit, and macOS rejects single socket sends >= 2 GiB (EINVAL): sends now go in 1 GiB pieces.
  tests/python/test_remote.py::test_split_model_weights_cached (Mac serve + iPhone). Qwen3-ASR on the iPhone:
  default tier 20/20 (CPU + GPU), 6 min incl. the first upload
- [x] ios_deploy.sh watchdog killed a healthy LokiHost mid-test (2026-11-06, iPhone slow tier: broken pipe): the
  `devicectl --console` connection dropped (CoreDevice "The connection was invalidated") while the app kept
  running, and the watchdog relaunched it with --terminate-existing. It now relaunches only once
  `devicectl device info processes` no longer lists LokiHost
- [x] iPhone GPU timeout on long prefills (Qwen3-ASR 111 s, T = 1473: "GPU timeout", then LokiHost kept failing):
  no single kernel was long (max 15.7 ms); ~76 dispatches per command buffer (params ring) = ~0.6 s of GPU work.
  GPU work budget per command buffer (docs/runtime.md). Forcing tiny budgets on the Mac exposed a real runtime race:
  the hazard tracker trusted a fresh segment that already held the previous kernel's later dispatches (mid-kernel
  commit); the first kernel of a segment now always gets a barrier. Mac: Qwen GPU slow tier with a 10 GFLOP budget
  28/28, full suites green, GPT-2 decode unchanged (2.06 ms)
- [x] iPhone GPU prefill is ~3.4x MLX Swift's (30 s: 497 vs ~145 ms): MLX uses the A-series GPU's f16 matrix units,
  our steel GEMMs didn't. **Done 2026-11-07 (gpu-gemm agent)**: Metal 4 tensor-op (`matmul2d`) f16 GEMMs for M > 1
  (`kernels/gpu/f16_tensor.metal`, docs/kernels/gemm.md "f16 tensor-op GEMM"), all fused epilogues kept, on by default
  on Apple10+ GPUs (`kTensorGemmDefault`, flavour "_nax" = MLX's fragment layout, 64x128 tiles); the M2 Max keeps
  steel (tensor ops are slower there). Validated on the phone: kernel tests --run-slow (layout-agnostic 193 passed,
  "_nax" GEMM family 138 passed), Qwen model tests --run-slow CPU + GPU 56 passed with each flavour, + GPU 28 passed
  with the final tile rule; transcripts golden. Recorded (iPhone GPU, from nominal): prefill 30 s 489 → 151.6 ms,
  300 s 5,267 → 1,955 (MLX 146 / 1,993), enc 30 s 161 → 59.4 (MLX 98.2), e2e 30 s 3,341 → 2,933 (MLX 2,941);
  docs/models/qwen3-asr.md "On the iPhone".
  - Left: prefill 2 s (40.2 vs MLX 35.9 ms) and 30 / 60 s (+4 %); e2e 111 s / 300 s are decode-bound (llama.cpp Metal
    ahead there). Ideas: split-K for small-M / large-K (MLX uses it for K >= 3 max(M, N)), a fused-epilogue cost check
    on fc1 (GELU: 5.5 vs 8 TF/s), 128x128 vs 64x128 per shape on the encoder (enc o M = 3900 [1024x1024] 1.24 vs
    3.15 ms in one noisy kernel-bench run).
- [x] Remote executor leaked every client's models and device tensors (one Session for the app's lifetime): on the
  phone a test run + a bench session accumulated Qwen executables (activation arenas of hundreds of MB each) and the
  GPU started timing out even on tiny calls. They are now freed when the client disconnects (ids are never reused;
  the blob store stays)
- [x] GPU memory kept after a model was freed (2026-11-06): `gpu_allocated` stayed 4.3 GB on the phone with 0 models
  after a GPU session, because residency-set removals were only committed at the next encode. `GpuContext::trim()`
  now commits them (server: on disconnect and FREE_TENSOR); Mac repro: 3.57 GB -> 11 MB after the disconnect
- [x] Remote executor `clear_blobs` message (server.cpp, remote.py): old blobs (whole models, superseded weights)
  can be removed from the phone (2026-11-06: BLOBS message, see "Iteration speed") (2026-11-06: not implemented, no `clear_blobs` in the code)

- [x] **Mutable inputs (user, 2026-11-05; plan decision 12)**: the KV cache is a caller-owned mutable input
  `kv f16[28, 2, P, 1024]` (P = the caller's capacity: `kv_cache(cap, device)`; iPhone max 6,656 = one 5 min pass,
  Mac 25,600 = one 20 min pass), not model-owned state. IR `%x: buffer<...> mutable` (`Region::mutableArgs`; text +
  `.lkm` v2, v1 still read); Python `lk.MutableType`; verifier: an `inout` operand must be a mutable input or
  persistent state (traced back through init for hoisted state), neither may be returned; runtime / Python refuse
  any binding that would copy (strided, read-only, wrong dtype, unwrappable GPU memory) and overlapping inputs.
  Found while doing it: **`inout` on an ordinary tensor input silently mutated the caller's array** (now a verify
  error) and strided buffer inputs were silently copied (writes lost). lit 27/27 (3 new), C++ unit tests pass;
  `build_generate` / `transcribe` / the Q4 tests use it
- [x] Remote executor device tensors (alloc / write / read / free, RUN2 binds them in place; mutable args never run
  on a copy; transparent `--device=iphone` path for tests) — agent, 2026-11-05. Built: protocol messages 12-15 +
  RUN2 input kind 3 + HELLO feature "device_tensors" (server.h, server.cpp), MEMORY reports live device tensors,
  `RemoteDevice.alloc_tensor / write_tensor / read_tensor / free_tensor` + `DeviceTensor`, conftest transparent
  path; fuzz cases + budget / churn test (C++ `[remote]` 6/6), tests/python/test_remote.py 13/13 (f32 rows, f16 CPU
  and GPU caches accumulating across RUN2 calls, mutable-inline refusals, transparent path) on Mac serve (12 + 1
  Mac-only skip with `--device=serve`) and on the iPhone 18 Pro Max (`--device=iphone`: 12 passed, 1 Mac-only skip;
  on-device Catch2 71/71 incl. the new fuzz cases)

- [x] **GPU errors were silent after the first commit** (found 2026-11-05 on Qwen3-ASR long passes: zeros returned
  as success): the reused `MTL4::CommitOptions` delivered feedback once. Fixed with fresh options per commit + the
  sentinel (docs/runtime.md, "GPU failures"); latency +0.01 ms/call. Unit 71/71, Python/kernels/GPT-2/Kokoro/e2e
  1311 passed. Also fixed: test_remote's `import conftest` resolved to tests/models/conftest.py in a mixed run.
  iPhone: redeploy + device suite deferred (phone disconnected)

### Q4 — Model correctness on 4 devices (optimized variant only — user, 2026-11-05)
Basic kernels are proven by their op tests (Q2); the model E2E tests run on `variant="opt"` only (they skip until Q5).
- [x] First full run on the basic graphs (stopped when the rule was set): transcripts 13/13 CPU + 13/13 GPU, encoder +
  prefill 15/15 CPU incl. the 20 min pass, 14/15 GPU; failing: long teacher-forced s300 / s1200 (CPU), s300 (GPU),
  GPU e_window1 encoder — the per-position "2× torch f16" bound may be too tight; re-check on opt with details
  (verified 2026-11-06: re-checked on opt — failures were the references' RoPE rounding (item below), all tiers pass
  Mac CPU + GPU, iPhone default 20/20 + slow 28/28 CPU + GPU)
- [x] (verified 2026-11-06: golden data captured, test file uses it — DEFAULT / SLOW / long cases incl. edge clips +
  isfinite checks; passes on all 4 devices) (written 2026-11-05, waits for the golden data) `tests/models/test_qwen3_asr.py`: stages vs G-f16w within 2× torch f16's error; transcripts = G-true by the
  full greedy loop on clips ≤ 111 s; 300 s / 1200 s teacher-forced (logits at 16 sampled positions); edge clips (no
  NaN / inf, transcript = G-true); `.lkm` weights f16 on disk
- [x] Test tiers (user, 2026-11-05: the full Qwen3-ASR model tests take ~1-1.5 h on the basic CPU kernels):
  `--run-slow` / `--run-long` in tests/conftest.py; default tier ~1-2 min per device; docs/correctness-testing.md
- [x] Correctness rules reworked (2026-11-06, docs/correctness-testing.md "Model-level f16 rules"): RMS ≤ 2x + max
  ≤ 4x torch f16 on the encoder output and every prompt row's lm_norm; logits checked by argmax (near-tie rule),
  per-row RMS ≤ 4x, pooled RMS ≤ 2x. The old "max ≤ 2x" failed equally precise CPU variants by rounding luck; fault
  injection (bf16-level weight noise, RoPE θ +1 %) shows the new rules catch real errors. All tiers pass on the Mac
  (60/60, CPU + GPU); iPhone default tier 20/20
- [x] (verified 2026-11-06: iPhone default 20/20 + slow 28/28 on CPU and GPU; 300 s peaks below, all < 5.5 GB; the
  memory numbers are only here, not yet in docs/models/qwen3-asr.md) iPhone CPU + GPU via `--device=iphone`; memory high-water on the 300 s case recorded, must be < 5.5 GB (else
  lower the iPhone `cap`). CPU e2e 300 s measured 2026-11-06 (fresh LokiHost, one bench session): resident peak
  4.69 GB (incl. the mmapped weights), phys_footprint peak 0.86 GB -> within budget. GPU e2e 300 s (fresh LokiHost): resident peak 0.28 GB, footprint
  peak 0.26 GB (Metal's no-copy weight wraps are not in either); gpu_allocated after: 0.38 GB with 0 models (the
  storage pool's bounded cache)

- [x] Long-context checks: the "2× torch f16" failures were the REFERENCES' fault, not lokinfer's (user asked to
  investigate before changing the rule): `.to(float16)` rounded RoPE's inv_freq in G-f16w and in the torch f16
  calibration (angles off by up to 3 rad at 17k); lokinfer is closer to f32 than torch at every layer
  (docs/models/qwen3-asr.md "Long-context accuracy"). Fixed `load_model` (buffers exact); recapturing G-f16w + torch
  f16, then rerun the long checks with the unchanged 2× rule. **Done 2026-11-05**: recaptured; all tiers (default +
  slow + long) on the optimized variant pass, Mac CPU 30/30 (8.5 min) and Mac GPU 30/30 (58 s), incl. the 20 min pass
  and both teacher-forced long checks, with the 2× rule unchanged. Q4 Mac done; iPhone deferred (disconnected)

### Q5 — Slightly optimized version (basic + obvious fusions)
- [x] Op contract `docs/impl-notes/qwen3-asr-f16-opt-kernels.md` (fused GEMM epilogues: bias+GELU, SwiGLU, residual
  add, bias+add+clamp; QK-norm + RoPE + KV write; packed-QKV window attention; flash GQA; fast LM-head GEMV), scaffold
  (`kernels/cpu/f16_fused.cpp`, `kernels/gpu/f16_fused.metal`), graph `variant="opt"` written (`_encoder_opt`,
  `_lm_layer_opt`; enabled by adding "opt" to `M.VARIANTS` once the kernels pass), concatenated weights folded to disk
  (`fold_opt`: 2.03 GB, memory-mapped, 3 s) — CPU and GPU kernel agents running
- [x] (closed 2026-11-07: the last two dropped by decision, see below) Concatenated QKV / gate-up weights; fused add + RMSNorm, QK-norm + RoPE + KV write, SwiGLU epilogue, bias +
  GELU(erf) epilogues, positions folded into `conv_out`; `tests/kernels/test_f16_fused.py` vs the basic kernels and
  f64 torch (2026-11-06: concat weights, epilogues, QK-norm+RoPE+KV, tests `test_f16_fused_{cpu,gpu}.py` done; still
  missing: positions not folded into `conv_out` (basic `add_pos_prefix` reused), no fused add + RMSNorm (rmsnorm+GEMM
  fused instead) — tick if those are dropped by decision). **Dropped, 2026-11-07** (user: "finish phase 1"): the
  per-kernel profiles show both are moot — the residual add lives in the GEMM epilogue (`gemm_add`) and the RMSNorm
  in the next GEMM (`rmsnorm_gemm*`), so no standalone add / RMSNorm is left in the layers (the final norm: 0.0–0.1 %
  of a call); `add_pos_prefix` is 0.013–0.017 ms, 0.0 % of the encoder on every device.
- [x] GEMM / GEMV per device: GPU MPS f16 GEMM + `gemv4`-style f16 GEMV; Mac CPU NEON FMLAL tiles + K-split GEMV;
  iPhone CPU SME2 FMOPA f16 path in `sme2_gemm` + the same GEMV (verified 2026-11-06: steel f16 GEMM kept after MPS
  A/B + f16 GEMV in f16_fused.metal, FMLAL tiles + K-split GEMV in f16_fused.cpp, FMOPA f16 in f16_sme2.cpp; iPhone
  CPU validation / `kSmeMinM` A/B tracked in the CPU opt kernels item)
- [x] Attention: half flash attention with f32 online softmax (encoder windows, LM prefill GQA hd 128), key-split
  GQA decode (any `cap`); CPU per-head NEON; encoder conv2d as im2col + f16 GEMM (verified 2026-11-06: GPU simdgroup
  flash + key-split decode + im2col conv (qwen3-asr.md "Q5 GPU kernels"); CPU flash core with online f32 softmax +
  FMLAL blocks in f16_fused.cpp, `audio_cnn_opt` im2col)
- [x] GPU opt kernels (agent, 2026-11-05): all 12 contract ops `<name>_opt_f16_gpu` in kernels/gpu/f16_fused.metal +
  f16_gpu.cpp (steel f16 GEMM 64x64x32 with fused epilogues / SwiGLU, f16 GEMV for M = 1, simdgroup flash attention
  with half K / V fragments + float softmax, key-split GQA decode, qknorm+RoPE+KV write, im2col conv); MPS vs steel
  measured (steel kept). tests/kernels/test_f16_fused_gpu.py 152/152 on the Mac and on the iPhone (default + slow).
  Speedups in docs/models/qwen3-asr.md "Q5 GPU kernels" (bench/kernels/bench_f16_fused_gpu.py): Mac complete;
  **needs iPhone validation**: the iPhone bench rows past the encoder o_proj (fc1, fc2, lm_head, qknorm, attention,
  audio_cnn) — the phone was taken mid-run (2026-11-06 22:20: done, docs/models/qwen3-asr.md "Q5 GPU kernels"; the iPhone bench
  skips the longest cases: decode pos 20000, prefill pos 13570, window S=15600) (2026-11-06: tests done on iPhone; those iPhone bench rows are still "-"
  in qwen3-asr.md)
- [x] `variant="opt"` passes Q4's model tests on 4 devices; speedup vs basic recorded per case kind (2026-11-06: Mac,
  docs/models/qwen3-asr.md "Fused vs basic"; basic rows for every case ≤ 300 s on the dashboard) (2026-11-06: tests
  pass on 4 devices; missing: model-level basic vs opt speedup per case kind — only per-kernel speedups recorded, no
  `lokinfer basic` rows)
- [x] Side measurement: Mac CPU `BFMMLA` (bf16) vs `FMLAL` (f16) tile throughput, recorded (no change now):
  FMLAL 4x16 tile 793 GFLOP/s on 8 P cores vs BFMMLA 396 (docs/models/qwen3-asr.md "Q5 CPU kernels")
- [x] CPU opt kernels (`kernels/cpu/f16_fused.cpp` + `f16_sme2.cpp`, `tests/kernels/test_f16_fused_cpu.py`, all 12 ops
  of the opt contract): Mac default + slow tiers green, speedups vs basic recorded (Mac); **needs iPhone validation**
  (phone taken 2026-11-05): default tier passed once on the phone, but the slow tier, the iPhone timing table, the
  SME2-vs-NEON A/B for `kSmeMinM` and one unreproduced phone hang (`test_audio_cnn_opt[3]`, first run) are open
  (2026-11-06: no evidence yet of test_f16_fused_cpu slow tier on iPhone; qwen3-asr.md still lists all four as open)
  2026-11-06 17:10: **slow tier on the iPhone: 106 passed, 2 skipped (2:46)**, the audio_cnn cases included, no hang.
  2026-11-06 23:11: the iPhone (and a fresh Mac) kernel timing table, `bench_f16_fused_gpu.py --cpu [--iphone]`
  (qwen3-asr.md "Q5 CPU kernels"). 2026-11-07: `kSmeMinM` SME2-vs-NEON A/B done on the iPhone (one deploy per
  setting): SME2 wins 1.6-4x at every M from 5 to 128 -> kSmeMinM 16 -> 5 (qwen3-asr.md "Q5 CPU kernels")

### Q6 — Baselines (all f16, each verified before timing)
Setup + verification done 2026-11-05 (docs/baselines.md § 8, impl-notes/iphone-baselines.md Qwen3-ASR section).
Verified = greedy transcript of s10 and s30 equal to the f32 CPU reference (`R.manual_generate`). Dashboard timings
are Q7 (shared harness); only indicative timings so far.
- [x] torch eager CPU, torch.compile CPU, torch MPS, torch.compile MPS: `bench/models/bench_qwen3asr_baselines.py`
  (enc / prefill / decode@C / e2e; `--verify`); all four verified token for token. torch CPU f16 = PyTorch's generic
  path, labelled. Open: "n/a: OOM" check of the 1200 s prefill (Q7)
- [x] MLX Python (Mac GPU): `runners/qwen3asr_mlx.py`, mlx 0.32.4 + mlx-audio 0.6.4 in `<baselines>/.venv-mlx`
  (`mlx_python` set), f16 weights `<baselines>/models/qwen3-asr-mlx-f16` (sinusoid table cast too); verified
  token for token
- [x] llama.cpp b12125 (`7d516e25`, patched converter for the `-hf` layout + M-RoPE sections, BLAS off): f16 GGUF +
  mmproj; Mac CPU / Metal verified (mtmd-cli + harness Mac CLI); iPhone CPU / Metal harness `<baselines>/llamacpp-ios/`
  (xcframework with `armv8.6-a+fp16`, BLAS off) verified on the phone. Notes: llama.cpp feeds up to 12 extra padded
  audio tokens (s10 161 vs 149); its activations / norms are f32 (only weights + KV are f16)
- [x] (2026-11-06: all long cases on the dashboard, CPU e2e 300 s 221 s) llama.cpp iPhone indicative / 5 min (`s300`, `n_ctx` ≥ 6,446) runs: the 2026-11-05 attempt was SIGKILLed by
  another session's `ios_deploy.sh` watchdog (`--terminate-existing`; `ios_deploy.sh` does not take the phone lock)
  (2026-11-06: 2-30 s iPhone llama.cpp rows are on the dashboard; s300 still not run)
- [x] MLX Swift (iPhone GPU): mlx-audio-swift @ `927d4e30` (patched: f16 sinusoid table, public submodules),
  mlx-swift 0.33.0, harness `<baselines>/qwen3asr-mlx-swift/` (enc / prefill / decode / e2e separately,
  `MLX_MEMLIMIT_MB=5500 MLX_CACHE_MB=1024`), f16 weights converted on the Mac; verified token for token on the iPhone
  and the Mac CLI; indicative iPhone numbers in iphone-baselines.md
- [x] (verified 2026-11-06: perf/results.json has 14 rows each for llama.cpp Mac CPU / Mac GPU, torch MPS,
  torch.compile CPU) Mac timings left for Q7 (indicative table in baselines.md § 8.3): llama.cpp Mac CPU / Metal (harness CLI
  `<baselines>/llamacpp-ios/llamacpp_asr_mac`, `llama-bench -p T` / `-n 1 -d C-1`, CPU with `-dev none`), torch MPS
  decode / e2e (crop fix landed after its run), torch.compile CPU (its run overlapped another model job). All model
  jobs under `/tmp/lokinfer-bigmodel.lock`
- [x] (verified 2026-11-06: ref_s60 / s111 / s300 / s1200 / s1800p1-2 .json in ~/.cache/lokinfer/qwen3asr/baselines/)
  References for s60 / s111 / s300 (and the f32-reference 1200 s) for the long cases: `prep_qwen3asr.py` /
  `bench_qwen3asr_baselines.ref()` compute them on first use (bounded-memory `R.load_model(torch.float32)`)

- [x] (2026-11-05) Fixed: `scripts/ios_deploy.sh` now takes `/tmp/lokinfer-phone.lock` itself (a deploy relaunches
  LokiHost and kills any app on the phone; one did kill the llama.cpp timing runs) unless `LOKINFER_PHONE_LOCK_HELD=1`
- [x] (2026-11-06 23:00, pushed) Re-push the old iPhone baseline data (Kokoro, `other/` models) before re-running those iPhone baselines: the
  Qwen3-ASR harness install reset the shared app's `Documents/baselines/` (iphone-baselines.md has the push commands);
  the phone now holds ≈ 8.2 GB of Qwen3-ASR baseline model files (2026-11-06: no evidence of the re-push)
- [x] Big-model jobs (anything holding the 4 GB weights) run under `/tmp/lokinfer-bigmodel.lock`, one at a time
  (2026-11-05, after a 15 GB swap episode; docs/benchmarking.md)

### Q7 — lokinfer benchmarks and the dashboard
- [x] (closed 2026-11-07 with the exceptions below, documented in qwen3-asr.md "Fused vs basic") `bench/models/bench_qwen3asr.py`: `lokinfer basic` (all enc / prefill / decode cases, e2e ≤ 300 s) and
  `lokinfer` (every case incl. e2e 1200 s and 30 min on the Mac) on 4 devices; ≥ 30 runs / ≥ 2 s, ≥ 3 runs when a
  run exceeds 10 s. 2026-11-06: `lokinfer fused` done (30 cases Mac CPU / GPU, 25 iPhone CPU / GPU); `lokinfer basic`
  on the Mac for every case ≤ 300 s, CPU enc / prefill / decode 1200 s (prefill 405 s per call vs 48 s fused), GPU
  enc 1200 s. Not timed: GPU decode / prefill 1200 s (basic's GPU attention is not key-split: one dispatch over
  ~15,600 keys hits the macOS GPU timeout; a basic-kernel limit, not fixed: basic is the oracle). iPhone basic (night
  2026-11-06): every case ≤ 30 s on CPU and GPU; longer clips not timed (single-dispatch attention vs the iOS GPU time
  limit)
- [x] perfdb + `perf/results.json` meta: Qwen3-ASR-1.7B first, "f16 on every device"; `perf/index.html`: shown first,
  one column, counted in the top totals (user) (verified 2026-11-06: meta.models lists Qwen first with the f16
  description (bench/perfdb.py); index.html orders by meta, one-col for many-case models, totals over all entries)
- [x] Gap summary + per-kernel profile (30 s case, each device) in `docs/models/qwen3-asr.md` (2026-11-06: "Per-kernel
  profile, 30 s clip", all 4 devices) (2026-11-06: gap tables
  for Mac + iPhone done; missing: the per-kernel profile of the 30 s case per device)
- [x] Kokoro / GPT-2 goal guard still green (`tests/perf/test_dashboard_goals.py`) (verified 2026-11-06: passes)

### Perf (Mac, after the Q7 measurements; user, 2026-11-05: "improve more perf")
- [x] CPU large-M GEMMs on the AMX (Accelerate): `gemmOpt` converts x and W column panels f16 -> f32 (exact) on the
  pool, one multithreaded `cblas_sgemm` per panel, same f32 epilogue (Mac only; SME2 iPhones keep FMOPA). Kernel:
  1.8-2.9x (gate|up T = 2048 139 -> 47.5 ms, LM down T = 401 14.5 -> 6.5). Model, Mac CPU: enc 30 s 732 -> 420 ms,
  prefill 30 s 1772 -> 879, prefill 300 s 19.9 -> 9.5 s, e2e 30 s 5.0 -> 4.1 s (decode-bound). f16 fused CPU
  tests 98/98, Qwen CPU default tier 10/10
- [x] GPU decode GEMV: `kOptGemvGroups` 32 -> 128 (M2 Max). In-model A/B (decode C = 480): 12.74 -> 12.25 (64) ->
  12.14 ms (128); 256: 13.02. N = 2048 GEMVs had 32 threadgroups for 38 GPU cores. The kernel bench reuses its
  weights from the SLC and showed no difference: judge GEMV changes in-model (prof-lokinfer qwen3asr_decode)
- [x] CPU decode GEMV: 1 K-split task per pool thread on the Mac (was 3): o 90 -> 100 GB/s, qkv 103 -> 115, down
  102 -> 107, swiglu 102 -> 105 (microbench, distinct weights > SLC). decode 30 s 35.0 -> 33.3 ms (llama.cpp CPU
  31.7), e2e 30 s 5.0 -> 3.93 s. Tried without gain: 16 weight rows per pass (partial traffic halved), a packed
  [K/8][N][8] linear-stream layout (standalone test 97 vs 95 GB/s). CPU streaming ceiling here ~124 GB/s
- [x] CPU encoder: vectorized GELU epilogue (`erfc4` / `geluErf4` in neon_math.h, Numerical Recipes' erfcc with an
  exact-square split: rel 6.0e-7; 5.7e-7 vs the f32-argument erfc the basic kernels use; tests/unit neon_math) and a
  batched AMX audio CNN on the Mac (conv weights f32 once per call, im2col straight to f32, 4 chunks per sgemm,
  conv_out over all chunks). enc 30 s 400 -> 320 (GELU) -> 281 ms; audio_cnn 160 -> 92.5, fc1 GEMMs 121 -> 70
- [x] CPU large-M GEMMs on BNNS (`BNNSMatMul` f16 x f16 -> f32 on the AMX, f32 accumulation; deprecated API, still
  in the macOS 27 SDK; the convert + sgemm path stays as fallback): no f32 weight copy, 2.3-2.5 TF/s (qkv T = 401
  4.91 -> 2.76 ms, gate|up 14.7 -> 8.28, down 6.49 -> 3.98). prefill 30 s 866 -> 552 ms, enc 30 s 281 -> 238.
  Audio CNN on BNNS too (conv1, K = 9, on the NEON tiles: 86 vs 105 ms; 2 chunks per GEMM, A/B 1/2/4/8 = 83 /
  82 / 86 / 152 ms); conv_out always on BNNS so a chunk's output is the same bit for bit whatever the chunk count
  (test_audio_cnn_opt's exact check caught the M-dependent engine choice). enc 30 s now ~214 ms (from 400)
- [x] CPU GEMV: two column blocks for very wide W (gate|up, 24,576 columns): 103 -> 108 GB/s (4 blocks: 101). Long-
  context CPU decode attention (C = 15,600: 28.9 ms of a 62 ms step, ~62 GB/s): more key splits did not help (2/4/8/16
  tasks per thread 28.9-32.1 ms): per-key compute + memory not overlapped; open
- [x] CPU prefill attention on the AMX (Mac, T >= 64): per (kv head, 64 query rows), key tiles of 256, QKᵀ and PV as
  sgemms (8 concurrent small sgemms: 2.2 TF/s vs ~0.6 for the NEON flash path), online f32 softmax unchanged. prefill
  20 min 83.4 -> 48.3 s, 300 s 8.4 -> 6.5, 111 s 2.2 -> 1.96. All attention tests (incl. slow) pass
- [x] CPU long-context decode attention (Mac, >= 1024 keys): tasks split the keys only and feed every kv head per
  64-key tile (whole 2 KB cache rows read in order instead of 256 B head slices 2 KB apart): attention at C = 15,600
  28.9 -> 22.2 ms; decode 20 min 70.7 -> 63.8 ms, 300 s 42.2 -> 39.9. (A fused in-register V conversion: no gain.)
- [x] CPU timing noise: on 8 pool threads over 8 P cores, any other busy process (the editor's renderer) delays every
  barrier; CPU medians drift 10-15 % above p10 at times. Measured 2026-11-07 (qwen3-asr.md "CPU decode GEMV
  streaming"): 7 pool threads -10 % always (rejected); 2 K-split tasks per thread +1-2 % quiet, -7-9 % with 8 busy
  user-interactive threads (decode 30 s 40.7 -> 37.9 ms); 1-4 busy threads cost nothing (they go to the E-cores).
  Kept 1 task per thread; rule instead: record CPU rows only at a 1-min load average < 2 (+ build lock)
- [x] CPU decode vs llama.cpp CPU (8 threads too): 33.3-34.0 vs 29.8-31.7 ms. llama.cpp's f16 dot products accumulate
  partly in f16 (fewer instructions per weight); ours must accumulate in f32 (contract). 2026-11-07: not the
  accumulation but the access pattern: the M = 1 GEMV now reads 2 W rows × 64 columns per step (2 streams per core,
  16 f32 accumulators; was 8 rows × 8 columns): 104 -> 115.5 GB/s of the 124 GB/s P-core ceiling, bit-identical
  outputs; LM head 4 -> 2 rows per pass (5.6 -> 4.9 ms). In-model A/B: decode 2 s ~33.9 -> ~31.3, 30 s ~36.7 ->
  ~31.1 ms; dashboard (quiet, load 1.9) 29.9 / 30.4 ms (llama.cpp 29.8 / 31.7). Left: llama.cpp's [N, K] dot layout reads ~1-2 % faster still (needs [N, K]
  weights in every prefill path)
- Night of 2026-11-05: the dashboard's short cases (2-30 s, both devices) re-recorded with every change above, display
  asleep (the aerial wallpaper's GPU use made earlier GPU timings noisy: prefill 10 s read 64-86 ms). With the display
  off the Mac also idle-sleeps (`sleep 1`): runs paused for up to ~14 min mid-sweep (perf_counter stops, wall clock
  does not); overnight work now runs under `caffeinate -i -s`. `/tmp/lokinfer-macbench.lock` vanished once while
  held (nothing in the repo deletes it; another session?): lk-lock recreates it, but two holders could then overlap
- [x] **iPhone memory: `fc-thrashing` jetsams** (2026-11-06; addressed 2026-11-07): with the phone short of RAM (free ~0.27 GB, compressor
  ~5.6 GB of other apps), iOS evicts the 4 GB of mmapped weights and re-reads them from flash every decode step
  (~3 s/step instead of 0.1 s), then kills processes (JetsamEvent reason LokiHost, kills `fc-thrashing`). Seen on CPU
  e2e 300 s. Every observed case followed a GPU session whose 4.3 GB of GPU allocations stayed (fixed above, the
  residency trim). 2026-11-07 (memory agent; docs/models/qwen3-asr.md "Memory under pressure"):
  - [x] KV cache sized to the pass: already everywhere (bench: min(MAX_CAP, prompt + max_new rounded up to 1024);
    tests: per case; `transcribe`: prompt + max_new)
  - [x] reproduced on purpose: `bench/models/pressure_qwen3asr_iphone.py` (decode steps timed while a balloon of
    random device tensors fills RAM). CPU graph unpinned: 2.8-10.6 GB of page-ins per 20 steps from the first 0.5 GB
    of balloon, max step 2.1-2.7 s. GPU graph: no page-ins up to 3.5 GB (its weights are resident Metal buffers,
    wired while GPU work runs, not charged to the footprint)
  - [x] pressure reported to the client: MEMORY + `available` (os_proc_available_memory), `pressure` (dispatch
    memory-pressure source), `pressure_events`, `pageins`, `weights_resident` (mincore), system free / file / compressor
    / wired; `RemoteDevice.memory_check()`; the iPhone Qwen bench warns and never records a case under pressure or
    with more page-ins than the weights' size; the test harness warns at connect
  - [x] PIN_MODEL (feature "pin_model"): mlock works on iOS (3.44 GB wired, 0 page-ins under the balloon) but the
    wired pages count in phys_footprint (+3.2 GB against the ~6 GB app limit), so it trades thrashing for jetsam
    risk. Mode "gpu" (a residency set of its own + requestResidency): wired without footprint on the Mac, but on
    the iPhone (2026-11-07 18:47) charged like mlock (+3.2 GB) while stopping the page-ins as well: on iOS any pin
    costs the weights' size in the app budget (CPU e2e 300 s then peaks ~4.3 GB of ~6: fits)
  - [x] (phone) `pressure_qwen3asr_iphone.py --device cpu --pin none gpu` and test_remote.py --device=iphone on
    the final build (2026-11-07 18:47-18:49)
  - [x] (phone, 21:18) A/B `bench_qwen3asr_iphone.py --device cpu --quick --pin none / gpu`, alternating: no cost
    (gpu as fast or faster in each pair); the bench now pins its CPU graphs by default (`--pin auto`); the test
    harness doesn't (headroom). The 46 s e2e 30 s of 19:13 re-checked: 5.1-5.4 s, an artifact of the phone dropping
  - [x] (phone, 21:22-21:40) iPhone per-case table (`measure_qwen3asr_memory.py`, qwen3-asr.md "Memory per case"):
    CPU footprint ≤ 0.64 GB (e2e 300 s, golden), GPU ≤ 0.17 GB; resident up to 4.47 GB (CPU, mapped weights)
  - [x] (phone, 21:50-21:55) Qwen model tests `--device=iphone --run-slow` 56 passed + 4 skipped, test_remote.py
    `--device=iphone` 17 passed + 3 skipped, on the final build. Left: only the 1-row items below (WRITE_TENSOR
    payload) and, if ever wanted, pinning in the test harness / an app-level "pin when the budget allows" policy
  - [x] per-case memory (Mac, `measure_qwen3asr_memory.py --where mac`; qwen3-asr.md "Memory per case"): CPU
    footprint = written KV + 0.1-0.25 GB of arenas / scratch (e2e 300 s peak 0.84 GB, iPhone 0.86); GPU ≤ 0.18 GB.
    The 2048-row prefill arenas add ≤ 0.2 GB: not shrunk (≤ 4 % of the working set, prefill speed risk); the
    weights' eviction is the problem
  - [ ] WRITE_TENSOR / PUT_BLOB hold the whole payload in a buffer: a 512 MiB write raised LokiHost's footprint
    ~1 GB transiently (big uploads only; read straight into the tensor / file if it matters)
- [x] iPhone CPU: re-measure kGemvTasksPerThread (was 3) + LM-head rows + kSmeMinM (2026-11-07, one deploy per
  setting, alternating, cool phone; qwen3-asr.md "CPU decode GEMV streaming" / "Q5 CPU kernels"): decode step C = 480
  tpt 1 / 2 / 3 = 48.7 / 40.8-42.5 / 49.4-50.1 ms -> 2; LM head 2 vs 4 rows 6.8 vs 7.0 ms -> 2; SME2 from M 5 (wins
  1.6-4x over the NEON tiles at every M 5-128; was 16). New kernels + final setting validated on the phone (fused CPU
  kernels + Qwen CPU model tests, slow tiers)
- [ ] iPhone: re-measure the GPU tile / GEMV-group rules (the CPU half of this item is done above)
- [x] GPU decode: rmsnorm folded into the next GEMV: new ops `rmsnorm_gemm_opt` / `rmsnorm_gemm_swiglu_opt` (CPU +
  GPU; contract doc updated). M = 1 on the GPU: one dispatch (`lk_f16o_rms_gemv`, each threadgroup normalizes the row
  with lk_f16_rmsnorm's exact sum order); else rmsnorm into a workspace output + the GEMM. Bit-identical to the
  composition (tests/kernels/test_f16_rms_gemm.py, 36 cases). Back-to-back A/B decode C = 480: 12.64 / 12.67 ->
  11.92 / 12.07 ms
- [ ] GPU decode: down / qkv / o GEMVs at ~300 GB/s vs swiglu 335 and lm_head 376
- [x] GPU prefill at short T: 64x64 steel tiles only from M >= 1024 (`kOptMin64Rows`, Mac): prefill 10 s 77.5 ->
  62.6 ms (llama.cpp Metal 68), 30 s 162 -> 145 (167), enc 30 s 56.7 -> 55.5, enc 300 s ~611 -> 540; a 2048-row
  chunk keeps 64x64 (736 vs 826 ms with 32x32). Threshold re-checked at M = 789 (prefill 60 s) with alternating runs
  (GPU timings drift ~10 % with heat between separate runs: only alternate A/B counts): 32x32 278-294 ms vs 64x64
  291-335, so 1024 stays
- [x] bench_qwen3asr.py: the KV cache is reused while big enough (it was reallocated + zeroed, 2.9 GB, whenever the
  capacity bucket changed: e2e 30 min's two passes alternate buckets)
- [ ] Note: prof-lokinfer's GPU profile shows ~0.9 ms of gaps at segment boundaries that unprofiled calls don't have
  (e2e 12.74 vs kernel sum 12.5 ms): a profiling artifact

### Iteration speed for large models (user, 2026-11-06: "being able to iterate / bench / profile quickly over large models is important"; playbook: docs/big-models.md)
- [x] Cost of `RemoteDevice.load(CompiledModel)` when every blob is cached (Qwen3-ASR generate, measured 2026-11-06):
  build 1.5 s + .lkm write 1.1 s + sha256 1.4 s (now 8 threads: 0.2 s) + 226 HAS_BLOB round trips (now one BLOBS
  query, 2026-11-06). Split straight from the in-memory model (2026-11-06: `_lokinfer.lkm_split` = C++ `lkmSplit`:
  the skeleton + zero-copy views of the constants; no .lkm written, no 3.4 GB in-memory copy; byte-identical to the
  file's split, tests/python/test_remote.py::test_lkm_split_matches_saved_file). Left: the graph build itself (1.5 s)
- [x] prof-lokinfer on the iPhone with mutable inputs: allocate them as device tensors and profile through RUN2
  (today: a hand-written set_profile / run2 / take_profile script) (2026-11-06: collect.py's iPhone path holds every
  input >= 1 MB as a device tensor, zeroed ones never sent, and times / profiles through RUN2)
- [x] tests/conftest.py transparent remote path: keep a device-resident copy per mutable host array (re-upload only
  when the host array changed: checksum), and build the result objects without re-running the graph on the Mac
  (2026-11-06: `_mirror` per argument slot / dtype / shape, blake2b of the last synced contents, LRU <= 1 GiB; the
  local run happens once per model, to learn the outputs' devices; read-back of mutable inputs stays)
- [x] (2026-11-06) Debug builds compiled shaders at -O0, where lk_f16o_gemm_swiglu_64x64 fails pipeline creation
  ("exceeds available stack space"): build/lokinfer-serve and `--device=serve` could not run Qwen3-ASR on the GPU.
  cmake/LokiMetal.cmake: always -O3 (cache var LOKI_METAL_OPT for a one-off -O0)
- [x] (2026-11-06) One LokiHost crash failed every later test of the session (108 tests on the dead connection):
  `RemoteDevice.reconnect()` + the harness reconnects before the next test (tests/conftest.py)
- [x] (2026-11-07) `--device=iphone` sessions piled up every test's models until iOS killed LokiHost: FREE_MODEL
  message (18, "free_model"); the harness frees each test's models by id (`LOKINFER_TEST_MEMLOG=1` prints the
  executor's memory after every test)
- [x] (2026-11-06) Test fixtures leaked `lokinfer-serve` processes when a call or shutdown failed (7 piled up; the
  next runs timed out on gpu_inspect): the fixtures now kill the server in `finally`
- [x] `--quick` mode for bench_qwen3asr*.py (1 warm-up + 5 runs, never recorded) for A/B during development; the full
  decision-11 protocol only when recording (2026-11-06: `--quick` on bench_qwen3asr.py and bench_qwen3asr_iphone.py,
  3 runs for calls > 10 s, iPhone e2e one pass; refuses `--record`. Mac GPU decode 2 s + enc 5 s + e2e 5 s in 8.5 s)
- [x] Remote `clear_blobs` + blob storage report (the phone accumulates superseded weights and whole-model blobs)
  (2026-11-06: message BLOBS (17), feature "blobs": `dev.blobs()`, `dev.clear_blobs(keep=...)`,
  `dev.remove_blobs(shas)`; tests/python/test_remote.py; the phone held 1,390 blobs / 4.79 GB). Mode 3 (feature
  "blobs_query", `dev.stored_blobs(shas)`): which blobs are stored in one round trip; the split upload uses it instead
  of one HAS_BLOB per constant
- [x] iPhone benches: thermal gating per device (cool to "nominal"/"fair" before each device's sweep; today the CPU
  sweep ran at "serious" after an hour of tests) (2026-11-06: per case, not per device: `bench_qwen3asr_iphone.py
  --cool-to nominal|fair --cool-max-wait S` (default nominal, 600 s), the wait is printed and the state recorded)

### Speech-to-text in lokinfer — NEXT STEP (designed 2026-11-06, not started; the user starts it)
Design + decisions: docs/impl-notes/speech-to-text-design.md; research: docs/impl-notes/research-speech-to-text.md.
**Ultimate goal (user): the full E2E binary faster than the PyTorch / MLX pipelines end to end (warm: model loaded,
call → text), with the same output as today's Python path (english-practice: transformers, English, greedy,
transcription_only).** Mac first, English only, Qwen3-ASR only with generic shared layers.
- [ ] Shared audio layer: WAV reader (FLLR / EXTENSIBLE / 8-16-24-32-bit / float), mean downmix, Kaiser-sinc polyphase
  resampler (kaiser_best constants; tests vs torchaudio ~1e-6), parameterized log-mel (≤ 1e-5 vs the HF extractor),
  quiet-point splitter (= reference.split_long)
- [ ] Shared text layer: byte-level BPE tokenizer in C++ (binary converted from tokenizer.json; NFC; Qwen2
  pre-tokenizer; tests + fuzz vs the Rust `tokenizers`), streaming detokenizer, generation loop (chunked prefill,
  session-owned KV cache, eos / max / capacity / cancel stops)
- [ ] Device-side argmax in the decode graph (4 bytes read back per token); token id fed back on the device;
  pipelined decode (step N+1 enqueued before reading token N), then K steps per commit with the ICB replay
- [ ] models/qwen3_asr: `SpeechToText::load` / `transcribe(pcm, frames, rate, channels, options)` (prompt ids,
  N_a, chunk layout, chunked prefill, `max_new_tokens = 128 + 8·s`, transcription_only parsing, capacity-based split);
  a features-in path for golden tests; Python mirror
- [ ] Model cache: `--model <name>` downloads the pinned HF snapshot and converts it once (C++: HTTP download,
  safetensors reader, weight folding, tokenizer binary) into `~/.cache/lokinfer/models/<name>`
- [ ] `utils/speech-to-text` binary: `--input FILE|-`, `--raw-f32 --rate --channels`, `--model`, `--device`,
  `--max-pass-seconds`, `--max-new-tokens`, `--format txt|json`, `--timings`
- [ ] Correctness: generated ids = transformers greedy `generate` on the golden clips (near-tie rule), text =
  `decode(transcription_only)`; > 20 min in one pass vs split (accuracy check)
- [ ] E2E benchmark (warm; 2 s … 30 min + 48 kHz stereo) vs qwen-asr / transformers MPS + CPU, mlx-audio Python +
  Swift, llama.cpp mtmd; on the dashboard; lokinfer must win
- Later: speed modes on top of the configurable options (deliberate splitting with parallel / batched passes,
  decoding beyond greedy, e.g. speculative), benchmarked as their own category, not against the reference-matched rows
- Later: iPhone binding, english-practice switch, timestamps, VAD, other languages, cold start

### Q8 — Docs
- [x] (2026-11-07: all listed gaps closed — perf-dashboard.md, baselines.md, iphone-baselines.md, status header,
  per-kernel profiles, the "Memory" section) `docs/models/qwen3-asr.md` (contract, gate results, correctness, memory, gap, bf16 side measurement),
  `docs/correctness-testing.md` (f16 calibrated-bound rule), `docs/perf-dashboard.md`, `docs/baselines.md`,
  `docs/runtime.md` if touched, `README.md` (models list) (2026-11-06: README, correctness-testing.md, baselines.md § 8,
  qwen3-asr.md mostly done; missing: perf-dashboard.md never mentions Qwen3-ASR; baselines.md table row 27 still says
  "not on the dashboard yet"; qwen3-asr.md status header says iPhone model tests / baselines / timings pending, no
  per-kernel profile or 300 s memory numbers; iphone-baselines.md llama.cpp section still says "no iPhone timing yet")
