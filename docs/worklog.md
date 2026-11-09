# Work log

Append-only diary, newest last: timestamped entries (`HH:MM–HH:MM (duration)`) of everything done — dev work
(reading code, investigating, each hypothesis tried and its outcome, writing code, debugging, decisions and why,
docs) as well as machine runs (builds, deploys, benchmarks, tests) — with where the details live (user, 2026-11-06 /
2026-11-07). Status and open items stay in [todo.md](todo.md); results in the
model / runtime docs.

## 2026-11-06 (Qwen3-ASR phase 1: iPhone validation, measurements, tooling)

Afternoon:
- GPU memory kept after a model was freed (4.3 GB `gpu_allocated` with 0 models on the phone): residency-set removals
  were only committed at the next encode. Fix: `GpuContext::trim()`, called by the server on disconnect and
  FREE_TENSOR (runtime.md "Residency commits"). Mac repro 3.57 GB → 11 MB.
- iPhone CPU e2e 300 s recorded (113.6 s, hot); GPU e2e 300 s 46.9 s (was 69.96 s while the phone was short of
  memory). Transcripts golden on both.
- Todo audit (agent): 11 phase-1 items ticked with evidence.
- Kokoro / GPT-2 goal guard re-checked by re-timing Kokoro CPU + GPU on the Mac: no regression (app71 outliers were
  noise: 55.5 vs 54.7 ms in isolation).
- iPhone baselines, long clips (60 / 111 / 300 s): llama.cpp Metal, MLX Swift, llama.cpp CPU (11 cases; e2e 300 s
  left). Every implementation's 5 min transcript has the same f16 near-tie (a comma after "imagination", token 48):
  `record_qwen3asr_iphone_baselines.py --max-word-diffs N` (word / token diff in each row's notes).
- llama.cpp CPU long batch hung once mid-encode (8 min no output) on a hot phone; restarted; a hot e2e 300 s call took
  ~9 min, stopped.
- `bench_qwen3asr.py`: e2e rows record the token count and first difference when not golden.
- `lokinfer basic` on the Mac: every case ≤ 300 s, then CPU enc / prefill / decode 1200 s and GPU enc 1200 s. GPU
  decode / prefill 1200 s can't run on basic (a > watchdog dispatch). Fused vs basic table in qwen3-asr.md.
- Per-kernel profiles, 30 s clip, all 4 devices (qwen3-asr.md "Per-kernel profile"); profiler FLOP counts fixed for
  `rmsnorm_gemm*` (weight = 3rd input).
- iPhone CPU fused-kernel tests, slow tier: 106 passed, 2 skipped; no hang.
- Docs sweep (agent): README, baselines.md, perf-dashboard.md, correctness-testing.md, profiling.md.

Evening (Mac-side while the phone was away):
- Remote BLOBS message (17, features "blobs" / "blobs_query"): report, clear, remove, "which are stored" in one round
  trip. A client bug (mode 3 sent to an older server) fixed by the separate feature flag.
- `--quick` bench mode (1 warm-up + 5 runs, never recorded).
- prof-lokinfer on the iPhone: inputs ≥ 1 MB held as device tensors, timing / profiling through RUN2.
- tests/conftest.py remote path: device mirrors per mutable argument (re-sent only when changed), the Mac run only
  on a model's first call.
- Debug builds compiled shaders at -O0 (steel GEMM pipeline creation failed: "exceeds available stack space"):
  cmake/LokiMetal.cmake always -O3.
- Test fixtures leaked `lokinfer-serve` processes on failures (7 piled up): killed in `finally`.
- `RemoteDevice.load(CompiledModel)`: split in memory (`lkm_split`), no .lkm written, no 3.4 GB copy; byte-identical
  test.
- A VS Code restart left the old Claude Code process running the same task in parallel for ~1 h (orphan, PPID 1);
  stopped it, checked its edits (one todo item rewritten).
- Mac default tier green: ctest 71/71, Python 1322 passed, models / e2e / perf 147 passed.

Night (phone back, autonomous):
- `bench_qwen3asr_iphone.py --cool-to / --cool-max-wait`: cool-down before every case.
- 21:44–22:20: every lokinfer iPhone row re-measured from "nominal" (CPU 24 cases, GPU 10): CPU decode 40–72 ms (was
  66–105 hot), CPU e2e 300 s 102 s, GPU decode 32–38 ms. One fresh iPhone table in qwen3-asr.md.
- 22:20: iPhone GPU kernel bench rows completed (qwen3-asr.md "Q5 GPU kernels").
- 22:34: llama.cpp CPU e2e 300 s on the phone after a 10 min idle cool-down (running).
- 22:25–22:45 (20 min): docs — the fresh iPhone comparison table, the iPhone kernel-bench rows, `worklog.md` and
  `time-accounting.md` (per-work-effort reference + where the time went, iPhone vs Mac).
- 22:34–23:00 (26 min incl. 10 min cool-down): llama.cpp CPU e2e 300 s on the iPhone: 221 s median (one call 565 s,
  throttled), transcript 4 words off (the near-tie). Recorded; lokinfer CPU is 2.2× faster (102 s).
- 23:00–23:02 (2 min): old Kokoro / other baseline data pushed back to the phone (126 MB cases + models + 2.7 GB other).
- 23:02: LokiHost redeployed, 71/71 on-device tests.
- 23:02–23:10 (8 min): docs — qwen3-asr.md table cell, todo (phone leftovers all done), time-accounting night rows.
- 23:10–23:25 (15 min): `bench_f16_fused_gpu.py --cpu` (the same 25 cases on the CPU ops: the Q5 CPU kernel table,
  Mac + iPhone) and `bench_qwen3asr_iphone.py --variant basic` (basic on the phone, timing only). Queued: the CPU kernel
  tables (Mac, then iPhone after cooling), then basic on the iPhone for clips ≤ 30 s (GPU, then CPU).
- 23:04–23:11 (7 min): CPU kernel tables, basic vs opt — Mac 2.4 min (GEMMs 3–14.5×, prefill attention 6–10×), iPhone
  4.3 min (GEMMs 2.4–7.5×, attention 1.4–3.7×, audio_cnn 4.8–6.1×). qwen3-asr.md "Q5 CPU kernels" rewritten (the
  2026-11-05 table predated BNNS / vector GELU).
- 23:11–23:24 (12.5 min incl. 62 s first load / basic weight upload): `lokinfer basic` on the iPhone GPU, 14 cases
  ≤ 30 s, from "nominal" each: e2e 5 s / 30 s 3.4 / 15.1 s (fused 0.87 / 3.34 s: 3.9 / 4.5×), transcripts golden.
- 23:24–23:41 (17 min): `lokinfer basic` on the iPhone CPU, 14 cases ≤ 30 s: e2e 5 s / 30 s 3.9 / 20.8 s (fused 1.95 /
  7.04 s: 2.0 / 3.0×), transcripts golden. Fused-vs-basic table now has iPhone columns.
- 23:41–23:47 (6 min): docs — fused-vs-basic iPhone columns, todo.
- 23:47: Mac slow tier started (Qwen 60 tests, then the rest of 1541), the end-of-task full validation.
- 23:41–23:48 (6 min): Mac slow tier green — Qwen 56 passed (1.5 min), everything else 1477 passed (4.4 min).
- 23:48: Mac long tier started (the 4 Qwen 1200 s tests).
- 23:48–23:51 (3 min): Mac long tier green (4 Qwen 1200 s tests: CPU 84 + 60 s, GPU 21 + 15 s). Mac fully validated.
- 23:52: iPhone validation of the new test-harness path started (fused GPU kernels + Python suite, slow tier), then the
  Qwen model slow tier on the phone.
- 23:51–23:55 (3.3 min): iPhone fused GPU kernels + Python suite, slow tier: LokiHost killed (signal 9, no jetsam
  report) during `test_attention_gqa_opt[25600-13570-2048-16-8-128]`; the 108 tests after it failed on the dead
  connection (cascade). Investigating (re-run alone queued).
- 00:00–00:12 (12 min): test harness reconnects after a server crash: `RemoteDevice.reconnect()` (same object, new
  connection, `session` counter; the watchdog relaunches LokiHost) + an autouse fixture and `remote_call` reconnect a
  broken session before the next test, model ids keyed by session, mirrors dropped. One crash now fails one test, not
  every test after it. tests/python/test_remote.py::test_reconnect_in_place.
- 23:55–23:59 (4 min): Qwen model slow tier on the iPhone through the new harness: 56 passed.
- 23:59–00:01 (1.3 min): `test_attention_gqa_opt` alone on the phone: 23 passed incl. both large cases → the kill was
  cumulative (all of a session's models stay loaded until disconnect, plus up to 1 GiB of the new mirrors). Mirror
  cap lowered to 256 MiB; the full kernel + Python run on the phone re-started (00:02).
- 00:01–00:05 (3.6 min): the full run again with 256 MiB mirrors: 234 passed, 1 failed — a second LokiHost kill (window
  attention S = 15,600), but the reconnect limited it to that one test. The models still pile up within a session.
- 00:05–00:12 (7 min): the harness now frees each test's models (a reconnect after any test that loaded one: the
  server frees a connection's models when it ends; the weights stay cached on the phone). `--device=serve` 217 passed
  (24 s). Phone re-run started 00:12.
- 00:12–00:33 (20.5 min): phone re-run with a reconnect after every model-loading test: 235 passed, 0 kills — but 5.7×
  slower (3.5 → 20.4 min: the connect over the phone tunnel costs seconds × ~235 tests).
- 00:33–00:36 (3 min): only free when the executor holds > 1 GiB (`dev.memory()` after the test: one round trip).
  serve 217 passed; phone re-run started 00:36.
- 00:36–00:41 (4.9 min): phone re-run with threshold freeing: 235 passed, 0 kills, 16 reconnects. Docs: python-api.md,
  big-models.md (harness memory behaviour). Qwen slow tier on the phone re-started with the final harness (00:42).
- 00:42–00:46 (4.2 min): Qwen slow tier on the phone with the threshold harness: 27 failed ("unknown device tensor"):
  the threshold reconnect freed the Qwen module fixture's own models / KV tensors. Reconnecting can't tell whose state
  it frees → replaced.
- 00:46–00:55 (9 min): FREE_MODEL message (18, feature "free_model", `dev.free_model(id)`): the harness frees exactly
  the models each test loaded through it (weakrefs; cached ids dropped), fixtures' own state untouched; no reconnect
  except after a crash. Local 19 passed, serve 218 passed. Deploy + both phone suites started 00:55.
- 00:55–00:59 (4 min): deploy (17 s, incremental) + phone suites: Qwen 56 passed; fused GPU + Python 235 passed, 1 kill
  (window attention S = 15,600).
- 00:59–01:05 (6 min): `LOKINFER_TEST_MEMLOG=1` (the executor's memory after every test) showed GPU allocations growing
  to 2.8 GB with models never freed: the harness kept weakrefs to the models and freed through them, but tests build
  models as temporaries, gone before teardown → ids lost. Fix: record the id at load, free by id. Mac: models 0 after
  every test, GPU 31 MB; phone: footprint peak 1.07 GB (was 3.1 GB).
- 01:05–01:15 (10 min): final phone + Mac runs: fused GPU + Python on the phone 236 passed, 0 kills, 3.6 min; Qwen
  slow tier on the phone 56 passed (4.1 min); Mac `--device=serve` (python + all kernels) 1221 passed; Mac local
  1324 passed.
- 01:17: full default-tier suite through the phone started (1543 tests), to validate the harness changes on every
  model / kernel test.
- 01:03–01:16 (13 min): full default-tier suite through the phone: 1364 passed, 1 failed, 0 kills.
- 01:16–01:35 (19 min): the failure = Kokoro `test_free_running_durations_match[app47m]` on the iPhone (36.7 dB vs a
  37.5 limit; Mac 37.8). Probably never passed on the phone: the only earlier phone run of this test was app59f alone,
  and the documented 37.8 matches the Mac (corrected 2026-11-07 after the user asked; first stated as a regression). With `LOKINFER_KOKORO_LSTM_T=1`: app47m 37.9 passes, real_speed125 41.8 fails (limit
  42.1). Documented in todo.md for a user decision; test not relaxed.
- 01:35–01:40: time-accounting.md night totals. End of the night's queue: nothing running.
- 07:39: Kokoro iPhone app47m SNR case: user decision — ignore for now (todo marked).
- 07:42–07:47 (5 min): closed Q8 (checked every listed doc gap: all fixed overnight except the 300 s memory numbers →
  new "Memory" section in qwen3-asr.md) and Q7 basic (ticked with its documented exceptions).
- 07:48: Kokoro app47m iPhone SNR item dropped from todo.md (user: "you can just drop it"); the test itself is
  unchanged (it still fails on the phone, passes on the Mac).
- 07:44–07:44 (~8 min) [coordinator]: user: "next tackle perf work now and iPhone memory pressure", "work in parallel,
  sub agents". Launched 3 agents, split by file area so they don't collide: [gpu-gemm agent] iPhone GPU f16 GEMM via
  Metal 4 tensor ops (kernels/gpu); [memory agent] iPhone memory pressure (src/remote, LokiHost, KV/arena sizing);
  [cpu-decode agent] Mac CPU decode vs llama.cpp + iPhone CPU GEMV + timing noise (kernels/cpu). Shared resources via
  the phone / big-model / Mac-bench / build locks; every agent logs here (append-only) with its tag.
- 07:44–07:58 (14 min) [memory agent]: read the brief's docs (big-models.md, qwen3-asr.md Memory / iPhone, runtime.md
  Memory + footprint table, todo fc-thrashing item), server.cpp (MEMORY, Session, endClient), remote.py, the iPhone
  Qwen bench, model.py (MAX_CAP / kv_cache), LokiHost main.mm + entitlements, measure_memory.py. Findings: the bench
  and transcribe() already size the KV cache to the pass (min(MAX_CAP, roundup(prompt + max_new, 1024))); the only
  existing per-case number is e2e 300 s (CPU footprint peak 0.86 GB incl. the 0.73 GB KV). Hypothesis for (4):
  file-backed weight pages are the cheapest thing for iOS to reclaim, so under pressure it evicts them (thrash) before
  jetsamming background apps; wiring them (mlock) or holding them as dirty anonymous memory would move the pressure
  onto other apps instead. To verify on the device.
- 07:44–07:52 (8 min) [cpu-decode agent]: read the brief's docs (big-models.md, qwen3-asr.md Q5 CPU / profile / Mac
  perf, the f16 opt-kernel contract, todo Perf items) and the code (gemvKSplit / gemvRange, runRmsGemmOpt,
  runLmHeadLastOpt, thread_pool.cpp). Decode step = 3.44 GB of f16 weights (28 × 100.7 MB layer GEMVs + 622 MB LM
  head): 33.3 ms = 103 GB/s, llama.cpp 29.8 ms = 115 GB/s. Hypotheses to measure, not theorize: (a) the CPU's real
  streaming ceiling (todo says ~124 GB/s; re-measure with a standalone read bench, contiguous vs our strided K-split
  pattern), (b) per-GEMV fixed costs: 2 parallelFor barriers per GEMV (K-split + reduction) × ~170 GEMVs per step,
  (c) layout: llama.cpp splits N with contiguous [N][K] rows per thread (no reduction pass). Swap already 5.8 GB used
  at start.
- 07:44–08:02 (18 min) [gpu-gemm agent]: iPhone GPU prefill gap (todo "~3.4x MLX Swift's"): read big-models.md,
  qwen3-asr.md (Q5 GPU kernels, profile, iPhone), the opt-kernel contract, f16_fused.metal / f16_gpu.cpp (steel GEMM
  dispatch), research notes on Metal 4 tensor ops. MLX 0.33 NAX GEMM (steel/gemm/nax.h, gemm_nax.h, matmul.cpp):
  no threadgroup staging; each simdgroup loads 16x16 half fragments straight from device memory into registers and
  runs `mpp::tensor_ops::matmul2d<(16, 32, 16), execution_simdgroup>` on cooperative tensors (f32 destination);
  phone config ('p' arch) 128x128 tiles, 4x4 simdgroups (32x32 per simdgroup), BK 512 (one mem_none barrier per BK),
  K step 32; gated on OS >= 26.2 and arch gen >= 18 (phone) / 17 (Mac). Note MLX passes relaxed_precision = true;
  plan: relaxed_precision = false (the contract's f32 accumulation), measure true only as a side note. Plan: new
  kernels/gpu/f16_tensor.metal with the same fused epilogues (bias, GELU, residual, clamp, SwiGLU with gate and up
  accumulators sharing the A fragments), dispatch for M > 1 when the GPU has neural accelerators (Apple10 family) or
  LOKINFER_F16_GEMM=tensor (Mac testing / A/B).
  (correction: that entry ran 07:44–07:46, 2 min, not 08:02.)
- 07:48–07:58 (10 min) [memory agent]: (correction: the entry above ran 07:44–07:47, 3 min, not 14.) Coded the
  pressure reporting + weight pinning in src/remote/server.cpp: a process-wide dispatch memory-pressure source
  (level + warn/critical event count); MEMORY gains available (os_proc_available_memory, iOS only, -1 on macOS),
  pressure, pressure_events, pageins (task_events_info: the thrash signal, a weight re-read from flash is a page-in),
  weights / weights_resident (mincore over the loaded models' mapped constants) / weights_locked, system_free /
  _file_backed / _compressor / _wired (host_statistics64). New PIN_MODEL message (19, feature "pin_model"): munlock /
  madvise WILLNEED / mlock (all or nothing) the model's constant pages. Python: RemoteDevice.pin_model,
  memory_check. Fuzz test now covers types up to 19 (skips SHUTDOWN / BLOBS / FREE_MODEL). Mac: unit [remote] incl.
  fuzz pass, test_remote.py 20/20 local, 17 + 3 skipped --device=serve.
- correction [cpu-decode agent]: my first entry "07:44–07:52 (8 min)" was written ahead of time; the real span was
  07:44–07:46 (2 min). From now on start/end times come from `date` at the step's start and end.
- correction [memory agent]: my two entries above had estimated end times. Real clock: reading 07:44–07:47 (3 min);
  coding the pressure reporting / PIN_MODEL + Mac tests 07:47–07:49 (2 min, not "07:48–07:58"). From here on every
  entry uses `date` at start and end.
- 07:46–07:50 (4 min) [cpu-decode agent]: baseline profile (prof-lokinfer decode C=480, Mac CPU): e2e 34.6 ms;
  swiglu 14.2 ms (99 GB/s), gemm_add 9.3 (101), lm_head 5.6 (110), qkv 4.3 (108). Standalone scratch microbench
  (10 layers of distinct Qwen decode weights, 1 GB, repo thread pool): pure 8-thread read ceiling 124 GB/s (4 thr
  108, 7 thr 116-119, 10-12 thr incl. E-cores 104-124: E-cores add nothing). Repo K-split GEMV (8 W rows per pass)
  102-104 GB/s. Hypotheses tested, alternating rounds: rows per pass 1 / 2 / 4 / 8 -> 103 / 110-111 / 100-104 /
  102-104 (2 rows best); 2 rows x 32 columns per iteration 111.5; llama.cpp-style dot layout [N][K] with 2 rows
  110-117 (needs a weight-layout change for every prefill path: out of scope unless needed); column-strip (128 B per
  row, registers) 49 (bad); N-split / 2D column blocks over [K, N] 63-90 (bad: strided); prefetch 4 groups ahead 75
  (bad); 2 tasks / thread with 2 rows 108. Load average ~4 (other agents) -> min values noisy, medians stable.
- 07:50–07:53 (3 min) [memory agent]: wrote bench/models/measure_qwen3asr_memory.py (per (graph, kind) a fresh
  LokiHost, clips ascending, MEMORY after each case incl. page-ins / weights resident / pressure) and
  bench/models/pressure_qwen3asr_iphone.py (decode steps timed while a balloon of random device tensors squeezes
  free memory, unpinned vs PIN_MODEL lock). bench_qwen3asr_iphone.py: warns and never records a case that starts /
  ends under memory pressure or pages in more than the weights' size. Checked: KV caches are already sized to the
  pass everywhere (bench: min(MAX_CAP, roundup(prompt + max_new, 1024)); tests: per-case capacity; transcribe:
  prompt + max_new), so that part of the todo is done. Deploy waiting on the phone lock (another agent holds it).
- correction [memory agent]: the entry above ended at 07:52 (2 min), not 07:53.
- 07:50–07:52 (2 min) [cpu-decode agent]: more microbench rounds (quiet, load ~3): one parallelFor with an in-task
  barrier instead of the separate reduction parallelFor: no gain (113.0 vs 113.1 GB/s), 2 / 4 tasks per thread worse
  (111 / 107). Columns per iteration with 2 rows: 16 / 32 / 64 = 112.0 / 112.7 / 115.1-115.8 GB/s (16 f32
  accumulators); 1 row x 64 cols 111 (but gate|up alone 121), 4 rows x 64 cols 109; non-temporal loads no change;
  1 / 2 / 4 column blocks for gate|up ~equal with 64-col tiles. Winner: 2 W rows x 64 columns per pass, 115.5 vs
  104 GB/s (+11 %), same per-lane FMLAL order as the 8-row kernel (bit-identical). Dot layout 2 rows 116.6
  (LM head is that layout: 4 -> 2 rows per pass to try). Next: put both in f16_fused.cpp, in-model A/B.
- 07:52–07:52 (1.5 min) [memory agent]: deployed (scripts/ios_deploy.sh: 71/71 on-device unit tests incl. the
  extended protocol fuzz); tests/python/test_remote.py --device=iphone 17 passed + 3 skipped (Mac-only ones). The new
  test's mlock of a 512 KiB weight passes on the phone: mlock is allowed for an iOS app (at least small). Started
  the pressure experiment (GPU decode, s30 context, balloon 0.5 GB steps, unpinned then mlock).
- 07:46–07:49 (3 min) [gpu-gemm agent]: wrote kernels/gpu/f16_tensor.metal (lk_f16t_gemm_{128x128,64x128,64x64}[_nt],
  lk_f16t_gemm_swiglu_*: MLX-style 16x16 register fragments + matmul2d (16, 32, 16), fused epilogues) and the
  dispatch in f16_gpu.cpp (encodeSteel / encodeSteelSwiglu route M > 1 to it when the GPU is Apple10+ or
  LOKINFER_F16_GEMM=tensor; LOKINFER_F16T_TILE forces a tile); added it to the metallib SOURCES (CMakeLists.txt,
  one-line edit). The kernel bench prints the phone's thermal state / LPM per case. iPhone steel GEMM baseline bench
  started: failed at once, LokiHost not running on the phone (connection refused) → baseline = the 2026-11-06 table.
- 07:50–07:51 (1 min) [gpu-gemm agent]: Mac, LOKINFER_F16_GEMM=tensor: every M > 1 GEMM wrong (block-diagonal garbage
  on an identity test). Cause: MLX's hard-coded cooperative-tensor lane layout holds only on neural-accelerator GPUs
  (MLX never runs NAX on the M2). Rewrote the kernel layout-agnostic: one matmul2d<(SM, SN, 32), execution_simdgroup>
  per simdgroup, the f32 destination cooperative tensor kept across the K loop, inputs filled element-wise via
  get_multidimensional_index (no layout assumption), same for the epilogue.
- 07:51–07:53 (2 min) [gpu-gemm agent]: then only cases with K >= 512 failed: with the BK barrier loop nested around
  the SK loop, only the last K step's product survived (accumulator lost between iterations on the M2; K = 480 fine,
  512 wrong). Flattened to one K loop with the barrier every BK / SK steps: fixed.
- 07:54–07:54 (<1 min) [gpu-gemm agent]: Mac test_f16_fused_gpu.py default tier with LOKINFER_F16_GEMM=tensor: 134
  passed, 18 skipped (slow). Next: slow tier on the Mac, deploy to the phone.
- 07:55–15:09 (7 h 14 min) [coordinator]: network outage (API unreachable): all 3 agents stopped mid-task (gpu-gemm:
  Mac default tier passing with the tensor-op GEMM; memory: adding tests for pressure fields + PIN_MODEL; cpu-decode:
  about to edit the M = 1 path / LM head rows). No locks or stray processes left.
- 15:09–15:09 [coordinator]: resumed all 3 agents from their transcripts; oversight monitor re-armed.
- 07:53–07:57 (4 min) [cpu-decode agent]: put the 2-row × 64-col M = 1 GEMV (gemvRange1) and a 2-row LM head into
  f16_fused.cpp with temporary env A/B toggles (LK_AB_OLDGEMV, LK_AB_LMH4; to be removed); built; fused CPU kernel
  tests 99 passed / 9 skipped; new vs old GEMV outputs bit-identical on 7 shapes (Qwen decode shapes + odd tails).
  Started the in-model baseline (old kernels), queued behind the memory agent's bigmodel lock.
- 07:55–15:10 (435 min) [cpu-decode agent]: network outage, no work. The queued baseline ran unattended meanwhile
  (finished 09:29; it also timed the basic variant by mistake: no --variant): old kernels decode 30 s 33.8 ms
  (p10 33.4). Mac was on battery.
- 15:09 [cpu-decode agent]: resumed; f16_fused.cpp checked: one pragma push (l.47) / one pop (end), edits intact.
- 07:55–07:57 (2 min) [gpu-gemm agent]: Mac slow tier with LOKINFER_F16_GEMM=tensor: 18 passed. Tests: new
  `gemm_path` fixture (GEMM-family + audio_cnn tests run on the device's own kernels and, on the Mac in-process, with
  the tensor-op kernels forced) + shapes for each tensor tile and K tails > 512: 224 passed, 34 skipped. Mac A/B
  (kernel bench, M = 2048, tensor vs steel back to back): qkv 8.44 vs 4.59 ms, o 4.68 vs 2.55, swiglu 21.4 vs 12.4,
  down 14.1 vs 6.39: tensor ops 1.8–2.2x slower on the M2 (matmul2d on the shader ALUs) → the Mac keeps steel.
  Deploy queued behind another agent's phone job, then failed ("no connected physical iPhone").
- 07:57–15:09 (432 min) [gpu-gemm agent]: gap: network outage, no work.
- 15:09–15:10 (1 min) [gpu-gemm agent]: re-checked state after the outage: no locks held, phone connected again,
  tree builds (ninja up to date), test_f16_fused_gpu.py on the Mac passes. Re-deploying to the phone.
- 15:10–15:11 (1 min) [cpu-decode agent]: in-model A/B, Mac CPU, 3 alternations back to back (old kernels via the
  temp env toggles vs new; load average 7–12 from other sessions): decode 2 s 33.2 / 34.7 / 33.9 -> 31.7 / 30.2 /
  31.9 ms; decode 30 s 37.3 / 38.5 / 34.4 -> 31.5 / 31.0 / 30.8 ms (~ -9 %). Profile C = 480 (new): swiglu 14.2 ->
  12.0 ms, gemm_add 9.3 -> 8.4, qkv 4.3 -> 4.2, lm_head 5.6 (4 rows) -> 4.9 (2 rows).
- 15:11–15:12 (1 min) [cpu-decode agent]: microbench, gate|up with 1 row x 64 cols (it read 121 GB/s alone) inside
  the 2-row kernel for the other GEMVs: no gain overall (115.8 vs 115.8 GB/s): dropped. o_proj (8 MB) stays the
  weakest (107 vs dot layout 115: fixed per-call cost of the reduction pass on a 70 us call): not worth a layout change.
- 15:12–15:12 (1 min) [cpu-decode agent]: removed the temp A/B toggles; LM head rows as a template (kLmHeadRows = 2);
  build; tests/kernels/test_f16_fused_cpu.py --run-slow 108 passed (Mac). Qwen CPU model tests (slow) started.
- 07:53–07:55 (2 min) [memory agent]: pressure experiment, iPhone GPU decode (s30 context, 20 steps per point),
  unpinned: balloon of random device tensors 0 → 3.5 GB: step 33.0–34.9 ms throughout, 0 page-ins, weights
  resident 3.21 / 3.21 GB; iOS reclaimed other apps instead (system file cache 2.42 → 0.88 GB, compressor 0.46 →
  3.50 GB), pressure "critical" at 3.5 GB. System wired 5.2–5.4 GB: the GPU path's weights are already held (Metal
  no-copy buffers in the residency set), so the GPU path doesn't thrash. Pinned phase: mlock of the 3.44 GB
  succeeded on iOS (no error), BUT the wired weights count in phys_footprint (footprint 0.19 → 3.40 GB, available
  5.81 → 2.60 GB): mlock moves the weights under the app's jetsam limit. The run then died with a broken pipe at
  the first pinned balloon step (jetsam or the network outage: to re-check).
- 07:55–15:10 (435 min) [memory agent]: GAP — network outage, no work. On resume (15:09): no locks held, tree builds
  (Debug + Release lokinfer-serve, unit tests), Python files parse, test_remote.py 20/20 on the Mac; no half-applied
  edit in my files. The deploy watchdog relaunched LokiHost ~150 times meanwhile; the phone is "connected" but the
  server refused connections at 15:10 (retrying).
- 15:12–15:13 (1 min) [cpu-decode agent]: Qwen CPU model tests --run-slow on the Mac: 28 passed, 2 skipped (the
  long tier) with the new GEMV / LM head (bit-identical kernels, so the goldens are unchanged by construction).
- 15:13–15:16 (3 min) [cpu-decode agent]: CPU timing noise (todo item). Microbench, 2-row GEMV, 5 alternating
  rounds: 7 pool threads always ~10 % slower (103 vs 115 GB/s quiet) -> rejected. Background spinners (1 ms busy /
  1 ms sleep, default or user-interactive QoS): 1-4 of them change nothing (the scheduler puts them on the E-cores);
  6-8 of them: tasks per thread 1 / 2 / 3 = 86-105 (min 76) / 98-113 / 93-98 GB/s; quiet 115 / 113 / 107. In-model
  A/B (temp env toggle), 3 rounds quiet: tpt 1 decode 2 s 30.1 / 30.1 / 32.1, 30 s 30.9 / 30.4 / 30.5 ms; tpt 2
  32.1 / 30.5 / 31.7, 31.1 / 31.0 / 31.2 (+1-2 %). With 8 spinners (2 rounds): tpt 1 38.0 / 40.5, 40.5 / 40.9; tpt 2
  34.5 / 36.0, 37.5 / 38.2 (-7-9 %). Decision: keep 1 task per thread (the quiet case is what the dashboard and
  llama.cpp comparison measure); tpt 2 documented as the robust choice for a loaded machine. Temp toggle removed.
- 15:17 [cpu-decode agent]: started the Mac CPU dashboard record (decode 2 s … 1200 s, e2e 5 / 30 / 111 / 300 s; est.
  ~8 min; e2e 1200 s / 30 min skipped: ~5 min per call). Phone busy with another agent's tests meanwhile.
- 15:10–15:11 (1 min) [gpu-gemm agent]: deploy to the phone (incremental, Catch2 all passed); gemm.md: new section
  "f16 tensor-op GEMM" (design, selection, the two M2 pitfalls). 15:11: test_f16_fused_gpu.py --device=iphone
  (default tier, tensor ops by default on the phone) started under the phone lock.
- 15:13–15:21 (8 min) [memory agent]: hypothesis from the GPU run: Metal wires resident no-copy buffers without
  charging phys_footprint, which is why the GPU graph's weights never thrashed; mlock does charge it. Added PIN_MODEL
  mode 3 "gpu": wrap the model's constant pages in no-copy MTLBuffers in the shared residency set + trim (commit),
  so a CPU graph's weights get the same protection (server-only change, existing gpuWrapNoCopy / gpuReleaseBuffer
  hooks; buffers dropped before the mapping on FREE_MODEL / disconnect / unpin). Fixed the PIN_MODEL comment (mlock
  counts in the footprint). Mac: build, unit [remote], test_remote.py 20/20 local + 17/3 skipped --device=serve.
  Phone busy (another agent's kernel tests hold the lock); waiting to deploy and run the CPU pressure experiment.
- correction [memory agent]: the entry above ended at 15:20 (7 min), not 15:21.
- 15:21–15:24 (3 min) [memory agent]: deployed the mode-3 server (waited for the phone lock; 71/71 on-device unit
  tests). Queued: test_remote.py --device=iphone + the CPU pressure experiment (unpinned / gpu / lock).
- 15:11–15:23 (12 min) [gpu-gemm agent]: test_f16_fused_gpu.py --device=iphone (tensor ops on by default there; the
  run waited ~5 min for the phone lock held by another agent): 139 passed, 119 skipped (slow + the Mac-only forced
  variants), 7.6 min.
- 15:23–15:25 (2 min) [gpu-gemm agent]: iPhone kernel bench M = 2048 (tensor ops, nominal): qkv 8.23 ms, o 4.13,
  swiglu 24.0, down 12.2 vs steel 13.1 / 7.38 / 39.8 / 19.6 (2026-11-06 table): 1.6x, ~4.2 TF/s; MLX reaches ~7.7 TF/s
  in-model. Hypothesis: the layout-agnostic element-wise fills (scalar loads through get_multidimensional_index) cost
  the difference; next: an MLX-layout fragment variant (vector-friendly loads; valid on neural-accelerator GPUs only,
  tested on the phone) and relaxed_precision as a side measurement.
- 15:25–15:27 (2 min) [gpu-gemm agent]: added the "_nax" kernel flavour to f16_tensor.metal (MLX's register
  fragments in the neural-accelerator lane layout, flat K loop, same epilogues; + one relaxed_precision 128x128
  kernel for a side measurement), selected by default on Apple10+ GPUs, LOKINFER_F16T_KERNEL=generic|nax|naxr
  overrides. Mac (generic forced via the tests): 171 passed. Deploy + phone GEMM/CNN tests started 15:27.
- 15:17–15:24 (7 min) [cpu-decode agent]: Mac CPU dashboard record (new kernels): decode 2 s 36.4 (noisy: load 7-9,
  macOS mediaanalysisd at 110 % CPU), 5 s 30.1, 10 s 30.5, 30 s 37.7 (noisy), 60 s 32.9, 111 s 33.0, 300 s 36.9,
  1200 s 57.0; e2e 5 s 847, 30 s 3282, 111 s 16,782, 300 s 45,148 ms. e2e 300 s transcript: 942 tokens, differs
  from golden at token 48 (the documented f16 near-tie, gap 0.004 < 0.05).
- 15:24–15:27 (3 min) [cpu-decode agent]: checked that token-48 difference is not mine: one process, s300
  transcribed with new / old (temp per-call toggle) / new kernels: 942 tokens each, identical token lists (old ==
  new). Toggle removed again. (The old row was overwritten by the record, so whether it matched before is unknown.)
- 15:29–15:30 [coordinator]: the cpu-decode agent recorded Mac CPU dashboard rows at load 7–9 (other agents
  building / loading weights): told it to restore any row it made noisier, and told all agents to record Mac
  timings only below load ~2 while holding the macbench + build locks (iPhone timings unaffected).
- 15:27–15:31 [cpu-decode agent]: coordinator: the 15:17 record ran at load 7-9. Rows it overwrote with SLOWER
  values: Mac CPU decode 2 s (33.25 -> 36.36) and decode 30 s (34.03 -> 37.73); every other recorded row came out
  faster than the row it replaced (decode 5 s … 1200 s, e2e 5 … 300 s). Cancelled a queued re-record that had no
  load check; new rule for my records: wait for loadavg < 2, hold bigmodel + macbench + build locks.
- 15:31–15:32 [cpu-decode agent]: docs: qwen3-asr.md new section "CPU decode GEMV streaming (Mac,
  2026-11-07)" (microbench table, in-model A/B, timing-noise results) + kernel description (2-row M = 1 GEMV, 2-row
  LM head); todo.md: Mac CPU decode vs llama.cpp and CPU timing noise items closed with the numbers. Queued: quiet-
  window re-record of Mac decode 2 s / 30 s (waits for load < 2, max 30 min), iPhone session 1 (deploy + decode
  2 s / 30 s, new kernel, tpt 3) behind two other agents' phone jobs.
- 15:33 [coordinator]: user took the iPhone ("I'm taking my iphone away", then going out, "keep being
  autonomous"): /tmp/lokinfer-phone-disabled set, watchdog + phone jobs stopped (the phone-lock holder was killed
  mid-session), all 3 agents told to continue Mac-only and list their "needs the phone" items in todo.md.
- 15:24–15:32 (8 min) [memory agent]: iPhone (phone lock held 15:27–15:32): test_remote.py --device=iphone 17 passed +
  3 skipped (incl. the new pressure / pin test with mode "gpu"). CPU pressure experiment (CPU generate graph, s30
  context, 20 decode steps per point): unpinned = THRASH reproduced: from the first 0.5 GB balloon on, 2.8–10.6 GB of
  page-ins per 20 steps, max step 2.1–2.7 s (median 40 → 48–73 ms), weights read back from flash each time.
  pin "gpu" (mode 3: buffers added to the queue's residency set): no page-ins up to 2.5 GB, then 1.7 / 3.5 GB at
  3.0 / 3.5 GB (max step ~1.1 s), and system wired stayed ~1.8 GB: the shared set is only made resident while GPU
  work runs, so it isn't real protection for a CPU-only model. pin "lock" (mlock): 0 page-ins, step 40–45 ms, wired
  +3.2 GB, but phys_footprint 0.36 → 3.57 GB (available 5.64 → 2.43): stopped at the first 0.5 GB balloon by my
  available-memory guard. Also seen: a 512 MiB WRITE_TENSOR raises the footprint ~1 GB transiently (the payload
  buffer), only relevant to big uploads. The experiment ended 15:32, before the user took the phone
  (/tmp/lokinfer-phone-disabled 15:32): nothing of mine was killed; Mac-only from here.
- 15:27–15:32 (5 min) [gpu-gemm agent]: deploy of the "_nax" flavour done (Catch2 passed) after waiting for the phone
  lock; the phone GEMM test run that followed never ran: the user took the iPhone at 15:32 (/tmp/lokinfer-phone-
  disabled). Not retried. Because the "_nax" kernels are unvalidated on the phone, the default flavour is now the
  layout-agnostic one (the kernels that passed the phone tests at 15:23); nax only through LOKINFER_F16T_KERNEL=nax.
  Mac-only from here.
- 15:33–15:33 [cpu-decode agent]: coordinator: the user took the iPhone (phone-disabled flag set 15:32).
  Killed my queued phone session (its deploy had been refused; the bench was waiting on the locks, never touched the
  phone). iPhone CPU GEMV re-measure written up in todo.md as "needs the phone", with the exact steps. Note: the new
  M = 1 GEMV / LM head are in the tree and will reach the phone with anyone's next deploy (bit-identical math).
- 15:33–15:35 (2 min) [memory agent]: PIN_MODEL mode 3 reworked: its own MTLResidencySet (no-copy buffers over the
  constant pages, commit + requestResidency, endResidency on unpin / free) instead of the queue's shared set. Mac:
  build, unit [remote], test_remote.py 20/20 local + 17/3 --device=serve; a 256 MiB weight pinned "gpu" raises system
  wired by ~205–256 MiB and phys_footprint by 2 MiB, "lock" raises the footprint by 256 MiB (and it stays after
  munlock). Its effect under iPhone pressure is still to measure (phone taken). bench_qwen3asr_iphone.py --pin
  none|gpu|lock (default none). Queued a Mac per-case memory run (measure_qwen3asr_memory.py --where mac) behind the
  big-model lock (another agent holds it); writing docs meanwhile.
- correction [memory agent]: the entry above ended at 15:34 (1 min), not 15:35.
- 15:33–15:34 (1 min) [gpu-gemm agent]: Mac: test_f16_fused_gpu.py + test_f16_rms_gemm.py --run-slow 294 passed (both
  GEMM paths via gemm_path); test_f16_rms_gemm.py with LOKINFER_F16_GEMM=tensor 36 passed; Qwen model tests on the
  Mac GPU with the tensor-op kernels forced (--run-slow, -k gpu, bigmodel lock): 28 passed, 2 skipped (long tier),
  transcripts golden. Docs: gemm.md (flavours, iPhone numbers), qwen3-asr.md (kernel table, iPhone note), todo.md
  ("Needs the phone" list under the prefill-gap item). 15:34: Qwen model tests, Mac default kernels, CPU + GPU slow
  tier started.
- 15:34–15:40 (6 min) [gpu-gemm agent]: Mac Qwen model tests, default kernels (steel), CPU + GPU slow tier: 56 passed,
  4 skipped (long); Mac tests/kernels default tier 1236 passed; tests/perf/test_dashboard_goals.py 3 passed (no Kokoro
  / GPT-2 code touched: f32 paths unchanged). f16_tensor.metal header updated; time-accounting.md: 3 rows. No Mac
  timings recorded (the Mac path is unchanged steel; machine busy, load ~5). Stopping here: everything left needs the
  phone (docs/todo.md, "Needs the phone" under the iPhone GPU prefill item).
- 15:34–15:38 (4 min) [memory agent]: docs: todo.md fc-thrashing item (done parts + a "needs the phone" list),
  qwen3-asr.md "Memory under pressure" (the iPhone balloon table), runtime.md "Memory pressure (remote executor)",
  big-models.md pointer, time-accounting rows. Mac per-case memory run 1 (15:34–15:37, big-model lock, swap steady
  5.8 GB): findings: the CPU path's transient peak is the call itself (CPU prefill 300 s: footprint 0.24 GB after,
  peak 0.66; e2e 300 s peak 0.85 GB = the iPhone's 0.86), the GPU path's footprint is tiny (e2e 300 s peak 0.18 GB:
  its arenas / KV writes aren't charged to the footprint), untouched KV pages cost nothing (anonymous zero-fill).
  Bug in my script: it read MEMORY after freeing the KV cache, and the GPU prefill session also paid the first upload
  of the weights (1.57 GB payload buffers); fixed (snapshot before the free) and re-running. Seen: Mac CPU e2e 300 s
  transcript NOT golden (942 tokens, golden 941) with the current tree's kernels/cpu (another agent's area; the
  known f16 comma near-tie at 300 s): flagged to the coordinator, not mine to fix.
- 15:42 [coordinator]: gpu-gemm agent finished (phone taken mid-validation): tensor-op GEMM 1.6–1.8× steel on
  the iPhone kernel bench (M = 2048), Mac all green; only the default-tier kernel tests ran on the phone. Made it
  opt-in (kTensorGemmDefault = false; LOKINFER_F16_GEMM=tensor) until the slow tier + Qwen model tests pass on the
  phone; LokiHost now reports LOKINFER_F16_GEMM / F16T_KERNEL / F16T_TILE in runtime_env; todo + gemm.md +
  qwen3-asr.md say so (~10 min).
- 15:38–15:44 (6 min) [memory agent]: Mac per-case memory run 2 (15:38–15:42, fixed script): table in qwen3-asr.md
  "Memory per case" (CPU footprint = written KV + 0.1-0.25 GB arenas, e2e 300 s peak 0.84 GB; GPU <= 0.18 GB).
  Decision: the CPU 2048-row prefill arenas (<= 0.2 GB) are not shrunk (small vs the 4.6 GB working set, prefill
  speed risk); the weights' eviction is the real issue (PIN_MODEL). CPU e2e 300 s transcript again 942 tokens, not
  golden (GPU golden), in both runs: flagged. Qwen model tests --run-slow on the Mac: 56 passed, 4 skipped (85 s);
  test_dashboard_goals.py 3 passed. Swap steady at 5.8 GB throughout. iPhone items left in docs/todo.md ("needs the
  phone"): mode-3 pressure run, --pin A/B with --quick, the iPhone per-case table, Qwen / remote tests on the phone.
- 15:45 [coordinator]: memory agent finished its Mac-side part (phone taken): pressure reporting in MEMORY,
  PIN_MODEL (default off), the per-case Mac memory table, the iPhone pressure reproduction (CPU path thrashes from
  +0.5 GB of pressure; GPU path holds). Its flag: Mac CPU e2e 300 s now 942 tokens vs golden 941 → sent to the
  cpu-decode agent to settle with the near-tie rule (G-true margin < 0.05) or fix / revert.
- 15:44–15:46 [cpu-decode agent]: coordinator: "Mac CPU fused e2e 300 s now 942 tokens, was golden".
  Evidence: (1) one process, old vs new kernels (temp toggle, 15:24–15:27): identical 942-token lists; (2) bench
  history ~/.cache/lokinfer/qwen3asr/bench/lokinfer.jsonl: Mac CPU fused e2e 300 s matches_golden False on
  2026-11-06 01:36 and 01:54 (before my change; only Mac GPU rows were True). Margin: the precision gate's G-true
  teacher-forced run logs s300 "min top-2 margin 0.004" at the f16 flip, token 48 (" and" 42.0922 vs "," 42.0880):
  < 0.05, so the difference is acceptable under the model-level rule. Not a regression of this change; documented
  in qwen3-asr.md ("CPU decode GEMV streaming"). (The iPhone CPU row being golden is consistent: different thread
  count / task split there.)
- 15:32–15:47 (15 min) [cpu-decode agent]: quiet-window re-record of the two rows the 15:17 run had overwritten with
  slower values: waited for load < 2 (15:32-15:46 at 2.0-8.3), took bigmodel + macbench + build locks, recorded at
  load 1.86: Mac CPU decode 2 s 29.86 ms (p10 29.58; was 33.25 before today, 36.36 noisy), decode 30 s 30.43
  (p10 30.22; was 34.03, 37.73 noisy). Restored-and-improved. qwen3-asr.md Mac table updated.
- 15:47 [cpu-decode agent]: final source (no temp toggles): test_f16_fused_cpu.py --run-slow re-run on the
  Mac, dashboard goals guard 3 passed (thread pool untouched, Kokoro / GPT-2 unaffected). iPhone tests / timings
  pending: the phone is with the user (todo "needs the phone").
- 15:51 [coordinator]: cpu-decode agent finished: Mac CPU decode 33.25 → 29.86 ms (2 s), 34.03 → 30.43 ms
  (30 s), bit-identical outputs; the s300 942-token transcript predates it (margin 0.004 < 0.05: near-tie, acceptable).
  All 3 agents done (Mac-side). Full Mac default tier after their combined changes (3.4 min): ctest 71/71, Python
  1415 passed, models / e2e / perf 147 passed. Phone items in todo.md "needs the phone" lists.
- 18:45 [coordinator]: iPhone back (user); flag cleared, deploying the combined changes.
- 18:46 [coordinator]: deploy 26 s (71/71 on-device). Resumed the 3 agents for their "needs the phone" lists, sharing
  the phone via the lock (≤ 15 min sessions; the GPU agent restores LokiHost's default env before releasing; the CPU
  agent's per-setting deploys only under the held lock).
- 18:46–18:47 (1 min) [memory agent]: the phone is back (coordinator, 18:46; LokiHost deployed with my final PIN_MODEL
  build). Wrote a session-start check (runtime_env empty, no Low Power Mode, memory summary); queued phone session 1:
  test_remote.py --device=iphone + pressure test CPU --pin none gpu (~8 min est.).
- 18:46–18:47 (1 min) [gpu-gemm agent]: phone back (coordinator: tensor GEMM now opt-in, kTensorGemmDefault = false;
  LokiHost env relaunch via bench/harness/lokihost.sh). Session helper: launch with env, check state()["runtime_env"],
  run, restore the default LokiHost, all inside one phone-lock hold. 18:47: session 1 = test_f16_fused_gpu.py +
  test_f16_rms_gemm.py --run-slow on the phone with LOKINFER_F16_GEMM=tensor (est. ~8 min).
- 18:50 [coordinator]: user: "let's finish phase 1". Open phase-1 items: (1) iPhone GPU prefill gap
  (gpu-gemm agent, on the phone now); (2) Q5's last two fusions → dropped by decision with profile evidence (no
  standalone add / RMSNorm left: residual in gemm_add's epilogue, norm in rmsnorm_gemm; add_pos_prefix 0.0 %);
  (3) the kSmeMinM SME2-vs-NEON A/B → added to the cpu-decode agent's phone list.
- 18:47–18:51 (4 min) [memory agent]: phone session 1 (18:47–18:49; runtime_env empty, nominal, no LPM):
  test_remote.py --device=iphone 17 passed + 3 skipped on the final build. Pressure test CPU: unpinned thrashes again
  (1.5-4.9 GB of page-ins per 20 steps at every balloon step, max step 0.86-1.87 s). PIN_MODEL "gpu" (own residency
  set + requestResidency): 0 page-ins, BUT on iOS it is charged to phys_footprint like mlock (footprint 0.24 → 3.44
  GB, available 5.76 → 2.56), so the balloon stopped at 0.5 GB (critical). Hypothesis "Metal residency is free on
  iOS" refuted for an explicitly requested set: only the GPU queue's own set during GPU work went uncharged. So on
  iOS there is no free pin: protecting 3.2 GB of weights costs 3.2 GB of the ~6 GB app budget, either way
  (CPU e2e 300 s would then peak at ~4.3 GB: it fits). Next: the speed A/B --pin none vs gpu.
- correction [memory agent]: the entry above ended at 18:50 (3 min), not 18:51.
- 18:46–18:51 [cpu-decode agent]: phone back. Queued session 1 (phone + bigmodel locks: device state /
  runtime_env check, test_f16_fused_cpu.py --run-slow and Qwen CPU model tests --run-slow on the iPhone) behind
  another agent's phone job. New item from the coordinator: kSmeMinM (16) SME2 vs NEON-tile A/B. Plan to cover all
  items in 3 deploys (per-kernel profiles separate the GEMV and LM-head effects): A = deployed (tpt 3, LM head 2
  rows, SME from M 16), B = tpt 1 + LM head 4 rows + SME from M 5, C = tpt 2 + SME never (M < 4096: NEON tiles);
  per deploy: device state, prof-lokinfer iphone-cpu decode C=480, a GEMM M sweep (M 5…128, qkv / enc o /
  conv_out shapes, scratch script), quick prefill 2 s / 5 s; then the final deploy.
- 18:50–19:01 (11 min) [memory agent]: docs updated with the iOS residency finding (qwen3-asr.md pressure table +
  bullet, runtime.md PIN_MODEL, todo); phone session 2 (quick A/B CPU --pin none / gpu, alternating x2, prefill /
  decode / e2e 30 s) queued since 18:50 behind the GPU agent's and the CPU agent's phone sessions (lock queue).
- correction [memory agent]: the entry above ended at 19:00 (10 min), not 19:01.
- 18:47–19:03 (16 min) [gpu-gemm agent]: phone session 1 (lock wait ~3 min, LokiHost with LOKINFER_F16_GEMM=tensor,
  runtime_env checked): test_f16_fused_gpu.py + test_f16_rms_gemm.py --run-slow: 193 passed, 101 skipped (Mac-only
  forced variants), 13.1 min; default LokiHost restored (runtime_env empty). 19:03: session 2 = Qwen model tests
  --run-slow on the phone (CPU + GPU) with tensor ops (est. ~5 min).
- 18:47–19:14 (27 min; 16 min of it waiting for the phone lock) [cpu-decode agent]: iPhone session 1 (19:03–19:14,
  11 min): device nominal, LPM off, build 2cb13d04822f, runtime_env empty. With the new M = 1 GEMV / 2-row LM head
  deployed: tests/kernels/test_f16_fused_cpu.py --run-slow --device=iphone 106 passed, 2 skipped (2.7 min);
  tests/models/test_qwen3_asr.py --run-slow -k cpu --device=iphone 28 passed, 2 skipped (7.4 min). Validated.
- 19:00–19:17 (17 min) [memory agent]: phone session 2 (lock 19:13–19:17; runtime_env empty; thermal "serious" at
  the start, cooled to nominal per case): quick A/B CPU, side 1 --pin none: prefill 30 s 1358.6 ms, decode 30 s
  44.07 ms, e2e 30 s 45,952 ms (!: dashboard 7,044; golden transcript, no pressure warning; unexplained, likely the
  e2e loop's per-call overhead or heat, to re-check). Side 2 --pin gpu: graphs loaded + pinned, then "No route to
  host": devicectl reports the iPhone "unavailable" (CoreDevice 4016, usage assertion not possible; the watchdog's
  relaunches fail). Not a jetsam signature (the device itself dropped off): I stopped my session and released the
  lock at 19:17; no retry. BLOCKER: the phone is disconnected / locked; left: the A/B, the iPhone per-case table,
  Qwen model tests --device=iphone.
- 19:17–19:18 [coordinator]: iPhone dropped off (devicectl "unavailable", CoreDevice 4016; watchdog relaunches
  failing). Memory agent stopped cleanly (its findings: on iOS a GPU pin counts against the app limit like mlock:
  footprint 0.24 → 3.44 GB; an e2e 30 s CPU of 46 s vs 7 s to re-check). gpu-gemm + cpu-decode agents told to pause
  phone work; asked the user to unlock / reconnect; a background check every 5 min reports when it's back.
- 19:03–19:18 (15 min) [gpu-gemm agent]: phone session 2 (Qwen model tests --run-slow with tensor ops) waited ~14 min
  for the phone lock behind two other agents; it got the lock at ~19:17, just as the phone dropped off (CoreDevice
  "unavailable"): nothing ran. Killed my session, restored the default LokiHost watchdog (bench/harness/lokihost.sh
  restore), released the lock; not retrying until the coordinator says the phone is back. todo.md: item 1 of
  "Needs the phone" done.
- 19:14–19:19 [cpu-decode agent]: session 1 had completed before the drop (19:03–19:14: fused CPU kernels
  slow tier 106 passed / 2 skipped, Qwen CPU model tests slow tier 28 passed / 2 skipped, both on the iPhone with the
  new kernels deployed). Session 2 (variant A timings: deployed tpt 3 / LM head 2 rows / SME from M 16) got the
  phone lock at 19:18:49, after the phone had dropped off (19:17, coordinator): stopped before any measurement, lock
  released. No retry loop; waiting for the coordinator. Prepared meanwhile: setvar.py (iPhone kGemvTasksPerThread /
  kLmHeadRows / kSmeMinM), measure_variant.sh (state, quick decode 2 s / 30 s + prefill 2 s / 5 s, decode
  profile, GEMM M sweep 5…128 on qkv / enc o / conv_out shapes). f16_fused.cpp: kLmHeadRows is now per platform
  (both 2) so iPhone variants never change the Mac build.
- 19:19 [cpu-decode agent]: Mac-side while the phone is away: build + LM-head tests after the per-platform
  constant edit (6 passed); todo.md "needs the phone" item updated (validation done, kSmeMinM A/B and the 3-deploy
  plan added, blocked on the phone); qwen3-asr.md notes the on-phone validation. Phone timings pending.
- 19:19 [coordinator]: all 3 agents paused cleanly (no lock held). Done on the phone before the drop: CPU
  kernels test_f16_fused_cpu --run-slow 106 passed + Qwen CPU model tests 28 passed (new GEMV / LM-head kernels
  validated on the iPhone); GPU tensor-op kernel slow tier 193 passed. Waiting on the phone for: Qwen tests with
  tensor ops (gate for the default), _nax / tile A/B, iPhone CPU setting sweep (3 deploys) + kSmeMinM, memory A/B.
- 21:10–21:11 [coordinator]: iPhone available again (unavailable 19:17 → ≤ 21:10). Redeployed LokiHost (30 s,
  71/71, nominal, LPM off, default env) and resumed the 3 agents on their phone lists.
- 21:11–21:11 (<1 min) [gpu-gemm agent]: phone back (21:10); session 2 again: Qwen model tests --run-slow (CPU + GPU) on the phone with LOKINFER_F16_GEMM=tensor (est. ~8 min incl. the 7 min CPU part).
- 21:10–21:12 [cpu-decode agent]: phone back (deploy A settings on it). Starting session A (no deploy needed).
- 21:10–21:12 (2 min) [memory agent]: the phone is back (coordinator, 21:10). Wrote a scratch e2e breakdown script
  (per-call device times, page-ins, thermal; 2 passes) and queued phone session 3: e2e 30 s CPU breakdown, then the
  quick A/B --pin none / gpu / none / gpu (prefill / decode / e2e 30 s), est. ~10 min.
- 21:11–21:16 (5 min) [gpu-gemm agent]: phone session 2 (LOKINFER_F16_GEMM=tensor, runtime_env checked): Qwen model
  tests --run-slow on the phone, CPU + GPU: 56 passed, 4 skipped (long tier), 4.1 min, transcripts golden; default
  LokiHost restored. → kTensorGemmDefault = true (on for Apple10+); deploy started 21:16.
- 21:12–21:18 (6 min) [memory agent]: phone session 3 started 21:17 (lock queue; runtime_env empty, nominal). e2e
  30 s CPU re-check, two passes with a per-call breakdown: enc 343-382 ms, prefill 1343-1489, 79 steps 3276-3760
  (median 40.9-48.2, max 54), total 5147 / 5446 ms, golden, 0 page-ins, pressure normal. So the 45,952 ms of 19:13
  was not the model: that run ended minutes before the phone dropped off the USB link (19:17), most likely the
  device / link stalling inside the device-timed calls. No action; the dashboard's 7,044 ms stands (this is faster).
- 21:18–21:22 (4 min) [memory agent]: session 3 A/B (CPU, --quick, alternating, all nominal except one fair; ms):
  none 1673 / 53.1 / 7424, gpu 1373 / 46.9 / 6001, none 1413 / 42.7 / 6397, gpu 1416 / 41.5 / 5467 (prefill 30 s
  / decode 30 s / e2e 30 s), all transcripts golden: pinning "gpu" costs no speed (as fast or faster in every
  pair). The second unpinned run warned "3.86 GB paged in during prefill 30s": a false positive, the first case after
  a load reading 3.80 GB of weights cold (the previous pinned process's pages had been evicted, system free 0.17 GB);
  the bench's thrash threshold is now 1.5x the weights. Session ended 21:21, lock released. Next: session 4, the
  iPhone GPU per-case table.
- 21:16–21:22 (6 min) [gpu-gemm agent]: deploy with tensor ops on by default on the phone (Catch2 71/71). 21:22:
  session 3 = "_nax" kernel tests (GEMM/CNN default tier) + kernel A/B generic / nax / naxr, opt only (est. ~9 min).
- 21:22–21:25 (3 min) [memory agent]: phone session 4 (21:22–21:25; runtime_env empty, nominal): iPhone GPU per-case
  table, fresh LokiHost per kind: footprint ≤ 0.17 GB everywhere (enc 300 s peak 0.17, e2e 300 s peak 0.15), GPU
  allocated up to 4.83 GB (weights wraps incl.), 0 page-ins, e2e 300 s golden. Bench default now `--pin auto`
  (gpu for the CPU graphs, none for the GPU ones); docs updated with the A/B. Queued session 5: CPU table.
- 21:16–21:18 (2 min) [cpu-decode agent]: iPhone session A (deployed: tpt 3, LM head 2 rows, SME from M 16; nominal,
  runtime_env empty). Decode profile C = 480: e2e 49.4 ms (p10 44.4); swiglu 21.5 ms (66 GB/s), o + down 14.7
  (64), qkv 7.5 (63), LM head 6.8 (92 GB/s). GEMM M sweep (ms; M 5 / 8 / 13 on NEON tiles, 16+ SME2): qkv 0.59 /
  0.59 / 0.85 | 0.37 (16) / 0.38 (24) / 0.38 (32) / 0.91 (44) / 0.92 (64) / 2.09 (128); enc o 0.058 / 0.054 /
  0.083 | 0.039 / 0.042 / 0.043 / 0.070 / 0.075 / 0.142; conv_out (K 7680) 0.63 / 0.59 / 0.84 | 0.29 / 0.30 / 0.33 /
  0.71 / 0.74 / 1.99. SME at M 16 is 1.6-2.9x faster than the tiles at M 5-13; SME time steps with each 32-row
  panel (M 32 -> 44: 2.4x). The quick bench lines were lost to a wrong grep (fixed); re-run separately.
- 21:18–21:22 (4 min) [cpu-decode agent]: debug run of the quick bench: decode 2 s 37.49 ms (A; build 067e9eaa4a84
  — another agent redeployed the same tree meanwhile; dashboard before the GEMV change: 40.0). A bench re-run queued.
- 21:22–21:30 (8 min) [gpu-gemm agent]: phone session 3 (lock wait ~3 min). "_nax" flavour (LOKINFER_F16T_KERNEL=nax):
  GEMM/CNN/rms-GEMM default tier 122 passed. Kernel A/B, opt only (ms, generic → nax; phone nominal → fair):
  qkv M=44 0.257 → 0.152, M=401 1.85 → 0.933, M=2048 8.38 → 4.24 (8.1 TF/s); o M=2048 4.24 → 2.24; swiglu M=2048
  24.2 → 13.4; down M=2048 12.8 → 6.83; enc qkv M=3900 7.14 → 3.56, o 2.37 → 1.14, fc1 9.09 → 5.92, fc2 8.12 → 4.04:
  nax ~1.9-2x the layout-agnostic kernels (the element-wise cooperative-tensor fills were the cost), ~3x steel.
  naxr (relaxed_precision) qkv: 0.932 / 4.23 ms = nax: no gain, so relaxed precision is irrelevant (kept off).
  Default LokiHost restored. Next: session 4 = nax slow tier (GEMM-family) + Qwen model tests with nax, then make nax
  the Apple10+ default.
- 21:30–21:35 (5 min) [cpu-decode agent]: session A quick bench (phone "fair" at start, cooled to nominal per case):
  decode 2 s 44.0, 30 s 45.9, prefill 2 s 227, prefill 5 s 351 ms. The 21:22 single run read decode 2 s 37.5:
  quick (5-run) phone numbers swing ~15 %, so the decisions use the decode profile (n ~ 50 timed calls + per-kernel
  times) and the GEMM sweep; the quick bench is a sanity check only. Session B (tpt 1, LM head 4 rows, SME from M 5:
  setvar + deploy + measure under the held phone lock) queued.
- 21:25–21:41 (16 min) [memory agent]: phone session 5 (lock 21:35–21:40, after a 10 min queue; runtime_env empty,
  nominal): iPhone CPU per-case table: footprint = written KV + ~0.1 GB (prefill 300 s 0.51, decode 0.60, e2e 0.64 GB;
  enc 300 s 0.15), resident up to 4.47 GB, available 5.36 GB at e2e 300 s, 0 page-ins after the first case, CPU e2e
  300 s golden (941 tokens) on the phone. Both iPhone tables in qwen3-asr.md "Memory per case" next to the Mac's;
  todo updated. Queued session 6: Qwen model tests --run-slow + test_remote.py on the phone.
- 21:40–21:44 [cpu-decode agent]: session B: setvar (tpt 1, LM head 4, SME from M 5) + deploy under the
  held phone lock (21:40:43–21:40:56, 71/71, build 93c02c87e414). The phone was at "serious" (heat from the sessions
  before); the measure script waited only in the bench, the profile / sweep would have run hot: stopped, added a
  wait-for-nominal (<= 10 min) before the profile and the sweep, re-queued B's measurement (B stays deployed; the
  tree holds the B constants until session C).
- 21:30–21:50 (20 min; 14 min of it waiting for the phone lock) [gpu-gemm agent]: phone session 4 with
  LOKINFER_F16T_KERNEL=nax: GEMM-family kernel tests --run-slow 138 passed (2.0 min); Qwen model tests --run-slow
  CPU + GPU 56 passed, 4 skipped (long), 4.0 min, transcripts golden. Phone ended "serious". → "_nax" is now the
  Apple10+ default flavour (f16_gpu.cpp); gemm.md updated with the 3-way kernel table. Deploy started 21:51.
- 21:41–21:55 (14 min) [memory agent]: phone session 6 (lock 21:50–21:55; runtime_env empty; thermal serious at the
  start): Qwen model tests --device=iphone --run-slow 56 passed + 4 skipped (4.4 min), test_remote.py
  --device=iphone 17 passed + 3 skipped, on the final build (with the other agents' changes). todo item closed (sub-
  items record what's done; left: the WRITE_TENSOR payload buffer, optional pinning policy for apps / the harness).
  test_dashboard_goals.py 3 passed. Done with the memory-pressure list.
- 21:56 [coordinator]: memory agent finished (fc-thrashing item closed): the 46 s e2e was an artifact (re-run 5.1–5.4 s);
  pin "gpu" costs no speed (A/B) → the iPhone bench pins CPU graphs by default (~3.8 GB of the ~6 GB app budget); the
  iPhone per-case memory table measured; Qwen + remote tests pass on the phone. gpu-gemm agent: "_nax" GEMM ~2x the
  generic tensor kernel, ~3x steel (qkv M=2048 4.24 ms, 8.1 TF/s), made the Apple10+ default after kernel + Qwen
  tests on the phone. Remaining: gpu-gemm in-model timings / recording, cpu-decode settings sweep.
- 21:44–22:03 (19 min; ~12 of it waiting for the phone lock, session 21:55–22:03) [cpu-decode agent]: session B
  measure (tpt 1, LM head 4 rows, SME from M 5; build 93c02c87e414; phone "serious" at start, cooled per case):
  quick decode 2 s 42.1, 30 s 42.9, prefill 2 s 222, 5 s 317 ms. Profile C = 480: e2e 48.8 ms (A 49.4); swiglu 21.4
  (A 21.5), o + down 15.9 (A 14.7), qkv 7.75 (A 7.49), LM head 4 rows 7.00 (A, 2 rows: 6.79). M sweep, SME now from
  M 5: qkv M 5 / 8 / 13 = 0.31 / 0.28 / 0.28 ms (A, NEON tiles: 0.59 / 0.59 / 0.85); enc o 0.035 (tiles 0.054-0.083);
  conv_out 0.26-0.29 (tiles 0.59-0.84): SME2 2-3x faster than the tiles at every M >= 5.
- 21:50–22:03 (13 min; mostly waiting for the phone lock) [gpu-gemm agent]: deploy with "_nax" as the Apple10+
  default (Catch2 71/71); Mac rebuilt, Mac f16 GPU kernel tests 260 passed. 22:04: session 6 = in-model
  --quick A/B steel vs default (prefill 2 / 30 / 300 s, enc 30 s; cool to fair, <= 60 s per case; est. <= 12 min).
- 22:04–22:06 (2 min) [cpu-decode agent]: session C (tpt 2, LM head 2, SME never = NEON tiles everywhere; deploy
  12 s, build 16fb19d08c4c, "fair" -> nominal): quick decode 2 s 34.4, 30 s 38.9, prefill 2 s 381, 5 s 624 ms (tiles:
  1.7-2x slower prefill). Profile C = 480: e2e 41.3 ms (A 49.4, B 48.8); swiglu 19.1, o + down 12.1, qkv 6.46, LM
  head 6.32. Tiles M sweep: qkv M 16 / 32 / 128 = 0.86 / 1.43 / 4.83 ms vs SME 0.28-0.37 / 0.31-0.38 / 1.77-2.09.
  Decision kSmeMinM: SME2 for every M above the GEMV range (5), it wins 2-3x at M 5-13 and 2.5-4x above. tpt 2 looks
  best (GEMVs -11-18 % vs A while the unchanged LM head moved -7 %: part phone state) -> confirming with an
  alternating deploy A/B in one session (tpt 3 / 2 / 3 / 2; deploys take ~15 s now).
- 22:10 [cpu-decode agent]: note for the other phone users: until my next session the phone runs variant C
  (iPhone CPU GEMMs on NEON tiles: CPU prefill ~1.7-2x slower; GPU unaffected). Tree set to tpt 2 / LM 2 / SME 5
  meanwhile, so any other deploy ships a sane build.
- 22:03–22:13 (10 min) [gpu-gemm agent]: phone session 6, in-model --quick A/B (GPU, cool to fair <= 60 s per case),
  steel (LOKINFER_F16_GEMM=steel) → default (tensor ops, "_nax"), ms: prefill 2 s 85.6 → 40.6, prefill 30 s 494.9 →
  165.4, prefill 300 s 4976 → 2065, enc 30 s 162.9 → 62.0 (MLX Swift: 35.9 / 146 / 1993 / 98.2). Prefill now
  1.1-1.13x MLX (was 2.4-3.4x), the encoder 1.6x faster than MLX. Default LokiHost restored.
- 22:06–22:18 (12 min; session 22:13–22:18) [cpu-decode agent]: iPhone tasks-per-thread A/B, alternating deploys in
  one session (LM head 2, SME from 5; ~15 s per deploy, 71/71 each), decode profile C = 480 e2e median: tpt 3 49.6
  / 50.1, tpt 2 40.8 / 42.5, tpt 1 48.7 ms (+ sessions A / B / C: tpt 3 49.4, tpt 1 48.8, tpt 2 41.3). tpt 2: GEMVs
  swiglu 17.3 vs 22.5 ms, o + down 10.9 vs 15.3, qkv 5.7 vs 7.7 (vs tpt 3) -> -15 % per decode step. LM head 2 vs
  4 rows: 6.8 vs 7.0 ms (keep 2). Chosen iPhone setting: kGemvTasksPerThread 2, kLmHeadRows 2, kSmeMinM 5.
- 22:19–22:20 [cpu-decode agent]: final iPhone constants in f16_fused.cpp (kGemvTasksPerThread 2 on iOS,
  kLmHeadRows 2 on both, kSmeMinM 5) with the measurements in the comments; Mac build + fused CPU tests 99 passed.
  Docs: qwen3-asr.md "Q5 CPU kernels" (kSmeMinM table) and "CPU decode GEMV streaming" (iPhone results); todo.md: CPU
  opt kernels item closed, iPhone CPU re-measure item closed (GPU half split off, still open). Final deploy + phone
  test session queued.
- 22:14–22:24 (10 min) [gpu-gemm agent]: phone session 7. prof-lokinfer prefill 30 s on iphone-gpu (default kernels):
  165.4 ms e2e; rmsnorm+gate/up+SwiGLU 75.2 ms (45.5 %, 7.51 TF/s), o/down + residual 43.2 ms (26.1 %, 8.72 TF/s),
  rmsnorm+QKV 24.9 ms (15.0 %, 7.58 TF/s), attention 9.1 ms (5.5 %), LM head 5.6 ms (3.4 %). Kernel-bench tile A/B
  (forced tile, phone "fair", sequential, opt only, ms 128x128 / 64x128 / 64x64): qkv M=401 0.944 / 0.862 / 0.869,
  qkv M=2048 4.18 / 3.91 / 3.83, swiglu M=2048 13.3 / 11.3 / 10.4, down M=2048 6.50 / 6.23 / 7.80, enc qkv M=3900
  3.55 / 3.38 / 5.71, enc o 1.24 / 3.15 / 3.61 (odd), fc1 6.33 / 5.08 / 5.70, fc2 4.05 / 3.85 / 5.19. 64x128 wins
  most LM shapes by 5-15 %; the kernel bench reuses its weights from the SLC, so decide in-model: session 8 =
  --quick in-model A/B default rule vs LOKINFER_F16T_TILE=64x128.
- 22:20–22:30 (10 min; session 22:24–22:30) [cpu-decode agent]: final deploy (tpt 2 / LM 2 / SME from 5; build
  bc00535e3bb4, 71/71, runtime_env empty) + phone validation: fused CPU kernels --run-slow 106 passed / 2 skipped,
  Qwen CPU model tests --run-slow 28 passed / 2 skipped (kSmeMinM change included). Final setting is what is
  deployed now.
- 22:24–22:37 (13 min) [gpu-gemm agent]: phone session 8, in-model --quick A/B, LOKINFER_F16T_TILE=64x128 vs the
  rule (128x128 from 32 tiles): prefill 30 s 150.6 vs 165.3 ms, prefill 300 s 1922 vs 2064, enc 30 s 59.9 vs 62.0,
  enc 300 s 565 vs 590 (each from nominal). → the auto rule now starts at 64x128 (64x64 below 32 tiles); 128x128
  only when forced. Default LokiHost restored. Next: Mac rebuild + tests, deploy, then record the GPU rows (cool).
- 22:38–22:38 (1 min) [gpu-gemm agent]: Mac rebuilt; Mac f16 GPU kernel tests --run-slow 294 passed. Started the
  record chain: deploy, Qwen GPU model tests on the phone (slow tier), then the iPhone GPU rows recorded in 7 chunks
  (prefill / enc 2 s ... 300 s, e2e 5 / 30 / 111 / 300 s; --record, cool to nominal per case, <= 300 s wait), each chunk
  its own phone-lock hold (est. 50-70 min in all incl. cool-downs; each chunk <= ~15 min).
- 22:30–22:38 (8 min) [cpu-decode agent]: recorded iPhone CPU decode rows (full protocol, 30 runs, nominal before
  each case, final build bc00535e3bb4): 2 s 40.0 -> 37.15, 5 s 45.6 -> 39.84, 10 s 49.8 -> 41.92, 30 s 51.6 ->
  44.42 ms (llama.cpp CPU 64.9-83.1). Queued 60 / 111 / 300 s.
- 22:38–22:40 (2 min) [gpu-gemm agent]: deploy with the 64x128 tile rule (Catch2 71/71); Qwen GPU model tests
  --run-slow on the phone with it: 28 passed, 2 skipped (long), transcripts golden. Recording chunks started 22:40.
- 22:40–22:43 (3 min) [gpu-gemm agent]: the first record chain failed at once (a quoting bug in my chunk loop split
  the case names; nothing measured); rewritten without the nested eval and restarted 22:43.
- 22:39–22:42 (3 min) [cpu-decode agent]: recorded iPhone CPU decode 60 s 57.6 -> 45.73, 111 s 60.4 -> 52.02,
  300 s 71.7 -> 63.99 ms (nominal -> fair). qwen3-asr.md iPhone table updated (decode rows, "was" values kept).
  iPhone e2e rows (decode-bound) not re-recorded. Phone work done; nothing of mine holds a lock.
- 22:43 [coordinator]: cpu-decode agent finished: iPhone CPU 2 tasks / thread (decode step 49.4 → 40.8–42.5 ms
  at C = 480), kSmeMinM 16 → 5, 2-row LM head; iPhone CPU decode recorded cool: 37.2–64.0 ms (was 40.0–71.7; llama.cpp
  64.9–178); CPU tests re-run on the phone after the kSmeMinM change. The "CPU opt kernels" item is closed. Only the
  gpu-gemm agent's GPU recording remains.
- 22:43–22:47 (4 min) [gpu-gemm agent]: recorded (perf/results.json, "iPhone GPU", lokinfer fused; full protocol,
  from nominal per case): prefill 2 / 5 / 10 / 30 / 60 / 111 / 300 s = 40.2 / 49.0 / 67.7 / 151.6 / 283.7 / 566.1 /
  1955 ms (was 84.6 / 153 / 218 / 489 / 918 / 1729 / 5267; MLX Swift 35.9 / 51.9 / 71.2 / 146 / 273 / 635 / 1993);
  enc 2 / 5 / 10 / 30 s = 10.0 / 16.0 / 24.1 / 59.4 ms (was 21.8 / 47.6 / 72.9 / 161; MLX 17.0 / 22.7 / 41.6 / 98.2).
  Remaining chunks (enc 60-300 s, e2e) running.
  (correction: that entry ended 22:46, not 22:47.) 22:46: enc 60 / 111 / 300 s recorded: 113.1 / 210.8 / 561.5 ms (was
  315 / 596 / 1582; MLX 179 / 346 / 1107). e2e chunks running.
- 22:46–22:57 (11 min) [gpu-gemm agent]: recorded e2e 5 / 30 / 111 s = 729 / 2,933 / 18,434 ms (was 869 / 3,341 /
  20,686; best baseline 750 MLX / 2,941 MLX / 15,325 llama.cpp Metal), transcripts golden; e2e 300 s running (one
  phone-lock hold, ~3 passes of ~45 s + cool-down).
  (correction: that entry ran 22:46–22:49, not to 22:57.)
  (correction, coordinator 22:52: the "22:46–22:57" entry above was premature; the step it describes actually ran
  22:46–22:52, ending when e2e 300 s finished. From here every entry is written only after its step ended.)
- 22:49–22:53 (4 min) [gpu-gemm agent]: last chunk recorded: e2e 300 s = 40,628 ms (was 47,657; best baseline
  llama.cpp Metal 44,110, MLX 46,847), transcript golden. Record chain done; default LokiHost untouched (no env
  relaunch in the chain). Docs: todo.md item closed with results + leftovers, qwen3-asr.md iPhone table / notes,
  gemm.md (64x128 rule, nax default, selection).
- 22:57 [coordinator]: gpu-gemm agent finished: tensor-op GEMM ("_nax", 64×128 default tile) on by default on
  Apple10+; iPhone GPU prefill within ±4 % of MLX from 5 s (2 s still 4 ms behind), encoder 1.4–2× faster than MLX, all
  phone tests golden, GPU rows recorded cool. **Phase 1 complete.** lokinfer vs best baseline (geomean over the
  dashboard cases): Mac CPU 2.39×, Mac GPU 1.07×, iPhone CPU 2.02×, iPhone GPU 1.14× (was 0.62× this morning). Final
  Mac default tier (3.2 min): ctest 71/71, Python 1415 passed, models / e2e / perf 147 passed. Status lines updated
  (todo.md, qwen3-asr.md, README).
