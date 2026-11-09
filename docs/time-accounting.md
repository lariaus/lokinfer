# Where the time goes (iPhone vs Mac)

Measured wall times of everything done on the Qwen3-ASR work, to plan the next experiments (user, 2026-11-06: "how
long everything takes … for the next experiments … measure all that"). Times come from the logs' start / end stamps.
The compact planning table is in [big-models.md](big-models.md) ("How long things take"); the step-by-step history in
[worklog.md](worklog.md).

## Reference: how long each kind of work takes

Wall time from start to "done and verified" (agent working time + machine time), from the 2026-11-06 examples below.
Use it to size the next experiments; a step that grows past ~20–30 min gets split.

| work effort | typical time | example (2026-11-06) |
|---|---|---|
| **iPhone round trip after any native change** (build, deploy, 71 on-device tests) | 4–5 min | 5 deploys |
| **runtime bug, found by measurement → fix → Mac repro → deploy → re-measure** | ~40 min | GPU memory kept after free (residency trim) |
| **small remote-protocol feature** (server message + client + test + docs) | ~15 min | BLOBS message |
| **follow-up compatibility fix** (old server on the phone) | ~10 min + a lost 10 min run | `blobs_query` flag |
| **bench script option** (code + doc) | 5–10 min | `--quick`, `--cool-to` |
| **test-harness change** (code + test update + suite in 2 modes) | ~20 min | conftest mirrors |
| **C++ API + binding + Python use + byte-identical test** | ~20 min | `lkm_split` |
| **build-system bug diagnosed from a test failure** | ~25 min | Debug -O0 shaders (3 wrong guesses first: env, metallib staleness) |
| **flaky-test investigation** | ~20 min | leaked `lokinfer-serve` processes |
| **device-memory bug in the test harness** (guess-and-fix vs measure) | 85 min guessing, 6 min once measured | `LOKINFER_TEST_MEMLOG` |
| **one iPhone baseline batch** (12 long-clip cases, one framework) | 6–8 min run + 10 min cool-down | MLX Swift, llama.cpp Metal |
| **llama.cpp CPU long clips on the iPhone** | 30+ min, hangs / heat-bound | needs a cool phone, alone |
| **re-measure every lokinfer iPhone row** (34 cases, cool per case) | ~40 min incl. deploy | night queue |
| **a full basic-kernel sweep on the Mac** (≤ 300 s, both devices) | ~65 min | 6 + 25.5 + 35 min |
| **basic 1200 s cases, CPU** | ~45 min | 405 s per prefill call |
| **per-kernel profiles, 3 stages × 2 devices** | Mac < 1 min, iPhone ~6 min | |
| **Mac default-tier test suite** (unit + Python + models) | ~4–5 min | |
| **Mac slow tier, everything** (1,533 tests) | ~6 min | Qwen 56 in 1.5 min + the rest in 4.4 min |
| **Mac long tier** (4 Qwen 1200 s tests) | 3 min | |
| **iPhone slow-tier kernel test file** | ~3 min | fused CPU kernels |
| **iPhone fused GPU kernels + Python suite, slow tier** (277 tests) | ~5 min | with the memory-threshold harness (20 min with a reconnect per test) |
| **iPhone Qwen model slow tier** (60 tests) | ~4 min | |
| **kernel timing table, basic vs opt** (25–28 cases, one device) | Mac CPU 2.4 min, iPhone CPU 4.3 min, iPhone GPU 4 min | `bench_f16_fused_gpu.py [--cpu] [--iphone]` |
| **docs sweep after a phase** (5 docs, agent) | ~3 min agent time + review | |
| **todo audit with evidence** (agent) | ~3 min + review | 22 items |
| **updating result tables in the docs** | 5–10 min per table | iPhone comparison table |
| **standalone CPU GEMV microbench round** (1 GB distinct weights, 5–7 variants × 6 alternating rounds) | ~5–20 s run + ~2 min to write a variant | 2026-11-07 decode GEMV |
| **in-model Mac CPU A/B** (decode 2 s + 30 s, 3 alternations, one process per side) | ~1.2 min | 2026-11-07 |
| **Mac CPU record, decode 2 s … 1200 s + e2e 5 … 300 s** | ~7 min (+ the wait for load < 2) | 2026-11-07 |
| **s300 transcript, one CPU pass** | ~45 s | 2026-11-07 |
| **iPhone kernel-constant variant** (setvar + incremental deploy 71/71 + decode profile) | ~1 min (deploy ~15 s) + cool-down; the phone-lock wait behind other agents was 5–16 min per session | 2026-11-07 tasks-per-thread A/B |
| **iPhone CPU decode record, 7 cases, cool per case** | ~12 min (two sessions) | 2026-11-07 |
| **iPhone memory-pressure experiment** (balloon 0 → 3.5 GB, 20 decode steps per point, 2-3 pin modes) | ~2 min GPU, ~5 min CPU (incl. graph load) | 2026-11-07 `pressure_qwen3asr_iphone.py` |
| **small server message + client + test, deployed** (MEMORY pressure fields, PIN_MODEL) | ~10 min code + 1.5 min deploy | 2026-11-07 |
| **new GPU kernel family, first correct version** (Metal 4 tensor-op GEMM: write, Mac debug of 2 toolchain/layout surprises, tests) | ~15 min | 2026-11-07 `f16_tensor.metal` |
| **iPhone fused GPU kernel tests, default tier** (139 + 119 skipped) | 7.6 min | 2026-11-07, incl. ~5 min waiting for the phone lock |
| **Qwen model tests, Mac GPU, slow tier** | ~0.5 min | 2026-11-07 (28 tests) |
| **iPhone fused GPU + rms-GEMM kernel tests, slow tier** (193 + 101 skipped) | 13 min | 2026-11-07, LokiHost relaunched with an env var |
| **iPhone Qwen model tests, slow tier, CPU + GPU** | ~4 min | 2026-11-07 (56 tests) |
| **iPhone in-model `--quick` A/B, 4 GPU cases × 2 sides** (cool to fair, <= 60 s per case) | ~7 min | 2026-11-07 tile / steel A/B |
| **waiting for the shared phone lock** (3 agents, <= 15 min sessions) | 3–15 min per session | 2026-11-07 evening |

## 2026-11-06 (afternoon → night)

### iPhone

| step | wall time | notes |
|---|---|---|
| `scripts/ios_deploy.sh` (iOS build, install, 71 on-device unit tests, relaunch) | ~4–5 min | 5 deploys today ≈ 22 min |
| lokinfer CPU e2e 300 s (3 passes) | 6.7 min | hot phone: 113.6 s per pass |
| lokinfer GPU e2e 300 s (fresh LokiHost) | 6.3 min | 46.9 s per pass |
| llama.cpp Metal long clips (12 cases incl. verify s300) | 6.4 min | + ~1 min install / data copy |
| MLX Swift long clips (12 cases) | 6 min | after a 10 min idle cool-down |
| llama.cpp CPU long clips, 1st attempt | ~9 min **lost** | hung mid-encode on a hot phone |
| llama.cpp CPU long clips, 2nd attempt (11 of 12 cases) | 32 min | stopped in e2e 300 s: ~9 min per call hot |
| CPU fused-kernel tests, slow tier (108) | 2.9 min | |
| lokinfer GPU long cases, attempt 1 | ~5 min | stopped when the app was closed on the phone (4 rows kept) |
| lokinfer GPU long cases, attempt 2 | ~10 min **lost** | 10 min cool-down, then my client bug (BLOBS mode 3 sent to an older server) |
| lokinfer GPU long cases, attempt 3 (6 cases) | 15 min | incl. the waits for "nominal" |
| per-kernel profiles (6 runs, CPU + GPU) | ~6 min | |
| lokinfer CPU re-measure while hot (20 of 24) | ~12 min + 10 min cool **superseded** | still "serious" after 10 min idle on the charger; replaced at night |
| phone unavailable (user) | 18:05 → 21:39 | Mac-only work |
| **night**: deploy | 4 min | |
| **night**: all lokinfer rows from "nominal" (CPU 24 cases, GPU 10) | 36 min | B1 decode 1 min, B2 prefill 5, B3 enc + e2e 5/30 s 2, B4 e2e 111/300 s 11, B5 GPU 17 |
| **night**: GPU kernel bench (25 cases, basic vs opt) | 4 min | |
| **night**: idle cool-down before llama.cpp CPU | 10 min | |
| **night**: llama.cpp CPU e2e 300 s (verify + 1 warm-up + 3 runs) | 26 min | 3.4 min verify, ~3.7 min per call, one 9.4 min throttled call |
| **night**: push Kokoro / other baseline data (~2.9 GB) | 2 min | |
| **night**: final deploy | < 1 min | nothing rebuilt |
| **night**: CPU kernel table, Mac (28 cases) | 2.4 min | `bench_f16_fused_gpu.py --cpu` |
| **night**: CPU kernel table, iPhone (25 cases) | 4.3 min | started at "nominal", no wait |
| **night**: `lokinfer basic` iPhone GPU (14 cases ≤ 30 s) | 12.5 min | incl. 62 s first load (basic weights uploaded) |
| **night**: `lokinfer basic` iPhone CPU (14 cases ≤ 30 s) | 17 min | prefill 30 s 8.3 s per call |

### Mac

| step | wall time | notes |
|---|---|---|
| GPU memory leak repro (local server, load + prefill + reconnects) | ~1 min per run | 4 runs |
| Kokoro CPU + GPU re-time (16 cases × 2 devices × 2 sweeps) | 7 min | + 1 min isolated re-check |
| per-kernel profiles (6 runs, CPU + GPU, 30 s clip) | < 1 min | |
| `lokinfer basic` ≤ 30 s (28 rows, both devices) | 6 min | |
| `lokinfer basic` 60–300 s, CPU (11 rows) | 25.5 min | e2e 300 s: 172 s per call |
| `lokinfer basic` 60–300 s, GPU (11 rows) | 35 min | e2e 300 s: 380 s per call (attention not key-split) |
| `lokinfer basic` 1200 s, GPU | 3.5 min, then fails | GPU watchdog (one > 10 s dispatch) |
| `lokinfer basic` 1200 s, CPU (3 rows) | 42 min | prefill 1200 s: 405 s per call × 4 |
| basic s300 divergence check (one transcription) | ~4 min | |
| test suites: Python non-model (1,322) | 60–77 s | |
| test suites: models + e2e + perf default tier (147) | 2:15–2:32 | Qwen default tier alone: 8 s |
| test suites: `--device=serve` (python + fused kernels, 314) | 65–90 s | |
| `--quick` bench, 3 Mac GPU cases | 8.5 s | vs minutes with the full protocol |
| full CMake rebuild after the shader-flag change | ~1–2 min | incremental `lokinfer-serve`: ~20 s |

### Night 2026-11-06 21:39 → 2026-11-07 01:35 (autonomous, phone available)

| block | time | what |
|---|---|---|
| lokinfer iPhone rows re-measured cool (34 cases) | 40 min | incl. deploy |
| iPhone kernel bench rows (GPU) + CPU kernel tables (Mac + iPhone) | 11 min | |
| llama.cpp CPU e2e 300 s + data push + redeploy | 28 min | incl. 10 min cool-down |
| `lokinfer basic` on the iPhone (28 cases ≤ 30 s) | 30 min | |
| Mac slow + long tiers | 9 min | all green |
| `--device=iphone` harness memory bug (4 fix attempts) | 85 min | found by measuring in 6 min |
| full default-tier suite through the phone + the Kokoro SNR finding | 32 min | 1364 passed, 1 failed (Kokoro app47m, for the user) |
| docs / logs / tables in between | ~25 min | overlapped with runs |

Where the phone time went tonight: ~3.4 h busy of 3.9 h (deploys < 1 min each when incremental; cool-downs ~20 min;
the rest measurements and tests). Mac: benches / tests ~1.5 h, the rest editing, analysis and docs alongside the
phone runs.

### Lost / duplicated time

| what | time | lesson |
|---|---|---|
| orphaned pre-restart Claude Code session doing the same work in parallel | ~1 h (duplicate load, one wrong todo edit) | check for orphans after a restart |
| leaked `lokinfer-serve` processes → test timeouts | ~20 min debugging | fixtures kill the server in `finally` |
| monitors / waiters matching their own command line (`pgrep -f`) | 3 stale watchers | `pgrep -f "[p]attern"` or wait on a PID |
| hot-phone measurements later superseded | ~45 min of phone time | cool before every case (`--cool-to`), time off the charger |
| llama.cpp CPU hang + hot e2e 300 s | ~40 min | run the long baseline cases alone on a cool phone |
| network outage, 2026-11-07 07:55–15:09 (API unreachable; 3 agents stopped mid-task) | 7 h 14 min | resume agents from their transcripts; check locks / half-applied edits first |
| `--device=iphone` session crashes from accumulated models (night, 23:51–01:15) | ~85 min of runs + fixes | 4 attempts: reconnect per test (works, 5.7× slower), memory-threshold reconnect (freed a fixture's state), FREE_MODEL by weakref (temporaries' ids lost), FREE_MODEL by id (works, same speed). Measure first: `LOKINFER_TEST_MEMLOG=1` found the cause in 6 min |

## 2026-11-07 (perf + memory with 3 parallel agents, finishing phase 1)

| block | time | notes |
|---|---|---|
| 3 agents, Mac-side work (07:44 → ~15:50) | ~1.5 h of work + a 7 h 14 min network outage (07:55–15:09) | GPU GEMM, CPU decode GEMV, memory pressure, in parallel by file area |
| phone taken (15:32 → 18:46), Mac-only | ~3 h 15 min | agents finished Mac-side parts; coordinator made the unvalidated tensor GEMM opt-in |
| phone validation, first try (18:46 → 19:17) | 31 min, then the phone dropped ("unavailable") until ~21:10 | kernel slow tier with tensor ops passed; CPU kernels validated |
| phone validation + sweeps + recording (21:10 → ~22:55) | ~1 h 45 min of phone time, 3 agents sharing it through the lock | tensor GEMM default (+ "_nax" ~2×), CPU settings sweep (3 deploys), memory A/B + table, GPU / CPU rows recorded cool |
| close phase 1 (status lines, comparison, final Mac tier) | ~10 min | |

Lessons: phone time is the bottleneck when 3 agents share it (lock queueing up to ~14 min per session); agents' first
worklog entries had future end times until told to log only after a step ends; an unvalidated kernel must not become
a device default (made opt-in until the phone tests passed).
