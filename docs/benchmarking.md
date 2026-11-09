# Benchmarking and profiling

How lokinfer measures speed, decides between kernel variants and detects regressions. How to *explain* a number
(per-kernel profiles on all four devices, Instruments traces, shader inspection, lokinfer vs PyTorch):
[profiling.md](profiling.md). Sections are marked
**[built]** or **[planned]**; the methodology is fixed now so every number produced from here on is comparable.

## Principles

- **Kernels are benchmarked through the kernel library, not the runtime** (original plan). Model and graph
  timings go through the runtime; kernel tuning does not.
- **Compare CPU to CPU and GPU to GPU**, and against the best framework that works on that device (PyTorch
  eager and torch.compile on CPU and MPS, MLX / MLX Swift on the Mac GPU; on the iPhone ExecuTorch XNNPACK on the
  CPU and MLX Swift on the GPU, [baselines.md](baselines.md)). **iPhone targets** (user decision, todo M0; computed in
  M2): an artificial target = the Mac result × (Mac peak / iPhone peak) per device class, peaks measured in the M2
  conv proof (`impl-notes/m2-conv-proof.md`); the goal is to beat it.
- **Every result records the machine state**: Low Power Mode, thermal state, concurrent CPU load, device,
  commit. Numbers measured before 2026-11-01 were taken in Low Power Mode; the Kokoro baselines were re-measured
  with it off (todo M0), MNIST, the runtime-overhead benchmark and the research numbers below were not yet.
- **Median plus spread** (p10/p90 or IQR) over rounds, after warm-up; re-run when the spread is large. Never a
  single run.
- **Report a percentage of the measured peak** next to absolute time (TFLOP/s, GB/s), so a kernel's headroom is
  visible without context.

## Local configuration

Machine-specific settings live in a git-ignored `lokinfer.local.json` at the repo root (copy the committed
`lokinfer.local.example.json`). Each key can be overridden by an environment variable:

| key | env | use |
|---|---|---|
| `iphone_udid` | `LOKINFER_IPHONE` | the test iPhone (remote executor, `scripts/ios_deploy.sh`, iPhone benches) |
| `ios_team_id` | `LOKINFER_TEAM` | Apple development team used to sign LokiHost |
| `ios_bundle_id` | `LOKINFER_BUNDLE_ID` | LokiHost bundle id (default `com.example.lokinfer.testhost`) |
| `baselines_dir` | `LOKINFER_BASELINES` | separate checkout of the reference-baseline harnesses (`<baselines>` in the docs) |
| `english_practice_dir` | `LOKINFER_ENGLISH_PRACTICE` | the english-practice app checkout (Kokoro reference and e2e corpus) |
| `mlx_python` | `LOKINFER_MLX_PYTHON` | Python of a separate venv with MLX, for the MLX baselines |
| `work_dir` | `LOKINFER_WORK_DIR` | downloads and scratch for research scripts (default `.cache/work` in the repo) |

In the docs, `<udid>`, `<team-id>`, `<bundle-id>`, `<baselines>` and `<english-practice>` stand for these values.

## Tools

| tool | status | use |
|---|---|---|
| `bench/runtime_overhead.cpp` (Google Benchmark) | built | runtime overhead vs direct kernel call |
| `bench/models/bench_mnist.py`, `bench_kokoro*.py`, `bench_m6_models.py`, `bench_remote_models.py` (iPhone), `bench_*_baselines.py` | built | model vs PyTorch / MLX / ExecuTorch baselines, Mac and iPhone |
| `perf/results.json` + `perf/index.html` (`scripts/perf_dashboard.sh`) | built | latest E2E time per model / device / case / implementation, grouped bar charts ([perf-dashboard.md](perf-dashboard.md)) |
| `bench/research/*` | built (one-off) | the research micro-benchmarks (Metal, AMX, Kokoro baselines) |
| `tools/kbench` (`bench/harness/`): kernel harness over every registered kernel at its real model shapes | built | `kbench run -d mac-cpu,mac-gpu,iphone-cpu,iphone-gpu`: one-kernel graphs, CPU wall time ÷ R, GPU counter-heap timestamps of R dependent dispatches in one command buffer ÷ R ([Kernel harness](#kernel-harness-built)) |
| `build-release/lokinfer_bench_kernels` (`bench/kernels/kernel_gbench.cpp`, Google Benchmark) | built | the same CPU cases called directly through the kernel registry (no runtime) |
| Kernel harness on the iPhone | built (remote executor: `kbench run -d iphone-cpu,iphone-gpu`; LokiHost `--bench` / `--bench-amx` for the conv proof and the AMX bench); first iPhone run pending | both devices, same cases |
| Results DB `perf/kernels.sqlite` + `kbench compare` | built | keyed by device, build id, kernel, shape, power state; history kept; compare two runs and flag regressions |
| Conv-proof benches (`bench/kernels/conv1d_cpu.cpp`, `conv1d_gpu.cpp`, `conv_bench.cpp`) and `bench/kernels/amx_bench.cpp` (`lokinfer-bench-conv amx`) | built | M2 conv candidates; multi-stream sgemm re-verification ([m2-conv-proof.md](impl-notes/m2-conv-proof.md)) |
| Per-kernel profiler in the runtime + `prof-lokinfer` (CPU wall time per kernel, Metal 4 counter-heap GPU timestamps per kernel, dispatch log, memory plans, IR per pass, HTML report; Mac + iPhone) | built | the optimize loop: [profiling.md](profiling.md) |
| `prof-torch` + `mlogs-viewer` (PyTorch reference per-op profile in the same format; localhost viewer, side-by-side compare) | built | lokinfer vs torch by op category: [profiling.md](profiling.md) |
| `xctrace` CLI (Metal System Trace, GPU counters / shader profiler, Time Profiler, CPU Counters) via `prof-lokinfer trace` | built | deep dives, both devices: [profiling.md](profiling.md) |
| Speed-regression tests (`tests/perf/test_kokoro_speed.py`) | built: Kokoro real117, Mac CPU + GPU, optional iPhone; planned: other models | fail thresholds = baseline × margin tuned from measured noise (below) |

## Performance dashboard

**[built]** End-to-end model numbers are reported as grouped bar charts (one per model and device; Y = time per
forward pass, X = input cases, one color per implementation) on a local web page. The `bench/models/bench_*.py`
scripts update `perf/results.json` automatically through `bench/perfdb.py`;
only the entries a run measured change. Open it with `scripts/perf_dashboard.sh`. Details, schema and naming
conventions: [perf-dashboard.md](perf-dashboard.md).
Reference baselines (frameworks, versions, methods, reproduction): [baselines.md](baselines.md).

## Kernel variants

Same-semantics kernels carry a variant suffix (`gemm_f32_cpu`, `gemm_f32_cpu_accelerate`, `gemm_f32_gpu`,
`gemm_f32_gpu_mps`). [planned] The benchmark DB (`perf/kernels.sqlite`, built) is meant to produce a **per-device variant table** (kernel family × shape
class → fastest variant) that the Python layer can consult, so switching a model to a faster variant is a table
update, not a code edit. This is the seed of the future `lk.nn` layer.

## Reference numbers (M2 Max)

Re-measured out of Low Power Mode (2026-11-02, AC; [m2-conv-proof.md](impl-notes/m2-conv-proof.md) "Multi-stream
AMX"): Accelerate `sgemm` ≈ 1.2–1.55 TFLOPS per single-threaded stream, 1.9–2.7 with its own threading, 3.0–3.4
aggregate from three concurrent single-threaded streams and up to 4.7 with six (512×768×768). Runtime overhead
(below): ≈ 320 ns per one-kernel CPU graph call, ≈ 94 µs per one-kernel GPU graph call. Still from the LPM research
(`research-kernels.md`, `research-metal-runtime.md`; not re-measured): one P-core NEON f32 ≈ 74 GFLOPS; GPU f32
≈ 9.8 TFLOPS peak, ≈ 335 GB/s DRAM; Metal 4 dependent dispatch ≈ 2 µs, independent ≈ 0.2 µs; CPU↔GPU round trip
≈ 120 µs.

## Runtime overhead [built]

`build-release/lokinfer_bench_runtime` (`bench/runtime_overhead.cpp`, Google Benchmark; run with
`--benchmark_min_warmup_time=1 --benchmark_min_time=0.4s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true`,
median). 2026-11-02, M2 Max, LPM off, AC:

| case | 1 024 floats | 1 M floats |
|---|---|---|
| direct `add_f32` call | 58 ns | 122.0 µs |
| one-kernel CPU graph (`Executable::run`) | 379 ns | 122.5 µs |
| one-kernel GPU graph | 94.4 µs | 433 µs |

The CPU graph costs ≈ 320 ns per call over the direct call (input checks, output allocation, one interpreter
step); the old LPM figure was ~200 ns. A GPU call is dominated by the submit-to-complete latency of one command
buffer. The kernel harness amortizes both by repeating the kernel R times per call (below); its runtime path and
the direct-call Google Benchmark path agree within 4 % (geometric mean over 64 CPU kernels).

## Kernel harness [built]

`tools/kbench` (code in `bench/harness/`; `tools/install-prof-tools.sh` links it). Goal: every registered kernel
benchmarked **once, at the shapes real models run it with**, on all four devices, results in a DB with history.

1. **Cases** (`kbench extract` → `bench/harness/cases.json`, checked in): builds every model of the profiler
   registry (`lokinfer.prof.targets`: Kokoro app59f, MNIST B=64, MobileNetV2 / ResNet-18 N=1, DCGAN G/D N=1, GPT-2
   prefill T=128 and decode P=512) for the CPU and the GPU, plus the basic Kokoro CPU graph, runs one profiled call
   on the Mac and pairs each exec kernel op of the final IR (attribute names and values, scalar operands) with the
   profile's concrete int arguments (symbolic dims resolved). Identical (kernel, signature, attributes) are merged;
   `instances` counts them per model. 2026-11-02: 446 cases (237 CPU, 209 GPU) over 117 of the 169 registered kernels; the 52
   kernels no model build runs (alternative variants: steel/gemv GEMMs, 2-D Winograd, attention variants, ...) have
   no real shape and are listed by `kbench coverage`.
2. **One case = one stand-alone graph** (`cases.case_ir`): exec inputs = the operands (seeded uniform [-1, 1];
   valid data where the kernel checks it: durations summing to `frames`, embedding ids in range, the Winograd
   transform table, decode position 512), outputs allocated once, the kernel **repeated R times on the same
   buffers** (R so one call lasts ≈ 2 ms, ≤ 64). On the GPU the repeats are write-after-write dependent dispatches in
   one command buffer (a barrier between each). Compiled with an empty pass pipeline (verifier only).
3. **Timing** (`kbench run`): the dashboard protocol (≥ 3 calls and ≥ 1 s warm-up, then ≥ 30 calls and ≥ 2 s,
   median, p10/p90). CPU kernels: wall time per call ÷ R. GPU kernels: then ≥ 10 profiled calls; the GPU time is
   last end − first begin of the Metal counter-heap timestamps of the R dispatches, ÷ R (the per-dispatch timestamp
   cost is spread over R). Wall time ÷ R is stored too. TF/s for GEMM / conv kernels (`analysis.flops_of`).
   Mac: in process. iPhone: the same graph through LokiHost's remote executor (`RUN` with warm-up / iteration
   counts, `PROFILE` / `TAKE_PROFILE` for the GPU timestamps).
4. **DB** `perf/kernels.sqlite` (`bench/harness/kdb.py`): `runs` (device, hardware, build id = hash of the
   `_lokinfer` module + metallib on the Mac, LokiHost's Mach-O UUID on the iPhone; Low Power Mode, power, thermal
   at start / end, harness `runtime` | `gbench`, protocol settings) and `results` (case key, kernel, signature,
   models, R, n, median / p10 / p90 / wall µs, GFLOP, TF/s, thermal after the case, error). Every run is kept.
   (`source_rev` is filled from `.git` when the tree is a repository; it is not one today.)
5. **Report**: `kbench ls`, `kbench show RUN`, `kbench compare A B [--threshold 0.1] [--fail]` (B / A per case;
   a regression must exceed the threshold **and** fall outside both runs' p10–p90; geometric mean; exit 1 on
   regressions with `--fail`), `kbench coverage`. `A`/`B` are run ids or `latest:<device>` (`latest:mac-gpu`).
6. **Google Benchmark path** (`kbench gbench-export DIR [--one-per-kernel]`, then
   `build-release/lokinfer_bench_kernels DIR [--part I/N] --benchmark_min_warmup_time=1 --benchmark_min_time=0.4s
   --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=gb.json`, then
   `kbench gbench-import gb.json --dir DIR`): CPU cases called directly through `KernelEntry::prepare/run`, no
   runtime (the original plan's "through the kernel library").

Usage rules: hold the locks (Mac runs under `/tmp/lokinfer-macbench.lock`, phone runs under
`/tmp/lokinfer-phone.lock`); use `--kernel REGEX` / `--model NAME` / `--part I/N` / `--append latest:<device>` for
short, targeted sessions — a change to one kernel re-runs that kernel's cases only, then `kbench compare`.
`kbench check` compiles and runs every case once without timing (finds broken cases in seconds).

**Results 2026-11-02 (Mac, LPM off, AC, thermal nominal throughout):** run 1 Mac CPU (237 cases / 64 kernels),
run 2 Mac GPU (209 / 53), run 3 Mac CPU direct calls (Google Benchmark, one shape per kernel, 64 cases). Median
p10–p90 spread 5 %. Fastest GEMM-class cases: GPU `gemm_f32_gpu_mps` 6.6–7.3 TF/s, `gemm_splitk_f32_gpu` 4.9;
CPU `conv1d_nobias_f32_cpu` (1×1 conv, Kokoro) 2.06 TF/s, `gemm_split_f32_cpu_accelerate` 1.94,
`gemm_bias_act_f32_cpu_accelerate` 1.84. The iPhone runs (CPU + GPU) are implemented and pending.

## CPU runtime variants [built]

The original plan's mono-thread vs thread pool vs P-core-pinned QoS, as environment knobs read once by
`ThreadPool::global()` (`src/runtime/thread_pool.cpp`): `LOKINFER_THREADS=N` (pool size incl. the caller; 1 =
mono-thread) and `LOKINFER_POOL_QOS=default|utility|user-initiated|user-interactive` (workers' QoS; the shipped
default is user-interactive, the way macOS / iOS let a thread ask for P-cores — there is no hard affinity).
`bench/harness/runtime_variants.py --mac` runs each variant in its own process (mono-thread also sets
`VECLIB_MAXIMUM_THREADS=1`); `--iphone` relaunches LokiHost with the variables (`bench/harness/lokihost.sh`, which
also restores the standard watchdog; LokiHost's HELLO reports them as `runtime_env`). Results in
`bench/harness/results_runtime_variants.json`. Dashboard protocol, median ms:

| Mac CPU (M2 Max, 2026-11-02, LPM off, AC) | MNIST B=64 | MobileNetV2 N=1 | Kokoro app59f |
|---|---|---|---|
| mono-thread | 0.161 | 4.01 | 278.8 |
| pool, default QoS | 0.165 | 1.69 | 114.2 |
| **pool, P-core QoS (shipped)** | **0.161** | **1.59** | **113.9** |

- The pool is 2.5× faster than mono-thread on MobileNetV2 and 2.45× on Kokoro; MNIST B=64 runs its three GEMMs
  inline below the pool threshold, so all variants tie.
- User-interactive QoS beats default QoS by 6 % on MobileNetV2 (many short parallel kernels, where a worker landing
  on an E-core delays the join) and ties on Kokoro (long kernels). The shipped default stays.
- iPhone: **needs an iPhone run** (`runtime_variants.py --iphone`, inside the phone lock).

## Speed-regression test [built]

`LOKINFER_PERF_TESTS=1 .venv/bin/python -m pytest -s tests/perf` times Kokoro real117 (CPU and GPU) with the dashboard
timing protocol and fails when slower than the per-device baseline in `tests/perf/kokoro_speed.json` × its margin
(margins from measured run-to-run noise). Add `LOKINFER_IPHONE=<udid>` to also check the phone (remote executor).
Update the baseline after an intended speedup. Opt-in because it needs an idle machine and ~30 s.

## Long runs: watch them

Large models (GBs of weights): see [big-models.md](big-models.md) for the full playbook (estimates, machine state, moving weights to the
iPhone, device-resident state, profiling, correctness at that size).

Rule (user, 2026-11-05): Before launching any benchmark, test sweep, deploy or agent job, estimate its total wall time (calls × runs × cases) and shrink anything over ~20-30 min. While it runs, check it actively at least every 5 minutes: progress in its log, who holds each `/tmp/lokinfer-*.lock` (`lsof -t`), stuck or zombie processes, the phone reachable, swap. Make sure nothing is blocked and unblock it at once. Never sit idle waiting for a notification: keep doing other unblocked work meanwhile. (A full-dashboard sweep once held the Mac for 4.5 h while nothing else happened.)

## Shared-resource locks (concurrent sessions) [built]

Several sessions (agents) may build, benchmark and use the iPhone at the same time. Timing runs must not overlap
(a concurrent GPU/CPU load or a second phone session skews numbers by 20 %+; an unlocked iPhone sweep on 2026-11-01
read Kokoro GPU real117 at 305 ms instead of 248 ms). Wrap the command in `scripts/lk-lock LOCKFILE COMMAND...`
(blocking exclusive `fcntl.flock`; macOS has no `flock(1)`):

| lock file | held while |
|---|---|
| `/tmp/lokinfer-build.lock` | building the shared `build/` dir (`cmake --build build`) — **also take the macbench lock** around any build (`scripts/lk-lock /tmp/lokinfer-macbench.lock scripts/lk-lock /tmp/lokinfer-build.lock cmake --build build`): a build loads every core and corrupts a concurrent Mac timing run |
| `/tmp/lokinfer-macbench.lock` | any Mac timing or profiling run (bench scripts, `prof-lokinfer`, `prof-torch`, pytest perf tests), and any build (Mac `cmake --build`, the iOS build inside `scripts/ios_deploy.sh`, which takes it itself) |
| `/tmp/lokinfer-phone.lock` | any iPhone session (remote benches, `prof-lokinfer -d iphone-*`, baseline apps); `scripts/ios_deploy.sh` takes it itself (it relaunches LokiHost, killing whatever runs on the phone) unless `LOKINFER_PHONE_LOCK_HELD=1` says the caller holds it |
| `/tmp/lokinfer-bigmodel.lock` | any Mac job that holds a full multi-GB model in memory (Qwen3-ASR: torch, torch.compile, MPS, MLX, llama.cpp, golden-data capture): one at a time, so the 32 GB Mac does not swap; check `sysctl vm.swapusage` first. Timed jobs take it before the macbench lock |

Lock order is always phone → bigmodel → macbench → build when nesting (never the reverse), so sessions can't deadlock.

Shell pipelines go through `sh -c '...'`: `scripts/lk-lock /tmp/lokinfer-macbench.lock sh -c 'prof-lokinfer run ... | grep e2e'`.
