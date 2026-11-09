# Working with large models: fast iteration

Lessons from bringing up Qwen3-ASR-1.7B (4 GB of f16 weights, 2026-11-05/06), the first model that is big for the
Mac's memory and for the iPhone. Everything that is cheap for a 100 MB model (rebuild, upload, a full benchmark sweep,
the test harness's conveniences) costs minutes to hours at 4 GB. Read this before porting the next large model.

## Before you launch anything

- **Estimate the wall time** (calls × runs × cases) and write the estimate down. Anything over ~20–30 min: shrink it
  (short cases first, long cases only where they matter, decision 11's slow rule: 1 warm-up + 3 runs for calls
  > 10 s) or split it into chunks. A full-dashboard sweep of every case once held the Mac for 4.5 h.
- **While it runs, look at it every 5 minutes**: its log, who holds each `/tmp/lokinfer-*.lock` (`lsof -t`), stuck
  processes, the phone reachable, swap. Never wait idle for a notification ([benchmarking.md](benchmarking.md),
  "Long runs: watch them"). A watcher loop that tests "still running?" with `pgrep -f PATTERN` matches its own
  command line and never ends: use `pgrep -f "[p]attern"` (2026-11-06).
- **One big model job at a time** (`/tmp/lokinfer-bigmodel.lock`); weights stay 2-byte and memory-mapped; f32 / f64
  references upcast per layer; long passes chunked. Check `sysctl vm.swapusage` before starting.

## Machine state (timings you can trust)

- **Display asleep → the Mac idle-sleeps** (`pmset sleep 1`): runs pause for minutes (`perf_counter` stops, the
  wall clock does not). Overnight work runs under `caffeinate -i -s` (system awake, display may sleep).
- **Display awake + aerial wallpaper → noisy GPU timings** (prefill 10 s read 64–86 ms across runs). Record GPU rows
  with the display asleep.
- **Other busy processes skew CPU medians**: with 8 pool threads on 8 P cores, any other core-hungry process (the
  editor's renderer) delays every barrier; CPU medians drift 10–15 % above p10 at times.
- **Heat drifts GPU timings ~10 % between separate runs**: only alternating A/B runs (A B A B, same process or back
  to back) decide small changes. The kernel bench reuses its weights from the 48 MB SLC: judge DRAM-bound kernels
  (decode GEMVs) in-model, not there.
- **iPhone Low Power Mode** throttles CPU / GPU (iOS can switch it on at low battery): the iPhone bench refuses to time
  with it on, and the dashboard never lets an LPM row replace a faster full-power one. Keep the phone charged.
- **iPhone**: Auto-Lock **Never** (a locked phone can't launch LokiHost; `devicectl` says "could not be, unlocked");
  after ~1 h of tests the phone sits at thermal "serious" and its numbers are throttled (the dashboard keeps the
  state; re-measure cool for final rows); the weekly provisioning-trust prompt blocks launches until trusted.

## The iPhone: moving 4 GB around

- **Weights go up once** (split transport, [runtime.md](runtime.md) / `RemoteDevice.load`): one blob per constant,
  keyed by sha256, cached on the phone across sessions and graphs. A graph or kernel change re-sends only the
  skeleton (KB); the CPU and GPU graphs share their weights. Whole-model uploads could never work: 2 GiB frame limit,
  and macOS rejects single socket sends ≥ 2 GiB (EINVAL). Since 2026-11-06 a `CompiledModel` is split in memory
  (`lkm_split`: no .lkm written, no copy of the weights) and the device answers "which blobs do you have" in one
  round trip (BLOBS query) instead of one HAS_BLOB per constant.
- **Large mutable state (KV caches) lives on the phone**: allocate it with `dev.alloc_tensor` and call the graph with
  `dev.run2(mid, [..., kv])`. The test harness's transparent path (`--device=iphone`) used to upload a mutable host
  array and copy it back on *every* call and also re-run the graph on the Mac: for a 5 min clip's 0.5 GB cache that
  moved 39 GB each way in one test run (stopped after 39 min; the device-resident version takes 72 s). Since
  2026-11-06 it keeps a device mirror per mutable argument (re-sent only when the host contents changed) and runs the
  graph on the Mac only on a model's first call; the read-back stays, so explicit device tensors are still the fast
  path for big caches. `prof-lokinfer -d iphone-*` holds inputs ≥ 1 MB on the phone and profiles through RUN2.
- **Long `--device=iphone` test sessions**: the server keeps a connection's models until it ends; the harness frees
  each test's own models (FREE_MODEL) and reconnects after a crash so it fails one test, not all the rest. Rejected:
  a reconnect per test (5.7× slower: 3.5 → 20 min) and a reconnect above a memory threshold (it also freed the Qwen
  module fixture's own models and KV tensors).
- **Time on the device**: RUN2's per-iteration times exclude the transfers; e2e = the sum of the calls' device
  times (model time only).
- **iOS GPU time limit per command buffer** (much shorter than macOS's): the runtime's work budget splits heavy
  graphs ([runtime.md](runtime.md), "GPU work budget"). A GPU timeout on the phone = too much work per buffer, not
  necessarily a slow kernel.
- **Each client's models are freed when it disconnects** (they used to pile up in LokiHost across test runs and
  benches until the GPU timed out). Within one session, load a big model once and keep its id.
- **Never wrap `scripts/ios_deploy.sh` in `lk-lock`**: it takes the locks itself (an outer lock deadlocked for 23
  min). Its watchdog relaunches LokiHost only when the app is really gone (a dropped `devicectl` console connection
  used to kill a healthy server mid-test).

- **Memory pressure on the phone** (other apps open): a CPU model's mmapped weights are the first thing iOS evicts,
  and a decode step then re-reads them from flash (seconds per step). Check `dev.memory_check()` / MEMORY's
  `pageins` around timed regions (the Qwen iPhone bench does), close other apps, and see PIN_MODEL
  ([runtime.md](runtime.md), "Memory pressure"). Reproduce it on purpose with `pressure_qwen3asr_iphone.py`.

## Benchmarks

- `bench/models/bench_qwen3asr.py` (Mac) / `bench_qwen3asr_iphone.py` (phone) are the templates: golden inputs,
  dashboard case names, decision-11 runs, `--case` to run a few, `--record` to write the dashboard.
- Reuse the caller's buffers the way the contract allows (one KV cache reused across calls and passes: a fresh 2.9 GB
  cache per call times the allocation and zeroing); don't add untimed extra calls (the golden check reuses the timed
  calls' output).
- Record the short cases first (minutes); long cases (20 min passes: ~5 min per call on the CPU) in their own run.
- **iPhone heat**: the phone reaches "serious" within minutes of model work and stays there while charging.
  `bench_qwen3asr_iphone.py` waits before every case until it is back to `--cool-to` (default nominal, at most
  `--cool-max-wait` 600 s); the baseline harness apps can't read the state, so give them a 10 min idle gap per launch.
- **`--quick`** for A/B while developing: 1 warm-up + 5 runs (3 for calls > 10 s), never recorded (seconds instead of
  minutes: 3 Mac GPU cases in 8.5 s). Record with the full protocol only.
- **Blob store on the phone**: `dev.blobs()` reports it, `dev.clear_blobs(keep=[...])` / `dev.remove_blobs([...])`
  prune it (superseded weights pile up: 4.8 GB on 2026-11-06). Removed weights go up again on the next load.

## Profiling

- `tools/prof-lokinfer run <target> -d mac-cpu|mac-gpu` with the model's targets (`qwen3asr_enc / _prefill /
  _decode`, `python/lokinfer/prof/targets.py`): per-kernel times in a minute. Per-dispatch GPU times from
  `raw_profile.json` (`[op, cpu_start, cpu_end, gpu_start, gpu_end, ints]`) show bandwidth per kernel.
- The GPU profile shows ~0.5 ms gaps at segment boundaries that unprofiled calls don't have: compare e2e with the
  kernel sum before chasing them.
- iPhone with mutable inputs: prof-lokinfer's remote path doesn't support them yet; use `dev.set_profile(mid, 1)`,
  `run2`, `dev.take_profile(mid)` directly (same record format).
- Microbenchmarks first for a new kernel idea (a standalone C++ file in the scratchpad: e.g. BNNS vs convert +
  sgemm, small concurrent sgemms), then in-model A/B.

## Correctness at this size

- Test tiers (default / `--run-slow` / `--run-long`, [correctness-testing.md](correctness-testing.md)): the default
  tier covers every code path in ~1 min per device.
- Model-level f16 rules: RMS + max on activations, argmax + loose RMS on logits, transcripts exact
  ([correctness-testing.md](correctness-testing.md), "Model-level f16 rules"). Check that a rule has power with
  injected faults (weight noise at bf16 precision, a wrong RoPE θ) before trusting it.
- No model-level E2E on the basic kernels for big models (hours): the basic kernels are proven by their op tests.
