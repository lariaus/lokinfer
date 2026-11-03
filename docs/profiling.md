# Profiling and debugging

How to find out where a lokinfer model spends its time, on each of the four devices (Mac CPU, Mac GPU, iPhone CPU,
iPhone GPU), and how to compare it with the PyTorch reference. Sections are marked **[built]** (implemented and
tested) or **[planned]**. Measurement rules (timing protocol, power state) are in [benchmarking.md](benchmarking.md)
and [perf-dashboard.md](perf-dashboard.md); this page is about *explaining* a number.

## Quick start [built]

```sh
tools/prof-lokinfer run kokoro -d mac-gpu --open      # profile + HTML report in the viewer
tools/prof-lokinfer run kokoro -d all                  # Mac CPU, Mac GPU, iPhone CPU, iPhone GPU
tools/prof-torch kokoro -d mps                         # the PyTorch reference, same inputs, same report format
tools/prof-lokinfer compare <lokinfer run> <torch run> --open
tools/mlogs-viewer                                     # browse every run (http://127.0.0.1:8766/)
```

`tools/install-prof-tools.sh` links the commands into `~/.local/bin` (`prof-minfer` is an alias of
`prof-lokinfer`). If your shell already defines another `prof-torch`, use `prof-lokinfer torch ...` or
`tools/prof-torch`. Every tool runs with the repo's `.venv`. Runs go to `prof-runs/` (git-ignored; override with
`LOKINFER_PROF_DIR`).

Models: `kokoro`, `mnist`, `resnet18`, `mobilenet_v2`, `dcgan_g`, `dcgan_d`, `gpt2`, `gpt2_decode` (one decode step at `-c "decode P=512"`; the cache is not prefilled, which does not change the work) (the benches' builds, cases and
inputs, `python/lokinfer/prof/targets.py`), or a saved `model.lkm` with `--inputs args.npz` (arrays in argument
order). `-c CASE` picks the input case (Kokoro: `real28`, `real117` (default), `rand510`).

## The tools

| command | status | what it does |
|---|---|---|
| `prof-lokinfer run MODEL -d DEVICE[,DEVICE] [-c CASE] [--serialize] [--open]` | built | e2e timing + per-kernel profile + IR after every pass + memory plan + timeline + `report.html` |
| `prof-lokinfer compare A B [--open]` | built | two runs side by side: e2e, per category, per kernel name (text, or the viewer) |
| `prof-lokinfer ls` / `report RUN` | built | list runs / regenerate a run's `report.html` |
| `prof-lokinfer trace MODEL -d DEVICE -t metal\|gpu-counters\|time\|cpu-counters\|system` | built | Instruments trace with `xctrace` (Mac process, or LokiHost on the iPhone) + a text summary |
| `prof-lokinfer sample MODEL -d mac-cpu\|mac-gpu --seconds N` | built (Mac) | CPU call-stack sampling with `sample(1)` |
| `prof-lokinfer metal list\|air\|air-stats\|source\|stats\|native\|info [FUNC...] [--iphone]` | built | shader and pipeline inspection (AIR, native binary sizes, pipeline limits) |
| `prof-lokinfer devices` | built | which devices can be profiled (Mac GPU, LokiHost reachable and new enough) |
| `prof-torch MODEL -d cpu\|mps [--compare RUN]` | built | per-op profile of the PyTorch reference, same run format |
| `mlogs-viewer [--run ID] [--compare A B] [--no-open] [--stop]` | built | localhost viewer for all runs |
| `kbench run / compare` | built | per-kernel timings at real model shapes in isolation (all four devices), history in `perf/kernels.sqlite`, regression report: [benchmarking.md](benchmarking.md) "Kernel harness" |

`prof-lokinfer` refuses to touch the iPhone while `bench_remote_models.py` is running (`--force` overrides).

## Devices: how each one is profiled [built]

| device | how | GPU timing |
|---|---|---|
| Mac CPU | in process (`CompiledModel._exe.set_profile`) | - |
| Mac GPU | in process | Metal 4 counter-heap timestamps around every GPU kernel |
| iPhone CPU | LokiHost's remote executor: the graph is compiled on the Mac, uploaded (`.lkm`), run on the phone; the protocol's `PROFILE` / `TAKE_PROFILE` messages switch the recorder on and fetch the records | - |
| iPhone GPU | same | same counter-heap timestamps, on the A-series GPU |

The phone must run a LokiHost built with this protocol (`scripts/ios_deploy.sh`; `prof-lokinfer devices` says so
when it is too old). Older clients and servers keep working: the new messages are optional and advertised in HELLO's
`"features"`. No relaunch with environment variables is needed any more (the old way, `DEVICECTL_CHILD_LOKINFER_PROFILE=2`
plus the console log, still works).

## How a run is measured [built]

1. **Build** the graph for the device (`K.build(weights, gpu=True)` etc.).
2. **E2E time**, no instrumentation, the dashboard protocol: warm-up ≥ 3 calls and ≥ 1 s, then ≥ `--runs` calls and
   ≥ 2 s; median, p10, p90. This is the number to compare with the dashboard.
3. **Profiled calls** (`--iters`, default 5): the runtime records, per call,
   - every kernel: CPU wall time (start, end; for GPU kernels this is the *encode* time), its int arguments (all
     dims + int attributes, so dynamic shapes are exact per call), and for GPU kernels two GPU timestamps;
   - for GPU kernels, the CPU time spent blocked on the previous in-flight command buffer before encoding
     ("wait in-flight segment": the runtime keeps one segment in flight);
   - events: `commit` (a CPU kernel starts the pending GPU work), `sync_to_cpu` / `sync_to_gpu`, `finish`;
   - every Metal dispatch: pipeline function, grid, threadgroup size (MPS work shows as `(MPS)`);
   - memory: every `mem.plan` (total, arena size, alignment, and each buffer's size, live range and offset) and
     every `buffer.alloc`.
4. **IR per pass**: the traced source IR, then the IR after each prefix of the default pipeline (`dce`,
   `bufferize`, `plan-memory`, `hoist-init`; each verified), as `ir/NN-pass.mlir`. A
   loaded `.lkm` only has its final IR.
5. **Machine state**: Low Power Mode, power source, thermal state (the phone's at the start and end).

### GPU timestamps

Default mode (`gpu_mode: stream`): the runtime writes a Metal 4 counter-heap timestamp (`writeTimestamp`, *precise*
granularity) right after the barrier of a kernel's first dispatch and after its last dispatch. The GPU row of a
kernel is `end - begin`: its own execution time, inside the normal command stream, with no extra commits. Timestamps
are converted to the CPU's `CLOCK_UPTIME_RAW` timeline (heap entries tick at `queryTimestampFrequency()` = 24 MHz,
paired with `sampleTimestamps()`), so the timeline shows CPU and GPU lanes on one axis. Measured overhead on Kokoro
GPU: profiled call 126.7 ms vs 125.4 ms unprofiled (+1 %). `LOKINFER_PROFILE_STAMPS=relaxed` lets Metal coalesce
the timestamps (cheaper, coarser).

`--serialize` (`gpu_mode: serialize`, also what `LOKINFER_PROFILE=1` does): commit + wait after every GPU kernel.
Per-kernel GPU times stay exact, but CPU/GPU overlap disappears and the call is slower (Kokoro: 182 vs 128 ms).
Use it only to rule out interference between concurrent dispatches.

Precise timestamps can serialize dispatches that would otherwise overlap (independent kernels without a barrier).
The *sum* of GPU kernel times is then an upper bound of the GPU's busy time; `GPU busy` in the report is the union
of the intervals.

### Zero cost when off

`setProfile` defaults to level 0; the checks are one branch per instruction and one per dispatch. Nothing is
allocated until profiling is switched on.

## Reading a report [built]

`report.html` (self-contained, in each run directory) and the viewer show the same tabs:

- **Summary**: E2E median (p10/p90, n); profiled call and its overhead; **GPU busy** (union of GPU kernel
  intervals) and its share of the e2e time; **CPU kernels** (sum of CPU kernel wall time); **sync / finish wait**
  (the CPU blocked on the GPU at `sync gpu_to_cpu` and at the end); **in-flight wait**; arena sizes; machine state;
  notes (Low Power Mode, throttling, serialize mode, profiler overhead > 10 %).
- **Kernels**: one row per kernel name: calls per iteration, total ms per iteration, % of the e2e median, mean /
  min / max per call, and for GPU kernels the encode and wait times. GPU kernels use their GPU time, CPU kernels their
  wall time; on a mixed graph both run concurrently, so the % column can add up to more than 100.
- **By shape**: the same, split by signature (`f32[1,512,128] f32[...] -> f32[...] {attrs}`).
- **TF/s** (Kernels, By shape, and the CLI summary): achieved arithmetic throughput of GEMM / GEMV / conv kernels
  over their own time, FLOPs = 2 · (output elements / last dim) · (weight elements) (`analysis.flops_of`). Winograd ops
  (`*wino*`) show no TF/s (their weights are the transformed ones). Compare it
  with the device ceiling before optimizing a kernel: f32 steel ≈ 6.3–6.5 TF on the M2 Max, 2.63–2.69 TF on the
  iPhone; Kokoro's convs already reach it on both (2026-11-01), so only fewer FLOPs or fusion can help them.
- **Instances**: one row per kernel *op* in the exec region. The op number links to its line in the final IR.
- **Categories**: coarse buckets (conv, conv_transpose, matmul, lstm, norm, signal, activation, elementwise, layout,
  attention, embedding) shared with prof-torch, so a lokinfer and a torch run can be compared op family by op family.
- **Timeline**: the median profiled call. Lanes: CPU kernels, CPU encode (GPU kernels being encoded), CPU sync
  (sync, finish, in-flight waits), GPU. Wheel to zoom, drag to pan, hover for name, op number, duration, signature.
  Gaps in the GPU lane are idle GPU; a long encode that is really a wait shows in the sync lane.
- **Memory**: each `mem.plan` as a picture (x = live range in op indices, y = offset), with the lower bound (peak of
  simultaneously live bytes) and the planner's efficiency (lower bound / total).
- **Metal dispatches**: per pipeline function, dispatches per call, the kernels using it, example grids.
- **IR per pass**: pick a pass, optionally highlight lines new since the previous pass; the final IR is annotated
  with each kernel's ms/iter.
- **Settings & files**: the raw profile, IR files, settings.

### Methodology

1. Start from the dashboard number (e2e median) and check the run's notes: Low Power Mode off, AC power, thermal
   nominal. A ⚠ run is not comparable.
2. On a GPU model, compare **GPU busy** with **E2E**. If GPU busy ≪ e2e, the GPU is starved: look at the timeline
   for CPU kernels on the critical path, sync waits, in-flight waits and encode gaps. If GPU busy ≈ e2e, the GPU
   kernels themselves are the work: go down the Kernels tab.
3. For the top kernels, check calls × mean against the expected FLOPs / bytes (dashboard peak numbers in
   [benchmarking.md](benchmarking.md)); *By shape* shows which shapes are slow.
4. For a slow GPU kernel: `prof-lokinfer metal stats FUNC` (threads per threadgroup, threadgroup memory),
   `metal air-stats FUNC` (loads / stores per address space, FMAs, barriers), `metal native FUNC` (native code size
   on this GPU), then an Instruments trace with GPU counters for occupancy / ALU / memory bound.
5. For a slow CPU kernel: `prof-lokinfer sample MODEL -d mac-cpu` or `trace -t time` (both devices) for the hot
   functions, `trace -t cpu-counters` (Mac) for the CPU bottleneck breakdown.
6. Compare with PyTorch by category (`prof-torch`, `compare`): a category where lokinfer is slower is the next
   kernel to write or tune.

## prof-torch [built]

`prof-torch MODEL -d cpu|mps` runs the registry model's PyTorch reference with the same inputs (Kokoro:
english-practice's `infer()` math, `bench/models/bench_kokoro.torch_infer`) and writes a run with `kind: torch`:

1. E2E time with the dashboard protocol, no instrumentation.
2. Per-op times with a `TorchDispatchMode` timer (leaf aten ops, after autograd). Each op is attributed to the
   innermost *leaf* `nn.Module` around it (`Conv1d`, `LSTM`, `Linear`, `AdaIN1d`, `LayerNorm`, `SineGen`, ...),
   which sets its category, e.g. `aten::mul @lstm` for the element-wise work of the MPS LSTM. On MPS every op is
   followed by `torch.mps.synchronize()`, so rows are real (serialized) GPU times and their sum exceeds the e2e time.
3. `torch.profiler` (CPU activity, `record_shapes`) over 2 calls: `torch_trace.json` (Chrome trace, open it in
   <https://ui.perfetto.dev>) and `torch_key_averages.txt`, as torch produces them.

`compare` maps lokinfer kernels and torch ops through the categories (the "per-op mapping"); kernels with the same
name only match between two lokinfer runs (e.g. before / after a change, CPU vs GPU, Mac vs iPhone).

## mlogs-viewer [built]

`tools/mlogs-viewer` starts `python -m lokinfer.prof.cli mlogs-viewer --foreground` in the background (log:
`prof-runs/viewer.log`) on `http://127.0.0.1:8766/` (`--port` or `MLOGS_PORT`; localhost only) and opens it. It is
idempotent (reuses a running server); `--stop` stops it. The run list filters by text; tick two runs and *Compare*.
Routes: `/api/runs`, `/api/run/<id>`, `/api/run/<id>/file/<path>`, `/api/compare?a=&b=`, `/runs/<id>/report.html`.
The viewer reads run directories only; it never runs models.

## Tool map: NVIDIA → Apple [built where marked]

| NVIDIA | Apple / lokinfer | status |
|---|---|---|
| `nsys` timeline | `prof-lokinfer run` timeline (CPU + GPU lanes, per kernel); `trace -t metal` (Metal System Trace: command buffers, encoders, GPU intervals, driver, residency, **GPU performance state** = DVFS level) | runtime timeline: built (Mac + iPhone); Metal System Trace: built, tested on the Mac |
| `ncu` per-kernel metrics | `trace -t gpu-counters` (Metal System Trace + the *Metal GPU Counters* instrument): the GPU's 31 counters sampled over time (ALU / F32 / F16 utilization and limiters, buffer load / store, threadgroup memory, last-level cache, compute occupancy, MMU / TLB, read / write bandwidth), summarized per counter (mean, mean while non-zero, max), plus the shader profiler's **per-shader GPU time** (`metal-shader-profiler-intervals`, by pipeline function); `metal stats` | built (Mac); iPhone [not yet run] |
| `ncu` source view / instruction-level hot spots | Instruments' shader profiler (needs the GUI: per-line costs from `-frecord-sources`, which the build embeds); the `gpu-shader-profiler-*` tables are exported when present | GUI only |
| PTX | **AIR** (LLVM IR for Apple GPUs): `metal air FUNC` disassembles the shipped metallib (`metal-objdump -d`), `metal source FUNC --emit-air` recompiles the `.metal` file to textual AIR (`metal -S`) with the build's flags, `metal air-stats` gives the static instruction mix | built |
| SASS | the **native GPU binary**: `metal native FUNC [--iphone]` compiles the function into an `MTLBinaryArchive` on the device and extracts the `applegpu_gNN` slice (`metal-lipo`); it reports the `__compute` section size. Apple ships no disassembler (`metal-objdump` has the `agx` targets registered but "no instruction printer", and `applegpu-nt -S` says "Plugin interface not implemented: AIRNTEmitAssembly"). Third-party reverse-engineered disassemblers exist for some generations (not integrated). | built (size only) |
| `cuobjdump --dump-resource-usage` (registers, shared memory) | `metal stats`: `maxTotalThreadsPerThreadgroup` (drops when a kernel uses many registers), `threadExecutionWidth` (SIMD width, 32), `staticThreadgroupMemoryLength`, pipeline creation time. Register counts are not exposed | built (Mac + iPhone) |
| `nvidia-smi` clocks / throttling | `trace`: `gpu-performance-state-intervals` (time per GPU performance state), `device-thermal-state-intervals`; runs record thermal state + Low Power Mode | built |
| `perf` / VTune hot spots | `sample MODEL` (Mac), `trace -t time` (Time Profiler, Mac + iPhone; top self / inclusive symbols) | built (Mac tested; iPhone recording verified by hand) |
| `perf stat` PMU counters | `trace -t cpu-counters` (CPU Counters template: CPU bottleneck breakdown, e.g. MNIST B=64 CPU: useful 73 %, processing 26 %, delivery 1 %, discarded 1 %; specific PMU events need a custom template passed by path, `-t /path/x.tracetemplate`) | built (Mac); iPhone [unverified] |
| CUDA events / CUPTI | Metal 4 counter heaps (`MTL4CounterHeap`, `writeTimestamp`), used by the runtime profiler | built |
| Nsight Graphics GPU capture | Xcode GPU frame capture (`MTLCaptureManager` → `.gputrace`) | planned (GUI to open; not wired) |

### What is not available on Apple GPUs

- No public ISA disassembler and no register count per kernel (only the indirect `maxTotalThreadsPerThreadgroup`).
- No hardware counters through a public API: `MTLDevice.counterSets` has only `timestamp` (`GPUTimestamp`), sampled
  only at stage boundaries (M2 Max: `supportsCounterSampling` is true for stage boundaries, false for dispatch /
  draw / blit boundaries), and Metal 4 counter heaps only hold timestamps. GPU counters come from Instruments.
- The shader profiler's per-line costs need the Instruments / Xcode GUI.
- `sample(1)` does not exist on iOS; use `trace -t time` against LokiHost.
- AMX (the matrix coprocessor behind Accelerate's BLAS/BNNS) has no counters of its own; it shows up as time inside
  Accelerate frames (`libBLAS`, `APL*`, `BNNS*`) in the Time Profiler, and its share of a CPU kernel can only be
  inferred (compare with the single-stream sgemm ≈ 1 TFLOP/s reference in benchmarking.md).

## xctrace details [built]

- Mac: `trace` starts the workload (`prof-lokinfer _workload`: builds the model, warms up ~1 s, prints `ready`,
  then calls it in an open-ended loop until killed), attaches `xctrace record --attach <pid>` for `--time-limit`
  (default 10 s), then kills the workload. Model build time is not in the trace.
- iPhone: the model is uploaded to LokiHost; a background thread keeps running it through the remote executor while
  `xctrace record --device <udid> --attach <pid>` records (xctrace can't find device processes by name; the pid comes
  from `devicectl device info processes`); the trace is pulled back to the Mac. Status: recording a Time Profiler
  trace attached to LokiHost's pid on the iPhone 18 Pro Max was verified by hand (2026-11-01); the integrated
  `trace -d iphone-*` path has not been run end to end yet (the phone was busy with benches).
- The trace (`*.trace`, open it in Instruments yourself if you want the GUI), `toc.xml`, the exported XML of the
  summarized tables and `trace_summary.txt/.json` go to `prof-runs/traces/<id>/`. Summarized tables:
  `time-profile` (top self / inclusive symbols, filtered to the process), `metal-gpu-intervals` (GPU busy per
  channel, filtered to the process), `gpu-performance-state-intervals`, `device-thermal-state-intervals`,
  `metal-shader-profiler-intervals` (GPU time per shader), `metal-gpu-counter-intervals`, `MetricAggregationForProcess`
  (CPU Counters: useful / processing / delivery / discarded breakdown and cycles), `gpu-counter-value`
  (streamed: ~1.1 GB of XML / 7.6 M samples for 3 s of Kokoro, summarized in ~20 s, then deleted). Exports parse the
  `id`/`ref` de-duplication.
- Cost: traces are large (a 3 s Kokoro GPU-counters trace is 127 MB) and exports take tens of seconds to minutes;
  keep `--time-limit` short (a few seconds of steady state is enough: the workload is already warm).
- Example (Kokoro real117, Mac GPU, `-t gpu-counters`, 3 s): GPU performance state *Maximum* 99.9 % of the time;
  per-shader time dominated by `lk_conv1d_lc_f32` (~90 % of sampled shader time); counters while non-zero: ALU
  limiter 67 %, ALU utilization 63 %, F32 utilization 53 %, buffer read limiter 9 %, threadgroup load limiter 11 %,
  **compute occupancy 22 %**: ALU-bound convs at low occupancy.

## Pitfalls

- **GPU DVFS warm-up.** The GPU clock ramps up under load. A cold GPU runs small graphs several times slower
  (iPhone: a 1-kernel call ~290 µs cold, < 50 µs warm). The runs warm up ≥ 1 s first; a Metal System Trace of a
  tiny model shows the GPU mostly in its *Minimum* performance state (MNIST on the M2 Max: 92 % of the time), which
  is why small-model GPU numbers are dominated by latency, not throughput.
- **iPhone thermal throttling.** Runs record the thermal state at start and end; anything other than `nominal` is
  flagged. Let the phone cool down between heavy runs (see the dashboard's timing protocol), keep it on power, and
  keep Low Power Mode off.
- **Profiler overhead.** Stream mode adds ~1 % on Kokoro; serialize mode removes overlap (call +40 %). Always read
  kernel times next to the unprofiled e2e number, never the profiled call total alone.
- **% of e2e on mixed graphs.** CPU and GPU kernels overlap, so per-kernel percentages can add up to more than 100.
- **Encode time includes waits.** A GPU kernel's CPU time may include waiting for the previous in-flight segment;
  the report separates it (`wait µs`, in-flight wait lane).
- **Torch MPS per-op times are serialized** (synchronize after each op); compare categories, not sums.
- **Concurrent benches.** Never profile on the phone while another bench drives it (the tool checks for
  `bench_remote_models.py`); results would interfere both ways. Serialize every Mac/iPhone timing session with
  `scripts/lk-lock` (`docs/benchmarking.md`).

### Metal kernel pitfalls found with these tools

- **Register arrays indexed by a runtime value spill to memory.** The first register-blocked 3×3 depthwise kernel
  indexed `in[o * stride + k]` with a runtime stride and was *slower* than the scalar kernel; making the stride a
  template parameter (all indices compile-time, loops unrolled) made it 4.4× faster (MobileNetV2 N=8 depthwise
  2.4 → 0.55 ms on the Mac). Same lesson as GPT-2's attention (fully unrolled: 3× faster, `docs/models/gpt2.md`).
- **Tiled kernels starve on small M.** A 32×32-tile GEMM at M = 1 runs on ⌈N/32⌉ threadgroups that each walk all of
  K (MNIST 784→128: 192 µs); split K across a threadgroup instead (`gemm_bias_smallm_f32`). The timeline's GPU lane
  and the dispatch grids in the report show this at a glance.
- **Mac-tuned tile choices don't transfer to the iPhone.** Check both: the steel GEMM that wins at M = 1024 on the
  Mac loses to the plain 32×32 kernel on the phone.

## Formats

### Raw profile (runtime → tools) [built]

`Executable::takeProfileJson()` / `CompiledModel._exe.take_profile()` / `RemoteDevice.take_profile(mid)`:

```json
{"version": 1, "uses_gpu": true, "num_exec_ops": 1001,
 "ops": [[op, "conv1d_lc_f32_gpu", "gpu", num_operands, [arg ranks], [arg dtypes]], ...],
 "calls": [{"call": 0, "total_ns": 126701917, "gpu_mode": "stream", "gpu_timestamps": true,
            "kernels": [[op, cpu_t0, cpu_t1, gpu_t0, gpu_t1, [ints], cpu_wait], ...],
            "events": [[op, "commit" | "sync_to_cpu" | "sync_to_gpu" | "finish", t0, t1], ...],
            "dispatches": [[op, "lk_conv1d_lc_f32", gx, gy, gz, tx, ty, tz, threadgroups], ...],
            "memory": [{"op": 0, "kind": "plan", "total": 1096704, "arena": 1654784, "align": 256,
                        "buffers": [[size, start, end, offset], ...]},
                       {"op": 996, "kind": "alloc", "bytes": 883200, "output": true}]}]}
```

Times are ns from the call start; GPU times are -1 when not measured; `op` is the instruction index in the exec
region, which is also the line `exec_line + 1 + op` of the final IR. Python: `exe.set_profile(level, gpu_serialize)`;
C++: `Executable::setProfile(ProfileOptions{1, false})`. Remote protocol: `PROFILE` (7), `TAKE_PROFILE` (8),
`GPU_INSPECT` (9), see `include/lokinfer/remote/server.h`.

### Run directory [built]

`run.json` (`schema: "lokinfer-prof/1"`): `kind` (`lokinfer` | `torch`), `model`, `case`, `device`, `backend`,
`created`, `machine_state`, `host`, `settings`, `summary`, `kernels`, `kernels_by_shape`, `ops` (lokinfer),
`categories`, `dispatches` / `memory` / `ir` (lokinfer), `timeline`, `notes`, `files`. Next to it:
`raw_profile.json`, `ir/NN-pass.mlir`, `report.html`, and for torch `torch_trace.json`, `torch_key_averages.txt`.
The analysis is in `python/lokinfer/prof/analysis.py` (pure Python, unit-tested in `tests/tools/`).

## Planned

- Init-region profile (weight preparation): today only exec calls are recorded.
- Xcode GPU capture (`.gputrace`) on request, for the shader profiler's per-line view (GUI to open it).
- Per-dispatch GPU timestamps for kernels with several dispatches (today: one interval per kernel).
- A per-kernel roofline column (FLOPs / bytes per kernel from shapes) next to the times.
