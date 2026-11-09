# The runtime

The runtime executes a compiled model. It is simple and explicit: the compiler already decided memory and hoisting,
the Python graph wrote the device boundaries, and the runtime only interprets a flat instruction list.

Code: `include/lokinfer/runtime/` and `src/runtime/` (executable, Metal 4 GPU context, kernel registry, thread pool).
Sections are marked **[built]** or **[planned]**.

## Loading (`Executable::create`) [built]

1. Each `kernel` op is resolved by name in the kernel registry (`findKernel`). This happens once, never per call. The
   registry's device for each kernel must agree with the device its name says (the compiler relies on names).
2. Kernel arguments are precomputed: pointers come from a **slot table** (one slot per IR value), and integer args
   (shape dims, int attrs) are evaluated once when static and recorded for per-call evaluation when symbolic.
3. Kernels with a `prepare` hook create their state here (GPU pipeline states, MPS kernel objects).
4. `init` runs once (including GPU work such as weight transposes). Its results become the exec region's state
   slots, and any symbols it defined persist across calls. Persistent buffers (`lk.persistent_buffer`, e.g.
   GPT-2's KV caches) are allocated here as init state; `inout` kernels write them in place on every call.

## A call (`Executable::run`) [built]

1. Input shapes bind the signature's symbols: a dim linear in one unbound symbol solves for it (`[N, N]`,
   `[600*F, 9]`), then every dim is checked. A dim that can't be bound, or a size of the wrong form, is an error.
2. The instructions run in order:
   - kernels go through one indirect call each. A kernel that detects invalid data-dependent input (an id out of
     range, `frames ≠ Σdur`) sets `ExecCtx::error` (a static string) instead of producing garbage, and the call
     returns that error right after the kernel;
   - `view` is pointer arithmetic (dynamic views are bounds-checked; static ones are checked by the verifier);
   - symbolic dims (kernel int args, view offsets, sizes) go through a per-call memo: each distinct dim is computed
     once per call (`DimEvaluator`);
   - `mem.plan` looks its result up in a per-plan cache keyed by symbol values, else runs the greedy planner on the
     evaluated sizes; the plans of a call share one grow-only arena per pool (CPU-touched / `gpu_only`), grown only
     at the pool's first plan of a call (`docs/compiler.md` "Memory planning at run time");
   - `dim.read` binds a data-dependent symbol, checked against any input-bound value;
   - `%y = sync gpu_to_cpu %x` commits the GPU segment and waits; `cpu_to_gpu` needs nothing (unified memory);
   - a CPU kernel that follows open GPU work first commits it without waiting (`GpuContext::commitAsync`): a CPU
     kernel never reads GPU results without an explicit `sync gpu_to_cpu`, so the two overlap.
3. Outputs are returned as owned tensors, **without copying**: each output is this call's own storage, so results
   stay valid after the next call (zero-offset views of it, e.g. reshapes, too). The only copies are outputs that
   *are* an input, a constant or init state, so callers never alias model memory.
   **Caller-provided outputs** [built 2026-11-02] (`run(inputs, outputs, callerOutputs)`, Python `model(x, out=y)`):
   the output's alloc is bound to the caller's memory (wrapped for the GPU like inputs, cached per output), so the
   model computes straight into it; outputs the model doesn't compute are copied into it.

Overhead (M2 Max, LPM off, 2026-11-02): a one-kernel CPU graph costs ≈ 320 ns per call on top of the direct kernel
call (379 vs 58 ns for a 1 024-float add); a one-kernel GPU graph ≈ 94 µs per call (`bench/runtime_overhead.cpp`,
[benchmarking.md](benchmarking.md) "Runtime overhead").

## Kernel ABI [built]

Every kernel is two layers (`include/lokinfer/runtime/kernel.h`):
- a **typed function** (`kernels::cpu::gemm_bias_f32(a, b, bias, out, m, n, k)`), the thing that's optimized and
  benchmarked directly;
- a **registry adapter** `run(ExecCtx&, const KernelArgs&)`, where `ptrs` = operands then outs, `ints` = every
  operand's shape dims, then every out's shape dims, then int/bool/int-list/dim attrs by attribute name, and
  `floats` = float attrs by attribute name. Structural attrs (`nohoist`, `inplace`, `inout`, `output`, `arena`,
  `device`, `persistent`, `gpu_only`; `isStructuralAttr`) are compiler-only and never passed.

Op signatures and output-type rules live **only in Python** (`python/lokinfer/ops/`). The C++ side maps names to code.
`tests/python/test_registry.py` checks the two agree (every Python op is registered on the device its name says,
every registered kernel has a Python definition).

## Memory (no copies of graph inputs, outputs or weights) [built]

All runtime memory is 16 KiB page-aligned host memory. When a model uses the GPU, memory is wrapped without copying
as a shared-storage `MTLBuffer` (unified memory: CPU and GPU see the same bytes):
- **runtime-allocated storage** comes from a size-class pool (pages up to 1 MiB, then 8 classes per power of two),
  because wrapping memory and changing the residency set are too expensive per call;
- **constants**: every `.lkm` weight (mmapped, 16 KiB-aligned) and every captured tensor are wrapped in place;
- **inputs**: GPU tensors are used as is; any other caller memory that comes with an owner (every Python tensor)
  is wrapped in place, the enclosing page range with an offset, so alignment doesn't matter. The wrap is cached
  per input, so calling again with the same tensor costs nothing. Only ownerless raw memory (C++ callers that pass
  no owner) is copied, because the runtime couldn't keep it mapped. The remote executor (`lokinfer-serve`, LokiHost)
  copies a RUN's inputs once into owned page-aligned memory before its timed calls, so they are wrapped like a
  Python caller's (before 2026-11-02 every timed iPhone GPU call copied them: MNIST B=1024, 3.2 MB per call).
  Its **device tensors** (`ALLOC_TENSOR`; RUN2 input kind 3) are owned page-aligned memory too, bound with no copy
  at all: the view points at the device tensor, with its owner, so a GPU model wraps it once and keeps the wrap
  across calls. A mutable input (`buffer<...> mutable`, e.g. Qwen3-ASR's KV cache) must be one: RUN / RUN2 refuse
  any other input for a mutable arg ("input i is mutable: pass a device tensor") instead of running it on a copy
  whose writes would be lost [built 2026-11-05].

Releasing a GPU buffer while a segment is open is deferred until that segment completes (no commit + wait mid-call).
Buffers only GPU kernels touch are planned in their own arena by the compiler (`gpu_only`, `docs/passes/bufferize.md`),
which keeps Kokoro's GPU arenas small (rand510 7.7 GB → 0.7 GB).

## GPU (Metal 4, `src/runtime/gpu.cpp`) [built]

- There is one process-wide context: device, `MTL4CommandQueue`, one argument table, one residency set holding
  every buffer, and **segment slots** (each a command allocator, a command buffer and a parameter ring buffer:
  Metal 4 has no `setBytes`; 2 on macOS and iOS).
- **Residency commits** [built 2026-11-06]: wrap / release only mark the residency set dirty; it is committed at the
  next encode. An uncommitted removal keeps the allocation alive, so `GpuContext::trim()` (commitAndWait + commit
  the set) must follow big frees outside a run: the remote server calls it when a client disconnects and on
  FREE_TENSOR. Before, a freed Qwen3-ASR model's 3.6 GB stayed allocated until the next GPU call (on the iPhone,
  through a following CPU-only session).
- **Thread safety** [built 2026-11-03]: the context is one shared encoder, guarded by a recursive mutex.
  `Executable::run` of a GPU executable holds it for the whole call; `commitAndWait`, buffer wrap/release and
  pipeline lookup take it themselves. So calls of different models (or `submit()` on a worker thread plus
  `Tensor.numpy()` on the caller's) never interleave their encoding; GPU calls from several threads serialize.
- One compute encoder per GPU segment. A segment ends at a `sync gpu_to_cpu` or at the end of the call (commit
  and wait), or before a CPU kernel (`commitAsync`: commit without waiting, its keep-alive objects and deferred
  releases held until it completes). **Several segments in flight** [built 2026-11-02]: with two slots
  (`LOKINFER_GPU_INFLIGHT`, 1-4; default 2 on macOS and iOS) the next segment is encoded while the previous one
  still runs (while nothing is in flight the last slot is reused, so single-segment calls behave as with one); its
  first dispatch then waits for the queue's earlier work (`barrierAfterQueueStages`, since Metal 4 doesn't order
  command buffers), and one autorelease pool spans all open / in-flight segments (pools are LIFO). Kokoro app59f
  Mac GPU (prof-lokinfer, interleaved): 1 slot 50.72 / 50.69 ms, 2 slots 50.18 / 50.09 ms, 3 slots 50.17 ms. The
  profile shows why the gain is small: with one slot the CPU waited 13.6 ms per call for the in-flight segment
  before encoding the next; with two that wait drops to 1-3 ms but reappears at the next `sync gpu_to_cpu`
  (24.6 → 36 ms): the call is GPU-bound. GPT-2 Mac GPU (prof-lokinfer, interleaved): prefill T=128 5.04 → 4.90 ms,
  decode 2.17 → 2.03 ms. iPhone 18 Pro Max (remote executor, interleaved): Kokoro app59f 65.6 → 64.9 ms, GPT-2
  decode 5.81 → 5.59 ms, but GPT-2 prefill T=128 bimodal at 2 slots (11.3 or 12.9 ms vs a steady 11.56 at 1), so iOS
  keeps one slot (the dashboard goal: Kokoro and GPT-2 must never regress). Waits spin on the shared event for up to
  2 ms before blocking (the blocking wait added ~100 µs of wake-up latency per call).
- **GPU failures** [built 2026-11-02]: every commit carries a Metal 4 commit-feedback handler; a command buffer
  that fails (GPU fault, timeout, out of memory) is recorded, waits stop blocking on the event it would never
  signal, and the call (or the next `sync gpu_to_cpu`, before any CPU kernel reads its results) returns
  `GPU command buffer failed: ...` instead of hanging the process. A timeout (Metal error 1) gets a hint (split the
  work: smaller chunks / fewer rows per call).
  **Each commit gets a fresh `MTL4::CommitOptions`** [fixed 2026-11-05]: a reused options object delivered feedback
  for its first commit only, so every later failure was silent and the call returned zeros as a success.
  **Sentinel** [2026-11-05, user's choice: correct per call, no latency]: the completion event can signal ~50 µs
  before a failed buffer's feedback arrives. The last dispatch of every command buffer, `lk_runtime_sentinel`
  (`kernels/gpu/elementwise.metal`), copies a per-segment marker into a shared 4096-slot buffer. A failed buffer is
  aborted before reaching it, so after the event wait, `commitAndWait` checks the markers: all present, success with
  no wait; one missing, it waits for that buffer's feedback (spin, then a bounded block), so the error belongs to
  the call that caused it. Cost ≈ 0.01 ms per call (GPT-2 decode 2.05 vs 2.04 ms); a per-call feedback wait cost
  +0.05 ms (2.5 %).
- **Barriers**: before each GPU kernel, the executable compares its read and write **byte ranges** with those of
  the dispatches since the last barrier. It inserts a barrier only on a RAW, WAW or WAR conflict. Independent
  dispatches run concurrently.
  The tracker resets with each new segment (command buffer), and the first kernel of a new segment always gets a
  barrier [fixed 2026-11-06]: a commit inside a kernel (work budget, params ring full) puts that kernel's later
  dispatches into the new buffer, where the next kernel would otherwise race them.
- **GPU work budget** [built 2026-11-06, iOS]: heavy kernels report their FLOPs (`GpuContext::addWork`: the steel
  GEMMs, prefill and window attention); once the open command buffer holds more than the budget (iPhone 2e11 FLOPs
  ≈ 0.1 s; none on the Mac; `LOKINFER_GPU_WORK_BUDGET_GFLOP` overrides), it is committed. iOS aborts command buffers
  that run too long (`MTL4CommandQueueErrorDomain` 1): a 1473-row Qwen3-ASR prefill had ~0.6 s of GPU work per
  buffer when only the params ring split them. Decode GEMVs report ~no work, so decode gets no extra commits.
- **Metal 4 command buffers don't retain resources.** Library kernels (MPS) hand their temporary objects to
  `GpuContext::keepAlive`, and the segment holds an autorelease pool until completion. Without this, MPS crashed
  after encoding.
- MPS kernels encode straight into lokinfer's encoder (`encodeWithMTL4CommandEncoder:`). They are ordinary kernels
  such as `gemm_f32_gpu_mps`.
- **Record / replay** [built 2026-11-02, opt-in: `LOKINFER_GPU_REPLAY=1` (`=verbose` logs the decision) or
  `RuntimeOptions::gpuReplay`; `Executable::gpuStrategy()` says what a model uses] (`src/runtime/gpu_replay.*`).
  Eligible: no `dim.read` (every size of a call is a function of its input shapes) and all GPU kernels in one
  segment (no CPU kernel or `gpu_to_cpu` sync between the first and the last). The first call with a given set of
  input shapes runs normally while the context tees every dispatch (pipeline, bound buffers, parameters, grid,
  barrier) into a recorder; the commands are then written once into an indirect command buffer (pipelines rebuilt
  with `supportIndirectCommandBuffers` and made resident, parameters copied into a buffer the replay owns). Later
  calls with the same shapes still interpret the non-GPU instructions (views, allocations, syncs), but a GPU kernel
  only compares its argument buffers with the recorded ones and re-binds the ICB commands whose argument moved
  (inputs, fresh outputs: one `setKernelBuffer` each), adds a barrier if the hazard tracker now wants one the
  recording didn't have, and the last GPU kernel encodes one `executeCommandsInBuffer`. Up to 4 shape signatures
  are kept. MPS kernels (they encode their own work) or a params-ring flush during the recording turn it off for
  that executable. Results are bit-identical to direct encoding (unit tests, incl. the barrier stress graph and
  moving inputs/outputs). **Measured: it loses, so it stays opt-in.** MobileNetV2 GPU N=1 on the Mac
  (prof-lokinfer, the runtime-overhead case): direct 1.117-1.128 ms, replay 1.152 / 1.152 ms (+3 %). As the
  research predicted (`impl-notes/research-metal-runtime.md` §3), direct encoding already overlaps the GPU, and an
  ICB's dependent chain runs slower on the GPU (≈2.6 vs 2.05 µs per dependent dispatch) than the CPU encode it
  saves. iPhone 18 Pro Max (remote executor, 3 interleaved rounds of 500 calls): direct 1.007-1.009 ms, replay
  1.048-1.049 ms (+4 %). It could still pay off where the CPU is the bottleneck (tiny graphs, energy).
  **Root cause, investigated 2026-11-02** (Metal System Trace, per-call command-buffer GPU time, no per-dispatch
  timestamps): the GPU itself runs the replayed call ~5.5 % slower (0.994 vs 0.942 ms; ≈1 µs more per command for
  55 dispatches). Ruled out by A/B: ICB-capable pipelines in the direct path (no change: 1.112 / 1.106 vs 1.116 /
  1.102 ms), `optimizeIndirectCommandBuffer` (1.141 vs replay 1.136 ms), dropping every ICB barrier (no gain, and
  incorrect). So it's the indirect execution's per-command cost, while the CPU work replay can save is only the
  ~55 µs of encoding (the interpreter still walks the graph) — on GPU-bound models they cancel.
- **Early commit** [built 2026-11-02, default `kEarlyCommit = 2`, `LOKINFER_GPU_EARLY_COMMIT=N`, 0 = off]: the
  investigation above showed the real per-call overhead is serial — CPU encodes every dispatch, then commits, then
  the GPU starts ~75 µs later. Committing the first 2 GPU kernels of the call right away lets the GPU start while
  the CPU encodes the rest into the next segment (2 in flight). Mac GPU: MNIST B=1 0.262 → 0.216 ms, GPT-2 decode
  2.10 → 2.04, prefill T=128 5.21 → 4.90, MobileNetV2 N=1 1.106 → 1.088, Kokoro app59f unchanged (49.2 / 49.1). iPhone: GPT-2 prefill T=128 11.44 →
  11.30 ms, MNIST B=1024 0.68 → 0.66, but a 3-kernel graph got slower (MNIST B=1 0.15 → 0.17), so on iOS only graphs
  with ≥ 8 GPU kernels commit early (`kEarlyMinKernels`). (An overlapped GPU→CPU readback of Kokoro's `F` is no longer needed: the duration predictor
  runs on the CPU in both builds.)

## Memory footprint per model [measured 2026-11-02]

`bench/models/measure_memory.py [mac|iphone]`: every (model, graph, case) runs in a **fresh executor process**
(Release `lokinfer-serve` on the Mac; LokiHost on the phone, restarted through SHUTDOWN + the deploy watchdog), with
the model blob uploaded beforehand so the upload buffer isn't counted; three calls; numbers are MiB above the idle
executor, read from the remote MEMORY message. **footprint** = `phys_footprint` (what iOS's per-app memory limit
counts; clean mmapped weight pages are not in it), **resident** = RSS (includes the mmapped weights once touched),
**GPU** = `MTLDevice.currentAllocatedSize` (includes the no-copy wraps of mmapped weights and of runtime memory, so
it overlaps the other two: unified memory).

| model | graph | case | Mac peak footprint | Mac peak resident | Mac GPU | iPhone peak footprint | iPhone peak resident | iPhone GPU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Kokoro-82M | CPU | app59f | 166 | 633 | – | 542 (steady 96) | 932 | – |
| Kokoro-82M | CPU | s28f | 834 | 1301 | – | 815 | 1282 | – |
| Kokoro-82M | GPU | app59f | 134 | 131 | 706 | 138 | 127 | 706 |
| Kokoro-82M | GPU | s28f | 318 | 316 | 1396 | 329 | 317 | 1396 |
| MNIST MLP | CPU | B=1 | 0.6 | 1.5 | – | 0.2 | 1.5 | – |
| MNIST MLP | GPU | B=1 | 7.4 | 7.6 | 3.6 | 9.4 | 3.4 | 3.4 |
| ResNet-18 | CPU | N=1 | 13.5 | 59 | – | 12.1 | 58 | – |
| ResNet-18 | GPU | N=1 | 9.7 | 10.1 | 93 | 18.9 | 5.8 | 93 |
| MobileNetV2 | CPU | N=1 | 13.3 | 28 | – | 11.3 | 27 | – |
| MobileNetV2 | GPU | N=1 | 9.6 | 10.1 | 27 | 15.6 | 5.6 | 27 |
| DCGAN G | CPU | N=1 | 2.1 | 16.3 | – | 1.5 | 16.0 | – |
| DCGAN G | GPU | N=1 | 10.2 | 11.3 | 18.4 | 17.7 | 6.1 | 18.4 |
| DCGAN D | CPU | N=1 | 2.5 | 13.6 | – | 2.1 | 13.5 | – |
| DCGAN D | GPU | N=1 | 7.3 | 7.7 | 13.8 | 11.7 | 3.5 | 13.8 |
| GPT-2 124M | CPU | prefill T=128 | 7.0 | 480 | – | 5.6 | 479 | – |
| GPT-2 124M | GPU | prefill T=128 | 21.5 | 23 | 633 | 22.0 | 11.9 | 633 |

Reading it: weights never count against the footprint (mmapped `.lkm`, wrapped without copies; GPT-2's 480 MiB
resident is its weights). Kokoro's footprint is activations and grows with the frame count: the CPU graph holds
834 / 815 MiB (Mac / iPhone) at s28f, the GPU graph 318 / 329 MiB (its `gpu_only` arena is planned separately and
freed at last use). The iPhone CPU app59f peak (542 MiB) is a transient of the first call (steady 96 MiB; Mac 166 for
both), not seen on the Mac — not investigated further. Everything is far below the phone's limit (LokiHost has the
increased-memory-limit entitlement).

## Memory pressure (remote executor) [built 2026-11-07]

Under system memory pressure iOS reclaims the cheapest memory first: clean file-backed pages, i.e. a CPU model's
memory-mapped weights, which then come back from flash at the next step (Qwen3-ASR's CPU decode: 2-3 s steps, then a
`fc-thrashing` jetsam; docs/models/qwen3-asr.md "Memory under pressure"). The executor makes that visible and
optionally prevents it:

- **MEMORY** (feature `memory_pressure`) adds `available` (`os_proc_available_memory()`: bytes before this app's
  jetsam limit; iOS only, -1 on macOS), `pressure` (the last level of a process-wide `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`
  source: normal / warn / critical; it only fires on transitions) and `pressure_events`, `pageins`
  (`task_events_info`: a weight re-read from flash is a page-in; a warm timed region has none), `weights` /
  `weights_resident` / `weights_locked` (the loaded models' mapped constants, `mincore`), `system_free` /
  `_file_backed` / `_compressor` / `_wired` (`host_statistics64`). `RemoteDevice.memory_check()` = (pressure is
  normal, a one-line summary). The iPhone Qwen bench never records a case that starts or ends under pressure or
  pages in more than the weights; the test harness prints a warning at connect.
- **PIN_MODEL** (feature `pin_model`), per model: `willneed` (madvise read-ahead, no protection), `lock` (mlock: works
  on iOS, but wired pages are charged to `phys_footprint`, i.e. against the app's limit, and on the Mac the footprint
  stays after munlock), `gpu` (no-copy MTLBuffers over the constant pages in a residency set of their own, committed
  + `requestResidency()`: not charged to the footprint on the Mac, but charged like mlock on iOS (Qwen3-ASR: +3.2
  GB); the GPU graphs' weights are held for free only by the queue's own set while GPU work runs), `unlock`. All or nothing; the pins go with FREE_MODEL / disconnect
  (before the mappings). Not on by default in the runtime; the iPhone Qwen bench pins its CPU graphs ("gpu": 0
  page-ins under pressure, no speed cost in an alternating A/B; on iOS the footprint grows by the weights' size).

## CPU kernels and the thread pool [built]

`rt::ThreadPool::global()` (`src/runtime/thread_pool.cpp`): persistent workers at `QOS_CLASS_USER_INTERACTIVE`, sized
to the P-core count, plus the E-cores on chips with < 4 P-cores (iPhone: 6 workers); the caller participates in
`parallelFor`, workers spin 50 µs before sleeping, the hand-off is lock-free, and `park()` puts spinning workers to
sleep before a multithreaded library call. Kernels decide themselves (user decision): small work runs inline on the
calling thread, work above a per-kernel threshold goes to the pool (e.g. the fused convs: 4 M MACs). The executor
itself runs ops one at a time. Variants for benchmarking (read once at pool creation): `LOKINFER_THREADS=N`
(pool size incl. the caller; 1 = mono-thread) and `LOKINFER_POOL_QOS=default|utility|user-initiated|user-interactive`.
Mac comparison (2026-11-02): the pool is 2.5× faster than mono-thread on MobileNetV2 / Kokoro, and user-interactive
QoS 6 % faster than default QoS on MobileNetV2 ([benchmarking.md](benchmarking.md) "CPU runtime variants"; iPhone
pending).

## Mixed CPU/GPU graphs [built]

Allowed, with **device boundaries written explicitly by the Python graph** (user decision 2026-11-01: devices are
explicit in the IR, never inferred). Constants live on their tensor's device; `ops.to_cpu` / `ops.to_gpu` emit
`%y = sync ... %x`. Memory is unified, so nothing is copied. The compiler only verifies residency
(`docs/verifier.md`). A GPU model is normally GPU end to end, with weight preparation hoisted into init.

## Profiling [built]

`Executable::setProfile(ProfileOptions{1, gpuSerialize})` records every following call: per kernel the CPU wall time
and int arguments, for GPU kernels two Metal 4 counter-heap timestamps (`GpuContext::profileStart` /
`profileKernelBegin` / `profileKernelEnd`; the begin stamp goes after the kernel's first barrier) and the time spent
waiting for the in-flight segment, plus syncs, every dispatch (pipeline function, grid) and every `mem.plan` /
`buffer.alloc`. `takeProfileJson()` returns and clears the records. Off by default (one branch per instruction and
per dispatch). `LOKINFER_PROFILE=1|2` still prints the per-kernel table to stderr (serializing GPU kernels).
The remote executor exposes it as `PROFILE` / `TAKE_PROFILE`, and `rt::gpuInspectJson` (pipeline statistics, native
binary archives) as `GPU_INSPECT`. Tools and formats: [profiling.md](profiling.md).

## Untrusted input [built]

The `.lkm` loader and the remote executor (the iPhone test host listens on the network) treat their bytes as
hostile: overflow-safe bounds, validated enums, ids and constant sizes, frame-size caps (and a payload buffer that
grows as bytes arrive, never to the claimed size up front), tensor shape ↔ byte-count checks, Metal function names
restricted to identifiers, and `verifyModel` before any model is executed. Device tensors are capped (4 GiB of
live device tensors, at most 4096, each ≤ the 2 GiB frame cap so READ_TENSOR fits one reply; memory is anonymous
pages, zeroed, committed only when touched), handles are looked up (an unknown or freed handle is an error), and a
WRITE_TENSOR must match the device tensor's dtype and shape exactly with no trailing bytes. The protocol is fuzzed in
the unit suite (`tests/unit/test_remote_fuzz.cpp`, docs/correctness-testing.md).
The executor serves one client at a time; a client that disappears without closing (Mac asleep, USB tunnel reset)
is detected by TCP keepalive (~25 s: 10 s idle, 3 probes 5 s apart) and the next client is accepted, and writes to
a closed client fail instead of raising SIGPIPE (`SO_NOSIGPIPE`) [built 2026-11-03].
