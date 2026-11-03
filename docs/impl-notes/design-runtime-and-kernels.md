# Design: Kernel ABI, Registry & Runtime

Status: **v1.1** (2026-11-01). Sections are marked **[built]** or **[planned]**. **Built:** kernel ABI + registry
(structural attrs kept out of the int ABI, registry ↔ Python check), executable/slot table, CPU interpreter, thread
pool (P-cores at USER_INTERACTIVE QoS, per-kernel work threshold), Metal 4 direct encode with byte-range barriers,
async commit before CPU kernels (`GpuContext::commitAsync`; two segments in flight by default, 2026-11-02), GPU
storage pool, no-copy GPU constants and inputs, MPS on our encoder, kernel error channel (`ExecCtx::error`),
persistent buffers + `inout` kernels, per-kernel profiler (`LOKINFER_PROFILE`, `Executable::setProfile`), remote
executor (+ RUN2: content-addressed / device-generated test inputs, on-device comparisons), ICB record/replay
(opt-in: measured slower than direct encode on MobileNetV2, docs/runtime.md), on-device reference sweeps through
the remote executor (`tests/python/test_device_sweeps.py`).
**Planned:** the mono-thread vs pool vs P-core runtime comparison, dim programs, `submit()/wait()`. Reviewed 2026-11-01 (`plan-review-2026-11-01.md`).

## 1. Two layers per kernel [built]

1. **Typed kernel function**, the thing we optimize and benchmark directly **[user: benchmark through the
   kernel library, no runtime]**:
   ```cpp
   // CPU
   void add_f32(const float* a, const float* b, float* out, int64_t n);
   // GPU: encodes one (or a few) dispatches into an encoder the caller owns
   void add_f32(gpu::Encoder& enc, const AddF32Pipeline& p, BufRef a, BufRef b, BufRef out, int64_t n);
   ```
   `BufRef = {MTL::Buffer*, uint64_t offset}`. No allocation, no locking, no virtual calls, no exceptions.
2. **Registry adapter**: a generic entry the runtime calls through one function pointer:
   ```cpp
   struct KernelArgs {
     void* const*   ptrs;    // CPU-visible pointers (unified memory: valid for GPU storage too)
     const BufRef*  bufs;    // GPU kernels: buffer + offset
     const int64_t* ints;    // operand dims, out dims, then int attrs in attribute-name order
     const double*  floats;  // float attrs in attribute-name order
     const void*    state;   // per-op prepared state (pipeline, packed params), from prepare()
     uint32_t numPtrs, numInts;
   };
   struct KernelEntry {
     const char* name; Device device;
     Status (*prepare)(PrepareCtx&, const Op&, void** state);  // at load: PSO, function constants (may be null)
     void   (*run)(ExecCtx&, const KernelArgs&);               // per call
     void   (*destroy)(void* state);                           // may be null
   };
   registerKernel({"add_f32_cpu", Device::Cpu, nullptr, runBinary<kernels::cpu::add_f32>, nullptr});
   ```
   (`include/lokinfer/runtime/kernel.h`; registration is explicit in `registerBuiltinKernels()`, not static
   constructors.) `ExecCtx` carries the GPU context and an `error` field: a kernel that detects invalid
   data-dependent input sets it (static string) and the executable returns it as an error right after the kernel.
   The overhead is one indirect call plus a few loads. Op signatures, attr meaning and output-type rules live in
   **Python only** **[user]**. The C++ registry only knows name → (device, prepare, run). Mismatches are
   caught by the per-kernel correctness tests, which run every kernel through a graph anyway.
   **Plan review P9, fixed in M0 [built]:** structural attrs (`nohoist`, `inplace`, `inout`, `output`, `arena`,
   `device`, `persistent`, `gpu_only`; `isStructuralAttr`) are never passed, so they can't shift the int layout, and
   `tests/python/test_registry.py` checks every Python op is registered on its name's device and every registered
   kernel has a Python def. Adapters still read the int layout by hard-coded indices.

**Naming**: `<op>[_<fusion>]*_<dtype(s)>_<device>[_<variant>]`, e.g. `gemm_bias_gelu_f16_gpu`,
`gemm_f32_cpu_accelerate`, `lstm_bidir_f32_cpu`. Library-backed variants (Accelerate, MPS, MPSGraph,
BNNS) are ordinary kernels with a suffix, so both versions can be benchmarked and the faster one picked **[user]**.

## 2. Kernel library layout [built; planned: `kernels/ref/`]

```
kernels/
  cpu/      *.cpp, NEON intrinsics, SME2 (sme2_conv.cpp); Accelerate-backed variants
  gpu/      *.metal (compiled into one metallib per platform) + host-side *.cpp / mps.mm (pipelines, dispatch)
  ref/      [planned] dead-simple C++ reference kernels (on-device correctness oracle on iPhone)
src/runtime/registry.cpp   the registry
```
Multithreaded CPU kernels use the process-wide pool `rt::ThreadPool::global()` (not passed through `ExecCtx`)
**[user: only when it measurably helps]**: each kernel runs inline below its own work threshold and on the pool
above it.

## 3. Executable (what the loader builds from the final IR) [built in a simpler form; planned: dim program]

Built: `Executable::Impl` (`src/runtime/executable.cpp`) holds one flat instruction list per region, a **slot
table** (one slot per IR value: pointer, `MTLBuffer` + offset, scalar) and, per instruction, the static ints plus a
list of symbolic ints re-evaluated per call (dims are evaluated recursively; there is no dim program yet). The
sketch below is the original design:

```cpp
struct Instr {
  void (*run)(ExecCtx&, const KernelArgs&);   // or a builtin opcode for sync/dim.read/mem.plan
  KernelArgs args;                            // points into per-model arrays below
  Device device;
};
struct Executable {
  std::vector<Instr> init, exec;
  DimProgram dims;          // symbol slots -> all sizes/offsets
  std::vector<void*> ptrTable;   // buffer slot -> current pointer (arena base + offset)
  std::vector<int64_t> intTable; // evaluated dims/attrs, referenced by Instr.args.ints
};
```
- **Static model**: ptr/int tables are fully filled at load, and `exec` is literally
  `for (auto& i : instrs) i.run(ctx, i.args);`.
- **Dynamic model**: per call → bind input symbols → run dim program → run `mem.plan` (fills ptrTable)
  → run instrs. `dim.read` binds a new symbol and re-runs the part of the dim program that depends on it.
- Inputs/outputs are slots in `ptrTable` overwritten per call (zero-copy from caller tensors).

## 4. CPU runtime **[user: try three, compare]** [built: mono-thread interpreter + the P-core thread pool; planned: the comparison]

**Correction (plan review P2):** for Kokoro, "mono-thread unless proven otherwise" is the wrong default. The
measured CPU lever is several concurrent single-threaded Accelerate GEMM streams (one per AMX unit) over time
tiles, plus NEON prologue/epilogue work on the remaining cores, so **kernels multithread through a thread pool**
from the start of the fused work. The mono-thread interpreter stays as the baseline for the comparison the
original plan asks for. Graph-level parallelism (independent branches) is secondary.

1. **Mono-thread**: the loop above.
2. **Thread pool** **[built]** (`src/runtime/thread_pool.cpp`): persistent workers (spin 50 µs, then sleep on a
   condition variable). Kernels use `parallelFor`. The job hand-off is lock-free: the caller marks the generation odd,
   waits until no worker is registered, publishes the job and makes the generation even; a worker that sees a new
   even generation registers (`active`) and re-checks it before reading the job (Dekker-style, so it never runs a
   stale job), and the mutex is only taken to sleep / wake sleepers. 2026-11-01: 8 trivial tasks on the M2 Max cost
   12 → 4-6 µs per `parallelFor` (the old hand-off took the mutex in every worker and always notified). Independent
   ops are *not* run concurrently in v1 (graph-level parallelism adds sync cost and Kokoro is mostly a chain).
3. **P-core pool** **[built, as the pool above]**: `QOS_CLASS_USER_INTERACTIVE` threads sized to the P-core count,
   plus the E-cores on chips with < 4 P-cores (iPhone 2 P + 4 E: 6 workers; Mac 8 P + 4 E: 8). macOS/iOS have no
   hard affinity API. A separate non-pinned pool variant for the comparison is [planned] (todo M6).
Default (user decision, todo M0): the interpreter is single-threaded; each kernel runs inline below a per-kernel work
estimate threshold (e.g. 4 M MACs for the fused convs) and on the pool above it.

## 5. GPU runtime **[user: direct encode vs record/replay, compare]** [built: direct encode; planned: record/replay]

Updated from `research-metal-runtime.md` (M2 Max measurements, taken in Low Power Mode, so absolute
numbers are pessimistic):
- **Metal 4 API only** (`MTL4CommandQueue/CommandBuffer/ComputeCommandEncoder/ArgumentTable`, residency
  sets). It's supported on the M2 Max (Apple8) and the iPhone. MPS/MPSGraph can encode on Metal 4 queues in the 27 SDKs.
  A dependent dispatch costs ≈2.05 µs on Metal 4 vs 3–3.9 µs on Metal 3. Independent dispatches cost ≈0.19 µs, and CPU encode
  ≈0.13–0.47 µs per dispatch. **Barriers between dependent kernels dominate, so fusion beats any runtime trick.**
- **Loader-built flat dispatch list** [built, with runtime barriers]: pipeline states are built at load (`prepare`);
  barriers are placed at encode time by comparing each dispatch's read/write byte ranges with those since the
  last barrier (RAW/WAW/WAR only; `inout` operands count as writes); untracked hazards; one compute encoder per GPU
  segment.
- **Direct encoding** is the default [built]. A segment is committed without waiting before a CPU kernel
  (`GpuContext::commitAsync`, one segment in flight) so CPU and GPU overlap; chunked commits inside a pure-GPU
  stretch are [planned].
  **ICB replay** is an optional per-model strategy: it costs ≈15–50 µs CPU per run regardless of size, but dependent
  chains run slower on the GPU (≈2.6 µs/dispatch). A Metal 4 command buffer cannot be committed twice.
- **Avoid mid-graph CPU waits**: a CPU↔GPU round trip is ≈120 µs, a warm segment wait ≈0.45–0.8 ms, and a GPU
  that has idled for 3 ms needs ≈440 µs to wake. Waits spin on the shared event for up to 2 ms before blocking
  [built]. **Data-dependent dims (superseded, todo M0):** Kokoro's duration predictor runs on the CPU in both
  builds, so `F` is a `dim.read` of a CPU tensor and never read back from the GPU; CPU/GPU overlap comes from async
  commit. GPU-driven dispatch with upper-bound allocation (229 µs vs 379 µs in the microbenchmark) is not viable for
  Kokoro (F ≤ 25 600 ⇒ 1.6 GB activations) and is dropped.
- **Mixed CPU/GPU** (e.g. LSTMs on the CPU): device boundaries are **explicit in the IR, written by the
  Python graph** (user decision 2026-11-01; `ops.to_cpu`/`ops.to_gpu` → the `sync` op); the compiler only
  verifies residency; nothing is copied (unified memory) [built]. Each `gpu_to_cpu` crossing still costs a
  commit + wait; LSTM placement was decided by measurement (M2): LSTMs on the CPU in both builds.
- **Memory** [built]: shared storage everywhere (same GPU bandwidth as private), one residency set holding every
  buffer. Weights are wrapped `newBufferWithBytesNoCopy` over 16 KiB-aligned mmapped `.lkm` data (≈60–90 µs for
  256 MB, full bandwidth, works on iOS), and captured tensors are wrapped in place too. Runtime storage comes from a
  size-class pool (pages up to 1 MiB, then 8 classes per power of two, ≤ 12.5 % waste). Buffers no CPU op touches
  (`gpu_only`) get their own arena (Kokoro GPU rand510 7.7 → 0.7 GB).
- **Pipelines** [built]: built at load from the offline metallib (runtime source compile is ≈0.5 s).
- **GEMM/conv families**: [built] `simdgroup_matrix` steel-style kernels (vendored MLX steel + lokinfer variants),
  Winograd convs and MPS, in f32 on both devices. [planned / experiment] `matmul2d` (Metal 4 tensor ops) on the
  iPhone's GPU neural accelerators: the f16 plan is superseded by the strict-f32 rule; only an f32 tensor-op
  experiment remains (todo M2/M5).
- **LSTMs**: on the CPU in both builds (decided by measurement, todo M2: a GPU per-step dispatch alone costs more
  than the CPU's ≈8–12 µs/step, and a persistent GPU kernel can't sync threadgroups per step).
- **Timing**: per-kernel GPU timestamps cost ≈10 µs. Kernel benchmarks repeat N dispatches in one command
  buffer and divide. The per-kernel profiler [built] uses Metal 4 counter-heap timestamps
  (`Executable::setProfile`, `prof-lokinfer`) or `LOKINFER_PROFILE=1|2` (serialized GPU kernels, table on stderr).

## 6. Python ↔ runtime [built]

- `CompiledModel.__call__(*tensors)`: the nanobind entry gets DLPack/buffer pointers. For GPU models with
  CPU-memory inputs that have an owner (every Python tensor), the enclosing page range is wrapped
  `newBufferWithBytesNoCopy` with an offset, cached per input; only ownerless raw memory (C++ callers) is copied.
  The model runs and outputs are wrapped as `lokinfer.Tensor` (zero-copy: each output is the call's own storage;
  outputs that are an input, a constant or init state are copied).
- The GIL is released during `run`.

## 7. Robustness (plan review, 2026-11-01) [built: loader, parsers, server checks; planned: remote-protocol fuzz tests]

The `.lkm` loader and the remote executor take untrusted bytes (the phone listens on the network). The loader
must bounds-check every offset/size with overflow-safe arithmetic, validate every enum, every value/symbol id
and every constant size against its type, and the server must cap frame sizes, validate tensor shapes against
`nbytes`, and run `verifyModel` before `Executable::create`. The text/dim parsers must be iterative or
depth-limited and reject integer overflow. `Executable::create` must never trust unverified IR. (Done in M0.)

## 8. iOS [built]

Same C++ runtime and kernels, linked into the test host app. It loads `.lkm` files plus the metallib from
the app bundle/data container. There is no compiler-side Python, but the C++ compiler passes are available
(they're needed anyway for `.lkm` → executable lowering).
