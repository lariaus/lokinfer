# Design: Compiler Pipeline & Passes

Status: **v1.1** (2026-11-01). Sections are marked **[built]** or **[planned]**. Built: dce, bufferize, plan-memory,
hoist-init (see `docs/passes/`), `lokinfer-opt --stats`; 2026-11-02: the < 1 ms pipeline (0.92 ms incl. per-pass
verification on the 3,800-op synthetic graph, Release M2 Max), the `mem.plan` cache keyed by symbol values, one arena
shared by the plans of a call (per pool), element-unit `mem.plan` offsets, memoized dim evaluation (instead of dim
programs, measured not worth it), `internalize` (a C++ function, not a pass: it needs the data), `bake-init`
(`rt::bakeInit`, in the runtime library: it runs init). Update 2026-11-01: verify / verify-devices / verify-buffers
are no longer passes but the always-on verifier (`docs/verifier.md`), run at load and after every pass; `insert-sync`
is gone (device boundaries are explicit in the IR).
**Planned, not built:** `init_only` constant marking, the alloc/free (MTLHeap) baseline comparison of the planner.
(`record-replay` is built in the runtime, opt-in: `design-decisions.md`.) Dropped: `gpu-driven-dims` (Kokoro's
`F` is computed on the CPU, so there is no readback to avoid).

## Pipeline [built]

```
dce → bufferize → plan-memory → hoist-init → (lower to executable)
```
The verifier runs on the input and after every pass (`runPasses(..., verifyEach)`); it is not a pass.
Export utilities (not pipeline passes, they need data / execution): `internalize` and `bake-init` [built,
`docs/compiler.md` "Export"].
Every pass is a `Status run(Model&)` that is independently runnable from `lokinfer-opt --pass=<name>`
and has its own FileCheck lit tests. **Target: whole pipeline < 1 ms for Kokoro.** Every pass is a single
(or a constant number of) linear walk(s) over the ops, and the planner is n·log n.

## verify (structural only) [built, as the always-on verifier: `docs/verifier.md`]

SSA definedness, operand/result counts per op kind, type equality on `view`/`return`, symbols defined
before use (input signature, `dim.read`, `mem.plan`), and no tensor types left after bufferization (verify
can run in a post-bufferize mode). **No shape inference** **[user]**.

## bufferize [built; planned: `model(x, out=y)`]

Tensor → buffer, destination-passing style.
- Graph inputs/outputs become buffers. Each kernel result gets `buffer.alloc` before the kernel and
  becomes an `outs()` operand.
- **In-place**: if a kernel declares `inplace = [out_k, in_j, ...]` and `in_j` is dead after this op (an alloc root
  of this region, not a graph input, constant, persistent buffer or view, and no other operand aliases it), the
  output reuses the input buffer and takes over its lifetime (in-place chains keep reusing it).
- `view` becomes a buffer alias of its source (no alloc). Alias sets are tracked for liveness.
- `const.*` become buffers that are never freed (owned by the model).
- **Frees**: after computing last uses (one backward walk), insert `buffer.free` right after the last use
  of each alias set's root alloc. A root touched by both GPU and CPU ops is freed no earlier than the next
  `sync gpu_to_cpu` after its last GPU use; a root no CPU op touches is marked `{gpu_only}` and freed at its last
  use (plan-memory gives it a separate arena). `buffer.alloc {persistent}` is never freed.
- **Outputs**: a returned buffer can't live in the reusable arena, or it would be overwritten by the next call.
  Bufferization marks returned allocs `{output}`. The runtime gives them their own allocation per call [built], or
  writes into a caller-provided buffer (`model(x, out=y)`, zero-copy into a torch/numpy tensor) [planned].

Example (FileCheck test sketch):
```
// CHECK-LABEL: exec(%x: buffer<f32[N]>, %y: buffer<f32[N]>)
// CHECK: %[[O:.*]] = buffer.alloc {output} : buffer<f32[N]>
// CHECK: kernel "add_f32_cpu"(%x, %y) outs(%[[O]])
// CHECK: return %[[O]]
```

## insert-sync → verify-devices [built as verification; the inference pass is removed]

**Decision 2026-11-01 (user):** device boundaries are explicit in the IR, written by the Python graph
(`ops.to_cpu` / `ops.to_gpu` → the existing `sync` op), never inferred. With unified memory nothing is copied;
a boundary is a sync point. The pass became the device check of the always-on verifier (`docs/verifier.md`,
`src/ir/verify_devices.cpp`): walk the graph tracking each value's residency (constant / persistent buffer `device`
attr, produced or written in place by a CPU or GPU kernel, an input, crossed a sync) and reject any kernel whose
operand is resident on the other device, and any `dim.read` of a GPU-resident value. CPU writes over pending GPU
reads are prevented by bufferize instead (CPU-touched buffers live until the next `sync gpu_to_cpu`). The runtime
executes `sync gpu_to_cpu` as commit + wait; `sync cpu_to_gpu` needs nothing (later GPU work is encoded after the
CPU wrote). The old inference pass had real bugs (alias-root granularity, missed CPU-after-GPU
write-after-read); verification keeps the detection and drops the guessing.

*Historical (the removed inference pass):* device ordering for mixed graphs **[user: mixed CPU/GPU allowed]**. Kernel device comes from the naming
convention: a kernel runs on the GPU iff one of its `_`-separated name tokens is `gpu`. Walk ops tracking which buffers
have pending GPU writes / CPU writes. Before a CPU kernel or `dim.read` that reads a buffer with a pending
GPU write, insert `sync gpu_to_cpu`, which the runtime turns into commit + wait. CPU→GPU needs no wait
(CPU code runs synchronously before later GPU encoding), but a `sync cpu_to_gpu` marker is kept
to split command-buffer segments. GPU→GPU hazards are left to the runtime (barriers between
dependent dispatches inside one encoder; see `design-runtime-and-kernels.md`).
Alternative for `dim.read` on GPU **[user: try both]** [dropped]: GPU-driven variants (indirect dispatch, upper-bound
allocation) as an optional pass `gpu-driven-dims`. For Kokoro this is not viable (F ≤ 25 600 ⇒ 1.6 GB
activations, `research-kokoro.md` §3.1), and no readback is needed anyway: the duration predictor runs on the CPU
in both builds, so `F` is a `dim.read` of a CPU tensor.

## plan-memory [built: 1–4; planned: plan cache, shared arena, 5's lower bound + tests, MTLHeap comparison]

Goal: one big arena + offsets instead of many allocs **[user]**.

1. **Liveness**: each alloc (alias-set root) gets an interval `[def_index, last_use_index]` from the
   bufferization frees. Inputs, constants and outputs are excluded.
2. **Static sizes** (all dims constant): the **greedy-by-size** algorithm (TFLite's arena planner,
   near-optimal in practice). Sort buffers by size descending, and place each at the lowest offset
   that doesn't overlap any already-placed buffer with an intersecting live interval. All offsets
   are 256-byte aligned (good for SIMD and Metal). Emit
   `%arena = buffer.alloc {arena = true} : buffer<u8[<total>]>` and `%b = view %arena[K]` (offset in result
   elements). The arena alloc is static, so
   **hoist-init moves it to init**, and exec does zero allocation.
3. **Symbolic sizes** (depend on input symbols): offsets can't be compile-time constants. Emit
   ```
   %arena = mem.plan sizes[s0, s1, ...] live[a0, b0, ...] align 256 -> M0_size, M0_o0, ... : buffer<u8[M0_size]>
   %b0 = view %arena[M0_o0] ...        // offsets in elements of each buffer's dtype: attr elem = [4, ...]
   ```
   `mem.plan` is executed by the runtime **at call time**. It evaluates the size dims (memoized per call),
   then runs the same greedy placement (O(n log n); n ≈ a few hundred buffers ≈ a few µs).
   The arena is a model-owned buffer that **grows only** (realloc to the high-water mark, at a call boundary).
   [built] A cache keyed by symbol values (8 results per plan; the key also covers the earlier plans sharing the
   arena) skips size evaluation and planning when shapes repeat.
4. **Data-dependent sizes** (depend on a `dim.read` symbol, e.g. Kokoro's frame count `F`): a second
   `mem.plan` placed **right after the `dim.read`** plans those buffers.
   [built 2026-11-02] The plans of a call share **one arena per pool** (CPU-touched / `gpu_only`): a later plan
   places its buffers around the earlier plans' buffers still live (fixed obstacles in the same greedy placement),
   instead of the "base = high-water mark" first sketched (which wastes the space of the dead first-plan buffers).
   The arena is only reallocated at the pool's first plan of a call, when none of its buffers is live; a later plan
   that doesn't fit uses a temporary overflow storage for that call and grows the arena for the next call.
   `live[...]` intervals are abstract op positions. Only their relative order matters, so later passes
   removing ops don't invalidate them.
5. Planner statistics are printed by `lokinfer-opt --stats` [built: static arena bytes vs. the max-live lower bound
   vs. the bytes without reuse, `mem.plan` buffer counts; checked in `tests/lit/plan_memory/stats.lkir`].
6. Within each plan, `gpu_only` buffers get their own arena, separate from CPU-touched ones [built].

The user's own words were "no idea if it's the right way" for memory planning, so this gets benchmarked
against the plain alloc/free baseline (MTLHeap sub-allocation) before it is locked in [planned].

## hoist-init [built; planned: `init_only` constants, record-replay]

Moves everything that doesn't depend on exec inputs into `init` **[user: as aggressively as legal]**.

**Implemented semantics** (src/compiler/analysis.cpp). There are two distinct notions:
- *memory-hoistable* values: constants, static non-output allocs (including the static arena), and
  constant-offset views of those. Their **memory** can be set up once.
- *constant contents*: a constant, or a buffer whose writers are all hoisted kernels.

A kernel is hoisted iff it is not `{nohoist = true}`, all its operands have constant contents, and all
its outs are memory-hoistable buffers that no per-call kernel also writes (otherwise the init contents
would be clobbered after the first call), and its results and `dim` attrs are static (no per-call symbols). This is computed as an optimistic fixpoint with in-sweep
propagation, which is linear in practice. plan-memory excludes constant-content buffers from the arena,
so hoisting them never conflicts with arena reuse.
- Forward walk: an op is hoistable if every operand is hoistable/const/init-produced and it has no
  per-call side effects. Hoistable ops are: kernels on constants (weight cast, packing, layout transforms,
  weight-norm folding), static allocs (incl. the static arena and persistent buffers), and constant views with
  static result types.
- `dim.read`, `sync`, and ops depending on inputs or symbols bound per call are never hoisted.
  `mem.plan` with symbolic sizes stays in exec **[user]**.
- Values defined in init and used in exec become **state slots** (init results ⇒ exec `state(...)` args).
- Buffers used only inside init (temporaries of weight transforms) are freed at the end of init.
- [planned] Constants consumed **only** by hoisted kernels are dropped from exec and marked `init_only`.
  [built] `bake-init` (export) evaluates init and stores its computed results as constants; the init work and the
  original weights only it read are removed, so the file stores the transformed weights instead.
- [planned] **GPU record/replay** **[user: record is part of init]**: a later pass `record-replay` wraps maximal
  fully-static GPU segments of exec in `gpu.replay @segN`. Their indirect command buffers are encoded in
  init and executed per call. Built after the basic runtime exists, kept only if benchmarks show a gain.

## Lowering to the executable [built; planned: dim programs]

Not an IR pass. The runtime "loader" turns the final IR into a flat instruction array (see
`design-runtime-and-kernels.md`): resolves kernel names to function pointers, precomputes static argument blocks
(symbolic ints are recorded and evaluated per call), compiles dim programs [planned] and creates Metal pipeline
states. This happens once per model load.
