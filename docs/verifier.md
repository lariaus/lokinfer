# IR verifier

**[built]** (everything on this page).

Verification is **not a pass**: like MLIR's verifier it never changes the IR, and it runs automatically
- when a model is loaded or built: Python `lk.compile` / `compile_text`, `lokinfer-opt` after parsing, the iPhone
  remote server before executing anything;
- after **every pass** in `runPasses` (`verifyEach`, on by default), so a broken invariant names the pass that broke
  it: `IR invalid after pass 'bufferize': verify: ...`. `lokinfer-opt --no-verify` turns both off (debugging a
  verifier bug).

API: `verifyModel(model, opts)` in `include/lokinfer/ir/verify.h` (`src/ir/verify.cpp`, `src/ir/verify_devices.cpp`).
It checks, in order:

1. **Structure**: every region ends in `return`; values are defined once, before use, in their region; init has no
   arguments and its results match exec's state; each op kind's operand/result counts and attrs (view bounds when
   static, inline constant sizes, `mem.plan` lists and its `elem` sizes (one per buffer, positive divisors of the alignment), `dim.read` index, `sync` kind); every symbol is bound before use.
2. **Buffer form**: once a model is bufferized (a kernel has `outs`, or a `buffer.free` / `mem.plan` exists) no
   tensor-typed value may remain. The executor also forces this check (`VerifyOptions::requireBuffers`) since it
   only runs buffer-form IR.
3. **Device boundaries** (below).

Earlier versions had `verify`, `verify-devices` and `verify-buffers` as pipeline passes; they were read-only, so they
became this always-on verifier (2026-11-01).

## Device boundaries

Device boundaries are **explicit in the IR, written by the Python graph, never inferred by the compiler**
(user decision 2026-11-01). The verifier only checks. It replaced the old `insert-sync` inference pass, which had
real bugs (alias-root granularity, a missed CPU-after-GPU write-after-read).

### The model

Memory is unified, so a device boundary moves no bytes; it's a **sync point**:
```
%ag = sync cpu_to_gpu %a : tensor<f32[1024]>     // ops.to_gpu(a): GPU kernels may use %ag
%oc = sync gpu_to_cpu %o : tensor<f32[1024]>     // ops.to_cpu(o): waits for the GPU work that made %o
```
The result is a zero-cost alias of the operand, resident on the target device. At runtime `gpu_to_cpu` commits
the GPU segment and waits; `cpu_to_gpu` needs no wait (later GPU work is encoded after the CPU wrote).

**Residency**:
- **Constants** are resident on their `device` attr (`"cpu"` / `"gpu"`), which Python sets from the captured
  tensor's device (`lk.from_numpy(w, device="gpu")`). A constant without the attr (hand-written IR) is usable by both.
- **Persistent buffers** (`buffer.alloc {persistent}`, e.g. a KV cache) are resident on their `device` attr, like
  constants; an `inout` operand a kernel writes in place becomes resident on that kernel's device.
- **Graph inputs and init state** are usable by both devices. (Exception: persistent buffers become init state
  after hoist-init and are written per call by `inout` kernels; their residency is checked when the model is built,
  before the pipeline, not on the hoisted form.)
- **Kernel results** are resident on the kernel's device. A kernel runs on the GPU iff a `_`-separated token of
  its name is `gpu`; the runtime checks at load that the registry agrees.
- **Views** inherit their source's residency; a **sync** result is resident on its target.

**Errors**: a kernel reading a value resident on the other device (`add ops.to_cpu(...)` in the message), and a
`dim.read` (a CPU read) of a GPU-resident value.

### Lifetimes across a boundary (bufferize)

A buffer a GPU kernel touched may still be read by that (uncommitted or in-flight) dispatch until the next
`sync gpu_to_cpu`, so if a CPU op also touches it, bufferize extends its free to that sync (or to the end of the
region). The memory planner therefore can't give its bytes to a later CPU kernel while the GPU may still read them.
A buffer no CPU op touches is marked `gpu_only` and freed at its real last use: GPU-to-GPU reuse is ordered by the
queue and the runtime's byte-range barriers, and plan-memory gives `gpu_only` buffers their own arena, never shared
with CPU-touched ones.

### Typical use

A GPU model is GPU end to end: weights on the GPU, weight preparation (e.g. `transpose_f32_gpu` on a constant)
hoisted into init, no syncs per call. Mixed models (e.g. a CPU LSTM between GPU convs) write their boundaries.

Tests: `tests/lit/verify/` (accepted explicit graph, rejected implicit crossing, rejected `dim.read` of GPU
data, write-after-read lifetime, rejected mixed tensor/buffer form), `tests/unit/test_runtime.cpp` (mixed graph executed end to end).
