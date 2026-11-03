# Plan review (2026-11-01)

A critical review of the lokinfer plan as it stands after day one: `original-plan.txt`, `design-decisions.md`,
the design docs, the research docs, `kokoro-port-plan.md` and `todo.md`, checked against what was actually built.
Nothing here is implemented yet. The point is to fix the plan before more code is written on top of it.

Verdict in one line: **the scaffolding (IR, compiler, runtime, Python, iOS executor) is sound, but the plan has
not yet de-risked the one thing the goal depends on, the generator conv chain, and it carries several design
gaps that would only show up late.**

---

## 1. What the plan gets right (keep)

- Tiny IR, no compiler fusion, hand-fused kernels, Python-only op definitions. This is the right shape for a
  one-model-at-a-time performance project, and it already works end to end.
- Testing through graphs with references from PyTorch *and* from other kernels in the project. The 25 basic
  Kokoro kernels are not throwaway: they are the **reference oracle** for every fused kernel and the on-device
  reference on the iPhone. The plan should say this explicitly, and they must never be "optimized" (keep them
  simple and obviously correct).
- The research is good where it counts: the Kokoro profile (92 % of MACs in generator convs, 75 % of CPU time)
  and the fusion proposal (`research-kokoro.md` §6) are the real plan. Everything else is support.
- Data-dependent `F` in the IR with one readback, the gather-instead-of-one-hot alignment, pushing the gather
  through linear ops, weight folds at load.

---

## 2. Strategic problems, ranked

### P1. The 90 % is not de-risked
Every number in the targets (CPU ≈ 1.4 s, GPU ≈ 0.4 s at 510 tokens) rests on one assumption: that a fused
implicit-GEMM conv1d reaches ~1–2 TFLOP/s on the CPU and ~4–5 TFLOP/s on the GPU **at Kokoro's exact shapes**
(C=128/256, K=3/7/11, dilation 1/3/5, L up to 135 k). Nobody has built or measured that kernel. The plan instead
schedules it as step 8–9 of 11, after the whole basic graph. If the assumption is off by 2×, the CPU goal
(beat 3.09 s) is still reachable but the GPU goal (beat MLX at 260 ms / torch.compile at 1.16 s) is not.

**Change:** add a milestone *before* fusion work: a standalone stage-1 conv benchmark (through the kernel
library, as the original plan asks) with explicit pass/kill thresholds (§4, M2). It also decides layout
(`[C, L]` vs time-major `[L, C]`, which kokoro.cpp found better) and the CPU algorithm (shifted GEMMs vs im2col),
which `research-kokoro.md` §6.4 says to "benchmark both" but the port plan never schedules.

### P2. The CPU plan ignores its own best lever
`research-kernels.md` measured Accelerate `sgemm` at **1.0–1.1 TFLOPS single-threaded, 2.0 with Accelerate's
own threading, and 3.4 TFLOPS aggregate from three concurrent single-threaded calls** (three AMX units). PyTorch
uses the default threading, so it is capped near 2.0. The plan mentions "prefer ≤ 2 concurrent GEMM streams;
measure" in passing. This is the single biggest CPU idea and should be the design center: **time-tile every
frame-rate conv into independent chunks, run one single-threaded `sgemm` stream per AMX unit, do the NEON
prologue/epilogue on the remaining cores.** That could move the CPU floor from ~1.4 s toward ~0.8 s. It must be
re-measured out of Low Power Mode first (all research numbers were taken in LPM; the E-cluster AMX oddity
needs re-verification).

Consequence for the runtime design: "mono-thread default, threads only if it helps" is wrong for Kokoro CPU.
Kernels need a thread pool from the start of the fused work. Graph-level parallelism (noise branch, harmonic
source on another thread) is secondary.

### P3. The model can't run free: no path from `ids + style + speed → audio`
The graph takes the SineGen noise as an **input of shape `[600·F, 9]`**, so the caller must already know F, the
model's own output length. The RNG kernel the design promised doesn't exist, and nothing in `todo.md` covers
it. Decision needed now, because it shapes the fused harmonic-source kernel: generate the noise **inside** the
harmonic-source kernel with a counter-based generator (Philox) seeded per call, with the explicit-noise input
kept only for tests (an optional input). Bit-matching torch's RNG stream is not needed (`research-kokoro.md`
§2.4: seed-to-seed audio correlation 0.995).

### P4. The iPhone has no target, and it is half the goal
`design-decisions.md` and the memory of the user's wider goal are explicit: Kokoro must run on **both** the Mac
and the iPhone. The plan has one line for it ("same on iPhone") and no definition of "beat PyTorch" there (there
is no PyTorch on iOS). The hardware is also different enough that the Mac strategies don't transfer:
- A20 Pro CPU: **2 P-cores**, so the multi-stream AMX trick (P2) buys little; SME2 is exposed directly.
- A20 Pro GPU: 7 cores with **neural accelerators reachable through Metal 4 `matmul2d`/MPP, f16**. On the M2 Max
  f16 gives no compute gain (10.1 vs 9.8 TFLOPS) and the generator is compute-bound, so f16 is irrelevant on the
  Mac and decisive on the phone.

**Change:** define iPhone targets as real-time factor and against the published on-device ports (CoreML/ANE
ports claim ~0.3 s for 30 s of audio on an iPhone 15 Pro Max, MLX-Swift ~3.3× real-time). Treat iPhone GPU
as its own kernel family (f16, tensor ops) from the conv proof onward, and benchmark every kernel on both
devices through the remote executor. Ask the user what "win" means on the phone (RTF? vs CoreML?).

### P5. The GPU plan contradicts itself on `F`, and mixed CPU/GPU is under-designed
- `design-runtime-and-kernels.md` makes GPU-driven dispatch (upper-bound allocation) the default; the Kokoro
  research shows the bound is F ≤ 25 600 (a 1.6 GB activation) and recommends one readback. The readback with
  overlap (commit the F-independent TextEncoder work while waiting for F) needs **multiple in-flight command
  buffers**, which the single-buffer GPU context can't do and the plan doesn't schedule.
- The researched GPU shape runs the LSTMs on the CPU concurrently with GPU convs. Every CPU↔GPU crossing costs
  ≥ 120 µs round trip and ~440 µs GPU wake after 3 ms idle; six LSTMs plus the readback is a floor of several
  ms, which matters against MLX's 67 ms at 16 tokens. The compiler's `insert-sync` also tracks hazards per
  alias root (one pending GPU write anywhere in an arena forces a sync on every later CPU touch of that arena)
  and misses CPU-after-GPU write-after-read; a Kokoro-sized mixed graph would sync at nearly every transition.

**Change:** decide the LSTM placement by measurement early (persistent-threadgroup GPU kernel vs CPU), and
only then invest in mixed graphs. Drop "GPU-driven" as the default; keep it as an experiment. Add the
multi-command-buffer context and byte-range sync analysis to the plan as prerequisites of the GPU Kokoro path.

### P6. Runtime micro-work is scheduled before it matters
ICB record/replay, three CPU runtime strategies, dependency-level barriers, compiler pipeline < 1 ms: with
~170 dispatches per forward and 2 µs per dependent dispatch, all of this is ≈ 0.3 ms against a 400 ms GPU
budget. The user's plan asks for these comparisons, so keep them, but **after** Kokoro beats the baselines, or
justify them with MobileNetV2 (many tiny ops), where they do matter.

### P7. Numerical strategy chases the wrong fidelity
The research shows the SineGen phase reaches ~5·10⁵ rad with f32 ulp ~0.03 rad, so bit-closeness to torch's
f32 cumsum is luck, not correctness. Chasing torch's op order constrains the fused kernel for no benefit.
**Change:** use a float64 numpy golden for the harmonic source and iSTFT, verify torch *and* lokinfer against
it, and judge audio by SNR against the reference. Keep exactness only where it is cheap and decisive: the
duration head (replicate torch's reduction order so F matches; a ±1 frame flip changes the output length).

### P8. Memory on the phone
GPU models currently copy every constant into pooled storage, and the pool rounds >1 MiB buffers to a power of
two: a 330 MB f32 Kokoro would use ~2–3× that on an iPhone. The `.lkm` format was designed for mmap →
`newBufferWithBytesNoCopy`; the plan must require that path for GPU constants (and f16 weights on the phone
halve it again).

### P9. The kernel ABI is a second source of truth
Kernel signatures are "Python-only", but every C++ adapter re-encodes the layout as magic int indices, and
structural attributes (`nohoist`, `inplace`) leak into the same int list, so the first RNG kernel (`nohoist`)
would read its attrs from the wrong slot. Before writing ~40 fused kernels: separate structural attrs from
kernel attrs in the IR, and add a consistency check (every registered kernel has a Python def, and the Python
def's attr order is what the adapter expects, e.g. via a generated header or a per-kernel self-test).

### P10. Scope: six other models
The port order is now MNIST → Kokoro → others (user-confirmed). The plan should say what the other models are
*for*: they only earn their place if they teach kernels Kokoro needs (GPT-2's attention/LN ≈ ALBERT; DCGAN-G's
transposed conv ≈ the upsamplers) or exercise runtime paths (MobileNetV2 for dispatch overhead). Port them to
"correct + a few fused kernels", not to "beat PyTorch", unless the user wants otherwise.

### P11. Process drift
- Design docs state unbuilt machinery as fact (dim programs, mem.plan cache, dce/bake-init/record-replay
  passes, chunked commits, submit/wait, caller-provided outputs). Mark each section *built / planned*.
- `design-decisions.md` is the Q&A log but doesn't record later decisions (port order, Metal-4-only ⇒ no OS
  fallback below 27).
- All research numbers are Low Power Mode numbers; "must beat" targets must be re-measured first.
- The user's documentation list is only half done (no correctness-testing, benchmarking, per-model, or
  kernel-optimization docs yet).

---

## 3. Ideas worth adding to the plan

1. **Library kernels first, at the exact shapes.** Try BNNS conv (CPU, AMX, fused activation) and MPS
   `MPSCNNConvolution` / MPSGraph conv (GPU) on stage-1 shapes before writing custom convs. If either hits the
   floor, the effort goes elsewhere. The plan allows library kernels but never schedules the trial.
2. **Built-in per-kernel profiler in the runtime** (CPU clock per instruction; GPU per-segment, optionally
   per-dispatch timestamps). The optimize loop (step 11) is only as fast as the profiler; xctrace alone is slow.
3. **Kernel variant tables per device.** Same-semantics kernels (`gemm_f32_cpu`, `..._accelerate`, `..._gpu`,
   `..._gpu_mps`) selected by a generated table keyed by device and shape, fed by the benchmark DB. This is the
   first step of the future `lk.nn` layer and removes hand-editing of models when a faster variant appears.
4. **Stats sharing across the three resblocks** and never materializing `mean_j` (already in the research;
   make it a plan item with a test).
5. **Time-major generator layout** as a first-class experiment in the conv proof (kokoro.cpp's finding).
6. **f16 activations only where bandwidth-bound or where the hardware has f16 compute** (iPhone), f32
   accumulation and f32 statistics everywhere.
7. **Speed regression tests per device from day one of the fused work**, thresholds tuned from measured noise,
   with Low Power Mode and thermal state recorded in every result (user decision).

---

## 4. Revised roadmap (milestones with exit criteria)

**M0 — fix the foundations (no new kernels)**
- Re-baseline Kokoro (torch eager/compile CPU+MPS, MLX) and the micro-benchmarks with Low Power Mode off.
- Decide and record: RNG inside the harmonic-source kernel; readback-with-overlap for F; iPhone targets;
  structural vs kernel attrs in the IR; thread pool for kernels.
- Mark design docs built/planned; log decisions in `design-decisions.md`.
- Exit: `kokoro-port-plan.md` v2 with the numbers and decisions above.

**M1 — Kokoro correct on CPU, basic graph** (kernels exist; tests have not been run)
- Stage-by-stage tests vs the bit-exact reference; durations exact on all 9 cases; audio SNR threshold.
- Free-running variant works (in-kernel RNG), explicit-noise variant for tests.
- Exit: `tests/models/test_kokoro.py` green; baseline timing of the unfused graph recorded (expect slower than
  PyTorch; that's fine).

**M2 — conv proof (the de-risk)**
- Standalone stage-1 and stage-0 conv benchmarks through the kernel library, Mac CPU / Mac GPU / iPhone GPU:
  shifted-GEMM vs im2col, multi-stream single-threaded sgemm vs Accelerate threading, `[C,L]` vs `[L,C]`,
  BNNS and MPS candidates, simdgroup f32 vs MPP f16 on the phone.
- Exit: measured TFLOP/s per device for k3/k7/k11 × d1/d3/d5 at L = 20F and 120F. Thresholds: CPU ≥ 1.8
  aggregate (then the 1.4 s target holds with margin); GPU ≥ 4.0 on M2 Max. Below that, revise targets *before*
  building the fused layer, not after.

**M3 — fused generator on CPU → beat 3.09 s / 850 ms**
- `conv1d_fused` with AdaIN/Snake prologue, stats epilogue, residual; polyphase ConvTranspose; shared stats
  per stage; fused harmonic source + STFT with in-kernel RNG; fused LSTM; token-rate GEMMs with gather.
- Each fused kernel tested against the composition of basic kernels (shape sweeps) and the stage references.
- Per-kernel profiler drives the order of work.

**M4 — GPU generator → beat MLX (short) and torch.compile-MPS (long)**
- Multi-command-buffer context, readback with overlap, LSTM placement by measurement, byte-range sync
  analysis if mixed graphs are kept.

**M5 — iPhone** — f16 tensor-op conv family, 2-P-core CPU strategy, remote benchmarks of every kernel, targets
from M0.

**M6 — the rest of the original plan**: ICB replay and CPU runtime comparison, speed-regression suite, the
documentation list, then the other models at "correct + instructive" depth.

---

## 5. Questions for the user — answered 2026-11-01 (see `design-decisions.md`)

1. iPhone: an **artificial target** derived from the Mac "beat PyTorch" result scaled by measured device peaks;
   the goal is to beat it.
2. Other models: as much performance as possible, not required to beat PyTorch.
3. Threading depends on the graph: small kernels run inline one by one; big kernels use the multithreaded
   runtime and run asynchronously (work-estimate threshold per device).
4. Noise: like PyTorch, generated internally from a settable seed; explicit-noise variant for parity tests only.

---

## Addendum (same day): user decision + code-level findings that change the plan

**User decision:** device boundaries are explicit in the IR, written by the Python graph, never inferred
(on unified memory nothing is copied; a boundary is a sync point). P5's sync-pass concerns are resolved by
design: the pass becomes a verifier, the Python API grows `to_cpu`/`to_gpu` which emit the existing `sync` op.

Foundation items the code reviews surfaced (all in `todo.md` M0; none affect the strategy, all would have
bitten later):
- hoist-init can hoist a kernel whose `dim<...>` attr references an exec input (unbound symbol at load);
- bufferize in-place reuse doesn't check aliases (a view of the destination among the operands) and in-place
  chains never reuse;
- outputs that are state/constants are returned as mutable aliases of model memory;
- the `.lkm` loader and the remote server trust their input (overflow in bounds checks, unvalidated enums/ids,
  unbounded frame sizes, no `verifyModel` before `create`); the phone exposes this on the network;
- the dim parser is recursive and overflows on long literals; division by a constant zero traps;
- GPU models copy every constant at load and every input per call (the zero-copy path exists only for
  runtime-allocated storage);
- missing runtime tests: mixed-device graphs, data-dependent dims end to end, non-empty init, in-place, outputs
  that are views/state, loader/protocol fuzzing.
