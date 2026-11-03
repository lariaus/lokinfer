# lokinfer — Design Decisions (Q&A log)

Answers from the initial design question session (2026-11-01), on top of
`original-plan.txt`. Overall rule from the user: **the goal is to be fast** —
any decision below can be revisited if measurements show a faster option.

## Project / tooling

- Fully independent project in its own repo (own venv, own Kokoro install, own docs).
  English-practice integration is a **separate, later task** — not part of this project.
- Build: **CMake + Ninja**, **C++20**.
- Third-party C++ deps: **CMake FetchContent, pinned versions**.
- Python bindings: **nanobind**.
- Python env: **own uv-managed `.venv` in lokinfer, Python 3.14** (same as english-practice).
- C++ unit tests: **Catch2**.
- Compiler pass tests: **pip `lit` + pip `filecheck`** (Python port of FileCheck).
- Git: the user inits and handles git themselves. Claude never runs git; only maintains `.gitignore`.
- Design docs live in **`docs/impl-notes/`**. `original-plan.txt` is the user's file (don't edit).
- Process: after this Q&A session, **fully autonomous**, with no approval gate before implementation.
  Design ↔ implement is a loop; update docs/todo as reality changes.

## Platform / Metal

- Targets: **macOS 27 / iOS 27 only** (M2 Max Mac, iPhone 18 Pro Max).
- Metal access: **metal-cpp**.
- Metal API generation: **research Metal 4 vs classic Metal 3, benchmark dispatch overhead, then decide**.
- MSL: **compiled offline to `.metallib` at build time**.
- Devices: **CPU + GPU only, no ANE**. Kernels may use any library (Accelerate/BLAS, MPS, MPSGraph, ...).

## Python development model

- Models are developed in Python **on the Mac only**.
- iOS has **no Python**. It only loads a binary model file and runs it/tests it through the C++ API.
- Kernel/op definitions (signatures, output-type rules) are **Python-side only**. C++ just looks kernels up by name.

## IR

- Textual IR: **custom MLIR-like syntax** with its own small parser/printer. No LLVM/MLIR dependency.
- Binary format: **custom mmappable format** (header + IR + aligned weight blob, mmap→MTLBuffer no-copy).
- Dtypes: **whatever the target models need**. That includes f32, f16, bf16, i32/i64, bool, and int8/quantized where useful.
  *Narrowed 2026-11-02: model math is strictly f32 (see "f32 only" in the log below); integer/bool dtypes stay.*
- Dim expressions: start with **`+ - * floordiv ceildiv mod min max`**. Add more as model support requires.
- **Data-dependent dims ARE in the IR** (a kernel can produce a dim value at runtime).
  Kokoro is a **single graph**, like its single PyTorch `forward()`.
  - GPU: **try both** a GPU→CPU readback sync and GPU-driven (indirect dispatch + upper-bound alloc), then benchmark.
    *Superseded 2026-11-02: `F` is computed on the CPU in both builds, so there is nothing to read back (log below).*
- Memory planning with dynamic/data-dependent shapes: **symbolic offsets evaluated at runtime**.
- Layout: start **contiguous row-major**, with **reshape/contiguous-slice as free view ops** (planner handles aliasing).
  User: *"these might change, really do what gives the best performance in the end."*
- External constants: compiled model **holds a zero-copy reference** (keeps the owner alive). **Internalize on export.**
- Error handling: **no exceptions anywhere** (status/expected). Runtime checks are **asserts for now**, to reconsider later.

## Tensor / runtime API

- Interop: **DLPack + numpy buffer protocol**, zero-copy where possible.
- Tensor has **device=cpu|gpu**. GPU memory = **shared-storage MTLBuffer** (unified memory).
- `model(...)` is **synchronous by default**. C++ also exposes submit()/wait().
- A graph **may mix CPU and GPU kernels**. The runtime inserts sync points.
  *Superseded 2026-11-01: boundaries are explicit `sync` ops written by the Python graph; the compiler only verifies them (log below).*

## Testing

- iPhone correctness: **both** a remote executor (pytest on the Mac drives the phone) **and** on-device reference sweeps.
- Mac↔iPhone transport: **TCP over local network / USB tunnel**.
- Standing permission: Claude may **build/install/launch the lokinfer test app on the iPhone anytime** without asking.
  If the phone is locked/disconnected or the weekly provisioning-trust issue appears, stop and tell the user.
  *Refined later (user): tell the user in one line, keep working on something else, retry in the background every
  5 minutes and re-ask while it is still blocked; never end the turn on that blocker.*
- Benchmarks: **run anytime**, repeat, and report median + variance. Re-run when noisy.
- Speed-regression thresholds: **Claude decides and tunes** based on observed noise.

## Models / reference data

- Weights: **pretrained when available**. Otherwise train briefly in PyTorch (**≤ ~5 min**). Otherwise random.
- Kokoro: **own install in lokinfer's venv, exactly what english-practice uses**:
  `kokoro==0.7.16`, `torch==2.14.1`, `misaki==0.7.4`,
  HF `hexgrad/Kokoro-82M` snapshot `f3ff3571791e39611d31c381e3a41a3af07b4987` (`config.json`, `kokoro-v1_0.pth`).
- Kokoro baseline = english-practice's `kokoro_model.py` `infer(ids, ref_s, speed)`: **CPU, float32**.
  "Model inference" and "E2E" goals both mean **this model forward only** (ids + style + speed → audio),
  not the text frontend.
- Kokoro decoder randomness: **noise is an explicit model input** for tests. Real use gets an RNG kernel.
  *Refined 2026-11-01 (user): like PyTorch, the shipped model generates its noise from a settable seed (log below).*
- Kokoro reference inputs: generated with **kokoro's own KPipeline in lokinfer's venv** (real sentences + voices),
  plus synthetic edge cases.

## Benchmark baselines

- Compare against **the best framework that works on each device**, not only PyTorch:
  PyTorch eager (CPU/MPS), torch.compile, MLX, CoreML (CPU/GPU only), ONNX Runtime. Always CPU vs CPU and GPU vs GPU.
- Tooling: try all of Google Benchmark (CPU), a custom Metal timing harness with GPU timestamps,
  Instruments/xctrace CLI, and a results DB with comparison reports. Keep what helps most.

## Later decisions (log continues here)

- 2026-11-01 — **Port order**: MNIST → **Kokoro** → the others (user confirmed; "Kokoro first, other models'
  kernels overlap anyway").
- 2026-11-01 — **Metal 4 only** (consequence: macOS 27 / iOS 27 minimum, no fallback; MPS/MPSGraph encode on our
  Metal 4 encoder through the 27-SDK entry points).
- 2026-11-01 — **Metal Toolchain** installed with the user's permission (`xcodebuild -downloadComponent MetalToolchain`).
- 2026-11-01 — **Low Power Mode** was on for every research measurement; the user turned it off. Every benchmark
  result now records power mode + thermal state; baselines are only updated with it off.
- 2026-11-01 — **iPhone**: standing permission to build/install/launch anytime; the user set Auto-Lock to Never
  for test sessions; `scripts/ios_deploy.sh` keeps the app in the foreground and relaunches it if it crashes.
- 2026-11-01 — **Plan review** (`plan-review-2026-11-01.md`): milestones M0–M6 replace the linear phase list.
  Open questions for the user: iPhone target definition; depth of the other model ports; thread pool by default;
  explicit-noise input test-only vs shipped. The proposed answers are in that doc and in `kokoro-port-plan.md` v2.
- 2026-11-01 — **Mixed CPU/GPU graphs: device boundaries are explicit in the IR, written by the Python
  graph, never inferred by the compiler.** The user's principle: "devices should be explicit in the IR and not
  inferred." With unified memory there is nothing to copy; a boundary is a **sync point** (`ops.to_cpu(x)` /
  `ops.to_gpu(x)` → the IR `sync` op). Consequences: the compiler's `insert-sync` inference pass is replaced by
  a *verification* (`verify-devices`) that every kernel's operands were produced on its device or passed
  through an explicit sync; the runtime executes the sync (commit + wait when leaving the GPU, segment
  boundary when entering it). Mixed graphs are opt-in and visible in the IR.
- 2026-11-01 — **Answers to the plan-review questions (user):**
  1. **Targets.** On the Mac: beat PyTorch. On the iPhone there is no PyTorch, so build an **artificial target**
     from what beating PyTorch means on the Mac and what the device can do, and beat that. Method (derived):
     measure the same peaks on both devices in M2 (GEMM/conv TFLOP/s per stream, memory bandwidth, dispatch
     cost), then `target_iphone = lokinfer_mac_time × (mac_peak / iphone_peak)` per device class (CPU, GPU),
     plus a real-time-factor sanity bound. Written down with its inputs so it can be audited.
  2. **Other models (ResNet-18, DCGAN G/D, GPT-2, MobileNetV2):** as much performance as reasonably possible,
     **not** required to beat PyTorch.
  3. **Threading depends on the graph.** Small kernels: run them directly, one by one, no threading. Big
     kernels: enable the multithreaded runtime and run those big kernels asynchronously. Design (derived): each
     kernel reports a work estimate; below a threshold the runtime calls it inline on the calling thread;
     above it the kernel's own parallel implementation runs on the pool, and independent big kernels may
     overlap. The threshold is tuned per device from measurements.
  4. **Noise / RNG: do like PyTorch.** The model generates the SineGen noise internally (as `randn_like` does),
     from a seed the caller can set (the equivalent of `torch.manual_seed`); no noise input in the shipped
     model. An explicit-noise variant may exist for parity tests only.
- 2026-11-01 — **Constants carry a device; GPU models are GPU end to end** (user: "weights can be on GPU too,
  constants can be on CPU or GPU, weight preparation like transposing should move to init"). Captured tensors keep
  their device in the IR (`device` attr); weight preparation runs as a kernel on the model's device and is hoisted
  into init. Mixed models stay possible with explicit boundaries.
- 2026-11-01 — **Avoid copying graph inputs and outputs whenever possible** (user, a general rule for E2E time).
  Implemented: owned caller memory is wrapped in place for GPU models (cached per input), outputs are the call's
  own storage, weights are wrapped from the mmapped file. Copies only where correctness needs them (outputs that
  alias inputs/constants/state, ownerless raw memory).

### Later decisions, continued (recorded 2026-11-02)

Each entry gives its date and its source: **user** (the user said it), **derived** (Claude decided it from a
measurement or from a user rule; it loses to any later user decision), or **user + derived** (the user set the
principle, the mechanism is derived). Where an earlier entry above already covers a point, it is not repeated.

- 2026-11-01 — **Verification is not a pass** (user request). The read-only passes `verify`, `verify-devices` and
  `verify-buffers` became an MLIR-style always-on verifier, run when a model is loaded or built and after every pass
  in `runPasses`, so an invalid IR names the pass that broke it. The default pipeline is
  `dce → bufferize → plan-memory → hoist-init`. Details: [../verifier.md](../verifier.md).
- 2026-11-01 — **Region result types in the IR** (user request): `init -> (...) {` and `exec(...) -> t {`, MLIR
  style, printed by the C++ printer and by the Python trace; the parser checks them against `return` (optional in
  hand-written IR).
- 2026-11-01 — **Explicit device boundaries, implementation** (user principle above; mechanism derived): `ops.to_cpu`
  / `ops.to_gpu` emit the `sync` op; the verifier tracks residency and rejects implicit crossings and CPU writes over
  pending GPU reads/writes. Constants carry their `device` (user). This removed the alias-root and write-after-read
  bugs of the old `insert-sync` inference pass.
- 2026-11-01 — **Structural attrs separate from kernel attrs** (derived, from the plan review): `nohoist`,
  `inplace`, `output`, `arena` never shift a kernel's int ABI; every registered kernel must have a Python definition
  with the attr order its adapter expects (checked).
- 2026-11-01 — **Untrusted input is hardened** (derived: the phone's remote executor listens on the network): the
  `.lkm` loader is overflow-safe and validates enums/ids/sizes, the server runs `verifyModel` before
  `Executable::create`, the remote protocol caps frame sizes and checks shape ↔ nbytes, parsers are depth-limited.
- 2026-11-01 — **Threading mechanism** (user rule 3 above; mechanism derived): `rt::ThreadPool` with persistent
  workers at USER_INTERACTIVE QoS, used only above a per-kernel work threshold (e.g. `conv1d_fused` inline below
  4 M MACs). Pool size = the P-cores, plus the E-cores on chips with fewer than 4 P-cores (derived, measured with the
  M2 conv proof: on the M2 Max 12 streams were slower than 8; on the iPhone 18 Pro Max 6 streams were fastest).
- 2026-11-01 — **LSTMs stay on the CPU in both builds** (derived, measured): a CPU step is ≈ 8-12 µs (sgemv of the
  1 MB `w_hh` from L2 + gates); a GPU per-step dispatch alone costs more, and a persistent GPU kernel cannot sync
  threadgroups per step.
- 2026-11-01 — **GPU frame-rate path is channel-last `[L, C]`** (derived, M2 conv proof: MLX steel conv channel-last
  6.2-6.75 TF vs ≤ 1.1 TF for the hand-written channel-first implicit GEMM). See
  [m2-conv-proof.md](m2-conv-proof.md).
- 2026-11-01 — **Kokoro representative shapes** (user request): optimize for the real workload, not only the three
  research cases. The app's TTS pack measures T 34-83 tokens (median 59), F 98-224 frames (median 161); cases `word`,
  `app{47,59,71}{f,m}` were added, and `app59f` is the default profiling case. Same day (user): shadowing is moving to
  5-10 s utterances plus occasional 20-60 s+ TTS (chunked ≤ 510-token forwards) → cases `s6`, `s9`, `s20`, `s28f`.
  Details: [../models/kokoro.md](../models/kokoro.md) "Representative shapes".
- 2026-11-01 — **Benchmark timing protocol** (user: "of course you need some warm up time, and many iterations"):
  warm-up ≥ 3 calls and ≥ ~1 s, then ≥ the bench's run count and ≥ ~2 s measured, median reported. iPhone cool-down
  (user: "wait a bit, but not much"): one capped wait for thermal "nominal" at the start of a sweep (≤ 5 min) and one
  short break per model (≤ 60 s), never per case or run; a throttled result never replaces a nominal one (derived).
  Details: [../perf-dashboard.md](../perf-dashboard.md).
- 2026-11-01 — **Dashboard coverage rule** (user): every model has Mac CPU, Mac GPU, iPhone CPU and iPhone GPU cards,
  each with lokinfer's numbers and measured baselines.
- 2026-11-01 — **Apples-to-apples baselines** (derived from the user's "compare against the best framework" rule):
  a baseline is recorded only if it runs the same model and weights in f32 (checked per framework, e.g. MLX with
  `MLX_ENABLE_TF32=0` on the iPhone), on the exact reference inputs, doing the same work (Kokoro: output length =
  600·F of the reference), E2E per call, with machine state recorded. Details: [../baselines.md](../baselines.md) §2.
- 2026-11-01, extended 2026-11-02 — **Benchmark lock rules** (derived, after an unlocked iPhone sweep read Kokoro GPU
  real117 at 305 ms instead of 248 ms, and builds during a Mac GPU bench read s6f at 70-195 ms instead of 69 ms):
  `scripts/lk-lock` serializes the shared build dir, Mac timing/profiling runs and iPhone sessions; any build (Mac
  or the iOS build in `scripts/ios_deploy.sh`) also takes the Mac bench lock; nesting order is always
  phone → macbench → build. The iPhone sweep runs GPU variants before CPU per model (long CPU runs heat the phone).
  Details: [../benchmarking.md](../benchmarking.md) "Shared-resource locks".
- 2026-11-02 — **`F` handling** (derived, todo triage): the duration predictor runs on the CPU in both builds, so `F`
  is a `dim.read` of a CPU tensor and is never read back from the GPU; the "readback vs GPU-driven" comparison of the
  initial Q&A is dropped. CPU/GPU overlap comes from async commits.
- 2026-11-02 — **f32 only, apples to apples** (todo triage; derived from the apples-to-apples rule, which requires f32
  weights and math from every baseline, so lokinfer is held to the same): no f16/bf16 model math or f16 weights, on
  either device; the f16 tensor-op conv/GEMM family planned for the iPhone (plan review, M5) is superseded. Metal 4
  tensor ops stay a candidate only in f32.
- 2026-11-02 — **Sustained vs burst measurements** (derived, measured): a minutes-long dashboard sweep reads mid-size
  Kokoro GPU cases ~20-25 % slower than a short loop (Mac s6f 87 vs 69 ms) while the thermal state stays "nominal".
  Rule: compare implementations only under the same protocol (sweep vs sweep, which the dashboard does); one-off app
  calls are closer to the short-loop numbers. `bench_kokoro_gpu.py` records the lower median of two full sweeps.
- 2026-11-02 — **Focus testing and benchmarking on the affected cases** (user: "focus testing / bench corresponding
  on the specific cases"; don't rerun long sweeps after every small change): a change runs the kernel tests and
  benchmark cases it touches (e.g. a K=7 conv change → its tests + the cases where K=7 matters; a CPU-only change → no
  GPU tests or benches), measured with `prof-lokinfer` or a short A/B on representative cases (e.g. `app59f` + one
  long case). Full sweeps and dashboard re-records run once at the end of a larger piece of work, or when numbers are
  known to be stale.
- 2026-11-02 — **GPU record/replay (ICB) is built but opt-in** (runtime agent; `LOKINFER_GPU_REPLAY=1`). Decided by
  measurement: MobileNetV2 GPU N=1 on the Mac, the runtime-overhead case, is 3 % slower replayed (1.152 vs
  1.117-1.128 ms), as `research-metal-runtime.md` §3 predicted (ICB dependent chains cost more GPU time than the
  overlapped CPU encode they save). Design choices: record by teeing the first call's dispatches (no separate
  warm-up call, the recording call is a normal call); eligibility is structural (no `dim.read`, GPU kernels in one
  segment) and the command list is cached per input-shape signature (≤ 4), because the models are compiled with
  symbolic shapes (a "static model only" rule excluded MobileNetV2 itself); per-call argument moves are patched
  with `setKernelBuffer`, never re-recorded; anything the recorder can't reproduce (MPS) turns it off for the
  executable rather than failing the call.
- 2026-11-02 — **Two GPU segments in flight by default on macOS, one on iOS** (`LOKINFER_GPU_INFLIGHT`, 1-4): Kokoro app59f Mac GPU 50.7 →
  50.1 ms (two interleaved A/B pairs, 3 no better); correctness rests on a queue barrier at the start of a segment
  opened while another is in flight and one autorelease pool spanning in-flight segments. The win is small because
  the call is GPU-bound (the CPU's wait moves from "in flight" to the next sync); GPT-2 Mac prefill T=128 5.04 →
  4.90, decode 2.17 → 2.03 ms. iPhone: Kokoro 65.6 → 64.9 and GPT-2 decode 5.81 → 5.59, but GPT-2 prefill T=128 went
  bimodal (11.3 or 12.9 ms vs a steady 11.56); rotating slots on every call was one cause (fixed: reuse the last slot
  while nothing is in flight), the rest is not understood, so under the "Kokoro + GPT-2 never regress" goal iOS keeps
  one slot. ICB replay on the iPhone: +4 % (1.048 vs 1.008 ms), same verdict as the Mac.
- 2026-11-02 — **Remote protocol additions** (backward compatible, advertised in HELLO "features"): MEMORY
  (task_vm_info + MTLDevice.currentAllocatedSize) and RUN2 (inputs inline, by content-addressed blob or generated
  on the device from a seed; outputs optionally returned; comparisons on the device against blobs or other
  outputs). Chosen over a per-test data push so that test data crosses the cable once and sweeps move no data at
  all. The generator is splitmix64 + Box-Muller, mirrored in numpy (`lokinfer.remote.generate`).
- 2026-11-02 — **pytest device parameter**: `--device=mac|serve|iphone`. On a remote device every compiled graph
  runs remotely by intercepting `CompiledModel.__call__` in `tests/conftest.py` (no test rewritten); the device's
  outputs are written into the Mac's result tensors so tests are unchanged; tests that ran nothing remotely are
  reported skipped, never passed. `serve` (a local lokinfer-serve) validates the remote path without the phone.
- 2026-11-02 — **Peak memory is measured per model in a fresh executor process** (`bench/models/measure_memory.py`),
  with the model blob uploaded by a throwaway process first, so the peaks contain neither other models nor the
  upload buffer. phys_footprint (what iOS's memory limit counts) is the headline number.
- 2026-11-02 — **Compiler pipeline < 1 ms, with the verifier still after every pass** (compiler agent): 5.1 → 0.92 ms
  on the 3,800-op synthetic graph (Release, M2 Max). Kept per-pass verification (the user's 2026-11-01 request) and made
  it cheap (dense tables: 0.35 → 0.06-0.08 ms per run) instead of verifying once at the end. Tried and reverted:
  running the device-residency check inside the structural verifier's walk (slower than two tight loops).
- 2026-11-02 — **mem.plan offsets are element offsets** (attr `elem = [bytes per element...]`): views use the symbol
  directly instead of `O // 4` (one dim node less per planned buffer at compile time, one division less per view per
  call). IR without `elem` (hand-written, older `.lkm`) keeps byte offsets.
- 2026-11-02 — **One arena per pool shared by all mem.plans of a call, plus a plan cache keyed by symbol values**:
  later plans place their buffers around the earlier plans' live buffers (obstacles in the same greedy placement),
  rather than the planned "base = high-water mark" (which wastes the dead first-plan space). The arena is never
  reallocated while any of its buffers can be live: only at the pool's first plan of a call; a later plan that
  doesn't fit gets a one-call overflow storage and grows the arena for the next call. Cache: 8 results per plan,
  most recent first, keyed by the symbols of its sizes and of the earlier plans it shares the arena with.
- 2026-11-02 — **Dim programs not built; dim evaluation is memoized per call** (`DimEvaluator`). Measured: on 1,000
  tiny symbolic-shape CPU kernels the symbolic overhead is ~20 ns/kernel, of which dim evaluation after memoization is
  ~4 %; the rest is the per-call views into the planned arena. A compiled evaluator would not pay for itself.
- 2026-11-02 — **`submit()/wait()` is a per-model worker thread in Python** (the native call releases the GIL),
  not an asynchronous runtime call: it gives overlap with Python work and keeps calls ordered with no runtime change.
  A runtime-level async call (commit without waiting, completion handler) can come later if a model needs the CPU
  thread itself back during GPU work.
- 2026-11-02 — **`model(x, out=y)`** binds the output's alloc (or the alloc a zero-offset view of it comes from) to the
  caller's memory; otherwise (outputs that are inputs/constants/state, partial views) the result is copied into `y`.
  Also: outputs that are zero-offset views of an output alloc (reshapes) are now returned without a copy.
- 2026-11-02 — **internalize is a C++ function, bake-init a runtime function, neither a pipeline pass**: both need
  data a `Status(Model&)` pass doesn't have (the external tensors; running init). bake-init bakes only "computed
  data" init results (plain allocs written by init kernels); arenas, persistent buffers, constants and models whose
  init defines symbols are left alone. Baked constants are resident on the device of the kernel that wrote them.
- 2026-11-02 — **torch MPS zero-copy both ways**: MPS → lokinfer was already in place (DLPack data = id<MTLBuffer>);
  lokinfer → MPS exports a kDLMetal DLPack array (`Tensor.torch(device="mps")`), which torch 2.14 imports sharing
  the buffer. `.torch()` stays a CPU view by default (unchanged behavior).

### 2026-11-02 — Goals reordered (user)
- **Kokoro-82M and GPT-2 124M: lokinfer #1 in every configuration (all devices × all dashboard cases), never
  regress.** Kokoro is the most important model. Enforced by `tests/perf/test_dashboard_goals.py` and a warning in
  `bench/perfdb.py` `record()`.
- Other models: within 5-10 % of the best baseline everywhere, ideally #1 — important but second.
