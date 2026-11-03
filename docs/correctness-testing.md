# Correctness testing

How lokinfer proves a kernel, a graph or a model is right. Sections are marked **[built]** or **[planned]**.
The rules come from `impl-notes/original-plan.txt` (test through graphs; references from another framework or
from another kernel in the project) and from the 2026-11-01 plan review.

## Levels

1. **Unit tests [built]** — C++ (Catch2, `tests/unit/`): dims, text and binary round trips, verifier, passes,
   runtime end to end on tiny graphs. Compiler passes have FileCheck tests (`tests/lit/`, one directory per pass).
   Python (`tests/python/`): tensor interop, tracing, compile, save/load, remote protocol.
2. **Kernel tests [built]** — `tests/kernels/` (Kokoro basic + fused CPU/GPU sets, GEMM, steel, Winograd,
   depthwise, transformer kernels): every kernel runs through a one-op graph
   (eager call) and is compared with the PyTorch functional op over **varied shapes**: odd sizes, length 1, every
   stride/dilation/padding combination the target models use, batch 1 and typical batch. Tolerances are stated
   per kernel in the test and justified (f32 reordering grows with the reduction length: use `rtol·√K`).
3. **Fused-kernel tests [built: vs the float64 torch composition of the ops a fused kernel replaces
   (`test_kokoro_conv_fused.py`, `test_winograd_gpu.py`, `test_transformer_fused.py`, ...); planned: vs the
   composition of lokinfer's basic kernels]** — a fused kernel is compared with the **composition of basic kernels**
   it replaces, over the same shape sweeps, in addition to PyTorch. The basic kernels are therefore a reference
   layer: kept simple, never optimized, and the oracle on devices without PyTorch (the iPhone).
4. **Stage tests [built, M1]** — a model graph is built with named intermediates; each is compared with the
   captured reference (`models/kokoro/reference.py` captures every stage of Kokoro bit-exactly from the reference
   PyTorch `infer()` math; `tests/models/test_kokoro.py::test_stages_forced_durations`, CPU and GPU builds).
   Token-rate stages must match to ~1e-4 relative; frame-rate stages to ~1e-3; anything looser needs a written
   reason (measured: token-rate and decoder ≤ 5e-5; `post` < 1e-2 and the harmonic source judged by magnitude +
   energy-weighted phase, both phase-sensitive, see the golden rule below).
5. **Model tests [built]** — the whole forward vs the reference outputs: exact where exactness is cheap
   and decisive (Kokoro durations: a ±1 frame flip changes the output length), metric-based where f32
   reordering makes bit-closeness meaningless (Kokoro audio: SNR against the reference, threshold set from the
   float64 golden below). Plus whole-dataset checks when they exist (MNIST test accuracy = 97.44 %). The other
   models (`tests/models/`) are compared with torchvision / Hugging Face / PyTorch outputs; GPT-2 generation must
   match HF `generate` token for token.
6. **Speed-regression tests [built: Kokoro; planned: other models]** — `tests/perf/`, see `docs/benchmarking.md`.

## Rules

- **Test through graphs, not kernels in isolation.** The runtime overhead is ~320 ns per one-kernel graph call (LPM off, 2026-11-02), so a
  one-op graph costs nothing and tests the ABI, the type rule and the kernel together.
- **Op definitions live in Python; the C++ adapter must agree.** [built] `tests/python/test_registry.py`: every
  Python op is registered on the device its name says, and every registered kernel has a Python definition.
  Attribute order needs no check: kernel attrs reach the adapter in attribute-name order and structural attrs are
  never passed (`isStructuralAttr`).
- **Golden references for phase-sensitive math.** [built] Kokoro's harmonic source accumulates ~5·10⁵ rad of
  phase; f32 ulp there is ~0.03 rad, so torch's own f32 result is not a bit-exact oracle. A float64 numpy golden
  of SineGen/STFT/iSTFT is the reference; both torch and lokinfer are measured against it, and the test
  tolerance is torch's own error (`reference.golden_audio`, used by the free-running SNR tests).
- **Forced inputs for data-dependent paths.** The Kokoro graph accepts forced durations so the decoder can be
  tested independently of duration flips; `repeat_interleave` must reject `frames ≠ Σdur` rather than pad.
- **Never silently widen a tolerance.** A failing stage test is a bug or a documented numerical difference.
- **Untrusted input is tested as such.** [built] The `.lkm` loader, text and dim parsers (`tests/unit/test_binary.cpp`,
  `test_text.cpp`, `test_dim.cpp`) and the remote protocol get fuzz-style cases (truncation, overflow, out-of-range
  enums, bad ids) that must return errors, never crash.
  Remote protocol: `tests/unit/test_remote_fuzz.cpp` (`[remote][fuzz]`, in the unit suite on the Mac and on the
  phone) drives the real server session (`lk::remote::serveConnection`, the code of `lokinfer-serve` and LokiHost)
  over a socketpair: malformed headers (bad magic/version, > 2 GiB, a 2 GiB claim then a hang-up with a footprint
  check, truncated header), ~50 hand-written hostile requests (every message type: bad shas, path traversal, hash
  mismatch, dtype 9 / 256, rank 17, negative / overflowing / `INT64_MAX` dims, shape↔nbytes mismatches both ways,
  truncated data, scalar byte counts, unknown model ids, generated-input and comparison abuse), 3000 seeded random /
  mutated frames, and 400 corrupted model blobs that are loaded and, when the loader and verifier accept them, run.
  After each, the session must still serve a valid RUN correctly. `tests/python/test_remote.py::
  test_hostile_frames_against_lokinfer_serve` repeats a few against a real `lokinfer-serve` process. Bugs it found
  (fixed): non-UTF-8 GPU_INSPECT names aborted the process inside Metal (`newFunction(nil)`); a header claiming 2 GiB
  made the server allocate 2 GiB before reading anything; dtype values ≥ 256 wrapped to F32 before the range check; a
  0-d tensor's byte count was not checked against its dtype.

## On the iPhone [built; run on the phone 2026-11-02]

`scripts/ios_deploy.sh` runs the C++ suite on the phone and then keeps the remote executor up.

**Python graph suite on the phone** (`tests/conftest.py`): `pytest --device=iphone` (or `LOKINFER_DEVICE=iphone`;
the UDID comes from `LOKINFER_IPHONE` or `devicectl`). Every compiled graph — `CompiledModel.__call__`, which also
serves eager kernel calls — runs on the phone through the remote executor: the model is uploaded once by content
hash, and the phone's outputs are what the test asserts on (they are written into the result tensors the same graph
produces on the Mac, so tests see ordinary `lokinfer.Tensor`s). A test that ran no graph on the phone (tensor
interop, save/load, registry) is reported **skipped**, never passed; `tests/tools` and `tests/perf` are Mac-only.
References stay on the Mac (PyTorch, float64 numpy, `~/.cache/lokinfer/<model>/ref/`). First phone run
(2026-11-02): sweeps + remote + MobileNetV2 + MNIST + GPU depthwise 62 passed / 1 skipped (Mac-only), Kokoro app59f
(CPU + GPU stages, free-running durations, SNR) 5 passed. `--device=serve` runs the same remote path against a local
`lokinfer-serve`, to validate it without the phone.

**On-device reference sweeps** (`tests/python/test_device_sweeps.py`, 26 cases): one graph computes an optimized
kernel and its oracle side by side — the fused MobileNetV2 block vs conv 1×1 → depthwise 3×3 → conv 1×1 (CPU), the
GPU depthwise conv vs the CPU one, the GPU `gemm_bias` (small-M, steel, tiled) vs the basic CPU GEMM — with inputs
**generated on the device from a seed** and the two outputs **compared on the device** (RUN2): only the small model
and a few numbers cross the cable. They run against a local `lokinfer-serve` by default, the phone with
`--device=iphone`.

**Test data crosses the cable once** (RUN2, `include/lokinfer/remote/server.h`): `RemoteDevice.put_tensor(x)`
stores an input or reference as a content-addressed blob (sha256 of the encoded tensor; the phone keeps blobs in
Application Support across sessions and reinstalls, like models; the client remembers which it already sent),
`run2(mid, [Blob, Generated(shape, seed), array], compare=[(output, Blob or other output, rtol, atol)],
return_outputs=False)` runs on them and returns per-output statistics computed on the device (mismatches at
`atol + rtol·|ref|`, worst index, max error, sums of squares for relative RMS / SNR). `lokinfer.remote.generate`
reproduces the device's generator (splitmix64 + Box-Muller) in numpy, so a Mac-side reference for generated inputs
needs no transfer either.

## Where things are

| what | where |
|---|---|
| C++ unit + E2E | `tests/unit/*.cpp`, `scripts/test.sh` |
| speed regression | `tests/perf/` (opt-in, `LOKINFER_PERF_TESTS=1`) |
| pass tests | `tests/lit/<pass>/*.lkir` |
| Python API | `tests/python/` |
| kernel sweeps | `tests/kernels/` |
| model stage/E2E | `tests/models/` |
| references | `models/<model>/reference.py` → `~/.cache/lokinfer/<model>/ref/` |
| remote protocol fuzzing | `tests/unit/test_remote_fuzz.cpp`, `tests/python/test_remote.py` |
| device parameter, on-device sweeps | `tests/conftest.py`, `tests/python/test_device_sweeps.py` |
