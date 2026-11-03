# Reference baselines: what they are and how they are measured

Every bar on the perf dashboard ([perf-dashboard.md](perf-dashboard.md), `perf/results.json`) whose impl does not
start with `lokinfer` (and is not `target`) is a **reference baseline**: the same model, run through another
framework, on the same device class. This page explains each one: what the framework is, where its copy of the
model comes from, how it is run and timed on each device, what makes it comparable, and how to reproduce it.

Hardware: **Mac** = M2 Max (`Mac CPU` / `Mac GPU`); **iPhone** = iPhone 18 Pro Max (iPhone19,3, iOS 27.0,
UDID `<udid>`; `iPhone CPU` / `iPhone GPU`). Everything is f32.

`<baselines>` is the separate checkout holding the baseline harnesses (key `baselines_dir` in `lokinfer.local.json`,
env `LOKINFER_BASELINES`); `<udid>`, `<team-id>` and `<bundle-id>` are the keys `iphone_udid`, `ios_team_id`,
`ios_bundle_id` ([benchmarking.md § Local configuration](benchmarking.md#local-configuration)).

Related pages: [impl-notes/iphone-baselines.md](impl-notes/iphone-baselines.md) (iPhone baselines in full detail),
[impl-notes/research-models.md](impl-notes/research-models.md) §8 (non-Kokoro Mac baselines, result table),
[impl-notes/research-kokoro.md](impl-notes/research-kokoro.md) §5 (Kokoro framework survey),
[impl-notes/kokoro-port-plan.md](impl-notes/kokoro-port-plan.md) (Kokoro targets).

## 1. Coverage (from `perf/results.json`, 2026-11-01)

Reference impls per model × device. `target` (an artificial goal, never a baseline) is not listed.

| model | Mac CPU | Mac GPU | iPhone CPU | iPhone GPU |
|---|---|---|---|---|
| Kokoro-82M | torch eager, torch.compile, ExecuTorch XNNPACK | torch MPS, torch.compile MPS, MLX, MLX Swift | ExecuTorch XNNPACK | MLX Swift |
| ResNet-18 | torch eager, torch eager tuned, torch.compile | torch MPS, torch.compile MPS, MLX | in progress (ExecuTorch XNNPACK) | none yet |
| MobileNetV2 | torch eager, torch.compile | torch MPS, torch.compile MPS, MLX | in progress (ExecuTorch XNNPACK) | none yet |
| DCGAN G | torch eager, torch eager tuned, torch.compile | torch MPS, torch.compile MPS, MLX | in progress (ExecuTorch XNNPACK) | none yet |
| DCGAN D | torch eager, torch eager tuned, torch.compile | torch MPS, torch.compile MPS, MLX | in progress (ExecuTorch XNNPACK) | none yet |
| GPT-2 124M | torch eager, torch.compile | torch MPS, torch.compile MPS, MLX | in progress (ExecuTorch XNNPACK) | none yet |
| MNIST MLP | torch eager | torch MPS | in progress (inputs prepared only) | none yet |

"In progress" = `<baselines>/other/` holds the shared inputs, weights and torch references for every
non-Kokoro model (`prep_data.py`) and ExecuTorch XNNPACK `.pte` programs for ResNet-18, MobileNetV2 and DCGAN G/D
(`export_et.py`); no iPhone harness run and no dashboard entries exist yet. No iPhone GPU reference exists for any
non-Kokoro model. The coverage rule (every model gets all four cards, each with measured baselines) is in
[perf-dashboard.md](perf-dashboard.md#card-headline-and-coverage).

**Not on the dashboard:** Core ML and ONNX Runtime have no entries. ONNX Runtime 1.31.1 was tried once for Kokoro
(research-kokoro.md §5.2: CPU EP ~2× slower than torch eager; CoreML EP fails to run the model) and not for the
other models (research-models.md §8: not installed). Core ML conversions of Kokoro are documented but not measured
(research-kokoro.md §5.3).

## 2. The apples-to-apples rule

A baseline number is recorded only if it does the same job as lokinfer:

1. **Same model and weights, f32.** Kokoro-82M v1.0 (`hexgrad/Kokoro-82M` @ `f3ff357`); torchvision
   `ResNet18_Weights.IMAGENET1K_V1`, `MobileNet_V2_Weights.IMAGENET1K_V2`; HF `hussamalafandi/DCGAN_CelebA` @
   `0c4e26b8`; HF `openai-community/gpt2` @ `607a30d7` (sdpa). Each framework's f32-ness is checked, not assumed
   (per-framework checks below).
2. **Exact reference inputs.** Kokoro: the cases of `models/kokoro/reference.py` (`R.load_case`: ids wrapped in
   `[0, …, 0]`, the voice row `ref_s`, `speed`). Other models: `np.random.default_rng(0)` inputs from
   `bench_other_baselines.py:make_inputs`.
3. **Same work.**
   - Kokoro: the output must be exactly the reference length, 600·F samples (same durations ⇒ same frame count ⇒
     same generator work). A framework that predicts its own durations may land one frame off: within 0.2 % the
     entry is recorded with the F difference in its notes, beyond that it is not recorded. The iPhone harnesses
     go further and **force** the reference durations.
   - Other models: the output is verified against torch eager CPU, rel err = max|y − ref| / max|ref| ≤ 1e-3,
     else not recorded (DCGAN D is verified on its pre-sigmoid logits, because the sigmoid saturates to 1.0).
4. **E2E per call.** One full forward per timed call, median after warm-up, with the GPU synced inside the call.
5. **Machine state recorded.** Low Power Mode, power source and thermal state go into each entry's `state`; the
   dashboard flags ⚠ when LPM is on, on battery, or thermal is not nominal. A ⚠ number must be re-measured.

## 3. Frameworks

### 3.1 torch eager (CPU) and torch MPS (Mac GPU)

**What.** PyTorch eager mode: each op dispatched from Python to ATen kernels as it executes. CPU uses PyTorch's
own CPU kernels (8 intra-op threads on the M2 Max); `torch MPS` is the same eager model on the `mps` device
(Metal Performance Shaders / MPSGraph). torch **2.14.1** in lokinfer's `.venv`. Mac only.

**Kokoro** (`bench/models/bench_kokoro.py`). The model is the `kokoro` package's `KModel` (kokoro 0.7.16) loaded
from the hexgrad snapshot (`R.load_model()` on CPU; a second `KModel` moved to `mps`). The timed function
`torch_infer` is english-practice's exact `kokoro_model.infer()` math: build `input_ids` from the ids, BERT, duration
encoder, LSTM, durations (predicted by the model itself), alignment, F0/N, text encoder, decoder, `.cpu()` readback
(which also syncs MPS). torch eager CPU **is** the reference implementation, so its durations equal the reference
ones by construction. Timing (`bench_kokoro.timeit`): ≥ 2 calls and ≥ 1 s of warm-up, then ≥ 7 calls and ≥ 2 s
measured, median. The first call's output length is checked against the reference (`length_check`, the §2 rule) for
both CPU and MPS; MPS's own durations matched the reference on every case so far (torch.compile too).

**Other models** (`bench/models/bench_other_baselines.py`, research-models.md §8). Models built with torchvision /
the pytorch/examples DCGAN classes (strict `load_state_dict`) / HF `GPT2LMHeadModel`; `eval()`, `inference_mode`.
GPT-2 prefill uses `use_cache=False, logits_to_keep=1`; decode runs one token at position 512 against a 512-token
`DynamicCache` built untimed and cropped back after each step. Timing: 3 warm-ups, then ≥ 10 runs and ≥ 2 s (≤ 1000),
median of per-call times; `torch.mps.synchronize()` inside each MPS call; inputs and outputs stay on the device.

**MNIST MLP** (`bench/models/bench_mnist.py`). `mnist_mlp.torch_reference()`, inputs `default_rng(0)`,
`inference_mode`; each round = mean over N calls sized for ~0.2 s, median of 7 rounds after 3 warm-ups; MPS
synced per call.

### 3.2 torch eager tuned (Mac CPU)

torch eager with **NNPACK disabled** (`torch.backends.nnpack.flags(enabled=False)`) and the model and input in
**channels_last** (research-models.md §4.4). Only for ResNet-18 and DCGAN G/D: skipped for MobileNetV2 (its
depthwise convs get 3–8× slower) and GPT-2 (no convs). Same harness and verification as §3.1. It exists because
default eager hits pathological NNPACK paths (DCGAN D N=64: 429 ms eager vs 18.7 ms tuned).

### 3.3 torch.compile and torch.compile MPS

**What.** PyTorch 2's compiler (TorchDynamo graph capture + Inductor codegen), same torch 2.14.1. Mac only.

- **Kokoro** (`bench/models/bench_kokoro_baselines.py --compile`): `m.decoder` and `m.bert` wrapped in
  `torch.compile(..., dynamic=True)`, the rest eager (as measured in research-kokoro.md §5.2). The first call
  compiles (untimed) and its output length is checked against the reference (`length_check`); then the
  `bench_kokoro.timeit` protocol (≥ 3 calls and ≥ 1 s warm-up, ≥ 7 calls and ≥ 2 s measured, median).
- **Other models** (`bench_other_baselines.py`): `torch.compile(model)` default mode, same timing/verification as
  §3.1. Known anomalies kept as measured: MobileNetV2 N=1 on CPU (~280 ms, slower than N=8) and DCGAN D N=64 on CPU
  (still on the NNPACK path, ~430 ms).

### 3.4 MLX (Python, Mac GPU)

**What.** Apple's array framework for Apple silicon: lazy graph, evaluated on the Metal GPU by `mx.eval`;
`mx.compile` traces and fuses a function. **mlx 0.32.4**, run as a subprocess in a **separate venv** (mlx is not a
lokinfer dependency) pointed to by `LOKINFER_MLX_PYTHON`. Mac GPU only (MLX CPU is unusable for Kokoro:
research-kokoro.md §5.2).

- **Kokoro** (`bench_kokoro_baselines.py --mlx` → `bench/models/runners/kokoro_mlx.py`): the **mlx-audio** Kokoro
  port (`mlx_audio.tts.utils.load_model`), with weights in `LOKINFER_MLX_MODEL`: fp32 safetensors converted from
  the hexgrad pth (548 tensors). The conversion step itself is not scripted in the repo. mlx-audio takes a phoneme
  string, so the ids are mapped back through the model's own vocab (an exact inverse). Durations are **MLX's own**
  (not forced): every case matches the reference length except rand510, F = 1100 vs 1101, and s28f, F = 1047 vs
  1048 (−0.09 / −0.10 % work, noted on the entries). f32 check: every parameter is `mx.float32`. Timing: ≥ 2 calls and ≥ 1 s of warm-up, then ≥ 7 calls,
  stopping at ≥ 30 calls or ≥ 2 s, median; `mx.eval(audio)` in the call. `LOKINFER_MLX_PYTHON` (key `mlx_python`) /
  `LOKINFER_MLX_MODEL` point at a local MLX venv / fp32 MLX weights dir. Fidelity: correlation with torch ≈ 0.88 (research-kokoro.md §5.2: different numerics; mlx-audio also clamps
  durations to 100 frames/token).
- **Other models** (`bench_other_baselines.py` → `bench/models/runners/other_mlx.py`): **our own direct
  transcriptions** of the PyTorch models (same ops, unfused eval BatchNorm, NCHW in/out with the NHWC transposes
  inside the timed call; GPT-2 from HF's `model.safetensors`, KV cache concatenated without keeping). Weights are
  the torch state dict passed as `.npz`. Each case is timed plain and `mx.compile`d; both outputs are verified
  against torch eager CPU and the faster valid one is recorded (it was `mx.compile` everywhere; the notes give both
  times). Same timing loop as §3.1.

The MLX interpreter comes from `mlx_python` in `lokinfer.local.json` (or `LOKINFER_MLX_PYTHON`, see
[benchmarking.md § Local configuration](benchmarking.md#local-configuration)): point it at a venv with
`mlx==0.32.4` + numpy (+ `mlx-audio` for Kokoro).

### 3.5 ExecuTorch XNNPACK (Kokoro: iPhone CPU, Mac CPU)

**What.** ExecuTorch is PyTorch's on-device runtime. A model is captured ahead of time with `torch.export`, lowered
(`to_edge_transform_and_lower` with the `XnnpackPartitioner`), and serialized into a `.pte` program; at run time the
ExecuTorch C++ runtime loads the program and executes it without Python. **XNNPACK** is its CPU backend (delegate):
the partitioned subgraphs run as XNNPACK graphs, the remaining ops through ExecuTorch's kernel libraries
(`kernels_optimized` is linked). Unlike PyTorch eager there is no Python or per-op dispatcher at run time, shapes
are exported ahead of time, and activations are memory-planned when the program is built (for Kokoro, planned for
the largest shapes N = 512, F = 2400, hence a 5.6 GB peak on the iPhone).

**Version.** ExecuTorch **1.5.1**: prebuilt xcframeworks (`executorch`, `backend_xnnpack`, `kernels_optimized`,
`threadpool`) from `https://ossci-ios.s3.amazonaws.com/executorch/<name>-1.5.1.zip`, `ios-arm64` and `macos-arm64`
slices, used through the C++ `Module` API.

**Model.** The published export `lee-futo/Kokoro-82M-executorch` (HF), `xnnpack_fp32/kokoro.pte` (325 MB, exported
with ExecuTorch 1.5.1 from hexgrad @ `f3ff357`; export scripts in its `export/`). One program, two methods:
`predict(input_ids, ref_s, speed) → d, t_en, duration` and `vocode(d, t_en, aln, ref_s, rand_ini, sine_noise) →
waveform`. The export rolls the six LSTMs into dynamic-length `scan` loops, bakes weight norm into the weights,
takes the vocoder noise as inputs, and uses Kokoro's `disable_complex=True` STFT path (`CustomSTFT`).
f32 check (`scripts/et_dtype_check.py`): all 4446 XNNPACK values are `xnn_datatype_fp32`; non-delegated tensors are
only Float/Long/Int/Bool.

**Harness.** `<baselines>/executorch/kokoro_et.{h,cpp}`, shared by the Mac CLI (`mac_main.cpp`,
`build_mac.sh`) and the iOS app (`executorch/ios/`: `main.mm`, CMake → Xcode project `EtKokoro`). One timed call =
`predict` + host-built alignment from the **forced reference durations** + `vocode` (fed the reference SineGen
noise) + copy of the waveform into a host vector. It also checks that ExecuTorch's own predicted durations equal
the reference ones (they do, on every case), the output length (600·F), correlation/SNR vs the reference audio, and
memory footprint. CPU work is synchronous, so no extra sync is needed. Threads set via the threadpool
(`_unsafe_reset_threadpool`): iPhone 6 (best of a 1/2/3/4/6 sweep; default 2), Mac 8 (default; 12 within noise).
Timing (both devices, since 2026-11-01 evening): ≥ 2 calls and ≥ 1 s of warm-up, then ≥ 7 calls, stopping at ≥ 30
calls or ≥ 2 s, median (`EtBenchOptions`: `warmup`/`minWarmS`, `runs`/`maxRuns`/`minRunS`). Cross-check: `executorch/kokoro_et_py.py`
runs the same cases through pip `executorch==1.5.1` (`<baselines>/.venv-et`) with identical output.

**Caveats.** Fidelity vs the reference audio is correlation ≈ 0.926 / 0.915 / 0.930: this comes from the
`disable_complex=True` STFT path baked into the export, not the runtime (eager kokoro 0.9.4 with
`disable_complex=True` gives the same 0.926; ExecuTorch vs that eager model: 0.9999). Its own predicted durations
equal the reference ones on every case (the 7 app/word cases too). The Mac rows are measured under the Mac bench
lock like every other Mac row (until 2026-11-01 evening they were calibration runs on a shared Mac).

### 3.6 MLX Swift (Kokoro: iPhone GPU, Mac GPU)

**What.** MLX's Swift bindings (**mlx-swift 0.30.2**): the same lazy array runtime and Metal GPU kernels as Python
MLX, usable in an iOS app. The model is **KokoroSwift**, the community port in `mlalma/kokoro-ios` @ `4d6d1d8`,
**patched** by `<baselines>/patches/kokoro-ios-lokinfer-baselines.patch`:

- `KokoroTTS.forward(paddedIds:refS:speed:forcedDurations:)`: the model forward from token ids (G2P skipped). It
  still computes the predicted durations and reads them back (the same sync as the unpatched code), then builds the
  alignment on the host from the **forced reference durations** (own durations would give F = 95/367/1098 on
  iPhone vs reference 95/368/1101).
- `SineGen.forcedNoise`: feeds the reference SineGen noise (for the correlation check; no change in work).
- `weightDtypes`: reports the dtype of every loaded weight.

**Weights.** `kokoro-v1_0.safetensors` from `mlalma/KokoroTestApp` (LFS `Resources/`): 548 tensors, all float32,
**bit-identical** to the hexgrad pth (`scripts/check_mlx_weights.py`); the app prints 548/548 float32 at load.

**Harness.** `<baselines>/mlx-swift/`: `App/Bench.swift` (shared), `App/AppMain.swift` (iOS app
`MlxKokoro`), `MacCLI/main.swift` (`MlxKokoroMac`), XcodeGen `project.yml`. One timed call = `forward` +
`eval(audio)` + `asArray` readback (GPU work complete and copied to host). Timing: same protocol as ExecuTorch (≥ 2
calls and ≥ 1 s warm-up, ≥ 7 calls, stop at ≥ 30 calls or ≥ 2 s, median) on both devices. Memory: MLX's default
memory config for every case but iPhone rand510, which needs `MLX_MEMLIMIT_MB=5500` and `MLX_CACHE_MB=2500` (with the
default 10.9 GB limit the buffer cache outgrows the app's ~6 GB and jetsam kills it), so it runs in its own launch.
The harness empties MLX's buffer cache (`Memory.clearCache()`) before each case: without it, nine cases in one iPhone
launch grew the cache to 5.2 GB and jetsam killed the app after the sixth. KokoroSwift's own durations equal the
reference on every case except rand510 (F = 1100 vs 1101) and s28f (1047 vs 1048), on both devices; the harness forces the reference ones.

**Caveats.** The port's audio is offset ~250–300 samples (about half a frame) from the reference, so the raw
correlation is ~0 (lag-aligned per segment: 0.84–0.97 on real28).

## 4. How the iPhone numbers are taken

Both iOS harnesses are minimal UIKit apps (UIScene lifecycle, a single label) that read their command line at
launch, run the benchmark on a background thread, print `RESULT {json}` lines to the console and `exit()`.

| step | how |
|---|---|
| build | ExecuTorch: `cmake -G Xcode` (`executorch/ios/CMakeLists.txt`) then `xcodebuild`; MLX Swift: `xcodegen generate` then `xcodebuild` (SwiftPM's CLI cannot build MLX's Metal shaders). Release, `-allowProvisioningUpdates` |
| sign | Automatic signing, team `<team-id>`, bundle id **`<bundle-id>`** (LokiHost's App ID): each app installs as an update of LokiHost, so no new App ID is used and the data container survives |
| entitlement | `com.apple.developer.kernel.increased-memory-limit` in both apps (ExecuTorch is killed by jetsam during the first forward without it; with it `os_proc_available_memory` ≈ 6.1 GB). The free team accepts it |
| inputs | `scripts/prep_cases.py` writes each reference case to `cases/<case>.kkc` (ids, `ref_s`, speed, reference durations, `rand_ini`, reference SineGen noise, reference audio). `xcrun devicectl device copy to --domain-type appDataContainer --domain-identifier <bundle-id>` puts the model and cases in `Documents/baselines/` |
| launch | `pkill -f lokinfer-ios-watchdog` first (LokiHost's relauncher would fight for the bundle id), then `xcrun devicectl device process launch --console --terminate-existing <bundle-id> -- --et-bench W R THREADS cases…` or `-- --mlx-bench W R FORCE_DUR FORCE_NOISE cases…` |
| state | Each `RESULT` carries `state_before` / `state_after` (thermal state, Low Power Mode, power from the battery state); the idle timer is disabled. Recorded numbers: LPM off, AC, thermal nominal before and after every case, except iPhone s28f (ExecuTorch) and s20f/s28f (MLX Swift), which ended at `fair` (impl-notes/iphone-baselines.md) |
| record | save the console to `<baselines>/results/<name>.log`; `scripts/record_baselines.py` asserts the length checks, takes the device state from `state_after` (Mac: from a `perfdb.machine_state()` JSON line appended to the log) and calls `perfdb.record` with the version/method notes |
| restore | `scripts/ios_deploy.sh` reinstalls LokiHost |

lokinfer depends on none of this: frameworks, apps, downloads and patches all live in `<baselines>/`.

## 5. Reproducing

Run from the repo root with Low Power Mode off, on AC, on a quiet machine. Each command overwrites only the
entries it measures.

| numbers | command |
|---|---|
| Kokoro torch eager + torch MPS | `.venv/bin/python bench/models/bench_kokoro.py [case …]` (no case = every dashboard case; also re-measures lokinfer basic/fused unless `--no-lokinfer`; `--no-mps`) |
| Kokoro torch.compile (CPU + MPS) | `.venv/bin/python bench/models/bench_kokoro_baselines.py --compile [case …]` |
| Kokoro MLX | `[LOKINFER_MLX_PYTHON=<mlx venv python> LOKINFER_MLX_MODEL=<fp32 mlx weights dir>] .venv/bin/python bench/models/bench_kokoro_baselines.py --mlx [case …]` |
| other models, all Mac impls | `.venv/bin/python bench/models/bench_other_baselines.py [--model resnet18\|mobilenetv2\|dcgan_g\|dcgan_d\|gpt2] [--impl NAME] [--no-record]` (MLX needs `LOKINFER_MLX_PYTHON`) |
| MNIST torch eager + MPS | `.venv/bin/python bench/models/bench_mnist.py` (also re-measures the lokinfer bars) |
| Kokoro ExecuTorch XNNPACK, MLX Swift (iPhone + Mac) | the command block in [impl-notes/iphone-baselines.md § Reproducing](impl-notes/iphone-baselines.md#reproducing), then `.venv/bin/python <baselines>/scripts/record_baselines.py et-iphone\|et-mac\|mlx-iphone-f32\|mlx-mac LOG…` |

Mac CLIs of the iPhone harnesses (run each under `scripts/lk-lock /tmp/lokinfer-macbench.lock`, append the
`perfdb.machine_state()` JSON to the log, then `record_baselines.py et-mac|mlx-mac LOG`):

```sh
B=<baselines>
$B/executorch/build_mac.sh && $B/executorch/kokoro_et_mac $B/downloads/executorch-kokoro/xnnpack_fp32/kokoro.pte \
  $B/cases 2 7 0 $CASES                                  # min-warmup min-runs threads(0 = default) cases
cd $B/mlx-swift && xcodegen generate && \
  xcodebuild -project MlxKokoro.xcodeproj -scheme MlxKokoroMac -configuration Release -derivedDataPath build/dd build
DYLD_FRAMEWORK_PATH=$PWD/build/dd/Build/Products/Release/PackageFrameworks build/dd/Build/Products/Release/MlxKokoroMac \
  $B/KokoroTestApp/Resources/kokoro-v1_0.safetensors $B/cases 2 7 1 1 $CASES
```

with `CASES` = the dashboard's Kokoro cases (`perfdb.DEFAULT_META`, e.g. `"word real28 app47f … s20f s28f rand510"`;
each must first be exported by `scripts/prep_cases.py`).

## 6. Where the code lives

| path | role |
|---|---|
| `bench/models/bench_kokoro.py` | Kokoro torch eager / MPS (english-practice's `infer()` math, `length_check`), plus lokinfer CPU |
| `bench/models/bench_kokoro_baselines.py` | Kokoro torch.compile (CPU/MPS) and MLX, with the output-length rule |
| `bench/models/runners/kokoro_mlx.py` | mlx-audio Kokoro runner (separate venv) |
| `bench/models/bench_other_baselines.py` | ResNet-18, MobileNetV2, DCGAN G/D, GPT-2: all torch impls + MLX, with verification |
| `bench/models/runners/other_mlx.py` | MLX transcriptions of those models (separate venv) |
| `bench/models/bench_mnist.py` | MNIST torch eager / MPS |
| `bench/perfdb.py` | `record` / `machine_state` into `perf/results.json` |
| `models/kokoro/reference.py` | Kokoro reference cases (ids, voice row, speed, durations, noise, audio) |
| `<baselines>/executorch/` | ExecuTorch harness, Mac CLI, iOS app (`ios/`) |
| `<baselines>/mlx-swift/` | MLX Swift harness, iOS app, Mac CLI |
| `<baselines>/kokoro-ios/`, `patches/` | patched KokoroSwift and the patch |
| `<baselines>/downloads/` | ExecuTorch 1.5.1 xcframeworks, the Kokoro `.pte` + its export scripts, kokoro 0.9.4 (fidelity check) |
| `<baselines>/scripts/` | `prep_cases.py`, `record_baselines.py`, f32 checks (`et_dtype_check.py`, `check_mlx_weights.py`), fidelity checks (`et_selfcheck.py`, `k094_check.py`) |
| `<baselines>/results/` | raw console logs of every recorded iPhone/Mac harness run |
| `<baselines>/other/` | in progress: non-Kokoro iPhone baselines (`prep_data.py`, `export_et.py`, `data/`) |

## 7. Known caveats on the current numbers

- The **MNIST** torch entries were recorded with `state.power = battery` (LPM off, thermal nominal), so the dashboard
  flags them ⚠; they need a re-run on AC. (The Kokoro Mac rows are all on AC, nominal.)
- Non-Kokoro Mac CPU rows vary by up to ~±15 % between runs (other sessions benchmarking concurrently; the harness
  waits up to 2 min for other processes < 200 % CPU and writes the load it saw into the notes). GPU rows: ~±5 %.
- Fidelity differs by framework even though the work is the same: ExecuTorch ≈ 0.93 correlation (export's STFT
  path), MLX Swift offset by ~half a frame, mlx-audio ≈ 0.88. These are timing baselines, not numerical references;
  the numerical reference is always torch eager CPU.
