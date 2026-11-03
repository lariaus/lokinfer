# iPhone reference baselines (ExecuTorch XNNPACK, MLX Swift)

**[built, measured 2026-11-01]** Reference timings of Kokoro-82M through other frameworks on the iPhone 18 Pro Max
(iPhone19,3, iOS 27.0), with M2 Max calibration runs of the same harnesses. lokinfer does **not** depend on any of
this: the frameworks, the measurement apps, the downloads and the patches all live outside the repo, in
`<baselines>/`. Only the numbers land here, in `perf/results.json` (impls `ExecuTorch XNNPACK` and
`MLX Swift`, see [perf-dashboard.md](../perf-dashboard.md)).

`<baselines>`, `<udid>`, `<team-id>` and `<bundle-id>` below are the local values of `baselines_dir`, `iphone_udid`,
`ios_team_id` and `ios_bundle_id` ([benchmarking.md § Local configuration](../benchmarking.md#local-configuration)).

## Results (median ms, E2E full forward)

| device | case | T | F | ExecuTorch XNNPACK (CPU) | MLX Swift (GPU) |
|---|---|---|---|---|---|
| iPhone | word | 9 | 55 | 110.7 | 81.5 |
| iPhone | real28 | 31 | 95 | 181.5 | 135.2 |
| iPhone | app47f | 47 | 118 | 228.7 | 171.9 |
| iPhone | app47m | 47 | 137 | 267.2 | 194.3 |
| iPhone | app59f | 59 | 147 | 288.9 | 210.3 |
| iPhone | app59m | 59 | 163 | 325.0 | 231.3 |
| iPhone | app71f | 71 | 174 | 349.6 | 253.5 |
| iPhone | app71m | 71 | 188 | 361.8 | 277.6 |
| iPhone | s6f | 92 | 238 | 449.8 | 354.3 |
| iPhone | s6m | 92 | 237 | 464.2 | 349.6 |
| iPhone | s9f | 141 | 328 | 679.4 | 494.6 |
| iPhone | s9m | 141 | 320 | 671.9 | 480.3 |
| iPhone | real117 | 126 | 368 | 734.2 * | 535.7 * |
| iPhone | s20f | 334 | 794 | 1812.3 | 1218.9 † |
| iPhone | s28f | 487 | 1048 | 2619.2 † | 1632.8 † |
| iPhone | rand510 | 512 | 1101 | 2682.4 * | 1705.0 * |
| Mac (M2 Max) | word | 9 | 55 | 222.0 | 46.0 |
| Mac (M2 Max) | real28 | 31 | 95 | 367.3 | 77.9 |
| Mac (M2 Max) | app47f | 47 | 118 | 458.4 | 94.9 |
| Mac (M2 Max) | app47m | 47 | 137 | 524.9 | 105.3 |
| Mac (M2 Max) | app59f | 59 | 147 | 573.0 | 115.6 |
| Mac (M2 Max) | app59m | 59 | 163 | 627.8 | 123.2 |
| Mac (M2 Max) | app71f | 71 | 174 | 672.4 | 134.4 |
| Mac (M2 Max) | app71m | 71 | 188 | 721.0 | 139.6 |
| Mac (M2 Max) | s6f | 92 | 238 | 885.9 | 175.9 |
| Mac (M2 Max) | s6m | 92 | 237 | 868.7 | 174.4 |
| Mac (M2 Max) | s9f | 141 | 328 | 1196.4 | 242.7 |
| Mac (M2 Max) | s9m | 141 | 320 | 1177.7 | 237.9 |
| Mac (M2 Max) | real117 | 126 | 368 | 1382.8 | 258.4 |
| Mac (M2 Max) | s20f | 334 | 794 | 2855.0 | 581.1 |
| Mac (M2 Max) | s28f | 487 | 1048 | 3815.1 | 774.9 |
| Mac (M2 Max) | rand510 | 512 | 1101 | 4283.8 | 835.1 |

ExecuTorch: 6 threads on the iPhone, the default 8 on the Mac. Every recorded number: Low Power Mode off, on AC,
thermal `nominal` before and after the case, except the three marked †, which ended at `fair` (s20f / s28f were each
measured in two or three sessions, every one starting `nominal` after a cool-down; the phone reached `fair` during
the long cases each time, and the fastest run is the one recorded; MLX Swift moves < 1 % between `fair` and
`nominal`, e.g. s6f 350.5 fair vs 354.3 nominal). Mac rows: measured 2026-11-01 evening under the Mac bench lock with the
current timing protocol (below). iPhone rows: measured 2026-11-01 evening with the current protocol, except the four
marked *, which are the earlier nominal runs (3 warm-ups + median of 7): the re-runs of real117 / rand510 with the new
protocol came at the end of a session, at thermal `fair`/`serious` and slower (ExecuTorch 773 / 2789 ms, MLX Swift
542 / 2070 ms), so `perfdb.record` kept the nominal ones. Thermal: the phone was shared with other sessions and
started each session hot (`fair`/`serious`); a cool-down of up to 5 min reached `nominal` each time, and it went
back to `fair` within about 2 min of CPU or GPU runs, so app71f/app71m were re-run in a short third session
(nominal). Logs: `results/et_iphone_v2.log` + `_v4` (app71m), `mlx_iphone_v3.log` + `_v4` (app71f/m),
`et_mac_v2.log`, `mlx_mac_v2.log`; s6/s9/s20/s28: `et_iphone_long.log` + `et_iphone_long2.log` (s20f/s28f),
`mlx_iphone_long3*.log`, `et_mac_v3.log`, `mlx_mac_v3.log`.

**iPhone MLX Swift runs with `MLX_ENABLE_TF32=0` (true f32), re-measured 2026-11-01.** On the A19 GPU, MLX ≥ 0.30 runs
f32 matmuls as TF32 by default (see the "Other models" section below). The first iPhone MLX run used that default;
the recorded numbers above are the f32 rerun, with everything else identical (forced durations, forced noise, same
cases, same memory settings, 3 warm-up + median of 7, one launch per case, thermal `nominal` before and after each).
TF32 is only 1 to 5 % faster:

| case | f32 (`MLX_ENABLE_TF32=0`, recorded) | TF32 (MLX default, first run) | TF32 speedup |
|---|---|---|---|
| real28 | 134.8 | 133.3 | 1.01× |
| real117 | 535.7 | 516.6 | 1.04× |
| rand510 | 1705.0 | 1630.1 | 1.05× |

A same-session TF32 control on real117, run right after the f32 runs (thermal `fair`), gave 519.3 ms, so the gap is
real and not drift. The Mac rows are unaffected: the M2 Max has no NAX and MLX stays f32 there.

## What makes these apples to apples

- **Same model and precision.** Kokoro-82M v1.0 (hexgrad, rev `f3ff357`), f32 everywhere:
  - ExecuTorch: every XNNPACK value in both methods is `xnn_datatype_fp32` (902 in `predict`, 3544 in `vocode`), and
    the non-delegated tensors are only Float, Long, Int and Bool. There is no fp16, bf16 or quantized tensor
    (`scripts/et_dtype_check.py`).
  - MLX: the `kokoro-v1_0.safetensors` file holds 548 tensors, all float32, and is **bit-identical** to the hexgrad
    pth (max |diff| = 0, `scripts/check_mlx_weights.py`). The app also prints the dtypes it loaded (548/548 float32).
- **Same inputs.** `scripts/prep_cases.py` writes each case from `R.load_case(name)` into a flat `.kkc` file: the ids
  wrapped in `[0, ..., 0]`, the `ref_s` row, `speed`, the reference `pred_dur`, the reference SineGen noise, and the
  reference audio.
- **Same work.** Durations are **forced** to the reference `pred_dur` in both harnesses, so `F` and the output length
  (600·F samples) equal the reference exactly. This is checked on every run.
  - ExecuTorch takes the alignment `aln` as an input. Its own predicted durations also equal the reference ones in
    every case.
  - MLX Swift computes its own durations. The patch makes it compute them, read them back (the same sync the
    unpatched code does), then build the alignment from the reference durations. In the f32 runs its own durations
    equal the reference on every case except rand510 (F = 1100 vs 1101) and s28f (1047 vs 1048), both devices; the first iPhone run, with
    MLX's default TF32, gave F = 95 / 367 / 1098 against the reference 95 / 368 / 1101.
  - The reference SineGen noise is fed to both (`sine_noise` input; `SineGen.forcedNoise` patch). This changes no
    work.
- **Full forward, E2E.** Each timed call runs the predictor (BERT, duration encoder, LSTM, durations), the host
  alignment, F0/N, the text encoder, the decoder and the generator, then reads the audio back into host memory.
  ExecuTorch also prints its predict/vocode split (iPhone rand510: 218 + 2458 ms).
- **Timing protocol** (both harnesses, both devices, since 2026-11-01 evening; the same as lokinfer's): at least 2
  warm-up calls and at least 1 s of warm-up, then at least 7 measured calls, stopping once 30 calls or 2 s have been
  measured; median. (The first measurements used 3 warm-ups + median of 7.) iPhone sessions: one cool-down to
  `nominal` (at most 5 min, read from LokiHost's remote state) before the session, then at most a 60 s break between
  the two implementations, never per case.

## Fidelity (output vs the PyTorch reference audio, same noise, same durations)

- **ExecuTorch: correlation 0.926 / 0.915 / 0.930** (SNR about 5 dB). This is not a runtime error. The published
  export bakes in Kokoro's `disable_complex=True` path (`CustomSTFT`). Eager kokoro 0.9.4 with
  `disable_complex=True` gives the same 0.926 on real28 against our reference (kokoro 0.7.16, `TorchSTFT`), while
  `disable_complex=False` gives 0.99998. ExecuTorch against that eager `disable_complex=True` model correlates at
  0.9999. The STFT done as a conv1d does the same order of work.
- **MLX Swift: raw correlation about 0.** The port's audio is offset by about 250 to 300 samples (roughly half a
  frame) from the reference, and the offset varies along the utterance. After aligning the lag per segment, the
  correlation is 0.84 to 0.97 (real28, Mac), and the magnitude spectrogram correlation is 0.957. research-kokoro.md
  §2.5 #5 already noted that the MLX port is not numerically faithful (0.88 for mlx-audio).

## Fairness notes

- **Threads (ExecuTorch).**
  - On the iPhone, the ExecuTorch default threadpool is 2 threads (the P-cores). A sweep on real117 gave 1: 1059,
    2: 866, 3: 765, 4: 758, 6: 736 ms. The recorded numbers use 6 threads, the best setting.
  - On the Mac, the default is 8 threads. 12 threads were within noise of 8, and 4 threads were 1.6× slower.
- **Memory.**
  - The ExecuTorch program plans its activations for the largest shapes (N = 512, F = 2400). The footprint is 2.5 GB
    after load and peaks at **5.6 GB**. Without the `com.apple.developer.kernel.increased-memory-limit` entitlement,
    jetsam kills the app (SIGKILL) during the first forward. With it, `os_proc_available_memory` is about 6.1 GB.
    The free team accepts this entitlement.
  - The MLX Swift harness empties MLX's buffer cache (`Memory.clearCache()`) before each case. Without it, the nine
    small cases in one launch grew the cache to 5.2 GB and jetsam killed the app after app59m (2026-11-01).
  - MLX's buffer cache grows past the app's limit on rand510 under the defaults (s20f and s28f are run the same way:
    one launch each with the rand510 memory settings) (`memoryLimit` = 10.9 GB, 1.5× the
    recommended working set). The recorded rand510 run sets `Memory.memoryLimit` = 5500 MB and `cacheLimit` =
    2500 MB through env vars (TF32 run: 1630 ms; a 1024 MB cache gave 1645 ms). With a 1024 MB cache, real117 (TF32)
    slows from 517 to 573 ms, so real28 and real117 are recorded with the defaults (MLX peak 0.7 and 1.6 GB). The
    cache sweep was done with TF32 on and was not repeated in f32.
- **Glue.** The alignment is built on the host in both harnesses (as KokoroSwift does), and it is included in the
  time.

## Reproducing

Everything is under `<baselines>/`:

| path | what |
|---|---|
| `downloads/executorch-kokoro/` | `lee-futo/Kokoro-82M-executorch` (HF): `xnnpack_fp32/kokoro.pte` (325 MB, exported with ExecuTorch 1.5.1), metadata, export scripts |
| `downloads/executorch-1.5.1/` | `executorch`, `backend_xnnpack`, `kernels_optimized`, `threadpool` xcframeworks from `https://ossci-ios.s3.amazonaws.com/executorch/<name>-1.5.1.zip` (the S3 prebuilts that ExecuTorch's `swiftpm-*` branches point at; there is no `swiftpm-1.5.1` branch, but the 1.5.1 zips exist) |
| `downloads/kokoro-0.9.4/` | the kokoro 0.9.4 wheel, unpacked (only used for the fidelity cross-check) |
| `cases/*.kkc` | the exported reference cases (`scripts/prep_cases.py`) |
| `executorch/kokoro_et.{h,cpp}` | ExecuTorch harness (C++ `Module` API), shared by the Mac CLI (`mac_main.cpp`, `build_mac.sh`) and the iOS app (`executorch/ios/`, CMake Xcode generator) |
| `executorch/kokoro_et_py.py` | the same run through the pip `executorch==1.5.1` runtime (cross-check: identical output) |
| `kokoro-ios/` | clone of `mlalma/kokoro-ios` @ `4d6d1d8` (mlx-swift 0.30.2), **patched**: `patches/kokoro-ios-lokinfer-baselines.patch` |
| `KokoroTestApp/` | clone of `mlalma/KokoroTestApp` (only its LFS `Resources/kokoro-v1_0.safetensors` is used) |
| `mlx-swift/` | MLX harness (`App/Bench.swift`), iOS app shell (`App/AppMain.swift`) and Mac CLI (`MacCLI/main.swift`), XcodeGen `project.yml` |
| `scripts/record_baselines.py` | parses the `RESULT {json}` lines of `results/*.log` and calls `perfdb.record` |
| `results/*.log` | raw console logs of every recorded run |

Both iOS apps are signed as **`<bundle-id>`** (team <team-id>), the LokiHost App ID. They install
as updates of LokiHost, so no new App ID is used and the data container survives. Afterwards, `scripts/ios_deploy.sh`
puts LokiHost back. Inputs go to the container's `Documents/baselines/` (`kokoro.pte`,
`kokoro-v1_0.safetensors`, `cases/`), and they are still there.

```sh
B=<baselines>; U=<udid>; L=$PWD   # run from the lokinfer repo root
CASES="word real28 app47f app47m app59f app59m app71f app71m s6f s6m s9f s9m real117 s20f s28f rand510"  # dashboard cases
SMALL="word real28 app47f app47m app59f app59m app71f app71m s6f s6m s9f s9m real117"  # MLX: default memory, one launch
BIG="s20f s28f rand510"   # MLX on the iPhone: one launch each with MLX_MEMLIMIT_MB=5500 MLX_CACHE_MB=2500
.venv/bin/python $B/scripts/prep_cases.py $CASES
pkill -f lokinfer-ios-watchdog   # LokiHost's relauncher would fight for the bundle id
cp() { xcrun devicectl device copy to --device $U --domain-type appDataContainer \
       --domain-identifier <bundle-id> --source "$1" --destination "Documents/baselines/$2"; }

# ExecuTorch (CPU)
cd $B/executorch/ios && cmake -S . -B build -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0
xcodebuild -project build/EtKokoro.xcodeproj -scheme EtKokoro -configuration Release -destination id=$U \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device $U build/Release-iphoneos/EtKokoro.app
cp $B/downloads/executorch-kokoro/xnnpack_fp32/kokoro.pte kokoro.pte
for c in $CASES; do cp $B/cases/$c.kkc cases/$c.kkc; done
xcrun devicectl device process launch --device $U --console --terminate-existing <bundle-id> \
  -- --et-bench 2 7 6 $CASES                            # min-warmup min-runs threads cases... -> results/et_iphone_v2.log
$B/executorch/build_mac.sh && $B/executorch/kokoro_et_mac $B/downloads/executorch-kokoro/xnnpack_fp32/kokoro.pte \
  $B/cases 2 7 0 $CASES                                 # Mac CPU (0 = default threads) -> results/et_mac_v2.log

# MLX Swift (GPU). Patch already applied in $B/kokoro-ios; xcodebuild is required (SwiftPM CLI can't build MLX's Metal shaders)
cd $B/mlx-swift && xcodegen generate
xcodebuild -project MlxKokoro.xcodeproj -scheme MlxKokoro -configuration Release -destination id=$U \
  -allowProvisioningUpdates -derivedDataPath build/dd build
xcrun devicectl device install app --device $U build/dd/Build/Products/Release-iphoneos/MlxKokoro.app
cp $B/KokoroTestApp/Resources/kokoro-v1_0.safetensors kokoro-v1_0.safetensors
xcrun devicectl device process launch --device $U --console --terminate-existing \
  --environment-variables '{"MLX_ENABLE_TF32":"0"}' <bundle-id> \
  -- --mlx-bench 2 7 1 1 $SMALL                         # min-warmup min-runs force_durations force_noise cases...
                                                        # -> results/mlx_iphone_v3.log
xcrun devicectl device process launch --device $U --console --terminate-existing \
  --environment-variables '{"MLX_ENABLE_TF32":"0","MLX_MEMLIMIT_MB":"5500","MLX_CACHE_MB":"2500"}' \
  <bundle-id> -- --mlx-bench 2 7 1 1 $c   # for c in $BIG -> results/mlx_iphone_v3_rand510.log,
                                                             #   mlx_iphone_long2_s20f.log, mlx_iphone_long2_s28f.log
xcodebuild -project MlxKokoro.xcodeproj -scheme MlxKokoroMac -configuration Release -derivedDataPath build/dd build
DYLD_FRAMEWORK_PATH=$PWD/build/dd/Build/Products/Release/PackageFrameworks build/dd/Build/Products/Release/MlxKokoroMac \
  $B/KokoroTestApp/Resources/kokoro-v1_0.safetensors $B/cases 2 7 1 1 $CASES   # -> results/mlx_mac_v2.log

# record: each mode takes the console logs to read (Mac logs end with a perfdb.machine_state() JSON line; run the
# Mac CLIs under scripts/lk-lock /tmp/lokinfer-macbench.lock)
R=".venv/bin/python $B/scripts/record_baselines.py"
cd $L && $R et-iphone et_iphone_v2.log && $R et-mac et_mac_v2.log && \
  $R mlx-iphone-f32 mlx_iphone_v3.log mlx_iphone_v3_rand510.log && $R mlx-mac mlx_mac_v2.log

# afterwards: put LokiHost back
scripts/ios_deploy.sh
```

Known gotchas:
- The ExecuTorch xcframework headers live under `Headers/ExecuTorch/`, but they include
  `<executorch/...>`. The harness symlinks `inc-{ios,macos}/executorch` to that directory and builds with
  `-Wno-nonportable-include-path`.
- `KokoroSwift` is a dynamic SwiftPM product, so the XcodeGen dependency needs `embed: true`. Without it, dyld
  fails at launch.
- The Mac CLI needs `DYLD_FRAMEWORK_PATH=.../PackageFrameworks`.
- ExecuTorch's threadpool header isn't shipped. The harness declares
  `executorch::extension::threadpool::get_threadpool()` / `_unsafe_reset_threadpool()` itself, and the symbols
  are in `libthreadpool`.

# Other models: ResNet-18, MobileNetV2, DCGAN G/D, GPT-2 124M, MNIST MLP

**[built, measured 2026-11-01]** The same two reference bars (`ExecuTorch XNNPACK` on "iPhone CPU", `MLX Swift`
on "iPhone GPU") for every other dashboard model, so each model has all four device cards. They use the same
weights, inputs and cases as the Mac baselines (`bench/models/bench_other_baselines.py`; MNIST MLP:
`models/mnist_mlp.py` + `bench/models/bench_mnist.py`). The apps and scripts live in `<baselines>/other/`.

## Results (iPhone 18 Pro Max, median ms, f32, LPM off, on AC)

| model | case | ExecuTorch XNNPACK (CPU, threads) | MLX Swift (GPU) |
|---|---|---|---|
| ResNet-18 | N=1 224x224 | 2.930 (6) | 2.698 |
| ResNet-18 | N=8 224x224 | 18.67 (6) | 14.31 |
| MobileNetV2 | N=1 224x224 | 1.491 (2) | 3.186 |
| MobileNetV2 | N=8 224x224 | 7.917 (6) | 20.26 |
| DCGAN G | N=1 | 1.738 (6) | 0.462 |
| DCGAN G | N=64 | 52.53 (6) | 11.44 |
| DCGAN D | N=1 | 0.192 (1) | 0.648 |
| DCGAN D | N=64 | 10.66 (6) | 6.157 |
| GPT-2 124M | prefill T=128 | 26.81 (2) | 12.75 |
| GPT-2 124M | prefill T=512 | 164.1 (6) | 41.77 |
| GPT-2 124M | decode P=512 | 8.691 (2) | 6.013 |
| MNIST MLP | B=1 | 0.0017 (2) | 0.139 |
| MNIST MLP | B=64 | 0.0121 (1) | 0.144 |
| MNIST MLP | B=1024 | 0.173 (1) | 0.264 |

Every entry is verified on the phone against the torch eager CPU reference (rel err = max|y - ref| / max|ref|).
ExecuTorch is at most 4.1e-4 (GPT-2) and at most 5e-6 elsewhere. MLX Swift is at most 5.3e-6. Every number is the
median of at least 10 runs and at least 2 s, after 3 warm-up runs.

## What makes these apples to apples

- **Same weights and inputs.** `other/prep_data.py` runs in lokinfer's `.venv` and imports `load_model`,
  `make_inputs`, `torch_runner` and `logits_variant` from `bench_other_baselines.py`. It writes the f32 state dicts
  (`<model>.safetensors`; GPT-2 is the HF file), the `default_rng(0)` inputs and the torch eager CPU outputs. For
  GPT-2 decode it also writes the 512 context ids and torch's KV cache.
- **ExecuTorch (CPU).**
  - `other/export_et.py` (in `.venv-et`: torch 2.15, executorch 1.5.1) builds one program per case with
    `torch.export` + `XnnpackPartitioner`, using `to_edge_transform_and_lower`.
  - The conv nets are the torchvision / pytorch-examples classes, with the same weights.
  - GPT-2 is a direct transcription of HF `GPT2LMHeadModel`. On the Mac it matches HF to 0 / 4e-7. Decode takes the
    512-token cache as 24 inputs and concatenates the new K/V, as torch's DynamicCache and the MLX runner do.
  - Graph inputs are not memory-planned (`alloc_graph_input=False`). The harness binds them once, so no input copy
    is timed.
  - f32 check (`scripts/et_dtype_check.py`): every XNNPACK value is `xnn_datatype_fp32`. All other tensors are
    Float, Long or Bool only.
  - DCGAN D is verified with a second program, the same graph minus the final Sigmoid (`*.logits.pte`), as on the
    Mac.
- **MLX Swift (GPU).**
  - `other/mlx/App/Bench.swift` is a line-by-line Swift port of `bench/models/runners/other_mlx.py`. MNIST is
    Flatten + matmul/bias/relu.
  - Like the Mac MLX baseline, each case is timed plain and with `compile()`, and the faster verified one is
    recorded. `eval` (GPU sync) runs inside each timed call.
  - The weights are checked to be all float32 at load.
- **`MLX_ENABLE_TF32=0` is required on the iPhone.** On the A19's GPU (NAX, "neural accelerators"), MLX ≥ 0.30 by
  default runs float32 matmuls as **TF32**. With the default, ResNet-18 N=8 gave rel err 2.2e-3, MNIST B=1024
  2.0e-3 and GPT-2 T=128 1.0e-3, which fails verification. The M2 Max has no NAX and stays f32 (≤ 5e-6). The
  recorded runs set `MLX_ENABLE_TF32=0` (true f32). For reference, TF32 is about 1.3× faster on GPT-2 prefill and
  the same elsewhere (`results/other_mlx_iphone_tf32default.log`).
  The Kokoro `MLX Swift` iPhone entries above were first measured with the default (TF32) and have since been
  re-measured and recorded with `MLX_ENABLE_TF32=0` (TF32 was 1 to 5 % faster there).
- **Threads (ExecuTorch).** Every case was swept at 1, 2, 4 and 6 threads. The recorded run is the fastest one that
  was thermal-nominal before and after (2-thread default; 6 wins on the larger convs and GPT-2 T=512, and 1 or 2 on
  the tiny ones). The sweep is in each entry's notes.
- **Thermal.** The phone is on AC and heats to `fair` within a few minutes of CPU runs. ExecuTorch is up to about
  1.5× slower at `fair` (GPT-2 decode 8.7 → 13.5 ms), so the CPU entries are all nominal runs. MLX is within 1 %
  between `fair` and `nominal`. Some GPU entries (DCGAN D N=64, GPT-2, MNIST) ended at `fair`, which their state
  records.

## Reproducing

```sh
B=<baselines>; O=$B/other; U=<udid>; L=$PWD   # run from the lokinfer repo root
.venv/bin/python $O/prep_data.py           # data/: weights, inputs, refs, cases.json
$B/.venv-et/bin/python $O/export_et.py                         # data/<id>.pte + <id>.et.txt (verifies via pip runtime)
$O/et/build_mac.sh && $O/et/other_et_mac $O/data 1 2 0 0 <ids> # Mac verification through the same C++ harness
cd $O/mlx && xcodegen generate && xcodebuild -project OtherMlx.xcodeproj -scheme OtherMlxMac -configuration Release \
  -derivedDataPath build/dd -clonedSourcePackagesDirPath build/spm build
DYLD_FRAMEWORK_PATH=$PWD/build/dd/Build/Products/Release/PackageFrameworks build/dd/Build/Products/Release/OtherMlxMac \
  $O/data 1 2 0 <ids>                                           # Mac verification of the MLX Swift ports
# iOS apps (bundle id <bundle-id>, install over LokiHost; pkill -f lokinfer-ios-watchdog first)
cd $O/et/ios && cmake -S . -B build -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 && \
  xcodebuild -project build/OtherEt.xcodeproj -scheme OtherEt -configuration Release -destination generic/platform=iOS \
  -allowProvisioningUpdates build
cd $O/mlx && xcodebuild -project OtherMlx.xcodeproj -scheme OtherMlx -configuration Release -destination generic/platform=iOS \
  -allowProvisioningUpdates -derivedDataPath build/dd -clonedSourcePackagesDirPath build/spm build
# copy every file of $O/data to Documents/baselines/other/ (devicectl device copy to ... --domain-type appDataContainer)
xcrun devicectl device process launch --device $U --console --terminate-existing <bundle-id> \
  -- --et-other 3 10 2 <threads> <ids>          # warmup min_runs min_seconds threads ids -> results/other_et_iphone_*.log
xcrun devicectl device process launch --device $U --console --terminate-existing \
  --environment-variables '{"MLX_ENABLE_TF32":"0"}' <bundle-id> \
  -- --mlx-other 3 10 2 <ids>                   # -> results/other_mlx_iphone_final.log
cd $L && .venv/bin/python $O/record_other.py   # (--dry-run to preview); then scripts/ios_deploy.sh
```

Case ids: `resnet18_n{1,8} mobilenetv2_n{1,8} dcgan_g_n{1,64} dcgan_d_n{1,64} gpt2_t128 gpt2_t512 gpt2_d512
mnist_b{1,64,1024}`. The phone's `Documents/baselines/other/` still holds the data, about 2.9 GB, mostly the three
622 MB GPT-2 programs. In those programs the tied `wte` weight is stored twice: once for the embedding, which is
not delegated, and once packed for XNNPACK's lm_head.
