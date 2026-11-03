# M2 — conv proof (results)

Milestone M2 de-risks the one kernel family the Kokoro goal depends on: the generator's 1-D convolutions (92 % of
the model's MACs). Measured through the kernel library only (no runtime), Kokoro rand510 shapes (F = 1101),
f32, M2 Max, Low Power Mode off, 2026-11-01. Benchmark: `bench/kernels/conv1d_cpu.cpp`
(`build-release/lokinfer_bench_conv1d_cpu`). The table below was re-measured on 2026-11-02 (LPM off, AC, thermal
nominal; ≥ 1 s warm-up, 5 × ≥ 0.4 s repetitions, median), raw JSON `bench/kernels/results_conv1d_cpu_m2table.json`
(exactly these rows: `--benchmark_filter='^BM_(Im2col|Shifted)/(128/132121/11/5|256/22020/7/3)/|^BM_Streams/.*/8/|^BM_TiledIm2col/.*/8/(512|2048)/'`).
The older `bench/kernels/results_conv1d_cpu.json` (2026-11-01) holds the d1/d5 im2col / shifted rows and the 2-8
stream rows, not the tiled ones; its stage-1 k11 d5 im2col (97 ms) and shifted (54 ms) agree with the re-run.

## CPU

Shapes: stage 1 = 128 → 128 channels, L = 120F + 1 = 132 121; stage 0 = 256 → 256, L = 20F = 22 020.
"Same" padding, stride 1. GFLOP/s = 2·Co·Ci·K·L / time.

| candidate | stage 1 k11 d5 | stage 0 k7 d3 |
|---|---|---|
| im2col + one Accelerate sgemm (Accelerate threading) — the basic kernel | 90.3 ms (0.53 TF) | 22.4 ms (0.90 TF) |
| shifted GEMMs: one sgemm per tap over a padded input, no im2col | 52.5 ms (0.91 TF) | 14.0 ms (1.44 TF) |
| shifted, time-split over 8 single-threaded streams | 57.8 ms (0.82 TF) | 11.9 ms (1.70 TF) |
| **time-tiled im2col (512-sample tiles, L2-resident) over 8 single-threaded streams** | **22.1 ms (2.15 TF)** | **11.2 ms (1.81 TF)** |
| time-tiled im2col, 2048-sample tiles | 141 ms (0.34 TF) | 66.5 ms (0.30 TF) |

(2026-11-01 values, same order: 97 / 54 / 66 / 23 (2.04 TF) / 149 ms and 22 / 14 / 12 / 11.7 (1.73 TF) / 65 ms.)

Findings:
- **The winner is time-tiled im2col with many single-threaded Accelerate streams.** Per tile the column block
  `[Ci·K, 512]` (≈ 2.9 MB at stage 1) stays in the core's share of the 16 MB L2, and the GEMM's inner dimension is
  Ci·K = 1 408 instead of Ci = 128, which is what Accelerate/AMX needs to run fast. Tiles of 2 048 overflow L2 and
  collapse to 0.3 TF.
- The research claim "3 concurrent single-thread sgemms = 3.4 TF" does not apply to the per-tap GEMMs (too thin),
  but many streams over L2-sized work do reach ≥ 2 TF. On plain sgemm shapes the claim holds out of LPM too (see
  "Multi-stream AMX, out of Low Power Mode" below: 3 streams = 3.39 TF, 6 streams = 4.70 TF).
- **Exit criterion met for the CPU** (≥ 1.8 TF on the heavy stage-1 shapes). PyTorch runs the same convs at
  0.5–0.8 TF (research-kokoro.md §4.4), so the conv share of the model can be ~2.5–4× faster than PyTorch's.

Consequences for M3 (CPU):
- `conv1d_fused` = time-tiled im2col over a thread pool, tile ≈ 512 (tune per K and device), single-threaded
  Accelerate per tile. The fused prologue (AdaIN affine + Snake/leaky) is applied **while building the column
  block** (it's already a per-element pass over the tile), and the epilogue (bias, residual, instance-norm partial
  stats) runs on the output tile while it's in cache. Fusion costs nothing extra in memory traffic.
- Estimated conv time at rand510: 24 stage-1 convs ≈ 24 × 7–23 ms (k3/7/11) + stage-0 ≈ ... → **≈ 0.4–0.5 s**,
  vs PyTorch's ≈ 1.3 s for the same convs. The CPU target (beat torch.compile CPU 2 067 ms; eager 2 280 ms) has
  wide margin.

## Multi-stream AMX, out of Low Power Mode (re-verification, 2026-11-02)

The research note (`research-kernels.md`, measured in LPM) claimed 1.0–1.1 TF per single-threaded Accelerate sgemm,
≈ 2.0 TF with Accelerate's own threading and **3.4 TF from 3 concurrent single-threaded sgemms**. Re-measured with
`bench/kernels/amx_bench.cpp` (`build-release/lokinfer-bench-conv amx 2 3 4 6 8`; LokiHost `--bench-amx` runs the
same code on the phone). M2 Max, LPM off, AC, thermal nominal. Protocol: all streams start together, ≥ 1 s warm-up,
then ≥ 30 calls and ≥ 2 s per stream, median per stream; aggregate = sum of the per-stream rates (streams with a
starved neighbour finish their window later, so uneven rows are an upper-bound estimate). GFLOP/s:

| GEMM (M×N×K) | 1 stream (UI QoS) | 1 stream (background QoS) | Accelerate threaded | 2 streams | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| 512×768×768 (the research shape) | 1548 | 1528 | 2728 | 3094 | **3386** | 3404 | **4704** | 3978 |
| 1024×1024×1024 | 1368 | 1372 | 2659 | 2754 | 2971 | 3009 | 3009 | 2866 |
| 128×512×1408 (Kokoro stage-1 k11 conv tile) | 1466 | 1467 | 2198 | 2920 | 3213 | 3738 | **3886** | 3661 |
| 59×2304×768 (ALBERT QKV at app59f) | 1204 | 1169 | 1888 | 2394 | 2553 | **3042** | 1912 | 1483 |

Findings:
- **The finding holds out of LPM, and the absolute numbers are higher**: one stream 1.2–1.55 TF (LPM: 1.0–1.1),
  Accelerate's threading 1.9–2.7 TF (LPM: 2.0), **3 streams 3.39 TF on the research shape (LPM: 3.4)**, and 6
  streams reach 4.7 TF there. Concurrent single-threaded streams beat Accelerate's own threading by 1.1–1.8× on
  every shape: this stays the center of the CPU strategy (it is what `conv1d_fused` / the wino convs do on the
  thread pool).
- Two streams scale perfectly (2 P-clusters, one AMX each). The 3rd and 4th streams mostly time-share: one stream
  drops to ≈ 0.2–0.3 TF while the others keep ≈ 1.5 TF, so the aggregate gain from 2 → 4 streams is 10–30 %. Six
  streams balance evenly (≈ 0.78 TF each on 512×768×768).
- Small-M GEMMs (59×2304×768) peak at 4 streams and collapse beyond (1.5 TF at 8): don't over-split short
  sequences.
- The "E-core AMX" oddity reproduces: a single stream at QoS background runs as fast as at user-interactive
  (e.g. 1467 vs 1466 GF/s). Not verified which cluster it ran on (no per-core sampling here); a likely reading is
  that an idle machine does not confine background work to the E-cluster, so QoS only matters under contention.
- iPhone 18 Pro Max (A19 Pro, SME): **needs an iPhone run** (`--bench-amx`, built into LokiHost; the phone was
  unavailable on 2026-11-02).

## GPU (M2 Max) — in progress

Benchmark: `bench/kernels/conv1d_gpu.{metal,cpp}` (`build-release/lokinfer_bench_conv1d_gpu`, GPU time = one
command buffer of 10 dependent dispatches / 10; output checked against a direct CPU computation).

| candidate | stage 1 k11 d5 | stage 0 k7 d3 |
|---|---|---|
| implicit-GEMM simdgroup_matrix f32 v1 (64×64×16 tiles, 24 KB threadgroup memory) | 72 ms (0.67 TF) | 31 ms (0.65 TF) |
| v2: coalesced tile loads, BK = 32, 1 KB edge scratch (two threadgroups per core) | 58 ms (0.82 TF) | 25 ms (0.81 TF) |
| PyTorch MPS conv1d on the same shapes (research-kokoro.md §4.4, LPM) | 5.2 TF | 4.75 TF |

Variants tried on the hand-written kernel (all correct, all ≈ 0.4–1.1 TF): BK 8/16/32, padded tile strides,
`max_total_threads_per_threadgroup(128)`, a no-threadgroup-memory "direct" variant (fragments straight from
device memory: 0.4 TF), with/without `-frecord-sources -gline-tables-only` (no consistent effect).

**Reference measurements (same machine, same day):**
| kernel | shape | time | TF |
|---|---|---|---|
| MLX `conv1d` (steel implicit GEMM, channel-last `[L, C]`) | stage 1 k11 d5 | 7.3 ms | 6.5 |
| MLX `conv1d` | stage 0 k11 d5 | 5.0 ms | 6.4 |
| MLX matmul (steel) | 128 × 132 121 × 1 408 | 6.5 ms | 7.3 |
| **MLX's steel GEMM kernel compiled into lokinfer, run through lokinfer's Metal 4 runtime** | 128 × 132 096 × 1 408 | 5.45 ms | **8.7** |

**Conclusion.** lokinfer's runtime path (Metal 4 queue, wrapped shared buffers, param ring, barriers) is not the
bottleneck: MLX's kernel runs *faster* through it than through MLX. The hand-written kernel is. Decision:
- **vendor MLX's steel kernels** (MIT, © Apple; GEMM loaders/MMA, implicit-GEMM conv) under
  `third_party/mlx_steel/` with the license, and build lokinfer's GPU GEMM/conv on them;
- **channel-last `[L, C]` layout for the GPU frame-rate path** (MLX and kokoro.cpp both use it; steel's conv
  loaders expect it);
- lokinfer's fused prologue/epilogue (AdaIN affine + Snake/leaky on load, bias/residual/stats on store) is
  added on top of steel's MMA core — that fusion is what MLX lacks (MLX's Kokoro spends most of its 858 ms
  outside the convs).

**Result with vendored steel** (`bench/kernels/conv1d_steel.metal`: MLX's `implicit_gemm_conv_2d<float, 64, 64,
16, 2, 2>` driven through lokinfer's runtime as a 1-D conv, channel-last `[L, Ci]` input, `[Co, K, Ci]` weights,
output checked against a CPU reference):

| shape | time | TF |
|---|---|---|
| stage 1 k3 d1 | 2.08 ms | 6.24 |
| stage 1 k7 d3 | 4.57 ms | 6.64 |
| stage 1 k11 d5 | 7.06 ms | **6.75** |
| stage 0 k3 d1 | 1.36 ms | 6.39 |
| stage 0 k7 d3 | 3.41 ms | 5.92 |
| stage 0 k11 d5 | 5.09 ms | 6.24 |

**GPU exit criterion met (≥ 4 TF)**, on par with MLX's own conv1d. At this rate the generator's ≈ 1.57 TFLOP cost
≈ 0.24 s on the M2 Max GPU (MLX's whole Kokoro forward at 510 tokens: 858 ms), so the GPU target has room;
the remaining work is fusion (AdaIN/Snake prologue on the steel loader, bias/residual/stats epilogue), the LSTMs
and the F readback.

Note: Xcode updated itself to 27.2 during the session; the Metal Toolchain is per Xcode version and had to be
re-downloaded (`xcodebuild -downloadComponent MetalToolchain`).

## LSTM (placement input for M3/M4)

Basic `lstm_bidir_f32_cpu` (input projection as one sgemm, both directions run one after the other, Accelerate
sgemv per step) vs `torch.nn.LSTM` on CPU, f32, M2 Max:

| T (steps) × input | lokinfer basic | per step (both dirs) | torch CPU |
|---|---|---|---|
| 130 × 640 | 4.1 ms | 31.5 µs | 5.0 ms |
| 512 × 640 | 14.6 ms | 28.6 µs | 18.4 ms |
| 1101 × 640 (the shared LSTM over F) | 31.9 ms | 29.0 µs | 40.2 ms |

All six Kokoro BiLSTMs at 510 tokens ≈ 105 ms with the basic kernel. Fused plan (M3): one thread per direction,
`W_hh` (1 MB) L2-resident, NEON/AMX matvec, gates fused: research target ≤ 5 µs/step/direction → ≈ 25 ms total.
GPU per-step dispatch alone costs ≈ 15 µs (research-metal-runtime.md), i.e. ≥ 33 ms for the shared LSTM before any
math, so the GPU path either uses a persistent one-dispatch-per-direction kernel or runs the LSTMs on the CPU
with explicit `ops.to_cpu`/`to_gpu` boundaries (each crossing ≈ 0.1–0.5 ms). Decided by measurement in M4.

## iPhone 18 Pro Max (2026-11-01, `LokiHost --bench`, Release; thermal/LPM not yet recorded by the host)

Same code as the Mac (`bench/kernels/conv_bench.cpp`), same rand510 shapes:

| shape | CPU tiled-im2col (best streams) | GPU steel conv |
|---|---|---|
| stage 1 k3 d1 | 9.03 ms, 1.44 TF (6) | 4.92 ms, 2.64 TF |
| stage 1 k7 d3 | 17.10 ms, 1.77 TF (6) | 11.33 ms, 2.68 TF |
| stage 1 k11 d5 | 24.57 ms, 1.94 TF (6) | 17.73 ms, 2.69 TF |
| stage 0 k3 d1 | 4.37 ms, 1.98 TF (6) | 3.29 ms, 2.63 TF |
| stage 0 k7 d3 | 9.70 ms, 2.08 TF (6) | 7.60 ms, 2.66 TF |
| stage 0 k11 d5 | 16.36 ms, 1.94 TF (4) | 11.92 ms, 2.66 TF |

- The CPU is close to the M2 Max: about 1.86 TF average vs 2.16 TF, ratio **0.86**. The best results use 6 streams,
  so the E-cores contribute: AMX throughput per stream, not the core count, is the limit.
- The GPU is f32 steel without tensor ops: about 2.66 TF vs 6.54 TF on the M2 Max, ratio **0.41**. Metal 4
  `matmul2d` in f16 (M5) is the main lever here.

### Artificial iPhone target (user decision: Mac "beat PyTorch" bar scaled by measured device peaks)

The conv share dominates (92 % of MACs), so the measured conv throughput ratio is the scale:

| | Mac bar | ratio | **iPhone target** |
|---|---|---|---|
| CPU rand510 | 2067 ms (torch.compile) | 0.86 | **≤ 2400 ms** |
| CPU real117 | 799 ms (eager) | 0.86 | **≤ 930 ms** |
| GPU rand510 | 858 ms (MLX) | 0.41 | **≤ 2090 ms** |
| GPU real117 | 193 ms (MLX) | 0.41 | **≤ 470 ms** |

The GPU targets are conservative: f16 tensor ops on the A20 should beat the f32 ratio. Refine them if a later
measurement (bandwidth, dispatch cost, tensor ops) shows a different ratio.

Open (M2, not yet run):
- tile size and stream sweep: partial Mac results below; finish it, and run it on the iPhone.
- Thermal state and Low Power Mode in the iOS host's bench output.
- iPhone GPU: Metal 4 tensor ops (`matmul2d` f16 with f32 accumulate).

### CPU tuning sweep (Mac, partial, 2026-11-01; LPM off; `lokinfer-bench-conv tune`)

TF per tile × raw-thread streams, plus the runtime thread pool (8 workers, one task per tile, dynamic):

| shape | tile | 4s | 6s | 8s | 10s | 12s | **pool** |
|---|---|---|---|---|---|---|---|
| s1 k3 | 256 | 1.20 | 1.45 | 1.63 | 1.48 | 1.48 | 1.84 |
| s1 k3 | 384 | 1.30 | 1.54 | 1.66 | 1.68 | 1.52 | 2.00 |
| s1 k3 | 512 | 1.13 | 1.37 | 1.52 | 1.45 | 1.17 | 1.54 |
| s1 k3 | 768 | 1.01 | 1.29 | 1.37 | 1.31 | 1.35 | **2.26** |
| s1 k3 | 1024 | 1.06 | 1.47 | 1.49 | 1.31 | 1.26 | 2.16 |
| s1 k7 | 256 | 1.41 | 1.71 | 1.92 | 1.99 | 1.77 | 2.25 |
| s1 k7 | 384 | 1.53 | 1.84 | 1.96 | 1.97 | 1.90 | 2.37 |

(The run was interrupted for the iPhone build. The `sudo xcodebuild -runFirstLaunch` install also ran during
the first rows, so the raw-thread columns read a little low.) Finding: **the persistent pool with dynamic tiles
beats raw threads with static strides** at every tile size, so M3's `conv1d_fused` uses the pool.

### Fused-conv tile sweep (M3, `bench/kernels/bench_conv_fused.py --sweep`, Mac)

The best tile puts the im2col block `[Ci·K, tile]` at **≈ 3 MB** for 6 of 7 Kokoro shapes (128 k7 → 768,
128 k11 → 512, 256 k7 → 384, 256 k11 → 256, 512 k3 → 512). Larger tiles collapse (256 k11 at tile 768: 0.97 TF
vs 2.42 TF at 256). Tiles above 768 never help, even at k3. Rule: `ops.conv_tile(ci, k)` = largest multiple of
128 with the block ≤ 3 MiB, clamped to [128, 768]. Fused resblock1 iteration vs the basic kernel chain:
4.20× / 3.73× / 3.47× (stage 1 k3/k7/k11), 2.56× (stage 0 k11).
