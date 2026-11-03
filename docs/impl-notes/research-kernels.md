# Research: Kernel Techniques for lokinfer (CPU NEON/AMX/SME, GPU Metal)

Status: **research v1** (2026-11-01). This is input to the kernel design, the per-area "Optimizing X kernels" docs,
and the todo list. Tags: **[M]** = measured on the dev Mac in this session, **[S]** = from a source (URL given),
**[I]** = inference or arithmetic. Inferences lose to measurements by default. Re-measure before relying on them.

Benchmark sources (throwaway, in a local scratch directory, not in the repo):
(`peak.cpp`, `lat.cpp`, `gemm.cpp`, `bw2.cpp`, `bw.cpp`, `amxq.cpp`, `gpu.mm`, `fma.mm`, `mpsg.mm`).
Fold the useful ones into `bench/` when the benchmark harness exists.

---

## 0. Hardware facts and roofline references

### 0.1 Dev Mac (verified via sysctl / system_profiler) [M]

| Item | Value |
|---|---|
| Model | Mac14,5, **Apple M2 Max**, macOS 27.0.1, Apple clang 21 |
| CPU | 8 P (Avalanche) + 4 E (Blizzard). **P-cores = 2 clusters × 4**, each cluster with its own 16 MB L2 (`hw.perflevel0.cpusperl2=4`). E: 4 MB L2 |
| L1D | P 128 KB, E 64 KB. Cache line **128 B** |
| CPU ISA | FEAT_FP16, FHM (FMLAL), DotProd (SDOT), **I8MM (SMMLA)**, **BF16 (BFDOT/BFMMLA)**. **No SME/SME2** (`FEAT_SME*=0`, `sme_max_svl_b=0`) |
| GPU | **30-core** M2 Max GPU (not 38: this Mac is the 30-core bin), "Metal Support: Metal 4", `maxThreadgroupMemoryLength=32768`, recommendedMaxWorkingSet 26.8 GB |
| RAM | 32 GB LPDDR5, 400 GB/s nominal |

> **Benchmark hygiene finding [M]:** this Mac was in **Low Power Mode** (`pmset -g` → `lowpowermode 1`, on AC)
> during every measurement in this session. A Python process was also using ~2.5 cores and 3 GB.
> - Under Low Power Mode the P-cores measured **~2.3 GHz**, using a 1-cycle integer-add chain, instead of ~3.5–3.7 GHz.
> - **All CPU numbers below are therefore ~35% below nominal.** The GPU seemed mostly unaffected (FMA peak was near the theoretical value).
>
> The benchmark harness must record `pmset -g` lowpowermode, thermal state, and the load average into every result.
> It should also refuse or flag runs taken in Low Power Mode.
> (`NSProcessInfo.isLowPowerModeEnabled` and `thermalState` cover the same on iOS.)

### 0.2 Roofline numbers

**CPU, one P-core [M, at ~2.3 GHz LPM clock; scale ×1.55 for nominal 3.5 GHz]**

| Instruction | Throughput | Peak per core | Per cycle |
|---|---|---|---|
| FMLA f32 (4 lanes) | 9.3 Ginst/s | **74 GFLOPS** | 4.0 inst/cycle → 4 FMA pipes |
| FMLA f16 (8 lanes) | 9.3 Ginst/s | **148 GFLOPS** | 2× f32 |
| FMLAL f16→f32 | 9.3 Ginst/s | 74 GFLOPS | Same as f32. Its value is halved weight bytes, not FLOPs |
| BFDOT | 2.3 Ginst/s | 37.5 GFLOPS | **1/4 rate** |
| BFMMLA | 1.2 Ginst/s | 37.6 GFLOPS | **1/8 rate**, so bf16 on M2 is slower than f32 FMLA. Do not use bf16 compute on M2 |
| SDOT int8 | 9.4 Ginst/s | **300 GOPS** | Full rate |
| SMMLA int8 (I8MM) | 4.6 Ginst/s | **297 GOPS** | Half issue rate, double work. Same peak as SDOT. Its advantage is the 2×2 output layout, which suits interleaved weights |

- FMLA latency: **3.9 cycles**. You need **≥16 independent accumulators** to saturate the 4 pipes (12 acc → 3.0/cycle, 8 → 2.0/cycle).
- **Nominal P-core peak** (3.5 GHz): ~112 GFLOPS f32, ~224 f16, ~450 GOPS int8 [I].
- **E-core** (background QoS) [M]: FMLA f32 27 GF, f16 49 GF, SDOT 104 GOPS, SMMLA 125 GOPS. That is roughly 1/3 of a P-core.

**CPU memory [M]**

Read bandwidth, best of 5, `bw2.cpp`, each thread reads its own chunk of the working set:

| Working set | 1T | 2T | 4T | 8T | 12T |
|---|---|---|---|---|---|
| 64 KB (L1) | 98 GB/s | 200 | 409 | 506 | 684 |
| 1 MB (L2) | 64 | 128 | 254 | 603 | 692 |
| 12 MB (L2) | 60 | 123 | 254 | 363 | 450 |
| 32 MB (L2+SLC) | 57 | 96 | 144 | 256 | 330 |
| 96 MB | 52 | 74 | 112 | 126 | 125 |
| **512 MB (DRAM)** | **42** | 64 | 89 | 108 | **116** |

- In LPM, one core gets ~42 GB/s from DRAM. The whole CPU caps at ~116 GB/s.
- Sources give ~100 GB/s single-core on M1 Max and ~220–250 GB/s CPU total at nominal clocks. Re-measure out of LPM.
- **The CPU can never use the full 400 GB/s. The GPU can.**

**CPU via Accelerate (AMX) [M]**

`cblas_sgemm` f32 reaches **~1.0–1.1 TFLOPS on one thread** (`VECLIB_MAXIMUM_THREADS=1`) for M ≥ 64.
- About 2.0 TFLOPS multi-threaded (default).
- Three concurrent single-threaded sgemms give 3.4 TFLOPS aggregate.
- That is consistent with **3 AMX units (2 P-cluster + 1 E-cluster)** at ~1.1 TF each in LPM.
- **Oddity [M]:** an sgemm called from a `QOS_CLASS_BACKGROUND` thread (E-core) also ran at ~1.05 TF. Either the E-AMX is as fast as a P-AMX under LPM, or the work was not E-confined. Re-verify out of LPM.

**GPU (30-core M2 Max) [M]**

| Metric | Measured | Notes |
|---|---|---|
| FMA f32 | **9.8 TFLOPS** | 16 independent chains per thread |
| FMA f16 | 10.1 TFLOPS | **f16 is not 2× faster on M2.** Equal rate [S: philipturner/metal-benchmarks] |
| Theoretical | 30 × 128 × 2 × 1.398 GHz = **10.7 TFLOPS** | |
| DRAM read | **330–340 GB/s** | 256 MB–1 GB working set |
| Copy (r+w) | ~360 GB/s | |
| 32 MB working set (SLC, 48 MB) | ~415 GB/s | |

- **Ridge point**: ~10 TFLOPS / 340 GB/s ≈ **30 FLOP/byte** on GPU.
- On CPU the ridge is ~1 TFLOP/s ÷ 100 GB/s ≈ 10 FLOP/byte with AMX, and ~0.6 ÷ 100 ≈ 6 with NEON on 8 cores.
- Everything elementwise, norm, GEMV, or small-K is memory- or latency-bound.

### 0.3 iPhone 18 Pro Max (A20 Pro). Nothing is verified locally yet; measure first.

- **CPU**: reported as 2 P + 4 E [S: macworld, ilounge].
- **SME2**: A18 Pro and A19 Pro expose SME2 to user code (Geekbench feature strings list `sme2`, `sme-f32f32`, `sme-i8i32`) [S]. For the A20 Pro, I found **no primary confirmation**, only low-reliability spec sites. It is almost certainly present, since Apple doesn't drop ISA [I].
  - **Detect at runtime**: `sysctlbyname("hw.optional.arm.FEAT_SME2")` and `hw.optional.arm.sme_max_svl_b`. Both keys exist; the second is present on this M2 and returns 0.
- **GPU**: 7-core GPU with **per-core Neural Accelerators** (the A19 Pro / M5-style matmul units). Reported +50% memory bandwidth (96-bit bus) [S: notebookcheck, press]. Reached only through Metal 4 **TensorOps / MPP `matmul2d`** (§2.2).
- **First on-device task**: run the ported `peak`, `bw2`, `amxq`, `gpu`, `fma` micro-benchmarks inside the lokinfer test app, and record the numbers in this doc.

---

## 1. CPU kernels

### 1.1 Apple AMX (M1–M3; still present on M4)

**What it is**
- An undocumented matrix coprocessor, **one per CPU cluster**. The M2 Max has 2 P-AMX and 1 E-AMX.
- It does f32/f16/f64 outer products into a big register file, plus bf16 on M2+ and int8/16.
- Theoretical rate is 512 f32 FLOP/cycle → ~1.6 TF per P-cluster on M1 [S: scalable.uni-jena.de/opt/sme/intro.html].
- Measured: ~2.7 TF f32 whole-chip on M1 Max [S: philipturner/amx-benchmarks].
- Reverse-engineered in corsix/amx [S: https://github.com/corsix/amx], which says "not documented nor supported, use at own risk".

**How to reach it**
- `cblas_sgemm`/`cblas_sgemv` (Accelerate BLAS), BNNS, and **BNNSGraph** (macOS 15 / iOS 18).
- An arXiv paper on M1 found BNNSGraph the fastest Accelerate f32 GEMM path, but **bit-exact on only 3 of 12 shapes** (max err 1.4e-3) [S: arXiv 2606.25426].
- The same paper's hand-written AMX kernel beat `cblas_sgemm` by ~2× geomean. It found pre-packing constant weights and the right panel width to be the big levers.

**Threading**
- `VECLIB_MAXIMUM_THREADS`, or the newer thread-local `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` (macOS 15+/iOS 18+) [S: developer.apple.com/documentation/accelerate/blassetthreading(_:)].
- Apple's advice: use single-threaded BLAS when the app has its own threading.
- **Don't oversubscribe**: on M4 Pro, capping GEMM threads at the number of matrix units gave 2729 vs 1030 GFLOPS at 1024³ [S: Eigen MR !3164].

**Measured here [M] (LPM, f32, row-major NN)**

| Shape (M×N×K) | cblas default threads | cblas 1 thread | NEON 8×12 BLIS (1T) | naive autovec (1T) |
|---|---|---|---|---|
| 512×768×768 | 858–1985 GF | 973–1108 GF | 66 GF | 18 GF |
| 64×768×768 | 1093 | 907 | 60 | 18.5 |
| 512×3072×768 | 879 | 285 (noisy) | 62 | – |
| 200×1024×512 (LSTM W_ih over T) | 812 | 374 | – | 14 |
| 512×516×1536 (conv1d k3 im2col) | 1012 | 440 | 64 | 19 |
| 8×768×768 | 419 | 420 | – | – |
| 1×768×768 (GEMV via sgemm) | 137 | 145 | – | – |

- The noise between runs comes from LPM plus the background load. The ranking is not affected by it.
- **Conclusion [M]:** for f32 GEMM with M ≥ ~8 on M2, **Accelerate (AMX) is 10–30× faster than the best possible NEON on one core**.
  - The NEON 8×12 kernel is at 89% of the per-core FMLA peak.
  - Even 8 P-cores of perfect NEON (~600 GF at LPM clocks) lose to one AMX thread.
- Custom NEON GEMM on M2 is therefore only worth it for:
  - (a) fused prologues/epilogues where a separate pass costs more than the GEMM;
  - (b) tiny shapes where Accelerate call overhead dominates (measure; the crossover is probably M·N·K < ~64K);
  - (c) int8 / f16-weight GEMV (bandwidth-bound, see 1.5).
- **Default plan [I]:**
  - GEMM-shaped ops on M2 CPU go to `cblas_sgemm` (variant `gemm_f32_cpu_accelerate`), then a cheap fused NEON epilogue pass (bias + activation) over the output while it is still in L2.
  - Try BNNSGraph per shape, with a tolerance-based (not bit-exact) test.
  - Direct AMX instructions are **not** a default. They are undocumented, change per generation, and carry App Review risk on iOS. Keep them as a possible later experiment, Mac-only.

### 1.2 SME / SME2 (M4+, A18 Pro+, presumably A20 Pro)

**Hardware facts [S: scalable.uni-jena.de/opt/sme/micro.html; "Hello SME!" arXiv 2409.18779]**
- Streaming vector length 512-bit. ZA = 64×64 B (4 f32 tiles of 16×16).
- Measured M4 single P-core:

| Instruction | P-core | E-core |
|---|---|---|
| FMOPA f32 | **2.0 TFLOPS** | 0.36 |
| FMOPA f16→f32 | ~2.0 | – |
| BFMOPA bf16→f32 | ~2.0 | – |
| SMOPA i8→i32 | **4.0 TOPS** | – |
| Streaming-SVE FMLA (non-outer-product) | **31 GFLOPS** | – |
| NEON FMLA (for comparison) | 107 GFLOPS | – |

- Widening f16/bf16 FMOPA give no speedup over f32.
- One thread nearly saturates a cluster's unit: 2.34 TF with 8 threads. SME and AMX appear to share the unit on M4.
- All 4 ZA tiles must be used to reach peak.
- **Non-streaming SVE is not supported** (SIGILL). **NEON is unavailable inside streaming mode.** Apple's optimization guide says to use SSVE "only in a supporting role".

**Streaming-mode switch cost [S]**

FMOPA throughput by number of FMOPAs inside each `smstart/smstop` block:

| FMOPAs per block | GFLOPS |
|---|---|
| 8 | 434 |
| 32 | 1606 |
| 64 | 1891 |
| 128 | 1947 |

- That is roughly 30 cycles per switch pair [I].
- Switching zeroes Z/P registers. Save/restore FPSR.
- **Rule**: enter streaming mode once per GEMM call (a whole K-loop over many tiles), never per micro-tile. Do any NEON epilogue outside streaming mode, or with SSVE ops on ZA rows.

**Toolchain [S]**
- Clang ≥ 18 / current Xcode: `-march=armv9.2-a+sme2`, or per-function `__attribute__((target("sme2")))`.
- ACLE `<arm_sme.h>`: `__arm_streaming`, `__arm_locally_streaming`, `__arm_new("za")`, `__arm_inout("za")`, `svmopa_za32_f32_m`, ….
- Older SDKs failed to link with a missing `__arm_sme_state`.
- It is allowed in user code on macOS 15+ and iOS 18+ (Geekbench and KleidiAI ship it).

**Libraries**
- **KleidiAI** (Arm) ships SME2 micro-kernels: f32, f16, bf16, int8/int4 → f32 GEMM. XNNPACK dispatches to them when SME2 is detected; Gemma decode improved +53% on M4 Pro [S].
- Accelerate on M4 very likely uses SME, but Apple has not documented it.

**Plan [I]**
- The M2 Mac has no SME, so the SME kernels target the **iPhone**.
- Phase 1: `cblas_sgemm` on iOS too (Accelerate will pick AMX or SME).
- Phase 2: a hand-written SME2 FMOPA GEMM, built in a separate translation unit with `target("sme2")` and runtime-dispatched on `FEAT_SME2`. Compare it against Accelerate and KleidiAI on the phone. The fused-epilogue gain is the reason to own it.
- SME2 kernels cannot be tested on the M2 at all. The phone is the only test device. Correctness goes through the on-device reference sweep.

### 1.3 NEON GEMM micro-kernels (fallback / fused / small)

**Register blocking** (32 × 128-bit V registers):
- **f32 8×12**: 24 accumulators + 2 A registers (via `fmla v.4s, vB, vA.s[i]`) + 3 B registers, 29 in total. That is 24 FMAs per 5 loads, well inside 3 load ports.
- **Measured 66 GF = 89% of the per-core peak** with plain intrinsics, KC=256, MC=128, packed A and B [M].
- Alternatives:
  - 12×8 [S: BLIS `bli_sgemm_armv8a_asm_8x12`]
  - 16×4 or 4×16 (XNNPACK's `4x16`/`6x8` aarch64 neonfma ld128 kernels) for skinny shapes
- **f16**: the same 8×24 halves layout (24 accumulators of `.8h`). Expect ~2× f32 (148 GF/core in LPM). Accumulating in f16 has precision risk for K > ~512. Split K and accumulate partials in f32, or use FMLAL (f32 accumulate, f32 rate).
- **Packing**: Goto/BLIS loops. KC×NR B panels stay in L1 (KC·NR·4 = 12 KB at KC=256). The MC×KC A block stays in L2.
  - **For inference, pre-pack weights at init** (the compiler's init-hoisting pass is a natural place).
  - Packing is a kernel variant with its own layout (`gemm_f32_cpu_neon_packedB12`).
- **Edge handling**: masked, specialised kernels for M%8 and N%12 tails (or pad at pack time). Don't branch in the inner loop.
- **int8 (W8A8 / W4A8, for GPT-2 later)**:
  - SMMLA and SDOT give the same 300 GOPS/core peak on M2 [M].
  - Use the llama.cpp/KleidiAI layout: interleave 4–8 weight rows so one SMMLA produces a 2×2 output block [S: llama.cpp PR #5780 `Q4_0_4_8`, ggml-cpu aarch64 repack].
  - Quantise activations per row to int8 on the fly.
- **bf16**: M2 BFMMLA/BFDOT run at 1/4–1/8 rate [M]. Use bf16 only as a *storage* format, widened with shifts, never for compute on M2.

### 1.4 f16 storage strategy (CPU)

- **f16 compute (FMLA .8h)**: 2× f32 throughput, but needs care with precision. Kokoro's reference is f32.
- **f16 weights, f32 activations**: halves bandwidth.
  - `FMLAL`/`FMLAL2` with f16 x needs no convert instructions [M: FMLAL runs at full FMLA rate].
  - With f32 x, convert weights with `vcvt_f32_f16`. That costs an extra instruction per 4 lanes.
  - Measured [M], single thread: FMLAL GEMV (`gemv_rows_f16` in `bw.cpp`) 26 µs for 768×768, against 51 µs for f32 GEMV.

### 1.5 Batch-1 GEMV

**Measured [M]** (y = W·x with row-major W[N,K], LPM; `gemm.cpp`, `bw.cpp`)

| Shape (N×K) | W f32 bytes | cblas_sgemv (default) | NEON f32 8-row, 1T / 4T | NEON f16-W FMLAL, 1T / 4T |
|---|---|---|---|---|
| 768×768 | 2.4 MB | **11.5 µs** (206 GB/s) | 51 / 27 µs | 26 / – µs |
| 3072×768 | 9.4 MB | **55 µs** | 540 / 122 µs | 167 / 58 µs |
| 1024×256 (LSTM W_hh) | 1 MB | **7.9 µs** | 32 / 12.5 µs | 17 / 8.7 µs |
| 2304×768 | 7 MB | **33 µs** | 342 / 122 µs | 141 / 52 µs |
| 50256×768 (GPT-2 lm_head) | 154 MB | 3.6–4.2 ms (37–43 GB/s) | 13.6 / 3.5 ms | 5.8 / **1.6 ms** |

**Conclusions [M+I]**
- For **L2-resident** GEMV (≤ ~16 MB per cluster), **`cblas_sgemv` (AMX) beats single-thread NEON by 3–10×** and beats 4-thread NEON f32.
- My NEON GEMV is under-tuned:
  - only 46 GB/s from L1/L2, against a 98 GB/s L1 read peak;
  - it needs more rows in flight, software prefetch, and 128 B aligned row starts.
  - Treat these numbers as a floor.
- For **DRAM-bound** GEMV (lm_head), Accelerate seems to use one AMX and gets ~40 GB/s. **Multi-threaded NEON with f16 weights wins (1.6 ms vs 3.6 ms)**. It should approach the CPU fabric cap (~116 GB/s in LPM, ~200+ nominal) with all P-cores across **both clusters**.
- Kokoro is mostly not GEMV-shaped, except the LSTM recurrence (1.6). GPT-2 decode is GEMV-shaped and belongs on the GPU anyway (~350 GB/s).

### 1.6 LSTM on CPU (Kokoro: bidirectional, H≈256 per direction, batch 1)

The standard design [S: ONNX Runtime `deep_cpu_lstm.cc` / `uni_directional_lstm.cc`; oneDNN `brgemm` RNN]:
1. **One GEMM for the input projection over all T timesteps and both directions**: X[T, In] · W_ih[In, 8H] (N = 2 directions × 4 gates × H). That is exactly the AMX-friendly shape (812 GF measured at 200×1024×512).
2. **Recurrence**: per step, z = W_hh · h_{t−1} (1024×256 GEMV) + precomputed row t. Then the fused gates: σ/tanh with polynomial approximations, c and h update, all in registers.
3. **Run the two directions on 2 threads**, one per direction, with no per-step synchronisation. Ideally they sit on the same P-cluster, because each W_hh is only 1 MB f32 (0.5 MB f16) and stays hot in L2.

**Per-step cost and options**
- Cost is ~8 µs (cblas_sgemv), ~8.7 µs (4T NEON f16), ~17 µs (1T NEON f16) [M, LPM].
- Fusing the gates into the GEMV epilogue saves a pass. Target **≤ 5 µs/step** out of LPM.
- At T≈200 frames × 2 directions in parallel, that is ~1–2 ms per LSTM layer. **Measure Kokoro's actual LSTM count and T** in step 4 of the porting process.
- **Option**: cblas_sgemv for W_hh·h (AMX) plus a NEON gate pass. The AMX call overhead is a few µs; compare it against a fused NEON kernel.
- **Splitting one direction's GEMV across cores** needs a barrier every step. That only pays off with spin barriers between same-cluster threads, and a fork-join costs ~0.2–0.7 µs [M]. Try it later.

### 1.7 Conv1d / ConvTranspose1d / depthwise on CPU

**Conv1d (Kokoro: C=512, k=3/5/7/11, dilation 1/3/5, T up to many thousands in the vocoder)**
- **Implicit GEMM, "shift-and-accumulate"**: out = Σ_k W_k[Cout×Cin] · X[Cin, t + k·d]. That is K accumulating GEMMs with B = a shifted view of X.
  - No im2col buffer is needed, and dilation is free.
  - With Accelerate, each tap is one `cblas_sgemm` with beta=1 on a strided view. This works directly in [C, T] row-major layout (ld = T).
- **im2col + one GEMM** (K = Cin·k): a single big AMX call that is often faster than k separate calls. The im2col buffer is k× the input, which is fine at Kokoro sizes (512×3×5000×4 = 30 MB; chunk over T to stay in L2).
- Benchmark both. Measured: 512×516×1536 im2col GEMM = 1 TF [M].
- **Winograd 1D**: F(2,3) gives 1.5× fewer multiplies, F(4,3) 2×. It is numerically fragile in f16 and for k = 7/11. On M2, AMX f32 GEMM is so far above NEON that Winograd on NEON never wins. **Skip Winograd on CPU.**

**ConvTranspose1d (Kokoro/iSTFTNet upsamplers: strides 10 and 6, kernels 20 and 12, to verify against the config)**
- Use the **polyphase / sub-pixel decomposition**: stride s with kernel K becomes s ordinary convs with ⌈K/s⌉ taps in low-rate space, outputs interleaved [S: Shi et al. arXiv 1609.07009].
- That means no wasted MACs (zero-insertion wastes (s−1)/s, i.e. 90% at s=10) and no scatter conflicts.
- Alternative: GEMM X[T,Cin]·W[Cin, Cout·K] followed by a col2im overlap-add (one big AMX GEMM plus a cheap scatter-add). Benchmark both.

**Depthwise (MobileNetV2 3×3)**
- Memory-bound. Use direct NEON in **channels-last** (NHWC) layout, vectorised over channels (XNNPACK `dwconv` up-N×K style).
- Use the indirection-buffer IGEMM for regular convs [S: Dukhan, "The Indirect Convolution Algorithm"].
- ResNet-18 and DCGAN on CPU: NHWC + im2col/indirection → Accelerate GEMM.

### 1.8 Multithreading on macOS/iOS

**Placement controls**
- **There is no hard affinity on Apple Silicon.** `thread_policy_set(THREAD_AFFINITY_POLICY)` returns `KERN_NOT_SUPPORTED` [S: developer.apple.com/forums/thread/703361].
- **QoS is the only lever**:
  - `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` or `USER_INITIATED`: the scheduler prefers P-cores.
  - `QOS_CLASS_BACKGROUND` confines a thread to E-cores [S: eclecticlight 2025-05-09; M: the E-core peak numbers above were obtained exactly this way].
  - Don't set raw priorities: they opt the thread out of QoS.

**Apple's guidance** [S: "Tuning your code's performance for Apple silicon", tech talk 110147]
- Use `dispatch_apply` with ≥ 3× as many chunks as cores, so work-stealing absorbs P/E asymmetry.
- Avoid static partitioning if E-cores might participate.
- Avoid `sched_yield` (it can cost up to 10 ms) and spinning-with-yield.
- Pad shared data to 128 B.
- Work items of 4–20 µs are "barely longer than waking a core".

**Measured fork-join latency, empty job, warm, LPM [M]**

| Threads | `dispatch_apply` (UI QoS) | Spin pool |
|---|---|---|
| 2 | 0.40 µs | 0.17 µs |
| 4 | 0.44 µs | 0.74 µs |
| 8 | 0.53 µs | 3.7 µs (one thread spinning per P-core while the machine is loaded) |
| 12 | 0.77 µs | 1.4 µs |

- **`dispatch_apply` is very competitive.** Spin pools only win at 2 threads, and degrade badly once spinners exceed the free P-cores.

**How other frameworks handle it**
- pthreadpool (XNNPACK) uses **GCD on Apple** by default [S].
- llama.cpp's own pool polls, with a configurable poll level, then falls back to a condition variable.
- `os_sync_wait_on_address`/`os_sync_wake_by_address_any` (macOS 14.4+/iOS 17.4+) give a futex for spin-then-block barriers [S].
- `os_workgroup` (audio workgroups) only matters if TTS ever runs inside a real-time audio deadline.

**Recommendation**
1. Runtime option 1: monothread (the default for small graphs, per the user's plan).
2. Runtime option 2: **GCD `dispatch_apply` at USER_INTERACTIVE**.
3. Runtime option 3: a persistent pool of N ≤ #P-cores threads at USER_INTERACTIVE that spin for ~20–50 µs and then block via `os_sync_wait_on_address`. This approximates "pinned P-cores only".
4. Never spin between `model()` calls.
5. Accelerate calls inside multithreaded kernels must use `BLASSetThreading(SINGLE)` to avoid nested parallelism.
6. **M2 Max topology** matters for bandwidth-bound kernels: use both P-clusters, because each has its own L2 and fabric port. Barrier-heavy work does better on ≤ 4 threads.
7. **iPhone (2 P-cores)**: realistically 2 compute threads. Accelerate/SME does the heavy lifting.

### 1.9 vDSP / BNNS / FFT on CPU

- **vDSP DFT** supports only lengths f·2ⁿ with f ∈ {1,3,5,15}: n ≥ 3 for complex, n ≥ 4 for real.
  - **n=20 is not supported** [S: Apple docs `vDSP_DFT_zrop_CreateSetup`].
  - Kokoro's iSTFT (n_fft=20, hop=5) therefore uses a direct inverse-DFT matrix: a 22×20 real basis with the window folded in, applied as one small GEMM or a fused NEON loop over frames, then overlap-add. It is trivially cheap (88 FMAs per output sample).
- **BNNSGraph** (macOS 15 / iOS 18) compiles a Core ML `.mlmodelc`: fusion, single-thread mode, no-alloc execution with a workspace.
  - Apple claims ≥ 2× over legacy BNNS, and it targets real-time audio [S: WWDC24 10211].
  - Worth an experiment as a library-variant kernel for whole sub-blocks, e.g. the vocoder resblock. It is opaque and not bit-exact.

---

## 2. GPU kernels (Metal / MSL)

### 2.1 simdgroup_matrix GEMM (M2 path): MLX "steel" design [S: mlx/backend/metal/kernels/steel/gemm/*.h, matmul.cpp]

**Structure**
- Threadgroup tile BM×BN×BK, with WM×WN simdgroups (32 threads each). Each simdgroup owns TM×TN 8×8 fragments (TM = BM/(8·WM)).
- K-loop:
  1. barrier
  2. cooperative `BlockLoader` copies the A and B tiles device → threadgroup memory
  3. barrier
  4. `simdgroup_load` the fragments, then `simdgroup_multiply_accumulate`
- Single-buffered threadgroup memory.
- **Fragments are held in float (AccumType) even for half inputs**. On M1/M2 the f16 and f32 matrix rates are equal (~102 FFMA/core-cycle, ~80% of scalar peak) [S: metal-benchmarks].
- **Threadgroup padding**: 16 bytes per row (`16/sizeof(T)` elements) to avoid bank conflicts.
- **Function constants** `align_M/align_N/align_K`, `has_batch`, `use_out_source`, `do_axpby` generate specialised pipelines with no bounds checks in the aligned case.
- **Swizzle**: `swizzle_log` remaps threadgroup ids into tile-high strips for L2 reuse. It is currently 0 in the non-NAX path.

**Parameters**
- The default is **BM=BN=64, BK=16, WM=WN=2** (128 threads, each simdgroup computing 32×32 = 16 fragments).
- Half on base/pro devices: 64×64×16 with 1×2 simdgroups. NT layout: 64×32×32. Small-shape variants: 32×64×16, 64×32×32.
- **Split-K** when M·N is small and K ≥ max(M,N): 16/32 tiles, 2–32 partitions, a float partial buffer, then a reduction dispatch.
- **Device class** comes from `MTLDevice.architecture.name` (last character g/p/s/c/d). **Print it on this Mac and on the phone** to see which MLX branch applies.

**Measured here [M]** (GPU time from command-buffer GPUStart/End, median)

| GEMM (M×N×K) | MPSMatrixMultiplication f32 | MPS f16 | MPSGraph f32 (pipelined) | naive sg-matrix kernel f16 (no threadgroup memory) |
|---|---|---|---|---|
| 512×768×768 | **5.7 TF** (105 µs) | 3.4 TF | 2.1 TF | **4.6 TF** (131 µs) |
| 64×768×768 | 0.46 TF (164 µs) | 1.2 TF (64 µs) | 0.65 | 0.6 |
| 512×3072×768 | 2.4 TF | 3.2 TF | 3.0 | 2.1 |
| 512×512×1536 | 5.4 TF | 1.6 TF | 3.9 | 1.4 |
| 2048³ | **8.0 TF** (75% of peak) | 7.2 TF | 6.6 | 4.8 |
| 4096³ | 3.4 TF (thermal/LPM?) | 8.2 TF | – | 5.0 |

- **MPS is erratic on mid-size shapes**: 0.46 TF at 64×768×768, and its f16 is slower than f32 at 512×768×768. A 30-line naive simdgroup kernel already beats MPS f16 at 512×768×768. **There is clear room for a tuned steel kernel at Kokoro/ALBERT shapes.**
- The same naive kernel in **f32** was 10–20× slower (0.2–0.75 TF). Direct-from-device f32 `simdgroup_load` with no threadgroup staging is pathological, probably from register pressure (16 float accumulator fragments + 8 f32 operands). Do f32 through threadgroup staging (steel) or with smaller per-simdgroup tiles.
- **MPSGraph f16 matmul measured absurdly slow** (0.05–0.46 TF, up to 37 ms at 2048³) on macOS 27 / M2. This is unexplained (possibly a regression or a bad path for MTLBuffer-backed tensor data). Re-test before relying on MPSGraph f16.
- **MPSGraph per-call CPU overhead**: a single synchronous `runWithMTLCommandQueue` takes **~300–600 µs** even for tiny matmuls [M]. MPSGraph is unusable per-op for a small model like Kokoro. It is acceptable only as one big subgraph per command buffer, encoded with `encodeToCommandBuffer`.

**Plan [I]**
- Implement a steel-style kernel (`gemm_f16_gpu_sg` / `gemm_f32_gpu_sg`), templated on BM/BN/BK/WM/WN, with alignment function constants.
- Start from 64×64×16 / 2×2 and 32×32×16 / 2×2. Add split-K for 64×768×768-like shapes, which have too few threadgroups: 30 cores need ≥ 60–120 threadgroups.
- Fused epilogues (bias, GELU/LeakyReLU/Snake, residual add, AdaIN apply) are applied in registers before the store. That is the main reason to own the GEMM.
- Keep `gemm_*_gpu_mps` as a library variant and pick per shape.

### 2.2 Metal 4: MTLTensor, Metal Performance Primitives, GPU Neural Accelerators

**Metal 4 core API** (M1+/A14+ supported, so this M2 qualifies) [S: developer.apple.com "Understanding the Metal 4 core API", WWDC25 205]
- `MTL4CommandQueue`, plus **reusable** `MTL4CommandBuffer` with `MTL4CommandAllocator`.
- `MTL4ArgumentTable` replaces `setBuffer`. One `MTL4ComputeCommandEncoder` handles compute, blit and acceleration structures.
- **All resources are untracked**: explicit barriers and mandatory residency sets.
- This is the low-CPU-overhead path, good for record/replay (see the runtime research doc).

**MPP / TensorOps** [S: MPP Programming Guide PDF 2026-03; WWDC26 session 330]
- `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, `mpp::tensor_ops::matmul2d<desc, execution_simdgroups<N>>` on `tensor<>` views, with cooperative tensors for register-resident fused epilogues (`get_destination_cooperative_tensor`).
- OS 27 adds fp8/fp4/int2, block scales and MX formats. A cooperative tensor can now feed matmul directly as an input, which enables flash-attention chains.

**Neural Accelerators**
- They are per-shader-core matmul units in M5 and A19 Pro+, so the **A20 Pro** in the iPhone 18 Pro Max has them. Reach them only via TensorOps/MPP.
- MLX "NAX" gates them on OS ≥ 26.2 and arch gen ≥ 17 (Mac) / ≥ 18 (phone).
- Uses 16×16 fragments with `matmul2d_descriptor(16,32,16,…)`, and big tiles (128×128×512 with 4×4 simdgroups; 64-row variants).
- Measured gains: M5 vs M4 is 3.3–4× on prefill [S: machinelearning.apple.com/research/exploring-llms-mlx-m5].

**On M2**
- MPP compiles and runs on the shader ALUs with no gain. llama.cpp disables it pre-M5/A19 ("M2 Ultra ~5% slower") [S: ggml-metal-device.m].

**MPP tuning recipe for NA devices** [S: MPP guide]
- 2×2 simdgroups per threadgroup, 32×32 per simdgroup for 16-bit types.
- **No threadgroup staging** (read device memory directly).
- A `threadgroup_barrier(mem_none)` every BK=128 to keep simdgroups in lockstep.
- A 1-D grid with Morton-order tile walk.
- Static extents for full tiles, dynamic only for edge tiles (selected by function constant).

**Plan**
- Two GEMM back ends: `*_gpu_sg` (simdgroup_matrix, M2 default) and `*_gpu_mpp` (TensorOps, iPhone 18 default).
- **Kernels may differ per device**, as the user's plan allows. Selection is static per compiled model and per device family.

### 2.3 GEMV on GPU

**Measured [M]** (naive kernel: 1 simdgroup per 4 rows, float4/half4 loads, simd_sum, 8 simdgroups per threadgroup)

| Shape | custom f32 | custom f16 | MPSMatrixVectorMultiplication f32 |
|---|---|---|---|
| 768×768 | 22 µs | 17 µs | 23 µs |
| 3072×768 | 39 µs (240 GB/s) | 28 µs | 49 µs |
| 1024×256 (LSTM W_hh) | 15 µs | 11 µs | 18 µs |
| 50256×768 (lm_head) | 437 µs (**353 GB/s**) | **222 µs** (348 GB/s) | 435 µs |

- A GEMV is bandwidth-bound once it is a few MB. **Small GEMVs (≤ 2 MB) are latency-bound at 10–20 µs**, even inside one command buffer. This includes the ~3 µs minimum kernel cost plus ramp-up. On the GPU these sizes lose to CPU AMX (cblas_sgemv 768×768 = 11.5 µs; 1024×256 = 7.9 µs).
- **Design** (MLX `gemv.h`, llama.cpp `mul_mv`) [S]:
  - Per thread TM×TN = 4×4, simdgroup SM×SN with SN=32 along K.
  - BM = 4 or 8 simdgroups per threadgroup (8 for N ≥ 4096).
  - Split K across simdgroups with a threadgroup-memory reduction when N is small.
  - 128-bit loads.
  - "GemvWide" for batch 2–15 (MLX) and `mul_mv_ext` for batch 2–8 (llama.cpp): read each weight row once for several vectors.
- **Target**: ≥ 85% of DRAM bandwidth (~300+ GB/s) for big GEMVs. For GPT-2 decode, that puts f16 weights at ~0.75 ms/token.

### 2.4 Convolutions on GPU

**MLX dispatch** [S: mlx conv.cpp]
1. **Depthwise 2D** fast path (k ≤ 7, stride ≤ 2, C%8==0).
2. **Winograd F(6×6,3×3)** for 3×3/stride-1 with C, O % 32 == 0 and enough work.
3. **Implicit GEMM** (`steel_conv`) for aligned channels.
4. Explicit im2col GEMM otherwise.

For 1D: a depthwise fast path, else reshaped to 2D. **Conv transpose = input-dilated conv → explicit GEMM (unfold)**, which is wasteful.

**Plan**
- **Kokoro Conv1d**: implicit GEMM (M = T, N = Cout, K = Cin·k, with the A-tile loader gathering shifted/dilated rows from [T, C] or [C, T]). Fuse:
  - **prologue**: AdaIN apply + Snake/LeakyReLU on the input as it is loaded into threadgroup memory;
  - **epilogue**: bias + residual add.
  - This removes 2–3 memory passes per resblock conv. The vocoder resblocks are the bulk of Kokoro FLOPs [I; verify with a profile].
- **ConvTranspose1d**: polyphase decomposition. Run s phase-convs in one dispatch (grid z = phase), each writing a strided output (out[t·s + p]).
- **ResNet-18**: implicit GEMM, plus Winograd F(4×4,3×3) or F(6×6,3×3) as a variant for 3×3 layers. Compare against MPSGraph conv.
- **MobileNetV2**: direct depthwise kernel (NHWC, half4 over channels). Pointwise = GEMM.
- **DCGAN**: ConvTranspose2d s=2, k=4 → 4 sub-pixel 2×2 convs.

### 2.5 Attention

**MLX** [S]
- `sdpa_vector` (q_len ≤ 8): 32 keys per iteration, online softmax, 2-pass split-K variant for long KV.
- Steel attention (q_len > 8): bq=32, bk=32 (D<128), wm=4.

**metal-flash-attention** (philipturner) [S]: 83–86% ALU utilisation on M1/M2-class. It blocks along D and uses deliberately skewed block aspect ratios.

**Kokoro ALBERT** (12 heads, D=64, L≈20–512, batch 1) [I]
- Attention FLOPs are tiny: ~0.8 GFLOP at L=500, about 0.1 ms.
- The issue is the **grid**: bq=32 at L=128 gives only 48 threadgroups for 30 cores. Use **bq=16**, one threadgroup per (head, q-block).
- K/V for one head at L ≤ 128 fit in threadgroup memory (128×64×2×2 B = 32 KB, just the limit; at L ≤ 64 it is 16 KB).
- A fused flash kernel mainly saves 2 dispatches plus an L² round trip (in SLC). Benchmark against GEMM + softmax + GEMM before over-investing.
- On iPhone 18, use the TensorOps path (WWDC26 shows FlashAttention with `reduce_rows`).

**GPT-2**
- decode = sdpa_vector design;
- prefill = flash with a causal mask.

### 2.6 Elementwise, reductions, norms

**Elementwise**
- At **30 FLOP/byte** ridge, everything here is bandwidth- or latency-bound.
- **1 MB at 340 GB/s ≈ 3 µs ≈ one kernel's fixed cost** [M: ~3.25 µs for a single tiny dispatch in its own command buffer; ~1.2 µs per dependent dispatch inside an encoder]. For Kokoro-sized tensors, **the number of kernels matters more than per-kernel tuning**. Fuse.
- MLX uses work-per-thread `size<65536 ? 1 : 8/sizeof(T)` and contiguous/strided/large-index variants [S].
- **Use 16-byte accesses** (float4 / half8 via 2×half4). Use grid-stride loops only for huge tensors.
- Generate fused elementwise chains from a small template set (unary∘binary∘…) rather than by hand per op. The Python side picks the variant.

**Row reductions** (LayerNorm/RMSNorm/softmax, row ≤ 4096) [S: MLX layer_norm.metal, rms_norm.metal, softmax.metal]
- One threadgroup per row, `N_READS=4` elements per thread kept in registers.
- `simd_sum` → threadgroup `local[32]` → second `simd_sum`.
- **x is read once** and the output is written from registers. LayerNorm is two-pass from registers (mean, then centred variance).
- For hidden 768: 192 threads per row.
- Rows > 4096 use a looped variant. Softmax uses an online max/sum.

**InstanceNorm / AdaIN** (Kokoro: normalise each channel over T, then (1+γ)x̂+β)
- MLX has no dedicated kernel. PyTorch MPS has GroupNorm.metal.
- With **[C, T] layout**: one threadgroup per channel and a looped reduction over T (Welford or two-pass in f32).
- With **[T, C]**: column reduction (each thread owns channels, loops over T, coalesced along C).
- Best: fuse the *apply* into the next conv's prologue, so AdaIN costs one stats pass (read x once) and zero extra writes. The stats themselves could come from the previous conv's epilogue via threadgroup partial sums and a tiny finalize, but that complicates things. Start with a separate stats kernel.

### 2.7 Dispatch overhead and LSTM recurrence on GPU

**Measured [M]** (GPU timestamps)

| What | Cost |
|---|---|
| 1 tiny dispatch per command buffer | 3.25 µs of GPU time |
| **Serial encoder**, back-to-back dependent tiny dispatches | **1.16 µs/dispatch** |
| **Concurrent encoder**, independent tiny dispatches | **0.19 µs/dispatch** |
| Concurrent encoder + `memoryBarrier` after each dispatch (the dependent-chain case) | 1.2–1.5 µs/dispatch |
| **CPU round trip** (encode + commit + waitUntilCompleted, 1 dispatch) | **~137 µs median** (128–157) |
| **CPU encode cost** (classic API, setPSO once + dispatch) | **0.14 µs per dispatch** |

- One `waitUntilCompleted` round trip costs ~140 µs, roughly 100 dependent dispatches. **Data-dependent shape readbacks (Kokoro durations) cost ~0.15 ms each.** Keep them to one per inference.
- Encoding is cheap. Record/replay (ICB / Metal 4 reusable command buffers) mainly saves *CPU* time; GPU-side cost is the per-dependent-dispatch ~1.2 µs.

**LSTM recurrence options** (W_hh 1024×256 per direction)
- **A. One dispatch per step**: ~1.2 µs barrier + ~10–15 µs GEMV latency measured for 1024×256 (dominated by ramp and too few threadgroups). That is ~15 µs × T × layers. **Worse than the CPU (~8 µs/step).**
- **B. Persistent single-threadgroup kernel per direction**: loops over T inside the kernel; h in threadgroup memory; `threadgroup_barrier` per step; W_hh streamed from L2/SLC.
  - One core reads ~64 B/cycle, so ~0.5 MB f16 per step takes ~5.6 µs [S: metal-benchmarks core bandwidth; I].
  - Run both directions (and independent layers) on different cores concurrently.
  - Variant: keep part of W_hh in registers across a 1024-thread threadgroup. 208 KB of registers per core can't hold 512 KB, but a 2–4-threadgroup split with an int4/int8 W_hh could. That is risky.
- **C. Multi-threadgroup persistent kernel with a global spin barrier**: **don't**. Metal gives no forward-progress or co-residency guarantee, so it can deadlock [S/I].
- **D. CPU hybrid**: GPU does the W_ih GEMM, convs and attention; CPU does the recurrence. Unified memory means no copies, at the price of ~2 sync points (~140 µs each [M]).
- PyTorch MPS uses MPSGraph `LSTMWithSourceTensor` [S: RnnOps.mm]. It is opaque, but a reasonable library variant to benchmark.
- **Plan**: implement B and D and benchmark them against MPSGraph LSTM. Expect B or D to win; decide by measurement on both devices. The phone's CPU/GPU balance differs.

### 2.8 STFT / iSTFT (n_fft=20, hop=5)

- No FFT library is needed. Use **one fused kernel**: one thread per output sample.
  - Each sample receives 4 overlapping frames (20/5).
  - Per frame, compute the inverse real DFT at offset k from 11 bins using a 20×11 cos/sin table in `constant` memory.
  - Fuse in Kokoro's `exp(mag)` and `cos/sin(phase)` (so no complex spectrum is materialised), the synthesis window, the overlap-add, and the window-square envelope normalisation.
  - That is ~88 FMAs per sample, memory-bound, a few µs per second of audio [I].
- Alternatives: a frames×22 · 22×20 GEMM + overlap-add (2 dispatches), or MPSGraph FFT (macOS 14+). Both are library or variant baselines only. MLX's FFT handles n=20 via a radix-5×4 Stockham [S].
- Same approach on CPU (§1.9).

---

## 3. When library kernels win (current evidence)

| Case | Winner now [M] | Notes |
|---|---|---|
| CPU f32 GEMM, M ≥ 8 (all Kokoro/ALBERT/GPT-2 prefill shapes) | **Accelerate cblas_sgemm (AMX)**, 10–30× over NEON | Own only the fused epilogue (separate NEON pass), or tiny shapes |
| CPU GEMV ≤ 16 MB (LSTM W_hh, ALBERT proj) | **cblas_sgemv** (8–12 µs) | Compare with a tuned fused NEON f16 GEMV + gates |
| CPU GEMV, DRAM-sized (lm_head) | **custom NEON f16 weights, multi-thread** (1.6 ms vs 3.6 ms) | Accelerate appears to use 1 AMX unit |
| CPU on iPhone (SME2) | Accelerate first, then a custom SME2 FMOPA kernel, then KleidiAI as reference | Not measurable on M2 |
| GPU GEMM, large (≥ 1–2k) | **MPSMatrixMultiplication** ~8 TF (75% peak) | A steel kernel should match it; MLX reaches similar |
| GPU GEMM, Kokoro-sized (64–512 × 768) | **Custom** (MPS erratic: 0.46 TF at 64×768×768; naive f16 kernel beats MPS f16) | Split-K / small tiles |
| GPU GEMV | **Custom** ≈ MPS for large; both latency-bound for small | Small GEMVs → CPU or fuse |
| MPSGraph | **Loses as a per-op library** (300–600 µs per sync call; f16 matmul anomaly) | Use only for whole subgraphs if at all |
| GPU on iPhone 18 | **MPP `matmul2d` (Neural Accelerators)** expected to win GEMM/conv/attention | Custom kernels must be written with TensorOps |

---

## 4. Reference implementations to study / reproduce

| Reference | What to take | Where |
|---|---|---|
| **MLX metal backend** | steel GEMM (tiling, loaders, alignment function constants, split-K, per-device params in `matmul.cpp`); gemv.h; conv (Winograd, implicit GEMM, depthwise); sdpa_vector + steel attention; layer_norm/rms_norm/softmax; NAX (Neural Accelerator) GEMM; command-buffer batching (20/40/50 ops per buffer by device class) | github.com/ml-explore/mlx/tree/main/mlx/backend/metal |
| **llama.cpp ggml-metal** | `mul_mv` variants (`_4`, `_short`, `ext` for batch 2–8), `mul_mm` 64×32 tile; tensor-API gating by device; multi-threaded command-buffer encoding | ggml/src/ggml-metal/ |
| **llama.cpp ggml-cpu (ARM)** | i8mm/dotprod repacked Q4_0/Q8_0 GEMM/GEMV (`Q4_0_4_8` interleave), KleidiAI integration (`GGML_CPU_KLEIDIAI`), threadpool polling | ggml/src/ggml-cpu/arch/arm, ggml-cpu/repack.cpp |
| **XNNPACK / KleidiAI** | NEON f32/f16 GEMM micro-kernels (4x16, 6x8 ld128), qs8 c4/c8 neondot/neoni8mm, IGEMM indirection, dwconv up-N×K, deconv subconvolution; KleidiAI SME2 kernels (`kai_<op>_<types>_<tile>_<tech>`) | github.com/google/XNNPACK, github.com/ARM-software/kleidiai |
| **ExecuTorch** | Uses XNNPACK as the CPU backend on iOS, so XNNPACK is the thing to study; MPS delegate (MPSGraph) as a GPU baseline | github.com/pytorch/executorch |
| **PyTorch MPS** | Hybrid: MPSGraph for most ops plus hand-written kernels in `aten/src/ATen/native/mps/kernels/` (Gemv, Matmul, GroupNorm, RMSNorm, SoftMax, MPP attention). GemmHeuristics.cpp (M5-tuned); `RnnOps.mm` (MPSGraph LSTM); known MPSGraph matmul bugs on Apple7/8 (K > 2^15 corruption) | github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/mps |
| **metal-flash-attention / metal-benchmarks** (philipturner) | Apple GPU microarchitecture numbers; attention blocking strategies | github.com/philipturner/metal-flash-attention, /metal-benchmarks |
| **tinygrad metal** | Whole-graph ICB record/replay with buffer patching (`ops_metal.py`); generated MSL with simdgroup 8×8 MMA | tinygrad/runtime/ops_metal.py, renderer/cstyle.py |
| **ONNX Runtime CPU LSTM** | Batched W_ih GEMM + per-step W_hh GEMM + fused gates | onnxruntime/core/providers/cpu/rnn/ |
| **Apple** | MPP Programming Guide (2026-03); "Understanding the Metal 4 core API"; "Tuning your code's performance for Apple silicon"; Apple Silicon CPU Optimization Guide v4 (dev login); WWDC22 10159 (GPU compute perf), WWDC24 10211 (BNNSGraph), WWDC25 205/262, WWDC26 330 | developer.apple.com |
| **corsix/amx, scalable.uni-jena.de SME pages, arXiv 2409.18779 / 2606.25426** | AMX/SME instruction behaviour and measured peaks | – |

---

## 5. Kernel-tuning workflow (recommendation)

1. **Microbench harness, per device** (C++, no runtime involved, as the user's plan requires):
   - **CPU**: Google Benchmark, or a minimal in-house loop: warmup, then 7 batches, report median plus min/max/IQR. Use `mach_absolute_time` and pin QoS to USER_INTERACTIVE. Optionally read cycle counts via `kperf` (Mac only, needs root, so skip by default).
   - **GPU**: a custom timer using command-buffer `GPUStartTime/GPUEndTime` for multi-dispatch kernels, and `MTLCounterSampleBuffer` stage-boundary timestamps for per-dispatch timing. Report GPU time and wall time separately. Always warm up the PSO first.
   - **Environment capture in every result row**: device model, OS build, Low Power Mode, thermal state, load average, power source, git-free build id (the hash of the kernel source file), date.
   - **Refuse or flag LPM runs** (this session's numbers are LPM-skewed).
2. **Parameter sweeps (autotuning)**:
   - Each tunable kernel exposes a small parameter struct, e.g. GEMM {BM,BN,BK,WM,WN,split_k,swizzle} on GPU and {MR,NR,KC,MC,threads} on CPU.
   - GPU kernels take these as **function constants or template instantiations** in the metallib. The tuner instantiates a grid, prunes invalid configurations (threadgroup memory ≤ 32 KB, threads ≤ 1024), and runs each on the model's real shapes, which come from step 4 of the porting process.
   - Results go to a **results DB** (SQLite or JSONL) keyed by (kernel, params, shape, dtype, device, os, build).
   - The best configuration per (shape-bucket, device) is emitted as a **generated header/table** (`tuned_params_<device>.inc`), or chosen at compile time by the Python op builder. That fits the "device/shape-specific ops" model.
   - **No runtime autotuning in production.** iOS gets its table from on-device tuning runs pushed through the remote executor.
3. **Library variants as first-class competitors**:
   - Every op with a library equivalent (Accelerate, BNNSGraph, MPS, MPSGraph) has a `_accelerate`/`_mps` variant in the same sweep.
   - The report always shows "best custom vs best library vs PyTorch/MLX baseline" for the same shape.
4. **Roofline annotation**:
   - Every result reports achieved GFLOPS and GB/s and **% of the measured device peak** from §0.2 (for example "GEMV 3072×768: 240 GB/s = 71% of 340").
   - This makes "is there headroom?" obvious, and marks latency-bound kernels (small tensors far below both rooflines). Those need fusion, not tuning.
5. **Reports**:
   - A CLI generates a Markdown or terminal comparison table: per-kernel deltas vs the previous run, and regressions above noise (noise = IQR from repeats).
   - Before claiming a result, use Instruments/xctrace CLI (`xctrace record --template 'Metal System Trace'`) only for deep dives into GPU occupancy and limiter counters. It is headless and needs no GUI.
6. **Correctness gate inside the tuner**: every swept configuration is checked against the reference kernel (max-abs/rel error) before its time counts. This stops fast-but-wrong tile configurations from winning.

---

## 6. Open questions / next measurements

1. **Re-run all CPU numbers out of Low Power Mode**, on an idle machine. Expect ~1.5× on NEON and on DRAM bandwidth.
2. Check the A20 Pro for SME2 (sysctl), and measure its FMOPA peak, NEON peak, bandwidth, GPU FMA, MPP `matmul2d` TFLOPS, and dispatch/round-trip costs.
3. Check the AMX on E-core oddity (sgemm under BACKGROUND QoS ran at P-speed).
4. Investigate the MPSGraph f16 matmul slowness on macOS 27.
5. Find the crossover where Accelerate call overhead loses to NEON for tiny GEMMs (MNIST MLP 784→128→64→10 at batch 1 is the canonical case: ~0.1 MFLOP, where call overhead matters).
6. Run Kokoro's real op/shape census, i.e. which convs, T ranges, LSTM count, ALBERT seq lengths. That decides which of the above actually matter.
7. Print `MTLDevice.architecture.name` on both devices, to map MLX's per-device tables.
