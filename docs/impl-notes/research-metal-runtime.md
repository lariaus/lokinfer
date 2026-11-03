# Research: Metal Runtime (metal-cpp, Metal 4 vs Metal 3, replay, sync, memory, timing, offline compile)

Status: **research v1** (2026-11-01). It feeds §5 of `design-runtime-and-kernels.md`.
Machines: Mac **M2 Max (30-core GPU, Apple8 family, 32 GB)** with macOS 27.0.1 and Xcode 27.0 (27A266a), and
iPhone 18 Pro Max (A20 Pro, iOS 27). The iPhone was not measured yet (see §11).

Constraint provenance follows the project rule:
- **[user]** marks a decision from `design-decisions.md` or `original-plan.txt`.
- **[measured]** marks a number from the micro-benchmarks in §10, run on the M2 Max.
- **[inferred]** marks my own conclusion, reasoning or recommendation. These lose by default if a measurement contradicts them.

> **Caveat on all numbers: the Mac had `lowpowermode 1`** (`pmset -g`, on AC power). The xctrace capture in §6
> confirms it: *"Minimum GPU Performance state due to active device conditions"*.
> Absolute latencies and clock-bound GPU times are pessimistic. Memory bandwidth was barely affected (≈330 of
> 400 GB/s). Relative comparisons (Metal 4 vs classic, ICB vs direct, sync vs no sync) should hold.
> **TODO**: re-run with Low Power Mode off before quoting any absolute number as a target.

---

## 0. TL;DR recommendations

1. **metal-cpp `release/metal-cpp_macOS27.0.1_iOS27.0.1`** (2026-09-22) covers **all of Metal 4** (30 `MTL4*.hpp` headers)
   and the Apple11 family. It is header-only. Fetch the GitHub tag tarball with FetchContent `URL` + `URL_HASH`
   (§1). It builds unchanged for iOS (verified by cross-compiling the benchmark for `arm64-apple-ios27.0`).
2. **Use the Metal 4 API (`MTL4CommandQueue`) as the only GPU path.** Both target devices support it (M2 Max reports
   `supportsFamily(Metal4)=1`), and **[user]** the targets are macOS/iOS 27 only. Measured on M2 Max:
   - Dependent dispatches cost **~2.05 µs GPU each with an intra-encoder barrier**. Classic Metal costs 3.0 µs
     (serial encoder) to 3.9 µs (concurrent + `memoryBarrier`).
   - Independent dispatches cost **0.19 µs** (classic: 0.26 µs).
   - CPU encode is about **0.13 to 0.47 µs per dispatch**.
   - Apple's own libraries now accept Metal 4 objects in the 27 SDK, so we lose nothing:
     `MPSNDArrayMatrixMultiplication encodeWithMTL4CommandEncoder:` and `MPSGraph runWithMTL4CommandQueue:`.
3. **Use one compute encoder per GPU segment, and barriers only between dependency levels.** Never split
   encoders or use queue barriers per dispatch: that measured **33 µs per dispatch** in Metal 4.
4. **Default to direct encoding from a precomputed flat plan. Treat ICB replay as an optional strategy.**
   - ICB replay cuts CPU cost to about 15 to 50 µs per run, regardless of dispatch count.
   - But on the GPU, ICB dependent chains are slower: 2.6 to 2.75 µs per dispatch vs 2.05 µs direct.
   - Direct encoding (0.3 to 0.5 µs per dispatch) overlaps GPU execution (≥2 µs per dependent dispatch), so the
     GPU is the bottleneck either way.
   - Metal 4 command buffers **cannot be re-committed**. Re-committing hits an assertion in
     `IOGPUMetal4CommandQueue commitFillArgs` **[measured]**. So "record" means either a precomputed plan or an ICB.
5. **Every CPU↔GPU round trip costs ≥ ~120 µs** (commit → GPU start ≈ 90 to 130 µs warm, ≈ 440 µs after 3 ms idle).
   - A mid-graph readback costs **~0.45 to 0.8 ms** in a warm pipeline.
   - Make data-dependent shapes GPU-driven: dims in a device buffer, `dispatchThreadgroups(indirectBuffer)`, and
     upper-bound allocation. Keep readback only as a fallback strategy to benchmark **[user: try both]**.
6. **Memory**:
   - Use one **Shared + HazardTrackingUntracked** arena `MTLBuffer`. Shared and Private have identical GPU bandwidth
     on Apple silicon (330 vs 333 GB/s **[measured]**).
   - Load weights by `mmap`ing the `.lkm` file and wrapping it with `newBufferWithBytesNoCopy`. Measured: 56 µs to
     wrap 256 MB, with full bandwidth. Data must be **16 KB-aligned**.
   - Use one `MTLResidencySet` added to the queue once at load.
   - Use no `MTLHeap`, unless we later need aliasing textures/tensors.
7. **Timing**:
   - `MTL4CommitFeedback.GPUStartTime/GPUEndTime` gives per-command-buffer GPU time.
   - Metal 4 counter heap `writeTimestamp(Precise)` gives per-dispatch times. Each timestamp costs ~10 µs of GPU
     time, so use it only for kernels ≫10 µs. `Relaxed` granularity produced garbage per dispatch.
   - For kernel benchmarking: K repetitions in one command buffer, then GPU time / K.
   - Full traces: `xcrun xctrace record --template 'Metal System Trace' --launch -- <bin>` plus
     `xctrace export --xpath`. Both work from the CLI **[measured]**.
8. **Offline compile**:
   - The **Metal Toolchain component is not installed on this Mac** (`xcodebuild -showComponent MetalToolchain` →
     `uninstalled`). Install it once with `xcodebuild -downloadComponent MetalToolchain`.
   - Build one metallib per platform with `-std=metal4.0` or later; macOS/iOS 27 adds MSL 4.1.
   - Use function constants for shape/tile specialization.
   - Create all pipeline states at load.
9. **GEMM/conv**:
   - **iPhone 18 Pro Max (A20 Pro) has GPU Neural Accelerators.** Use MPP `matmul2d`/`convolution2d` (TensorOps)
     there.
   - **M2 Max has none.** TensorOps falls back to shader code: 6.7 TFLOPS of ~10.7 peak with a naive MPP kernel
     **[measured]**. ggml measured it ~5% slower than `simdgroup_matrix` on M2 Ultra. Use `simdgroup_matrix`
     kernels on the Mac.
   - Plan **two GEMM/conv kernel families**, selected by GPU family.

---

## 1. metal-cpp

**Version.** Tags on `github.com/apple/metal-cpp`, queried via the GitHub API on 2026-11-01:

| Tag | Date | Changelog |
|---|---|---|
| `release/metal-cpp_macOS27.0.1_iOS27.0.1` | 2026-09-22 | "Add the Metal APIs to support the **Apple11** GPU family" |
| `release/metal-cpp_macOS27_iOS27` | 2026-06-08 | "Add all the new Metal APIs in macOS 27 and iOS 27" |
| `macOS26.4` | | |
| `macOS26` | | Added Metal 4 and Apple10 |
| `macOS15.2` | | |

Apple's developer page (developer.apple.com/metal/cpp) now just links to the GitHub repo.

**Metal 4 coverage**, checked in the 27.0.1 headers, all present:
- `MTL4CommandQueue` (`commit`, `signalEvent`, `wait`, `addResidencySet`), `MTL4CommandBuffer`
  (`beginCommandBuffer(allocator)`, `useResidencySet`, `writeTimestampIntoHeap`), `MTL4CommandAllocator`.
- `MTL4ComputeCommandEncoder`: dispatch, indirect dispatch by `MTL::GPUAddress`, `executeCommandsInBuffer`,
  copies, `fillBuffer`, `writeTimestamp`.
- `MTL4CommandEncoder` barriers: `barrierAfterEncoderStages`, `barrierAfterQueueStages`, `barrierAfterStages`;
  fences.
- `MTL4ArgumentTable` (`setAddress(gpuAddress[, stride], index)`, `setResource`, `setTexture`).
- `MTL4Compiler`, `MTL4CompilerTask`, `MTL4Archive`, `MTL4PipelineDataSetSerializer`.
- `MTL4MachineLearningCommandEncoder` and pipeline, `MTL4Counters` (`CounterHeap`, `TimestampGranularity`),
  `MTL4CommitFeedback`.
- `MTLResidencySet`, `MTLTensor`.
- The new 27 APIs: `ForwardProgressUsage`, `ContentionRelief`, `optimizeForPersistentKernel`,
  `recommendedPersistentThreadgroupsPerGrid…`, FP8/FP4 tensor dtypes, tensor auxiliary planes.

**Not covered**: MPS, MPSGraph and Accelerate. Those need a small **Objective-C++ (`.mm`) shim**. metal-cpp
objects are toll-free (`(__bridge id<MTLBuffer>)buf`), so the shim is trivial.

**CMake.** The repo has no `CMakeLists.txt`, so we use it as an INTERFACE library. We use the URL plus hash, not
git, **[user: FetchContent, pinned]**:

```cmake
include(FetchContent)
FetchContent_Declare(metal_cpp
  URL      https://github.com/apple/metal-cpp/archive/refs/tags/release/metal-cpp_macOS27.0.1_iOS27.0.1.tar.gz
  URL_HASH SHA256=3a2f456c5373748e1b8de0896d1dbe53d6e0a1d3b8b9e5235c1a8abae6f373f5
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(metal_cpp)          # just populates (no CMakeLists in the archive)
add_library(lk_metal_cpp INTERFACE)
target_include_directories(lk_metal_cpp SYSTEM INTERFACE ${metal_cpp_SOURCE_DIR})   # SYSTEM: silence warnings
target_link_libraries(lk_metal_cpp INTERFACE "-framework Metal" "-framework Foundation")
```

The SHA256 is that of the tarball I downloaded on 2026-11-01. GitHub tag archives are stable in practice, but not
contractually. If the hash ever breaks, re-download and re-pin.

Exactly one `.cpp` in the GPU runtime must contain:

```cpp
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
```

Skip QuartzCore/`CA_PRIVATE_IMPLEMENTATION`: we have no layers, and the benchmark links fine without QuartzCore.

**iOS.**
- Same headers and same code. The `iphoneos` SDK has the same frameworks (Metal, MPP, MPS, MPSGraph).
- Verified: `xcrun -sdk iphoneos clang++ -target arm64-apple-ios27.0 -std=c++20 … -framework Metal -framework Foundation`
  builds and links the full benchmark.
- In CMake: `-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 -G Xcode` (or Ninja plus a toolchain file).
  Nothing metal-cpp specific is needed.

**metal-cpp usage notes [inferred, from the README plus the benchmarks]:**
- Objects from `new*`/`alloc` are owned (+1). Everything else is autoreleased.
- Classic `queue->commandBuffer()` and `computeCommandEncoder()` are autoreleased, so every run needs an
  `NS::AutoreleasePool`. The Metal 4 path creates almost everything once with `new*` (allocators, command buffers,
  argument tables), so per-run autorelease traffic is minimal. Keep a pool around `run()` anyway, plus one per
  worker thread.
- Use `NS::SharedPtr`/`NS::TransferPtr` (MLX does) for RAII.
- No exceptions anywhere **[user]**. metal-cpp throws none; errors come back via `NS::Error**`.

---

## 2. Metal 4 vs classic Metal (Metal 3) for many small compute kernels

### 2.1 What Metal 4 changes (WWDC25 205 "Discover Metal 4", 262, headers)
- **Device-created objects.** `MTL4CommandBuffer`s are created from the device, decoupled from queues, and
  reusable. Memory comes from an explicit `MTL4CommandAllocator`; `reset()` it only after the GPU has finished.
  Allocators aren't thread-safe, so use one per thread.
- **Unified compute encoder.** Compute, blit (copy/fill) and acceleration-structure work go in one encoder, so there
  are fewer encoders.
- **Concurrency by default and no automatic hazard tracking.** We express dependencies with **stage-to-stage
  barriers**: `barrierAfterEncoderStages(after, before, visibility)` inside an encoder,
  `barrierAfterQueueStages` across encoders, plus fences and events.
- **`MTL4ArgumentTable` replaces `setBuffer`/`setBytes`.** You `setAddress(buffer->gpuAddress() + offset, slot)`.
  **There is no `setBytes`**, so params must live in a buffer.
  - **Bindings are snapshotted at each dispatch** **[measured]**. I set slot 0 to X, dispatched, set it to Y and
    dispatched again; both buffers were incremented. So one table can be mutated between dispatches.
- **Residency sets are mandatory.** Every resource must be in a residency set attached to the queue or the command
  buffer. Since Xcode 27, pipeline states referenced by an ICB must also be added to a residency set
  (doc change noted in dotnet/macios Xcode 27 API diff).
- **`MTL4Compiler`** is separate from the device. It follows the caller's QoS and gives explicit control of
  compilation.
- **Shader ML**: `MTLTensor`, `metal_tensor`, and the Metal Performance Primitives (TensorOps)
  `matmul2d`/`convolution2d`.
- **`MTL4MachineLearningCommandEncoder`** runs CoreML ML-programs (converted to an `MTLPackage`) on the GPU timeline.
  Not for us **[user: no ANE, hand-written kernels]**, but it is a possible benchmark baseline.
- **macOS/iOS 27 additions:** MSL 4.1; persistent-kernel support on pipeline descriptors (`forwardProgressUsage`,
  `contentionRelief`, `optimizeForPersistentKernel`, `recommendedPersistentThreadgroupsPerGrid…`), with no header
  docs yet; FP8/FP4/int2 tensors and scale planes; and the deprecation of `StorageModeManaged`. The persistent-kernel
  APIs are worth a later look for LSTM-style sequential loops (§9.6).

### 2.2 Device support

| Feature | GPUs | M2 Max (Apple8) | iPhone 18 Pro Max (A20 Pro) |
|---|---|---|---|
| Metal 4 core API (MTL4 queue/encoders, argument tables, barriers, residency sets, MTLTensor, MPP API) | Apple7+ = **M1/A14 and later** (WWDC25-205) | **yes** (`supportsFamily(Metal4)=1` **[measured]**) | yes |
| Apple9 features: dynamic caching (on-chip memory allocated dynamically → better occupancy), HW ray tracing, mesh shading | A17 Pro, M3, M4 | **no** (`Apple9=0` **[measured]**) | yes |
| **GPU Neural Accelerators** (TensorOps run on per-core matrix units) | Apple10+ = **M5 / A19 and later** | **no**: TensorOps falls back to shader code | **yes**. Apple's A20 Pro announcement (2026-09-09): Neural Accelerators in the GPU cores, with "2× FP8" vs A19. Apple11 was added to metal-cpp the same month, so A20 is most likely Apple11 **[inferred]** |
| Counter sampling points | | `AtStageBoundary` only (draw/dispatch/blit = 0) **[measured]** | expected the same |

The iPhone family number should be confirmed with the same `supportsFamily` probe in the iOS bench app.

### 2.3 What others use, and why

**ggml-metal (llama.cpp, master 2026-11):**
- **Classic** `MTLCommandQueue`, `commandBufferWithUnretainedReferences`.
- `computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent`, with `memoryBarrierWithScope:MTLBarrierScopeBuffers`
  only between dependent nodes. The graph is split across `n_cb` command buffers encoded in parallel by
  `dispatch_apply`; its comment says the optimal `n_cb` is 1 or 2.
- Residency sets (macOS 15+), and `newBufferWithBytesNoCopy` over the mmapped model.
- Detects the "tensor API" via `supportsFamily(MTLGPUFamilyMetal4)` and a test compile of a `matmul2d` kernel. It
  **disables it by default on anything that isn't M5/M6/A19/A20**: *"M2 Ultra ~5% slower; M4 no significant
  difference."*

**MLX (main 2026-11):**
- **Classic** API through metal-cpp, with one concurrent compute encoder per command buffer.
- Tracks the inputs and outputs of the previous dispatches and inserts `memoryBarrier(BarrierScopeBuffers)` only on
  a real dependency. Uses `MTLFence`s between encoders.
- All buffers are `StorageModeShared | HazardTrackingModeUntracked`. Uses residency sets.
- Commits every N ops: `max_ops_per_buffer` is 20 on phones, 40 on base/Pro, 50 on Max/Ultra.
- Neural-accelerator ("NAX") GEMM kernels need macOS 26.2+ and arch gen ≥17 (Mac, i.e. M5) or ≥18 (phone, i.e. A19).
  They use Metal 4 TensorOps in the kernels, but **not** the MTL4 queue.

**PyTorch MPS (main):**
- Classic `MTLCommandQueue` plus `MPSCommandBuffer` with `commitAndContinue`.
- Mostly MPSGraph-cached graphs plus MPS kernels plus some custom Metal kernels.
- A heap-based caching allocator (`MPSHeapAllocatorImpl`). No Metal 4.

**Why nobody uses the MTL4 queue yet [inferred]:**
- It needs OS 26+.
- MPS/MPSGraph only gained MTL4 entry points in the 27 SDK.
- Their dispatch patterns (a concurrent encoder plus barriers) already approximate Metal 4 semantics.

**We have no backward-compatibility constraint**, and the MTL4 barrier was cheaper in our measurements (§2.4).

### 2.4 Measurements (M2 Max, Low Power Mode ON, §10 bench2-B)

Setup: tiny kernel (256 threads, 1 threadgroup); 40 back-to-back command buffers × 200 dispatches; GPU time per
dispatch is the median of per-command-buffer `GPUEnd−GPUStart`.

| Variant | GPU µs/dispatch | CPU encode µs/dispatch |
|---|---|---|
| classic, concurrent encoder, chain + `memoryBarrier(buffers)` (MLX/ggml style) | **3.89** | 1.22 |
| classic, serial encoder, chain (implicit serialization) | 2.99 | 0.38 |
| classic, one encoder per dispatch, chain | 2.07 | 1.03 |
| classic, concurrent encoder, independent | 0.26 | 0.40 |
| **MTL4, one encoder, chain + `barrierAfterEncoderStages(Dispatch→Dispatch)`** | **2.05** | 0.47 |
| **MTL4, one encoder, independent** | **0.19** | 0.32 |
| MTL4, one encoder per dispatch + `barrierAfterQueueStages` | **33.5** (avoid!) | 2.55 |
| classic ICB replay, chain (`setBarrier`) | 2.59 | 0.12 |
| MTL4 ICB replay, chain | 2.75 | 0.05 |

A single-command-buffer run (bench1 §2, N=1000, less warm) shows the same ordering:
- CPU encode per dispatch: MTL4 0.13 µs (independent) and 0.29 µs (chain); classic concurrent+barrier 0.56 µs;
  classic serial 0.19 to 0.30 µs.
- GPU per dependent dispatch: MTL4 2.4 µs; classic concurrent+barrier 3.56 µs; classic serial 2.0 µs (noisy because
  of clock ramp).

Takeaways:
- **The dominant per-kernel cost on the GPU is the dependency barrier: ~2 µs per dependent dispatch.** It is not the
  CPU encode, and not the launch: independent dispatches pipeline at ~0.2 µs.
  - Example: a Kokoro-sized graph of ~1000 mostly-dependent kernels pays ~2 ms of pure barrier/launch overhead.
    That makes **fusion (fewer kernels) the main lever** **[user: hand-fused kernels]**, more than any runtime trick.
- Only place barriers where there is a real RAW/WAR/WAW dependency. The compiler should schedule independent ops
  next to each other, by dependency level, so they share one barrier.
- Hazard tracking: tracked vs untracked made no measurable difference to encode cost in the serial encoder at small
  N. Untracked is still required: Metal 4 has no tracking anyway, and MLX does the same.

---

## 3. Record / replay

### 3.1 Indirect Command Buffers (ICB) for compute
- **Creation.** `MTLIndirectCommandBufferDescriptor`:
  - `commandTypes = ConcurrentDispatch` (or `ConcurrentDispatchThreads`).
  - `inheritPipelineState=false` and `inheritBuffers=false`, so each command carries its own state.
  - `maxKernelBufferBindCount` ≤ 31.
  - Then `device->newIndirectCommandBuffer(desc, maxCount, options)`.
  - Pipelines must be created with `supportIndirectCommandBuffers = true`.
- **Commands.** `indirectComputeCommand(i)` gives:
  - `setComputePipelineState`, `setKernelBuffer(buf, offset, index[, stride])`, `setThreadgroupMemoryLength`.
  - `concurrentDispatchThreadgroups`/`concurrentDispatchThreads`.
  - **`setBarrier()`**, which makes this command wait for all previous ones, and `clearBarrier`, `reset`.
  - **No `setBytes`**: params go in buffers. **No indirect (GPU-sized) dispatch inside an ICB command.**
- **Limits.**
  - The size is fixed at creation (`maxCount`).
  - The 16,384-command limit per `executeCommandsInBuffer` range applies only to the Mac2 family (Intel/AMD Macs);
    Apple-silicon families don't have it (Apple docs, `executeCommandsInBuffer:withRange:`).
- **Execution.**
  - `encoder->executeCommandsInBuffer(icb, NS::Range)`, available on both classic and MTL4 encoders.
  - Or `executeCommandsInBuffer(icb, indirectRangeBuffer)`, where the GPU decides which range runs: the range is
    a `MTLIndirectCommandBufferExecutionRange` in a buffer, 4-byte aligned.
  - The ICB, every buffer it references, and (MTL4, Xcode 27) the PSOs must all be resident.
- **GPU-encoded ICBs.** A compute kernel can write commands into an ICB through an argument buffer
  (MSL `command_buffer` / `compute_command`). This is how to get data-dependent grid sizes inside a replayed ICB.
- **Updating between replays (CPU).**
  - Rebinding one buffer (`setKernelBuffer`) costs **0.04 to 0.08 µs** **[measured]**.
  - Re-recording a whole command costs **0.12 to 0.18 µs** **[measured]** (first touch of a fresh ICB: 1.7 µs/cmd).
  - So a CPU "patch list" per run is cheap: new input/output offsets and new grids for dims that depend on inputs.
  - Data that changes per run (dims, scalars) should live in a params buffer that the commands point at. Then only
    grids need patching.

**Measured (bench1 §3):**

| | N=10 | N=100 | N=1000 |
|---|---|---|---|
| ICB record (CPU, once) | 6 to 17 µs | 13 to 18 µs | 115 to 136 µs |
| ICB replay CPU per run (cb + encoder + execute + commit) | 17 to 19 µs | 15 to 25 µs | 19 µs (indep) / 41 to 50 µs (chain) |
| direct MTL4 encode CPU per run | ~16 µs | ~33 to 40 µs | 130 µs (indep) to 290 µs (chain) |
| GPU per dispatch, ICB chain vs direct MTL4 chain | 6.1 vs 4.2 | 4.7 vs 3.7 | 1.8 to 2.3 vs 2.4 |

Steady state (bench2-B): ICB chain 2.6 to 2.75 µs vs direct MTL4 2.05 µs per dispatch.

**Anomaly.** With *classic* ICBs, 40 back-to-back command buffers showed **~300 to 440 µs gaps between command
buffers**: wall time 2.5 to 4.3 µs per dispatch vs GPU time 0.29 to 2.6 µs. The MTL4 ICB path did not show this
(wall 1.3 µs per dispatch). This is another reason to stay on MTL4.

### 3.2 Metal 4 equivalents of "record"
- An MTL4 command buffer **cannot be committed twice**. Doing so aborted with `allocator && storage` in
  `IOGPUMetal4CommandQueue commitFillArgs` **[measured]**. You must `beginCommandBuffer` and re-encode, so command
  buffers are not a replay mechanism.
- Re-encoding is cheap. Allocator `reset()` plus `beginCommandBuffer` is a few µs, and command buffers and argument
  tables are long-lived objects. So **"record" in Metal 4 = a precomputed flat dispatch plan** (§9.2). It is cheap to
  replay with direct encoding, and ICBs are used where CPU time must be near zero.

### 3.3 Expected savings [inferred from the numbers]
- **CPU:** at 1000 dispatches, direct encoding costs ~0.13 to 0.47 ms CPU per run; ICB costs ~0.02 to 0.05 ms.
- **GPU:** ICB loses ~0.6 µs per *dependent* dispatch (~0.6 ms for 1000 dispatches), and is equal for independent
  ones.
- **Wall time:** with chunked commits, CPU encoding overlaps GPU execution. GPU work for a dependent dispatch is
  ≥2 µs, which is above the ≤0.5 µs encode, so the wall time is GPU-bound and **direct ≈ or better than ICB on the
  M2 Max**.
- **Where ICB should win:**
  - (a) Latency of tiny graphs, where the first dispatch waits for the whole encode.
  - (b) Mixed CPU/GPU graphs, where the CPU must run CPU kernels instead of encoding.
  - (c) Energy on iPhone.
- **[user]** asks for both strategies, so both are kept, with direct as the default. The choice is re-benchmarked
  per model and per device.

---

## 4. GPU-driven dispatch for data-dependent shapes

**APIs.**
- MTL4: `dispatchThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerThreadgroup)` and
  `dispatchThreads(GPUAddress)`.
- Classic: `dispatchThreadgroups(buffer, offset, tgSize)`.
- The argument layout is `MTLDispatchThreadgroupsIndirectArguments {uint32 threadgroupsPerGrid[3]}`.
- The producing kernel, or a tiny "dim kernel", writes those 3 uints plus the dim values. Consumer kernels read their
  dims from a device buffer, not from constants.

**Measured (bench1 §4).** One-shot: a produce kernel then a consume kernel, with n=1000 decided on the GPU.

| Strategy | Wall per inference |
|---|---|
| GPU-driven: produce → barrier → indirect consume, one command buffer | **229 µs** |
| Readback: produce, commit+wait, CPU reads n, second command buffer consume, commit+wait | **379 µs** (+150 µs) |
| Upper-bound grid (16 threadgroups) with early exit, one command buffer | **171 µs** |

**Measured (bench2-C).** Warm pipeline: 30 segments × 100 chained dispatches, per segment.

| Sync between segments | µs/segment |
|---|---|
| none (pipelined) | **194** |
| `MTLSharedEvent` spin, encode the next segment after the wake | 1008 |
| event spin, **next segment pre-encoded** | 651 |
| `waitUntilSignaledValue` | 769 |
| `waitUntilCompleted` | 937 |

bench1 shows the same effect: a 200-dispatch chain took 605 µs, and 1233 µs with one mid-graph CPU wait.

**Conclusion [inferred].**
- A mid-graph CPU sync costs **≈0.45 to 0.8 ms**:
  - the GPU drains;
  - ~15 µs for the GPU→CPU wake;
  - ~90 to 130 µs from commit to GPU start;
  - re-encode;
  - the GPU clocks drop and ramp up again.
- That is the cost of hundreds of kernels. **GPU-driven is the default for data-dependent dims**:
  - **Upper-bound grid plus early exit** when the bound is ≤ ~2× the typical size. It is the cheapest: no extra
    dependency.
  - **Indirect dispatch** when the bound is loose. Kokoro's frame count comes from predicted durations, and its upper
    bound is `max_tokens × max_duration`.
  - **Allocation by upper bound.** The planner sizes buffers that depend on a data-dependent dim with its declared
    upper bound. Kokoro needs one at compile time: `input_types` or an attribute on `dim.produce`.
- The CPU still needs the actual value to size *outputs* returned to the user. Read it after the final completion,
  at no extra sync, and return views trimmed to it.
- Keep **readback** as an alternative strategy flag (`sync gpu_to_cpu`, then the CPU evaluates the dims and encodes
  the rest) **[user: try both, benchmark]**. If used, always pre-encode whatever doesn't depend on the value before
  waiting.

---

## 5. Memory

**Storage modes.**
- Apple silicon has unified memory. Measured GPU bandwidth (256 MB buffers, bench1 §7):
  - Shared→Shared copy: 333 GB/s (read + write).
  - Private→Private copy: 330 GB/s.
  - Read-only: 322 GB/s (shared) vs 321 GB/s (private).
  - The nominal M2 Max bandwidth is 400 GB/s.
- **No GPU benefit from Private for buffers** **[measured]**. Private can enable lossless compression for
  *textures*, which we don't use.
- `Managed` is deprecated on Apple silicon in 27.
- → **Everything is `StorageModeShared | HazardTrackingModeUntracked`** **[user: shared-storage GPU tensors]**,
  matching MLX. CPU kernels in mixed graphs and numpy/DLPack interop then read the same memory with zero copies.
- Cache coherency: the CPU sees GPU writes after completion has been observed (an event or completion). No flush
  calls are needed.

**`newBufferWithBytesNoCopy` on mmapped weights.**
- Measured:
  - It works on a `MAP_PRIVATE` file mapping with either `PROT_READ` or `PROT_READ|PROT_WRITE`.
  - Wrapping 256 MB takes 56 to 92 µs, and `contents()` equals the mmap pointer.
  - GPU read bandwidth is 322 GB/s, equal to a normal buffer, once the pages are resident.
- Requirements:
  - The pointer must be **page-aligned and the length a multiple of the page size**. The page size is **16 KB**
    on both macOS and iOS arm64 (`vm.pagesize=16384`).
  - → The `.lkm` weight blob must start at a 16 KB file offset, and each wrapped region must be padded to 16 KB.
- It works on iOS too: ggml uses the same call on iOS for mmapped GGUF files.
- On iOS, read-only file-backed pages are *clean* memory. The kernel can evict and re-read them, and they count far
  less against the jetsam limit than dirty allocations. For an app-embedded model this is the right way to hold the
  weights (Kokoro fp32 ≈ 330 MB) **[inferred]**.
- Map weights `PROT_READ`, so a buggy kernel that writes to them faults loudly.
- Weights that init-hoisting transforms (repacking, dtype conversion) become dirty arena memory instead.
- **First-touch cost.** The first GPU use of fresh pages is slow. Freshly allocated buffers ran a 64 MB copy at
  ~700 µs on first use vs 400 µs warm (bench3), and mmapped pages fault in from disk.
  → At load: `requestResidency()` on the residency set, and pre-fault the arena once (`fillBuffer` or a memset)
  during init.

**Arena / MTLHeap.**
- A placement heap with explicit offsets works. Two buffers placed at offset 0 alias the same GPU address
  **[measured]**; `heapBufferSizeAndAlign(1 MB)` gives alignment 256.
- But for buffers, **one big `MTLBuffer` plus offsets is simpler and equivalent**:
  - In MTL4, binding is `gpuAddress() + offset`, so sub-allocations cost nothing.
  - The planner's output maps directly: `BufRef{arena, offset}` → GPU address.
  - Alignment: 256 B for GPU sub-buffers, which is ≥ the CPU SIMD and cache-line needs.
  - Max single buffer: `maxBufferLength` = 20.1 GB on the M2 Max (iPhone: check, it will be much smaller).
- A heap is only needed if we later want aliasable `MTLTexture`/opaque-layout `MTLTensor`s inside the arena.
- Dynamic shapes **[user: symbolic offsets evaluated at run time]**: allocate the arena with capacity for the
  evaluated size, and grow by geometric reallocation when a call exceeds it. Growing means:
  - a new buffer;
  - swapping it in the residency set, then `commit()`;
  - the GPU-address ptrTable is recomputed. With direct encoding nothing else is cached. With ICBs, re-patch the
    commands.

**Hazard tracking.**
- Untracked everywhere. The compiler/runtime computes dependencies from the IR's buffer liveness and memory-plan
  offsets. Two ops that touch overlapping arena ranges are dependent, *including WAR on reused offsets*, which is
  where memory planning creates dependencies.
- Barriers go inside an encoder, `MTLFence`s across encoders (rare), and `MTLSharedEvent`s across queues or the CPU.

**Residency sets.**
- One per model or executable, containing: arena, weight buffers, param buffer, dim buffer, ICBs, and (with ICBs)
  PSOs.
- `commit()` and `requestResidency()` at load; `q4->addResidencySet(rs)` once.
- After that there is no per-command-buffer cost (vs classic `useResource`).
- Input/output tensors passed zero-copy from the caller are `MTLBuffer`s outside the set. Two options:
  - add them to a small per-call residency set via `cb->useResidencySet()` (cheap if the set is unchanged);
  - or copy them into arena slots.
  Benchmark both.

---

## 6. Timing

**Per command buffer.**
- Classic `cb->GPUStartTime()/GPUEndTime()`, and MTL4 `MTL4CommitFeedback::GPUStartTime/GPUEndTime` via
  `CommitOptions::addFeedbackHandler`.
- The values are host-time seconds, in the same clock as `mach_absolute_time`. You can subtract the CPU commit time
  to get submit latency. That is how §7's latency decomposition was made.

**Per dispatch.**
- M2 Max supports only `CounterSamplingPointAtStageBoundary` **[measured]**. With the classic API, per-kernel
  timestamps need **one encoder per kernel** via `ComputePassDescriptor.sampleBufferAttachments`.
  - Measured: 3.4 → 12.2 µs GPU per tiny dispatch.
  - The resolved values are **nanoseconds**: 64 MB copies resolved to ~483,000 per kernel.
  - Device timestamps (`sampleTimestamps`, MTL4 heaps) are 24 MHz ticks = mach ticks on this machine. Don't mix
    the two.
- Metal 4: `MTL4CounterHeap` (type `Timestamp`, 8 B entries) with
  `encoder->writeTimestamp(granularity, heap, index)` inside one encoder. Resolve with
  `heap->resolveCounterRange()`, which returns ticks; divide by `queryTimestampFrequency()` = 24 MHz.
  - `Precise`: correct for large kernels. 8 × 64 MB copies read 388 to 400 µs each, ~1 µs apart.
    **But each timestamp costs ~10 µs of GPU time.** A 100-dispatch tiny chain went from ~360 µs to 2.8 ms, and
    per-dispatch deltas read ~7 µs vs a true ~2 to 3.5 µs.
  - `Relaxed`: values were zero or garbage between dispatches. Unusable per dispatch.

**Recommended kernel-benchmark method [inferred]** (kernel library only, no runtime **[user]**):
1. Warm up: ~50 ms of the same kernel, to get clocks up.
2. Encode **K back-to-back dispatches** of the kernel into one encoder:
   - with a barrier between them, for latency-like numbers;
   - without, for throughput.
   K is adaptive, so that GPU time per command buffer is ≥5 to 10 ms.
3. Time = commit-feedback `(GPUEnd−GPUStart)/K`.
4. Repeat ≥10 times; report median and p10/p90 **[user: median + variance]**.
5. Subtract the measured barrier/launch floor (~2 µs dependent, ~0.2 µs independent) only when comparing to
   per-dispatch numbers from other frameworks.

**Per-kernel breakdown inside a model** (profile mode): MTL4 Precise timestamps around each dispatch. Calibrate the
empty-pair cost once and subtract it. Only trust it for kernels ≥ ~50 µs; otherwise use the method above.

**xctrace from the CLI** works without the GUI **[measured]**:

```
xcrun xctrace record --template 'Metal System Trace' --output run.trace --launch -- ./lk_bench args
xcrun xctrace export --input run.trace --toc
xcrun xctrace export --input run.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' > gpu.xml
```

Useful schemas:
- `metal-gpu-intervals`: per-encoder GPU intervals, with a "CPU to GPU Latency" column.
- `gpu-performance-state-intervals`: shows the induced/minimum performance state; this is how Low Power Mode was
  spotted.
- `metal-gpu-counter-intervals`, `gpu-shader-profiler-*`, `metal-residency-set-*`, `metal-driver-intervals`,
  `metal-command-buffer-completed`.

A small Python parser over the XML can turn this into per-kernel tables. Label the encoders and debug groups with
kernel names.

**Benchmark hygiene [inferred, from the data]:**
- Low Power Mode off.
- Warm up and keep the GPU busy. Idle → commit-to-start latency was 442 µs vs 127 µs back-to-back.
- On iPhone, also log `ProcessInfo.thermalState`, and wait for cool-down between runs.

---

## 7. Offline compile, specialization, MSL version, TensorOps

**Toolchain.**
- On this Mac, `xcrun metal` fails with *"missing Metal Toolchain"*. Since Xcode 26, the Metal compiler is a
  separately downloaded component.
- One-time setup: **`xcodebuild -downloadComponent MetalToolchain`**. Not done by me: it is a system-level install.
  The implementation phase must run it (or ask) before building metallibs.

**Commands** (per platform, since AIR targets differ: air64-apple-macos vs ios):

```
xcrun -sdk macosx   metal -std=metal4.0 -mmacosx-version-min=27.0 -O3 -c k.metal -o k.macos.air
xcrun -sdk iphoneos metal -std=metal4.0 -mios-version-min=27.0   -O3 -c k.metal -o k.ios.air
xcrun -sdk macosx   metallib *.macos.air -o lokinfer_kernels.macos.metallib
xcrun -sdk iphoneos metallib *.ios.air   -o lokinfer_kernels.ios.metallib
# debug/profiling builds: add -frecord-sources -gline-tables-only (source-level shader profiler)
```

- `MTLLanguageVersion4_1` exists in the 27 SDK, so `-std=metal4.1` should be accepted. Verify after installing the
  toolchain.
- TensorOps/MPP need MSL ≥ 4.0 (ggml switches the language version only when `has_tensor` is set).
  `MPPTensorOpsAvailability.h` gates newer features on a deployment target ≥ 26.2 or 27.0.
- CMake: `add_custom_command` per `.metal` → `.air`, then one `metallib` per platform. Put it in the iOS app bundle;
  on macOS, next to the library or embedded as a resource. Load with `device->newLibrary(NS::URL)`.

**Runtime compile cost.**
- Compiling an MPP `matmul2d` kernel *from source* at runtime took **450 to 600 ms** per library **[measured]**.
- Offline metallibs remove the frontend cost. **PSO creation still compiles AIR to GPU ISA at runtime.** The OS
  shader cache makes this cheap after the first launch.
- Create every PSO in the kernel's `prepare()` at load, in parallel with `MTL4Compiler` tasks or async
  `newComputePipelineState`.
- Later, consider `MTL4Archive`/`MTL4PipelineDataSetSerializer` harvesting, so cold starts on the iPhone skip
  backend compilation.

**Function constants.**
- `[[function_constant(i)]]` gives specialization of shapes, tile sizes and epilogue flags from one AIR function.
  Dead code and loops on constant trip counts are folded at PSO creation.
- Classic: `MTL::FunctionConstantValues`. MTL4: `MTL4SpecializedFunctionDescriptor`/`MTL4LibraryFunctionDescriptor`
  with constant values.
- The PSO cache key is (function name, constant values). This fits **[user: many shape/dtype-specific variants]**.
  The Python side picks the kernel name plus constants; C++ only creates the PSO.

**TensorOps / Metal Performance Primitives (`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`).**
- `mpp::tensor_ops::matmul2d<desc, execution_simdgroups<N>|execution_thread>` computes C = A·B + C.
  - Inputs: `tensor<device …>` (host `MTLTensor`, or `tensor_inline` built from raw device pointers inside the kernel;
    ggml and our bench do the latter, so classic `setBuffer` works).
  - Output can be a **cooperative tensor**, so fused epilogues (bias, activation, residual) are done in registers
    before `store()`.
  - Data types (27 SDK header): f16/bf16/f32 mixes, int8×int8→int32, f16×int8/uint8, int4/int2 and fp8 (e4m3/e5m2)
    and fp4 weights with f16/bf16 activations.
- `convolution2d`: NHWC activations, HWIO weights, strides, dilations and groups; **threadgroup scope only**.
  Conv1d is `H=1`.
- HW acceleration: **Neural Accelerators in each GPU core on M5/A19 and later**. Apple claims GEMM math rate
  "4 to 8× depending on precision", and prefill up to 4×. On M1 to M4, "TensorOps falls back to optimized shader
  implementations" (Apple tech talk 111432).
- **M2 Max, measured** (4096³, f16×f16→f32, naive MPP kernel from the header example, Low Power Mode on):

  | Tile | TFLOPS |
  |---|---|
  | 64×32 | 3.2 |
  | **64×64** | **6.7** |
  | 128×64 | 5.8 |

  The M2 Max 30-core peak is ≈10.7 TFLOPS (f16 = f32 rate on Apple8). ggml found TensorOps ~5% slower than its
  `simdgroup_matrix` kernels on M2 Ultra.
- **Implications [inferred]:**
  - **iPhone 18 Pro Max:** GEMM/conv-heavy layers should use MPP `matmul2d`/`convolution2d` with cooperative-tensor
    epilogues. Expect compute-bound GEMM/conv to be several times faster per core than shader code, so iPhone GEMM
    could approach or beat the M2 Max despite fewer cores. Memory-bound ops (elementwise, norms, small convs, most of
    Kokoro's decoder glue) are limited by phone bandwidth instead. A20 Pro is reported at +50% vs A19 Pro.
  - **M2 Max:** hand-written `simdgroup_matrix` (MLX "steel"-style) GEMM/conv. Keep the MPP path as a variant to
    benchmark; it's cheap to try.
  - Kernel naming already allows per-device variants (`…_gpu_mpp`, `…_gpu_sgmm`) **[user: different kernels per
    device OK]**.

---

## 8. CPU↔GPU sync for mixed graphs

**Measured latencies** (M2 Max, Low Power Mode on, bench1 §1, §5 and bench2-A):

| Path | Median | p10 | Notes |
|---|---|---|---|
| empty command buffer commit → `waitUntilCompleted` | 16 to 25 µs | 12 µs | no GPU work; short-circuits |
| 1 tiny dispatch, commit → `waitUntilCompleted` | 132 µs | 124 µs | |
| … → `MTLSharedEvent::waitUntilSignaledValue` | 121 µs | 94 µs | |
| … → spin on `event->signaledValue()` | 117 µs | 80 µs | best latency; burns a core |
| … → spin on `cb->status()` | 287 µs | 168 µs | **bad**, avoid |
| MTL4: commit + `signalEvent` → spin / wait | 116 / 138 µs | 91 / 117 µs | same as classic |
| CPU `setSignaledValue` → GPU (waiting on the event) runs 1 dispatch → CPU sees completion | 117 µs | 105 µs | CPU→GPU handoff |

Decomposition (1 tiny dispatch):

| | commit → GPU start | GPU | GPU end → CPU wake |
|---|---|---|---|
| back-to-back | 127 µs | 7.7 µs | 15 µs |
| after a 3 ms idle | **442 µs** | | |
| right after a heavy kernel | 164 µs | | |

**Design [inferred]:**
- **A crossing costs ~120 µs at best** and drains the GPU, so the runtime minimizes crossings.
- Prefer running small glue ops on the GPU to avoid a segment switch. The kernel choice is explicit in the IR
  **[user]**, but the docs and tooling should warn when a graph alternates devices.
- **GPU → CPU** (CPU kernel consumes a GPU result):
  - End the GPU segment with `q4->signalEvent(ev, v)` and commit.
  - The CPU waits by spinning on `ev->signaledValue()` for up to ~200 µs, then falls back to
    `waitUntilSignaledValue`.
  - Before waiting, **pre-encode and commit all later GPU work that doesn't depend on the CPU result**. It is gated
    by `q4->wait(ev2, v2)`, the GPU-side wait on a CPU-signaled event.
- **CPU → GPU** (GPU kernel consumes a CPU result):
  - The GPU segment is already committed behind `q4->wait(ev2, v2)`.
  - When the CPU kernel finishes, it calls `ev2->setSignaledValue(v2)`. The GPU picks it up with no encode on the
    critical path.
- Memory is shared, so no copies are needed. The event's signal/wait gives the needed ordering and visibility.
- `model()` is synchronous: final signal, then spin-then-block wait. `submit()`/`wait()` expose the event value to
  C++ callers **[user]**.
- One `MTLSharedEvent` per executable with a monotonically increasing value. A second event is only needed for
  CPU→GPU gates.

---

## 9. Proposed GPU runtime design

### 9.1 Objects (created once, at model load)
- `GpuContext` (per process): `MTL::Device`, one `MTL4::CommandQueue`, `MTL4::Compiler`, a ring of
  `kInFlight = 3` × {`MTL4::CommandAllocator`, `MTL4::CommandBuffer`}, one `MTL4::ArgumentTable`
  (`maxBufferBindCount = 16`), and `MTLSharedEvent`s.
- `GpuExecutable` (per compiled model):
  - The residency set (§5).
  - The **arena**: one Shared/Untracked `MTLBuffer`, pre-faulted.
  - **Weights**: no-copy `MTLBuffer`s over the mmapped `.lkm`.
  - The **param buffer** (Shared): per-dispatch dims/scalars blocks.
  - The **dim buffer** (Shared): runtime dim values and indirect-dispatch args.
  - PSOs from `prepare()`.
  - The **dispatch plan** (§9.2).

### 9.2 Flat dispatch plan (the "record" step, built in init)

```cpp
struct GpuDispatch {
  const MTL::ComputePipelineState* pso;
  uint8_t  nbind;  uint8_t barrier_before;      // barrier only if this op depends on the previous level
  uint32_t bind_slot[8];                        // index into the GPU-address table (arena/weights/io/param/dim)
  uint64_t bind_off[8];                         // static offsets (or symbolic → patched per call)
  MTL::Size grid, tg;                           // static grid, or:
  int32_t  indirect_args_off;                   // ≥0 → dispatchThreadgroups(dimBuf.gpuAddress()+off, tg)
  uint32_t param_off;                           // this op's dims/scalars block in the param buffer
};
```

- **Kernel ABI on the GPU side [inferred, to align with `design-runtime-and-kernels.md` §1]:**
  - Tensor/buffer arguments in buffer slots `0..n-1`.
  - Then one `constant Params&` slot (dims/scalars) and optionally a `device const DimVals*` slot.
  - The same kernel works with the MTL4 argument table (`setAddress`), the classic `setBuffer` (benchmarks,
    MPS-interop tests) and ICB `setKernelBuffer`.
  - `BufRef{buffer, offset}` resolves to `buffer->gpuAddress() + offset` once at load. Per call, only inputs/outputs
    and symbolic offsets change.
- **Barrier placement**:
  - The compiler computes dependency levels from the memory plan's arena ranges (RAW/WAR/WAW).
  - The runtime emits `barrierAfterEncoderStages(Dispatch, Dispatch, None)` once per level boundary.
  - Within a level, dispatches run concurrently. MTL4 dispatches are concurrent by default: 0.19 µs each vs 2 µs
    for dependent ones.
- Params for static shapes are written into the param buffer at init. For dynamic shapes, the CPU dim program
  writes them per call: a memcpy into shared memory, no GPU sync.

### 9.3 Strategy A: direct encode (default)
Per call:
1. Wait until the ring slot's previous use has completed (event value). Then `allocator.reset()` and
   `cb.beginCommandBuffer(alloc)`, then `enc = cb.computeCommandEncoder()` and `enc.setArgumentTable(table)`.
2. For each dispatch:
   - `[barrier]`;
   - `setComputePipelineState` (skip if unchanged);
   - `table.setAddress` for each binding;
   - `dispatchThreadgroups` (or the indirect variant).
3. **Chunked commit.** The first chunk is ~8 to 16 dispatches, so the GPU starts within ~30 µs of encode start.
   Later chunks are ~64 to 256 dispatches, or are cut by estimated GPU time.
   - Each chunk is its own `MTL4::CommandBuffer` from the ring.
   - Chunk boundaries are queue-level boundaries: put them **at a level boundary**. The next chunk's first dispatch
     needs a `barrierAfterQueueStages(Dispatch, Dispatch)` only if it depends on the previous chunk.
   - Measure the cost of a cross-command-buffer dependency on the queue before settling chunk sizes. An encoder per
     dispatch with queue barriers measured 33 µs/dispatch, so chunk boundaries must be rare.
4. `q4.signalEvent(ev, ++v)`, then wait (§8).

Expected CPU cost: ~0.15 to 0.5 µs/dispatch, hidden behind GPU execution except for the first chunk.

### 9.4 Strategy B: ICB replay (optional, per model, `gpu.replay` segments)
- **Init**: record each segment into one `MTL::IndirectCommandBuffer`:
  - commands with PSO, bindings and grid;
  - `setBarrier()` on the first command of each level;
  - ICB, PSOs and buffers added to the residency set.
- **Per call**:
  - **Patch list.** Commands whose bindings or grid depend on the call get `setKernelBuffer`/
    `concurrentDispatchThreadgroups` re-issued: 0.04 to 0.18 µs each.
  - Then one `executeCommandsInBuffer(icb, range)` per segment, in one command buffer.
- **Data-dependent grids**: upper-bound grid plus early exit, or a tiny GPU "ICB patch" kernel that writes
  `compute_command` dispatch sizes, or `executeCommandsInBuffer(icb, indirectRangeBuffer)` to select the command
  range on the GPU.
- Expected: ~20 to 50 µs CPU per call regardless of size; ~0.6 µs/dependent dispatch slower on the GPU (M2 Max).
  Worth it for tiny graphs, mixed CPU/GPU graphs, and possibly on iPhone (energy). Decide by benchmark
  **[user: compare]**.

### 9.5 Data-dependent dims (Kokoro durations → frames)
- Default (GPU-driven, §4):
  - The producing kernel writes the dim value into the dim buffer.
  - A tiny generic **`lk_dim_eval` kernel** evaluates the derived dims (the IR's dim expressions, compiled to a small
    bytecode in the param buffer) and the indirect-dispatch args, all on the GPU.
  - Consumer kernels read their dims from the dim buffer and use indirect dispatch.
  - Buffers are planned with the declared upper bound.
  - After the final completion, the CPU reads the dim buffer to size and trim the returned tensors.
- Alternative strategy (flag): readback sync at `dim.read`. Pre-encode independent work before waiting.

### 9.6 Long sequential loops (LSTM over time, iSTFT frames) [inferred]
- At ~2 µs per dependent dispatch, a per-timestep dispatch loop is too slow: hundreds of steps × layers = ms.
- Kernels must loop over time internally: one dispatch per layer and direction, with weights in registers or
  threadgroup memory, and `threadgroup_barrier` between steps. This needs a single threadgroup or a cluster with
  device-scope sync.
- macOS/iOS 27's `forwardProgressUsage`/`optimizeForPersistentKernel` APIs may make multi-threadgroup persistent
  kernels legal. They are undocumented in the headers, so research them when the LSTM kernels are written.

### 9.7 Profiling hooks
- A build/run flag enables MTL4 Precise timestamps around each dispatch, with the calibrated overhead subtracted.
- Every dispatch is labeled through `pushDebugGroup` with the IR op name, so xctrace and the Metal debugger show
  kernel names.
- Per-call `MTL4CommitFeedback` GPU time and the event-based wall time are always available.

---

## 10. Benchmark sources and raw results

The benchmark is in a local scratch directory (not in the repo).
That directory is ephemeral, so **TODO**: port it into `lokinfer/bench/gpu_runtime/` as a tool. It is plain
metal-cpp C++20 and needs no Metal toolchain, because MSL is compiled at runtime from source strings.

| File | What it measures |
|---|---|
| `bench.cpp` | §1 round trips; §2 encode/GPU cost for N=10/100/1000 across 5 classic and 2 MTL4 variants; §3 ICB record/update/replay (classic and MTL4); §4 data-dependent strategies; §5 CPU→GPU event; §6 MTL4 timestamps; §7 bandwidth shared/private/no-copy mmap; §8 placement heap aliasing. Output: `run1.txt` |
| `bench2.cpp` | A latency decomposition; B warm steady state (table §2.4); C mid-graph sync cost; D timestamps (MTL4 Relaxed/Precise raw, classic stage boundary). Output: `run2.txt` |
| `bench3.cpp` | classic counter units (ns) |
| `bench4.cpp` | MTL4 argument-table snapshot semantics and the re-commit assertion |
| `bench_mpp.cpp` | MPP `matmul2d` TFLOPS on M2 Max |
| `mst.trace` | xctrace Metal System Trace sample |

Build: `clang++ -std=c++20 -O2 -I metal-cpp -I . benchX.cpp -framework Metal -framework Foundation -framework QuartzCore -o benchX`

The rep counts are fixed, which is intentional: these are bounded measurement runs, not live-driven tests.

## 11. Open items / TODO for the implementation phase
1. **Re-run all benches with Low Power Mode off.** Then run them **on the iPhone 18 Pro Max** through the test host
   app, adding the `supportsFamily(Apple10/Apple11)` and `maxBufferLength` probes.
2. Install the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`). Verify the `-std=metal4.1` spelling
   and the AIR → metallib → `newLibrary(URL)` flow for both platforms.
3. Measure the cost of queue-level dependencies between chunked MTL4 command buffers, to size chunks in §9.3.
4. Compare the bindless params-block ABI with direct buffer slots, for kernel performance and encode cost.
5. Measure zero-copy caller I/O buffers via a per-call `useResidencySet` vs copying into the arena.
6. Measure PSO creation time from a metallib (cold vs OS-cached) on both devices, and decide whether `MTL4Archive`
   harvesting is needed.
7. Benchmark GEMM: MPP vs `simdgroup_matrix` vs MPS (`MPSNDArrayMatrixMultiplication encodeWithMTL4CommandEncoder`)
   on both devices, for the Kokoro shapes.
8. Research the persistent-kernel APIs (27) for the LSTM/sequential kernels.

## Sources
- metal-cpp: https://github.com/apple/metal-cpp (tags, README changelog), https://developer.apple.com/metal/cpp/
- WWDC25 "Discover Metal 4": https://developer.apple.com/videos/play/wwdc2025/205/
- WWDC25 "Combine Metal 4 machine learning and graphics": https://developer.apple.com/videos/play/wwdc2025/262/
- Tech talk "Accelerate your machine learning workloads with the M5 and A19 GPUs": https://developer.apple.com/videos/play/tech-talks/111432/
- Apple ML research, MLX on M5: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- ICB docs: https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer, https://developer.apple.com/documentation/metal/indirect-command-encoding, `executeCommandsInBuffer:withRange:` (16,384 range limit on Mac2 only)
- Xcode 27 Metal API diff: https://github.com/dotnet/macios/wiki/Metal-macOS-xcode27.0-b1
- Metal Toolchain unbundling: https://origin-devforums.apple.com/forums/thread/787221
- A20 Pro GPU Neural Accelerators: https://www.notebookcheck.net/Apple-A20-Pro-debuts-with-7-core-GPU-32-core-Neural-Engine-and-50-more-memory-bandwidth.1395027.0.html, https://aiweekly.co/alerts/apples-iphone-18-pro-debuts-2nm-a20-pro-with-32-neural-engine-cores-2x-fp8-on
- Source read: ggml `ggml/src/ggml-metal/ggml-metal-{device,context}.m` (llama.cpp master), MLX `mlx/backend/metal/{device.cpp,allocator.cpp,resident.cpp}` (main), PyTorch `aten/src/ATen/mps/{MPSStream,MPSAllocator}.mm` (main), all downloaded 2026-11-01.
- Local SDK headers: Metal.framework (MTL4*.h, MTLDevice.h, MTLComputePipeline.h, MTLLibrary.h), MetalPerformancePrimitives.framework (MPPTensorOpsMatMul2d.h, MPPTensorOpsConvolution2d.h), MPSNDArrayKernel.h, MPSGraph.h (MacOSX27 SDK).
