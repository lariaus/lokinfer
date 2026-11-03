// GPU microbench: FMA peak f32/f16, read/copy bandwidth, dispatch overhead, MPS matmul f32/f16, MPSGraph matmul,
// a simple simdgroup_matrix GEMM, and a simple GEMV.
// Build: clang++ -O3 -std=c++20 -fobjc-arc gpu.mm -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation -o gpu
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

template<typename T> kernel void fma_peak(device T* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  T a0 = T(gid)*T(1e-3), a1=a0+T(1), a2=a0+T(2), a3=a0+T(3), a4=a0+T(4), a5=a0+T(5), a6=a0+T(6), a7=a0+T(7);
  T b = T(0.999), c = T(1e-4);
  for (uint i = 0; i < iters; i++) {
    a0 = fma(a0,b,c); a1 = fma(a1,b,c); a2 = fma(a2,b,c); a3 = fma(a3,b,c);
    a4 = fma(a4,b,c); a5 = fma(a5,b,c); a6 = fma(a6,b,c); a7 = fma(a7,b,c);
  }
  out[gid] = a0+a1+a2+a3+a4+a5+a6+a7;
}
template [[host_name("fma_f32")]] kernel void fma_peak<float>(device float*, constant uint&, uint);
template [[host_name("fma_f16")]] kernel void fma_peak<half>(device half*, constant uint&, uint);

kernel void read_bw(device const float4* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  float4 s = 0;
  for (uint i = gid; i < n4; i += gsz) s += in[i];
  if (s.x == -1.0f) out[gid] = s.y;   // never true, keeps loads alive
}
kernel void copy_bw(device const float4* in [[buffer(0)]], device float4* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = in[gid];
}
kernel void empty_k(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) { if (gid == 0xffffffff) out[0] = 1; }

// Simple simdgroup_matrix GEMM: C[M,N] = A[M,K] * B[K,N], row-major, M,N multiple of 32, K multiple of 8.
// Threadgroup = 4 simdgroups (2x2), each computing 32x32 of a 64x64 block, loading straight from device memory.
template<typename T>
kernel void sg_gemm(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]], device T* C [[buffer(2)]],
                    constant uint3& MNK [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  uint M = MNK.x, N = MNK.y, K = MNK.z;
  uint row0 = tg.y * 64 + (sg / 2) * 32, col0 = tg.x * 64 + (sg % 2) * 32;
  if (row0 >= M || col0 >= N) return;
  simdgroup_matrix<float, 8, 8> acc[4][4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  for (uint k = 0; k < K; k += 8) {
    simdgroup_matrix<T, 8, 8> a[4], b[4];
    for (int i = 0; i < 4; i++) simdgroup_load(a[i], A + (row0 + 8 * i) * K + k, K);
    for (int j = 0; j < 4; j++) simdgroup_load(b[j], B + k * N + col0 + 8 * j, N);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
    simdgroup_matrix<T, 8, 8> o; o.thread_elements()[0] = T(acc[i][j].thread_elements()[0]); o.thread_elements()[1] = T(acc[i][j].thread_elements()[1]);
    simdgroup_store(o, C + (row0 + 8 * i) * N + col0 + 8 * j, N);
  }
}
template [[host_name("sg_gemm_f32")]] kernel void sg_gemm<float>(device const float*, device const float*, device float*, constant uint3&, uint2, uint);
template [[host_name("sg_gemm_f16")]] kernel void sg_gemm<half>(device const half*, device const half*, device half*, constant uint3&, uint2, uint);

// GEMV y[N] = W[N,K] x[K]: one simdgroup per 4 rows (half4/float4 loads, simd_sum).
template<typename T, typename T4>
kernel void gemv(device const T4* W [[buffer(0)]], device const T4* x [[buffer(1)]], device T* y [[buffer(2)]], constant uint2& NK [[buffer(3)]],
                 uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint N = NK.x, K4 = NK.y / 4;
  uint r0 = (tgid * 8 + sg) * 4;   // 8 simdgroups per threadgroup, 4 rows each
  if (r0 >= N) return;
  float acc[4] = {0, 0, 0, 0};
  for (uint k = lane; k < K4; k += 32) {
    float4 xv = float4(x[k]);
    for (int r = 0; r < 4; r++) acc[r] += dot(float4(W[(r0 + r) * K4 + k]), xv);
  }
  for (int r = 0; r < 4; r++) { float s = simd_sum(acc[r]); if (lane == 0) y[r0 + r] = T(s); }
}
template [[host_name("gemv_f32")]] kernel void gemv<float, float4>(device const float4*, device const float4*, device float*, constant uint2&, uint, uint, uint);
template [[host_name("gemv_f16")]] kernel void gemv<half, half4>(device const half4*, device const half4*, device half*, constant uint2&, uint, uint, uint);
)MSL";

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLLibrary> lib;
static id<MTLComputePipelineState> pso(const char* n) {
  NSError* e = nil;
  id<MTLFunction> f = [lib newFunctionWithName:@(n)];
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
  if (!p) { printf("pso %s: %s\n", n, e.localizedDescription.UTF8String); exit(1); }
  return p;
}
// GPU time of a command buffer encoded by `enc`, median over reps (uses GPUStartTime/GPUEndTime)
template <class F> double gpu_time(F enc, int reps = 15) {
  std::vector<double> ts;
  for (int r = 0; r < reps + 2; r++) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    enc(cb);
    [cb commit]; [cb waitUntilCompleted];
    if (r >= 2) ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
  }
  std::sort(ts.begin(), ts.end()); return ts[ts.size() / 2];
}
static void dispatch1(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, MTLSize grid, MTLSize tg) {
  [e setComputePipelineState:p]; [e dispatchThreads:grid threadsPerThreadgroup:tg];
}

int main() {
  @autoreleasepool {
    dev = MTLCreateSystemDefaultDevice(); q = [dev newCommandQueue];
    printf("Device: %s  maxThreadgroupMem=%lu  recommendedMaxWorkingSet=%.1fGB\n", dev.name.UTF8String, (unsigned long)dev.maxThreadgroupMemoryLength, dev.recommendedMaxWorkingSetSize / 1e9);
    NSError* err = nil; MTLCompileOptions* opt = [MTLCompileOptions new]; opt.mathMode = MTLMathModeFast;
    lib = [dev newLibraryWithSource:@(kSrc) options:opt error:&err];
    if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }

    // ---- FMA peak ----
    for (const char* n : {"fma_f32", "fma_f16"}) {
      auto p = pso(n); uint iters = 4096; size_t threads = 1 << 20;
      id<MTLBuffer> out = [dev newBufferWithLength:threads * 4 options:MTLResourceStorageModeShared];
      double t = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoder]; [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1]; dispatch1(e, p, MTLSizeMake(threads, 1, 1), MTLSizeMake(256, 1, 1)); [e endEncoding]; });
      printf("%s: %.2f TFLOPS\n", n, 2.0 * 8 * iters * threads / t / 1e12);
    }
    // ---- bandwidth ----
    {
      size_t bytes = 1ull << 30; id<MTLBuffer> a = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
      id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
      id<MTLBuffer> o = [dev newBufferWithLength:1 << 24 options:MTLResourceStorageModePrivate];
      auto pr = pso("read_bw"), pc = pso("copy_bw");
      for (size_t sz : {8ull << 20, 32ull << 20, 256ull << 20, 1ull << 30}) {
        uint n4 = (uint)(sz / 16);
        double tr = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoder]; [e setBuffer:a offset:0 atIndex:0]; [e setBuffer:o offset:0 atIndex:1];
          [e setBytes:&n4 length:4 atIndex:2]; dispatch1(e, pr, MTLSizeMake(30 * 1024 * 4, 1, 1), MTLSizeMake(1024, 1, 1)); [e endEncoding]; });
        double tc = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoder]; [e setBuffer:a offset:0 atIndex:0]; [e setBuffer:b offset:0 atIndex:1];
          dispatch1(e, pc, MTLSizeMake(sz / 16, 1, 1), MTLSizeMake(1024, 1, 1)); [e endEncoding]; });
        printf("bandwidth %5zu MB: read %6.1f GB/s   copy %6.1f GB/s (r+w)\n", sz >> 20, sz / tr / 1e9, 2.0 * sz / tc / 1e9);
      }
    }
    // ---- dispatch overhead ----
    {
      auto p = pso("empty_k"); id<MTLBuffer> o = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
      for (int nd : {1, 100, 1000}) {
        double t = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoder]; [e setComputePipelineState:p]; [e setBuffer:o offset:0 atIndex:0];
          for (int i = 0; i < nd; i++) [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; });
        printf("dispatch: %4d tiny dispatches in 1 encoder (serial): GPU %.2f us total, %.2f us/dispatch\n", nd, t * 1e6, t * 1e6 / nd);
      }
      for (int nd : {100, 1000}) {
        double t = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; [e setComputePipelineState:p]; [e setBuffer:o offset:0 atIndex:0];
          for (int i = 0; i < nd; i++) [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; });
        printf("dispatch: %4d tiny dispatches concurrent encoder: GPU %.2f us/dispatch\n", nd, t * 1e6 / nd);
      }
      // with memory barrier between each (dependent chain as in an LSTM recurrence)
      for (int nd : {100, 1000}) {
        double t = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; [e setComputePipelineState:p]; [e setBuffer:o offset:0 atIndex:0];
          for (int i = 0; i < nd; i++) { [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; } [e endEncoding]; });
        printf("dispatch: %4d tiny dispatches concurrent + barrier each: GPU %.2f us/dispatch\n", nd, t * 1e6 / nd);
      }
      // CPU-side round trip: commit + wait of an (almost) empty command buffer
      std::vector<double> ts;
      for (int r = 0; r < 200; r++) { double t0 = now(); id<MTLCommandBuffer> cb = [q commandBuffer]; auto e = [cb computeCommandEncoder];
        dispatch1(e, p, MTLSizeMake(32, 1, 1), MTLSizeMake(32, 1, 1)); [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts.push_back(now() - t0); }
      std::sort(ts.begin(), ts.end());
      printf("CPU round trip (encode+commit+wait, 1 dispatch): median %.1f us, p10 %.1f us, p90 %.1f us\n", ts[100] * 1e6, ts[20] * 1e6, ts[180] * 1e6);
      ts.clear();
      for (int r = 0; r < 100; r++) { double t0 = now(); id<MTLCommandBuffer> cb = [q commandBuffer]; auto e = [cb computeCommandEncoder]; [e setComputePipelineState:p]; [e setBuffer:o offset:0 atIndex:0];
        for (int i = 0; i < 100; i++) [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; ts.push_back(now() - t0); [cb commit]; [cb waitUntilCompleted]; }
      std::sort(ts.begin(), ts.end());
      printf("CPU encode cost: 100 dispatches = %.1f us (%.2f us/dispatch)\n", ts[50] * 1e6, ts[50] * 1e4);
    }

    // ---- matmul: MPS vs MPSGraph vs simple simdgroup kernel ----
    struct S { int M, N, K; const char* w; } shapes[] = {{512, 768, 768, "512x768x768"}, {64, 768, 768, "64x768x768"}, {512, 3072, 768, "512x3072x768"},
                                                        {512, 512, 1536, "512x512x1536 (conv1d k3)"}, {2048, 2048, 2048, "2048^3"}, {4096, 4096, 4096, "4096^3"}};
    printf("\n%-26s | %-28s | %-28s | %-28s\n", "GEMM (TFLOPS, GPU time)", "MPSMatrixMultiplication", "MPSGraph matmul", "simple sg_gemm (no tg mem)");
    for (bool f16 : {false, true}) {
      printf("-- %s --\n", f16 ? "f16" : "f32");
      int es = f16 ? 2 : 4; MPSDataType dt = f16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
      auto psg = pso(f16 ? "sg_gemm_f16" : "sg_gemm_f32");
      for (auto s : shapes) {
        int M = s.M, N = s.N, K = s.K; double flop = 2.0 * M * N * K;
        id<MTLBuffer> A = [dev newBufferWithLength:(size_t)M * K * es options:MTLResourceStorageModeShared];
        id<MTLBuffer> B = [dev newBufferWithLength:(size_t)K * N * es options:MTLResourceStorageModeShared];
        id<MTLBuffer> C = [dev newBufferWithLength:(size_t)M * N * es options:MTLResourceStorageModeShared];
        memset(A.contents, 0, A.length); memset(B.contents, 0, B.length);
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * es dataType:dt]];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N * es dataType:dt]];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * es dataType:dt]];
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1 beta:0];
        double tm = gpu_time([&](id<MTLCommandBuffer> cb) { [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC]; });
        // MPSGraph
        MPSGraph* g = [MPSGraph new];
        MPSGraphTensor* ta = [g placeholderWithShape:@[@(M), @(K)] dataType:dt name:nil];
        MPSGraphTensor* tb = [g placeholderWithShape:@[@(K), @(N)] dataType:dt name:nil];
        MPSGraphTensor* tc = [g matrixMultiplicationWithPrimaryTensor:ta secondaryTensor:tb name:nil];
        MPSGraphTensorData* da = [[MPSGraphTensorData alloc] initWithMTLBuffer:A shape:@[@(M), @(K)] dataType:dt];
        MPSGraphTensorData* db = [[MPSGraphTensorData alloc] initWithMTLBuffer:B shape:@[@(K), @(N)] dataType:dt];
        MPSGraphTensorData* dc = [[MPSGraphTensorData alloc] initWithMTLBuffer:C shape:@[@(M), @(N)] dataType:dt];
        MPSGraphExecutable* ex = [g compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:dev] feeds:@{ta : [[MPSGraphShapedType alloc] initWithShape:@[@(M), @(K)] dataType:dt], tb : [[MPSGraphShapedType alloc] initWithShape:@[@(K), @(N)] dataType:dt]}
                                     targetTensors:@[tc] targetOperations:nil compilationDescriptor:nil];
        double tg = gpu_time([&](id<MTLCommandBuffer> cb) {
          MPSCommandBuffer* mcb = [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
          [ex encodeToCommandBuffer:mcb inputsArray:@[da, db] resultsArray:@[dc] executionDescriptor:nil];
        });
        // wall-clock for MPSGraph run (includes CPU overhead)
        double tgw; { std::vector<double> ws; for (int r = 0; r < 12; r++) { double t0 = now(); [ex runWithMTLCommandQueue:q inputsArray:@[da, db] resultsArray:@[dc] executionDescriptor:nil]; ws.push_back(now() - t0); } std::sort(ws.begin(), ws.end()); tgw = ws[6]; }
        double ts = -1;
        if (M % 32 == 0 && N % 32 == 0) {
          uint32_t mnk[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0};
          ts = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoder]; [e setComputePipelineState:psg]; [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
            [e setBytes:mnk length:16 atIndex:3]; [e dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; [e endEncoding]; });
        }
        printf("%-26s | %6.2f TF %8.1f us        | %6.2f TF %7.1f us (wall %6.0f) | %6.2f TF %8.1f us\n", s.w, flop / tm / 1e12, tm * 1e6, flop / tg / 1e12, tg * 1e6, tgw * 1e6,
               ts > 0 ? flop / ts / 1e12 : 0, ts * 1e6);
      }
    }
    // ---- GEMV ----
    printf("\n%-22s | %-24s | %-24s | %-24s\n", "GEMV (batch 1)", "custom f32", "custom f16", "MPSMatrixVectorMult f32");
    struct G { int N, K; const char* w; } gv[] = {{768, 768, "768x768"}, {3072, 768, "3072x768"}, {1024, 256, "1024x256 LSTM rec"}, {2304, 768, "2304x768"}, {50256, 768, "50256x768 lm_head"}};
    for (auto s : gv) {
      int N = s.N, K = s.K;
      id<MTLBuffer> W = [dev newBufferWithLength:(size_t)N * K * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> x = [dev newBufferWithLength:K * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> y = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
      double t[2];
      for (int h = 0; h < 2; h++) {
        auto p = pso(h ? "gemv_f16" : "gemv_f32"); uint32_t nk[2] = {(uint32_t)N, (uint32_t)K};
        t[h] = gpu_time([&](id<MTLCommandBuffer> cb) { auto e = [cb computeCommandEncoder]; [e setComputePipelineState:p]; [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:x offset:0 atIndex:1]; [e setBuffer:y offset:0 atIndex:2];
          [e setBytes:nk length:8 atIndex:3]; [e dispatchThreadgroups:MTLSizeMake((N + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding]; });
      }
      MPSMatrix* mW = [[MPSMatrix alloc] initWithBuffer:W descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * 4 dataType:MPSDataTypeFloat32]];
      MPSVector* vx = [[MPSVector alloc] initWithBuffer:x descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:K dataType:MPSDataTypeFloat32]];
      MPSVector* vy = [[MPSVector alloc] initWithBuffer:y descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:N dataType:MPSDataTypeFloat32]];
      MPSMatrixVectorMultiplication* mv = [[MPSMatrixVectorMultiplication alloc] initWithDevice:dev transpose:NO rows:N columns:K alpha:1 beta:0];
      double tm = gpu_time([&](id<MTLCommandBuffer> cb) { [mv encodeToCommandBuffer:cb inputMatrix:mW inputVector:vx resultVector:vy]; });
      printf("%-22s | %7.1f us %6.0f GB/s | %7.1f us %6.0f GB/s | %7.1f us %6.0f GB/s\n", s.w, t[0] * 1e6, (double)N * K * 4 / t[0] / 1e9, t[1] * 1e6, (double)N * K * 2 / t[1] / 1e9,
             tm * 1e6, (double)N * K * 4 / tm / 1e9);
    }
  }
}
