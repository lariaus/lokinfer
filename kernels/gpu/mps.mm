// Apple-library GPU kernels (MetalPerformanceShaders) encoded into lokinfer's
// own Metal 4 compute encoder (MPS gained MTL4 entry points in the 27 SDKs).
// Objective-C++ with ARC; metal-cpp objects are bridged as plain pointers.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../../src/runtime/gpu_internal.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

namespace {

struct MpsGemmState {
  MPSNDArrayMatrixMultiplication* __strong kernel;
};

Status prepareMpsGemm(PrepareCtx& pc, const Op&, void** state) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)pc.gpu->device();
  auto* s = new MpsGemmState();
  s->kernel = [[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:2];
  s->kernel.alpha = 1.0;
  s->kernel.beta = 0.0;
  *state = s;
  return Status::ok();
}

void destroyMpsGemm(void* s) { delete static_cast<MpsGemmState*>(s); }

// Row-major [rows, cols] f32 matrix -> MPSNDArray (dimension 0 = fastest = cols).
MPSNDArray* ndarray(const BufRef& b, int64_t rows, int64_t cols) {
  NSUInteger sizes[2] = {static_cast<NSUInteger>(cols), static_cast<NSUInteger>(rows)};
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                          dimensionCount:2
                                                          dimensionSizes:sizes];
  d.preferPackedRows = YES;
  return [[MPSNDArray alloc] initWithBuffer:(__bridge id<MTLBuffer>)b.mtlBuffer offset:b.offset descriptor:d];
}

// ints: a(M, K), b(K, N), out(M, N)
void runMpsGemm(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  int64_t m = a.ints[0], k = a.ints[1], n = a.ints[3];
  // No local autorelease pool: Metal 4 command buffers don't retain
  // resources, so MPS's temporaries must outlive the segment. GpuContext keeps
  // a pool open per segment and retains these arrays until completion.
  MPSNDArray* A = ndarray(a.bufs[0], m, k);
  MPSNDArray* B = ndarray(a.bufs[1], k, n);
  MPSNDArray* C = ndarray(a.bufs[2], m, n);
  id<MTL4ComputeCommandEncoder> enc = (__bridge id<MTL4ComputeCommandEncoder>)(void*)g.beginDispatch();
  [static_cast<const MpsGemmState*>(a.state)->kernel encodeWithMTL4CommandEncoder:enc
                                                                      sourceArrays:@[ A, B ]
                                                                  destinationArray:C];
  g.noteExternalDispatch();
  g.keepAlive((__bridge_retained void*)A);
  g.keepAlive((__bridge_retained void*)B);
  g.keepAlive((__bridge_retained void*)C);
}

// out[M, N] = a[M, K] @ b[N, K]^T — B is read through a transposed descriptor (no copy; GPT-2's tied LM head).
// ints: a(M, K), b(N, K), out(M, N)
void runMpsGemmNt(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  int64_t m = a.ints[0], k = a.ints[1], n = a.ints[2];
  MPSNDArray* A = ndarray(a.bufs[0], m, k);
  NSUInteger sizes[2] = {static_cast<NSUInteger>(k), static_cast<NSUInteger>(n)};  // physical [N, K]
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                          dimensionCount:2
                                                          dimensionSizes:sizes];
  d.preferPackedRows = YES;
  [d transposeDimension:0 withDimension:1];  // logical [K, N]
  MPSNDArray* B = [[MPSNDArray alloc] initWithBuffer:(__bridge id<MTLBuffer>)a.bufs[1].mtlBuffer
                                              offset:a.bufs[1].offset
                                          descriptor:d];
  MPSNDArray* C = ndarray(a.bufs[2], m, n);
  id<MTL4ComputeCommandEncoder> enc = (__bridge id<MTL4ComputeCommandEncoder>)(void*)g.beginDispatch();
  [static_cast<const MpsGemmState*>(a.state)->kernel encodeWithMTL4CommandEncoder:enc
                                                                      sourceArrays:@[ A, B ]
                                                                  destinationArray:C];
  g.noteExternalDispatch();
  g.keepAlive((__bridge_retained void*)A);
  g.keepAlive((__bridge_retained void*)B);
  g.keepAlive((__bridge_retained void*)C);
}

}  // namespace

void registerGpuMpsKernels() {
  registerKernel({"gemm_f32_gpu_mps", Device::Gpu, prepareMpsGemm, runMpsGemm, destroyMpsGemm});
  registerKernel({"gemm_nt_f32_gpu_mps", Device::Gpu, prepareMpsGemm, runMpsGemmNt, destroyMpsGemm});
}

}  // namespace lk::rt
