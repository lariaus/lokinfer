// Dispatch functions for the GPU kernels in kernels/gpu/*.metal.
#include <Metal/Metal.hpp>

#include "../../src/runtime/gpu_internal.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

namespace {

struct PsoState {
  MTL::ComputePipelineState* pso;
};

template <const char* Function>
Status preparePso(PrepareCtx& pc, const Op&, void** state) {
  MTL::ComputePipelineState* pso = pc.gpu->pipeline(Function);
  if (!pso) return Status::error(std::string("missing GPU kernel function ") + Function);
  *state = new PsoState{pso};
  return Status::ok();
}

void destroyPso(void* s) { delete static_cast<PsoState*>(s); }

void bindArgs(GpuContext& g, const KernelArgs& a) {
  for (uint32_t i = 0; i < a.numPtrs; ++i) g.bind(i, static_cast<MTL::Buffer*>(a.bufs[i].mtlBuffer), a.bufs[i].offset);
}

// Same-shape binary op; n = product of operand 0 dims (rank = numInts / 3).
void runBinary(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  int64_t rank = a.numInts / 3, n = 1;
  for (int64_t i = 0; i < rank; ++i) n *= a.ints[i];
  struct {
    uint32_t n;
  } params{static_cast<uint32_t>(n)};
  g.beginDispatch();
  bindArgs(g, a);
  g.bindParams(3, &params, sizeof params);
  uint64_t threads = (static_cast<uint64_t>(n) + 3) / 4;
  g.dispatch(static_cast<const PsoState*>(a.state)->pso, threads, 1, 1, 256, 1, 1);
}

// ints: a(M, K), b(K, N), out(M, N)
void runGemm(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t m, n, k;
  } params{static_cast<uint32_t>(a.ints[0]), static_cast<uint32_t>(a.ints[3]), static_cast<uint32_t>(a.ints[1])};
  g.beginDispatch();
  bindArgs(g, a);
  g.bindParams(3, &params, sizeof params);
  g.dispatchThreadgroups(static_cast<const PsoState*>(a.state)->pso, (params.n + 31) / 32, (params.m + 31) / 32, 1,
                         128, 1, 1);
}

// ints: in(R, C), out(C, R); buffers in, out, params.
void runTranspose(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, cols;
  } params{static_cast<uint32_t>(a.ints[0]), static_cast<uint32_t>(a.ints[1])};
  g.beginDispatch();
  bindArgs(g, a);
  g.bindParams(2, &params, sizeof params);
  g.dispatchThreadgroups(static_cast<const PsoState*>(a.state)->pso, (params.cols + 31) / 32, (params.rows + 31) / 32,
                         1, 32, 8, 1);
}

constexpr char kAdd[] = "add_f32";
constexpr char kMul[] = "mul_f32";
constexpr char kGemm[] = "gemm_f32";
constexpr char kGemmBias[] = "gemm_bias_f32";
constexpr char kGemmBiasRelu[] = "gemm_bias_relu_f32";
constexpr char kGemmBiasSmallM[] = "gemm_bias_smallm_f32";
constexpr char kGemmBiasReluSmallM[] = "gemm_bias_relu_smallm_f32";
constexpr char kTranspose[] = "transpose_f32";

}  // namespace

void registerGpuKernels() {
  registerKernel({"add_f32_gpu", Device::Gpu, preparePso<kAdd>, runBinary, destroyPso});
  registerKernel({"mul_f32_gpu", Device::Gpu, preparePso<kMul>, runBinary, destroyPso});
  registerKernel({"gemm_f32_gpu", Device::Gpu, preparePso<kGemm>, runGemm, destroyPso});
  registerKernel({"transpose_f32_gpu", Device::Gpu, preparePso<kTranspose>, runTranspose, destroyPso});
}

}  // namespace lk::rt
