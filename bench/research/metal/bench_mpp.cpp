#include "common.inc"
static const char* kMM = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void mm(device half* pA [[buffer(0)]], device half* pB [[buffer(1)]], device float* pC [[buffer(2)]],
               constant int3& mnk [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = mnk.x, N = mnk.y, K = mnk.z;
  auto A = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(pA, dextents<int32_t, 2>(K, M));
  auto B = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(pB, dextents<int32_t, 2>(N, K));
  auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(pC, dextents<int32_t, 2>(N, M));
  constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
  matmul2d<d, execution_simdgroups<4>> op;
  auto mA = A.slice(0, tgid.y * TM);
  auto mB = B.slice(tgid.x * TN, 0);
  auto mC = C.slice(tgid.x * TN, tgid.y * TM);
  op.run(mA, mB, mC);
}
)MSL";
int main() {
  mach_timebase_info_data_t tb; mach_timebase_info(&tb); g_tb = (double)tb.numer / tb.denom;
  auto top = NS::AutoreleasePool::alloc()->init();
  auto dev = MTL::CreateSystemDefaultDevice(); auto q = dev->newCommandQueue(); NS::Error* err = nullptr;
  const int M = 4096, N = 4096, K = 4096;
  int tiles[][2] = {{64, 32}, {64, 64}, {128, 64}};
  auto A = dev->newBuffer((size_t)M * K * 2, MTL::ResourceStorageModeShared);
  auto B = dev->newBuffer((size_t)K * N * 2, MTL::ResourceStorageModeShared);
  auto C = dev->newBuffer((size_t)M * N * 4, MTL::ResourceStorageModeShared);
  __fp16* a = (__fp16*)A->contents(); __fp16* b = (__fp16*)B->contents();
  for (size_t i = 0; i < (size_t)M * K; i++) a[i] = (__fp16)((i % 7) * 0.125f);
  for (size_t i = 0; i < (size_t)K * N; i++) b[i] = (__fp16)((i % 5) * 0.25f);
  for (auto& t : tiles) {
    auto opts = MTL::CompileOptions::alloc()->init();
    opts->setLanguageVersion(MTL::LanguageVersion4_0);
    char defs[64];
    std::string src = "#define TM " + std::to_string(t[0]) + "\n#define TN " + std::to_string(t[1]) + "\n" + kMM;
    uint64_t t0 = now_ticks();
    auto lib = dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), opts, &err);
    if (!lib) { printf("compile failed: %s\n", err->localizedDescription()->utf8String()); return 1; }
    auto pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("mm", NS::UTF8StringEncoding)), &err);
    if (!pso) { printf("pso failed: %s\n", err->localizedDescription()->utf8String()); return 1; }
    uint64_t t1 = now_ticks();
    // Tensors over buffers: dims are (innermost first): A is K x M (extent0 = K), B is N x K, C is N x M
    auto mkT = [&](MTL::Buffer* buf, int inner, int outer, MTL::TensorDataType dt) {
      auto td = MTL::TensorDescriptor::alloc()->init();
      NS::Integer e[2] = {inner, outer}; NS::Integer s[2] = {1, inner};
      td->setDimensions(MTL::TensorExtents::alloc()->init(2, e));
      td->setStrides(MTL::TensorExtents::alloc()->init(2, s));
      td->setDataType(dt);
      td->setUsage(MTL::TensorUsageCompute);
      auto tt = buf->newTensor(td, 0, &err);
      if (!tt) printf("tensor fail: %s\n", err->localizedDescription()->utf8String());
      return tt;
    };
    
    std::vector<double> v;
    for (int r = 0; r < 12; r++) {
      memset(C->contents(), 0, C->length());
      auto cb = q->commandBuffer(); auto e = cb->computeCommandEncoder();
      e->setComputePipelineState(pso);
      e->setBuffer(A, 0, 0); e->setBuffer(B, 0, 1); e->setBuffer(C, 0, 2); int mnk[4] = {M, N, K, 0}; e->setBytes(mnk, 16, 3);
      e->dispatchThreadgroups(MTL::Size(N / t[1], M / t[0], 1), MTL::Size(pso->threadExecutionWidth() * 4, 1, 1));
      e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
      if (r >= 2) v.push_back(2.0 * M * N * K / (cb->GPUEndTime() - cb->GPUStartTime()) / 1e12);
    }
    // spot check C[0][0..2]
    float* c = (float*)C->contents();
    double ref = 0; for (int k = 0; k < K; k++) ref += (double)(float)a[k] * (float)b[(size_t)k * N + 0];
    char nm[128]; snprintf(nm, sizeof nm, "MPP matmul2d f16xf16->f32 4096^3 tile %dx%d (compile %.0f ms) C00 %.1f ref %.1f", t[0], t[1], ticks_to_us(t1 - t0) / 1000, c[0], ref);
    report(nm, v, "TFLOPS");
  }
  top->release();
}
