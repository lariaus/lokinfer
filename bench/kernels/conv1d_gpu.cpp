// M2 conv proof (GPU): times the implicit-GEMM conv1d kernel on Kokoro's
// generator shapes. GPU time = (one command buffer with N dispatches) / N,
// barriers between dispatches (dependent, like a real conv chain).
// Usage: lokinfer_bench_conv1d_gpu [reps]   (needs LOKINFER_BENCH_METALLIB or the default path)
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../../src/runtime/gpu_internal.h"
#include "lokinfer/runtime/runtime.h"

using namespace lk::rt;

namespace {

struct Shape {
  uint32_t c, l, k, d;
};

double runCase(GpuContext& g, MTL::ComputePipelineState* pso, const Shape& s, int n, int reps) {
  uint32_t pad = s.d * (s.k - 1) / 2, lo = s.l + 2 * pad - s.d * (s.k - 1);
  auto x = allocateStorage(size_t(s.c) * s.l * 4, &g);
  auto w = allocateStorage(size_t(s.c) * s.c * s.k * 4, &g);
  auto o = allocateStorage(size_t(s.c) * lo * 4, &g);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1, 1);
  for (size_t i = 0; i < size_t(s.c) * s.l; ++i) static_cast<float*>(x->data)[i] = u(rng);
  for (size_t i = 0; i < size_t(s.c) * s.c * s.k; ++i) static_cast<float*>(w->data)[i] = u(rng);
  struct {
    uint32_t ci, co, l, lo, k, d, pad;
  } prm{s.c, s.c, s.l, lo, s.k, s.d, pad};
  std::vector<double> times;
  for (auto w0 = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - w0 < std::chrono::seconds(1);) {
    for (int i = 0; i < n; ++i) {  // warm-up: ramp the GPU clock before timing
      g.beginDispatch();
      g.bind(0, static_cast<MTL::Buffer*>(x->mtlBuffer), 0);
      g.bind(1, static_cast<MTL::Buffer*>(w->mtlBuffer), 0);
      g.bind(2, static_cast<MTL::Buffer*>(o->mtlBuffer), 0);
      g.bindParams(3, &prm, sizeof prm);
      g.dispatchThreadgroups(pso, (lo + 63) / 64, (s.c + 63) / 64, 1, 128, 1, 1);
    }
    g.commitAndWait();
  }
  for (int r = 0; r < reps + 1; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
      g.beginDispatch();
      g.bind(0, static_cast<MTL::Buffer*>(x->mtlBuffer), 0);
      g.bind(1, static_cast<MTL::Buffer*>(w->mtlBuffer), 0);
      g.bind(2, static_cast<MTL::Buffer*>(o->mtlBuffer), 0);
      g.bindParams(3, &prm, sizeof prm);
      g.dispatchThreadgroups(pso, (lo + 63) / 64, (s.c + 63) / 64, 1, 128, 1, 1);
    }
    g.commitAndWait();
    auto t1 = std::chrono::steady_clock::now();
    if (r > 0) times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / n);
  }
  // Correctness: compare a few output samples with a direct CPU computation.
  const float* xs = static_cast<const float*>(x->data);
  const float* ws = static_cast<const float*>(w->data);
  const float* os = static_cast<const float*>(o->data);
  double maxErr = 0;
  std::mt19937 pick(7);
  for (int q = 0; q < 200; ++q) {
    uint32_t co = pick() % s.c, t = pick() % lo;
    if (q == 0) t = 0;
    if (q == 1) t = lo - 1;
    double ref = 0;
    for (uint32_t ci = 0; ci < s.c; ++ci)
      for (uint32_t k = 0; k < s.k; ++k) {
        int ti = int(t) + int(k * s.d) - int(pad);
        if (ti >= 0 && ti < int(s.l)) ref += double(ws[(size_t(co) * s.c + ci) * s.k + k]) * xs[size_t(ci) * s.l + ti];
      }
    maxErr = std::max(maxErr, std::abs(ref - os[size_t(co) * lo + t]));
  }
  if (maxErr > 1e-3) std::printf("  !! max abs error %.3g\n", maxErr);
  std::sort(times.begin(), times.end());
  return times.front();  // best of reps (after warm-up)
}

// Direct kernel: padded input [Ci, LP], reordered weights [Co, K, Ci], output [Co, LOP].
double runDirect(GpuContext& g, MTL::ComputePipelineState* pso, const Shape& s, int n, int reps) {
  uint32_t pad = s.d * (s.k - 1) / 2, lo = s.l + 2 * pad - s.d * (s.k - 1);
  uint32_t lop = (lo + 63) / 64 * 64, lp = lop + (s.k - 1) * s.d;
  auto xp = allocateStorage(size_t(s.c) * lp * 4, &g);
  auto wr = allocateStorage(size_t(s.c) * s.c * s.k * 4, &g);
  auto o = allocateStorage(size_t(s.c) * lop * 4, &g);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> x(size_t(s.c) * s.l), w(size_t(s.c) * s.c * s.k);
  for (auto& v : x) v = u(rng);
  for (auto& v : w) v = u(rng);
  float* xps = static_cast<float*>(xp->data);
  std::fill(xps, xps + size_t(s.c) * lp, 0.f);
  for (uint32_t ci = 0; ci < s.c; ++ci)
    std::copy(x.begin() + size_t(ci) * s.l, x.begin() + size_t(ci + 1) * s.l, xps + size_t(ci) * lp + pad);
  float* wrs = static_cast<float*>(wr->data);
  for (uint32_t co = 0; co < s.c; ++co)
    for (uint32_t ci = 0; ci < s.c; ++ci)
      for (uint32_t k = 0; k < s.k; ++k) wrs[(size_t(co) * s.k + k) * s.c + ci] = w[(size_t(co) * s.c + ci) * s.k + k];
  struct {
    uint32_t ci, co, lp, lop, k, d;
  } prm{s.c, s.c, lp, lop, s.k, s.d};
  std::vector<double> times;
  for (int r = 0; r < reps + 1; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
      g.beginDispatch();
      g.bind(0, static_cast<MTL::Buffer*>(xp->mtlBuffer), 0);
      g.bind(1, static_cast<MTL::Buffer*>(wr->mtlBuffer), 0);
      g.bind(2, static_cast<MTL::Buffer*>(o->mtlBuffer), 0);
      g.bindParams(3, &prm, sizeof prm);
      g.dispatchThreadgroups(pso, lop / 64, s.c / 64, 1, 128, 1, 1);
    }
    g.commitAndWait();
    auto t1 = std::chrono::steady_clock::now();
    if (r > 0) times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / n);
  }
  const float* os = static_cast<const float*>(o->data);
  double maxErr = 0;
  std::mt19937 pick(7);
  for (int q = 0; q < 200; ++q) {
    uint32_t co = pick() % s.c, t = q == 0 ? 0 : q == 1 ? lo - 1 : pick() % lo;
    double ref = 0;
    for (uint32_t ci = 0; ci < s.c; ++ci)
      for (uint32_t k = 0; k < s.k; ++k) {
        int ti = int(t) + int(k * s.d) - int(pad);
        if (ti >= 0 && ti < int(s.l)) ref += double(w[(size_t(co) * s.c + ci) * s.k + k]) * x[size_t(ci) * s.l + ti];
      }
    maxErr = std::max(maxErr, std::abs(ref - os[size_t(co) * lop + t]));
  }
  if (maxErr > 1e-3) std::printf("  !! max abs error %.3g\n", maxErr);
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

// MLX steel implicit-GEMM conv, channel-last: x [L, Ci], w [Co, K, Ci], out [Lo, Co].
struct MLXConvParams2 {
  int N, C, O, iS[2], wS[2], oS[2], str[2], pad[2], kdil[2], idil[2];
  int64_t in_strides[4], wt_strides[4], out_strides[4];
  int groups;
  bool flip;
};
struct ImplicitGemmConv2DParams {
  int M, N, K, gemm_k_iterations, inp_jump_w, inp_jump_h, inp_jump_c, tiles_n, tiles_m, swizzle_log;
};

double runSteel(GpuContext& g, MTL::ComputePipelineState* pso, const Shape& s, int n, int reps) {
  const int C = int(s.c), L = int(s.l), K = int(s.k), d = int(s.d);
  const int pad = d * (K - 1) / 2, lo = L + 2 * pad - d * (K - 1);
  auto x = allocateStorage(size_t(L) * C * 4, &g);
  auto w = allocateStorage(size_t(C) * K * C * 4, &g);
  auto o = allocateStorage(size_t(lo) * C * 4, &g);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1, 1);
  float* xs = static_cast<float*>(x->data);
  float* ws = static_cast<float*>(w->data);
  for (size_t i = 0; i < size_t(L) * C; ++i) xs[i] = u(rng);
  for (size_t i = 0; i < size_t(C) * K * C; ++i) ws[i] = u(rng);
  MLXConvParams2 cp{1, C, C, {1, L}, {1, K}, {1, lo}, {1, 1}, {0, pad}, {1, d}, {1, 1},
                    {int64_t(L) * C, int64_t(L) * C, C, 1}, {int64_t(K) * C, int64_t(K) * C, C, 1},
                    {int64_t(lo) * C, int64_t(lo) * C, C, 1}, 1, false};
  const int bm = 64, bn = 64, bk = 16;
  const int M = lo, N = C, Kg = K * C;
  const int tn = (N + bn - 1) / bn, tm = (M + bm - 1) / bm;
  const int gemmK = K * ((C + bk - 1) / bk);
  const int ijw = int(cp.in_strides[2]) * d, ijh = int(cp.in_strides[1]) * 1;
  ImplicitGemmConv2DParams gp{M, N, Kg, gemmK, ijw, ijh - (K - 1) * ijw, bk - 0 * ijh - (K - 1) * ijw, tn, tm, 0};
  std::vector<double> times;
  auto encode = [&] {
    g.beginDispatch();
    g.bind(0, static_cast<MTL::Buffer*>(x->mtlBuffer), 0);
    g.bind(1, static_cast<MTL::Buffer*>(w->mtlBuffer), 0);
    g.bind(2, static_cast<MTL::Buffer*>(o->mtlBuffer), 0);
    g.bindParams(3, &cp, sizeof cp);
    g.bindParams(4, &gp, sizeof gp);
    g.dispatchThreadgroups(pso, tn, tm, 1, 32, 2, 2);
  };
  for (auto w0 = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - w0 < std::chrono::seconds(1);) {
    for (int i = 0; i < n; ++i) encode();
    g.commitAndWait();
  }
  for (int r = 0; r < reps + 1; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) encode();
    g.commitAndWait();
    if (r > 0) times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n);
  }
  const float* os = static_cast<const float*>(o->data);
  double maxErr = 0;
  std::mt19937 pick(7);
  for (int q = 0; q < 200; ++q) {
    int co = int(pick() % C), t = q == 0 ? 0 : q == 1 ? lo - 1 : int(pick() % lo);
    double ref = 0;
    for (int k = 0; k < K; ++k) {
      int ti = t + k * d - pad;
      if (ti < 0 || ti >= L) continue;
      for (int ci = 0; ci < C; ++ci) ref += double(ws[(size_t(co) * K + k) * C + ci]) * xs[size_t(ti) * C + ci];
    }
    maxErr = std::max(maxErr, std::abs(ref - os[size_t(t) * C + co]));
  }
  if (maxErr > 1e-3) std::printf("  !! max abs error %.3g\n", maxErr);
  std::sort(times.begin(), times.end());
  return times.front();
}

}  // namespace

int main(int argc, char** argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 5;
  if (const char* p = std::getenv("LOKINFER_BENCH_METALLIB")) setMetallibPath(p);
  auto g = GpuContext::shared();
  if (!g) {
    std::fprintf(stderr, "%s\n", g.status().message().c_str());
    return 1;
  }
  const char* variants[] = {"conv1d_igemm_bk8_p0", "conv1d_igemm_bk8_p4", "conv1d_igemm_bk16_p0", "conv1d_igemm_bk16_p4"};
  // Kokoro rand510 (F = 1101): stage 1 = 128 ch x (120F+1), stage 0 = 256 ch x 20F.
  const Shape shapes[] = {{128, 132121, 3, 1}, {128, 132121, 7, 3},  {128, 132121, 11, 5},
                          {256, 22020, 3, 1},  {256, 22020, 7, 3},   {256, 22020, 11, 5}};
  {  // MMA ceiling: 4 simdgroups x 16 MMAs (8x8x8, 1024 flops each) per iteration
    MTL::ComputePipelineState* pso = (*g)->pipeline("mma_peak_f32");
    auto o = allocateStorage(1 << 20, *g);
    uint32_t iters = 4096, groups = 30 * 64;
    double best = 1e30;
    for (int r = 0; r < 8; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      (*g)->beginDispatch();
      (*g)->bind(0, static_cast<MTL::Buffer*>(o->mtlBuffer), 0);
      (*g)->bindParams(1, &iters, sizeof iters);
      (*g)->dispatchThreadgroups(pso, groups, 1, 1, 128, 1, 1);
      (*g)->commitAndWait();
      best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    double flops = double(groups) * 4 * 16 * iters * 1024;
    std::printf("simdgroup_matrix f32 MMA ceiling (Metal 4 queue): %.2f TF\n", flops / best * 1e-12);
    // Same pipeline through a classic MTLCommandQueue (what MLX uses).
    MTL::CommandQueue* q = (*g)->device()->newCommandQueue();
    MTL::Buffer* ob = (*g)->device()->newBuffer(1 << 20, MTL::ResourceStorageModeShared);
    MTL::Buffer* ib = (*g)->device()->newBuffer(&iters, sizeof iters, MTL::ResourceStorageModeShared);
    double best3 = 1e30;
    for (int r = 0; r < 8; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      MTL::CommandBuffer* cb = q->commandBuffer();
      MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
      e->setComputePipelineState(pso);
      e->setBuffer(ob, 0, 0);
      e->setBuffer(ib, 0, 1);
      e->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(128, 1, 1));
      e->endEncoding();
      cb->commit();
      cb->waitUntilCompleted();
      best3 = std::min(best3, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("simdgroup_matrix f32 MMA ceiling (classic queue): %.2f TF\n", flops / best3 * 1e-12);
  }
  {
    MTL::ComputePipelineState* pso = (*g)->pipeline("steel_conv1d_lc_f32_bm64_bn64");
    if (!pso) return 1;
    std::printf("steel conv1d, channel-last [L, C] (MLX steel, vendored)\n");
    for (const Shape& s : shapes) {
      uint32_t pad = s.d * (s.k - 1) / 2, lo = s.l + 2 * pad - s.d * (s.k - 1);
      double ms = runSteel(**g, pso, s, 10, reps);
      double tf = 2.0 * s.c * s.c * s.k * lo / (ms * 1e-3) * 1e-12;
      std::printf("  (%u, %u, %u, %u)  %8.2f ms  %6.2f TF\n", s.c, s.l, s.k, s.d, ms, tf);
    }
  }
  {
    MTL::ComputePipelineState* pso = (*g)->pipeline("conv1d_direct_f32");
    if (!pso) return 1;
    std::printf("conv1d_direct_f32\n");
    for (const Shape& s : shapes) {
      uint32_t pad = s.d * (s.k - 1) / 2, lo = s.l + 2 * pad - s.d * (s.k - 1);
      double ms = runDirect(**g, pso, s, 10, reps);
      double tf = 2.0 * s.c * s.c * s.k * lo / (ms * 1e-3) * 1e-12;
      std::printf("  (%u, %u, %u, %u)  %8.2f ms  %6.2f TF\n", s.c, s.l, s.k, s.d, ms, tf);
    }
  }
  if (std::getenv("LOKINFER_BENCH_ALL") == nullptr) return 0;
  for (const char* v : variants) {
    MTL::ComputePipelineState* pso = (*g)->pipeline(v);
    if (!pso) return 1;
    std::printf("%s (max threads %lu)\n", v, (unsigned long)pso->maxTotalThreadsPerThreadgroup());
    for (const Shape& s : shapes) {
      uint32_t pad = s.d * (s.k - 1) / 2, lo = s.l + 2 * pad - s.d * (s.k - 1);
      double ms = runCase(**g, pso, s, 10, reps);
      double tf = 2.0 * s.c * s.c * s.k * lo / (ms * 1e-3) * 1e-12;
      std::printf("  (%u, %u, %u, %u)  %8.2f ms  %6.2f TF\n", s.c, s.l, s.k, s.d, ms, tf);
    }
  }
  return 0;
}
