#include "conv_bench.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "../../src/runtime/gpu_internal.h"
#include "lokinfer/runtime/runtime.h"
#include "lokinfer/runtime/thread_pool.h"

namespace lk::bench {

namespace {

using clk = std::chrono::steady_clock;

struct Shape {
  int c, l, k, d;
};

// Kokoro rand510 (F = 1101): stage 1 = 128 ch x (120F+1), stage 0 = 256 ch x 20F.
const Shape kShapes[] = {{128, 132121, 3, 1}, {128, 132121, 7, 3}, {128, 132121, 11, 5},
                         {256, 22020, 3, 1},  {256, 22020, 7, 3},  {256, 22020, 11, 5}};

int outLen(const Shape& s) { return s.l; }  // "same" padding, stride 1
int padOf(const Shape& s) { return s.d * (s.k - 1) / 2; }
double flops(const Shape& s) { return 2.0 * s.c * s.c * s.k * outLen(s); }

std::vector<float> rnd(size_t n, uint32_t seed) {
  std::mt19937 r(seed);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> v(n);
  for (auto& x : v) x = u(r);
  return v;
}

// Direct reference for sample (co, t). Channel-first x [C, L], w [Co, Ci, K].
double refCF(const Shape& s, const float* x, const float* w, int co, int t) {
  double acc = 0;
  for (int ci = 0; ci < s.c; ++ci)
    for (int k = 0; k < s.k; ++k) {
      int ti = t + k * s.d - padOf(s);
      if (ti >= 0 && ti < s.l) acc += double(w[(size_t(co) * s.c + ci) * s.k + k]) * x[size_t(ci) * s.l + ti];
    }
  return acc;
}

// ---- CPU: time-tiled im2col, `streams` single-threaded Accelerate sgemms.
double cpuTiled(const Shape& s, int streams, int tile, double& maxErr, bool pool = false) {
  const int lo = outLen(s), pad = padOf(s), lp = s.l + 2 * pad, kk = s.c * s.k;
  auto x = rnd(size_t(s.c) * s.l, 1), w = rnd(size_t(s.c) * s.c * s.k, 2);
  std::vector<float> xpad(size_t(s.c) * lp, 0.f), out(size_t(s.c) * lo);
  for (int i = 0; i < s.c; ++i) std::memcpy(xpad.data() + size_t(i) * lp + pad, x.data() + size_t(i) * s.l, s.l * 4);
  const int ntiles = (lo + tile - 1) / tile;
  // Pool variant (what the M3 engine uses): one task per tile, per-thread col buffers.
  auto tileWork = [&](float* col, int ti) {
    int t0 = ti * tile, n = std::min(tile, lo - t0);
    for (int i = 0; i < s.c; ++i)
      for (int t = 0; t < s.k; ++t)
        std::memcpy(col + size_t(i * s.k + t) * n, xpad.data() + size_t(i) * lp + t * s.d + t0, n * 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s.c, n, kk, 1.f, w.data(), kk, col, n, 0.f, out.data() + t0,
                lo);
  };
  auto runPool = [&] {
    rt::ThreadPool::global().parallelFor(ntiles, [&](int64_t ti) {
      thread_local std::vector<float> col;
      thread_local bool single = (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED), true);
      (void)single;
      if (col.size() < size_t(kk) * tile) col.resize(size_t(kk) * tile);
      tileWork(col.data(), int(ti));
    });
  };
  auto run = [&] {
    if (pool) return runPool();
    std::vector<std::thread> th;
    for (int sidx = 0; sidx < streams; ++sidx)
      th.emplace_back([&, sidx] {
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        std::vector<float> col(size_t(kk) * tile);
        for (int ti = sidx; ti < ntiles; ti += streams) {
          int t0 = ti * tile, n = std::min(tile, lo - t0);
          for (int i = 0; i < s.c; ++i)
            for (int t = 0; t < s.k; ++t)
              std::memcpy(col.data() + size_t(i * s.k + t) * n, xpad.data() + size_t(i) * lp + t * s.d + t0, n * 4);
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s.c, n, kk, 1.f, w.data(), kk, col.data(), n, 0.f,
                      out.data() + t0, lo);
        }
      });
    for (auto& t : th) t.join();
  };
  run();  // warm-up
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    auto t0 = clk::now();
    run();
    best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
  }
  std::mt19937 pick(7);
  maxErr = 0;
  for (int q = 0; q < 100; ++q) {
    int co = int(pick() % s.c), t = q == 0 ? 0 : q == 1 ? lo - 1 : int(pick() % lo);
    maxErr = std::max(maxErr, std::abs(refCF(s, x.data(), w.data(), co, t) - out[size_t(co) * lo + t]));
  }
  return best;
}

// ---- GPU: vendored MLX steel implicit-GEMM conv, channel-last.
struct MLXConvParams2 {
  int N, C, O, iS[2], wS[2], oS[2], str[2], pad[2], kdil[2], idil[2];
  int64_t in_strides[4], wt_strides[4], out_strides[4];
  int groups;
  bool flip;
};
struct ImplicitGemmConv2DParams {
  int M, N, K, gemm_k_iterations, inp_jump_w, inp_jump_h, inp_jump_c, tiles_n, tiles_m, swizzle_log;
};

double gpuSteel(rt::GpuContext& g, MTL::ComputePipelineState* pso, const Shape& s, double& maxErr) {
  const int C = s.c, L = s.l, K = s.k, d = s.d, pad = padOf(s), lo = outLen(s);
  auto x = rt::allocateStorage(size_t(L) * C * 4, &g);
  auto w = rt::allocateStorage(size_t(C) * K * C * 4, &g);
  auto o = rt::allocateStorage(size_t(lo) * C * 4, &g);
  auto xv = rnd(size_t(L) * C, 1), wv = rnd(size_t(C) * K * C, 2);
  std::memcpy(x->data, xv.data(), xv.size() * 4);
  std::memcpy(w->data, wv.data(), wv.size() * 4);
  MLXConvParams2 cp{1, C, C, {1, L}, {1, K}, {1, lo}, {1, 1}, {0, pad}, {1, d}, {1, 1},
                    {int64_t(L) * C, int64_t(L) * C, C, 1}, {int64_t(K) * C, int64_t(K) * C, C, 1},
                    {int64_t(lo) * C, int64_t(lo) * C, C, 1}, 1, false};
  const int bm = 64, bn = 64, bk = 16, tn = (C + bn - 1) / bn, tm = (lo + bm - 1) / bm;
  const int ijw = C * d;
  ImplicitGemmConv2DParams gp{lo, C, K * C, K * ((C + bk - 1) / bk), ijw, L * C - (K - 1) * ijw, bk - (K - 1) * ijw,
                              tn, tm, 0};
  auto encode = [&] {
    g.beginDispatch();
    g.bind(0, static_cast<MTL::Buffer*>(x->mtlBuffer), 0);
    g.bind(1, static_cast<MTL::Buffer*>(w->mtlBuffer), 0);
    g.bind(2, static_cast<MTL::Buffer*>(o->mtlBuffer), 0);
    g.bindParams(3, &cp, sizeof cp);
    g.bindParams(4, &gp, sizeof gp);
    g.dispatchThreadgroups(pso, tn, tm, 1, 32, 2, 2);
  };
  const int n = 10;
  for (auto w0 = clk::now(); clk::now() - w0 < std::chrono::seconds(1);) {  // ramp the GPU clock
    for (int i = 0; i < n; ++i) encode();
    g.commitAndWait();
  }
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    auto t0 = clk::now();
    for (int i = 0; i < n; ++i) encode();
    g.commitAndWait();
    best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t0).count() / n);
  }
  const float* os = static_cast<const float*>(o->data);
  std::mt19937 pick(7);
  maxErr = 0;
  for (int q = 0; q < 100; ++q) {
    int co = int(pick() % C), t = q == 0 ? 0 : q == 1 ? lo - 1 : int(pick() % lo);
    double ref = 0;
    for (int k = 0; k < K; ++k) {
      int ti = t + k * d - pad;
      if (ti < 0 || ti >= L) continue;
      for (int ci = 0; ci < C; ++ci) ref += double(wv[(size_t(co) * K + k) * C + ci]) * xv[size_t(ti) * C + ci];
    }
    maxErr = std::max(maxErr, std::abs(ref - os[size_t(t) * C + co]));
  }
  return best;
}

}  // namespace

void runConvTune() {
  std::printf("LOKINFER_BENCH conv-tune begin\n");
  std::printf("P-cores %d, pool threads %d, hw threads %u\n", rt::performanceCores(), rt::ThreadPool::global().threads(),
              std::thread::hardware_concurrency());
  for (const Shape& s : kShapes) {
    for (int tile : {256, 384, 512, 768, 1024}) {
      std::printf("C=%d K=%d d=%d tile %4d:", s.c, s.k, s.d, tile);
      for (int streams : {4, 6, 8, 10, 12}) {
        if (unsigned(streams) > std::thread::hardware_concurrency()) continue;
        double e = 0, ms = cpuTiled(s, streams, tile, e);
        std::printf("  %2ds %.2f", streams, flops(s) / (ms * 1e-3) * 1e-12);
      }
      double e = 0, ms = cpuTiled(s, 0, tile, e, true);
      std::printf("  pool %.2f TF%s\n", flops(s) / (ms * 1e-3) * 1e-12, e > 1e-2 ? "  !! WRONG" : "");
      std::fflush(stdout);
    }
  }
  std::printf("LOKINFER_BENCH conv-tune end\n");
  std::fflush(stdout);
}

void runConvBench(const std::string& metallib, bool cpu, bool gpu) {
  std::printf("LOKINFER_BENCH conv begin\n");
  if (cpu) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    for (const Shape& s : kShapes) {
      double bestMs = 1e30, err = 0;
      int bestStreams = 0;
      for (int streams : {1, 2, 4, 6, 8, 12}) {
        if (unsigned(streams) > hw) continue;
        double e = 0, ms = cpuTiled(s, streams, 512, e);
        err = std::max(err, e);
        if (ms < bestMs) bestMs = ms, bestStreams = streams;
      }
      std::printf("cpu tiled-im2col C=%d L=%d K=%d d=%d: %.2f ms  %.2f TF  (%d streams)%s\n", s.c, s.l, s.k, s.d, bestMs,
                  flops(s) / (bestMs * 1e-3) * 1e-12, bestStreams, err > 1e-2 ? "  !! WRONG" : "");
      std::fflush(stdout);
    }
  }
  if (gpu) {
    rt::setMetallibPath(metallib);
    auto g = rt::GpuContext::shared();
    if (!g) {
      std::printf("gpu: %s\n", g.status().message().c_str());
    } else if (MTL::ComputePipelineState* pso = (*g)->pipeline("steel_conv1d_lc_f32_bm64_bn64")) {
      for (const Shape& s : kShapes) {
        double err = 0, ms = gpuSteel(**g, pso, s, err);
        std::printf("gpu steel-conv C=%d L=%d K=%d d=%d: %.2f ms  %.2f TF%s\n", s.c, s.l, s.k, s.d, ms,
                    flops(s) / (ms * 1e-3) * 1e-12, err > 1e-2 ? "  !! WRONG" : "");
        std::fflush(stdout);
      }
    }
  }
  std::printf("LOKINFER_BENCH conv end\n");
  std::fflush(stdout);
}

}  // namespace lk::bench
