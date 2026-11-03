// M2 conv proof (CPU): Kokoro generator conv1d shapes, through the kernel
// library only (no runtime). Candidates:
//   im2col   : the basic kernel (im2col + one Accelerate sgemm, Accelerate threading)
//   shifted  : one sgemm per tap over a zero-padded input (no im2col copy), Accelerate threading
//   streams/N: shifted, split into N time tiles, each tile on its own thread with a
//              single-threaded sgemm (BLASSetThreading) — one stream per AMX unit
// Reports GFLOP/s (2·Co·Ci·K·Lout per conv). Layout [C, L] (channel-first, batch 1).
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "lokinfer/kernels/cpu_kokoro.h"

namespace {

struct Shape {
  int64_t c, l, k, d;
};

std::vector<float> randomVec(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = u(rng);
  return v;
}

int64_t outLen(const Shape& s) { return s.l + 2 * (s.d * (s.k - 1) / 2) - s.d * (s.k - 1); }  // "same" padding

// out[Co, Lo] = sum_k W[k][Co, Ci] @ xpad[Ci, k*d : k*d + Lo]; wTaps is [K, Co, Ci], xpad has row stride lp.
void shiftedGemm(const float* wTaps, const float* xpad, int64_t lp, float* out, int64_t co, int64_t ci, int64_t k,
                 int64_t d, int64_t t0, int64_t t1, int64_t lo) {
  for (int64_t tap = 0; tap < k; ++tap)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<__LAPACK_int>(co),
                static_cast<__LAPACK_int>(t1 - t0), static_cast<__LAPACK_int>(ci), 1.f, wTaps + tap * co * ci,
                static_cast<__LAPACK_int>(ci), xpad + tap * d + t0, static_cast<__LAPACK_int>(lp), tap ? 1.f : 0.f,
                out + t0, static_cast<__LAPACK_int>(lo));
}

struct Fixture {
  Shape s;
  int64_t lo, pad, lp;
  std::vector<float> x, w, wTaps, b, xpad, out;
  explicit Fixture(Shape sh) : s(sh) {
    lo = outLen(s);
    pad = s.d * (s.k - 1) / 2;
    lp = s.l + 2 * pad;
    x = randomVec(s.c * s.l, 1);
    w = randomVec(s.c * s.c * s.k, 2);
    b = randomVec(s.c, 3);
    out.assign(s.c * lo, 0.f);
    wTaps.resize(w.size());  // [Co, Ci, K] -> [K, Co, Ci] (done once at load in a real model)
    for (int64_t o = 0; o < s.c; ++o)
      for (int64_t i = 0; i < s.c; ++i)
        for (int64_t t = 0; t < s.k; ++t) wTaps[(t * s.c + o) * s.c + i] = w[(o * s.c + i) * s.k + t];
    xpad.assign(s.c * lp, 0.f);
    for (int64_t i = 0; i < s.c; ++i) std::memcpy(xpad.data() + i * lp + pad, x.data() + i * s.l, s.l * sizeof(float));
  }
  double flops() const { return 2.0 * s.c * s.c * s.k * lo; }
};

void report(benchmark::State& st, const Fixture& f) {
  st.counters["GFLOPs"] = benchmark::Counter(f.flops() * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
}

Shape shapeOf(const benchmark::State& st) { return Shape{st.range(0), st.range(1), st.range(2), st.range(3)}; }

void BM_Im2col(benchmark::State& st) {
  Fixture f(shapeOf(st));
  for (auto _ : st) {
    lk::kernels::cpu::conv1d_f32(f.x.data(), f.w.data(), nullptr, f.out.data(), f.s.c, f.s.l, f.s.c, f.s.k, 1,
                                 f.pad, f.s.d);
    benchmark::DoNotOptimize(f.out.data());
  }
  report(st, f);
}

void BM_Shifted(benchmark::State& st) {
  Fixture f(shapeOf(st));
  for (auto _ : st) {
    // Includes the padding copy a real kernel would fuse into the producer.
    for (int64_t i = 0; i < f.s.c; ++i)
      std::memcpy(f.xpad.data() + i * f.lp + f.pad, f.x.data() + i * f.s.l, f.s.l * sizeof(float));
    shiftedGemm(f.wTaps.data(), f.xpad.data(), f.lp, f.out.data(), f.s.c, f.s.c, f.s.k, f.s.d, 0, f.lo, f.lo);
    benchmark::DoNotOptimize(f.out.data());
  }
  report(st, f);
}

void BM_Streams(benchmark::State& st) {
  Fixture f(shapeOf(st));
  const int streams = static_cast<int>(st.range(4));
  for (auto _ : st) {
    for (int64_t i = 0; i < f.s.c; ++i)
      std::memcpy(f.xpad.data() + i * f.lp + f.pad, f.x.data() + i * f.s.l, f.s.l * sizeof(float));
    std::vector<std::thread> th;
    for (int t = 0; t < streams; ++t) {
      int64_t t0 = f.lo * t / streams, t1 = f.lo * (t + 1) / streams;
      th.emplace_back([&, t0, t1] {
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        shiftedGemm(f.wTaps.data(), f.xpad.data(), f.lp, f.out.data(), f.s.c, f.s.c, f.s.k, f.s.d, t0, t1, f.lo);
      });
    }
    for (auto& t : th) t.join();
    benchmark::DoNotOptimize(f.out.data());
  }
  report(st, f);
}

// Time-tiled im2col: each tile of `tile` output samples builds its own
// [Ci*K, tile] column block (L2-resident) and runs one sgemm with inner dim
// Ci*K (instead of Ci per tap); tiles are spread over `streams` threads, each
// with a single-threaded sgemm.
void tiledIm2col(const Fixture& f, float* out, int streams, int64_t tile) {
  const int64_t ci = f.s.c, co = f.s.c, k = f.s.k, d = f.s.d, kk = ci * k;
  const int64_t ntiles = (f.lo + tile - 1) / tile;
  // weights as [Co, Ci*K] with column index (i*K + t) = the PyTorch layout already
  auto work = [&](int sidx) {
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    std::vector<float> col(static_cast<size_t>(kk * tile));
    for (int64_t ti = sidx; ti < ntiles; ti += streams) {
      int64_t t0 = ti * tile, n = std::min(tile, f.lo - t0);
      for (int64_t i = 0; i < ci; ++i)
        for (int64_t t = 0; t < k; ++t)
          std::memcpy(col.data() + (i * k + t) * n, f.xpad.data() + i * f.lp + t * d + t0, n * sizeof(float));
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<__LAPACK_int>(co), static_cast<__LAPACK_int>(n),
                  static_cast<__LAPACK_int>(kk), 1.f, f.w.data(), static_cast<__LAPACK_int>(kk), col.data(),
                  static_cast<__LAPACK_int>(n), 0.f, out + t0, static_cast<__LAPACK_int>(f.lo));
    }
  };
  std::vector<std::thread> th;
  for (int sidx = 0; sidx < streams; ++sidx) th.emplace_back(work, sidx);
  for (auto& t : th) t.join();
}

void BM_TiledIm2col(benchmark::State& st) {
  Fixture f(Shape{st.range(0), st.range(1), st.range(2), st.range(3)});
  const int streams = static_cast<int>(st.range(4));
  const int64_t tile = st.range(5);
  for (auto _ : st) {
    for (int64_t i = 0; i < f.s.c; ++i)
      std::memcpy(f.xpad.data() + i * f.lp + f.pad, f.x.data() + i * f.s.l, f.s.l * sizeof(float));
    tiledIm2col(f, f.out.data(), streams, tile);
    benchmark::DoNotOptimize(f.out.data());
  }
  report(st, f);
}

void TiledArgs(benchmark::Benchmark* b) {
  for (int s : {1, 3, 4, 8})
    for (int64_t tile : {512, 2048}) {
      b->Args({128, 132121, 11, 5, s, tile});
      b->Args({256, 22020, 7, 3, s, tile});
    }
}

// Kokoro rand510 (F = 1101): stage 1 = 128 ch × (120F+1), stage 0 = 256 ch × 20F.
void Stage1(benchmark::Benchmark* b) {
  for (int64_t k : {3, 7, 11})
    for (int64_t d : {1, 5}) b->Args({128, 132121, k, d});
}
void Stage0(benchmark::Benchmark* b) {
  for (int64_t k : {3, 7, 11})
    for (int64_t d : {1, 5}) b->Args({256, 22020, k, d});
  b->Args({256, 22020, 7, 3});  // the m2-conv-proof.md table's stage-0 column
}
void StreamsArgs(benchmark::Benchmark* b) {
  for (int s : {2, 3, 4, 8}) {
    b->Args({128, 132121, 11, 5, s});
    b->Args({256, 22020, 7, 3, s});
  }
}

}  // namespace

BENCHMARK(BM_Im2col)->Apply(Stage1)->Apply(Stage0)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Shifted)->Apply(Stage1)->Apply(Stage0)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Streams)->Apply(StreamsArgs)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_TiledIm2col)->Apply(TiledArgs)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
