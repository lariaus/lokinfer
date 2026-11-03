// Shared SME2 GEMM engine (kernels/cpu/sme2_gemm.cpp) against a float64 reference. On CPUs without SME2 (the M2 Max)
// the engine's scalar path runs, which still checks the packing / B-addressing / epilogue contract; the iPhone test
// host (A19) runs the FMOPA kernels. `[sme2bench]` (hidden) times the vision GEMM shapes against Accelerate.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <pthread.h>
#include <pthread/qos.h>

#include "../../kernels/cpu/sme2_gemm.h"
#include "lokinfer/runtime/thread_pool.h"

using namespace lk::kernels::cpu;

namespace {

std::vector<float> randv(size_t n, std::mt19937& g) {
  std::normal_distribution<float> d(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

double actD(double v, sme2::Act a) {
  switch (a) {
    case sme2::Act::None: return v;
    case sme2::Act::Relu: return std::max(v, 0.0);
    case sme2::Act::Relu6: return std::min(std::max(v, 0.0), 6.0);
    case sme2::Act::Leaky: return v >= 0 ? v : 0.2 * v;
  }
  return v;
}

struct Case {
  int64_t m, n, k, taps = 1, dil = 0;
  bool packedB = false, bias = false, biasRow = false, res = false, acc = false;
  sme2::Act act = sme2::Act::None;
};

void check(const Case& t, std::mt19937& g) {
  const int64_t ldc = t.n + 3;
  const auto a = randv(static_cast<size_t>(t.m * t.k), g);
  // B: taps == 1 -> [k, n] row-major (ldb = n + 5); taps > 1 -> channel-first windows: rows of win samples per
  // channel (k = ci * taps), element (c * taps + kk, j) = x[c * win + kk * dil + j].
  const int64_t ci = t.k / t.taps, win = t.n + (t.taps - 1) * t.dil + 2, ldb = t.taps == 1 ? t.n + 5 : win;
  const auto bm = randv(static_cast<size_t>((t.taps == 1 ? t.k : ci) * ldb), g);
  auto bAt = [&](int64_t kk, int64_t j) {
    return t.taps == 1 ? bm[kk * ldb + j] : bm[(kk / t.taps) * win + (kk % t.taps) * t.dil + j];
  };
  const auto bias = randv(static_cast<size_t>(std::max(t.m, t.n)), g);
  const auto res = randv(static_cast<size_t>(t.m * ldc), g);
  auto c = randv(static_cast<size_t>(t.m * ldc), g);
  const auto c0 = c;
  std::vector<float> ap(sme2::packedSize(t.m, t.k));
  sme2::packPanels(a.data(), t.k, t.m, t.k, ap.data());
  std::vector<float> bp;
  sme2::BOperand bo{bm.data(), ldb, 32, t.taps, t.dil};
  if (t.packedB) {
    bp.resize(sme2::packedSize(t.n, t.k));
    sme2::packPanelsKMajor(bm.data(), ldb, t.k, t.n, bp.data());
    bo = {bp.data(), 32, 32 * t.k};
  }
  sme2::Epilogue e;
  if (t.bias) e.bias = bias.data(), e.biasPerRow = t.biasRow;
  if (t.res) e.res = res.data(), e.ldres = ldc;
  e.accumulate = t.acc;
  e.act = t.act;
  sme2::gemm(ap.data(), t.m, t.k, bo, t.n, c.data(), ldc, e);
  double maxErr = 0;
  for (int64_t i = 0; i < t.m; ++i) {
    for (int64_t j = 0; j < t.n; ++j) {
      double s = 0;
      for (int64_t kk = 0; kk < t.k; ++kk) s += double(a[i * t.k + kk]) * double(bAt(kk, j));
      if (t.acc) s += c0[i * ldc + j];
      if (t.bias) s += bias[t.biasRow ? i : j];
      if (t.res) s += res[i * ldc + j];
      maxErr = std::max(maxErr, std::abs(actD(s, t.act) - double(c[i * ldc + j])));
    }
    for (int64_t j = t.n; j < ldc; ++j) REQUIRE(c[i * ldc + j] == c0[i * ldc + j]);  // nothing past n written
  }
  INFO("m " << t.m << " n " << t.n << " k " << t.k << " taps " << t.taps << " err " << maxErr);
  REQUIRE(maxErr < 1e-4 * std::sqrt(double(t.k)) + 1e-5);
}

double medianUs(const std::function<void()>& fn) {
  using clk = std::chrono::steady_clock;
  const auto w0 = clk::now();
  while (std::chrono::duration<double>(clk::now() - w0).count() < 0.3) fn();
  std::vector<double> ts;
  const auto t0 = clk::now();
  while (ts.size() < 30 || std::chrono::duration<double>(clk::now() - t0).count() < 0.3) {
    const auto a = clk::now();
    fn();
    ts.push_back(std::chrono::duration<double, std::micro>(clk::now() - a).count());
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

}  // namespace

TEST_CASE("sme2 gemm matches the reference", "[sme2]") {
  std::mt19937 g(7);
  INFO("SME2 available: " << sme2::available());
  const std::vector<Case> cases = {
      {32, 32, 16},
      {1, 1, 1},
      {49, 17, 5},
      {64, 64, 576, 1, 0, false, true, false, false, false, sme2::Act::Relu},
      {49, 512, 300, 1, 0, true, true, false, true, false, sme2::Act::Relu},
      {100, 70, 33, 1, 0, false, true, true, false, true, sme2::Act::Relu6},
      {37, 45, 24, 1, 0, true, false, false, false, false, sme2::Act::Leaky},
      {40, 50, 3 * 7, 7, 2, false, true, true, false, false, sme2::Act::None},  // Kokoro-style conv windows
      {64, 96, 4 * 3, 3, 1, false, false, false, true, true, sme2::Act::None},
  };
  for (const auto& t : cases) check(t, g);
}

TEST_CASE("sme2 gemm vs Accelerate on the vision shapes", "[.sme2bench]") {
  if (!sme2::available()) return;
  struct S {
    const char* name;
    int64_t m, n, k;
  };
  const std::vector<S> shapes = {
      {"r18 stem N1", 12544, 64, 147}, {"r18 l1 N1", 3136, 64, 576},   {"r18 l2 N1", 784, 128, 1152},
      {"r18 l3 N1", 196, 256, 2304},   {"r18 l4 N1", 49, 512, 4608},   {"r18 l4a N1", 49, 512, 2304},
      {"r18 l1 N8", 25088, 64, 576},   {"r18 l3 N8", 1568, 256, 2304}, {"r18 l4 N8", 392, 512, 4608},
      {"mb pw 56 e", 3136, 144, 24},   {"mb pw 56 p", 3136, 24, 144},  {"mb pw 14 e", 196, 384, 64},
      {"mb pw 14 p", 196, 64, 384},    {"mb pw 7 e", 49, 960, 160},    {"mb pw 7 p", 49, 160, 960},
      {"mb last", 49, 1280, 320},      {"mnist fc", 64, 128, 784},     {"mnist fc1k", 1024, 128, 784},
      {"gpt qkv 128", 128, 2304, 768}, {"gpt o 128", 128, 768, 768},   {"gpt fc 128", 128, 3072, 768},
      {"gpt pr 128", 128, 768, 3072},  {"albert qkv 59", 59, 2304, 768}, {"albert ffn 59", 59, 2048, 768},
      {"gpt fc 512", 512, 3072, 768},
  };
  std::mt19937 g(1);
  auto& pool = lk::rt::ThreadPool::global();
  std::printf("SME2BENCH shape m n k | accel1 accelMT sme1 sme1pre smeB smePool2 smePoolAll (us) | best GF/s\n");
  for (const auto& s : shapes) {
    const auto a = randv(static_cast<size_t>(s.m * s.k), g), b = randv(static_cast<size_t>(s.k * s.n), g);
    std::vector<float> c(static_cast<size_t>(s.m * s.n)), ap(sme2::packedSize(s.m, s.k)),
        bp(sme2::packedSize(s.n, s.k));
    sme2::packPanels(a.data(), s.k, s.m, s.k, ap.data());
    sme2::packPanelsKMajor(b.data(), s.n, s.k, s.n, bp.data());
    const sme2::BOperand bo{b.data(), s.n}, bop{bp.data(), 32, 32 * s.k};
    auto blas = [&](BLAS_THREADING th) {
      BLASSetThreading(th);
      return medianUs([&] {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(s.m), int(s.n), int(s.k), 1.f, a.data(), int(s.k),
                    b.data(), int(s.n), 0.f, c.data(), int(s.n));
      });
    };
    const double accel1 = blas(BLAS_THREADING_SINGLE_THREADED), accelMT = blas(BLAS_THREADING_MULTI_THREADED);
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    const double sme1 = medianUs([&] { sme2::gemmA(a.data(), s.k, s.m, s.k, bo, s.n, c.data(), s.n); });
    const double smePre = medianUs([&] { sme2::gemm(ap.data(), s.m, s.k, bo, s.n, c.data(), s.n); });
    const double smeB = medianUs([&] { sme2::gemm(ap.data(), s.m, s.k, bop, s.n, c.data(), s.n); });
    // Pool split along the dimension with more 32-blocks (packing per task included).
    auto pooled = [&](int64_t tasks) {
      const int64_t mb = (s.m + 31) / 32, nb = (s.n + 31) / 32;
      const bool byM = mb >= nb;
      const int64_t blocks = byM ? mb : nb, t = std::min(tasks, blocks), per = (blocks + t - 1) / t;
      return medianUs([&] {
        pool.parallelFor(t, [&](int64_t i) {
          const int64_t b0 = i * per * 32, b1 = std::min((i + 1) * per * 32, byM ? s.m : s.n);
          if (b0 >= b1) return;
          if (byM)
            sme2::gemmA(a.data() + b0 * s.k, s.k, b1 - b0, s.k, bo, s.n, c.data() + b0 * s.n, s.n);
          else
            sme2::gemm(ap.data(), s.m, s.k, {b.data() + b0, s.n}, b1 - b0, c.data() + b0, s.n);
        });
      });
    };
    const double p2 = pooled(2), pAll = pooled(pool.threads());
    const double best = std::min({accel1, accelMT, sme1, smePre, smeB, p2, pAll});
    std::printf("SME2BENCH %-12s %5lld %4lld %4lld | %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f | %6.0f\n", s.name,
                (long long)s.m, (long long)s.n, (long long)s.k, accel1, accelMT, sme1, smePre, smeB, p2, pAll,
                2.0 * double(s.m) * double(s.n) * double(s.k) / best / 1e3);
    std::fflush(stdout);
  }
  BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
}

TEST_CASE("sme2 gather packing equals packing the explicit rows", "[sme2]") {
  std::mt19937 g(3);
  for (const int64_t seglen : {16, 32, 20, 3})
    for (const int64_t rows : {1, 31, 50, 64}) {
      const int64_t segs = 9, k = segs * seglen;
      const auto src = randv(static_cast<size_t>(rows * segs * seglen + 64), g);
      std::vector<const float*> ptrs(static_cast<size_t>(rows * segs));
      std::vector<float> flat(static_cast<size_t>(rows * k));
      std::mt19937 pick(static_cast<uint32_t>(rows));
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t s = 0; s < segs; ++s) {
          const float* p = src.data() + (pick() % static_cast<uint32_t>(rows * segs)) * seglen;
          ptrs[static_cast<size_t>(r * segs + s)] = p;
          std::copy(p, p + seglen, flat.begin() + r * k + s * seglen);
        }
      std::vector<float> a(sme2::packedSize(rows, k), -1.f), b(sme2::packedSize(rows, k), -2.f);
      sme2::packPanels(flat.data(), k, rows, k, a.data());
      sme2::packPanelsGather(ptrs.data(), rows, segs, seglen, b.data());
      INFO("seglen " << seglen << " rows " << rows);
      REQUIRE(a == b);
    }
}

TEST_CASE("sme2 gemm with a row-offset B and a column mask", "[sme2]") {
  std::mt19937 g(11);
  for (const int64_t k : {9, 150, 576})
    for (const auto act : {sme2::Act::None, sme2::Act::Relu}) {
      const int64_t m = 70, n = 101, ldc = n + 1, src = 4096;
      const auto a = randv(static_cast<size_t>(m * k), g), x = randv(static_cast<size_t>(src + n + 64), g);
      std::vector<int64_t> off(static_cast<size_t>(k));
      for (auto& o : off) o = static_cast<int64_t>(g() % static_cast<uint32_t>(src));
      std::vector<float> mask(static_cast<size_t>(n)), bias = randv(static_cast<size_t>(m), g);
      for (auto& v : mask) v = (g() % 4 == 0) ? 0.f : 1.f;
      std::vector<float> ap(sme2::packedSize(m, k)), c(static_cast<size_t>(m * ldc), 7.f);
      sme2::packPanels(a.data(), k, m, k, ap.data());
      sme2::BOperand bo{x.data(), 0};
      bo.rowOff = off.data();
      sme2::Epilogue e;
      e.bias = bias.data();
      e.biasPerRow = true;
      e.act = act;
      e.colMask = mask.data();
      sme2::gemm(ap.data(), m, k, bo, n, c.data(), ldc, e);
      double maxErr = 0;
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          double s = bias[i];
          for (int64_t kk = 0; kk < k; ++kk) s += double(a[i * k + kk]) * double(x[off[kk] + j]);
          maxErr = std::max(maxErr, std::abs(actD(s, act) * mask[j] - double(c[i * ldc + j])));
        }
        REQUIRE(c[i * ldc + n] == 7.f);
      }
      INFO("k " << k << " err " << maxErr);
      REQUIRE(maxErr < 1e-4 * std::sqrt(double(k)) + 1e-5);
    }
}

TEST_CASE("sme2 planar 3x3 conv GEMMs (row-offset B)", "[.sme2bench]") {
  if (!sme2::available()) return;
  struct S {
    const char* name;
    int64_t hw, c, n;
  };
  const std::vector<S> shapes = {{"56x56x64 N1", 56, 64, 1},   {"28x28x128 N1", 28, 128, 1}, {"14x14x256 N1", 14, 256, 1},
                                 {"7x7x512 N1", 7, 512, 1},    {"56x56x64 N8", 56, 64, 8},   {"14x14x256 N8", 14, 256, 8},
                                 {"7x7x512 N8", 7, 512, 8}};
  std::mt19937 g(5);
  std::printf("SME2PLANAR shape | us | useful GF/s\n");
  for (const auto& s : shapes) {
    const int64_t wp = s.hw + 2, img = wp * wp, plane = s.n * img, k = 9 * s.c, o0 = wp + 1,
                  cols = (s.n - 1) * img + s.hw * wp + s.hw + 1 - o0;
    const auto x = randv(static_cast<size_t>(s.c * plane), g), a = randv(static_cast<size_t>(s.c * k), g),
               bias = randv(static_cast<size_t>(s.c), g);
    std::vector<float> ap(sme2::packedSize(s.c, k)), out(static_cast<size_t>(s.c * plane)),
        mask(static_cast<size_t>(cols), 1.f);
    sme2::packPanels(a.data(), k, s.c, k, ap.data());
    std::vector<int64_t> off(static_cast<size_t>(k));
    for (int64_t t = 0; t < 9; ++t)
      for (int64_t c = 0; c < s.c; ++c) off[static_cast<size_t>(t * s.c + c)] = c * plane + (t / 3 - 1) * wp + t % 3 - 1;
    sme2::BOperand bo{x.data() + o0, 0};
    bo.rowOff = off.data();
    sme2::Epilogue e;
    e.bias = bias.data();
    e.biasPerRow = true;
    e.act = sme2::Act::Relu;
    e.colMask = mask.data();
    const double us = medianUs([&] { sme2::gemm(ap.data(), s.c, k, bo, cols, out.data() + o0, plane, e); });
    std::printf("SME2PLANAR %-14s %8.1f us %6.0f GF/s\n", s.name, us,
                2.0 * double(s.n * s.hw * s.hw) * double(s.c) * double(k) / us / 1e3);
    std::fflush(stdout);
  }
}

// Hidden: which cluster's SME unit runs what. Threads at utility QoS land on the efficiency cores. Times one stream of
// sme2::gemm and of single-threaded Accelerate sgemm (512 x 512 x 512) on a P-core thread and on E-core threads, and
// 4 concurrent E-core streams, alone and while a P-core thread runs sme2::gemm (does the E cluster add throughput?).
TEST_CASE("sme2 vs Accelerate per cluster", "[.sme2cluster]") {
  if (!sme2::available()) return;
  constexpr int64_t m = 512, n = 512, k = 512;
  std::mt19937 g(5);
  const auto a = randv(m * k, g), b = randv(k * n, g);
  std::vector<float> ap(sme2::packedSize(m, k));
  sme2::packPanels(a.data(), k, m, k, ap.data());
  auto runFor = [&](bool sme, double secs, std::vector<float>& c, int* cpuOut) {
    using clk = std::chrono::steady_clock;
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    int64_t calls = 0;
    const auto t0 = clk::now();
    while (std::chrono::duration<double>(clk::now() - t0).count() < secs) {
      if (sme)
        sme2::gemm(ap.data(), m, k, {b.data(), n}, n, c.data(), n);
      else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, a.data(), k, b.data(), n, 0.f, c.data(), n);
      ++calls;
    }
    size_t cpu = 0;
    pthread_cpu_number_np(&cpu);
    if (cpuOut) *cpuOut = static_cast<int>(cpu);
    return 2.0 * m * n * k * calls / std::chrono::duration<double>(clk::now() - t0).count() / 1e9;
  };
  auto onQos = [&](qos_class_t q, std::function<void()> fn) {
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_set_qos_class_np(&at, q, 0);
    auto* f = new std::function<void()>(std::move(fn));
    pthread_t th;
    pthread_create(&th, &at, [](void* p) -> void* {
      auto* ff = static_cast<std::function<void()>*>(p);
      (*ff)();
      delete ff;
      return nullptr;
    }, f);
    return th;
  };
  for (int sme = 1; sme >= 0; --sme) {  // N concurrent user-interactive streams (they spill onto the E cores)
    for (int nt : {1, 2, 3, 4, 6}) {
      std::vector<double> gf(static_cast<size_t>(nt), 0.0);
      std::vector<int> cpus(static_cast<size_t>(nt), -1);
      std::vector<std::vector<float>> cs(static_cast<size_t>(nt), std::vector<float>(m * n));
      std::vector<pthread_t> ths;
      for (int i = 0; i < nt; ++i)
        ths.push_back(onQos(QOS_CLASS_USER_INTERACTIVE, [&, i] { gf[i] = runFor(sme, 2.0, cs[i], &cpus[i]); }));
      for (auto t : ths) pthread_join(t, nullptr);
      double sum = 0;
      std::string detail;
      for (int i = 0; i < nt; ++i) sum += gf[i], detail += std::to_string(cpus[i]) + ":" + std::to_string(int(gf[i])) + " ";
      std::printf("STREAMS %-10s x%d total %7.0f GF/s  (cpu:GF/s %s)\n", sme ? "sme2::gemm" : "Accelerate", nt, sum,
                  detail.c_str());
      std::fflush(stdout);
    }
  }
  for (int sme = 1; sme >= 0; --sme) {
    const char* name = sme ? "sme2::gemm" : "Accelerate";
    for (int ecount : {0, 1, 4}) {
      for (int withP : {1, 0}) {
        if (!withP && ecount == 0) continue;
        std::vector<double> gf(static_cast<size_t>(ecount + 1), 0.0);
        std::vector<int> cpus(static_cast<size_t>(ecount + 1), -1);
        std::vector<std::vector<float>> cs(static_cast<size_t>(ecount + 1), std::vector<float>(m * n));
        std::vector<pthread_t> ths;
        for (int i = 0; i < ecount; ++i)
          ths.push_back(onQos(QOS_CLASS_BACKGROUND, [&, i] { gf[i + 1] = runFor(sme, 1.5, cs[i + 1], &cpus[i + 1]); }));
        if (withP)
          ths.push_back(onQos(QOS_CLASS_USER_INTERACTIVE, [&] { gf[0] = runFor(sme, 1.5, cs[0], &cpus[0]); }));
        for (auto t : ths) pthread_join(t, nullptr);
        double eSum = 0;
        std::string ec;
        for (int i = 0; i < ecount; ++i) eSum += gf[i + 1], ec += std::to_string(cpus[i + 1]) + " ";
        std::printf("CLUSTER %-10s P=%d (cpu %d) %7.0f GF/s | E x%d (cpus %s) %7.0f GF/s | total %7.0f\n", name, withP,
                    cpus[0], gf[0], ecount, ec.c_str(), eSum, gf[0] + eSum);
        std::fflush(stdout);
      }
    }
  }
}
