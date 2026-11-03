// Multi-stream AMX / SME re-verification (docs/impl-notes/m2-conv-proof.md "Multi-stream AMX, out of LPM"): does the
// CPU's matrix throughput scale with concurrent single-threaded Accelerate sgemm streams, vs one call with
// Accelerate's own threading? Shared by the Mac CLI (`lokinfer-bench-conv amx`) and the iOS host (`--bench-amx`).
//
// Per shape: (a) one thread, Accelerate single-threaded, QoS user-interactive; (b) the same at QoS background (the
// "E-core AMX" oddity of research-kernels.md); (c) one call with Accelerate's default threading; (d) N concurrent
// threads, each its own single-threaded sgemm on its own buffers. Protocol (docs/perf-dashboard.md): >= 1 s warm-up
// (all streams running), then >= 30 calls and >= 2 s per stream, median per stream; aggregate = sum of per-stream
// FLOP rates.
#include "amx_bench.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <pthread.h>
#include <sys/qos.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace lk::bench {
namespace {

using Clock = std::chrono::steady_clock;

struct Gemm {
  int m, n, k;
  const char* what;
};

struct Stream {
  std::vector<float> a, b, c;
  std::vector<double> ts;  // seconds per call (measured phase)
  explicit Stream(const Gemm& g) : a(size_t(g.m) * g.k, 0.1f), b(size_t(g.k) * g.n, 0.2f), c(size_t(g.m) * g.n) {}
  void call(const Gemm& g) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, g.m, g.n, g.k, 1.f, a.data(), g.k, b.data(), g.n, 0.f,
                c.data(), g.n);
  }
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

// Runs `streams` threads concurrently (all start together); returns the aggregate GFLOP/s and the slowest/fastest
// stream's rate. single = BLASSetThreading(single) in each thread.
struct Result {
  double agg, minStream, maxStream;
  size_t calls;
};

Result runConcurrent(const Gemm& g, int streams, bool single, qos_class_t qos) {
  std::vector<Stream> ss;
  ss.reserve(streams);
  for (int i = 0; i < streams; ++i) ss.emplace_back(g);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  const double flop = 2.0 * g.m * g.n * g.k;
  std::vector<std::thread> th;
  for (int i = 0; i < streams; ++i)
    th.emplace_back([&, i] {
      pthread_set_qos_class_self_np(qos, 0);
      if (single) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
      ready.fetch_add(1);
      while (!go.load()) {
      }
      Stream& s = ss[i];
      auto t0 = Clock::now();
      int k = 0;
      while (k < 3 || Clock::now() - t0 < std::chrono::milliseconds(1000)) s.call(g), ++k;  // warm-up >= 1 s
      t0 = Clock::now();
      while (s.ts.size() < 30 || Clock::now() - t0 < std::chrono::milliseconds(2000)) {
        auto a = Clock::now();
        s.call(g);
        s.ts.push_back(std::chrono::duration<double>(Clock::now() - a).count());
      }
      if (single) BLASSetThreading(BLAS_THREADING_MULTI_THREADED);
    });
  while (ready.load() < streams) {
  }
  go = true;
  for (auto& t : th) t.join();
  Result r{0, 1e30, 0, 0};
  for (auto& s : ss) {
    const double gf = flop / median(s.ts) / 1e9;
    r.agg += gf;
    r.minStream = std::min(r.minStream, gf);
    r.maxStream = std::max(r.maxStream, gf);
    r.calls += s.ts.size();
  }
  return r;
}

}  // namespace

void runAmxBench(const std::vector<int>& streamCounts) {
  // 512x768x768: the research claim's shape; 1024^3: a big square GEMM; 128x512x1408: one Kokoro stage-1 k11 conv
  // tile (Co x tile x Ci*K, the M2 CPU winner's per-stream GEMM); 59x2304x768: ALBERT QKV at app59f (T = 59).
  const Gemm shapes[] = {{512, 768, 768, "research shape"},
                         {1024, 1024, 1024, "square"},
                         {128, 512, 1408, "Kokoro conv tile (Co x tile x Ci*K)"},
                         {59, 2304, 768, "ALBERT QKV, T=59"}};
  std::printf("AMX_BENCH begin (GFLOP/s; median per stream over >= 30 calls and >= 2 s after >= 1 s warm-up)\n");
  for (const Gemm& g : shapes) {
    std::printf("AMX_BENCH shape %dx%dx%d (%s)\n", g.m, g.n, g.k, g.what);
    auto r = runConcurrent(g, 1, true, QOS_CLASS_USER_INTERACTIVE);
    std::printf("AMX_BENCH   1 stream single-threaded, QoS user-interactive: %8.1f\n", r.agg);
    r = runConcurrent(g, 1, true, QOS_CLASS_BACKGROUND);
    std::printf("AMX_BENCH   1 stream single-threaded, QoS background:       %8.1f\n", r.agg);
    r = runConcurrent(g, 1, false, QOS_CLASS_USER_INTERACTIVE);
    std::printf("AMX_BENCH   1 call, Accelerate default threading:           %8.1f\n", r.agg);
    for (int n : streamCounts) {
      if (n < 2) continue;
      r = runConcurrent(g, n, true, QOS_CLASS_USER_INTERACTIVE);
      std::printf("AMX_BENCH   %2d concurrent single-threaded streams:          %8.1f aggregate (per stream %.0f-%.0f)\n",
                  n, r.agg, r.minStream, r.maxStream);
    }
    std::fflush(stdout);
  }
  std::printf("AMX_BENCH end\n");
  std::fflush(stdout);
}

}  // namespace lk::bench
