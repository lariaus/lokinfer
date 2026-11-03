// Shared helpers for kernels/cpu/kokoro_*.cpp (internal header).
#pragma once

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <pthread.h>
#include <sys/sysctl.h>

#include <cstdint>
#include <functional>
#include <vector>

#include "lokinfer/runtime/thread_pool.h"
#include "neon_math.h"

namespace lk::kernels::cpu::detail {

// Grow-only thread-local scratch (one buffer per slot). Allocates only when a larger size than ever
// seen on this thread is requested; steady-state calls do not allocate.
inline float* scratch(size_t n, int slot = 0) {
  thread_local std::vector<float> bufs[8];
  std::vector<float>& b = bufs[slot];
  if (b.size() < n) b.resize(n);
  return b.data();
}

inline __LAPACK_int li(int64_t v) { return static_cast<__LAPACK_int>(v); }

// Small GEMMs run with single-threaded BLAS: Accelerate's internal multithreading costs more than it saves below a
// few tens of MFLOP (inline convs of small convnets on the iPhone). The calling thread's setting is restored.
constexpr double kSmallGemmFlops = 5e7;

// C[m, n] = alpha * op(A) @ op(B) + beta * C, row-major with explicit leading dims.
inline void sgemm(bool transA, bool transB, int64_t m, int64_t n, int64_t k, float alpha, const float* a, int64_t lda,
                  const float* b, int64_t ldb, float beta, float* c, int64_t ldc) {
  const bool small = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) < kSmallGemmFlops;
  const BLAS_THREADING prev = small ? BLASGetThreading() : BLAS_THREADING_SINGLE_THREADED;
  if (small && prev != BLAS_THREADING_SINGLE_THREADED) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
  // A multi-threaded BLAS call: our pool workers still spinning from the previous parallelFor would compete with
  // Accelerate's threads for the cores; send them to sleep first.
  if (!small && prev != BLAS_THREADING_SINGLE_THREADED) rt::ThreadPool::global().park();
  cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, li(m), li(n),
              li(k), alpha, a, li(lda), b, li(ldb), beta, c, li(ldc));
  if (small && prev != BLAS_THREADING_SINGLE_THREADED) BLASSetThreading(prev);
}

// Sum of n floats: 4x4-lane NEON accumulators within 1024-element blocks, blocks combined in double.
inline double sumBlocked(const float* x, int64_t n) {
  double total = 0.0;
  for (int64_t i0 = 0; i0 < n; i0 += 1024) {
    int64_t e = i0 + 1024 < n ? i0 + 1024 : n, i = i0;
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= e; i += 16) {
      a0 = vaddq_f32(a0, vld1q_f32(x + i));
      a1 = vaddq_f32(a1, vld1q_f32(x + i + 4));
      a2 = vaddq_f32(a2, vld1q_f32(x + i + 8));
      a3 = vaddq_f32(a3, vld1q_f32(x + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < e; ++i) s += x[i];
    total += s;
  }
  return total;
}

// Sum of (x - mean)^2, same blocking as sumBlocked.
inline double sumSqDevBlocked(const float* x, int64_t n, float mean) {
  double total = 0.0;
  float32x4_t m = vdupq_n_f32(mean);
  for (int64_t i0 = 0; i0 < n; i0 += 1024) {
    int64_t e = i0 + 1024 < n ? i0 + 1024 : n, i = i0;
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= e; i += 16) {
      float32x4_t d0 = vsubq_f32(vld1q_f32(x + i), m), d1 = vsubq_f32(vld1q_f32(x + i + 4), m);
      float32x4_t d2 = vsubq_f32(vld1q_f32(x + i + 8), m), d3 = vsubq_f32(vld1q_f32(x + i + 12), m);
      a0 = vfmaq_f32(a0, d0, d0);
      a1 = vfmaq_f32(a1, d1, d1);
      a2 = vfmaq_f32(a2, d2, d2);
      a3 = vfmaq_f32(a3, d3, d3);
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < e; ++i) {
      float d = x[i] - mean;
      s += d * d;
    }
    total += s;
  }
  return total;
}

// Two-pass mean / biased variance.
inline void meanVar(const float* x, int64_t n, float& mean, float& var) {
  mean = static_cast<float>(sumBlocked(x, n) / static_cast<double>(n));
  var = static_cast<float>(sumSqDevBlocked(x, n, mean) / static_cast<double>(n));
}

// out[i] = (x[i] - m) * a + b  (mean subtracted first: no cancellation for large means)
inline void affineCentered(const float* x, float* out, int64_t n, float m, float a, float b) {
  float32x4_t vm = vdupq_n_f32(m), va = vdupq_n_f32(a), vb = vdupq_n_f32(b);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vfmaq_f32(vb, vsubq_f32(vld1q_f32(x + i), vm), va));
  for (; i < n; ++i) out[i] = (x[i] - m) * a + b;
}

// parallelFor on the runtime pool with single-threaded BLAS inside every task (one sgemm stream per core).
// BLAS threading is a per-thread setting: workers keep SINGLE for good; the calling thread (which also runs
// tasks) is restored afterwards so its later sgemm calls stay multithreaded.
inline void parallelForSt(int64_t n, const std::function<void(int64_t)>& fn) {
  const BLAS_THREADING prev = BLASGetThreading();
  if (prev != BLAS_THREADING_SINGLE_THREADED) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
  thread_local bool workerSingle = false;
  rt::ThreadPool::global().parallelFor(n, [&](int64_t i) {
    if (!workerSingle) BLASSetThreading(BLAS_THREADING_SINGLE_THREADED), workerSingle = true;
    fn(i);
  });
  if (prev != BLAS_THREADING_SINGLE_THREADED) BLASSetThreading(prev);
}

// CPU numbers >= this are performance cores (Apple numbers the efficiency cores first); 0 = no split (single
// cluster or unknown).
inline int perfCoreSplit() {
  static const int split = [] {
    int e = 0, p = 0;
    size_t len = sizeof e;
    if (sysctlbyname("hw.perflevel1.logicalcpu", &e, &len, nullptr, 0) != 0) return 0;
    len = sizeof p;
    if (sysctlbyname("hw.perflevel0.logicalcpu", &p, &len, nullptr, 0) != 0) return 0;
    return e > 0 && p > 0 ? e : 0;
  }();
  return split;
}

inline bool onPerfCore() {
  size_t cpu = 0;
  if (pthread_cpu_number_np(&cpu) != 0) return true;
  return static_cast<int>(cpu) >= perfCoreSplit();
}

}  // namespace lk::kernels::cpu::detail
