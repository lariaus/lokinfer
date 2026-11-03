// CPU GEMM: a hand-written NEON baseline and the Accelerate (AMX) variant.
// The hand-written kernel is a straightforward register-blocked micro-kernel
// (4x16 tile, 16 f32 accumulators); research-kernels.md has the tuning plan.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <TargetConditionals.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>

#include "kokoro_common.h"
#include "neon_math.h"
#include "lokinfer/kernels/cpu.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

namespace {

enum class Epi { None, Bias, BiasRelu };

template <Epi E>
inline float32x4_t epilogue(float32x4_t v, const float* bias) {
  if constexpr (E == Epi::None) return v;
  v = vaddq_f32(v, vld1q_f32(bias));
  if constexpr (E == Epi::BiasRelu) v = vmaxq_f32(v, vdupq_n_f32(0.f));
  return v;
}

template <Epi E>
inline float epilogue1(float v, const float* bias) {
  if constexpr (E == Epi::None) return v;
  v += *bias;
  if constexpr (E == Epi::BiasRelu) v = v > 0.f ? v : 0.f;
  return v;
}

// out[i0:i0+R, j0:j0+16] = epi(a[i0:i0+R, :] @ b[:, j0:j0+16]); R in 1..4
// (R < 4 covers small-M / GEMV shapes without falling back to scalar code).
template <Epi E, int R>
inline void tile16(const float* a, const float* b, const float* bias, float* out, int64_t n, int64_t k, int64_t i0,
                   int64_t j0) {
  float32x4_t c[R][4];
  for (auto& r : c)
    for (auto& v : r) v = vdupq_n_f32(0.f);
  for (int64_t p = 0; p < k; ++p) {
    const float* bp = b + p * n + j0;
    float32x4_t b0 = vld1q_f32(bp), b1 = vld1q_f32(bp + 4), b2 = vld1q_f32(bp + 8), b3 = vld1q_f32(bp + 12);
    for (int r = 0; r < R; ++r) {
      float32x4_t av = vdupq_n_f32(a[(i0 + r) * k + p]);
      c[r][0] = vfmaq_f32(c[r][0], av, b0);
      c[r][1] = vfmaq_f32(c[r][1], av, b1);
      c[r][2] = vfmaq_f32(c[r][2], av, b2);
      c[r][3] = vfmaq_f32(c[r][3], av, b3);
    }
  }
  const float* bj = bias ? bias + j0 : nullptr;
  for (int r = 0; r < R; ++r) {
    float* o = out + (i0 + r) * n + j0;
    for (int q = 0; q < 4; ++q) vst1q_f32(o + 4 * q, epilogue<E>(c[r][q], bj ? bj + 4 * q : nullptr));
  }
}

// Column tail (n % 16): 4-wide vectors then scalars, any R.
template <Epi E>
inline void tailColumns(const float* a, const float* b, const float* bias, float* out, int64_t n, int64_t k,
                        int64_t i, int64_t j0) {
  int64_t j = j0;
  for (; j + 4 <= n; j += 4) {
    float32x4_t acc = vdupq_n_f32(0.f);
    for (int64_t p = 0; p < k; ++p) acc = vfmaq_f32(acc, vdupq_n_f32(a[i * k + p]), vld1q_f32(b + p * n + j));
    vst1q_f32(out + i * n + j, epilogue<E>(acc, bias ? bias + j : nullptr));
  }
  for (; j < n; ++j) {
    float s = 0.f;
    for (int64_t p = 0; p < k; ++p) s += a[i * k + p] * b[p * n + j];
    out[i * n + j] = epilogue1<E>(s, bias ? bias + j : nullptr);
  }
}

template <Epi E>
void gemmEpi(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k) {
  int64_t nFull = n / 16 * 16;
  // Large M: blocks of 4-row tiles on the pool (MNIST B = 1024 on the M2 Max: 2.65 ms single-threaded).
  constexpr double kParMacs = 4e6;
  if (m >= 16 && static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) >= kParMacs) {
    const int64_t quads = m / 4, threads = rt::ThreadPool::global().threads();
    const int64_t per = std::max<int64_t>(1, (quads + 3 * threads - 1) / (3 * threads));
    rt::ThreadPool::global().parallelFor((quads + per - 1) / per, [&](int64_t t) {
      for (int64_t q = t * per; q < std::min(quads, (t + 1) * per); ++q)
        for (int64_t j = 0; j < nFull; j += 16) tile16<E, 4>(a, b, bias, out, n, k, 4 * q, j);
    });
    const int64_t i0 = quads * 4, rem = m - i0;
    for (int64_t j = 0; j < nFull; j += 16) {
      if (rem == 3) tile16<E, 3>(a, b, bias, out, n, k, i0, j);
      else if (rem == 2) tile16<E, 2>(a, b, bias, out, n, k, i0, j);
      else if (rem == 1) tile16<E, 1>(a, b, bias, out, n, k, i0, j);
    }
    if (nFull < n)
      for (int64_t r = 0; r < m; ++r) tailColumns<E>(a, b, bias, out, n, k, r, nFull);
    return;
  }
  int64_t i = 0;
  for (; i + 4 <= m; i += 4)
    for (int64_t j = 0; j < nFull; j += 16) tile16<E, 4>(a, b, bias, out, n, k, i, j);
  int64_t rem = m - i;
  for (int64_t j = 0; j < nFull; j += 16) {
    if (rem == 3) tile16<E, 3>(a, b, bias, out, n, k, i, j);
    else if (rem == 2) tile16<E, 2>(a, b, bias, out, n, k, i, j);
    else if (rem == 1) tile16<E, 1>(a, b, bias, out, n, k, i, j);
  }
  if (nFull < n)
    for (int64_t r = 0; r < m; ++r) tailColumns<E>(a, b, bias, out, n, k, r, nFull);
}

// In-place epilogue over a row-major [m, n] result (Accelerate variants).
template <Epi E>
void applyEpilogue(float* out, const float* bias, int64_t m, int64_t n) {
  for (int64_t i = 0; i < m; ++i) {
    float* row = out + i * n;
    int64_t j = 0;
    for (; j + 4 <= n; j += 4) vst1q_f32(row + j, epilogue<E>(vld1q_f32(row + j), bias + j));
    for (; j < n; ++j) row[j] = epilogue1<E>(row[j], bias + j);
  }
}

using LI = __LAPACK_int;

// Small-M GEMMs (M <= 8: GEMV-like, e.g. a decode step) on big weights are bandwidth-bound and Accelerate runs them
// on too few cores: split N into column blocks on the runtime pool (single-threaded BLAS per block) so every core
// streams its share of B.
constexpr int64_t kSmallM = 8, kSplitBytes = int64_t{4} << 20;

bool splitN(int64_t m, int64_t n, int64_t k) { return m <= kSmallM && n * k * 4 >= kSplitBytes && n >= 256; }

void sgemm(const float* a, const float* b, float* out, int64_t m, int64_t n, int64_t k) {
  if (!splitN(m, n, k)) {
    rt::ThreadPool::global().park();  // see gemmBiasAct
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(m), LI(n), LI(k), 1.0f, a, LI(k), b, LI(n), 0.0f, out,
                LI(n));
    return;
  }
  const int64_t blocks = std::min<int64_t>(4 * rt::ThreadPool::global().threads(), n / 64);
  const int64_t nb = (n / blocks + 15) / 16 * 16;
  detail::parallelForSt((n + nb - 1) / nb, [&](int64_t i) {
    const int64_t n0 = i * nb, cols = std::min(nb, n - n0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(m), LI(cols), LI(k), 1.0f, a, LI(k), b + n0, LI(n),
                0.0f, out + n0, LI(n));
  });
}

// --- GPT-2 CPU prefill: Accelerate GEMM + fused epilogues, split on the pool ----------------------------------

// act: 0 none, 1 gelu_tanh (NEON exp), 2 gelu_tanh (vvtanhf), 3 relu. Rows [r0, r1) x cols [c0, c1) of out (ld n).
void epilogueBlock(float* out, const float* bias, int64_t n, int64_t r0, int64_t r1, int64_t c0, int64_t c1, int act) {
  const int64_t w = c1 - c0;
  for (int64_t r = r0; r < r1; ++r) {
    float* row = out + r * n + c0;
    const float* b = bias ? bias + c0 : nullptr;
    if (act == 2) {
      float* buf = detail::scratch(static_cast<size_t>(w), 2);
      for (int64_t j = 0; j < w; ++j) {
        const float x = row[j] + (b ? b[j] : 0.f);
        row[j] = x;
        buf[j] = 0.7978845608028654f * (x + 0.044715f * x * x * x);
      }
      const int wi = static_cast<int>(w);
      vvtanhf(buf, buf, &wi);
      for (int64_t j = 0; j < w; ++j) row[j] = 0.5f * row[j] * (1.f + buf[j]);
      continue;
    }
    int64_t j = 0;
    for (; j + 4 <= w; j += 4) {
      float32x4_t v = vld1q_f32(row + j);
      if (b) v = vaddq_f32(v, vld1q_f32(b + j));
      if (act == 1) v = detail::geluTanh4(v);
      if (act == 3) v = vmaxq_f32(v, vdupq_n_f32(0.f));
      vst1q_f32(row + j, v);
    }
    for (; j < w; ++j) {
      float v = row[j] + (b ? b[j] : 0.f);
      row[j] = act == 3 ? std::max(v, 0.f) : act ? detail::geluTanh1(v) : v;
    }
  }
}

#if TARGET_OS_IPHONE
constexpr int64_t kIphoneRowBlocks = -4;
// The generic Accelerate kernels (Kokoro's ALBERT, weights shared across layers and cache-hot) prefer two row blocks:
// an ALBERT-like encoder at T = 59: 8.9 ms (2 blocks) vs 9.8 (4) vs 11.0 (multithreaded Accelerate); T = 141: 18.4
// vs 20.7 vs 18.7. Small-N / small GEMMs stay on one Accelerate call (MNIST B=64 / 1024: 2 blocks 53 / 208 us vs
// 12 / 162 us). A/B 2026-11-01.
constexpr int64_t kIphoneGenericRowBlocks = -2;
bool iphoneRowSplit(int64_t m, int64_t n, int64_t k) {
  return m >= 16 && n >= 512 && 2.0 * double(m) * double(n) * double(k) >= 5e7;
}
#endif

// split = 0 (auto), per platform (A/B 2026-11-01, GPT-2 124M):
//  - small M (<= 8, decode GEMVs) on big weights: K split over 3 (Mac) / 2 (iPhone) tasks per pool thread (Mac decode
//    10.9 -> 7.9 ms; iPhone 6.2 vs 6.0 ms, a wash);
//  - iPhone, M >= 16, N >= 512, >= 50 MFLOP: four row blocks, each one single-threaded Accelerate stream (T = 128 prefill 26.7 -> 23.0 ms,
//    T = 512 80.1 -> 77.4; 2 blocks 23.7 / 87.5, 3 blocks 25.8;
//    Accelerate's own threading underuses the SME units at these sizes);
//  - else one multithreaded Accelerate call (the M2 Max spreads it over both P-clusters' AMX: 1 stream is 1.7x
//    slower).
int64_t autoSplit(int64_t m, int64_t n, int64_t k) {
  const int64_t threads = rt::ThreadPool::global().threads();
#if TARGET_OS_IPHONE
  constexpr int64_t kTasksPerThread = 2;
  if (m > 8) return iphoneRowSplit(m, n, k) ? kIphoneRowBlocks : 0;
#else
  constexpr int64_t kTasksPerThread = 3;
#endif
  if (m <= 8 && n * k * 4 >= (int64_t{1} << 20)) return 5000 + kTasksPerThread * threads;
  return 0;
}

void gemmBiasAct(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k,
                 int act, int64_t split) {
  if (split == 0) split = autoSplit(m, n, k);
  if (split == 6000) split = 0;  // explicit: one multithreaded Accelerate call (A/B knob)
  if (split == 0) {
    // Multi-threaded Accelerate: park the pool first, or workers still spinning from the previous parallelFor compete
    // with Accelerate's threads (Kokoro ALBERT [59,768]x[768,2304]: 224 us in-model vs 112 us alone, 2026-11-01).
    rt::ThreadPool::global().park();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(m), LI(n), LI(k), 1.0f, a, LI(k), b, LI(n), 0.0f, out,
                LI(n));
    if (!bias && act == 0) return;
    // Epilogue on the pool only when it is heavy (gelu on GPT-2's [128, 3072]); a pool round right after a
    // multithreaded Accelerate call otherwise costs more than it saves (iPhone MNIST B=1024: 292 vs 162 us).
    const int64_t rb = std::max<int64_t>(1, (m + 4 * rt::ThreadPool::global().threads() - 1) /
                                                (4 * rt::ThreadPool::global().threads()));
    const int64_t tasks = (act == 1 || act == 2) && m * n >= 65536 ? (m + rb - 1) / rb : 1;
    const int64_t rbs = tasks == 1 ? m : rb;
    rt::ThreadPool::global().parallelFor(tasks, [&](int64_t i) {
      epilogueBlock(out, bias, n, i * rbs, std::min(m, (i + 1) * rbs), 0, n, act);
    });
    return;
  }
  if (split >= 5000) {  // small M (decode GEMV): K split into `split - 5000` contiguous row ranges of B (streams
                        // whole rows, unlike column blocks), partial [m, n] sums reduced by column chunks after
    const int64_t tasks = std::max<int64_t>(1, std::min<int64_t>(split - 5000, k / 16));
    const int64_t kb = (k + tasks - 1) / tasks;
    float* parts = detail::scratch(static_cast<size_t>(tasks * m * n), 0);
    detail::parallelForSt(tasks, [&](int64_t t) {
      const int64_t k0 = t * kb, kk = std::min(kb, k - k0);
      float* pt = parts + t * m * n;
      if (kk <= 0) {
        std::fill(pt, pt + m * n, 0.f);
        return;
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(m), LI(n), LI(kk), 1.0f, a + k0, LI(k), b + k0 * n,
                  LI(n), 0.0f, pt, LI(n));
    });
    const int64_t cc = 256, chunks = (m * n + cc - 1) / cc;
    rt::ThreadPool::global().parallelFor(chunks, [&](int64_t c) {
      const int64_t e0 = c * cc, e1 = std::min(m * n, e0 + cc);
      for (int64_t e = e0; e < e1; e += 4) {
        if (e + 4 <= e1) {
          float32x4_t acc = vld1q_f32(parts + e);
          for (int64_t t = 1; t < tasks; ++t) acc = vaddq_f32(acc, vld1q_f32(parts + t * m * n + e));
          vst1q_f32(out + e, acc);
        } else {
          for (int64_t x = e; x < e1; ++x) {
            float v = 0.f;
            for (int64_t t = 0; t < tasks; ++t) v += parts[t * m * n + x];
            out[x] = v;
          }
        }
      }
      if (bias || act)
        for (int64_t r = e0 / n; r <= (e1 - 1) / n; ++r) {
          const int64_t c0 = std::max(e0, r * n) - r * n, c1 = std::min(e1, (r + 1) * n) - r * n;
          epilogueBlock(out, bias, n, r, r + 1, c0, c1, act);
        }
    });
    return;
  }
  if (split > 0) {
    const int64_t nb = std::max<int64_t>(16, (n / split + 15) / 16 * 16);
    detail::parallelForSt((n + nb - 1) / nb, [&](int64_t i) {
      const int64_t n0 = i * nb, cols = std::min(nb, n - n0);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(m), LI(cols), LI(k), 1.0f, a, LI(k), b + n0, LI(n),
                  0.0f, out + n0, LI(n));
      if (bias || act) epilogueBlock(out, bias, n, 0, m, n0, n0 + cols, act);
    });
    return;
  }
  const int64_t mb = std::max<int64_t>(1, (m + (-split) - 1) / (-split));
  detail::parallelForSt((m + mb - 1) / mb, [&](int64_t i) {
    const int64_t m0 = i * mb, rows = std::min(mb, m - m0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(rows), LI(n), LI(k), 1.0f, a + m0 * k, LI(k), b, LI(n),
                0.0f, out + m0 * n, LI(n));
    if (bias || act) epilogueBlock(out, bias, n, m0, m0 + rows, 0, n, act);
  });
}

}  // namespace

void gemm_f32(const float* a, const float* b, float* out, int64_t m, int64_t n, int64_t k) {
  gemmEpi<Epi::None>(a, b, nullptr, out, m, n, k);
}
void gemm_bias_f32(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k) {
  gemmEpi<Epi::Bias>(a, b, bias, out, m, n, k);
}
void gemm_bias_relu_f32(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n,
                        int64_t k) {
  gemmEpi<Epi::BiasRelu>(a, b, bias, out, m, n, k);
}

// EXPERIMENT knob (Mac): LOKINFER_GEMM_SPLIT=<split> routes the generic Accelerate GEMM kernels through gemmBiasAct
// with that split (see gemm_accel_f32_cpu), to A/B blockings in-model.
int64_t envSplit() {
  static const int64_t v = [] {
    const char* e = std::getenv("LOKINFER_GEMM_SPLIT");
    return e ? std::atoll(e) : INT64_MIN;
  }();
  return v;
}

void gemm_f32_accelerate(const float* a, const float* b, float* out, int64_t m, int64_t n, int64_t k) {
  if (envSplit() != INT64_MIN) return gemmBiasAct(a, b, nullptr, out, m, n, k, 0, envSplit());
#if TARGET_OS_IPHONE
  if (iphoneRowSplit(m, n, k)) return gemmBiasAct(a, b, nullptr, out, m, n, k, 0, kIphoneGenericRowBlocks);
#endif
  sgemm(a, b, out, m, n, k);
}
void gemm_bias_f32_accelerate(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n,
                              int64_t k) {
  if (envSplit() != INT64_MIN) return gemmBiasAct(a, b, bias, out, m, n, k, 0, envSplit());
#if TARGET_OS_IPHONE
  if (iphoneRowSplit(m, n, k)) return gemmBiasAct(a, b, bias, out, m, n, k, 0, kIphoneGenericRowBlocks);
#endif
  sgemm(a, b, out, m, n, k);
  applyEpilogue<Epi::Bias>(out, bias, m, n);
}
void gemm_bias_relu_f32_accelerate(const float* a, const float* b, const float* bias, float* out, int64_t m,
                                   int64_t n, int64_t k) {
  if (envSplit() != INT64_MIN) return gemmBiasAct(a, b, bias, out, m, n, k, 3, envSplit());
#if TARGET_OS_IPHONE
  if (iphoneRowSplit(m, n, k)) return gemmBiasAct(a, b, bias, out, m, n, k, 3, kIphoneGenericRowBlocks);
#endif
  sgemm(a, b, out, m, n, k);
  applyEpilogue<Epi::BiasRelu>(out, bias, m, n);
}

void gemmBiasActAccelerate(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n,
                           int64_t k, int act, int64_t split) {
  gemmBiasAct(a, b, bias, out, m, n, k, act, split);
}

void transpose_f32(const float* in, float* out, int64_t rows, int64_t cols) {
  // 8x8 blocked for cache friendliness. Hoisted into init on constants (MNIST weights); per call on activations
  // (Kokoro CPU graph: [T, C] <-> [C, T] layout changes around the text encoder / ASR features).
  constexpr int64_t B = 8;
  for (int64_t i0 = 0; i0 < rows; i0 += B)
    for (int64_t j0 = 0; j0 < cols; j0 += B)
      for (int64_t i = i0; i < std::min(i0 + B, rows); ++i)
        for (int64_t j = j0; j < std::min(j0 + B, cols); ++j) out[j * rows + i] = in[i * cols + j];
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

// ints: a(M, K), b(K, N), out(M, N)
template <void (*Fn)(const float*, const float*, float*, int64_t, int64_t, int64_t)>
void runGemm(ExecCtx&, const KernelArgs& a) {
  Fn(static_cast<const float*>(a.ptrs[0]), static_cast<const float*>(a.ptrs[1]), static_cast<float*>(a.ptrs[2]),
     a.ints[0], a.ints[3], a.ints[1]);
}

// ints: a(M, K), b(K, N), bias(N), out(M, N)
template <void (*Fn)(const float*, const float*, const float*, float*, int64_t, int64_t, int64_t)>
void runGemmBias(ExecCtx&, const KernelArgs& a) {
  Fn(static_cast<const float*>(a.ptrs[0]), static_cast<const float*>(a.ptrs[1]), static_cast<const float*>(a.ptrs[2]),
     static_cast<float*>(a.ptrs[3]), a.ints[0], a.ints[3], a.ints[1]);
}

// out[M, N] = a[M, K] @ b[N, K]^T (Accelerate, transposed B: GPT-2's tied LM head reads wte [V, D] directly).
// ints: a(M, K), b(N, K), out(M, N)
void runGemmNt(ExecCtx&, const KernelArgs& a) {
  using LI = __LAPACK_int;
  const int64_t m = a.ints[0], k = a.ints[1], n = a.ints[2];
  const float* A = static_cast<const float*>(a.ptrs[0]);
  const float* B = static_cast<const float*>(a.ptrs[1]);
  float* C = static_cast<float*>(a.ptrs[2]);
  if (!(m <= 8 && n * k * 4 >= (int64_t{4} << 20))) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, LI(m), LI(n), LI(k), 1.f, A, LI(k), B, LI(k), 0.f, C, LI(n));
    return;
  }
  // GEMV-like (decode LM head over 50k vocab rows): B's row blocks are contiguous; one block per pool task.
  const int64_t blocks = std::min<int64_t>(4 * rt::ThreadPool::global().threads(), n / 64);
  const int64_t nb = (n / blocks + 15) / 16 * 16;
  kernels::cpu::detail::parallelForSt((n + nb - 1) / nb, [&](int64_t i) {
    const int64_t n0 = i * nb, rows = std::min(nb, n - n0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, LI(m), LI(rows), LI(k), 1.f, A, LI(k), B + n0 * k, LI(k), 0.f,
                C + n0, LI(n));
  });
}

// ints: a(M, K), b(K, N), bias(N), out(M, N), act, split
void runGemmBiasAct(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::gemmBiasActAccelerate(static_cast<const float*>(a.ptrs[0]), static_cast<const float*>(a.ptrs[1]),
                                      static_cast<const float*>(a.ptrs[2]), static_cast<float*>(a.ptrs[3]), a.ints[0],
                                      a.ints[3], a.ints[1], static_cast<int>(a.ints[7]), a.ints[8]);
}

// ints: a(M, K), b(K, N), out(M, N), split
void runGemmSplit(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::gemmBiasActAccelerate(static_cast<const float*>(a.ptrs[0]), static_cast<const float*>(a.ptrs[1]),
                                      nullptr, static_cast<float*>(a.ptrs[2]), a.ints[0], a.ints[3], a.ints[1], 0,
                                      a.ints[6]);
}

// ints: in(R, C), out(C, R)
void runTranspose(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::transpose_f32(static_cast<const float*>(a.ptrs[0]), static_cast<float*>(a.ptrs[1]), a.ints[0],
                              a.ints[1]);
}

}  // namespace

void registerCpuGemmKernels() {
  using namespace kernels::cpu;
  registerKernel({"gemm_f32_cpu", Device::Cpu, nullptr, runGemm<gemm_f32>, nullptr});
  registerKernel({"gemm_f32_cpu_accelerate", Device::Cpu, nullptr, runGemm<gemm_f32_accelerate>, nullptr});
  registerKernel({"gemm_bias_f32_cpu", Device::Cpu, nullptr, runGemmBias<gemm_bias_f32>, nullptr});
  registerKernel({"gemm_bias_relu_f32_cpu", Device::Cpu, nullptr, runGemmBias<gemm_bias_relu_f32>, nullptr});
  registerKernel({"gemm_bias_f32_cpu_accelerate", Device::Cpu, nullptr, runGemmBias<gemm_bias_f32_accelerate>, nullptr});
  registerKernel({"gemm_bias_relu_f32_cpu_accelerate", Device::Cpu, nullptr, runGemmBias<gemm_bias_relu_f32_accelerate>,
                  nullptr});
  registerKernel({"transpose_f32_cpu", Device::Cpu, nullptr, runTranspose, nullptr});
  registerKernel({"gemm_nt_f32_cpu_accelerate", Device::Cpu, nullptr, runGemmNt, nullptr});
  registerKernel({"gemm_bias_act_f32_cpu_accelerate", Device::Cpu, nullptr, runGemmBiasAct, nullptr});
  registerKernel({"gemm_split_f32_cpu_accelerate", Device::Cpu, nullptr, runGemmSplit, nullptr});
}

}  // namespace lk::rt
