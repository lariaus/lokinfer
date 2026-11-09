// f16 optimized CPU kernels (Qwen3-ASR phase 1, docs/impl-notes/qwen3-asr-f16-opt-kernels.md, plan Q5): the obvious
// fusions + a reasonable kernel per device. Tested against the composition of the basic f16 kernels
// (tests/kernels/test_f16_fused_cpu.py). f16 storage, f32 accumulation and epilogue math, one rounding at the output.
//
// GEMMs (`gemm*_opt`, one engine, `gemmOpt`):
//  - M <= 4 (decode GEMVs): K split over the pool (GPT-2's scheme): each task streams a contiguous range of W rows,
//    8 rows at a time (M = 1: 2 rows × 64 columns, gemvRange1), into an f32 partial [M, N] that stays in L1/L2; the partials are summed by column chunks with
//    the fused epilogue. Bandwidth-bound: every core streams its share of W once.
//  - larger M, NEON: 4 × 16 register tiles of FMLAL / FMLAL2 by element (f16 × f16 → f32; x as 8 k values per
//    register, w rows of 16 columns), K in order; tasks = (row block, column block), each W column strip reused from
//    L1/L2 by every row quad of the block.
//  - larger M on the Mac (no SME2): Accelerate's f32 GEMM on the AMX units. x and column panels of W are converted
//    f16 -> f32 (exact) on the pool, one multithreaded cblas_sgemm per panel, the same f32 epilogue. 1.6-2.8x the
//    NEON tiles at Qwen3-ASR's prefill / encoder shapes (2026-11-05 microbench: gate|up T = 2048 49 vs 139 ms).
//  - larger M on SME2 CPUs (the iPhone): widening FMOPA panels (f16_sme2.cpp), x packed in k pairs per 32-row panel
//    chunk, P cores only (one SME unit per cluster; the efficiency cluster's is far slower, iphone-cpu-sme2.md).
// Epilogues (f32): + bias, GELU(erf), SwiGLU (w = [gate | up]: silu(g)·u), + residual, clamp; one f16 rounding.
//
// Attention (`attention_gqa_opt`, `attention_window_qkv_opt`): flash-style. A task holds a block of queries (rows ×
// the q heads sharing one kv head, or one window's rows of one head) and walks the keys in tiles of 64: scores by
// 4 × 4 FMLAL dot-product blocks, online f32 softmax (running max / sum, accumulator rescaled), V tile converted to
// f32 once per tile and shared by every query of the block. Short query blocks over long contexts (decode) split the
// keys over tasks; the partial (max, sum, accumulator) triples are merged afterwards. Never a [T, P] matrix.
//
// Kernel ABI (lokinfer/runtime/kernel.h): ptrs = operands then outputs; ints = every operand's shape dims, then the
// outputs' dims, then int attrs in attribute-name order; floats = float attrs in attribute-name order.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <TargetConditionals.h>
#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "f16_sme2.h"
#include "kokoro_common.h"
#include "neon_math.h"
#include "lokinfer/runtime/kernel.h"
#include "lokinfer/runtime/thread_pool.h"

// FMLAL / FMLAL2 (FEAT_FHM): default on the Mac target, not on the generic iOS arm64 target (every iOS 27 CPU has it).
#pragma clang attribute push(__attribute__((target("fp16fml"))), apply_to = function)

namespace lk::rt {

namespace {

using h16 = __fp16;
using kernels::cpu::detail::exp4;
using kernels::cpu::detail::geluErf4;

const h16* H(const KernelArgs& a, int i) { return static_cast<const h16*>(a.ptrs[i]); }
h16* Hout(const KernelArgs& a, int i) { return static_cast<h16*>(a.ptrs[i]); }
int64_t I64(const KernelArgs& a, int i) { return *static_cast<const int64_t*>(a.ptrs[i]); }

// --- helpers --------------------------------------------------------------------------------------------------

inline void load8(const h16* p, float32x4_t& lo, float32x4_t& hi) {
  const float16x8_t h = vld1q_f16(p);
  lo = vcvt_f32_f16(vget_low_f16(h));
  hi = vcvt_high_f32_f16(h);
}

inline void store8(h16* p, float32x4_t lo, float32x4_t hi) {
  vst1q_f16(p, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
}

// Grow-only thread-local scratch, one per slot (steady-state calls do not allocate).
template <typename T>
T* scratch(size_t n, int slot) {
  thread_local std::vector<T> bufs[6];
  std::vector<T>& b = bufs[slot];
  if (b.size() < n) b.resize(n);
  return b.data();
}

// The basic kernels' GELU (f16_basic.cpp): 0.5·v·erfc(-v/√2), accurate in the negative tail.
inline float geluErf(float v) { return 0.5f * v * std::erfc(-v * 0.70710678118654752f); }

// fn(begin, end) over [0, n) in contiguous ranges on the pool; inline when the work is small.
void parallelRanges(int64_t n, int64_t work, const std::function<void(int64_t, int64_t)>& fn) {
  ThreadPool& pool = ThreadPool::global();
  const int64_t tasks = work < (int64_t{1} << 16) ? 1 : std::min<int64_t>(n, 4 * pool.threads());
  if (tasks <= 1) {
    if (n > 0) fn(0, n);
    return;
  }
  const int64_t per = (n + tasks - 1) / tasks;
  pool.parallelFor((n + per - 1) / per, [&](int64_t i) { fn(i * per, std::min(n, (i + 1) * per)); });
}

// Tasks on the pool (inline when there is one).
void runTasks(int64_t n, const std::function<void(int64_t)>& fn) {
  if (n == 1) return fn(0);
  ThreadPool::global().parallelFor(n, fn);
}

// --- GEMM epilogue ------------------------------------------------------------------------------------------------

struct Epi {
  const h16* bias = nullptr;  // [N]
  const h16* res = nullptr;   // [M, N]
  bool gelu = false;
  bool swiglu = false;  // w is [K, 2N]: out = silu(y[:, :N])·y[:, N:]
  float lim = 0.f;      // > 0: clamp to ±lim (after the residual)
};

// out[r, c0:c0+n] from the f32 sums v (gate) and u (up, SwiGLU only).
inline void finish(const float* v, const float* u, const Epi& e, int64_t r, int64_t c0, int64_t n, int64_t ldo,
                   h16* out) {
  h16* o = out + r * ldo + c0;
  const h16* b = e.bias ? e.bias + c0 : nullptr;
  const h16* rs = e.res ? e.res + r * ldo + c0 : nullptr;
  int64_t j = 0;
  {
    const float32x4_t one = vdupq_n_f32(1.f), lim = vdupq_n_f32(e.lim), nlim = vdupq_n_f32(-e.lim);
    for (; j + 8 <= n; j += 8) {
      float32x4_t a0 = vld1q_f32(v + j), a1 = vld1q_f32(v + j + 4);
      if (b) {
        float32x4_t b0, b1;
        load8(b + j, b0, b1);
        a0 = vaddq_f32(a0, b0);
        a1 = vaddq_f32(a1, b1);
      }
      if (e.gelu) {
        a0 = geluErf4(a0);
        a1 = geluErf4(a1);
      }
      if (u) {
        a0 = vmulq_f32(vdivq_f32(a0, vaddq_f32(one, exp4(vnegq_f32(a0)))), vld1q_f32(u + j));
        a1 = vmulq_f32(vdivq_f32(a1, vaddq_f32(one, exp4(vnegq_f32(a1)))), vld1q_f32(u + j + 4));
      }
      if (rs) {
        float32x4_t r0, r1;
        load8(rs + j, r0, r1);
        a0 = vaddq_f32(r0, a0);
        a1 = vaddq_f32(r1, a1);
      }
      if (e.lim > 0.f) {
        a0 = vminq_f32(vmaxq_f32(a0, nlim), lim);
        a1 = vminq_f32(vmaxq_f32(a1, nlim), lim);
      }
      store8(o + j, a0, a1);
    }
  }
  for (; j < n; ++j) {
    float a = v[j];
    if (b) a += static_cast<float>(b[j]);
    if (e.gelu) a = geluErf(a);
    if (u) a = a / (1.f + std::exp(-a)) * u[j];
    if (rs) a = static_cast<float>(rs[j]) + a;
    if (e.lim > 0.f) a = std::min(e.lim, std::max(-e.lim, a));
    o[j] = static_cast<h16>(a);
  }
}

// --- NEON tile GEMM -------------------------------------------------------------------------------------------------

// c[MR][16] = x[0:MR, 0:k] @ w[0:k, 0:16] (x row stride ldx, w row stride ldw), FMLAL by element: per 8 k, one
// register of x per row, two of w per k.
template <int MR>
inline void tile4x16(const h16* x, int64_t ldx, const h16* w, int64_t ldw, int64_t k, float* c) {
  float32x4_t acc[MR][4];
  for (int r = 0; r < MR; ++r)
    for (int q = 0; q < 4; ++q) acc[r][q] = vdupq_n_f32(0.f);
  int64_t kk = 0;
  for (; kk + 8 <= k; kk += 8) {
    float16x8_t xv[MR];
    for (int r = 0; r < MR; ++r) xv[r] = vld1q_f16(x + r * ldx + kk);
    const h16* wp = w + kk * ldw;
#define LK_STEP(L)                                                    \
  {                                                                   \
    const float16x8_t w0 = vld1q_f16(wp + (L) * ldw);                 \
    const float16x8_t w1 = vld1q_f16(wp + (L) * ldw + 8);             \
    for (int r = 0; r < MR; ++r) {                                    \
      acc[r][0] = vfmlalq_laneq_low_f16(acc[r][0], w0, xv[r], L);     \
      acc[r][1] = vfmlalq_laneq_high_f16(acc[r][1], w0, xv[r], L);    \
      acc[r][2] = vfmlalq_laneq_low_f16(acc[r][2], w1, xv[r], L);     \
      acc[r][3] = vfmlalq_laneq_high_f16(acc[r][3], w1, xv[r], L);    \
    }                                                                 \
  }
    LK_STEP(0) LK_STEP(1) LK_STEP(2) LK_STEP(3) LK_STEP(4) LK_STEP(5) LK_STEP(6) LK_STEP(7)
#undef LK_STEP
  }
  for (; kk < k; ++kk) {
    const float16x8_t w0 = vld1q_f16(w + kk * ldw), w1 = vld1q_f16(w + kk * ldw + 8);
    for (int r = 0; r < MR; ++r) {
      const float16x8_t xv = vdupq_n_f16(x[r * ldx + kk]);
      acc[r][0] = vfmlalq_low_f16(acc[r][0], w0, xv);
      acc[r][1] = vfmlalq_high_f16(acc[r][1], w0, xv);
      acc[r][2] = vfmlalq_low_f16(acc[r][2], w1, xv);
      acc[r][3] = vfmlalq_high_f16(acc[r][3], w1, xv);
    }
  }
  for (int r = 0; r < MR; ++r)
    for (int q = 0; q < 4; ++q) vst1q_f32(c + r * 16 + 4 * q, acc[r][q]);
}

inline void tileRows(int64_t mr, const h16* x, int64_t ldx, const h16* w, int64_t ldw, int64_t k, float* c) {
  switch (mr) {
    case 4: return tile4x16<4>(x, ldx, w, ldw, k, c);
    case 3: return tile4x16<3>(x, ldx, w, ldw, k, c);
    case 2: return tile4x16<2>(x, ldx, w, ldw, k, c);
    default: return tile4x16<1>(x, ldx, w, ldw, k, c);
  }
}

// w columns [c0, c0 + nc) (nc <= 16) packed into a contiguous zero-padded [k, 16] strip (the row stride of w is
// N·2 bytes: read in place, a strip touches a new page every few k and the tile kernel ran at ~1/4 of its packed speed).
const h16* packStrip(const h16* w, int64_t ldw, int64_t k, int64_t c0, int64_t nc, int slot) {
  h16* s = scratch<h16>(static_cast<size_t>(k * 16), slot);
  const h16* src = w + c0;
  if (nc == 16) {
    for (int64_t kk = 0; kk < k; ++kk, src += ldw) {
      vst1q_f16(s + kk * 16, vld1q_f16(src));
      vst1q_f16(s + kk * 16 + 8, vld1q_f16(src + 8));
    }
    return s;
  }
  for (int64_t kk = 0; kk < k; ++kk, src += ldw) {
    std::memcpy(s + kk * 16, src, static_cast<size_t>(nc) * sizeof(h16));
    std::memset(s + kk * 16 + nc, 0, static_cast<size_t>(16 - nc) * sizeof(h16));
  }
  return s;
}

void gemmTiles(const h16* x, const h16* w, const Epi& e, h16* out, int64_t m, int64_t k, int64_t n) {
  const int64_t ldw = e.swiglu ? 2 * n : n;
  const int64_t strips = (n + 15) / 16;
  const int64_t want = 3 * ThreadPool::global().threads();
  int64_t sp = 4, mb = 128;
  auto count = [&] { return ((strips + sp - 1) / sp) * ((m + mb - 1) / mb); };
  while (count() < want && sp > 1) sp /= 2;
  while (count() < want && mb > 4) mb /= 2;
  if (2.0 * double(m) * double(n) * double(k) < 4e6) sp = strips, mb = m;  // small: one task
  const int64_t cb = (strips + sp - 1) / sp, rb = (m + mb - 1) / mb;
  runTasks(cb * rb, [&](int64_t t) {
    const int64_t s0 = (t % cb) * sp, s1 = std::min(strips, s0 + sp);
    const int64_t r0 = (t / cb) * mb, r1 = std::min(m, r0 + mb);
    float cg[4 * 16], cu[4 * 16];
    for (int64_t s = s0; s < s1; ++s) {
      const int64_t c0 = s * 16, nc = std::min<int64_t>(16, n - c0);
      const h16* wg = packStrip(w, ldw, k, c0, nc, 0);
      const h16* wu = e.swiglu ? packStrip(w, ldw, k, n + c0, nc, 1) : nullptr;
      for (int64_t i = r0; i < r1; i += 4) {
        const int64_t mr = std::min<int64_t>(4, r1 - i);
        tileRows(mr, x + i * k, k, wg, 16, k, cg);
        if (wu) tileRows(mr, x + i * k, k, wu, 16, k, cu);
        for (int64_t r = 0; r < mr; ++r) finish(cg + r * 16, wu ? cu + r * 16 : nullptr, e, i + r, c0, nc, n, out);
      }
    }
  });
}

// --- K-split GEMV (M <= 4) -------------------------------------------------------------------------------------------

constexpr int64_t kGemvMaxM = 4;
// K-split tasks per pool thread. Mac, 1: Qwen3-ASR decode GEMVs (DRAM-bound, distinct weights) o 90 -> 100 GB/s,
// qkv 103 -> 115, down 102 -> 107, swiglu 102 -> 105 vs 3 (2026-11-05; 2: in between, 4: worse): fewer partials
// to reduce and no task hand-offs; 2026-11-07 with the 2-row GEMV: 1 = 115, 2 = 113, 3 = 107 GB/s quiet (2 is 7-9 %
// faster with 8 busy threads competing, docs/models/qwen3-asr.md "CPU decode GEMV streaming"). iPhone (6 threads:
// 2 P + 4 E cores), 2026-11-07, decode step C = 480 (profile, alternating deploys): 1 / 2 / 3 = 48.7 / 40.8-42.5 /
// 49.6-50.1 ms: 2 balances the P / E speed gap without the extra partials of 3.
#if TARGET_OS_IPHONE
constexpr int64_t kGemvTasksPerThread = 2;
#else
constexpr int64_t kGemvTasksPerThread = 1;
#endif

// part[m, 0:ncol] += x[m, k0:k1] @ w[k0:k1, 0:ncol] (part zeroed first), 8 rows of w per pass.
template <int M>
void gemvRange(const h16* x, int64_t ldx, const h16* w, int64_t ldw, int64_t ncol, int64_t k0, int64_t k1, float* part,
               int64_t ldp) {
  for (int r = 0; r < M; ++r) std::memset(part + r * ldp, 0, static_cast<size_t>(ncol) * sizeof(float));
  const int64_t n8 = ncol / 8 * 8;
  int64_t kk = k0;
  for (; kk + 8 <= k1; kk += 8) {
    float16x8_t xv[M];
    for (int r = 0; r < M; ++r) xv[r] = vld1q_f16(x + r * ldx + kk);
    const h16* wr = w + kk * ldw;
    for (int64_t j = 0; j < n8; j += 8) {
      float16x8_t wv[8];
      for (int q = 0; q < 8; ++q) wv[q] = vld1q_f16(wr + q * ldw + j);
      for (int r = 0; r < M; ++r) {
        float* p = part + r * ldp + j;
        float32x4_t lo = vld1q_f32(p), hi = vld1q_f32(p + 4);
#define LK_G(L)                                           \
  lo = vfmlalq_laneq_low_f16(lo, wv[L], xv[r], L);         \
  hi = vfmlalq_laneq_high_f16(hi, wv[L], xv[r], L);
        LK_G(0) LK_G(1) LK_G(2) LK_G(3) LK_G(4) LK_G(5) LK_G(6) LK_G(7)
#undef LK_G
        vst1q_f32(p, lo);
        vst1q_f32(p + 4, hi);
      }
    }
    for (int64_t j = n8; j < ncol; ++j)
      for (int r = 0; r < M; ++r) {
        float s = part[r * ldp + j];
        for (int q = 0; q < 8; ++q) s += static_cast<float>(x[r * ldx + kk + q]) * static_cast<float>(wr[q * ldw + j]);
        part[r * ldp + j] = s;
      }
  }
  for (; kk < k1; ++kk) {
    const h16* wr = w + kk * ldw;
    for (int r = 0; r < M; ++r) {
      const float xs = static_cast<float>(x[r * ldx + kk]);
      const float16x8_t xv = vdupq_n_f16(x[r * ldx + kk]);
      float* p = part + r * ldp;
      int64_t j = 0;
      for (; j < n8; j += 8) {
        const float16x8_t wv = vld1q_f16(wr + j);
        vst1q_f32(p + j, vfmlalq_low_f16(vld1q_f32(p + j), wv, xv));
        vst1q_f32(p + j + 4, vfmlalq_high_f16(vld1q_f32(p + j + 4), wv, xv));
      }
      for (; j < ncol; ++j) p[j] += xs * static_cast<float>(wr[j]);
    }
  }
}

// M = 1: part[0:ncol] = x[k0:k1] @ w[k0:k1, 0:ncol], 2 W rows per pass × 64 columns per step (16 f32 accumulators).
// The same FMLAL sequence per output lane as gemvRange<1> (rows in k order into one f32 accumulator: bit-identical),
// but 2 streams per core instead of 8 and a quarter of the partial's load / store traffic per step. Mac, 2026-11-07
// (decode weights >> SLC, alternating rounds): 104 -> 115.5 GB/s; rows per pass 1 / 2 / 4 / 8 with 16-column steps:
// 103 / 110 / 100 / 104; 2 rows × 16 / 32 / 64 columns: 112 / 112.7 / 115.5.
void gemvRange1(const h16* x, const h16* w, int64_t ldw, int64_t ncol, int64_t k0, int64_t k1, float* part) {
  std::memset(part, 0, static_cast<size_t>(ncol) * sizeof(float));
  const int64_t n64 = ncol / 64 * 64, n8 = ncol / 8 * 8;
  int64_t kk = k0;
  for (; kk + 2 <= k1; kk += 2) {
    const float16x8_t x0 = vdupq_n_f16(x[kk]), x1 = vdupq_n_f16(x[kk + 1]);
    const h16* w0 = w + kk * ldw;
    const h16* w1 = w0 + ldw;
    int64_t j = 0;
    for (; j < n64; j += 64) {
      float* p = part + j;
      float32x4_t a[16];
      for (int i = 0; i < 16; ++i) a[i] = vld1q_f32(p + 4 * i);
      for (int i = 0; i < 8; ++i) {
        const float16x8_t v = vld1q_f16(w0 + j + 8 * i);
        a[2 * i] = vfmlalq_low_f16(a[2 * i], v, x0);
        a[2 * i + 1] = vfmlalq_high_f16(a[2 * i + 1], v, x0);
      }
      for (int i = 0; i < 8; ++i) {
        const float16x8_t v = vld1q_f16(w1 + j + 8 * i);
        a[2 * i] = vfmlalq_low_f16(a[2 * i], v, x1);
        a[2 * i + 1] = vfmlalq_high_f16(a[2 * i + 1], v, x1);
      }
      for (int i = 0; i < 16; ++i) vst1q_f32(p + 4 * i, a[i]);
    }
    for (; j < n8; j += 8) {
      float32x4_t lo = vld1q_f32(part + j), hi = vld1q_f32(part + j + 4);
      const float16x8_t v0 = vld1q_f16(w0 + j), v1 = vld1q_f16(w1 + j);
      lo = vfmlalq_low_f16(lo, v0, x0);
      hi = vfmlalq_high_f16(hi, v0, x0);
      lo = vfmlalq_low_f16(lo, v1, x1);
      hi = vfmlalq_high_f16(hi, v1, x1);
      vst1q_f32(part + j, lo);
      vst1q_f32(part + j + 4, hi);
    }
    for (; j < ncol; ++j) {  // f16 × f16 products are exact in f32: one rounding per row, as FMLAL
      part[j] += static_cast<float>(x[kk]) * static_cast<float>(w0[j]);
      part[j] += static_cast<float>(x[kk + 1]) * static_cast<float>(w1[j]);
    }
  }
  for (; kk < k1; ++kk) {
    const h16* wr = w + kk * ldw;
    const float xs = static_cast<float>(x[kk]);
    const float16x8_t xv = vdupq_n_f16(x[kk]);
    int64_t j = 0;
    for (; j < n8; j += 8) {
      const float16x8_t v = vld1q_f16(wr + j);
      vst1q_f32(part + j, vfmlalq_low_f16(vld1q_f32(part + j), v, xv));
      vst1q_f32(part + j + 4, vfmlalq_high_f16(vld1q_f32(part + j + 4), v, xv));
    }
    for (; j < ncol; ++j) part[j] += xs * static_cast<float>(wr[j]);
  }
}

void gemvKSplit(const h16* x, const h16* w, const Epi& e, h16* out, int64_t m, int64_t k, int64_t n) {
  const int64_t ncol = e.swiglu ? 2 * n : n;
  const bool small = double(m) * double(k) * double(ncol) < double(1 << 18);
  const int64_t tpt = kGemvTasksPerThread;
  // Very wide W (SwiGLU's gate|up, 24,576 columns): two column blocks, so each task's f32 partial (48 KB) stays in L1
  // while W streams (Mac, 2026-11-05: 103 -> 108 GB/s; 4 blocks 101)
  const int64_t cbs = small || ncol < 16384 ? 1 : 2;  // column blocks
  const int64_t cbw = ((ncol + cbs - 1) / cbs + 7) / 8 * 8;
  const int64_t tasks = small ? 1 : std::max<int64_t>(1, std::min<int64_t>(tpt * ThreadPool::global().threads() / cbs, k / 64));
  const int64_t kb = ((k + tasks - 1) / tasks + 7) / 8 * 8;  // multiple of 8 (whole FMLAL lane groups per task)
  const int64_t nt = (k + kb - 1) / kb;
  // The partials live in the calling thread's scratch (written by every task, read by the reduction).
  float* parts = scratch<float>(static_cast<size_t>(nt * m * ncol), 2);
  runTasks(nt * cbs, [&](int64_t task) {
    const int64_t t = task / cbs, c0 = (task % cbs) * cbw, cw = std::min(cbw, ncol - c0);
    if (cw <= 0) return;
    const int64_t k0 = t * kb, k1 = std::min(k, k0 + kb);
    float* p = parts + t * m * ncol + c0;
    switch (m) {
      case 1: return gemvRange1(x, w + c0, ncol, cw, k0, k1, p);
      case 2: return gemvRange<2>(x, k, w + c0, ncol, cw, k0, k1, p, ncol);
      case 3: return gemvRange<3>(x, k, w + c0, ncol, cw, k0, k1, p, ncol);
      default: return gemvRange<4>(x, k, w + c0, ncol, cw, k0, k1, p, ncol);
    }
  });
  constexpr int64_t kChunk = 256;
  const int64_t chunks = (n + kChunk - 1) / kChunk;
  runTasks(small ? 1 : m * chunks, [&](int64_t c) {
    const int64_t c0all = small ? 0 : c;
    const int64_t cEnd = small ? m * chunks : c + 1;
    float g[kChunk], u[kChunk];
    for (int64_t cc = c0all; cc < cEnd; ++cc) {
      const int64_t r = cc / chunks, c0 = (cc % chunks) * kChunk, w8 = std::min(kChunk, n - c0);
      for (int64_t j = 0; j < w8; j += 4) {
        const bool full = j + 4 <= w8;
        float32x4_t sg = vdupq_n_f32(0.f), su = sg;
        for (int64_t t = 0; t < nt; ++t) {
          const float* p = parts + (t * m + r) * ncol + c0 + j;
          if (full) {
            sg = vaddq_f32(sg, vld1q_f32(p));
            if (e.swiglu) su = vaddq_f32(su, vld1q_f32(p + n));
          } else {
            for (int64_t q = 0; q < w8 - j; ++q) {
              sg[q] += p[q];
              if (e.swiglu) su[q] += p[n + q];
            }
          }
        }
        if (full) {
          vst1q_f32(g + j, sg);
          vst1q_f32(u + j, su);
        } else {
          for (int64_t q = 0; q < w8 - j; ++q) g[j + q] = sg[q], u[j + q] = su[q];
        }
      }
      finish(g, e.swiglu ? u : nullptr, e, r, c0, w8, n, out);
    }
  });
}

// --- SME2 GEMM (large M on SME2 CPUs) ---------------------------------------------------------------------------------

// Tasks run on the P cores only when the CPU has two clusters (the iPhone: the efficiency cluster's SME unit is ~9x
// slower); any task left over runs on the caller.
void runPerfTasks(int64_t n, const std::function<void(int64_t)>& fn) {
  namespace d = kernels::cpu::detail;
  if (d::perfCoreSplit() <= 0) return runTasks(n, fn);
  std::atomic<int64_t> next{0};
  ThreadPool::global().parallelFor(ThreadPool::global().threads(), [&](int64_t) {
    if (!d::onPerfCore()) return;
    for (int64_t t; (t = next.fetch_add(1, std::memory_order_relaxed)) < n;) fn(t);
  });
  for (int64_t t; (t = next.fetch_add(1, std::memory_order_relaxed)) < n;) fn(t);
}

void gemmSme(const h16* x, const h16* w, const Epi& e, h16* out, int64_t m, int64_t k, int64_t n) {
  namespace s2 = kernels::cpu::sme2f16;
  const int64_t ldw = e.swiglu ? 2 * n : n;
  constexpr int64_t kChunkPanels = 8, kCols = 128;
  const int64_t panels = (m + 31) / 32, cbs = (n + kCols - 1) / kCols;
  uint32_t* packed = scratch<uint32_t>(static_cast<size_t>(std::min(panels, kChunkPanels) * 32 * k / 2), 3);
  for (int64_t p0 = 0; p0 < panels; p0 += kChunkPanels) {
    const int64_t p1 = std::min(panels, p0 + kChunkPanels);
    runTasks(p1 - p0, [&](int64_t i) {
      const int64_t r0 = (p0 + i) * 32;
      s2::packPairs(x + r0 * k, k, std::min<int64_t>(32, m - r0), k, packed + i * 32 * (k / 2));
    });
    runPerfTasks((p1 - p0) * cbs, [&](int64_t t) {
      const int64_t pi = t / cbs, c0 = (t % cbs) * kCols, nc = std::min(kCols, n - c0);
      const int64_t r0 = (p0 + pi) * 32, nr = std::min<int64_t>(32, m - r0);
      float* cg = scratch<float>(32 * kCols, 4);
      float* cu = scratch<float>(32 * kCols, 5);
      const uint32_t* ap = packed + pi * 32 * (k / 2);
      s2::panelGemm(ap, k, w + c0, ldw, nc, nr, cg, kCols);
      if (e.swiglu) s2::panelGemm(ap, k, w + n + c0, ldw, nc, nr, cu, kCols);
      for (int64_t r = 0; r < nr; ++r)
        finish(cg + r * kCols, e.swiglu ? cu + r * kCols : nullptr, e, r0 + r, c0, nc, n, out);
    });
  }
}

// --- Accelerate (AMX) GEMM (large M on the Mac) ----------------------------------------------------------------------

#if !TARGET_OS_IPHONE
constexpr int64_t kAmxMinM = 16;

inline void cvtF32(const h16* s, float* d, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const float16x8_t v = vld1q_f16(s + i);
    vst1q_f32(d + i, vcvt_f32_f16(vget_low_f16(v)));
    vst1q_f32(d + i + 4, vcvt_high_f32_f16(v));
  }
  for (; i < n; ++i) d[i] = static_cast<float>(s[i]);
}

// Grow-only buffers of the calling (executor) thread: x, gate / up panels, gate / up results.
float* amxBuf(int slot, size_t n) {
  thread_local std::vector<float> bufs[5];
  if (bufs[slot].size() < n) bufs[slot].resize(n);
  return bufs[slot].data();
}

// BNNS f16 x f16 -> f32 matmul (AMX f16, f32 accumulation): ~2x convert + sgemm (gate|up T = 401: 7.8 vs 15.4 ms,
// down 3.8 vs 6.5; 2.6 TF/s, 2026-11-05), no f32 copy of W. BNNSMatMul is deprecated in favour of BNNSGraph (macOS
// 15 SDK) but present in the macOS 27 SDK; gemmAmx's sgemm path stays as the fallback if it ever fails.
BNNSNDArrayDescriptor bnnsMatrix(const void* p, BNNSDataType t, int64_t rows, int64_t cols) {
  BNNSNDArrayDescriptor d{};
  d.layout = BNNSDataLayoutRowMajorMatrix;
  d.size[0] = static_cast<size_t>(cols);  // the fastest dimension first
  d.size[1] = static_cast<size_t>(rows);
  d.data = const_cast<void*>(p);
  d.data_type = t;
  d.data_scale = 1;
  return d;
}

bool gemmBnns(const h16* x, const h16* w, const Epi& e, h16* out, int64_t m, int64_t k, int64_t n) {
  const int64_t nc = e.swiglu ? 2 * n : n;  // W [K, nc] contiguous; SwiGLU: gate and up in one matmul
  const int64_t rb = std::min(m, std::max<int64_t>(16, ((int64_t{8} << 20) / nc) / 16 * 16));  // C <= ~32 MB
  thread_local std::vector<char> work;
  float* c = amxBuf(3, static_cast<size_t>(rb * nc));
  for (int64_t r0 = 0; r0 < m; r0 += rb) {
    const int64_t mr = std::min(rb, m - r0);
    BNNSNDArrayDescriptor A = bnnsMatrix(x + r0 * k, BNNSDataTypeFloat16, mr, k);
    BNNSNDArrayDescriptor B = bnnsMatrix(w, BNNSDataTypeFloat16, k, nc);
    BNNSNDArrayDescriptor C = bnnsMatrix(c, BNNSDataTypeFloat32, mr, nc);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    const ssize_t ws = BNNSMatMulWorkspaceSize(false, false, 1.f, &A, &B, &C, nullptr);
    if (ws < 0) return false;
    if (work.size() < static_cast<size_t>(ws)) work.resize(static_cast<size_t>(ws));
    ThreadPool::global().park();  // BNNS runs its own threads
    if (BNNSMatMul(false, false, 1.f, &A, &B, &C, ws > 0 ? work.data() : nullptr, nullptr) != 0) return false;
#pragma clang diagnostic pop
    parallelRanges(mr, mr * n, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i)
        finish(c + i * nc, e.swiglu ? c + i * nc + n : nullptr, e, r0 + i, 0, n, n, out);
    });
  }
  return true;
}

void gemmAmx(const h16* x, const h16* w, const Epi& e, h16* out, int64_t m, int64_t k, int64_t n) {
  if (gemmBnns(x, w, e, out, m, k, n)) return;
  using LI = __LAPACK_int;
  const int64_t ldw = e.swiglu ? 2 * n : n;
  // W panels of <= ~16 MB of f32 (L2-friendly for the conversion, big enough N for the AMX GEMM)
  const int64_t nb = std::min(n, std::max<int64_t>(256, ((int64_t{4} << 20) / k) / 64 * 64));
  float* xf = amxBuf(0, static_cast<size_t>(m * k));
  float* wg = amxBuf(1, static_cast<size_t>(k * nb));
  float* wu = e.swiglu ? amxBuf(2, static_cast<size_t>(k * nb)) : nullptr;
  float* cg = amxBuf(3, static_cast<size_t>(m * nb));
  float* cu = e.swiglu ? amxBuf(4, static_cast<size_t>(m * nb)) : nullptr;
  parallelRanges(m, m * k, [&](int64_t r0, int64_t r1) { cvtF32(x + r0 * k, xf + r0 * k, (r1 - r0) * k); });
  for (int64_t c0 = 0; c0 < n; c0 += nb) {
    const int64_t nc = std::min(nb, n - c0);
    parallelRanges(k, k * nc, [&](int64_t k0, int64_t k1) {
      for (int64_t kk = k0; kk < k1; ++kk) {
        cvtF32(w + kk * ldw + c0, wg + kk * nc, nc);
        if (wu) cvtF32(w + kk * ldw + n + c0, wu + kk * nc, nc);
      }
    });
    // multithreaded Accelerate: park the pool first (spinning workers compete with Accelerate's threads, gemm.cpp)
    ThreadPool::global().park();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(m), LI(nc), LI(k), 1.f, xf, LI(k), wg, LI(nc), 0.f, cg,
                LI(nc));
    if (wu)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(m), LI(nc), LI(k), 1.f, xf, LI(k), wu, LI(nc), 0.f,
                  cu, LI(nc));
    parallelRanges(m, m * nc, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) finish(cg + r * nc, wu ? cu + r * nc : nullptr, e, r, c0, nc, n, out);
    });
  }
}
#endif

// SME2 for every M above the GEMV range. iPhone 18 Pro Max, 2026-11-07 (ms, SME2 vs NEON tiles): qkv [2048x4096] M 5 /
// 8 / 13: 0.31 / 0.28 / 0.28 vs 0.59 / 0.59 / 0.85; M 16 / 32 / 128: 0.28-0.37 / 0.31-0.38 / 1.8-2.1 vs 0.86 / 1.43 /
// 4.8; enc o and conv_out (K 7680) likewise 1.6-4x; prefill 2 s / 5 s on the tiles 381 / 624 vs 222-227 / 317-351 ms.
constexpr int64_t kSmeMinM = 5;

void gemmOpt(const h16* x, const h16* w, const Epi& e, h16* out, int64_t m, int64_t k, int64_t n) {
  if (m <= 0 || n <= 0) return;
  if (m <= kGemvMaxM) return gemvKSplit(x, w, e, out, m, k, n);
  if (m >= kSmeMinM && k % 2 == 0 && kernels::cpu::sme2::available()) return gemmSme(x, w, e, out, m, k, n);
#if !TARGET_OS_IPHONE
  if (m >= kAmxMinM && !kernels::cpu::sme2::available()) return gemmAmx(x, w, e, out, m, k, n);
#endif
  gemmTiles(x, w, e, out, m, k, n);
}

// ptrs: x, w, [b], [res], out. ints: x(M, K), w(K, N or 2N), [b(N)], [res(M, N)], out(M, N). floats: [lim]
template <bool Bias, bool Res, bool Gelu, bool Clamp, bool Swiglu>
void runGemmOpt(ExecCtx&, const KernelArgs& a) {
  Epi e;
  int p = 2, i = 4;
  if (Bias) e.bias = H(a, p++), i += 1;
  if (Res) e.res = H(a, p++), i += 2;
  e.gelu = Gelu;
  e.swiglu = Swiglu;
  if (Clamp) e.lim = static_cast<float>(a.floats[0]);
  gemmOpt(H(a, 0), H(a, 1), e, Hout(a, p), a.ints[0], a.ints[1], a.ints[i + 1]);
}

// --- rmsnorm + GEMM -------------------------------------------------------------------------------------------------

// rmsnorm_gemm[_swiglu]_opt: ptrs x, g, w, out, ws. ints x(M, K), g(K), w(K, Nw), out(M, N), ws(M, K); floats eps.
// The basic rmsnorm (f16_basic.cpp rmsGroups: sequential sum of squares, the same roundings) into the workspace ws,
// then the optimized GEMM on it (the CPU gains nothing from fusing them; the op exists for the GPU's decode).
template <bool Swiglu>
void runRmsGemmOpt(ExecCtx&, const KernelArgs& a) {
  const int64_t m = a.ints[0], k = a.ints[1], nw = a.ints[4];
  const h16 *x = H(a, 0), *g = H(a, 1);
  h16* ws = Hout(a, 4);
  const float eps = static_cast<float>(a.floats[0]);
  parallelRanges(m, m * k, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const h16* xr = x + i * k;
      float sq = 0.f;
      for (int64_t j = 0; j < k; ++j) sq += static_cast<float>(xr[j]) * static_cast<float>(xr[j]);
      const float r = 1.f / std::sqrt(sq / static_cast<float>(k) + eps);
      for (int64_t j = 0; j < k; ++j) {
        const h16 h = static_cast<h16>(static_cast<float>(xr[j]) * r);
        ws[i * k + j] = static_cast<h16>(static_cast<float>(g[j]) * static_cast<float>(h));
      }
    }
  });
  Epi e;
  e.swiglu = Swiglu;
  gemmOpt(ws, H(a, 2), e, Hout(a, 3), m, k, Swiglu ? nw / 2 : nw);
}

// --- QK-norm + RoPE + KV write ------------------------------------------------------------------------------------

// The basic composition's math: rmsnorm over each hd-wide head (h = f16(x·r), y = f16(g·h)), then rotate_half RoPE
// of the f16 y at position p (cos / sin as rope_f16_cpu), one rounding. NEON (hd % 16 == 0): the sum of squares in 4
// lanes (another order than the basic kernel's sequential sum: r may differ in its last bit, an h at a rounding
// midpoint then flips; the tests require >= 99.9 % bit-exact outputs).
inline void normRopeHead(const h16* x, const h16* g, int64_t hd, float eps, const float* cs, const float* sn,
                         h16* o) {
  const int64_t half = hd / 2;
  h16 y[256];
  if (hd % 16 == 0) {
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = s0;
    for (int64_t j = 0; j < hd; j += 8) {
      float32x4_t lo, hi;
      load8(x + j, lo, hi);
      s0 = vfmaq_f32(s0, lo, lo);
      s1 = vfmaq_f32(s1, hi, hi);
    }
    const float r = 1.f / std::sqrt(vaddvq_f32(vaddq_f32(s0, s1)) / static_cast<float>(hd) + eps);
    const float32x4_t rv = vdupq_n_f32(r);
    for (int64_t j = 0; j < hd; j += 8) {
      float32x4_t lo, hi, g0, g1;
      load8(x + j, lo, hi);
      store8(y + j, vmulq_f32(lo, rv), vmulq_f32(hi, rv));  // h
      load8(y + j, lo, hi);
      load8(g + j, g0, g1);
      store8(y + j, vmulq_f32(g0, lo), vmulq_f32(g1, hi));  // y = g·h
    }
    for (int64_t j = 0; j < half; j += 8) {
      float32x4_t l0, l1, h0, h1;
      load8(y + j, l0, l1);
      load8(y + j + half, h0, h1);
      const float32x4_t c0 = vld1q_f32(cs + j), c1 = vld1q_f32(cs + j + 4);
      const float32x4_t n0 = vld1q_f32(sn + j), n1 = vld1q_f32(sn + j + 4);
      store8(o + j, vfmsq_f32(vmulq_f32(l0, c0), h0, n0), vfmsq_f32(vmulq_f32(l1, c1), h1, n1));
      store8(o + j + half, vfmaq_f32(vmulq_f32(h0, c0), l0, n0), vfmaq_f32(vmulq_f32(h1, c1), l1, n1));
    }
    return;
  }
  float sq = 0.f;
  for (int64_t j = 0; j < hd; ++j) sq += static_cast<float>(x[j]) * static_cast<float>(x[j]);
  const float r = 1.f / std::sqrt(sq / static_cast<float>(hd) + eps);
  for (int64_t j = 0; j < hd; ++j) {
    const h16 h = static_cast<h16>(static_cast<float>(x[j]) * r);
    y[j] = static_cast<h16>(static_cast<float>(g[j]) * static_cast<float>(h));
  }
  for (int64_t j = 0; j < half; ++j) {
    const float lo = static_cast<float>(y[j]), hi = static_cast<float>(y[j + half]);
    o[j] = static_cast<h16>(lo * cs[j] - hi * sn[j]);
    o[j + half] = static_cast<h16>(hi * cs[j] + lo * sn[j]);
  }
}

// ptrs: qkv, qg, kg, pos, kc, vc, out. ints: qkv(T, W), qg(hd), kg(hd), pos(1), kc(P, Dk), vc(P, Dk), out(T, Dq).
// floats: eps, theta. q = rope(rmsnorm_heads(qkv[:, :Dq], qg)) -> out; k likewise (kg) -> kc[pos + i];
// v = qkv[:, Dq + Dk:] -> vc[pos + i].
void runQkNormRopeKv(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], wdt = a.ints[1], hd = a.ints[2], cap = a.ints[5], dk = a.ints[6], dq = a.ints[10];
  const int64_t pos = I64(a, 3);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "qknorm_rope_kv: pos + T exceeds the cache";
    return;
  }
  const float eps = static_cast<float>(a.floats[0]), theta = static_cast<float>(a.floats[1]);
  const int64_t half = hd / 2;
  float invFreq[128];
  for (int64_t j = 0; j < half; ++j)
    invFreq[j] = 1.f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(hd));
  const h16 *qkv = H(a, 0), *qg = H(a, 1), *kg = H(a, 2);
  h16* kc = static_cast<h16*>(a.ptrs[4]);
  h16* vc = static_cast<h16*>(a.ptrs[5]);
  h16* out = Hout(a, 6);
  parallelRanges(t, t * wdt * 4, [&](int64_t i0, int64_t i1) {
    float cs[128], sn[128];
    for (int64_t i = i0; i < i1; ++i) {
      const float p = static_cast<float>(pos + i);
      for (int64_t j = 0; j < half; ++j) {
        double sd, cd;
        __sincos(static_cast<double>(p * invFreq[j]), &sd, &cd);
        cs[j] = static_cast<float>(cd);
        sn[j] = static_cast<float>(sd);
      }
      const h16* row = qkv + i * wdt;
      for (int64_t h0 = 0; h0 < dq; h0 += hd) normRopeHead(row + h0, qg, hd, eps, cs, sn, out + i * dq + h0);
      for (int64_t h0 = 0; h0 < dk; h0 += hd) normRopeHead(row + dq + h0, kg, hd, eps, cs, sn, kc + (pos + i) * dk + h0);
      std::memcpy(vc + (pos + i) * dk, row + dq + dk, static_cast<size_t>(dk) * sizeof(h16));
    }
  });
}

// --- flash attention core -------------------------------------------------------------------------------------------

constexpr int64_t kMaxHd = 256, kQMax = 128, kKT = 64;

// Running softmax state of up to kQMax queries: max m, sum l, f32 accumulators [nq][hd].
struct Flash {
  float m[kQMax], l[kQMax];
  float* acc;  // [kQMax][kMaxHd]
  float* s;    // [kQMax][kKT] scores / probabilities of the current tile
  float* vf;   // [kKT][kMaxHd] the current V tile in f32
};

Flash& flashScratch() {
  thread_local std::vector<float> buf(static_cast<size_t>(kQMax * kMaxHd + kQMax * kKT + kKT * kMaxHd));
  thread_local Flash f{{}, {}, buf.data(), buf.data() + kQMax * kMaxHd, buf.data() + kQMax * kMaxHd + kQMax * kKT};
  return f;
}

// s[i][j] = scale · q_i · k_j for QN queries × 4 keys (FMLAL dot products over hd, hd % 8 == 0).
template <int QN>
inline void scores4(const h16* const* q, const h16* const* k, int64_t hd, float scale, float* s, int64_t lds) {
  float32x4_t acc[QN][4];
  for (int i = 0; i < QN; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = vdupq_n_f32(0.f);
  for (int64_t c = 0; c < hd; c += 8) {
    float16x8_t kv[4];
    for (int j = 0; j < 4; ++j) kv[j] = vld1q_f16(k[j] + c);
    for (int i = 0; i < QN; ++i) {
      const float16x8_t qv = vld1q_f16(q[i] + c);
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = vfmlalq_low_f16(acc[i][j], qv, kv[j]);
        acc[i][j] = vfmlalq_high_f16(acc[i][j], qv, kv[j]);
      }
    }
  }
  for (int i = 0; i < QN; ++i) {
    // Horizontal sums of the 4 key accumulators as one vector.
    const float32x4_t s01 = vpaddq_f32(acc[i][0], acc[i][1]), s23 = vpaddq_f32(acc[i][2], acc[i][3]);
    vst1q_f32(s + i * lds, vmulq_f32(vpaddq_f32(s01, s23), vdupq_n_f32(scale)));
  }
}

// Feeds keys [j0, j1) (rows of k / v at stride ld) to nq queries (qp[i], f16, hd contiguous); query i sees only keys
// < lim[i]. The state f must be initialized (flashInit).
void flashKeys(Flash& f, const h16* const* qp, const int64_t* lim, int64_t nq, const h16* k, const h16* v, int64_t ld,
               int64_t j0, int64_t j1, int64_t hd, float scale) {
  int64_t limMax = 0;
  for (int64_t i = 0; i < nq; ++i) limMax = std::max(limMax, lim[i]);
  j1 = std::min(j1, limMax);
  const float ninf = -std::numeric_limits<float>::infinity();
  for (int64_t t0 = j0; t0 < j1; t0 += kKT) {
    const int64_t nk = std::min(kKT, j1 - t0);
    // 1) scores (key tails: the last key repeated, results ignored)
    for (int64_t jb = 0; jb < nk; jb += 4) {
      const h16* kp[4];
      for (int j = 0; j < 4; ++j) kp[j] = k + (t0 + std::min(jb + j, nk - 1)) * ld;
      int64_t i = 0;
      for (; i + 4 <= nq; i += 4) scores4<4>(qp + i, kp, hd, scale, f.s + i * kKT + jb, kKT);
      switch (nq - i) {
        case 3: scores4<3>(qp + i, kp, hd, scale, f.s + i * kKT + jb, kKT); break;
        case 2: scores4<2>(qp + i, kp, hd, scale, f.s + i * kKT + jb, kKT); break;
        case 1: scores4<1>(qp + i, kp, hd, scale, f.s + i * kKT + jb, kKT); break;
        default: break;
      }
    }
    // 2) online softmax per query; p overwrites s (0 for masked keys)
    for (int64_t i = 0; i < nq; ++i) {
      float* s = f.s + i * kKT;
      const int64_t nv = std::clamp<int64_t>(lim[i] - t0, 0, nk);
      if (nv == 0) {
        std::fill(s, s + nk, 0.f);
        continue;
      }
      float bm = ninf;
      for (int64_t j = 0; j < nv; ++j) bm = std::max(bm, s[j]);
      if (bm > f.m[i]) {
        const float corr = std::exp(f.m[i] - bm);  // 0 for the first tile
        f.l[i] *= corr;
        float* acc = f.acc + i * kMaxHd;
        const float32x4_t cv = vdupq_n_f32(corr);
        for (int64_t c = 0; c < hd; c += 4) vst1q_f32(acc + c, vmulq_f32(vld1q_f32(acc + c), cv));
        f.m[i] = bm;
      }
      const float32x4_t mv = vdupq_n_f32(f.m[i]);
      float32x4_t lv = vdupq_n_f32(0.f);
      int64_t j = 0;
      for (; j + 4 <= nv; j += 4) {
        const float32x4_t p = exp4(vsubq_f32(vld1q_f32(s + j), mv));
        vst1q_f32(s + j, p);
        lv = vaddq_f32(lv, p);
      }
      float l = vaddvq_f32(lv);
      for (; j < nv; ++j) l += (s[j] = std::exp(s[j] - f.m[i]));
      for (; j < nk; ++j) s[j] = 0.f;
      f.l[i] += l;
    }
    // 3) V tile in f32
    for (int64_t j = 0; j < nk; ++j) {
      const h16* vr = v + (t0 + j) * ld;
      float* d = f.vf + j * kMaxHd;
      for (int64_t c = 0; c < hd; c += 8) {
        float32x4_t lo, hi;
        load8(vr + c, lo, hi);
        vst1q_f32(d + c, lo);
        vst1q_f32(d + c + 4, hi);
      }
    }
    // 4) acc[i] += Σ_j p[i][j] · v[j]: two queries × 32 columns in registers
    for (int64_t i = 0; i < nq; i += 2) {
      const bool two = i + 1 < nq;
      const float* p0 = f.s + i * kKT;
      const float* p1 = two ? p0 + kKT : p0;
      float* a0 = f.acc + i * kMaxHd;
      float* a1 = two ? a0 + kMaxHd : a0;
      for (int64_t c = 0; c < hd; c += 32) {
        const int64_t cw = std::min<int64_t>(32, hd - c);  // a multiple of 8
        float32x4_t x0[8], x1[8];
        for (int q = 0; q < cw / 4; ++q) x0[q] = vld1q_f32(a0 + c + 4 * q), x1[q] = vld1q_f32(a1 + c + 4 * q);
        if (cw == 32) {
          for (int64_t j = 0; j < nk; ++j) {
            const float* vr = f.vf + j * kMaxHd + c;
            const float32x4_t s0 = vdupq_n_f32(p0[j]), s1 = vdupq_n_f32(p1[j]);
            for (int q = 0; q < 8; ++q) {
              const float32x4_t vv = vld1q_f32(vr + 4 * q);
              x0[q] = vfmaq_f32(x0[q], s0, vv);
              x1[q] = vfmaq_f32(x1[q], s1, vv);
            }
          }
        } else {
          for (int64_t j = 0; j < nk; ++j) {
            const float* vr = f.vf + j * kMaxHd + c;
            const float32x4_t s0 = vdupq_n_f32(p0[j]), s1 = vdupq_n_f32(p1[j]);
            for (int q = 0; q < cw / 4; ++q) {
              const float32x4_t vv = vld1q_f32(vr + 4 * q);
              x0[q] = vfmaq_f32(x0[q], s0, vv);
              x1[q] = vfmaq_f32(x1[q], s1, vv);
            }
          }
        }
        for (int q = 0; q < cw / 4; ++q) {
          vst1q_f32(a0 + c + 4 * q, x0[q]);
          if (two) vst1q_f32(a1 + c + 4 * q, x1[q]);
        }
      }
    }
  }
}

void flashInit(Flash& f, int64_t nq, int64_t hd) {
  for (int64_t i = 0; i < nq; ++i) {
    f.m[i] = -std::numeric_limits<float>::infinity();
    f.l[i] = 0.f;
    std::fill(f.acc + i * kMaxHd, f.acc + i * kMaxHd + hd, 0.f);
  }
}

// out = acc / l, one rounding.
inline void flashStore(const float* acc, float l, int64_t hd, h16* out) {
  const float32x4_t inv = vdupq_n_f32(1.f / l);
  for (int64_t c = 0; c < hd; c += 8) store8(out + c, vmulq_f32(vld1q_f32(acc + c), inv), vmulq_f32(vld1q_f32(acc + c + 4), inv));
}

// ptrs: q, kc, vc, pos, out. ints: q(T, Dq), kc(cap, Dk), vc(cap, Dk), pos(1), out(T, Dq), kv_heads, q_heads.
// Task = (kv head, block of query rows, key split): the block's rows × the group's q heads share every K / V tile.
#if !TARGET_OS_IPHONE
// Prefill attention on the Mac's AMX (T >= kAmxAttnMinT): per (kv head, block of kAmxAttnQB query rows), key tiles of
// kAmxAttnKT: S = scale · Q Kᵀ for the block's queries (all `group` heads of the kv head, f32 copies) with one sgemm,
// the causal mask + the online f32 softmax (exp4), then acc += P V with a second sgemm; one f16 rounding at the end.
// The same math as flashKeys (f32 scores, f32 softmax and accumulation); the NEON dot products cost ~1/4 of the AMX
// rate at long contexts (prefill 20 min: ~28 TFLOP of attention).
constexpr int64_t kAmxAttnMinT = 64, kAmxAttnQB = 64, kAmxAttnKT = 256;  // 8 threads, [128 q x 128] x [256 keys]: 2.2 TF/s (128 keys: 2.0)

void attentionAmx(const h16* q, const h16* kc, const h16* vc, h16* out, int64_t t, int64_t dq, int64_t dk, int64_t pos,
                  int64_t kvHeads, int64_t group, int64_t hd, float scale) {
  using LI = __LAPACK_int;
  const int64_t blocks = (t + kAmxAttnQB - 1) / kAmxAttnQB;
  const float ninf = -std::numeric_limits<float>::infinity();
  runTasks(kvHeads * blocks, [&](int64_t task) {
    // the blocks with the most keys (the last rows) first
    const int64_t h = task % kvHeads, b = blocks - 1 - task / kvHeads;
    const int64_t i0 = b * kAmxAttnQB, nr = std::min(t, i0 + kAmxAttnQB) - i0, nq = nr * group;
    thread_local std::vector<float> buf;
    const size_t need = static_cast<size_t>(nq * hd * 2 + 2 * kAmxAttnKT * hd + nq * kAmxAttnKT + 2 * nq);
    if (buf.size() < need) buf.resize(need);
    float *qf = buf.data(), *acc = qf + nq * hd, *kf = acc + nq * hd, *vf = kf + kAmxAttnKT * hd;
    float *sc = vf + kAmxAttnKT * hd, *m = sc + nq * kAmxAttnKT, *l = m + nq;
    // queries g-major: row g·nr + r is query row i0 + r of q head h·group + g
    for (int64_t g = 0; g < group; ++g)
      for (int64_t r = 0; r < nr; ++r) cvtF32(q + (i0 + r) * dq + (h * group + g) * hd, qf + (g * nr + r) * hd, hd);
    std::fill(m, m + nq, ninf);
    std::fill(l, l + nq, 0.f);
    std::fill(acc, acc + nq * hd, 0.f);
    const int64_t nkeys = pos + i0 + nr;  // keys visible to the block's last row
    for (int64_t t0 = 0; t0 < nkeys; t0 += kAmxAttnKT) {
      const int64_t nk = std::min(kAmxAttnKT, nkeys - t0);
      for (int64_t j = 0; j < nk; ++j) {
        cvtF32(kc + (t0 + j) * dk + h * hd, kf + j * hd, hd);
        cvtF32(vc + (t0 + j) * dk + h * hd, vf + j * hd, hd);
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, LI(nq), LI(nk), LI(hd), scale, qf, LI(hd), kf, LI(hd), 0.f,
                  sc, LI(nk));
      for (int64_t qi = 0; qi < nq; ++qi) {
        float* row = sc + qi * nk;
        const int64_t nv = std::clamp<int64_t>(pos + i0 + qi % nr + 1 - t0, 0, nk);
        if (nv == 0) {
          std::fill(row, row + nk, 0.f);
          continue;
        }
        float bm = ninf;
        for (int64_t j = 0; j < nv; ++j) bm = std::max(bm, row[j]);
        if (bm > m[qi]) {
          const float corr = std::exp(m[qi] - bm);  // 0 for the first tile
          l[qi] *= corr;
          float* ar = acc + qi * hd;
          const float32x4_t cv = vdupq_n_f32(corr);
          for (int64_t c = 0; c < hd; c += 4) vst1q_f32(ar + c, vmulq_f32(vld1q_f32(ar + c), cv));
          m[qi] = bm;
        }
        const float32x4_t mv = vdupq_n_f32(m[qi]);
        float32x4_t lv = vdupq_n_f32(0.f);
        int64_t j = 0;
        for (; j + 4 <= nv; j += 4) {
          const float32x4_t pv = exp4(vsubq_f32(vld1q_f32(row + j), mv));
          vst1q_f32(row + j, pv);
          lv = vaddq_f32(lv, pv);
        }
        float ls = vaddvq_f32(lv);
        for (; j < nv; ++j) ls += (row[j] = std::exp(row[j] - m[qi]));
        for (; j < nk; ++j) row[j] = 0.f;
        l[qi] += ls;
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, LI(nq), LI(hd), LI(nk), 1.f, sc, LI(nk), vf, LI(hd), 1.f,
                  acc, LI(hd));
    }
    for (int64_t qi = 0; qi < nq; ++qi)
      flashStore(acc + qi * hd, l[qi], hd, out + (i0 + qi % nr) * dq + (h * group + qi / nr) * hd);
  });
}
#endif

#if !TARGET_OS_IPHONE
// Decode (T = 1) over long contexts on the Mac: tasks split the keys only and walk them in tiles of kKT, feeding every
// kv head per tile, so each task reads whole cache rows (all heads, dk wide) in order. One task per kv head (the
// generic path) read 256-byte head slices 2 KB apart while the other heads' tasks read the rest of those rows at other
// times. Per-head online-softmax states, merged across the key splits as in the generic path.
constexpr int64_t kDecodeInterleaveMinKeys = 1024;

void decodeInterleaved(const h16* q, const h16* kc, const h16* vc, h16* out, int64_t dk, int64_t pos, int64_t kvHeads,
                       int64_t group, int64_t hd, float scale) {
  const int64_t keys = pos + 1;
  const int64_t splits = std::clamp<int64_t>(2 * ThreadPool::global().threads(), 1, keys / 256);
  const int64_t rec = hd + 2;
  float* parts = scratch<float>(static_cast<size_t>(splits * kvHeads * group * rec), 2);
  runTasks(splits, [&](int64_t sp) {
    const int64_t ks0 = keys * sp / splits, ks1 = keys * (sp + 1) / splits;
    thread_local std::vector<float> buf;
    thread_local std::vector<Flash> fs;
    const size_t need = static_cast<size_t>(kvHeads * group * kMaxHd + kQMax * kKT + kKT * kMaxHd);
    if (buf.size() < need) buf.resize(need);
    if (fs.size() < static_cast<size_t>(kvHeads)) fs.resize(static_cast<size_t>(kvHeads));
    float* shared = buf.data() + kvHeads * group * kMaxHd;  // scores and the f32 V tile: used within one call
    for (int64_t h = 0; h < kvHeads; ++h) {
      fs[h].acc = buf.data() + h * group * kMaxHd;
      fs[h].s = shared;
      fs[h].vf = shared + kQMax * kKT;
      flashInit(fs[h], group, hd);
    }
    const h16* qp[kQMax];
    int64_t lim[kQMax];
    for (int64_t t0 = ks0; t0 < ks1; t0 += kKT) {
      const int64_t t1 = std::min(ks1, t0 + kKT);
      for (int64_t h = 0; h < kvHeads; ++h) {
        for (int64_t g = 0; g < group; ++g) qp[g] = q + (h * group + g) * hd, lim[g] = keys;
        flashKeys(fs[h], qp, lim, group, kc + h * hd, vc + h * hd, dk, t0, t1, hd, scale);
      }
    }
    for (int64_t h = 0; h < kvHeads; ++h)
      for (int64_t g = 0; g < group; ++g) {
        float* r = parts + ((sp * kvHeads + h) * group + g) * rec;
        r[0] = fs[h].m[g];
        r[1] = fs[h].l[g];
        std::memcpy(r + 2, fs[h].acc + g * kMaxHd, static_cast<size_t>(hd) * sizeof(float));
      }
  });
  float acc[kMaxHd];
  for (int64_t h = 0; h < kvHeads; ++h)
    for (int64_t g = 0; g < group; ++g) {
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t sp = 0; sp < splits; ++sp) m = std::max(m, parts[((sp * kvHeads + h) * group + g) * rec]);
      float l = 0.f;
      std::fill(acc, acc + hd, 0.f);
      for (int64_t sp = 0; sp < splits; ++sp) {
        const float* r = parts + ((sp * kvHeads + h) * group + g) * rec;
        if (r[1] == 0.f) continue;
        const float c = std::exp(r[0] - m);
        l += r[1] * c;
        const float32x4_t cv = vdupq_n_f32(c);
        for (int64_t d = 0; d < hd; d += 4) vst1q_f32(acc + d, vfmaq_f32(vld1q_f32(acc + d), cv, vld1q_f32(r + 2 + d)));
      }
      flashStore(acc, l, hd, out + (h * group + g) * hd);
    }
}
#endif

void runAttentionGqaOpt(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], dq = a.ints[1], cap = a.ints[2], dk = a.ints[3];
  const int64_t kvHeads = a.ints[9], qHeads = a.ints[10], hd = dq / qHeads, group = qHeads / kvHeads;
  const int64_t pos = I64(a, 3);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "attention_gqa: pos + T exceeds the cache";
    return;
  }
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const h16 *q = H(a, 0), *kc = H(a, 1), *vc = H(a, 2);
  h16* out = Hout(a, 4);
#if !TARGET_OS_IPHONE
  if (t >= kAmxAttnMinT && hd % 8 == 0 && hd <= kMaxHd && !kernels::cpu::sme2::available())
    return attentionAmx(q, kc, vc, out, t, dq, dk, pos, kvHeads, group, hd, scale);
  if (t == 1 && pos + 1 >= kDecodeInterleaveMinKeys && group <= kQMax && hd % 8 == 0 && hd <= kMaxHd)
    return decodeInterleaved(q, kc, vc, out, dk, pos, kvHeads, group, hd, scale);
#endif
  const int64_t qb = std::max<int64_t>(1, std::min<int64_t>(16, kQMax / group));
  const int64_t blocks = (t + qb - 1) / qb, base = kvHeads * blocks;
  const int64_t threads = ThreadPool::global().threads();
  const int64_t keys = pos + t;
  int64_t splits = 1;
  if (base < 2 * threads) splits = std::clamp<int64_t>((2 * threads + base - 1) / base, 1, std::max<int64_t>(1, keys / 256));
  const int64_t nqMax = qb * group, rec = hd + 2;
  // Split partials: [task][query] of (m, l, acc[hd]).
  float* parts = splits > 1 ? scratch<float>(static_cast<size_t>(base * splits * nqMax * rec), 2) : nullptr;
  runTasks(base * splits, [&](int64_t task) {
    const int64_t sp = task % splits, bt = task / splits, h = bt % kvHeads, b = bt / kvHeads;
    const int64_t i0 = b * qb, i1 = std::min(t, i0 + qb), nq = (i1 - i0) * group;
    const h16* qp[kQMax];
    int64_t lim[kQMax];
    for (int64_t i = i0; i < i1; ++i)
      for (int64_t g = 0; g < group; ++g) {
        qp[(i - i0) * group + g] = q + i * dq + (h * group + g) * hd;
        lim[(i - i0) * group + g] = pos + i + 1;
      }
    const int64_t n = pos + i1, ks0 = n * sp / splits, ks1 = n * (sp + 1) / splits;
    Flash& f = flashScratch();
    flashInit(f, nq, hd);
    flashKeys(f, qp, lim, nq, kc + h * hd, vc + h * hd, dk, ks0, ks1, hd, scale);
    for (int64_t qi = 0; qi < nq; ++qi) {
      if (splits == 1) {
        const int64_t i = i0 + qi / group, g = qi % group;
        flashStore(f.acc + qi * kMaxHd, f.l[qi], hd, out + i * dq + (h * group + g) * hd);
      } else {
        float* r = parts + (task * nqMax + qi) * rec;
        r[0] = f.m[qi];
        r[1] = f.l[qi];
        std::memcpy(r + 2, f.acc + qi * kMaxHd, static_cast<size_t>(hd) * sizeof(float));
      }
    }
  });
  if (splits == 1) return;
  // Merge the splits: m = max m_s, l = Σ l_s·e^(m_s - m), acc = Σ acc_s·e^(m_s - m).
  runTasks(base > 1 && t * dq > 4096 ? base : 1, [&](int64_t bt0) {
    const int64_t bEnd = base > 1 && t * dq > 4096 ? bt0 + 1 : base;
    float acc[kMaxHd];
    for (int64_t bt = bt0; bt < bEnd; ++bt) {
      const int64_t h = bt % kvHeads, b = bt / kvHeads;
      const int64_t i0 = b * qb, i1 = std::min(t, i0 + qb), nq = (i1 - i0) * group;
      for (int64_t qi = 0; qi < nq; ++qi) {
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t sp = 0; sp < splits; ++sp) m = std::max(m, parts[((bt * splits + sp) * nqMax + qi) * rec]);
        float l = 0.f;
        std::fill(acc, acc + hd, 0.f);
        for (int64_t sp = 0; sp < splits; ++sp) {
          const float* r = parts + ((bt * splits + sp) * nqMax + qi) * rec;
          if (r[1] == 0.f) continue;  // no key of this split is visible to the query
          const float c = std::exp(r[0] - m);
          l += r[1] * c;
          const float32x4_t cv = vdupq_n_f32(c);
          for (int64_t d = 0; d < hd; d += 4) vst1q_f32(acc + d, vfmaq_f32(vld1q_f32(acc + d), cv, vld1q_f32(r + 2 + d)));
        }
        const int64_t i = i0 + qi / group, g = qi % group;
        flashStore(acc, l, hd, out + i * dq + (h * group + g) * hd);
      }
    }
  });
}

// ptrs: qkv, out. ints: qkv(S, 3D), out(S, D), heads, window. q = qkv[:, :D], k = qkv[:, D:2D], v = qkv[:, 2D:];
// non-causal attention inside blocks of `window` rows. Task = (window, head, block of <= 128 query rows).
void runAttentionWindowQkv(ExecCtx&, const KernelArgs& a) {
  const int64_t s = a.ints[0], w3 = a.ints[1], d = a.ints[3], heads = a.ints[4], window = a.ints[5], hd = d / heads;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const h16* qkv = H(a, 0);
  h16* out = Hout(a, 1);
  const int64_t wins = (s + window - 1) / window, qbs = (window + kQMax - 1) / kQMax;
  auto task = [&](int64_t tk) {
    const int64_t qbi = tk % qbs, h = (tk / qbs) % heads, wi = tk / (qbs * heads);
    const int64_t w0 = wi * window, w1 = std::min(s, w0 + window);
    const int64_t i0 = w0 + qbi * kQMax, i1 = std::min(w1, i0 + kQMax);
    if (i0 >= i1) return;
    const int64_t nq = i1 - i0;
    const h16* qp[kQMax];
    int64_t lim[kQMax];
    for (int64_t i = 0; i < nq; ++i) qp[i] = qkv + (i0 + i) * w3 + h * hd, lim[i] = w1;
    Flash& f = flashScratch();
    flashInit(f, nq, hd);
    flashKeys(f, qp, lim, nq, qkv + d + h * hd, qkv + 2 * d + h * hd, w3, w0, w1, hd, scale);
    for (int64_t i = 0; i < nq; ++i) flashStore(f.acc + i * kMaxHd, f.l[i], hd, out + (i0 + i) * d + h * hd);
  };
  const int64_t n = wins * heads * qbs;
  if (s * window * d < (int64_t{1} << 16)) {
    for (int64_t i = 0; i < n; ++i) task(i);
  } else {
    ThreadPool::global().parallelFor(n, task);
  }
}

// --- LM head --------------------------------------------------------------------------------------------------------

// out[r0:r1] = table[r0:r1] · x: R table rows per pass (R streams per core), FMLAL dot products.
template <int R>
void lmHeadRows(const h16* x, const h16* table, int64_t d, int64_t r0, int64_t r1, float* out) {
  const int64_t d8 = d / 8 * 8;
  int64_t r = r0;
  for (; r + R <= r1; r += R) {
    float32x4_t acc[R][2];
    for (auto& p : acc) p[0] = p[1] = vdupq_n_f32(0.f);
    const h16* tr = table + r * d;
    for (int64_t c = 0; c < d8; c += 8) {
      const float16x8_t xv = vld1q_f16(x + c);
      for (int j = 0; j < R; ++j) {
        const float16x8_t tv = vld1q_f16(tr + j * d + c);
        acc[j][0] = vfmlalq_low_f16(acc[j][0], tv, xv);
        acc[j][1] = vfmlalq_high_f16(acc[j][1], tv, xv);
      }
    }
    for (int j = 0; j < R; ++j) {
      float sum = vaddvq_f32(vaddq_f32(acc[j][0], acc[j][1]));
      for (int64_t c = d8; c < d; ++c) sum += static_cast<float>(x[c]) * static_cast<float>(tr[j * d + c]);
      out[r + j] = sum;
    }
  }
  if (R > 1 && r < r1) lmHeadRows<1>(x, table, d, r, r1, out);
}

// Table rows per pass. Mac, 2026-11-07: 2 rows 4.94 vs 4 rows 5.62 ms in-model (decode C = 480; microbench of the
// same dot-product layout over decode-sized weights: 2 / 4 / 1 rows 116.7 / 110 / 111 GB/s): 2 streams per core
// stream best. iPhone, 2026-11-07: 2 rows 6.8 vs 4 rows 7.0 ms.
constexpr int kLmHeadRows = 2;

// ptrs: x, table, out(f32). ints: x(T, D), table(V, D), out(1, V). out[v] = x[T-1] · table[v]: kLmHeadRows table rows
// at a time, FMLAL dot products, every core streaming a contiguous range of rows. The per-row sum order does not
// depend on kLmHeadRows (bit-identical results).
void runLmHeadLastOpt(ExecCtx&, const KernelArgs& a) {
  const int64_t t = a.ints[0], d = a.ints[1], v = a.ints[2];
  const h16* x = H(a, 0) + (t - 1) * d;
  const h16* table = H(a, 1);
  float* out = static_cast<float*>(a.ptrs[2]);
  auto rows = [&](int64_t r0, int64_t r1) { lmHeadRows<kLmHeadRows>(x, table, d, r0, r1, out); };
  if (v * d < (int64_t{1} << 18)) return rows(0, v);
  const int64_t tasks = std::min<int64_t>(4 * ThreadPool::global().threads(), (v + 3) / 4);
  const int64_t per = ((v + tasks - 1) / tasks + 3) / 4 * 4;
  ThreadPool::global().parallelFor((v + per - 1) / per, [&](int64_t i) { rows(i * per, std::min(v, (i + 1) * per)); });
}

// --- audio CNN front end ------------------------------------------------------------------------------------------

// im2col of a 3×3, stride 2, pad 1 conv over NHWC x[h, w, c]: rows (ho, wo), columns (kt, kf, ci); zeros outside.
void im2col3x3s2(const h16* x, int64_t h, int64_t w, int64_t c, h16* cols, int64_t ho, int64_t wo) {
  const size_t seg = static_cast<size_t>(c) * sizeof(h16);
  parallelRanges(ho * wo, ho * wo * 9 * c / 8, [&](int64_t p0, int64_t p1) {
    for (int64_t p = p0; p < p1; ++p) {
      const int64_t oy = p / wo, ox = p % wo;
      h16* row = cols + p * 9 * c;
      for (int64_t kt = 0; kt < 3; ++kt)
        for (int64_t kf = 0; kf < 3; ++kf) {
          const int64_t iy = 2 * oy - 1 + kt, ix = 2 * ox - 1 + kf;
          h16* dst = row + (kt * 3 + kf) * c;
          if (iy < 0 || iy >= h || ix < 0 || ix >= w)
            std::memset(dst, 0, seg);
          else
            std::memcpy(dst, x + (iy * w + ix) * c, seg);
        }
    }
  });
}

// [Co, K] -> [K, Co] (conv weights [Co, kt, kf, Ci] as GEMM B operands), 8 × 8 blocks on the pool.
void transposeInto(const h16* w, int64_t co, int64_t k, h16* t) {
  parallelRanges((k + 7) / 8, co * k, [&](int64_t b0, int64_t b1) {
    for (int64_t jb = b0 * 8; jb < std::min(k, b1 * 8); jb += 8)
      for (int64_t o0 = 0; o0 < co; o0 += 8)
        for (int64_t j = jb; j < std::min(k, jb + 8); ++j)
          for (int64_t o = o0; o < std::min(co, o0 + 8); ++o) t[j * co + o] = w[o * k + j];
  });
}

#if !TARGET_OS_IPHONE
// The Mac audio CNN: kAmxCnnBatch chunks per conv-layer GEMM on BNNS (f16 x f16 -> f32 on the AMX, the bias + GELU
// epilogue rounding each layer to f16, the contract), conv weights transposed once per call, conv_out over every
// chunk at once. The generic per-chunk path converted weights and im2col matrices f16 -> f32 for every chunk and ran
// conv_out with M = 13 on the NEON tiles (enc 30 s audio_cnn 160 -> 82 ms). conv1 (K = 9) stays on the NEON tiles.
constexpr int64_t kAmxCnnBatch = 2;  // enc 30 s audio_cnn: 1 chunk 83.1 ms, 2 81.9, 4 86.1, 8 152 (2026-11-05)

void gemmBnnsOr(const h16* x, const h16* w, const Epi& e, h16* out, int64_t m, int64_t k, int64_t n) {
  if (!gemmBnns(x, w, e, out, m, k, n)) gemmTiles(x, w, e, out, m, k, n);
}

void audioCnnAmx(const float* feats, int64_t n, const h16* w1, const h16* b1, const h16* w2, const h16* b2,
                 const h16* w3, const h16* b3, const h16* wOut, h16* out) {
  constexpr int64_t kF = 128, kT = 100, kC = 480;
  constexpr int64_t kH1 = 50, kW1 = 64, kH2 = 25, kW2 = 32, kH3 = 13, kW3 = 16, kOut = 1024;
  constexpr int64_t kP1 = kH1 * kW1, kP2 = kH2 * kW2, kP3 = kH3 * kW3;
  std::vector<h16> w1t(9 * kC), w2t(9 * kC * kC), w3t(9 * kC * kC);
  transposeInto(w1, kC, 9, w1t.data());
  transposeInto(w2, kC, 9 * kC, w2t.data());
  transposeInto(w3, kC, 9 * kC, w3t.data());
  const int64_t bmax = std::min(n, kAmxCnnBatch);
  std::vector<h16> x0(kT * kF), a1(bmax * kP1 * kC), a2(bmax * kP2 * kC), a3(n * kP3 * kC);
  std::vector<h16> cols(static_cast<size_t>(bmax * std::max(kP1 * 9, kP2 * 9 * kC)));
  Epi conv;
  conv.gelu = true;
  for (int64_t c0 = 0; c0 < n; c0 += kAmxCnnBatch) {
    const int64_t nb = std::min(kAmxCnnBatch, n - c0);
    for (int64_t i = 0; i < nb; ++i) {
      const float* f = feats + (c0 + i) * kF * kT;
      for (int64_t t = 0; t < kT; ++t)
        for (int64_t q = 0; q < kF; ++q) x0[t * kF + q] = static_cast<h16>(f[q * kT + t]);
      im2col3x3s2(x0.data(), kT, kF, 1, cols.data() + i * kP1 * 9, kH1, kW1);
    }
    conv.bias = b1;
    gemmTiles(cols.data(), w1t.data(), conv, a1.data(), nb * kP1, 9, kC);  // K = 9: the NEON tiles beat BNNS (86 vs 105 ms)
    for (int64_t i = 0; i < nb; ++i)
      im2col3x3s2(a1.data() + i * kP1 * kC, kH1, kW1, kC, cols.data() + i * kP2 * 9 * kC, kH2, kW2);
    conv.bias = b2;
    gemmBnnsOr(cols.data(), w2t.data(), conv, a2.data(), nb * kP2, 9 * kC, kC);
    for (int64_t i = 0; i < nb; ++i)
      im2col3x3s2(a2.data() + i * kP2 * kC, kH2, kW2, kC, cols.data() + i * kP3 * 9 * kC, kH3, kW3);
    conv.bias = b3;
    gemmBnnsOr(cols.data(), w3t.data(), conv, a3.data() + c0 * kP3 * kC, nb * kP3, 9 * kC, kC);
  }
  // conv_out: chunk c's [H3, W3 * C] = its a3 rows read as [13, 7680] (the f16 a3 layout [H3, W3, C]). Always the
  // BNNS engine (not gemmOpt's M-dependent choice), so a chunk's output does not depend on how many chunks the call
  // has (bit for bit; test_audio_cnn_opt)
  gemmBnnsOr(a3.data(), wOut, Epi{}, out, n * kH3, kW3 * kC, kOut);
}
#endif

// ptrs: features, w1, b1, w2, b2, w3, b3, w_out, out. ints: as audio_cnn_f16_cpu. Per 1 s chunk: im2col + the
// optimized GEMM with the bias + GELU epilogue per conv layer (each output rounded to f16), then conv_out.
void runAudioCnnOpt(ExecCtx&, const KernelArgs& a) {
  constexpr int64_t kF = 128, kT = 100, kC = 480;
  constexpr int64_t kH1 = 50, kW1 = 64, kH2 = 25, kW2 = 32, kH3 = 13, kW3 = 16, kOut = 1024;
  const int64_t n = a.ints[0];
  const float* feats = static_cast<const float*>(a.ptrs[0]);
  const h16 *b1 = H(a, 2), *b2 = H(a, 4), *b3 = H(a, 6), *wOut = H(a, 7);
  h16* out = Hout(a, 8);
#if !TARGET_OS_IPHONE
  if (!kernels::cpu::sme2::available())
    return audioCnnAmx(feats, n, H(a, 1), b1, H(a, 3), b2, H(a, 5), b3, wOut, out);
#endif
  std::vector<h16> w1t(9 * kC), w2t(9 * kC * kC), w3t(9 * kC * kC);
  transposeInto(H(a, 1), kC, 9, w1t.data());
  transposeInto(H(a, 3), kC, 9 * kC, w2t.data());
  transposeInto(H(a, 5), kC, 9 * kC, w3t.data());
  std::vector<h16> x0(kT * kF), cols(kH2 * kW2 * 9 * kC), a1(kH1 * kW1 * kC), a2(kH2 * kW2 * kC), a3(kH3 * kW3 * kC);
  Epi conv;
  conv.gelu = true;
  for (int64_t ch = 0; ch < n; ++ch) {
    const float* f = feats + ch * kF * kT;
    for (int64_t t = 0; t < kT; ++t)
      for (int64_t q = 0; q < kF; ++q) x0[t * kF + q] = static_cast<h16>(f[q * kT + t]);
    im2col3x3s2(x0.data(), kT, kF, 1, cols.data(), kH1, kW1);
    conv.bias = b1;
    gemmOpt(cols.data(), w1t.data(), conv, a1.data(), kH1 * kW1, 9, kC);
    im2col3x3s2(a1.data(), kH1, kW1, kC, cols.data(), kH2, kW2);
    conv.bias = b2;
    gemmOpt(cols.data(), w2t.data(), conv, a2.data(), kH2 * kW2, 9 * kC, kC);
    im2col3x3s2(a2.data(), kH2, kW2, kC, cols.data(), kH3, kW3);
    conv.bias = b3;
    gemmOpt(cols.data(), w3t.data(), conv, a3.data(), kH3 * kW3, 9 * kC, kC);
    gemmOpt(a3.data(), wOut, Epi{}, out + ch * kH3 * kOut, kH3, kW3 * kC, kOut);
  }
}

}  // namespace

void registerCpuF16FusedKernels() {
  registerKernel({"rmsnorm_gemm_opt_f16_cpu", Device::Cpu, nullptr, runRmsGemmOpt<false>, nullptr});
  registerKernel({"rmsnorm_gemm_swiglu_opt_f16_cpu", Device::Cpu, nullptr, runRmsGemmOpt<true>, nullptr});
  registerKernel({"gemm_opt_f16_cpu", Device::Cpu, nullptr, runGemmOpt<false, false, false, false, false>, nullptr});
  registerKernel({"gemm_bias_opt_f16_cpu", Device::Cpu, nullptr, runGemmOpt<true, false, false, false, false>, nullptr});
  registerKernel(
      {"gemm_bias_gelu_opt_f16_cpu", Device::Cpu, nullptr, runGemmOpt<true, false, true, false, false>, nullptr});
  registerKernel({"gemm_add_opt_f16_cpu", Device::Cpu, nullptr, runGemmOpt<false, true, false, false, false>, nullptr});
  registerKernel(
      {"gemm_bias_add_opt_f16_cpu", Device::Cpu, nullptr, runGemmOpt<true, true, false, false, false>, nullptr});
  registerKernel(
      {"gemm_bias_add_clamp_opt_f16_cpu", Device::Cpu, nullptr, runGemmOpt<true, true, false, true, false>, nullptr});
  registerKernel(
      {"gemm_swiglu_opt_f16_cpu", Device::Cpu, nullptr, runGemmOpt<false, false, false, false, true>, nullptr});
  registerKernel({"qknorm_rope_kv_opt_f16_cpu", Device::Cpu, nullptr, runQkNormRopeKv, nullptr});
  registerKernel({"attention_gqa_opt_f16_cpu", Device::Cpu, nullptr, runAttentionGqaOpt, nullptr});
  registerKernel({"attention_window_qkv_opt_f16_cpu", Device::Cpu, nullptr, runAttentionWindowQkv, nullptr});
  registerKernel({"audio_cnn_opt_f16_cpu", Device::Cpu, nullptr, runAudioCnnOpt, nullptr});
  registerKernel({"lm_head_last_opt_f16_cpu", Device::Cpu, nullptr, runLmHeadLastOpt, nullptr});
}

}  // namespace lk::rt

#pragma clang attribute pop
