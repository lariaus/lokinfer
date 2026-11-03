// Shared SME2 GEMM engine (see sme2_gemm.h). One pass of `block` computes a 32 (A rows) × 32 (B columns) block of C
// in the four 16×16 f32 ZA tiles: per k, one 2-vector load of the packed A column pair, one of the B row pair, and
// four FMOPAs (tiles 0..3 = rows 0-15 × cols 0-15, rows 0-15 × cols 16-31, rows 16-31 × cols 0-15, rows 16-31 ×
// cols 16-31). The epilogue reads the tiles back by horizontal slices (one C row each) and applies accumulate / bias /
// residual / activation in streaming SVE before a predicated store.
//
// Loop order: A panels in chunks of <= 4 MiB (they stay in L2 — SME loads come from L2 on Apple cores), B 32-column
// blocks outside the chunk's panels so each B block (weights) streams once per chunk.
#include "sme2_gemm.h"

#include <arm_neon.h>

#include <algorithm>
#include <vector>

#if defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>
#endif

namespace lk::kernels::cpu::sme2 {

namespace {

constexpr int64_t kChunkBytes = int64_t{4} << 20;

inline const float* bAddr(const BOperand& b, int64_t kk, int64_t j) {
  if (b.rowOff) return b.b + (j / 32) * b.panel + b.rowOff[kk] + j % 32;
  return b.b + (j / 32) * b.panel + (kk / b.taps) * b.ldb + (kk % b.taps) * b.dil + j % 32;
}

inline float actS(float v, const Epilogue& e) {
  switch (e.act) {
    case Act::None: return v;
    case Act::Relu: return v > 0.f ? v : 0.f;
    case Act::Relu6: return std::min(std::max(v, 0.f), 6.f);
    case Act::Leaky: return v >= 0.f ? v : v * e.leaky;
  }
  return v;
}

// Scalar reference (CPUs without SME2: only reached by tests, callers check available()).
void gemmScalar(const float* ap, int64_t m, int64_t k, const BOperand& b, int64_t n, float* c, int64_t ldc,
                const Epilogue& e) {
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      const float* a = ap + (i / 32) * 32 * k + i % 32;
      float s = 0.f;
      for (int64_t kk = 0; kk < k; ++kk) s += a[kk * 32] * *bAddr(b, kk, j);
      float* o = c + i * ldc + j;
      if (e.accumulate) s += *o;
      if (e.bias) s += e.bias[e.biasPerRow ? i : j];
      if (e.res) s += e.res[i * e.ldres + j];
      *o = actS(s, e) * (e.colMask ? e.colMask[j] : 1.f);
    }
}

void packGatherScalar(const float* const* ptrs, int64_t rows, int64_t segs, int64_t seglen, float* dst) {
  const int64_t k = segs * seglen;
  for (int64_t p = 0; p * 32 < rows; ++p)
    for (int64_t kk = 0; kk < k; ++kk)
      for (int64_t j = 0; j < 32; ++j) {
        const int64_t r = p * 32 + j;
        dst[(p * k + kk) * 32 + j] = r < rows ? ptrs[r * segs + kk / seglen][kk % seglen] : 0.f;
      }
}

}  // namespace

void packPanelsKMajor(const float* src, int64_t ld, int64_t k, int64_t cols, float* dst) {
  for (int64_t p = 0; p * 32 < cols; ++p) {
    const int64_t nc = std::min<int64_t>(32, cols - p * 32);
    float* d = dst + p * 32 * k;
    for (int64_t kk = 0; kk < k; ++kk) {
      const float* s = src + kk * ld + p * 32;
      int64_t j = 0;
      for (; j < nc; ++j) d[kk * 32 + j] = s[j];
      for (; j < 32; ++j) d[kk * 32 + j] = 0.f;
    }
  }
}

#if defined(__ARM_FEATURE_SME2)

namespace {

// ZA starts from: zero, or C (accumulate), or the residual; a per-column bias is added as one outer product
// ones ⊗ bias (4 FMOPAs), a per-row bias as bias ⊗ ones. Rows past nr / columns past nt are never stored.
inline void initZa(int64_t nr, svbool_t pb0, svbool_t pb1, const float* c, int64_t ldc, const float* res,
                   int64_t ldres, bool accumulate, const float* biasC, const float* biasR) __arm_streaming
    __arm_inout("za") {
  svzero_za();
  const float* src = accumulate ? c : res;
  const int64_t ld = accumulate ? ldc : ldres;
  if (src) {
    for (int64_t i = 0; i < nr && i < 16; ++i) {
      svld1_hor_za32(0, static_cast<uint32_t>(i), pb0, src + i * ld);
      svld1_hor_za32(1, static_cast<uint32_t>(i), pb1, src + i * ld + 16);
    }
    for (int64_t i = 16; i < nr; ++i) {
      svld1_hor_za32(2, static_cast<uint32_t>(i - 16), pb0, src + i * ld);
      svld1_hor_za32(3, static_cast<uint32_t>(i - 16), pb1, src + i * ld + 16);
    }
  }
  const svbool_t all = svptrue_b32();
  const svfloat32_t ones = svdup_n_f32(1.f);
  if (biasC) {
    const svfloat32_t b0 = svld1_f32(pb0, biasC), b1 = svld1_f32(pb1, biasC + 16);
    svmopa_za32_f32_m(0, all, pb0, ones, b0);
    svmopa_za32_f32_m(1, all, pb1, ones, b1);
    svmopa_za32_f32_m(2, all, pb0, ones, b0);
    svmopa_za32_f32_m(3, all, pb1, ones, b1);
  }
  if (biasR) {
    const svbool_t pr0 = svwhilelt_b32_s64(0, nr), pr1 = svwhilelt_b32_s64(16, nr);
    const svfloat32_t r0 = svld1_f32(pr0, biasR), r1 = svld1_f32(pr1, biasR + 16);
    svmopa_za32_f32_m(0, all, all, r0, ones);
    svmopa_za32_f32_m(1, all, all, r0, ones);
    svmopa_za32_f32_m(2, all, all, r1, ones);
    svmopa_za32_f32_m(3, all, all, r1, ones);
  }
}

// C[0:nr, 0:nt] of one 32×32 block. a: the A panel ([k][32]); b: B at column j0 (multiple of 32); c: C at (i0, j0);
// biasC: column bias at j0 (or null); biasR: row bias at i0 (or null); res: residual at (i0, j0) (or null, or folded
// into the post-pass).
// Mode 0: B rows at kk · ldb; 1: (kk / taps) · ldb + (kk % taps) · dil; 2: rowOff[kk].
template <int Mode>
inline void block(const float* a, int64_t k, const BOperand& bo, const float* b, int64_t nt, int64_t nr, float* c,
                  int64_t ldc, const float* biasC, const float* biasR, const float* res, int64_t ldres,
                  bool accumulate, const Epilogue* epi) __arm_streaming __arm_inout("za") {
  const svbool_t all = svptrue_b32();
  const svbool_t pb0 = svwhilelt_b32_s64(0, nt), pb1 = svwhilelt_b32_s64(16, nt);
  initZa(nr, pb0, pb1, c, ldc, res, ldres, accumulate, biasC, biasR);
  constexpr bool Taps1 = Mode == 0;
  const int64_t outer = Mode == 1 ? k / bo.taps : k, taps = Mode == 1 ? bo.taps : 1;
  if (Mode == 2) {
    const int64_t* off = bo.rowOff;
    if (nt == 32) {
      const svcount_t pc = svptrue_c32();
      int64_t kk = 0;
      for (; kk + 2 <= k; kk += 2, a += 64) {
        const svfloat32x2_t av = svld1_f32_x2(pc, a), bv = svld1_f32_x2(pc, b + off[kk]);
        const svfloat32x2_t av2 = svld1_f32_x2(pc, a + 32), bv2 = svld1_f32_x2(pc, b + off[kk + 1]);
        svmopa_za32_f32_m(0, all, all, svget2(av, 0), svget2(bv, 0));
        svmopa_za32_f32_m(1, all, all, svget2(av, 0), svget2(bv, 1));
        svmopa_za32_f32_m(2, all, all, svget2(av, 1), svget2(bv, 0));
        svmopa_za32_f32_m(3, all, all, svget2(av, 1), svget2(bv, 1));
        svmopa_za32_f32_m(0, all, all, svget2(av2, 0), svget2(bv2, 0));
        svmopa_za32_f32_m(1, all, all, svget2(av2, 0), svget2(bv2, 1));
        svmopa_za32_f32_m(2, all, all, svget2(av2, 1), svget2(bv2, 0));
        svmopa_za32_f32_m(3, all, all, svget2(av2, 1), svget2(bv2, 1));
      }
      if (kk < k) {
        const svfloat32x2_t av = svld1_f32_x2(pc, a), bv = svld1_f32_x2(pc, b + off[kk]);
        svmopa_za32_f32_m(0, all, all, svget2(av, 0), svget2(bv, 0));
        svmopa_za32_f32_m(1, all, all, svget2(av, 0), svget2(bv, 1));
        svmopa_za32_f32_m(2, all, all, svget2(av, 1), svget2(bv, 0));
        svmopa_za32_f32_m(3, all, all, svget2(av, 1), svget2(bv, 1));
      }
    } else {
      for (int64_t kk = 0; kk < k; ++kk, a += 32) {
        const float* bp = b + off[kk];
        const svfloat32_t a0 = svld1_f32(all, a), a1 = svld1_f32(all, a + 16);
        const svfloat32_t b0 = svld1_f32(pb0, bp), b1 = svld1_f32(pb1, bp + 16);
        svmopa_za32_f32_m(0, all, pb0, a0, b0);
        svmopa_za32_f32_m(1, all, pb1, a0, b1);
        svmopa_za32_f32_m(2, all, pb0, a1, b0);
        svmopa_za32_f32_m(3, all, pb1, a1, b1);
      }
    }
  } else if (nt == 32) {
    const svcount_t pc = svptrue_c32();
    if (Taps1) {
      const int64_t ldb = bo.ldb;
      int64_t kk = 0;
      for (; kk + 2 <= k; kk += 2, a += 64, b += 2 * ldb) {
        const svfloat32x2_t av = svld1_f32_x2(pc, a), bv = svld1_f32_x2(pc, b);
        const svfloat32x2_t av2 = svld1_f32_x2(pc, a + 32), bv2 = svld1_f32_x2(pc, b + ldb);
        svmopa_za32_f32_m(0, all, all, svget2(av, 0), svget2(bv, 0));
        svmopa_za32_f32_m(1, all, all, svget2(av, 0), svget2(bv, 1));
        svmopa_za32_f32_m(2, all, all, svget2(av, 1), svget2(bv, 0));
        svmopa_za32_f32_m(3, all, all, svget2(av, 1), svget2(bv, 1));
        svmopa_za32_f32_m(0, all, all, svget2(av2, 0), svget2(bv2, 0));
        svmopa_za32_f32_m(1, all, all, svget2(av2, 0), svget2(bv2, 1));
        svmopa_za32_f32_m(2, all, all, svget2(av2, 1), svget2(bv2, 0));
        svmopa_za32_f32_m(3, all, all, svget2(av2, 1), svget2(bv2, 1));
      }
      if (kk < k) {
        const svfloat32x2_t av = svld1_f32_x2(pc, a), bv = svld1_f32_x2(pc, b);
        svmopa_za32_f32_m(0, all, all, svget2(av, 0), svget2(bv, 0));
        svmopa_za32_f32_m(1, all, all, svget2(av, 0), svget2(bv, 1));
        svmopa_za32_f32_m(2, all, all, svget2(av, 1), svget2(bv, 0));
        svmopa_za32_f32_m(3, all, all, svget2(av, 1), svget2(bv, 1));
      }
    } else {
      for (int64_t o = 0; o < outer; ++o) {
        const float* brow = b + o * bo.ldb;
        for (int64_t t = 0; t < taps; ++t, a += 32) {
          const svfloat32x2_t av = svld1_f32_x2(pc, a), bv = svld1_f32_x2(pc, brow + t * bo.dil);
          svmopa_za32_f32_m(0, all, all, svget2(av, 0), svget2(bv, 0));
          svmopa_za32_f32_m(1, all, all, svget2(av, 0), svget2(bv, 1));
          svmopa_za32_f32_m(2, all, all, svget2(av, 1), svget2(bv, 0));
          svmopa_za32_f32_m(3, all, all, svget2(av, 1), svget2(bv, 1));
        }
      }
    }
  } else {
    for (int64_t o = 0; o < outer; ++o) {
      const float* brow = b + o * bo.ldb;
      for (int64_t t = 0; t < taps; ++t, a += 32) {
        const float* bp = brow + t * bo.dil;
        const svfloat32_t a0 = svld1_f32(all, a), a1 = svld1_f32(all, a + 16);
        const svfloat32_t b0 = svld1_f32(pb0, bp), b1 = svld1_f32(pb1, bp + 16);
        svmopa_za32_f32_m(0, all, pb0, a0, b0);
        svmopa_za32_f32_m(1, all, pb1, a0, b1);
        svmopa_za32_f32_m(2, all, pb0, a1, b0);
        svmopa_za32_f32_m(3, all, pb1, a1, b1);
      }
    }
  }
  if (epi) {  // activation + column mask on the way out of ZA (worth it for long K; see gemmSme)
    const svfloat32_t zero = svdup_n_f32(0.f);
    const svfloat32_t m0 = epi->colMask ? svld1_f32(pb0, epi->colMask) : svdup_n_f32(1.f);
    const svfloat32_t m1 = epi->colMask ? svld1_f32(pb1, epi->colMask + 16) : svdup_n_f32(1.f);
    const Act act = epi->act;
    const float lk = epi->leaky;
    auto fin = [&](svfloat32_t v, svfloat32_t m) __arm_streaming {
      if (act == Act::Relu) v = svmax_n_f32_x(all, v, 0.f);
      else if (act == Act::Relu6) v = svmin_n_f32_x(all, svmax_n_f32_x(all, v, 0.f), 6.f);
      else if (act == Act::Leaky) v = svsel_f32(svcmpge_n_f32(all, v, 0.f), v, svmul_n_f32_x(all, v, lk));
      return svmul_f32_x(all, v, m);
    };
    for (int64_t i = 0; i < nr && i < 16; ++i) {
      const uint32_t sl = static_cast<uint32_t>(i);
      svst1_f32(pb0, c + i * ldc, fin(svread_hor_za32_f32_m(zero, all, 0, sl), m0));
      svst1_f32(pb1, c + i * ldc + 16, fin(svread_hor_za32_f32_m(zero, all, 1, sl), m1));
    }
    for (int64_t i = 16; i < nr; ++i) {
      const uint32_t sl = static_cast<uint32_t>(i - 16);
      svst1_f32(pb0, c + i * ldc, fin(svread_hor_za32_f32_m(zero, all, 2, sl), m0));
      svst1_f32(pb1, c + i * ldc + 16, fin(svread_hor_za32_f32_m(zero, all, 3, sl), m1));
    }
    return;
  }
  for (int64_t i = 0; i < nr && i < 16; ++i) {
    svst1_hor_za32(0, static_cast<uint32_t>(i), pb0, c + i * ldc);
    svst1_hor_za32(1, static_cast<uint32_t>(i), pb1, c + i * ldc + 16);
  }
  for (int64_t i = 16; i < nr; ++i) {
    svst1_hor_za32(2, static_cast<uint32_t>(i - 16), pb0, c + i * ldc);
    svst1_hor_za32(3, static_cast<uint32_t>(i - 16), pb1, c + i * ldc + 16);
  }
}

// N-tail block (nt <= 16): 64 rows (two A panels) x 16 columns, the four ZA tiles stacked along M, so a narrow last
// column chunk costs half the FMOPAs per row of a 32 x 32 block (Kokoro's decoder Winograd GEMMs at L = F: n = 37).
// Plain (no bias / residual / accumulate / ZA epilogue) only; modes 0 and 1. iPhone interleaved A/B (2026-11-02,
// nominal): Kokoro app59f 98.0 / 99.6 ms vs 100.3 / 101.2 without, s9f 208.4 / 213.3 vs 213.5 / 213.2.
template <int Mode>
inline void block64x16(const float* a0, const float* a1, int64_t k, const BOperand& bo, const float* b, int64_t nt,
                       int64_t nr, float* c, int64_t ldc) __arm_streaming __arm_inout("za") {
  svzero_za();
  const svbool_t all = svptrue_b32();
  const svbool_t pb = svwhilelt_b32_s64(0, nt);
  const int64_t outer = Mode == 1 ? k / bo.taps : k, taps = Mode == 1 ? bo.taps : 1, dil = Mode == 1 ? bo.dil : 0;
  for (int64_t o = 0; o < outer; ++o) {
    const float* brow = b + o * bo.ldb;
    for (int64_t t = 0; t < taps; ++t, a0 += 32, a1 += 32) {
      const svfloat32_t bv = svld1_f32(pb, brow + t * dil);
      svmopa_za32_f32_m(0, all, pb, svld1_f32(all, a0), bv);
      svmopa_za32_f32_m(1, all, pb, svld1_f32(all, a0 + 16), bv);
      svmopa_za32_f32_m(2, all, pb, svld1_f32(all, a1), bv);
      svmopa_za32_f32_m(3, all, pb, svld1_f32(all, a1 + 16), bv);
    }
  }
  for (int64_t i = 0; i < 16 && i < nr; ++i) svst1_hor_za32(0, static_cast<uint32_t>(i), pb, c + i * ldc);
  for (int64_t i = 16; i < 32 && i < nr; ++i) svst1_hor_za32(1, static_cast<uint32_t>(i - 16), pb, c + i * ldc);
  for (int64_t i = 32; i < 48 && i < nr; ++i) svst1_hor_za32(2, static_cast<uint32_t>(i - 32), pb, c + i * ldc);
  for (int64_t i = 48; i < nr; ++i) svst1_hor_za32(3, static_cast<uint32_t>(i - 48), pb, c + i * ldc);
}

// Panels [p0, p1) × all n (B 32-column blocks outside: each streams once per call).
template <int Mode>
__arm_new("za") __arm_locally_streaming void chunkSme(const float* ap, int64_t m, int64_t k, const BOperand& b,
                                                      int64_t n, float* c, int64_t ldc, const Epilogue& e,
                                                      const float* res, int64_t p0, int64_t p1, bool epiZa) {
  for (int64_t j0 = 0; j0 < n; j0 += 32) {
    const float* bj = b.b + (j0 / 32) * b.panel;
    const int64_t nt = n - j0 < 32 ? n - j0 : 32;
    Epilogue ej = e;
    if (ej.colMask) ej.colMask += j0;
    const bool plain = !e.bias && !res && !e.accumulate && !epiZa;
    for (int64_t p = p0; p < p1; ++p) {
      const int64_t i0 = p * 32, nr = m - i0 < 32 ? m - i0 : 32;
      if (Mode != 2 && plain && nt <= 16 && p + 1 < p1) {
        block64x16<Mode>(ap + p * 32 * k, ap + (p + 1) * 32 * k, k, b, bj, nt, m - i0 < 64 ? m - i0 : 64,
                         c + i0 * ldc + j0, ldc);
        ++p;
        continue;
      }
      block<Mode>(ap + p * 32 * k, k, b, bj, nt, nr, c + i0 * ldc + j0, ldc,
                   e.bias && !e.biasPerRow ? e.bias + j0 : nullptr, e.bias && e.biasPerRow ? e.bias + i0 : nullptr,
                   res ? res + i0 * e.ldres + j0 : nullptr, e.ldres, e.accumulate, epiZa ? &ej : nullptr);
    }
  }
}

// Non-streaming NEON pass over rows [r0, r1) of C: + residual (only when accumulate also is set), the activation,
// the column mask.
void postPass(float* c, int64_t ldc, int64_t r0, int64_t r1, int64_t n, const Epilogue& e, bool addRes) {
  const float32x4_t zero = vdupq_n_f32(0.f), six = vdupq_n_f32(6.f), sl = vdupq_n_f32(e.leaky);
  const float* mk = e.colMask;
  for (int64_t i = r0; i < r1; ++i) {
    float* o = c + i * ldc;
    const float* r = addRes ? e.res + i * e.ldres : nullptr;
    int64_t j = 0;
    for (; j + 4 <= n; j += 4) {
      float32x4_t v = vld1q_f32(o + j);
      if (r) v = vaddq_f32(v, vld1q_f32(r + j));
      switch (e.act) {
        case Act::None: break;
        case Act::Relu: v = vmaxq_f32(v, zero); break;
        case Act::Relu6: v = vminq_f32(vmaxq_f32(v, zero), six); break;
        case Act::Leaky: v = vbslq_f32(vcgeq_f32(v, zero), v, vmulq_f32(v, sl)); break;
      }
      if (mk) v = vmulq_f32(v, vld1q_f32(mk + j));
      vst1q_f32(o + j, v);
    }
    for (; j < n; ++j) o[j] = actS(o[j] + (r ? r[j] : 0.f), e) * (mk ? mk[j] : 1.f);
  }
}

template <int Mode>
void gemmSme(const float* ap, int64_t m, int64_t k, const BOperand& b, int64_t n, float* c, int64_t ldc,
             const Epilogue& e) {
  // Activation / mask: read back from ZA per row when K is long (the readback is ~10 % of a K = 576 block; at K = 24 it
  // tripled the time, so short K stores straight from ZA and runs a NEON pass per chunk instead).
  const bool resInZa = e.res && !e.accumulate;
  const bool needEpi = e.act != Act::None || e.colMask != nullptr;
  // Row-gather B (planar convs): from K = 256 (ResNet-18's planar stem, K = 147: ZA readback 288 -> 235 us at N = 1,
  // 2.49 -> 2.30 ms at N = 8 with the NEON pass instead, iPhone 2026-11-02).
  const bool epiZa = needEpi && !(e.res && e.accumulate) && k >= (Mode == 2 ? 256 : 128);
  const bool post = !epiZa && (needEpi || (e.res && e.accumulate));
  const int64_t panels = (m + 31) / 32;
  // Chunk: packed A <= 4 MiB; with a post-pass also C rows <= 256 KiB (still in cache for the NEON pass).
  int64_t chunk = std::max<int64_t>(1, kChunkBytes / (k * 128));
  if (post) chunk = std::min(chunk, std::max<int64_t>(1, (int64_t{256} << 10) / (n * 4 * 32)));
  for (int64_t p0 = 0; p0 < panels; p0 += chunk) {
    const int64_t p1 = std::min(panels, p0 + chunk);
    chunkSme<Mode>(ap, m, k, b, n, c, ldc, e, resInZa ? e.res : nullptr, p0, p1, epiZa);
    if (post) postPass(c, ldc, p0 * 32, std::min(m, p1 * 32), n, e, e.res && e.accumulate);
  }
}

// One 32-row panel: 16-column steps of k, rows loaded as horizontal ZA slices of tiles 0 (rows 0-15) and 1 (16-31),
// read back as vertical slices (one k each) into dst[kk][0:16] / dst[kk][16:32].
__arm_new("za") __arm_locally_streaming void packSme(const float* src, int64_t ld, int64_t rows, int64_t k,
                                                     float* dst) {
  const svbool_t all = svptrue_b32();
  for (int64_t p = 0; p * 32 < rows; ++p) {
    const int64_t nr = std::min<int64_t>(32, rows - p * 32);
    const float* s = src + p * 32 * ld;
    float* d = dst + p * 32 * k;
    for (int64_t k0 = 0; k0 < k; k0 += 16) {
      const svbool_t pk = svwhilelt_b32_s64(k0, k);
      if (nr < 32) svzero_za();
      for (int64_t i = 0; i < std::min<int64_t>(16, nr); ++i)
        svld1_hor_za32(0, static_cast<uint32_t>(i), pk, s + i * ld + k0);
      for (int64_t i = 16; i < nr; ++i) svld1_hor_za32(1, static_cast<uint32_t>(i - 16), pk, s + i * ld + k0);
      const int64_t kn = std::min<int64_t>(16, k - k0);
      for (int64_t j = 0; j < kn; ++j) {
        svst1_ver_za32(0, static_cast<uint32_t>(j), all, d + (k0 + j) * 32);
        svst1_ver_za32(1, static_cast<uint32_t>(j), all, d + (k0 + j) * 32 + 16);
      }
    }
  }
}

// Gather variant: row r's k values are segs segments of seglen floats at ptrs[r * segs + s].
__arm_new("za") __arm_locally_streaming void packGatherSme(const float* const* ptrs, int64_t rows, int64_t segs,
                                                           int64_t seglen, float* dst) {
  const svbool_t all = svptrue_b32();
  const int64_t k = segs * seglen;
  for (int64_t p = 0; p * 32 < rows; ++p) {
    const int64_t nr = rows - p * 32 < 32 ? rows - p * 32 : 32;
    const float* const* pr = ptrs + p * 32 * segs;
    float* d = dst + p * 32 * k;
    for (int64_t sg = 0; sg < segs; ++sg)
      for (int64_t k0 = 0; k0 < seglen; k0 += 16) {
        const svbool_t pk = svwhilelt_b32_s64(k0, seglen);
        if (nr < 32) svzero_za();
        for (int64_t i = 0; i < nr && i < 16; ++i) svld1_hor_za32(0, static_cast<uint32_t>(i), pk, pr[i * segs + sg] + k0);
        for (int64_t i = 16; i < nr; ++i) svld1_hor_za32(1, static_cast<uint32_t>(i - 16), pk, pr[i * segs + sg] + k0);
        const int64_t kn = seglen - k0 < 16 ? seglen - k0 : 16;
        float* dk = d + (sg * seglen + k0) * 32;
        for (int64_t j = 0; j < kn; ++j) {
          svst1_ver_za32(0, static_cast<uint32_t>(j), all, dk + j * 32);
          svst1_ver_za32(1, static_cast<uint32_t>(j), all, dk + j * 32 + 16);
        }
      }
  }
}

}  // namespace

void packPanelsGather(const float* const* ptrs, int64_t rows, int64_t segs, int64_t seglen, float* dst) {
  if (available()) return packGatherSme(ptrs, rows, segs, seglen, dst);
  packGatherScalar(ptrs, rows, segs, seglen, dst);
}

void packPanels(const float* src, int64_t ld, int64_t rows, int64_t k, float* dst) {
  if (available()) return packSme(src, ld, rows, k, dst);
  for (int64_t p = 0; p * 32 < rows; ++p)
    for (int64_t kk = 0; kk < k; ++kk)
      for (int64_t j = 0; j < 32; ++j) {
        const int64_t r = p * 32 + j;
        dst[(p * k + kk) * 32 + j] = r < rows ? src[r * ld + kk] : 0.f;
      }
}

void gemm(const float* ap, int64_t m, int64_t k, const BOperand& b, int64_t n, float* c, int64_t ldc,
          const Epilogue& e) {
  if (m <= 0 || n <= 0) return;
  if (!available()) return gemmScalar(ap, m, k, b, n, c, ldc, e);
  if (b.rowOff) return gemmSme<2>(ap, m, k, b, n, c, ldc, e);
  if (b.taps == 1) return gemmSme<0>(ap, m, k, b, n, c, ldc, e);
  gemmSme<1>(ap, m, k, b, n, c, ldc, e);
}

#else

void packPanelsGather(const float* const* ptrs, int64_t rows, int64_t segs, int64_t seglen, float* dst) {
  packGatherScalar(ptrs, rows, segs, seglen, dst);
}

void packPanels(const float* src, int64_t ld, int64_t rows, int64_t k, float* dst) {
  for (int64_t p = 0; p * 32 < rows; ++p)
    for (int64_t kk = 0; kk < k; ++kk)
      for (int64_t j = 0; j < 32; ++j) {
        const int64_t r = p * 32 + j;
        dst[(p * k + kk) * 32 + j] = r < rows ? src[r * ld + kk] : 0.f;
      }
}

void gemm(const float* ap, int64_t m, int64_t k, const BOperand& b, int64_t n, float* c, int64_t ldc,
          const Epilogue& e) {
  if (m > 0 && n > 0) gemmScalar(ap, m, k, b, n, c, ldc, e);
}

#endif

void gemmA(const float* a, int64_t lda, int64_t m, int64_t k, const BOperand& b, int64_t n, float* c, int64_t ldc,
           const Epilogue& e) {
  thread_local std::vector<float> buf;
  const size_t need = packedSize(m, k);
  if (buf.size() < need) buf.resize(need);
  packPanels(a, lda, m, k, buf.data());
  gemm(buf.data(), m, k, b, n, c, ldc, e);
}

}  // namespace lk::kernels::cpu::sme2
