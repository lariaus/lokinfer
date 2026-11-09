// SME2 f16 GEMM panel (see f16_sme2.h). Per k pair: the packed A pair-vectors of rows 0-15 and 16-31 (one 2-vector
// load), rows 2kk / 2kk + 1 of w at the block's 32 columns (two loads) zipped into the column-pair vectors of
// columns 0-15 and 16-31, and four widening FMOPAs (tiles 0..3 = rows 0-15 × cols 0-15, rows 0-15 × cols 16-31,
// rows 16-31 × cols 0-15, rows 16-31 × cols 16-31). Each FMOPA adds x[i, 2kk]·w[2kk, j] + x[i, 2kk+1]·w[2kk+1, j]
// to the f32 tile element. The tiles are stored as f32 rows; the caller runs the f16 epilogue (NEON).
#include "f16_sme2.h"

#include <cstring>

#include "sme2_gemm.h"

#if defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>
#endif

namespace lk::kernels::cpu::sme2f16 {

void packPairs(const __fp16* x, int64_t ldx, int64_t m, int64_t k, uint32_t* dst) {
  // The u32 view of x: [m, k / 2] with row stride ldx / 2; packPanels moves 32-bit words only (ZA slices / copies).
  sme2::packPanels(reinterpret_cast<const float*>(x), ldx / 2, m, k / 2, reinterpret_cast<float*>(dst));
}

#if defined(__ARM_FEATURE_SME2)

namespace {

__arm_new("za") __arm_locally_streaming void panelSme(const uint32_t* ap, int64_t k, const __fp16* w, int64_t ldw,
                                                      int64_t n, int64_t nr, float* c, int64_t ldc) {
  const svbool_t all16 = svptrue_b16();
  const svcount_t pc = svptrue_c16();
  const int64_t k2 = k / 2;
  const float16_t* wb = reinterpret_cast<const float16_t*>(w);
  for (int64_t j0 = 0; j0 < n; j0 += 32) {
    const int64_t nt = n - j0 < 32 ? n - j0 : 32;
    const svbool_t pl = svwhilelt_b16_s64(0, nt);  // the block's columns in a w row (zero past n)
    const svbool_t pb0 = svwhilelt_b32_s64(0, nt), pb1 = svwhilelt_b32_s64(16, nt);
    svzero_za();
    const float16_t* a = reinterpret_cast<const float16_t*>(ap);
    const float16_t* b = wb + j0;
    int64_t kk = 0;
    for (; kk + 2 <= k2; kk += 2, a += 128, b += 4 * ldw) {
      const svfloat16x2_t av = svld1_f16_x2(pc, a), av2 = svld1_f16_x2(pc, a + 64);
      const svfloat16_t r0 = svld1_f16(pl, b), r1 = svld1_f16(pl, b + ldw);
      const svfloat16_t r2 = svld1_f16(pl, b + 2 * ldw), r3 = svld1_f16(pl, b + 3 * ldw);
      const svfloat16_t z0 = svzip1_f16(r0, r1), z1 = svzip2_f16(r0, r1);
      const svfloat16_t z2 = svzip1_f16(r2, r3), z3 = svzip2_f16(r2, r3);
      svmopa_za32_f16_m(0, all16, all16, svget2(av, 0), z0);
      svmopa_za32_f16_m(1, all16, all16, svget2(av, 0), z1);
      svmopa_za32_f16_m(2, all16, all16, svget2(av, 1), z0);
      svmopa_za32_f16_m(3, all16, all16, svget2(av, 1), z1);
      svmopa_za32_f16_m(0, all16, all16, svget2(av2, 0), z2);
      svmopa_za32_f16_m(1, all16, all16, svget2(av2, 0), z3);
      svmopa_za32_f16_m(2, all16, all16, svget2(av2, 1), z2);
      svmopa_za32_f16_m(3, all16, all16, svget2(av2, 1), z3);
    }
    if (kk < k2) {
      const svfloat16x2_t av = svld1_f16_x2(pc, a);
      const svfloat16_t r0 = svld1_f16(pl, b), r1 = svld1_f16(pl, b + ldw);
      const svfloat16_t z0 = svzip1_f16(r0, r1), z1 = svzip2_f16(r0, r1);
      svmopa_za32_f16_m(0, all16, all16, svget2(av, 0), z0);
      svmopa_za32_f16_m(1, all16, all16, svget2(av, 0), z1);
      svmopa_za32_f16_m(2, all16, all16, svget2(av, 1), z0);
      svmopa_za32_f16_m(3, all16, all16, svget2(av, 1), z1);
    }
    for (int64_t i = 0; i < nr && i < 16; ++i) {
      svst1_hor_za32(0, static_cast<uint32_t>(i), pb0, c + i * ldc + j0);
      svst1_hor_za32(1, static_cast<uint32_t>(i), pb1, c + i * ldc + j0 + 16);
    }
    for (int64_t i = 16; i < nr; ++i) {
      svst1_hor_za32(2, static_cast<uint32_t>(i - 16), pb0, c + i * ldc + j0);
      svst1_hor_za32(3, static_cast<uint32_t>(i - 16), pb1, c + i * ldc + j0 + 16);
    }
  }
}

}  // namespace

void panelGemm(const uint32_t* ap, int64_t k, const __fp16* w, int64_t ldw, int64_t n, int64_t nr, float* c,
               int64_t ldc) {
  if (n > 0 && nr > 0) panelSme(ap, k, w, ldw, n, nr, c, ldc);
}

#else

// Scalar reference (only for builds without SME2 support; callers check sme2::available() first).
void panelGemm(const uint32_t* ap, int64_t k, const __fp16* w, int64_t ldw, int64_t n, int64_t nr, float* c,
               int64_t ldc) {
  const __fp16* a = reinterpret_cast<const __fp16*>(ap);
  for (int64_t i = 0; i < nr; ++i)
    for (int64_t j = 0; j < n; ++j) {
      float s = 0.f;
      for (int64_t kk = 0; kk < k; ++kk) s += static_cast<float>(a[(kk / 2) * 64 + 2 * i + kk % 2]) * w[kk * ldw + j];
      c[i * ldc + j] = s;
    }
}

#endif

}  // namespace lk::kernels::cpu::sme2f16
