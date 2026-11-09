// SME2 f16 GEMM panel (f16_sme2.cpp): widening FMOPA (f16 × f16 → f32, two k per outer product) into the four
// 16×16 f32 ZA tiles, one 32-row × 32-column block of C per pass over K. Compiled with +sme2 on every Apple target;
// sme2::available() (sme2_conv.h) says whether this CPU runs it (the iPhone yes, the M2 Max no). Used by the f16
// optimized GEMMs (f16_fused.cpp) for large M on SME2 CPUs.
//
// Operands: A (rows of C) PACKED in k pairs: 32-row panels [k/2][32] of u32 = (x[r, 2kk], x[r, 2kk + 1]) — the f16
// pairs are adjacent in a row-major x, so packing is sme2::packPanels on the u32 view (packPairs). B is the f16
// weight w[k, n] row-major, read in place: rows 2kk and 2kk + 1 are zipped into pairs in registers.
#pragma once

#include <cstdint>

#include "sme2_conv.h"  // sme2::available()

namespace lk::kernels::cpu::sme2f16 {

// x [m, k] f16 (row stride ldx, ldx and k even) -> dst [ceil(m / 32)][k / 2][32] u32 (zero rows past m).
void packPairs(const __fp16* x, int64_t ldx, int64_t m, int64_t k, uint32_t* dst);

// c[0:nr, 0:n] (f32, row stride ldc) = A panel (32 rows, from packPairs; k even) @ w[0:k, 0:n] (f16, row stride
// ldw). nr <= 32. SME2 CPUs only (callers check sme2::available()).
void panelGemm(const uint32_t* ap, int64_t k, const __fp16* w, int64_t ldw, int64_t n, int64_t nr, float* c,
               int64_t ldc);

}  // namespace lk::kernels::cpu::sme2f16
