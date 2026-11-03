// SME2 helpers for the Kokoro CPU kernels, SME2 CPUs only (runtime check; the iPhone 18 Pro Max test phone): the runtime check itself and the
// LSTM step GEMV with the whole gate vector in ZA. The conv GEMMs (FMOPA 32x32 blocks, pre-packed weight panels, B
// rows read straight from the conv window) moved into the shared engine sme2_gemm.{h,cpp} (2026-11-02): a conv tile is
// sme2::gemm(wp, co, ci·k, {xw, win, 32, k, d}, n, o, ldo) with wp from sme2::packPanels(W [co, ci·k]).
#include "sme2_conv.h"

#include <sys/sysctl.h>

#include <algorithm>

#if defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>
#endif

namespace lk::kernels::cpu::sme2 {

bool available() {
#if defined(__ARM_FEATURE_SME2)
  static const bool ok = [] {
    int v = 0;
    size_t len = sizeof v;
    return sysctlbyname("hw.optional.arm.FEAT_SME2", &v, &len, nullptr, 0) == 0 && v == 1;
  }();
  return ok;
#else
  return false;
#endif
}

#if defined(__ARM_FEATURE_SME2)

// y[0:n] += Σ_j wt[j·ldw + i] · x[j] (j < k), n a multiple of 64: per 256 outputs, 4 partial sums in ZA (partial
// j mod 4: 4 partials x 4 groups of 4 vectors = all 16 vg1x4 groups), one multi-vector FMLA per row and 64 outputs
// with x[j] broadcast; then y += (p0 + p1) + (p2 + p3). Bit-identical to kokoro_lstm.cpp's NEON gemvT4 (whose
// summation order the Kokoro SNR tests validate); the single-accumulator version moved app47m's SNR below the limit.
__arm_new("za") __arm_locally_streaming void gemvT(const float* wt, int64_t ldw, int64_t k, int64_t n, const float* x,
                                                  float* y) {
  const svcount_t pc = svptrue_c32();
  const svbool_t all = svptrue_b32();
  for (int64_t base = 0; base < n; base += 256) {
    const int64_t groups = n - base >= 256 ? 4 : (n - base) / 64;
    svzero_za();
    for (int64_t j = 0; j < k; ++j) {
      const svfloat32_t xv = svdup_n_f32(x[j]);
      const float* row = wt + j * ldw + base;
      const uint32_t p = static_cast<uint32_t>(j & 3) * 4;
      for (int64_t b = 0; b < groups; ++b)
        svmla_single_za32_f32_vg1x4(p + static_cast<uint32_t>(b), svld1_f32_x4(pc, row + 64 * b), xv);
    }
    for (int64_t b = 0; b < groups; ++b) {
      const uint32_t g = static_cast<uint32_t>(b);
      const svfloat32x4_t a0 = svread_za32_f32_vg1x4(g), a1 = svread_za32_f32_vg1x4(4 + g);
      const svfloat32x4_t a2 = svread_za32_f32_vg1x4(8 + g), a3 = svread_za32_f32_vg1x4(12 + g);
      float* yb = y + base + 64 * b;
      const svfloat32x4_t yv = svld1_f32_x4(pc, yb);
      auto sum = [&](svfloat32_t v, svfloat32_t q0, svfloat32_t q1, svfloat32_t q2, svfloat32_t q3) __arm_streaming {
        return svadd_f32_x(all, v, svadd_f32_x(all, svadd_f32_x(all, q0, q1), svadd_f32_x(all, q2, q3)));
      };
      const svfloat32x4_t out = svcreate4_f32(
          sum(svget4(yv, 0), svget4(a0, 0), svget4(a1, 0), svget4(a2, 0), svget4(a3, 0)),
          sum(svget4(yv, 1), svget4(a0, 1), svget4(a1, 1), svget4(a2, 1), svget4(a3, 1)),
          sum(svget4(yv, 2), svget4(a0, 2), svget4(a1, 2), svget4(a2, 2), svget4(a3, 2)),
          sum(svget4(yv, 3), svget4(a0, 3), svget4(a1, 3), svget4(a2, 3), svget4(a3, 3)));
      svst1_f32_x4(pc, yb, out);
    }
  }
}

#else

void gemvT(const float*, int64_t, int64_t, int64_t, const float*, float*) {}

#endif

}  // namespace lk::kernels::cpu::sme2
