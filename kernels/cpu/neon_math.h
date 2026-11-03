// Vectorized f32 math for the CPU kernels (internal header): polynomial transcendentals on 4 NEON lanes, shared by
// every CPU kernel family (Kokoro signal / Snake / attention softmax / GELU epilogues).
//
// Accuracy (measured against double, tests/kernels/test_kokoro_kernels.py::test_neon_math):
//   exp4      rel 2.2e-7 (|x| < 80)   log4    abs 1e-7 on [0.5, 1], rel 1e-7 elsewhere
//   sin4/cos4 abs 1.2e-7, |x| < 1e4 (f32 Cody-Waite reduction by pi); sinSq4 abs 2.5e-7, |x| < 1e4
//   sin4Wide  abs 1.2e-7, |x| < 1.6e7 (reduction in double: sin of the exact f32 argument)
//   tanh4     abs 8.4e-8              atan2_4 abs 2.7e-7 rad
// (tests/unit/test_sme2_conv.cpp "neon_math", 2026-11-02, M2 Max)
// LSTM gates keep Accelerate's vvexpf / vvtanhf on purpose: their ulp-level differences move the chaotic F0 path.
#pragma once

#include <arm_neon.h>

#include <cmath>

namespace lk::kernels::cpu::detail {

// exp(x) for 4 lanes: 2^n * p(r), r = x - n ln2 in [-ln2/2, ln2/2], degree-6 Taylor (rel. error ~1.2e-7).
inline float32x4_t exp4(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.f)), vdupq_n_f32(88.f));
  const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
  float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
  r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
  float32x4_t p = vdupq_n_f32(1.f / 720.f);
  p = vfmaq_f32(vdupq_n_f32(1.f / 120.f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.f / 24.f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.f / 6.f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
  const int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), e));
}

// gelu_tanh(x) = 0.5 x (1 + tanh(u)) = x / (1 + exp(-2u)), u = sqrt(2/pi) (x + 0.044715 x^3).
inline float32x4_t geluTanh4(float32x4_t x) {
  const float32x4_t x2 = vmulq_f32(x, x);
  const float32x4_t u2 = vmulq_f32(vmulq_f32(x, vfmaq_f32(vdupq_n_f32(1.f), x2, vdupq_n_f32(0.044715f))),
                                   vdupq_n_f32(-2.f * 0.7978845608028654f));
  return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.f), exp4(u2)));
}

inline float geluTanh1(float x) {
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + std::tanh(u));
}

// sin(r) for |r| <= pi/2: degree-11 odd polynomial (least-squares fit on Chebyshev nodes, fit error 2e-11 in double).
inline float32x4_t sinPoly4(float32x4_t r) {
  const float32x4_t r2 = vmulq_f32(r, r);
  float32x4_t p = vdupq_n_f32(-2.384194136e-8f);
  p = vfmaq_f32(vdupq_n_f32(2.752230066e-6f), p, r2);
  p = vfmaq_f32(vdupq_n_f32(-1.984079643e-4f), p, r2);
  p = vfmaq_f32(vdupq_n_f32(8.333330420e-3f), p, r2);
  p = vfmaq_f32(vdupq_n_f32(-1.666666660e-1f), p, r2);
  return vfmaq_f32(r, vmulq_f32(p, r2), r);
}

// x = j·pi + r, |r| <= pi/2 (Cody-Waite, 3 parts; j·C1 exact for |j| < 2^16, so |x| < ~1e4 for full accuracy).
inline float32x4_t reducePi4(float32x4_t x, float32x4_t& j) {
  j = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(0.318309886183790671f)));
  float32x4_t r = vfmsq_f32(x, j, vdupq_n_f32(3.140625f));
  r = vfmsq_f32(r, j, vdupq_n_f32(9.67502593994140625e-4f));
  return vfmsq_f32(r, j, vdupq_n_f32(1.509957990978376432e-7f));
}

// (-1)^j · v (j integral).
inline float32x4_t signByParity4(float32x4_t v, float32x4_t j) {
  const uint32x4_t odd = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(j)), 31);
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), odd));
}

// sin(x)^2 (sign of sin drops out of the square). Measured max abs error 2.5e-7 for |x| < 1e4.
inline float32x4_t sinSq4(float32x4_t x) {
  float32x4_t j;
  const float32x4_t sn = sinPoly4(reducePi4(x, j));
  return vmulq_f32(sn, sn);
}

// sin(x), |x| < ~1e4.
inline float32x4_t sin4(float32x4_t x) {
  float32x4_t j;
  return signByParity4(sinPoly4(reducePi4(x, j)), j);
}

// cos(x) = (-1)^(j+1) sin(r), x = (j + 1/2)·pi + r; |x| < ~1e4.
inline float32x4_t cos4(float32x4_t x) {
  const float32x4_t j = vrndnq_f32(vfmsq_f32(vmulq_f32(x, vdupq_n_f32(0.318309886183790671f)), vdupq_n_f32(1.f),
                                             vdupq_n_f32(0.5f)));
  const float32x4_t jh = vaddq_f32(j, vdupq_n_f32(0.5f));  // exact: j + 1/2 representable for |j| < 2^22
  float32x4_t r = vfmsq_f32(x, jh, vdupq_n_f32(3.140625f));
  r = vfmsq_f32(r, jh, vdupq_n_f32(9.67502593994140625e-4f));
  r = vfmsq_f32(r, jh, vdupq_n_f32(1.509957990978376432e-7f));
  return signByParity4(sinPoly4(r), vaddq_f32(j, vdupq_n_f32(1.f)));
}

// sin(x) of the exact f32 argument for |x| < 1.6e7: the reduction by pi runs in double (two lanes at a time), so huge
// phases (Kokoro's SineGen reaches ~1e6 rad) keep full accuracy.
inline float32x4_t sin4Wide(float32x4_t x) {
  const float64x2_t invPi = vdupq_n_f64(0.318309886183790671537767526745);
  const float64x2_t piHi = vdupq_n_f64(3.141592653589793116), piLo = vdupq_n_f64(1.2246467991473532e-16);
  const float64x2_t x0 = vcvt_f64_f32(vget_low_f32(x)), x1 = vcvt_high_f64_f32(x);
  const float64x2_t j0 = vrndnq_f64(vmulq_f64(x0, invPi)), j1 = vrndnq_f64(vmulq_f64(x1, invPi));
  const float64x2_t r0 = vfmsq_f64(vfmsq_f64(x0, j0, piHi), j0, piLo);
  const float64x2_t r1 = vfmsq_f64(vfmsq_f64(x1, j1, piHi), j1, piLo);
  const float32x4_t r = vcvt_high_f32_f64(vcvt_f32_f64(r0), r1);
  const uint32x4_t odd = vshlq_n_u32(
      vuzp1q_u32(vreinterpretq_u32_s64(vcvtq_s64_f64(j0)), vreinterpretq_u32_s64(vcvtq_s64_f64(j1))), 31);
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sinPoly4(r)), odd));
}

// ln(x) for x > 0 (Cephes logf: x = 2^e·m, m in [sqrt(1/2), sqrt(2)), degree-8 polynomial; finite inputs only).
inline float32x4_t log4(float32x4_t x) {
  const int32x4_t bits = vreinterpretq_s32_f32(x);
  float32x4_t e = vcvtq_f32_s32(vsubq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(126)));
  float32x4_t m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(bits, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F000000)));
  const uint32x4_t small = vcltq_f32(m, vdupq_n_f32(0.707106781186547524f));  // m in [0.5, 1)
  e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(small, vreinterpretq_u32_f32(vdupq_n_f32(1.f)))));
  m = vsubq_f32(vaddq_f32(m, vreinterpretq_f32_u32(vandq_u32(small, vreinterpretq_u32_f32(m)))), vdupq_n_f32(1.f));
  const float32x4_t z = vmulq_f32(m, m);
  float32x4_t p = vdupq_n_f32(7.0376836292e-2f);
  p = vfmaq_f32(vdupq_n_f32(-1.1514610310e-1f), p, m);
  p = vfmaq_f32(vdupq_n_f32(1.1676998740e-1f), p, m);
  p = vfmaq_f32(vdupq_n_f32(-1.2420140846e-1f), p, m);
  p = vfmaq_f32(vdupq_n_f32(1.4249322787e-1f), p, m);
  p = vfmaq_f32(vdupq_n_f32(-1.6668057665e-1f), p, m);
  p = vfmaq_f32(vdupq_n_f32(2.0000714765e-1f), p, m);
  p = vfmaq_f32(vdupq_n_f32(-2.4999993993e-1f), p, m);
  p = vfmaq_f32(vdupq_n_f32(3.3333331174e-1f), p, m);
  float32x4_t y = vmulq_f32(vmulq_f32(p, m), z);
  y = vfmaq_f32(y, e, vdupq_n_f32(-2.12194440e-4f));
  y = vfmsq_f32(y, z, vdupq_n_f32(0.5f));
  return vfmaq_f32(vaddq_f32(m, y), e, vdupq_n_f32(0.693359375f));
}

// tanh(x): odd polynomial for |x| < 0.625 (Cephes tanhf), 1 - 2 / (exp(2|x|) + 1) above, sign restored.
inline float32x4_t tanh4(float32x4_t x) {
  const float32x4_t ax = vabsq_f32(x);
  const float32x4_t z = vmulq_f32(x, x);
  float32x4_t p = vdupq_n_f32(-5.70498872745e-3f);
  p = vfmaq_f32(vdupq_n_f32(2.06390887954e-2f), p, z);
  p = vfmaq_f32(vdupq_n_f32(-5.37397155531e-2f), p, z);
  p = vfmaq_f32(vdupq_n_f32(1.33314422036e-1f), p, z);
  p = vfmaq_f32(vdupq_n_f32(-3.33332819422e-1f), p, z);
  const float32x4_t small = vfmaq_f32(x, vmulq_f32(p, z), x);
  const float32x4_t e = exp4(vminq_f32(vaddq_f32(ax, ax), vdupq_n_f32(40.f)));
  float32x4_t big = vsubq_f32(vdupq_n_f32(1.f), vdivq_f32(vdupq_n_f32(2.f), vaddq_f32(e, vdupq_n_f32(1.f))));
  big = vbslq_f32(vdupq_n_u32(0x80000000u), x, big);  // copysign
  return vbslq_f32(vcltq_f32(ax, vdupq_n_f32(0.625f)), small, big);
}

// atan2(y, x) (Cephes atanf on [0, 1] with the tan(pi/8) split, then the octant fix-ups); atan2(+-0, x < 0) = +-pi,
// atan2(0, 0) = 0.
inline float32x4_t atan2_4(float32x4_t y, float32x4_t x) {
  const float32x4_t ax = vabsq_f32(x), ay = vabsq_f32(y);
  const uint32x4_t swap = vcgtq_f32(ay, ax);
  const float32x4_t num = vminq_f32(ax, ay), den = vmaxq_f32(ax, ay);
  float32x4_t t = vdivq_f32(num, den);
  t = vbslq_f32(vceqq_f32(den, vdupq_n_f32(0.f)), vdupq_n_f32(0.f), t);
  const uint32x4_t mid = vcgtq_f32(t, vdupq_n_f32(0.4142135623730950f));
  t = vbslq_f32(mid, vdivq_f32(vsubq_f32(t, vdupq_n_f32(1.f)), vaddq_f32(t, vdupq_n_f32(1.f))), t);
  const float32x4_t z = vmulq_f32(t, t);
  float32x4_t p = vdupq_n_f32(8.05374449538e-2f);
  p = vfmaq_f32(vdupq_n_f32(-1.38776856032e-1f), p, z);
  p = vfmaq_f32(vdupq_n_f32(1.99777106478e-1f), p, z);
  p = vfmaq_f32(vdupq_n_f32(-3.33329491539e-1f), p, z);
  float32x4_t r = vfmaq_f32(t, vmulq_f32(p, z), t);
  r = vbslq_f32(mid, vaddq_f32(r, vdupq_n_f32(0.785398163397448310f)), r);
  r = vbslq_f32(swap, vsubq_f32(vdupq_n_f32(1.570796326794896619f), r), r);
  r = vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.f)), vsubq_f32(vdupq_n_f32(3.141592653589793238f), r), r);
  return vbslq_f32(vdupq_n_u32(0x80000000u), y, r);  // copysign(r, y)
}

}  // namespace lk::kernels::cpu::detail
