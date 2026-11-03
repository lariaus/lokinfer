// Kokoro signal kernels: hn-NSF harmonic source (torch-exact op order), STFT -> (mag, phase), iSTFT from logits.
// FP contraction is off in this file: every fused multiply-add below is an explicit std::fma that mirrors
// one PyTorch performs (verified bit-exact against torch 2.14 CPU for the phase interpolation).
#include <algorithm>
#include <cmath>
#include <cstring>

#include "kokoro_common.h"
#include "lokinfer/kernels/cpu_kokoro.h"
#include "lokinfer/runtime/kernel.h"

#pragma clang fp contract(off)

namespace lk::kernels::cpu {

namespace {

constexpr int kHarmonics = 9;     // fundamental + 8 overtones
constexpr int64_t kUp = 300;      // prod(upsample_rates) * hop = 10 * 6 * 5
constexpr float kSampleRate = 24000.f;
constexpr float kVoicedThreshold = 10.f;
constexpr float kSineAmp = 0.1f;
constexpr float kNoiseStd = 0.003f;

// torch.remainder for floats.
inline float torchRemainder(float a, float b) {
  float m = std::fmod(a, b);
  if (m != 0.f && ((b < 0.f) != (m < 0.f))) m += b;
  return m;
}

// Periodic hann window of length n (scipy get_window('hann', n, fftbins=True)), computed in double.
void hannPeriodic(float* w, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n)));
}

}  // namespace

namespace {

// Counter-based Philox4x32-10 (Salmon et al. 2011): stateless, so every output sample's noise depends only on
// (seed, sample index) and the kernel splits across threads without changing results. Sample i uses counters
// (i_lo, i_hi, blk, 0x6C6B6E7A) for blk = 0..2, key = seed; Box-Muller on the word pairs gives its 9 normals.
// Philox4x32-10 on 4 counters at once (lanes = 4 consecutive samples): same stream as philox4x32 per lane.
inline void philox4x32x4(uint32x4_t c[4], uint32_t k0, uint32_t k1) {
  const uint32x2_t m0 = vdup_n_u32(0xD2511F53u), m1 = vdup_n_u32(0xCD9E8D57u);
  for (int r = 0; r < 10; ++r) {
    const uint64x2_t p0l = vmull_u32(vget_low_u32(c[0]), m0), p0h = vmull_high_u32(c[0], vcombine_u32(m0, m0));
    const uint64x2_t p1l = vmull_u32(vget_low_u32(c[2]), m1), p1h = vmull_high_u32(c[2], vcombine_u32(m1, m1));
    const uint32x4_t lo0 = vuzp1q_u32(vreinterpretq_u32_u64(p0l), vreinterpretq_u32_u64(p0h));
    const uint32x4_t hi0 = vuzp2q_u32(vreinterpretq_u32_u64(p0l), vreinterpretq_u32_u64(p0h));
    const uint32x4_t lo1 = vuzp1q_u32(vreinterpretq_u32_u64(p1l), vreinterpretq_u32_u64(p1h));
    const uint32x4_t hi1 = vuzp2q_u32(vreinterpretq_u32_u64(p1l), vreinterpretq_u32_u64(p1h));
    const uint32x4_t n0 = veorq_u32(veorq_u32(hi1, c[1]), vdupq_n_u32(k0));
    const uint32x4_t n2 = veorq_u32(veorq_u32(hi0, c[3]), vdupq_n_u32(k1));
    c[0] = n0;
    c[1] = lo1;
    c[2] = n2;
    c[3] = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
}

// Standard normals of samples i..i+3 (lane = sample): nz[h], h < kHarmonics. Same Philox counters and Box-Muller
// pairing as the scalar version this replaced (2026-11-02); log / sqrt / sin / cos by the NEON polynomials (ulp-level differences only).
inline void normals4(uint64_t seed, int64_t i, float32x4_t* nz) {
  const uint32_t k0 = static_cast<uint32_t>(seed), k1 = static_cast<uint32_t>(seed >> 32);
  const uint32_t idx[4] = {static_cast<uint32_t>(i), static_cast<uint32_t>(i + 1), static_cast<uint32_t>(i + 2),
                           static_cast<uint32_t>(i + 3)};
  const uint32x4_t lo = vld1q_u32(idx);
  const uint32_t hiv[4] = {static_cast<uint32_t>(static_cast<uint64_t>(i) >> 32),
                           static_cast<uint32_t>(static_cast<uint64_t>(i + 1) >> 32),
                           static_cast<uint32_t>(static_cast<uint64_t>(i + 2) >> 32),
                           static_cast<uint32_t>(static_cast<uint64_t>(i + 3) >> 32)};
  const uint32x4_t hi = vld1q_u32(hiv);
  const float32x4_t scale = vdupq_n_f32(2.3283064e-10f), half = vdupq_n_f32(0.5f);
  for (int blk = 0; blk * 4 < kHarmonics + 1; ++blk) {
    uint32x4_t c[4] = {lo, hi, vdupq_n_u32(static_cast<uint32_t>(blk)), vdupq_n_u32(0x6C6B6E7Au)};
    philox4x32x4(c, k0, k1);
    for (int q = 0; q < 2; ++q) {
      const int id = blk * 4 + 2 * q;
      if (id >= kHarmonics) break;
      const float32x4_t u1 = vmulq_f32(vaddq_f32(vcvtq_f32_u32(c[2 * q]), half), scale);
      const float32x4_t u2 = vmulq_f32(vcvtq_f32_u32(c[2 * q + 1]), scale);
      const float32x4_t r = vsqrtq_f32(vmulq_f32(vdupq_n_f32(-2.f), detail::log4(u1)));
      const float32x4_t t = vmulq_f32(vdupq_n_f32(6.2831853f), u2);
      nz[id] = vmulq_f32(r, detail::cos4(t));
      if (id + 1 < kHarmonics) nz[id + 1] = vmulq_f32(r, detail::sin4(t));
    }
  }
}

// noise4(i, nz) fills nz[h] (h < kHarmonics) with the noise of samples i..i+3.
template <typename Noise4>
void harmonicSource(const float* f0, Noise4 noise4, const float* w, const float* b, float* out, int64_t n) {
  // SineGen._f02sine, per frame j (80 Hz) and harmonic h:
  //   rad[s, h]  = (f0_up[s] * (h+1) / 24000) % 1              (f0_up = nearest x300; rand_ini only hits s = 0: dead)
  //   rad_d[j,h] = linear 1/300 downsample = rad at s = 300j+149/150 (same frame) * 0.5 each = rad of frame j, exactly
  //   phase      = cumsum(rad_d) (torch accumulates float cumsum in double) * 2 * pi
  //   up         = linear x300 upsample of (phase * 300)
  float* p300 = detail::scratch(static_cast<size_t>(n * kHarmonics), 0);
  const float twoPi[2] = {2.f, static_cast<float>(M_PI)};
  double acc[kHarmonics] = {};
  for (int64_t j = 0; j < n; ++j)
    for (int h = 0; h < kHarmonics; ++h) {
      float fn = f0[j] * static_cast<float>(h + 1);
      float rad = torchRemainder(fn / kSampleRate, 1.f);
      acc[h] += static_cast<double>(rad);
      float phase = (static_cast<float>(acc[h]) * twoPi[0]) * twoPi[1];
      p300[j * kHarmonics + h] = phase * static_cast<float>(kUp);
    }

  // upsample_linear1d (align_corners=False, scale_factor=300): torch's area_pixel_compute_source_index is
  // fma(scale, dst + 0.5, -0.5) in float and the interpolation is fma(t0, w0, t1 * w1). 4 samples per step (NEON):
  // the 4 samples share their frame (300 is a multiple of 4), sin by the double-reduced polynomial (the phase
  // reaches ~1e6 rad), tanh by the NEON polynomial.
  const float scale = static_cast<float>(1.0 / static_cast<double>(kUp));
  const float ampV = kNoiseStd;            // uv * 0.003 + (1 - uv) * 0.1 / 3 with uv = 1
  const float ampU = (1.f * kSineAmp) / 3.f;  //                                  with uv = 0
  const float32x4_t vb = vdupq_n_f32(b[0]);
  const int64_t s = n * kUp;  // a multiple of 4
  // Every output sample depends only on its index (phase table above, per-sample Philox counter), so the
  // samples split into independent blocks on the pool.
  constexpr int64_t kBlock = 4096;
  auto block = [&](int64_t blk) {
    const int64_t iEnd = std::min(s, (blk + 1) * kBlock);
    const float lane[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    const float32x4_t vlane = vld1q_f32(lane), zero = vdupq_n_f32(0.f), one = vdupq_n_f32(1.f);
    const int32x4_t last = vdupq_n_s32(static_cast<int32_t>(n - 1));
    for (int64_t i = blk * kBlock; i < iEnd; i += 4) {
      float32x4_t real = vfmaq_f32(vdupq_n_f32(-0.5f), vdupq_n_f32(scale),
                                   vaddq_f32(vdupq_n_f32(static_cast<float>(i)), vlane));
      real = vmaxq_f32(real, zero);
      const int32x4_t i0 = vminq_s32(vcvtmq_s32_f32(real), last);
      const float32x4_t l1 = vminq_f32(vmaxq_f32(vsubq_f32(real, vcvtq_f32_s32(i0)), zero), one);
      const float32x4_t l0 = vsubq_f32(one, l1);
      const int32x4_t i1 = vsubq_s32(i0, vreinterpretq_s32_u32(vcltq_s32(i0, last)));  // + 1 below the last frame
      int32_t a0[4], a1[4];
      vst1q_s32(a0, i0);
      vst1q_s32(a1, i1);
      const float* r0[4] = {p300 + a0[0] * kHarmonics, p300 + a0[1] * kHarmonics, p300 + a0[2] * kHarmonics,
                            p300 + a0[3] * kHarmonics};
      const float* r1[4] = {p300 + a1[0] * kHarmonics, p300 + a1[1] * kHarmonics, p300 + a1[2] * kHarmonics,
                            p300 + a1[3] * kHarmonics};
      const float f = f0[i / kUp];
      const bool voiced = f > kVoicedThreshold;
      const float32x4_t uv = vdupq_n_f32(voiced ? 1.f : 0.f), amp = vdupq_n_f32(voiced ? ampV : ampU);
      float32x4_t nz[kHarmonics];
      noise4(i, nz);
      float32x4_t y = zero;
      for (int h = 0; h < kHarmonics; ++h) {
        float32x4_t t0 = vdupq_n_f32(r0[0][h]), t1 = vdupq_n_f32(r1[0][h]);
        t0 = vld1q_lane_f32(r0[1] + h, t0, 1);
        t0 = vld1q_lane_f32(r0[2] + h, t0, 2);
        t0 = vld1q_lane_f32(r0[3] + h, t0, 3);
        t1 = vld1q_lane_f32(r1[1] + h, t1, 1);
        t1 = vld1q_lane_f32(r1[2] + h, t1, 2);
        t1 = vld1q_lane_f32(r1[3] + h, t1, 3);
        const float32x4_t phase = vfmaq_f32(vmulq_f32(t1, l1), t0, l0);
        const float32x4_t sine = vmulq_f32(detail::sin4Wide(phase), vdupq_n_f32(kSineAmp));
        const float32x4_t v = vaddq_f32(vmulq_f32(sine, uv), vmulq_f32(amp, nz[h]));
        y = vaddq_f32(y, vmulq_f32(v, vdupq_n_f32(w[h])));
      }
      vst1q_f32(out + i, detail::tanh4(vaddq_f32(y, vb)));
    }
  };
  const int64_t blocks = (s + kBlock - 1) / kBlock;
  if (blocks <= 1) {
    for (int64_t blk = 0; blk < blocks; ++blk) block(blk);
  } else {
    rt::ThreadPool::global().parallelFor(blocks, block);
  }
}

}  // namespace

void harmonic_source_f32(const float* f0, const float* noise, const float* w, const float* b, float* out, int64_t n) {
  harmonicSource(
      f0,
      [noise](int64_t i, float32x4_t* nz) {
        const float* p = noise + i * kHarmonics;
        for (int h = 0; h < kHarmonics; ++h) {
          const float v[4] = {p[h], p[kHarmonics + h], p[2 * kHarmonics + h], p[3 * kHarmonics + h]};
          nz[h] = vld1q_f32(v);
        }
      },
      w, b, out, n);
}

void harmonic_source_rng_f32(const float* f0, uint64_t seed, const float* w, const float* b, float* out, int64_t n) {
  harmonicSource(f0, [seed](int64_t i, float32x4_t* nz) { normals4(seed, i, nz); }, w, b, out, n);
}

void stft_magphase_f32(const float* x, float* out, int64_t s, int64_t nFft, int64_t hop) {
  const int64_t bins = nFft / 2 + 1, pad = nFft / 2, frames = s / hop + 1;
  // Windowed DFT basis: re[k][m] = w[m] cos(2 pi k m / N), im[k][m] = -w[m] sin(2 pi k m / N).
  float* tab = detail::scratch(static_cast<size_t>(2 * bins * nFft + nFft), 2);
  float* win = tab + 2 * bins * nFft;
  hannPeriodic(win, nFft);
  for (int64_t k = 0; k < bins; ++k)
    for (int64_t m = 0; m < nFft; ++m) {
      double ang = 2.0 * M_PI * static_cast<double>((k * m) % nFft) / static_cast<double>(nFft);
      tab[k * nFft + m] = static_cast<float>(win[m] * std::cos(ang));
      tab[(bins + k) * nFft + m] = static_cast<float>(-win[m] * std::sin(ang));
    }
  float* mag = out;
  float* ph = out + bins * frames;
  // Residue rows of the reflect-padded signal: xr[rho][j] = xpad[hop·j + rho] (xpad[q] = x[reflect(q - pad)]), so
  // sample m = hop·a + rho of frames t..t+3 is the vector xr[rho][t + a .. t + a + 3] (needs nFft % hop == 0).
  const int64_t taps = nFft / hop, len = frames + taps + 4;
  float* xr = detail::scratch(static_cast<size_t>(hop * len), 3);
  for (int64_t rho = 0; rho < hop; ++rho)
    for (int64_t j = 0; j < len; ++j) {
      int64_t q = hop * j + rho - pad;
      if (q < 0) q = -q;
      if (q >= s) q = 2 * (s - 1) - q;
      xr[rho * len + j] = (q >= 0 && q < s) ? x[q] : 0.f;  // q out of range only past the last frame (unused)
    }
  // Frames are independent: blocks of frames on the pool, 4 frames per NEON step.
  constexpr int64_t kBlock = 2048;
  const bool vec = nFft % hop == 0 && nFft <= 64;
  auto block = [&](int64_t blk) {
    const int64_t tEnd = std::min(frames, (blk + 1) * kBlock);
    int64_t t = blk * kBlock;
    if (vec) {
      for (; t + 4 <= tEnd; t += 4) {
        float32x4_t fr[64];
        for (int64_t m = 0; m < nFft; ++m) fr[m] = vld1q_f32(xr + (m % hop) * len + t + m / hop);
        for (int64_t k = 0; k < bins; ++k) {
          const float* cr = tab + k * nFft;
          const float* ci = tab + (bins + k) * nFft;
          float32x4_t re = vdupq_n_f32(0.f), im = re;
          for (int64_t m = 0; m < nFft; ++m) {
            re = vaddq_f32(re, vmulq_n_f32(fr[m], cr[m]));
            im = vaddq_f32(im, vmulq_n_f32(fr[m], ci[m]));
          }
          if (k == 0 || 2 * k == nFft) im = vdupq_n_f32(0.f);  // real-input FFT: exactly +0 at DC / Nyquist
          vst1q_f32(mag + k * frames + t, vsqrtq_f32(vfmaq_f32(vmulq_f32(re, re), im, im)));
          vst1q_f32(ph + k * frames + t, detail::atan2_4(im, re));
        }
      }
    }
    float* frame = detail::scratch(static_cast<size_t>(nFft), 1);
    for (; t < tEnd; ++t) {
      for (int64_t m = 0; m < nFft; ++m) {
        int64_t q = t * hop + m - pad;  // reflect pad
        if (q < 0) q = -q;
        if (q >= s) q = 2 * (s - 1) - q;
        frame[m] = x[q];
      }
      for (int64_t k = 0; k < bins; ++k) {
        const float* cr = tab + k * nFft;
        const float* ci = tab + (bins + k) * nFft;
        float re = 0.f, im = 0.f;
        for (int64_t m = 0; m < nFft; ++m) {
          re += cr[m] * frame[m];
          im += ci[m] * frame[m];
        }
        if (k == 0 || 2 * k == nFft) im = 0.f;
        mag[k * frames + t] = std::hypot(re, im);
        ph[k * frames + t] = std::atan2(im, re);
      }
    }
  };
  const int64_t blocks = (frames + kBlock - 1) / kBlock;
  if (blocks <= 1) {
    for (int64_t b = 0; b < blocks; ++b) block(b);
  } else {
    rt::ThreadPool::global().parallelFor(blocks, block);
  }
}

bool istft_logits_f32(const float* x, float* out, int64_t l, int64_t nFft, int64_t hop) {
  const int64_t bins = nFft / 2 + 1, pad = nFft / 2;
  constexpr int64_t kMaxBins = 64;
  if (bins > kMaxBins) return false;  // Kokoro: n_fft = 20 (the Python rule bounds n_fft)
  // Inverse real DFT (backward norm 1/N) with the synthesis window folded in:
  //   y[m] = w[m]/N * (Re X0 + Re X_{N/2} (-1)^m + 2 sum_{k=1}^{N/2-1} (Re Xk cos - Im Xk sin)(2 pi k m / N))
  float* tab = detail::scratch(static_cast<size_t>(2 * bins * nFft + nFft), 2);
  float* win = tab + 2 * bins * nFft;
  hannPeriodic(win, nFft);
  for (int64_t k = 0; k < bins; ++k) {
    const bool edge = k == 0 || 2 * k == nFft;
    for (int64_t m = 0; m < nFft; ++m) {
      double ang = 2.0 * M_PI * static_cast<double>((k * m) % nFft) / static_cast<double>(nFft);
      double c = (edge ? 1.0 : 2.0) * static_cast<double>(win[m]) / static_cast<double>(nFft);
      tab[k * nFft + m] = static_cast<float>(c * std::cos(ang));
      tab[(bins + k) * nFft + m] = edge ? 0.f : static_cast<float>(-c * std::sin(ang));
    }
  }
  // 1) Per-frame synthesis, 4 frames per NEON step (exp / sin / cos polynomials), into fr[m][t] (frame-minor).
  float* fr = detail::scratch(static_cast<size_t>(nFft * (l + 4)), 1);
  const int64_t ld = l + 4;
  constexpr int64_t kBlock = 2048;
  auto block = [&](int64_t blk) {
    float32x4_t re[kMaxBins], im[kMaxBins];
    const int64_t tEnd = std::min(l, (blk + 1) * kBlock);
    for (int64_t t = blk * kBlock; t < tEnd; t += 4) {
      const int64_t nv = std::min<int64_t>(4, tEnd - t);
      for (int64_t k = 0; k < bins; ++k) {
        float32x4_t lm, lp;
        if (nv == 4) {
          lm = vld1q_f32(x + k * l + t);
          lp = vld1q_f32(x + (bins + k) * l + t);
        } else {
          float a[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
          for (int64_t q = 0; q < nv; ++q) a[q] = x[k * l + t + q], b[q] = x[(bins + k) * l + t + q];
          lm = vld1q_f32(a);
          lp = vld1q_f32(b);
        }
        const float32x4_t mag = detail::exp4(lm);
        const float32x4_t p = detail::sin4Wide(lp);  // |p| <= 1: cos / sin below need no wide reduction
        re[k] = vmulq_f32(mag, detail::cos4(p));
        im[k] = vmulq_f32(mag, detail::sin4(p));
      }
      for (int64_t m = 0; m < nFft; ++m) {
        float32x4_t v = vdupq_n_f32(0.f);
        for (int64_t k = 0; k < bins; ++k)
          v = vaddq_f32(v, vaddq_f32(vmulq_n_f32(re[k], tab[k * nFft + m]), vmulq_n_f32(im[k], tab[(bins + k) * nFft + m])));
        vst1q_f32(fr + m * ld + t, v);  // t + 4 <= l + 4: padded row
      }
    }
  };
  const int64_t blocks = (l + kBlock - 1) / kBlock;
  // 2) Overlap-add as a gather per output sample, frames in ascending order (the serial loop's summation order),
  // divided by the window-square envelope summed the same way; output blocks on the pool.
  const int64_t len = hop * (l - 1);
  constexpr int64_t kOutBlock = 8192;
  auto ola = [&](int64_t blk) {
    const int64_t iEnd = std::min(len, (blk + 1) * kOutBlock);
    for (int64_t i = blk * kOutBlock; i < iEnd; ++i) {
      const int64_t p = pad + i;
      const int64_t tHi = std::min(l - 1, p / hop);
      const int64_t tLo = p >= nFft ? (p - nFft) / hop + 1 : 0;
      float y = 0.f, env = 0.f;
      for (int64_t t = tLo; t <= tHi; ++t) {
        const int64_t m = p - t * hop;
        y += fr[m * ld + t];
        env += win[m] * win[m];
      }
      out[i] = y / env;
    }
  };
  auto& pool = rt::ThreadPool::global();
  if (blocks <= 1) {
    for (int64_t b = 0; b < blocks; ++b) block(b);
  } else {
    pool.parallelFor(blocks, block);
  }
  const int64_t oblocks = (len + kOutBlock - 1) / kOutBlock;
  if (oblocks <= 1) {
    for (int64_t b = 0; b < oblocks; ++b) ola(b);
  } else {
    pool.parallelFor(oblocks, ola);
  }
  return true;
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

const float* F(const KernelArgs& a, int i) { return static_cast<const float*>(a.ptrs[i]); }
float* O(const KernelArgs& a, int i) { return static_cast<float*>(a.ptrs[i]); }

// ints: f0(n), noise(300n, 9), w(9), b(1), out(300n)
void runHarmonicSource(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::harmonic_source_f32(F(a, 0), F(a, 1), F(a, 2), F(a, 3), O(a, 4), a.ints[0]);
}

// operands f0(2F), w(9), b(1), seed (i64 host scalar); out(600F). ints: f0, w, b, out shapes.
void runHarmonicSourceRng(ExecCtx&, const KernelArgs& a) {
  int64_t seed;
  std::memcpy(&seed, a.ptrs[3], sizeof seed);
  kernels::cpu::harmonic_source_rng_f32(F(a, 0), static_cast<uint64_t>(seed), F(a, 1), F(a, 2), O(a, 4), a.ints[0]);
}

// ints: x(S), out(N+2, S/hop+1), hop, n_fft
void runStft(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::stft_magphase_f32(F(a, 0), O(a, 1), a.ints[0], a.ints[4], a.ints[3]);
}

// ints: x(N+2, L), out(hop(L-1)), hop, n_fft
void runIstft(ExecCtx& ctx, const KernelArgs& a) {
  if (!kernels::cpu::istft_logits_f32(F(a, 0), O(a, 1), a.ints[1], a.ints[4], a.ints[3]))
    ctx.error = "istft_logits_f32_cpu: n_fft too large (at most 126)";
}

}  // namespace

void registerCpuKokoroSignalKernels() {
  registerKernel({"harmonic_source_f32_cpu", Device::Cpu, nullptr, runHarmonicSource, nullptr});
  registerKernel({"harmonic_source_rng_f32_cpu", Device::Cpu, nullptr, runHarmonicSourceRng, nullptr});
  registerKernel({"stft_magphase_f32_cpu", Device::Cpu, nullptr, runStft, nullptr});
  registerKernel({"istft_logits_f32_cpu", Device::Cpu, nullptr, runIstft, nullptr});
}

}  // namespace lk::rt
