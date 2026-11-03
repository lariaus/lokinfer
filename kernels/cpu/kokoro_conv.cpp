// 1-D convolution family (Kokoro, channel-first [C, L]): conv1d (tiled im2col + Accelerate sgemm),
// conv_transpose1d (sgemm + col2im), depthwise conv_transpose1d, nearest upsample, reflect pad.
#include <algorithm>
#include <atomic>
#include <cstring>

#include "kokoro_common.h"
#include "sme2_gemm.h"
#include "lokinfer/kernels/cpu_kokoro.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

namespace {

// Column-tile budget (floats) for im2col / col2im temporaries: 8 MB.
constexpr int64_t kTileBudget = int64_t{1} << 21;

int64_t tileWidth(int64_t rowsPerCol, int64_t total) {
  int64_t tw = kTileBudget / std::max<int64_t>(rowsPerCol, 1);
  tw = std::max<int64_t>(tw, 64);
  return std::min(tw, total);
}

// out[c, :] = bias[c] (or 0)
void fillBias(float* out, const float* bias, int64_t c, int64_t l) {
  for (int64_t ch = 0; ch < c; ++ch) std::fill(out + ch * l, out + (ch + 1) * l, bias ? bias[ch] : 0.f);
}

}  // namespace

void conv1d_f32(const float* x, const float* w, const float* bias, float* out, int64_t ci, int64_t l, int64_t co,
                int64_t k, int64_t stride, int64_t padding, int64_t dilation) {
  const int64_t lo = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  if (lo <= 0) return;
  if (bias) fillBias(out, bias, co, lo);
  const float beta = bias ? 1.f : 0.f;
  if (k == 1 && stride == 1 && padding == 0) {  // pointwise: plain GEMM, no im2col
    detail::sgemm(false, false, co, lo, ci, 1.f, w, ci, x, l, beta, out, lo);
    return;
  }
  const int64_t ck = ci * k;
  const int64_t tw = tileWidth(ck, lo);
  float* col = detail::scratch(static_cast<size_t>(ck * tw));
  for (int64_t j0 = 0; j0 < lo; j0 += tw) {
    const int64_t n = std::min(tw, lo - j0);
    for (int64_t c = 0; c < ci; ++c) {
      const float* xr = x + c * l;
      for (int64_t kk = 0; kk < k; ++kk) {
        float* row = col + (c * k + kk) * n;
        const int64_t off = kk * dilation - padding;  // input pos = (j0 + jj) * stride + off
        if (stride == 1) {
          // valid jj: 0 <= j0 + jj + off < l
          int64_t a = std::clamp<int64_t>(-off - j0, 0, n), b = std::clamp<int64_t>(l - off - j0, 0, n);
          if (b < a) b = a;
          std::fill(row, row + a, 0.f);
          if (b > a) std::memcpy(row + a, xr + j0 + a + off, static_cast<size_t>(b - a) * sizeof(float));
          std::fill(row + b, row + n, 0.f);
        } else {
          for (int64_t jj = 0; jj < n; ++jj) {
            int64_t p = (j0 + jj) * stride + off;
            row[jj] = (p >= 0 && p < l) ? xr[p] : 0.f;
          }
        }
      }
    }
    detail::sgemm(false, false, co, n, ck, 1.f, w, ck, col, n, beta, out + j0, lo);
  }
}

void conv_transpose1d_f32(const float* x, const float* w, const float* bias, float* out, int64_t ci, int64_t l,
                          int64_t co, int64_t k, int64_t stride, int64_t padding, int64_t outputPadding) {
  const int64_t lo = (l - 1) * stride - 2 * padding + k + outputPadding;
  if (lo <= 0) return;
  fillBias(out, bias, co, lo);
  const int64_t ck = co * k;
  const int64_t tl = tileWidth(ck, l);
  float* y = detail::scratch(static_cast<size_t>(ck * tl));
  for (int64_t j0 = 0; j0 < l; j0 += tl) {
    const int64_t n = std::min(tl, l - j0);
    // y[co*k + kk, jj] = sum_ci w[ci, co, kk] * x[ci, j0 + jj]   (w viewed as [ci, co*k])
    detail::sgemm(true, false, ck, n, ci, 1.f, w, ck, x + j0, l, 0.f, y, n);
    for (int64_t c = 0; c < co; ++c) {
      float* o = out + c * lo;
      for (int64_t kk = 0; kk < k; ++kk) {
        const float* yr = y + (c * k + kk) * n;
        const int64_t off = j0 * stride + kk - padding;  // out pos = jj * stride + off
        // valid jj: 0 <= jj * stride + off < lo
        int64_t a = off >= 0 ? 0 : (-off + stride - 1) / stride;
        int64_t b = lo - off <= 0 ? 0 : std::min(n, (lo - off + stride - 1) / stride);
        for (int64_t jj = a; jj < b; ++jj) o[jj * stride + off] += yr[jj];
      }
    }
  }
}

void conv_transpose1d_dw_f32(const float* x, const float* w, const float* bias, float* out, int64_t c, int64_t l,
                             int64_t k, int64_t stride, int64_t padding, int64_t outputPadding) {
  const int64_t lo = (l - 1) * stride - 2 * padding + k + outputPadding;
  if (lo <= 0) return;
  // Gather form per output phase: out[o] = bias + Σ_j x[j] · w[o + p - j·s] over j ascending (the scatter loop's
  // accumulation order, so results are unchanged), as one pass per output instead of a strided scatter; channels
  // on the pool (Kokoro's upsample blocks: 0.75 ms -> 0.06 ms on the Mac, app59f).
  auto chan = [&](int64_t ch) {
    const float* xr = x + ch * l;
    const float* wr = w + ch * k;
    float* o = out + ch * lo;
    const float b = bias ? bias[ch] : 0.f;
    for (int64_t q = 0; q < lo; ++q) {
      const int64_t t = q + padding;  // = j·s + kk
      int64_t jHi = std::min(l - 1, t / stride);
      int64_t jLo = t - (k - 1) <= 0 ? 0 : (t - (k - 1) + stride - 1) / stride;
      float v = b;
      for (int64_t j = jLo; j <= jHi; ++j) v += xr[j] * wr[t - j * stride];
      o[q] = v;
    }
  };
  if (static_cast<double>(c) * static_cast<double>(lo) * static_cast<double>(k) < 2e5) {
    for (int64_t ch = 0; ch < c; ++ch) chan(ch);
    return;
  }
  const int64_t threads = rt::ThreadPool::global().threads();
  const int64_t cpt = std::max<int64_t>(1, (c + 2 * threads - 1) / (2 * threads));
  rt::ThreadPool::global().parallelFor((c + cpt - 1) / cpt, [&](int64_t t) {
    for (int64_t ch = t * cpt; ch < std::min(c, (t + 1) * cpt); ++ch) chan(ch);
  });
}

// Polyphase ConvTranspose1d (Kokoro generator ups[0] K20 s10, ups[1] K12 s6), leaky prologue, + residual epilogue,
// optional 1-sample left reflect pad between the two (ups[1]'s ReflectionPad1d((1, 0))):
//   out[co, j·s + r] = Σ_c Σ_{tt < taps} Wr[co, c, tt] · act(x)[c, j + q_r - (taps-1) + tt]
// with r' = (r + p) mod s, q_r = (r + p) div s, Wr[co, c, tt] = w[c, co, r' + (taps-1-tt)·s] (K = taps·s,
// 2p = (taps-1)·s, so Lo = L·s and every phase has L outputs). Rows (r, co) of phases with the same q_r form one
// GEMM over the shared leaky window: no col2im scatter, no zero taps, same FLOPs as the GEMM + col2im form.
// wp = ops.convt_poly_weights_cpu: 32-row panels [ceil(rows_q / 32)][Ci·taps][32] of group q = 0 then q = 1,
// reduction index c·taps + tt (sme2::packPanels of the [rows, Ci·taps] matrix). Phases: 1) leaky window [Ci, L + taps] on the pool,
// 2) the GEMM panels -> T [s·Co, L] (SME2: P cores only; else single-threaded Accelerate per task), 3) interleave
// + bias + residual (+ reflect) on the pool.
void conv_transpose1d_poly_f32(const float* x, const float* wp, const float* bias, const float* res, float* out,
                               int64_t ci, int64_t l, int64_t co, int64_t stride, int64_t padding, int64_t taps,
                               int64_t padLeft, float slope, const float* x2, const float* x3, float inScale) {
  const int64_t s = stride, lp = l + taps + 1, ck = ci * taps;
  const int64_t rows0 = (s - padding) * co, rows1 = padding * co;  // phases r < s - p have q_r = 0
  const int64_t pan0 = (rows0 + 31) / 32, pan1 = (rows1 + 31) / 32;
  float* xw = detail::scratch(static_cast<size_t>(ci * lp), 7);
  float* tb = detail::scratch(static_cast<size_t>(s * co * l), 6);
  auto& pool = rt::ThreadPool::global();
  const int64_t threads = pool.threads();
  // 1) window: xw[c, i] = leaky(x[c, i - (taps - 1)]), zeros outside.
  const int64_t cpt = std::max<int64_t>(1, (ci + 2 * threads - 1) / (2 * threads));
  pool.parallelFor((ci + cpt - 1) / cpt, [&](int64_t t) {
    for (int64_t c = t * cpt; c < std::min(ci, (t + 1) * cpt); ++c) {
      float* row = xw + c * lp;
      const float* xr = x + c * l;
      std::fill(row, row + taps - 1, 0.f);
      float* dst = row + taps - 1;
      const float32x4_t vs = vdupq_n_f32(slope), z = vdupq_n_f32(0.f), vin = vdupq_n_f32(inScale);
      int64_t i = 0;
      if (x2) {  // ((x + x2) + x3) * inScale first (add3_scale_f32_cpu's op order)
        const float* r2 = x2 + c * l;
        const float* r3 = x3 + c * l;
        for (; i + 4 <= l; i += 4) {
          const float32x4_t v =
              vmulq_f32(vaddq_f32(vaddq_f32(vld1q_f32(xr + i), vld1q_f32(r2 + i)), vld1q_f32(r3 + i)), vin);
          vst1q_f32(dst + i, vbslq_f32(vcgeq_f32(v, z), v, vmulq_f32(v, vs)));
        }
        for (; i < l; ++i) {
          const float v = (xr[i] + r2[i] + r3[i]) * inScale;
          dst[i] = v >= 0.f ? v : v * slope;
        }
      }
      for (; i + 4 <= l; i += 4) {
        const float32x4_t v = vld1q_f32(xr + i);
        vst1q_f32(dst + i, vbslq_f32(vcgeq_f32(v, z), v, vmulq_f32(v, vs)));
      }
      for (; i < l; ++i) dst[i] = xr[i] >= 0.f ? xr[i] : xr[i] * slope;
      std::fill(dst + l, row + lp, 0.f);
    }
  });
  // 2) GEMMs: task = (panel, column chunk).
  const bool sme = sme2::available();
  const int64_t chunk = sme ? 256 : 512, nch = (l + chunk - 1) / chunk, npan = pan0 + pan1;
  auto gemmTask = [&](int64_t task) {
    const int64_t pi = task / nch, j0 = (task % nch) * chunk, n = std::min(chunk, l - j0);
    const int64_t q = pi < pan0 ? 0 : 1;
    const int64_t r0 = q == 0 ? pi * 32 : rows0 + (pi - pan0) * 32;
    const int64_t rows = std::min<int64_t>(32, (q == 0 ? rows0 : rows0 + rows1) - r0);
    const float* a = wp + pi * ck * 32;
    const float* b = xw + q + j0;
    float* o = tb + r0 * l + j0;
    if (sme) {
      sme2::gemm(a, rows, ck, sme2::BOperand{b, lp, 32, taps, 1}, n, o, l);
    } else {
      // A_tt^T = panel rows c·taps + tt (lda = taps·32), B_tt = the window shifted by tt, read in place.
      for (int64_t tt = 0; tt < taps; ++tt)
        detail::sgemm(true, false, rows, n, ci, 1.f, a + tt * 32, taps * 32, b + tt, lp, tt ? 1.f : 0.f, o, l);
    }
  };
  const int64_t ntasks = npan * nch;
  if (sme && detail::perfCoreSplit() > 0) {
    // One SME unit per cluster; the efficiency cluster's runs these ~9x slower (iphone-cpu-sme2.md): P cores only.
    std::atomic<int64_t> next{0};
    detail::parallelForSt(threads, [&](int64_t) {
      if (!detail::onPerfCore()) return;
      for (int64_t t; (t = next.fetch_add(1, std::memory_order_relaxed)) < ntasks;) gemmTask(t);
    });
    for (int64_t t; (t = next.fetch_add(1, std::memory_order_relaxed)) < ntasks;) gemmTask(t);  // no P core ran
  } else {
    detail::parallelForSt(ntasks, gemmTask);
  }
  // 3) out[co, padLeft + j·s + r] = T[r·Co + co, j] + bias[co] + res[co, ...]; reflect: out[co, 0] = unpadded[1].
  const int64_t lo = l * s + padLeft;
  const int64_t opt = std::max<int64_t>(1, (co + 2 * threads - 1) / (2 * threads));
  pool.parallelFor((co + opt - 1) / opt, [&](int64_t t) {
    for (int64_t c = t * opt; c < std::min(co, (t + 1) * opt); ++c) {
      const float bc = bias[c];
      float* orow = out + c * lo + padLeft;
      const float* rrow = res + c * lo + padLeft;
      const float* trow = tb + c * l;
      const int64_t rs = co * l;  // phase stride in T
      for (int64_t j = 0; j < l; ++j) {
        float* o = orow + j * s;
        const float* rr = rrow + j * s;
        for (int64_t r = 0; r < s; ++r) o[r] = (trow[r * rs + j] + bc) + rr[r];
      }
      if (padLeft) out[c * lo] = (trow[1 * rs] + bc) + res[c * lo];  // unpadded position 1 = (j 0, r 1)
    }
  });
}

void upsample_nearest1d_f32(const float* x, float* out, int64_t c, int64_t l, int64_t scale) {
  for (int64_t ch = 0; ch < c; ++ch) {
    const float* xr = x + ch * l;
    float* o = out + ch * l * scale;
    if (scale == 2) {
      int64_t i = 0;
      for (; i + 4 <= l; i += 4) {
        float32x4_t v = vld1q_f32(xr + i);
        float32x4x2_t vv = {{v, v}};
        vst2q_f32(o + 2 * i, vv);
      }
      for (; i < l; ++i) o[2 * i] = o[2 * i + 1] = xr[i];
    } else {
      for (int64_t i = 0; i < l; ++i) std::fill(o + i * scale, o + (i + 1) * scale, xr[i]);
    }
  }
}

void pad_reflect1d_f32(const float* x, float* out, int64_t c, int64_t l, int64_t left, int64_t right) {
  const int64_t lo = left + l + right;
  for (int64_t ch = 0; ch < c; ++ch) {
    const float* xr = x + ch * l;
    float* o = out + ch * lo;
    for (int64_t i = 0; i < left; ++i) o[i] = xr[left - i];
    std::memcpy(o + left, xr, static_cast<size_t>(l) * sizeof(float));
    for (int64_t i = 0; i < right; ++i) o[left + l + i] = xr[l - 2 - i];
  }
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

const float* F(const KernelArgs& a, int i) { return static_cast<const float*>(a.ptrs[i]); }
float* O(const KernelArgs& a, int i) { return static_cast<float*>(a.ptrs[i]); }

// ints: x(Ci, L), w(Co, Ci, K), b(Co), out(Co, Lo), dilation, padding, stride
void runConv1d(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::conv1d_f32(F(a, 0), F(a, 1), F(a, 2), O(a, 3), n[0], n[1], n[2], n[4], n[10], n[9], n[8]);
}

// ints: x(Ci, L), w(Co, Ci, K), out(Co, Lo), dilation, padding, stride
void runConv1dNoBias(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::conv1d_f32(F(a, 0), F(a, 1), nullptr, O(a, 2), n[0], n[1], n[2], n[4], n[9], n[8], n[7]);
}

// ints: x(Ci, L), w(Ci, Co, K), b(Co), out(Co, Lo), output_padding, padding, stride
void runConvTranspose1d(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::conv_transpose1d_f32(F(a, 0), F(a, 1), F(a, 2), O(a, 3), n[0], n[1], n[3], n[4], n[10], n[9], n[8]);
}

// ints: x(Ci, L), wp(P, Ci·taps, 32), b(Co), res(Co, Lo), out(Co, Lo), pad_left, padding, stride, taps; floats: slope
void runConvTranspose1dPoly(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::conv_transpose1d_poly_f32(F(a, 0), F(a, 1), F(a, 2), F(a, 3), O(a, 4), n[0], n[1], n[5], n[12], n[11],
                                          n[13], n[10], static_cast<float>(a.floats[0]), nullptr, nullptr, 1.f);
}

// conv_transpose1d_poly_mean3_f32_cpu(x, x2, x3, wp, b, res): the input is ((x + x2) + x3) * in_scale.
// ints: x, x2, x3 (Ci, L), wp(3), b(1), res(2), out(2), pad_left, padding, stride, taps; floats: in_scale, slope
void runConvTranspose1dPolyMean3(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::conv_transpose1d_poly_f32(F(a, 0), F(a, 3), F(a, 4), F(a, 5), O(a, 6), n[0], n[1], n[9], n[16], n[15],
                                          n[17], n[14], static_cast<float>(a.floats[1]), F(a, 1), F(a, 2),
                                          static_cast<float>(a.floats[0]));
}

// ints: x(C, L), w(C, 1, K), b(C), out(C, Lo), output_padding, padding, stride
void runConvTranspose1dDw(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::conv_transpose1d_dw_f32(F(a, 0), F(a, 1), F(a, 2), O(a, 3), n[0], n[1], n[4], n[10], n[9], n[8]);
}

// ints: x(C, L), out(C, L*s), scale
void runUpsampleNearest1d(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::upsample_nearest1d_f32(F(a, 0), O(a, 1), a.ints[0], a.ints[1], a.ints[4]);
}

// ints: x(C, L), out(C, Lo), left, right
void runPadReflect1d(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::pad_reflect1d_f32(F(a, 0), O(a, 1), a.ints[0], a.ints[1], a.ints[4], a.ints[5]);
}

}  // namespace

void registerCpuKokoroConvKernels() {
  registerKernel({"conv1d_f32_cpu", Device::Cpu, nullptr, runConv1d, nullptr});
  registerKernel({"conv1d_nobias_f32_cpu", Device::Cpu, nullptr, runConv1dNoBias, nullptr});
  registerKernel({"conv_transpose1d_f32_cpu", Device::Cpu, nullptr, runConvTranspose1d, nullptr});
  registerKernel({"conv_transpose1d_poly_f32_cpu", Device::Cpu, nullptr, runConvTranspose1dPoly, nullptr});
  registerKernel({"conv_transpose1d_poly_mean3_f32_cpu", Device::Cpu, nullptr, runConvTranspose1dPolyMean3, nullptr});
  registerKernel({"conv_transpose1d_dw_f32_cpu", Device::Cpu, nullptr, runConvTranspose1dDw, nullptr});
  registerKernel({"upsample_nearest1d_f32_cpu", Device::Cpu, nullptr, runUpsampleNearest1d, nullptr});
  registerKernel({"pad_reflect1d_f32_cpu", Device::Cpu, nullptr, runPadReflect1d, nullptr});
}

}  // namespace lk::rt
