// Normalization kernels (Kokoro): layernorm, adalayernorm, adain1d. Two-pass stats (kokoro_common.h).
#include <cmath>

#include "kokoro_common.h"
#include "lokinfer/kernels/cpu_kokoro.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

namespace {

// out = ((x - mean) * rstd) * g + b per element; g/b per column (g == nullptr: g = 1 + gb[:C], b = gb[C:]).
void normRows(const float* x, const float* gamma, const float* beta, bool ada, float* out, int64_t rows, int64_t cols,
              float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float* o = out + r * cols;
    float mean, var;
    detail::meanVar(xr, cols, mean, var);
    float rstd = 1.0f / std::sqrt(var + eps);
    float32x4_t vm = vdupq_n_f32(mean), vr = vdupq_n_f32(rstd), one = vdupq_n_f32(1.f);
    int64_t c = 0;
    for (; c + 4 <= cols; c += 4) {
      float32x4_t n = vmulq_f32(vsubq_f32(vld1q_f32(xr + c), vm), vr);
      float32x4_t g = ada ? vaddq_f32(one, vld1q_f32(gamma + c)) : vld1q_f32(gamma + c);
      float32x4_t b = ada ? vld1q_f32(gamma + cols + c) : vld1q_f32(beta + c);
      vst1q_f32(o + c, vfmaq_f32(b, n, g));
    }
    for (; c < cols; ++c) {
      float n = (xr[c] - mean) * rstd;
      float g = ada ? 1.f + gamma[c] : gamma[c];
      float b = ada ? gamma[cols + c] : beta[c];
      o[c] = n * g + b;
    }
  }
}

}  // namespace

void layernorm_f32(const float* x, const float* gamma, const float* beta, float* out, int64_t rows, int64_t cols,
                   float eps) {
  normRows(x, gamma, beta, false, out, rows, cols, eps);
}

void adalayernorm_f32(const float* x, const float* gb, float* out, int64_t rows, int64_t cols, float eps) {
  normRows(x, gb, nullptr, true, out, rows, cols, eps);
}

void adain1d_f32(const float* x, const float* gb, float* out, int64_t c, int64_t l, float eps) {
  for (int64_t ch = 0; ch < c; ++ch) {
    const float* xr = x + ch * l;
    float mean, var;
    detail::meanVar(xr, l, mean, var);
    float rstd = 1.0f / std::sqrt(var + eps);
    // (1 + g) * (x - mean) * rstd + b  ==  (x - mean) * a + b
    detail::affineCentered(xr, out + ch * l, l, mean, (1.f + gb[ch]) * rstd, gb[c + ch]);
  }
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

const float* F(const KernelArgs& a, int i) { return static_cast<const float*>(a.ptrs[i]); }
float* O(const KernelArgs& a, int i) { return static_cast<float*>(a.ptrs[i]); }

// ints: x(R, C), gamma(C), beta(C), out(R, C); floats: eps
void runLayerNorm(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::layernorm_f32(F(a, 0), F(a, 1), F(a, 2), O(a, 3), a.ints[0], a.ints[1], static_cast<float>(a.floats[0]));
}

// ints: x(R, C), gb(2C), out(R, C); floats: eps
void runAdaLayerNorm(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::adalayernorm_f32(F(a, 0), F(a, 1), O(a, 2), a.ints[0], a.ints[1], static_cast<float>(a.floats[0]));
}

// ints: x(C, L), gb(2C), out(C, L); floats: eps
void runAdain1d(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::adain1d_f32(F(a, 0), F(a, 1), O(a, 2), a.ints[0], a.ints[1], static_cast<float>(a.floats[0]));
}

}  // namespace

void registerCpuKokoroNormKernels() {
  registerKernel({"layernorm_f32_cpu", Device::Cpu, nullptr, runLayerNorm, nullptr});
  registerKernel({"adalayernorm_f32_cpu", Device::Cpu, nullptr, runAdaLayerNorm, nullptr});
  registerKernel({"adain1d_f32_cpu", Device::Cpu, nullptr, runAdain1d, nullptr});
}

}  // namespace lk::rt
