#include <arm_neon.h>

#include "lokinfer/kernels/cpu.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

namespace {

template <typename Op>
inline void binary_f32(const float* a, const float* b, float* out, int64_t n, Op op) {
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t a0 = vld1q_f32(a + i), a1 = vld1q_f32(a + i + 4), a2 = vld1q_f32(a + i + 8), a3 = vld1q_f32(a + i + 12);
    float32x4_t b0 = vld1q_f32(b + i), b1 = vld1q_f32(b + i + 4), b2 = vld1q_f32(b + i + 8), b3 = vld1q_f32(b + i + 12);
    vst1q_f32(out + i, op(a0, b0));
    vst1q_f32(out + i + 4, op(a1, b1));
    vst1q_f32(out + i + 8, op(a2, b2));
    vst1q_f32(out + i + 12, op(a3, b3));
  }
  for (; i + 4 <= n; i += 4) vst1q_f32(out + i, op(vld1q_f32(a + i), vld1q_f32(b + i)));
  for (; i < n; ++i) out[i] = vgetq_lane_f32(op(vdupq_n_f32(a[i]), vdupq_n_f32(b[i])), 0);
}

}  // namespace

void add_f32(const float* a, const float* b, float* out, int64_t n) {
  binary_f32(a, b, out, n, [](float32x4_t x, float32x4_t y) { return vaddq_f32(x, y); });
}

void mul_f32(const float* a, const float* b, float* out, int64_t n) {
  binary_f32(a, b, out, n, [](float32x4_t x, float32x4_t y) { return vmulq_f32(x, y); });
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

// Same-shape binary op: n = product of operand 0's dims (ints[0..rank)).
int64_t numel(const KernelArgs& a, int64_t rank) {
  int64_t n = 1;
  for (int64_t i = 0; i < rank; ++i) n *= a.ints[i];
  return n;
}

// Rank = numInts / 3 (two operands + one out, same shape, no int attrs).
template <void (*Fn)(const float*, const float*, float*, int64_t)>
void runBinary(ExecCtx&, const KernelArgs& a) {
  Fn(static_cast<const float*>(a.ptrs[0]), static_cast<const float*>(a.ptrs[1]), static_cast<float*>(a.ptrs[2]),
     numel(a, a.numInts / 3));
}

}  // namespace

void registerCpuElementwiseKernels() {
  registerKernel({"add_f32_cpu", Device::Cpu, nullptr, runBinary<kernels::cpu::add_f32>, nullptr});
  registerKernel({"mul_f32_cpu", Device::Cpu, nullptr, runBinary<kernels::cpu::mul_f32>, nullptr});
}

}  // namespace lk::rt
