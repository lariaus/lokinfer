// Typed CPU kernel functions: the layer that's benchmarked directly
// (design-runtime-and-kernels.md §1). No allocation, no exceptions.
#pragma once

#include <cstdint>

namespace lk::kernels::cpu {

void add_f32(const float* a, const float* b, float* out, int64_t n);
void mul_f32(const float* a, const float* b, float* out, int64_t n);

// out[M,N] = a[M,K] @ b[K,N], row-major, hand-written NEON.
void gemm_f32(const float* a, const float* b, float* out, int64_t m, int64_t n, int64_t k);
// Same contract through Accelerate (cblas_sgemm, AMX on Apple silicon).
void gemm_f32_accelerate(const float* a, const float* b, float* out, int64_t m, int64_t n, int64_t k);

// Fused epilogues: out = a @ b + bias[N] (then ReLU).
void gemm_bias_f32(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n, int64_t k);
void gemm_bias_relu_f32(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n,
                        int64_t k);
void gemm_bias_f32_accelerate(const float* a, const float* b, const float* bias, float* out, int64_t m, int64_t n,
                              int64_t k);
void gemm_bias_relu_f32_accelerate(const float* a, const float* b, const float* bias, float* out, int64_t m,
                                   int64_t n, int64_t k);

// out[C, R] = in[R, C]^T
void transpose_f32(const float* in, float* out, int64_t rows, int64_t cols);

}  // namespace lk::kernels::cpu
