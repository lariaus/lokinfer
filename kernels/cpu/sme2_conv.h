// SME2 kernels for the Kokoro CPU kernels (sme2_conv.cpp; the conv GEMMs use the shared engine sme2_gemm.h). Compiled
// with +sme2 on every Apple target; available() says whether this CPU runs them (hw.optional.arm.FEAT_SME2: the
// iPhone 18 Pro Max test phone yes, the M2 Max no); callers fall back to Accelerate otherwise.
#pragma once

#include <cstdint>

namespace lk::kernels::cpu::sme2 {

bool available();

// y[0:n] += wt^T x: wt [k, ldw] row-major (rows of n used), n a multiple of 64; 4 partial sums per output (j mod 4)
// combined (p0 + p1) + (p2 + p3), then added to y (the LSTM step GEMV, kokoro_lstm.cpp).
void gemvT(const float* wt, int64_t ldw, int64_t k, int64_t n, const float* x, float* y);

}  // namespace lk::kernels::cpu::sme2
