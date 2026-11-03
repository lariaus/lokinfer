// Shared SME2 GEMM engine (sme2_gemm.cpp): FMOPA outer products into the four 16×16 f32 ZA tiles (one 32×32 block
// of C per pass over K). Compiled with +sme2 on every Apple target; sme2::available() (sme2_conv.h) says whether this
// CPU runs it (A19 / M4 and later). Callers fall back to Accelerate otherwise (the M2 Max never gets here).
//
// Both operands of FMOPA are k-major (for each k: 32 values along M, 32 along N):
//  - A (the "row" operand, C's rows) is always PACKED: panels [ceil(m / 32)][k][32] (zero rows past m). Pack weights
//    at prepare time (Kokoro: W [Co, Ci·K]) or activations per call (vision: im2col rows [P, K]).
//  - B (the "column" operand) is read in place: element (kk, j) at
//        b + (j / 32) · panel + (kk / taps) · ldb + (kk % taps) · dil + j % 32
//    · plain row-major [k, n]:                taps = 1, ldb = row stride, panel = 32 (default)
//    · panels from packPanelsKMajor:          taps = 1, ldb = 32, panel = 32 · k
//    · channel-first conv windows (Kokoro):   taps = K, ldb = window row stride, dil = dilation, panel = 32
//    · any row gather (planar 2-D convs):     rowOff[kk] replaces (kk / taps) · ldb + (kk % taps) · dil
// Callers split work over threads themselves: rows by 32-row panel offsets of A (ap + (m0 / 32) · 32 · k, m0 a multiple
// of 32), columns by offsetting b / c / bias.
#pragma once

#include <cstddef>
#include <cstdint>

#include "sme2_conv.h"  // sme2::available()

namespace lk::kernels::cpu::sme2 {

enum class Act : int { None = 0, Relu = 1, Relu6 = 2, Leaky = 3 };  // Leaky: slope `leaky`

struct BOperand {
  const float* b;
  int64_t ldb;
  int64_t panel = 32;
  int64_t taps = 1;
  int64_t dil = 0;
  const int64_t* rowOff = nullptr;  // k entries (floats); when set, ldb / taps / dil are ignored
};

struct Epilogue {
  const float* bias = nullptr;  // per column (C's n), or per row when biasPerRow
  bool biasPerRow = false;
  const float* res = nullptr;   // C += res (same [m, n] indexing, row stride ldres) before act
  int64_t ldres = 0;
  bool accumulate = false;      // C += A·B (the old C is added before bias/res/act)
  Act act = Act::None;
  float leaky = 0.2f;
  const float* colMask = nullptr;  // per column of C (0 / 1), multiplied in after the activation
};

// Floats needed by packPanels / packPanelsKMajor for `rows` rows (or columns) of length k.
inline size_t packedSize(int64_t rows, int64_t k) { return static_cast<size_t>((rows + 31) / 32 * 32 * k); }

// src [rows, k] row-major (row stride ld) -> dst [ceil(rows / 32)][k][32] (the transpose; zero past rows).
// Streaming-mode ZA transposes on SME2 CPUs, plain loops elsewhere.
void packPanels(const float* src, int64_t ld, int64_t rows, int64_t k, float* dst);

// Gather packing (implicit im2col): row r's k = segs · seglen values are segs runs of seglen floats at
// ptrs[r · segs + s] (s < segs); point padding taps at a zero run. Efficient when seglen is a multiple of 16.
void packPanelsGather(const float* const* ptrs, int64_t rows, int64_t segs, int64_t seglen, float* dst);

// src [k, cols] k-major (row stride ld) -> dst [ceil(cols / 32)][k][32] (no transpose; zero past cols).
void packPanelsKMajor(const float* src, int64_t ld, int64_t k, int64_t cols, float* dst);

// C[m, n] (row stride ldc) = epi(A · B). ap from packPanels (m rows, k columns).
void gemm(const float* ap, int64_t m, int64_t k, const BOperand& b, int64_t n, float* c, int64_t ldc,
          const Epilogue& e = {});

// Convenience: A [m, k] row-major (lda) packed into this thread's scratch, then gemm.
void gemmA(const float* a, int64_t lda, int64_t m, int64_t k, const BOperand& b, int64_t n, float* c, int64_t ldc,
           const Epilogue& e = {});

}  // namespace lk::kernels::cpu::sme2
