// Bidirectional 1-layer LSTM (Kokoro): one sgemm projects every timestep of both directions
// (x @ w_ih^T + b), then a sequential recurrence per direction with one sgemv per step. The two
// directions run concurrently on the pool (each keeps its 4H x H w_hh block in its core's cache);
// gate nonlinearities are vectorized (vvexpf / vvtanhf over the step's 4H gates).
// Tried 2026-11-01: splitting each direction across 2 cores by hidden unit with a per-step spin barrier — 8.1 vs
// 10.4 us/step in a standalone microbench, but slower inside the kernel (T=147 1.52 -> 1.88 ms: 4 sgemv calls per
// step + pool hand-off), so not kept.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "kokoro_common.h"
#include "sme2_conv.h"
#include "lokinfer/kernels/cpu_kokoro.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

namespace {

// In-place sigmoid of n values: 1 / (1 + exp(-v)).
void sigmoidInPlace(float* v, int n) {
  for (int i = 0; i < n; ++i) v[i] = -v[i];
  vvexpf(v, v, &n);
  int i = 0;
  const float32x4_t one = vdupq_n_f32(1.f);
  for (; i + 4 <= n; i += 4) vst1q_f32(v + i, vdivq_f32(one, vaddq_f32(one, vld1q_f32(v + i))));
  for (; i < n; ++i) v[i] = 1.f / (1.f + v[i]);
}

// y[0:n] += Σ_j wt[j·ldw + i] · x[j] (j < k) on NEON with 4 partial sums per output (partial j mod 4, fused
// multiply-adds in j order), combined as (p0 + p1) + (p2 + p3), then added to y. sme2::gemvT computes exactly the same
// operations (bit-identical), so this order is what the SNR tests validate on the Mac. One sequential accumulator over
// k (the old SME2 kernel; also sgemv on the transposed matrix) moved the chaotic F0 path: app47m 37.8 -> 36.1 dB vs
// the f64 golden (limit 37.5); 2 partials: 36.0; 4 partials: 37.9 and all 26 free-running cases pass (2026-11-02).
void gemvT4(const float* wt, int64_t ldw, int64_t k, int64_t n, const float* x, float* y) {
  for (int64_t i0 = 0; i0 < n; i0 += 16) {  // n % 16 == 0
    float32x4_t acc[4][4];
    for (auto& a : acc)
      for (auto& v : a) v = vdupq_n_f32(0.f);
    const float* w = wt + i0;
    int64_t j = 0;
    for (; j + 4 <= k; j += 4, w += 4 * ldw) {
      for (int p = 0; p < 4; ++p) {
        const float xv = x[j + p];
        const float* r = w + p * ldw;
        acc[p][0] = vfmaq_n_f32(acc[p][0], vld1q_f32(r), xv);
        acc[p][1] = vfmaq_n_f32(acc[p][1], vld1q_f32(r + 4), xv);
        acc[p][2] = vfmaq_n_f32(acc[p][2], vld1q_f32(r + 8), xv);
        acc[p][3] = vfmaq_n_f32(acc[p][3], vld1q_f32(r + 12), xv);
      }
    }
    for (int p = 0; j < k; ++j, ++p, w += ldw)  // k % 4 tail: partial j mod 4 as in sme2::gemvT
      for (int q = 0; q < 4; ++q) acc[p][q] = vfmaq_n_f32(acc[p][q], vld1q_f32(w + 4 * q), x[j]);
    for (int q = 0; q < 4; ++q) {
      const float32x4_t s = vaddq_f32(vaddq_f32(acc[0][q], acc[1][q]), vaddq_f32(acc[2][q], acc[3][q]));
      vst1q_f32(y + i0 + 4 * q, vaddq_f32(vld1q_f32(y + i0 + 4 * q), s));
    }
  }
}

// Gates of one step: gates[0:4H] += w_hh h. whhT (transposed layout, lstm_bidir_t): this direction's [H, 4H] column
// block of w_hh^T [H, 8H] -> SME2 (y in ZA) when available, else sgemv on the transposed matrix; otherwise whh is
// this direction's [4H, H] row block.
void stepGemv(const float* whh, bool transposed, int64_t hidden, const float* h, float* gates) {
  const int64_t g4 = 4 * hidden;
  if (!transposed) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, detail::li(g4), detail::li(hidden), 1.f, whh, detail::li(hidden), h, 1,
                1.f, gates, 1);
  } else if (sme2::available() && g4 % 64 == 0) {
    sme2::gemvT(whh, 2 * g4, hidden, g4, h, gates);
  } else if (g4 % 16 == 0) {
    gemvT4(whh, 2 * g4, hidden, g4, h, gates);
  } else {
    cblas_sgemv(CblasRowMajor, CblasTrans, detail::li(hidden), detail::li(g4), 1.f, whh, detail::li(2 * g4), h, 1, 1.f,
                gates, 1);
  }
}

// One direction. gx: [T, 8H] input projections (this direction's 4H block at column `col`),
// whh: this direction's [4H, H] block. Writes h_t to out[t, outCol : outCol + H].
void recur(const float* gx, const float* whh, bool transposed, float* out, int64_t t, int64_t hidden, int64_t col,
           int64_t outCol, bool reverse, float* work) {
  const int64_t g4 = 4 * hidden, ld = 8 * hidden;
  const int hn = static_cast<int>(hidden);
  float* h = work;
  float* c = work + hidden;
  float* gates = work + 2 * hidden;  // [i, f, g, o] x H
  float* tc = work + 6 * hidden;     // tanh(c)
  std::memset(h, 0, static_cast<size_t>(2 * hidden) * sizeof(float));
  for (int64_t s = 0; s < t; ++s) {
    const int64_t ti = reverse ? t - 1 - s : s;
    std::memcpy(gates, gx + ti * ld + col, static_cast<size_t>(g4) * sizeof(float));
    if (s > 0) stepGemv(whh, transposed, hidden, h, gates);
    float* ig = gates;
    float* fg = gates + hidden;
    float* gg = gates + 2 * hidden;
    float* og = gates + 3 * hidden;
    sigmoidInPlace(ig, 2 * hn);  // i and f are adjacent
    sigmoidInPlace(og, hn);
    vvtanhf(gg, gg, &hn);
    for (int64_t j = 0; j < hidden; ++j) c[j] = fg[j] * c[j] + ig[j] * gg[j];
    vvtanhf(tc, c, &hn);
    float* o = out + ti * 2 * hidden + outCol;
    for (int64_t j = 0; j < hidden; ++j) h[j] = o[j] = og[j] * tc[j];
  }
}

}  // namespace

// One BiLSTM's operands (lstmBidirN).
struct LstmArgs {
  const float *x, *wih, *whh, *b;
  float* out;
  int64_t t, in, hidden;
  bool whhT;
};

// n independent BiLSTMs: the input projections (one sgemm each), then all 2n direction recurrences concurrently on the
// pool (one core each). n = 2 overlaps the TextEncoder LSTM with the first duration-encoder LSTM (2026-11-02).
void lstmBidirN(const LstmArgs* ls, int n) {
  float* gx[2];
  for (int k = 0; k < n; ++k) {
    const LstmArgs& l = ls[k];
    const int64_t g8 = 8 * l.hidden;
    gx[k] = detail::scratch(static_cast<size_t>(l.t * g8), k == 0 ? 0 : 2);
    for (int64_t i = 0; i < l.t; ++i) std::memcpy(gx[k] + i * g8, l.b, static_cast<size_t>(g8) * sizeof(float));
    // gx[T, 8H] = x[T, I] @ w_ih[8H, I]^T + b
    detail::sgemm(false, true, l.t, g8, l.in, 1.f, l.x, l.in, l.wih, l.in, 1.f, gx[k], g8);
  }
  auto dir = [&](int64_t i) {
    const LstmArgs& l = ls[i / 2];
    const int64_t d = i % 2, hidden = l.hidden;
    float* work = detail::scratch(static_cast<size_t>(7 * hidden), 1);  // per thread
    const float* w = l.whhT ? l.whh + d * 4 * hidden : l.whh + d * 4 * hidden * hidden;
    recur(gx[i / 2], w, l.whhT, l.out, l.t, hidden, d * 4 * hidden, d * hidden, d == 1, work);
  };
  int64_t tmax = 0;
  for (int k = 0; k < n; ++k) tmax = std::max(tmax, ls[k].t);
  static const int pcores = [] {
    int p = 0;
    size_t len = sizeof p;
    return sysctlbyname("hw.perflevel0.logicalcpu", &p, &len, nullptr, 0) == 0 && p > 0 ? p : 8;
  }();
  if (tmax < 4) {
    for (int64_t i = 0; i < 2 * n; ++i) dir(i);
  } else if (2 * n <= pcores) {
    detail::parallelForSt(2 * n, dir);
  } else {
    // Fewer P cores than recurrences (iPhone: 2 P + 4 E; a recurrence on an E core is several times slower): one
    // BiLSTM at a time, as separate kernels would run. Needs iPhone validation (2026-11-02, phone unavailable).
    for (int k = 0; k < n; ++k) detail::parallelForSt(2, [&](int64_t d) { dir(2 * k + d); });
  }
}

// whhT: false = w_hh [8H, H] (torch layout), true = w_hh^T [H, 8H] (lstm_bidir_t_f32_cpu).
void lstmBidir(const float* x, const float* wih, const float* whh, bool whhT, const float* b, float* out, int64_t t,
               int64_t in, int64_t hidden) {
  const LstmArgs l{x, wih, whh, b, out, t, in, hidden, whhT};
  lstmBidirN(&l, 1);
}

void lstm_bidir_f32(const float* x, const float* wih, const float* whh, const float* b, float* out, int64_t t,
                    int64_t in, int64_t hidden) {
  lstmBidir(x, wih, whh, false, b, out, t, in, hidden);
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

// ints: x(T, I), w_ih(8H, I), w_hh(8H, H), b(8H), out(T, 2H)
void runLstmBidir(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::lstm_bidir_f32(static_cast<const float*>(a.ptrs[0]), static_cast<const float*>(a.ptrs[1]),
                               static_cast<const float*>(a.ptrs[2]), static_cast<const float*>(a.ptrs[3]),
                               static_cast<float*>(a.ptrs[4]), a.ints[0], a.ints[1], a.ints[5]);
}

// ints: x(T, I), w_ih(8H, I), w_hh_t(H, 8H), b(8H), out(T, 2H)
void runLstmBidirT(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::lstmBidir(static_cast<const float*>(a.ptrs[0]), static_cast<const float*>(a.ptrs[1]),
                          static_cast<const float*>(a.ptrs[2]), true, static_cast<const float*>(a.ptrs[3]),
                          static_cast<float*>(a.ptrs[4]), a.ints[0], a.ints[1], a.ints[4]);
}

// Two independent BiLSTMs: ints x1(T1, I1), w_ih1(8H, I1), w_hh1(8H, H), b1(8H), x2(T2, I2), w_ih2(8H, I2), w_hh2(8H, H),
// b2(8H), out1(T1, 2H), out2(T2, 2H)
void runLstmBidir2(ExecCtx&, const KernelArgs& a) {
  auto f = [&](int i) { return static_cast<const float*>(a.ptrs[i]); };
  const int64_t* n = a.ints;
  const kernels::cpu::LstmArgs ls[2] = {
      {f(0), f(1), f(2), f(3), static_cast<float*>(a.ptrs[8]), n[0], n[1], n[5], false},
      {f(4), f(5), f(6), f(7), static_cast<float*>(a.ptrs[9]), n[7], n[8], n[12], false}};
  kernels::cpu::lstmBidirN(ls, 2);
}

}  // namespace

void registerCpuKokoroLstmKernels() {
  registerKernel({"lstm_bidir_f32_cpu", Device::Cpu, nullptr, runLstmBidir, nullptr});
  registerKernel({"lstm_bidir2_f32_cpu", Device::Cpu, nullptr, runLstmBidir2, nullptr});
  registerKernel({"lstm_bidir_t_f32_cpu", Device::Cpu, nullptr, runLstmBidirT, nullptr});
}

}  // namespace lk::rt
