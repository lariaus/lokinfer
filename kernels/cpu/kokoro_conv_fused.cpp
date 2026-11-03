// Fused frame-rate conv1d for the Kokoro generator (M3; kokoro-port-plan.md "conv1d_fused").
//
//   out = conv1d(act((x - mean) * scale + beta), w) + bias [+ residual]        stride 1, zero padding
//
// act ∈ {none, snake(alpha), leaky(slope)}; the per-input-channel affine is AdaIN
// (affine = [mean; (1 + g) / std; b] from adain_affine_f32_cpu). Centering first (instead of
// folding into x * s + t) keeps full precision when the variance is tiny next to the mean. The output is
// processed in time tiles of `tile` samples (the M2 winner: time-tiled im2col + single-threaded
// sgemm per tile, tiles spread over the runtime thread pool). Per tile and input channel the
// transformed input window [t0 - pad, t0 + n - pad + (k-1)d) is built once (zeros outside [0, L),
// like torch, which pads after the activation), im2col copies from it, and the epilogue adds
// bias/residual and writes per-(channel, 128-sample chunk) stats (mean, M2) for the next AdaIN while
// the tile is still in L2 (the chunk is fixed so stats don't depend on the conv tile; tiles are
// multiples of it). adain_affine combines the partials exactly, in double.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <pthread.h>
#include <sys/sysctl.h>

#include <memory>
#include <time.h>
#include <cstring>

#include "kokoro_common.h"
#include "sme2_gemm.h"
#include "lokinfer/kernels/cpu_kokoro.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

namespace {

// Below this many MACs the whole conv runs inline on the calling thread (pool hand-off ~5-20 µs). One constant for
// both devices: every Kokoro conv that reaches this kernel is far above it (the smallest, the F0/N blocks at L = F
// for a one-word input, are > 10 M MACs; app59f's are 100 M - 3 G), so a per-device value would not change any Kokoro
// decision; only the inline/pool choice for tiny convs (< ~10 M MACs, not in the model) depends on the device.
constexpr double kParallelMacs = 4e6;

// Tile stats over n values: mean and M2 = sum (v - mean)^2 (two passes, in cache).
void tileStats(const float* v, int64_t n, float* st) {
  const float mean = static_cast<float>(detail::sumBlocked(v, n) / static_cast<double>(n));
  st[0] = mean;
  st[1] = static_cast<float>(detail::sumSqDevBlocked(v, n, mean));
}

// dst[i] = act((src[i] - m) * a + b) for n values.
void transformRow(const float* src, float* dst, int64_t n, float m, float a, float b, ConvAct act, float alpha,
                  float slope) {
  int64_t i = 0;
  const float32x4_t vm = vdupq_n_f32(m), va = vdupq_n_f32(a), vb = vdupq_n_f32(b);
  for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vfmaq_f32(vb, vsubq_f32(vld1q_f32(src + i), vm), va));
  for (; i < n; ++i) dst[i] = (src[i] - m) * a + b;
  if (act == ConvAct::Leaky) {
    for (int64_t j = 0; j < n; ++j) dst[j] = dst[j] >= 0.f ? dst[j] : dst[j] * slope;
  } else if (act == ConvAct::Snake) {
    // x + (1/a)·sin(a·x)², sin² from the NEON polynomial (detail::sinSq4), no vvsinf pass.
    const float inv = 1.f / alpha;
    const float32x4_t va2 = vdupq_n_f32(alpha), vinv = vdupq_n_f32(inv);
    int64_t j = 0;
    for (; j + 4 <= n; j += 4) {
      const float32x4_t v = vld1q_f32(dst + j);
      vst1q_f32(dst + j, vfmaq_f32(v, vinv, detail::sinSq4(vmulq_f32(va2, v))));
    }
    if (j < n) {  // tail through a padded vector
      float t[4] = {0.f, 0.f, 0.f, 0.f};
      std::memcpy(t, dst + j, static_cast<size_t>(n - j) * sizeof(float));
      const float32x4_t v = vld1q_f32(t);
      vst1q_f32(t, vfmaq_f32(v, vinv, detail::sinSq4(vmulq_f32(va2, v))));
      std::memcpy(dst + j, t, static_cast<size_t>(n - j) * sizeof(float));
    }
  }
}

}  // namespace

int64_t statChunks(int64_t l) { return (l + kStatChunk - 1) / kStatChunk; }

namespace {

// Stats of out[c, t0 : t0 + n) (t0 a multiple of kStatChunk) into stats[c, t0 / chunk ...].
void writeStats(const float* row, int64_t t0, int64_t n, float* stats, int64_t nchunks, int64_t c) {
  for (int64_t j = 0; j < n; j += kStatChunk)
    tileStats(row + j, std::min(kStatChunk, n - j), stats + (c * nchunks + (t0 + j) / kStatChunk) * 2);
}

// Bias, residual, out scale and the next AdaIN's chunk stats on out[:, t0 : t0 + n) (o = out + t0), in cache.
void epilogue(const ConvFusedArgs& p, float* o, int64_t lo, int64_t t0, int64_t n, int64_t nchunks, int64_t c0 = 0,
              int64_t c1 = -1) {
  if (c1 < 0) c1 = p.co;
    for (int64_t c = c0; c < c1; ++c) {
      float* orow = o + c * lo;
      const float bias = p.bias ? p.bias[c] : 0.f;
      const float32x4_t vb = vdupq_n_f32(bias);
      int64_t j = 0;
      if (p.residual && p.outScale != 1.f) {
        const float* r = p.residual + c * lo + t0;
        const float32x4_t vs = vdupq_n_f32(p.outScale);
        for (; j + 4 <= n; j += 4)
          vst1q_f32(orow + j, vmulq_f32(vaddq_f32(vaddq_f32(vld1q_f32(orow + j), vb), vld1q_f32(r + j)), vs));
        for (; j < n; ++j) orow[j] = (orow[j] + bias + r[j]) * p.outScale;
      } else if (p.residual) {
        const float* r = p.residual + c * lo + t0;
        for (; j + 4 <= n; j += 4) vst1q_f32(orow + j, vaddq_f32(vaddq_f32(vld1q_f32(orow + j), vb), vld1q_f32(r + j)));
        for (; j < n; ++j) orow[j] = orow[j] + bias + r[j];
      } else if (p.bias) {
        for (; j + 4 <= n; j += 4) vst1q_f32(orow + j, vaddq_f32(vld1q_f32(orow + j), vb));
        for (; j < n; ++j) orow[j] += bias;
      }
      if (p.stats) writeStats(orow, t0, n, p.stats, nchunks, c);
    }
}

}  // namespace

void conv1d_fused_f32(const ConvFusedArgs& p) {
  const int64_t ci = p.ci, l = p.l, co = p.co, k = p.k, d = p.dilation, pad = p.padding, tile = p.tile;
  const int64_t lo = l + 2 * pad - d * (k - 1);
  if (lo <= 0) return;
  const int64_t ntiles = (lo + tile - 1) / tile, nchunks = statChunks(lo), ck = ci * k, span = (k - 1) * d;

  auto runTile = [&](int64_t ti) {
    const int64_t t0 = ti * tile, n = std::min(tile, lo - t0), win = n + span;
    float* xw = detail::scratch(static_cast<size_t>(ci * (tile + span)), 1);
    float* col = detail::scratch(static_cast<size_t>(ck * tile), 2);
    // Transformed input window per channel: positions [t0 - pad, t0 - pad + win).
    const int64_t p0 = t0 - pad, a = std::clamp<int64_t>(-p0, 0, win), b = std::clamp<int64_t>(l - p0, a, win);
    for (int64_t c = 0; c < ci; ++c) {
      float* row = xw + c * win;
      std::fill(row, row + a, 0.f);
      const float* af = p.affine;
      const float m = af ? af[c] : 0.f, sc = af ? af[ci + c] : 1.f, be = af ? af[2 * ci + c] : 0.f;
      if (b > a) transformRow(p.x + c * l + p0 + a, row + a, b - a, m, sc, be, p.act, p.alpha ? p.alpha[c] : 0.f, p.slope);
      std::fill(row + b, row + win, 0.f);
      for (int64_t kk = 0; kk < k; ++kk)
        std::memcpy(col + (c * k + kk) * n, row + kk * d, static_cast<size_t>(n) * sizeof(float));
    }
    float* o = p.out + t0;
    detail::sgemm(false, false, co, n, ck, 1.f, p.w, ck, col, n, 0.f, o, lo);
    epilogue(p, o, lo, t0, n, nchunks);
  };

  const double macs = static_cast<double>(co) * static_cast<double>(ck) * static_cast<double>(lo);
  if (macs < kParallelMacs || ntiles == 1) {
    for (int64_t ti = 0; ti < ntiles; ++ti) runTile(ti);
    return;
  }
  detail::parallelForSt(ntiles, runTile);
}

// --- Winograd variant (F(4,3) for K = 3, nested F(4,4) for K = 4..12; docs/impl-notes/winograd-conv.md) ----------
//
// Per time tile [t0, t0 + n): the transformed (AdaIN affine + act) input window xw is built as above; residue class ρ
// (output t = t0 + ρ + d·m) is an undilated conv over xw[ρ + d·j], so Winograd tile q of class ρ covers m = 4q..4q+3.
// V[ξ][c][col], col = ρ·QS + q (QS = Qmax + G - 1 transformed tiles per class, G tap groups that read tiles q + g):
// V = BT · xw[ρ + d(4q + i)]. Then per point ξ one GEMM M[ξ][co][col] = Σ_{c, g} U[ξ][c, g][co] · V[ξ][c][col + g]
// (SME2: B rows read in place, shifted by g; otherwise Accelerate on a [Ci·G, cols] copy), and the output transform
// y[t0 + ρ + d(4q + j)] = Σ_ξ AT[j][ξ] M[ξ][co][ρ·QS + q] (j < 4) followed by the usual epilogue.

namespace {

constexpr int64_t kWinoTile = 4;  // outputs per Winograd tile

// Input transform of one channel window row (n valid samples needed per class).
// The transforms as compile-time tables (ops.winograd.transforms(4, r) in f32; checked against the op's `tr` input),
// so the unrolled loops drop the zero terms and the multiplies by ±1.
template <int A>
struct WinoTables;
template <>
struct WinoTables<6> {  // F(4, 3), points {0, ±1, ±2, ∞}
  static constexpr float bt[6][6] = {{0.800000011920929f, 0.f, -1.f, 0.f, 0.20000000298023224f, 0.f},
                                     {0.f, 1.f, 1.f, -0.25f, -0.25f, 0.f},
                                     {0.f, -1.f, 1.f, 0.25f, -0.25f, 0.f},
                                     {0.f, -1.f, -0.5f, 1.f, 0.5f, 0.f},
                                     {0.f, 1.f, -0.5f, -1.f, 0.5f, 0.f},
                                     {0.f, 0.800000011920929f, 0.f, -1.f, 0.f, 0.20000000298023224f}};
  static constexpr float at[4][6] = {{1.f, 1.f, 1.f, 1.f, 1.f, 0.f},
                                     {0.f, 1.f, -1.f, 2.f, -2.f, 0.f},
                                     {0.f, 1.f, 1.f, 4.f, 4.f, 0.f},
                                     {0.f, 1.f, -1.f, 8.f, -8.f, 1.f}};
};
template <>
struct WinoTables<7> {  // F(4, 4), points {0, ±1, ±2, -1/2, ∞}
  static constexpr float bt[7][7] = {
      {0.4000000059604645f, 0.800000011920929f, -0.5f, -1.f, 0.10000000149011612f, 0.20000000298023224f, 0.f},
      {0.f, 0.3333333432674408f, 1.f, 0.5833333134651184f, -0.25f, -0.1666666716337204f, 0.f},
      {0.f, 0.4444444477558136f, 0.4444444477558136f, -1.f, -0.1111111119389534f, 0.2222222238779068f, 0.f},
      {0.f, -0.4000000059604645f, -1.f, 0.f, 1.f, 0.4000000059604645f, 0.f},
      {0.f, -0.5f, -0.75f, 1.f, 0.75f, -0.5f, 0.f},
      {0.f, -0.800000011920929f, 0.f, 1.f, 0.f, -0.20000000298023224f, 0.f},
      {0.f, 0.4000000059604645f, 0.800000011920929f, -0.5f, -1.f, 0.10000000149011612f, 0.20000000298023224f}};
  static constexpr float at[4][7] = {{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f},
                                     {0.f, 1.f, -1.f, 2.f, -2.f, -0.5f, 0.f},
                                     {0.f, 1.f, 1.f, 4.f, 4.f, 0.25f, 0.f},
                                     {0.f, 1.f, -1.f, 8.f, -8.f, -0.125f, 1.f}};
};

template <int A>
bool winoTablesMatch(const float* tr) {  // to ~1 ulp (the caller may have derived the table in another precision)
  auto near = [](float a, float b) { return std::fabs(a - b) <= 1e-6f * std::max(1.f, std::fabs(b)); };
  for (int i = 0; i < A * A; ++i)
    if (!near(tr[i], WinoTables<A>::bt[i / A][i % A])) return false;
  for (int i = 0; i < 4 * A; ++i)
    if (!near(tr[A * A + i], WinoTables<A>::at[i / A][i % A])) return false;
  return true;
}

// acc = Σ_i c[i] · in[i] over the nonzero c[i] (c a row of a constexpr table; folded after unrolling).
template <int A>
inline float32x4_t dotConst(const float (&c)[A], const float32x4_t* in) {
  float32x4_t acc = vdupq_n_f32(0.f);
  bool first = true;
#pragma clang loop unroll(full)
  for (int i = 0; i < A; ++i) {
    if (c[i] == 0.f) continue;
    if (first) {
      acc = c[i] == 1.f ? in[i] : c[i] == -1.f ? vnegq_f32(in[i]) : vmulq_n_f32(in[i], c[i]);
      first = false;
    } else if (c[i] == 1.f) {
      acc = vaddq_f32(acc, in[i]);
    } else if (c[i] == -1.f) {
      acc = vsubq_f32(acc, in[i]);
    } else {
      acc = vfmaq_n_f32(acc, in[i], c[i]);
    }
  }
  return acc;
}

// Residue rows: sub[ρ][j] = row[ρ + d·j], j < len (sub row stride `len`).
inline void deinterleave(const float* row, int64_t d, int64_t len, float* sub) {
  int64_t j = 0;
  if (d == 3) {
    for (; j + 4 <= len; j += 4) {
      const float32x4x3_t v = vld3q_f32(row + 3 * j);
      vst1q_f32(sub + j, v.val[0]);
      vst1q_f32(sub + len + j, v.val[1]);
      vst1q_f32(sub + 2 * len + j, v.val[2]);
    }
  } else if (d == 5) {
    // 20 samples = 4 per class: class ρ takes floats ρ, ρ+5, ρ+10, ρ+15 (byte tables over v0..v3, then v4 for > 15).
    uint8x16_t lo[5], hi[5];
    for (int r = 0; r < 5; ++r) {
      uint8_t a[16], b[16];
      for (int m = 0; m < 4; ++m) {
        const int f = r + 5 * m;
        for (int by = 0; by < 4; ++by) {
          a[4 * m + by] = f < 16 ? static_cast<uint8_t>(4 * f + by) : 0xFF;
          b[4 * m + by] = f >= 16 ? static_cast<uint8_t>(4 * (f - 16) + by) : 0xFF;
        }
      }
      lo[r] = vld1q_u8(a);
      hi[r] = vld1q_u8(b);
    }
    for (; j + 4 <= len; j += 4) {
      const float* src = row + 5 * j;
      uint8x16x4_t t;
      t.val[0] = vreinterpretq_u8_f32(vld1q_f32(src));
      t.val[1] = vreinterpretq_u8_f32(vld1q_f32(src + 4));
      t.val[2] = vreinterpretq_u8_f32(vld1q_f32(src + 8));
      t.val[3] = vreinterpretq_u8_f32(vld1q_f32(src + 12));
      const uint8x16_t v4 = vreinterpretq_u8_f32(vld1q_f32(src + 16));
      for (int r = 0; r < 5; ++r)
        vst1q_f32(sub + r * len + j, vreinterpretq_f32_u8(vqtbx1q_u8(vqtbl4q_u8(t, lo[r]), v4, hi[r])));
    }
  } else if (d == 2) {
    for (; j + 4 <= len; j += 4) {
      const float32x4x2_t v = vld2q_f32(row + 2 * j);
      vst1q_f32(sub + j, v.val[0]);
      vst1q_f32(sub + len + j, v.val[1]);
    }
  }
  for (int64_t rho = 0; rho < d; ++rho)
    for (int64_t jj = j; jj < len; ++jj) sub[rho * len + jj] = row[rho + d * jj];
}

// row[ρ + d·j] = sub[ρ][j] for ρ + d·j < n (sub row stride `len`).
inline void interleave(const float* sub, int64_t d, int64_t len, int64_t n, float* row) {
  const int64_t full = n / d;  // j < full is in range for every ρ
  int64_t j = 0;
  if (d == 3) {
    for (; j + 4 <= full; j += 4) {
      float32x4x3_t v;
      v.val[0] = vld1q_f32(sub + j);
      v.val[1] = vld1q_f32(sub + len + j);
      v.val[2] = vld1q_f32(sub + 2 * len + j);
      vst3q_f32(row + 3 * j, v);
    }
  } else if (d == 5) {
    // Inverse of deinterleave's d = 5 case: out float f = 4·k + e comes from class f % 5, lane f / 5.
    uint8x16_t lo[5], hi[5];
    for (int k = 0; k < 5; ++k) {
      uint8_t a[16], b[16];
      for (int e = 0; e < 4; ++e) {
        const int f = 4 * k + e, r = f % 5, m = f / 5;
        for (int by = 0; by < 4; ++by) {
          a[4 * e + by] = r < 4 ? static_cast<uint8_t>(16 * r + 4 * m + by) : 0xFF;
          b[4 * e + by] = r == 4 ? static_cast<uint8_t>(4 * m + by) : 0xFF;
        }
      }
      lo[k] = vld1q_u8(a);
      hi[k] = vld1q_u8(b);
    }
    for (; j + 4 <= full; j += 4) {
      uint8x16x4_t t;
      for (int r = 0; r < 4; ++r) t.val[r] = vreinterpretq_u8_f32(vld1q_f32(sub + r * len + j));
      const uint8x16_t v4 = vreinterpretq_u8_f32(vld1q_f32(sub + 4 * len + j));
      for (int k = 0; k < 5; ++k)
        vst1q_f32(row + 5 * j + 4 * k, vreinterpretq_f32_u8(vqtbx1q_u8(vqtbl4q_u8(t, lo[k]), v4, hi[k])));
    }
  } else if (d == 2) {
    for (; j + 4 <= full; j += 4) {
      float32x4x2_t v;
      v.val[0] = vld1q_f32(sub + j);
      v.val[1] = vld1q_f32(sub + len + j);
      vst2q_f32(row + 2 * j, v);
    }
  }
  for (int64_t t = d * j; t < n; ++t) row[t] = sub[(t % d) * len + t / d];
}

// Input transform of one channel's window row: V[ξ][col], col = ρ·qs + q (ξ stride vStride).
template <int A>
void winoInput(const float* row, int64_t d, int64_t qs, float* v, int64_t vStride, float* sub) {
  using Tb = WinoTables<A>;
  const int64_t len = kWinoTile * qs + 8;  // samples per class (4 tiles of vectorized reads past the end stay in it)
  const float* base = row;
  if (d > 1) {
    deinterleave(row, d, len, sub);
    base = sub;
  }
  for (int64_t rho = 0; rho < d; ++rho) {
    const float* s = base + (d > 1 ? rho * len : 0);
    float* vcol = v + rho * qs;
    int64_t q = 0;
    for (; q + 4 <= qs; q += 4) {
      // in[i] for tiles q..q+3 = s[4q + i]: deinterleave two 16-sample blocks.
      const float32x4x4_t lo4 = vld4q_f32(s + kWinoTile * q), hi4 = vld4q_f32(s + kWinoTile * q + 4);
      const float32x4_t in[8] = {lo4.val[0], lo4.val[1], lo4.val[2], lo4.val[3],
                                 hi4.val[0], hi4.val[1], hi4.val[2], hi4.val[3]};
#pragma clang loop unroll(full)
      for (int xi = 0; xi < A; ++xi) vst1q_f32(vcol + xi * vStride + q, dotConst<A>(Tb::bt[xi], in));
    }
    for (; q < qs; ++q)
      for (int xi = 0; xi < A; ++xi) {
        float acc = 0.f;
        for (int i = 0; i < A; ++i) acc += Tb::bt[xi][i] * s[kWinoTile * q + i];
        vcol[xi * vStride + q] = acc;
      }
  }
}

// Output transform for one output channel: M[ξ][col] (ξ stride mStride) -> orow[ρ + d(4q + j)] for t < n.
template <int A>
void winoOutput(const float* m, int64_t mStride, int64_t d, int64_t qs, int64_t n, float* orow, float* stage) {
  using Tb = WinoTables<A>;
  const int64_t len = kWinoTile * qs;
  float* dst0 = d == 1 ? orow : stage;
  for (int64_t rho = 0; rho < d && rho < n; ++rho) {
    const int64_t cnt = (n - rho + d - 1) / d;  // outputs of this class in the tile
    const int64_t nq = (cnt + kWinoTile - 1) / kWinoTile;
    float* dst = dst0 + rho * len;
    const float* mc = m + rho * qs;
    int64_t q = 0;
    const int64_t nqFull = d == 1 ? cnt / kWinoTile : nq;  // d == 1 writes straight into orow: no overrun
    for (; q + 4 <= nqFull; q += 4) {
      float32x4_t mv[A];
#pragma clang loop unroll(full)
      for (int xi = 0; xi < A; ++xi) mv[xi] = vld1q_f32(mc + xi * mStride + q);
      float32x4x4_t y;
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j) y.val[j] = dotConst<A>(Tb::at[j], mv);
      vst4q_f32(dst + kWinoTile * q, y);
    }
    for (; q < nq; ++q)
      for (int j = 0; j < 4; ++j) {
        if (d == 1 && kWinoTile * q + j >= cnt) break;
        float acc = 0.f;
        for (int xi = 0; xi < A; ++xi) acc += Tb::at[j][xi] * mc[xi * mStride + q];
        dst[kWinoTile * q + j] = acc;
      }
  }
  if (d > 1) interleave(stage, d, len, n, orow);
}

// The window row of channel c for positions [p0, p0 + win): act(affine(x)), zeros outside [0, l).
void windowRow(const ConvFusedArgs& p, int64_t c, int64_t p0, int64_t win, float* row) {
  const int64_t ci = p.ci, l = p.l;
  const int64_t a = std::clamp<int64_t>(-p0, 0, win), b = std::clamp<int64_t>(l - p0, a, win);
  std::fill(row, row + a, 0.f);
  const float* af = p.affine;
  const float m = af ? af[c] : 0.f, sc = af ? af[ci + c] : 1.f, be = af ? af[2 * ci + c] : 0.f;
  const float* src = p.x + c * l + p0 + a;
  if (b > a && p.x2) {  // generator stage output mean3(rb0, rb1, rb2) read straight from the three resblock outputs
    const float* s2 = p.x2 + c * l + p0 + a;
    const float* s3 = p.x3 + c * l + p0 + a;
    float* d = row + a;
    const float32x4_t vs = vdupq_n_f32(p.inScale);
    int64_t i = 0;
    for (; i + 4 <= b - a; i += 4)
      vst1q_f32(d + i, vmulq_f32(vaddq_f32(vaddq_f32(vld1q_f32(src + i), vld1q_f32(s2 + i)), vld1q_f32(s3 + i)), vs));
    for (; i < b - a; ++i) d[i] = (src[i] + s2[i] + s3[i]) * p.inScale;
    src = d;  // transformRow below runs in place
  }
  if (b > a) transformRow(src, row + a, b - a, m, sc, be, p.act, p.alpha ? p.alpha[c] : 0.f, p.slope);
  std::fill(row + b, row + win, 0.f);
}

// Winograd geometry of one time tile of n outputs.
struct WinoGeom {
  int64_t qmax, qs, cols, ncols, win;
  WinoGeom(int64_t n, int64_t d, int64_t g) {
    qmax = (((n + d - 1) / d) + kWinoTile - 1) / kWinoTile;  // Winograd tiles per residue class
    qs = qmax + g - 1;                                        // + the tiles only read shifted by a tap group
    cols = d * qs;
    ncols = cols - (g - 1);  // GEMM columns (the last g - 1 only feed shifted reads)
    win = d * (kWinoTile * qs + 8);  // window samples: class rows of 4·qs + 8 (vectorized reads)
  }
};

// M[ξ] rows [r0, r0 + rows) = U[ξ] rows [r0, r0 + rows) x V[ξ] for one Winograd point ξ, SME2 or Accelerate.
inline void winoGemmPoint(int xi, const float* up, int64_t co, int64_t ci, int64_t g, const float* v,
                          const WinoGeom& gm, int64_t r0, int64_t rows, float* m, bool sme) {
  const int64_t cb = (co + 31) / 32, ck = ci * g;
  const float* uxi = up + (static_cast<int64_t>(xi) * cb + r0 / 32) * ck * 32;
  float* mxi = m + static_cast<int64_t>(xi) * rows * gm.cols;
  const float* vxi = v + static_cast<int64_t>(xi) * ci * gm.cols;
  if (sme) {
    sme2::gemm(uxi, rows, ck, sme2::BOperand{vxi, gm.cols, 32, g, 1}, gm.ncols, mxi, gm.cols);
  } else if (g <= 2) {
    // B read in place, one accumulating sgemm per tap group (A_g^T = panel rows c·G + g, lda = G·32): no V copy.
    // Mac 2026-11-02: K = 3 stage-1 0.89 -> 0.74 ms, decoder 1024 x 147 0.52 -> 0.35, K = 7 1.16 -> 0.91; K = 11
    // (G = 3) is slower this way (1.69 -> 2.05: three K = Ci passes over the same M) and keeps the copy.
    for (int64_t bi = 0; bi * 32 < rows; ++bi)
      for (int64_t gg = 0; gg < g; ++gg)
        detail::sgemm(true, false, std::min<int64_t>(32, rows - bi * 32), gm.ncols, ci, 1.f, uxi + bi * ck * 32 + gg * 32,
                      g * 32, vxi + gg, gm.cols, gg ? 1.f : 0.f, mxi + bi * 32 * gm.cols, gm.cols);
  } else {
    float* col = detail::scratch(static_cast<size_t>(ck * gm.ncols), 0);
    for (int64_t c = 0; c < ci; ++c)
      for (int64_t gg = 0; gg < g; ++gg)
        std::memcpy(col + (c * g + gg) * gm.ncols, vxi + c * gm.cols + gg, static_cast<size_t>(gm.ncols) * sizeof(float));
    for (int64_t bi = 0; bi * 32 < rows; ++bi)
      detail::sgemm(true, false, std::min<int64_t>(32, rows - bi * 32), gm.ncols, ck, 1.f, uxi + bi * ck * 32, 32,
                    col, gm.ncols, 0.f, mxi + bi * 32 * gm.cols, gm.cols);
  }
}

// M[ξ][rows][cols] = U[ξ] rows [r0, r0 + rows) x V[ξ] (all A points), SME2 or Accelerate.
template <int A>
void winoGemm(const float* up, int64_t co, int64_t ci, int64_t g, const float* v, const WinoGeom& gm, int64_t r0,
              int64_t rows, float* m, bool sme) {
  for (int xi = 0; xi < A; ++xi) winoGemmPoint(xi, up, co, ci, g, v, gm, r0, rows, m, sme);
}

using detail::onPerfCore;
using detail::perfCoreSplit;

template <int A>
void conv1dWino(const ConvFusedArgs& p, const float* up, const float* tr, int64_t groups) {
  const int64_t ci = p.ci, l = p.l, co = p.co, k = p.k, d = p.dilation, pad = p.padding, g = groups;
  const int64_t lo = l + 2 * pad - d * (k - 1);
  if (lo <= 0) return;
  if (!winoTablesMatch<A>(tr)) {
    std::fprintf(stderr, "conv1d_*_wino: transform table doesn't match the kernel's (use ops.wino_weights_cpu)\n");
    return;
  }
  const int64_t nchunks = statChunks(lo), threads = rt::ThreadPool::global().threads();
  const bool sme = sme2::available();
  const double macs = static_cast<double>(co) * static_cast<double>(ci * k) * static_cast<double>(lo) * 0.5;
  const bool parallel = macs >= kParallelMacs && threads > 1;
  // EXPERIMENT (A/B): tile attr bits >= 20 = mode. bit 0: E cores run the GEMMs on Accelerate instead of SME2;
  // bit 1: pipeline GEMMs claimed per (tile, point), E cores take them when they have no transform work.
  const int64_t tile = p.tile & ((int64_t{1} << 20) - 1);
  const int mode = static_cast<int>(p.tile >> 20);
  const bool eAccel = (mode & 1) && perfCoreSplit() > 0;
  auto smeHere = [&] { return sme && (!eAccel || onPerfCore()); };
  // Two-phase single tile below 2 tiles per thread (iPhone 2026-11-01, Ci = 256 L = 2940 K = 11: 1.36 ms vs 1.51 ms for
  // the P/E pipeline over 6 tiles; Ci = 128 L = 6601 (13 tiles): pipeline 0.88 vs 0.98 ms).
  if (parallel && (lo + tile - 1) / tile < 2 * threads) {
    // Short sequence: one tile, two parallel phases. 1) the transformed input V over channel blocks, 2) per 32
    // output channels: the A GEMMs, the output transform and the epilogue.
    const WinoGeom gm(lo, d, g);
    float* v = detail::scratch(static_cast<size_t>(A * ci * gm.cols + 32), 5);
    const int64_t cpt = std::max<int64_t>(1, (ci + 2 * threads - 1) / (2 * threads));
    detail::parallelForSt((ci + cpt - 1) / cpt, [&](int64_t t) {
      float* row = detail::scratch(static_cast<size_t>(gm.win), 1);
      float* sub = detail::scratch(static_cast<size_t>(gm.win), 4);
      for (int64_t c = t * cpt; c < std::min(ci, (t + 1) * cpt); ++c) {
        windowRow(p, c, -pad, gm.win, row);
        winoInput<A>(row, d, gm.qs, v + c * gm.cols, ci * gm.cols, sub);
      }
    });
    detail::parallelForSt((co + 31) / 32, [&](int64_t bi) {
      const int64_t r0 = bi * 32, rows = std::min<int64_t>(32, co - r0);
      float* m = detail::scratch(static_cast<size_t>(A * rows * gm.cols + 32), 3);
      float* stage = detail::scratch(static_cast<size_t>(gm.win), 4);
      winoGemm<A>(up, co, ci, g, v, gm, r0, rows, m, smeHere());
      for (int64_t c = 0; c < rows; ++c)
        winoOutput<A>(m + c * gm.cols, rows * gm.cols, d, gm.qs, lo, p.out + (r0 + c) * lo, stage);
      epilogue(p, p.out, lo, 0, lo, nchunks, r0, r0 + rows);
    });
    return;
  }
  const int64_t ntiles = (lo + tile - 1) / tile;
  // The three stages of a tile: 1) window + input transform -> v, 2) the A GEMMs -> m, 3) output transform + epilogue.
  auto stage1 = [&](int64_t ti, float* v) {
    const int64_t t0 = ti * tile;
    const WinoGeom gm(std::min(tile, lo - t0), d, g);
    float* row = detail::scratch(static_cast<size_t>(gm.win), 1);
    float* sub = detail::scratch(static_cast<size_t>(gm.win), 4);
    for (int64_t c = 0; c < ci; ++c) {
      windowRow(p, c, t0 - pad, gm.win, row);
      winoInput<A>(row, d, gm.qs, v + c * gm.cols, ci * gm.cols, sub);
    }
  };
  auto stage2 = [&](int64_t ti, const float* v, float* m) {
    const WinoGeom gm(std::min(tile, lo - ti * tile), d, g);
    winoGemm<A>(up, co, ci, g, v, gm, 0, co, m, smeHere());
  };
  auto stage2p = [&](int64_t ti, int xi, const float* v, float* m, bool useSme) {
    const WinoGeom gm(std::min(tile, lo - ti * tile), d, g);
    winoGemmPoint(xi, up, co, ci, g, v, gm, 0, co, m, useSme);
  };
  auto stage3 = [&](int64_t ti, const float* m) {
    const int64_t t0 = ti * tile, n = std::min(tile, lo - t0);
    const WinoGeom gm(n, d, g);
    float* sub = detail::scratch(static_cast<size_t>(gm.win), 4);
    float* o = p.out + t0;
    for (int64_t c = 0; c < co; ++c) winoOutput<A>(m + c * gm.cols, co * gm.cols, d, gm.qs, n, o + c * lo, sub);
    epilogue(p, o, lo, t0, n, nchunks);
  };
  auto runTile = [&](int64_t ti) {
    const WinoGeom gm(std::min(tile, lo - ti * tile), d, g);
    float* v = detail::scratch(static_cast<size_t>(A * ci * gm.cols + 32), 2);
    float* m = detail::scratch(static_cast<size_t>(A * co * gm.cols + 32), 3);
    stage1(ti, v);
    stage2(ti, v, m);
    stage3(ti, m);
  };
  if (parallel && sme && ntiles > 1 && perfCoreSplit() > 0) {
    // SME2 parts with an efficiency cluster (iPhone 18 Pro Max: 2 P + 4 E cores, one SME unit per cluster; the E unit runs these
    // GEMMs ~9x slower, measured 2026-11-01): the GEMMs run on the P cores, the transforms mostly on the E cores,
    // pipelined through a ring of R tile buffers. A thread classifies itself per step (pthread_cpu_number_np).
    // Ring depth: 16 tiles (iPhone sweep 2026-11-01: 4 slower than no split, 8-24 within 5 %), capped at ~16 MB.
    const int64_t bytesPerSlot = static_cast<int64_t>(A) * (ci + co) * WinoGeom(tile, d, g).cols * 4;
    const int64_t R = std::clamp<int64_t>((16 << 20) / bytesPerSlot, 4, 16);
    const WinoGeom g0(tile, d, g);
    const size_t vsz = static_cast<size_t>(A * ci * g0.cols + 32), msz = static_cast<size_t>(A * co * g0.cols + 32);
    float* vring = detail::scratch(vsz * R, 5);
    float* mring = detail::scratch(msz * R, 6);
    std::unique_ptr<std::atomic<int>[]> state(new std::atomic<int>[ntiles]);
    for (int64_t i = 0; i < ntiles; ++i) state[i].store(0, std::memory_order_relaxed);
    std::atomic<int64_t> next1{0}, next2{0}, next3{0};
    const bool points = (mode & 2) != 0;
    std::unique_ptr<std::atomic<int>[]> pdone(new std::atomic<int>[points ? ntiles : 1]);
    for (int64_t i = 0; i < (points ? ntiles : 1); ++i) pdone[i].store(0, std::memory_order_relaxed);
    const int64_t units = ntiles * A;
    auto claim = [](std::atomic<int64_t>& c, int64_t t) { return c.compare_exchange_strong(t, t + 1); };
    detail::parallelForSt(rt::ThreadPool::global().threads(), [&](int64_t) {
      int64_t idleSince = -1;
      while (next3.load(std::memory_order_acquire) < ntiles) {
        const bool perf = onPerfCore();
        bool did = false;
        auto try2u = [&](bool allow) {  // one (tile, point) GEMM unit
          const int64_t u = next2.load(std::memory_order_acquire), t = u / A;
          if (!allow || u >= units || state[t].load(std::memory_order_acquire) < 1 || !claim(next2, u)) return false;
          stage2p(t, static_cast<int>(u % A), vring + (t % R) * vsz, mring + (t % R) * msz, sme && (perf || !eAccel));
          if (pdone[t].fetch_add(1, std::memory_order_acq_rel) + 1 == A) state[t].store(2, std::memory_order_release);
          return true;
        };
        auto try2 = [&](bool allow) {
          const int64_t t = next2.load(std::memory_order_acquire);
          if (!allow || t >= ntiles || state[t].load(std::memory_order_acquire) < 1 || !claim(next2, t)) return false;
          stage2(t, vring + (t % R) * vsz, mring + (t % R) * msz);
          state[t].store(2, std::memory_order_release);
          return true;
        };
        auto try1 = [&]() {
          const int64_t t = next1.load(std::memory_order_acquire);
          if (t >= ntiles || (t >= R && state[t - R].load(std::memory_order_acquire) < 3) || !claim(next1, t)) return false;
          stage1(t, vring + (t % R) * vsz);
          state[t].store(1, std::memory_order_release);
          return true;
        };
        auto try3 = [&]() {
          const int64_t t = next3.load(std::memory_order_acquire);
          if (t >= ntiles || state[t].load(std::memory_order_acquire) < 2 || !claim(next3, t)) return false;
          stage3(t, mring + (t % R) * msz);
          state[t].store(3, std::memory_order_release);
          return true;
        };
        if (points) {
          // E cores take a point GEMM when they have no transform work, except in the last tile (an E-core unit is
          // several times slower than a P-core one).
          const int64_t u = next2.load(std::memory_order_relaxed);
          did = perf ? (try2u(true) || try1() || try3()) : (try3() || try1() || try2u(units - u > A));
        } else if (perf) {
          did = try2(true) || try1() || try3();
        } else {
          // E cores take a GEMM only when it has been ready for 2 ms with nobody taking it (no P core in the job).
          const bool stale = idleSince >= 0 && static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) - idleSince > 2000000;
          did = try3() || try1() || try2(stale);
        }
        if (did) {
          idleSince = -1;
        } else if (idleSince < 0) {
          idleSince = static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
        }
      }
    });
    return;
  }
  if (!parallel || ntiles == 1) {
    for (int64_t ti = 0; ti < ntiles; ++ti) runTile(ti);
    return;
  }
  detail::parallelForSt(ntiles, runTile);
}

}  // namespace

void conv1d_wino_f32(const ConvFusedArgs& p, const float* up, const float* tr, int64_t groups) {
  if (p.k == 3)
    conv1dWino<6>(p, up, tr, groups);
  else
    conv1dWino<7>(p, up, tr, groups);
}

void channel_stats_f32(const float* x, float* stats, int64_t c, int64_t l) {
  const int64_t nchunks = statChunks(l);
  auto one = [&](int64_t ch) { writeStats(x + ch * l, 0, l, stats, nchunks, ch); };
  if (static_cast<double>(c) * static_cast<double>(l) < kParallelMacs) {
    for (int64_t ch = 0; ch < c; ++ch) one(ch);
    return;
  }
  rt::ThreadPool::global().parallelFor(c, one);
}

void adain_affine_f32(const float* stats, const float* gb, float* affine, int64_t c, int64_t l, float eps) {
  const int64_t ntiles = statChunks(l), tile = kStatChunk;
  for (int64_t ch = 0; ch < c; ++ch) {
    // Exact parallel combine of (count, mean, M2) partials in double, two passes (no per-chunk division):
    // mean = Σ n_i m_i / N, M2 = Σ M2_i + Σ n_i (m_i - mean)². Every chunk but the last has n_i = tile.
    const float* sc = stats + ch * ntiles * 2;
    const double nLast = static_cast<double>(l - (ntiles - 1) * tile), nT = static_cast<double>(tile);
    double sumFull = 0;
    for (int64_t ti = 0; ti + 1 < ntiles; ++ti) sumFull += sc[ti * 2];
    const double mean = (sumFull * nT + nLast * sc[(ntiles - 1) * 2]) / static_cast<double>(l);
    double m2 = 0, dev = 0;
    for (int64_t ti = 0; ti + 1 < ntiles; ++ti) {
      const double dm = sc[ti * 2] - mean;
      m2 += sc[ti * 2 + 1];
      dev += dm * dm;
    }
    const double dl = sc[(ntiles - 1) * 2] - mean;
    m2 += sc[(ntiles - 1) * 2 + 1] + dev * nT + nLast * dl * dl;
    const float m = static_cast<float>(mean), var = static_cast<float>(m2 / static_cast<double>(l));  // biased
    affine[ch] = m;
    affine[c + ch] = (1.f + gb[ch]) * (1.0f / std::sqrt(var + eps));
    affine[2 * c + ch] = gb[c + ch];
  }
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

const float* F(const KernelArgs& a, int i) { return static_cast<const float*>(a.ptrs[i]); }
float* O(const KernelArgs& a, int i) { return static_cast<float*>(a.ptrs[i]); }

using kernels::cpu::ConvAct;
using kernels::cpu::ConvFusedArgs;

// conv1d_{act}[_res]_f32_cpu(x, w, b, affine, [alpha,] [res]) outs(out, stats)
// ints: x(Ci, L), w(Co, Ci, K), b(Co), affine(3, Ci), [alpha(Ci)], [res(Co, Lo)], out(Co, Lo),
//       stats(Co, ceil(Lo / 128), 2), then attrs dilation, padding, tile; floats (by name): [oscale (res)], [slope]
template <ConvAct Act, bool Res>
void runConvFused(ExecCtx&, const KernelArgs& a) {
  constexpr bool hasAlpha = Act == ConvAct::Snake;
  const int64_t* n = a.ints;
  ConvFusedArgs p;
  p.x = F(a, 0);
  p.w = F(a, 1);
  p.bias = F(a, 2);
  p.affine = F(a, 3);
  int pi = 4;
  if (hasAlpha) p.alpha = F(a, pi++);
  if (Res) p.residual = F(a, pi++);
  p.out = O(a, pi++);
  p.stats = O(a, pi++);
  p.ci = n[0];
  p.l = n[1];
  p.co = n[2];
  p.k = n[4];
  const int attr = 8 + (hasAlpha ? 1 : 0) + (Res ? 2 : 0) + 2 + 3;
  p.dilation = n[attr];
  p.padding = n[attr + 1];
  p.tile = n[attr + 2];
  p.act = Act;
  int fi = 0;
  if (Res && Act == ConvAct::Leaky) p.outScale = static_cast<float>(a.floats[fi++]);
  if (Act == ConvAct::Leaky) p.slope = static_cast<float>(a.floats[fi++]);
  kernels::cpu::conv1d_fused_f32(p);
}

// conv1d_{snake,leaky,affine}_wino[_res]_f32_cpu(x, u, tr, b, affine, [alpha,] [res]) outs(out, stats)
// ints: x(Ci, L), u(A, Co/32, Ci·G, 32), tr(A·A + 4A), b(Co), affine(3, Ci), [alpha(Ci)], [res(Co, Lo)], out(Co, Lo),
//       stats(Co, S, 2), then attrs dilation, groups, kernel, padding, tile; floats (by name): [oscale (leaky res)], [slope]
template <ConvAct Act, bool Res>
void runConvWino(ExecCtx&, const KernelArgs& a) {
  constexpr bool hasAlpha = Act == ConvAct::Snake;
  const int64_t* n = a.ints;
  ConvFusedArgs p;
  p.x = F(a, 0);
  const float* u = F(a, 1);
  const float* tr = F(a, 2);
  p.bias = F(a, 3);
  p.affine = F(a, 4);
  int pi = 5;
  if (hasAlpha) p.alpha = F(a, pi++);
  if (Res) p.residual = F(a, pi++);
  p.out = O(a, pi++);
  p.stats = O(a, pi++);
  p.ci = n[0];
  p.l = n[1];
  p.co = n[7];
  const int attr = 15 + (hasAlpha ? 1 : 0) + (Res ? 2 : 0);
  p.dilation = n[attr];
  const int64_t groups = n[attr + 1];
  p.k = n[attr + 2];
  p.padding = n[attr + 3];
  p.tile = n[attr + 4];
  p.act = Act;
  int fi = 0;
  if (Res && Act == ConvAct::Leaky) p.outScale = static_cast<float>(a.floats[fi++]);
  if (Act == ConvAct::Leaky) p.slope = static_cast<float>(a.floats[fi++]);
  kernels::cpu::conv1d_wino_f32(p, u, tr, groups);
}

// conv1d_leaky_wino_mean3_f32_cpu(x, x2, x3, u, tr, b, affine) outs(out, stats): conv1d_leaky_wino on
// ((x + x2) + x3) * in_scale. ints: x, x2, x3 (Ci, L) each, u(4), tr(1), b(1), affine(2), out(2), stats(3), then attrs
// dilation, groups, kernel, padding, tile; floats: in_scale, slope
void runConvWinoMean3(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  ConvFusedArgs p;
  p.x = F(a, 0);
  p.x2 = F(a, 1);
  p.x3 = F(a, 2);
  const float* u = F(a, 3);
  const float* tr = F(a, 4);
  p.bias = F(a, 5);
  p.affine = F(a, 6);
  p.out = O(a, 7);
  p.stats = O(a, 8);
  p.ci = n[0];
  p.l = n[1];
  p.co = n[11];
  const int attr = 6 + 4 + 1 + 1 + 2 + 2 + 3;
  p.dilation = n[attr];
  const int64_t groups = n[attr + 1];
  p.k = n[attr + 2];
  p.padding = n[attr + 3];
  p.tile = n[attr + 4];
  p.act = ConvAct::Leaky;
  p.inScale = static_cast<float>(a.floats[0]);
  p.slope = static_cast<float>(a.floats[1]);
  kernels::cpu::conv1d_wino_f32(p, u, tr, groups);
}

// ints: x(C, L), stats(C, S, 2)
void runChannelStats(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::channel_stats_f32(F(a, 0), O(a, 1), a.ints[0], a.ints[1]);
}

// ints: stats(C, S, 2), gb(2C), affine(3, C), length; floats: eps
void runAdainAffine(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::adain_affine_f32(F(a, 0), F(a, 1), O(a, 2), n[0], n[6], static_cast<float>(a.floats[0]));
}

}  // namespace

void registerCpuKokoroConvFusedKernels() {
  registerKernel({"conv1d_snake_f32_cpu", Device::Cpu, nullptr, runConvFused<ConvAct::Snake, false>, nullptr});
  registerKernel({"conv1d_snake_res_f32_cpu", Device::Cpu, nullptr, runConvFused<ConvAct::Snake, true>, nullptr});
  registerKernel({"conv1d_snake_wino_f32_cpu", Device::Cpu, nullptr, runConvWino<ConvAct::Snake, false>, nullptr});
  registerKernel({"conv1d_snake_wino_res_f32_cpu", Device::Cpu, nullptr, runConvWino<ConvAct::Snake, true>, nullptr});
  registerKernel({"conv1d_leaky_wino_f32_cpu", Device::Cpu, nullptr, runConvWino<ConvAct::Leaky, false>, nullptr});
  registerKernel({"conv1d_leaky_wino_res_f32_cpu", Device::Cpu, nullptr, runConvWino<ConvAct::Leaky, true>, nullptr});
  registerKernel({"conv1d_leaky_wino_mean3_f32_cpu", Device::Cpu, nullptr, runConvWinoMean3, nullptr});
  registerKernel({"conv1d_affine_wino_f32_cpu", Device::Cpu, nullptr, runConvWino<ConvAct::None, false>, nullptr});
  registerKernel({"conv1d_leaky_f32_cpu", Device::Cpu, nullptr, runConvFused<ConvAct::Leaky, false>, nullptr});
  registerKernel({"conv1d_leaky_res_f32_cpu", Device::Cpu, nullptr, runConvFused<ConvAct::Leaky, true>, nullptr});
  registerKernel({"conv1d_affine_f32_cpu", Device::Cpu, nullptr, runConvFused<ConvAct::None, false>, nullptr});
  registerKernel({"channel_stats_f32_cpu", Device::Cpu, nullptr, runChannelStats, nullptr});
  registerKernel({"adain_affine_f32_cpu", Device::Cpu, nullptr, runAdainAffine, nullptr});
}

}  // namespace lk::rt
