// f16 basic CPU kernels (Qwen3-ASR phase 1, docs/impl-notes/qwen3-asr-port-plan.md Q2): simple and readable, the
// reference oracle for the fused f16 kernels. f16 storage (NEON f16 loads / stores), f32 math and accumulation, one
// rounding to f16 at the output. The op contract (signatures, semantics) is docs/impl-notes/qwen3-asr-f16-kernels.md;
// the type rules live in python/lokinfer/ops/f16.py.
//
// Kernel ABI (lokinfer/runtime/kernel.h): ptrs = operands then outputs; ints = every operand's shape dims, then the
// outputs' dims, then int / dim attrs in attribute-name order; floats = float attrs in attribute-name order.
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "lokinfer/runtime/kernel.h"
#include "lokinfer/runtime/thread_pool.h"

namespace lk::rt {

namespace {

using h16 = __fp16;

const h16* H(const KernelArgs& a, int i) { return static_cast<const h16*>(a.ptrs[i]); }
h16* Hout(const KernelArgs& a, int i) { return static_cast<h16*>(a.ptrs[i]); }
int64_t I64(const KernelArgs& a, int i) { return *static_cast<const int64_t*>(a.ptrs[i]); }

// --- helpers --------------------------------------------------------------------------------------------------

inline void load8(const h16* p, float32x4_t& lo, float32x4_t& hi) {
  const float16x8_t h = vld1q_f16(p);
  lo = vcvt_f32_f16(vget_low_f16(h));
  hi = vcvt_high_f32_f16(h);
}

inline void store8(h16* p, float32x4_t lo, float32x4_t hi) {
  vst1q_f16(p, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
}

// 0.5·v·(1 + erf(v/√2)) written as 0.5·v·erfc(-v/√2): no cancellation in the negative tail (1 + erf(z) for z << 0
// would lose all f32 digits right where f16 still resolves the result).
inline float geluErf(float v) { return 0.5f * v * std::erfc(-v * 0.70710678118654752f); }

// fn(begin, end) over [0, n) in contiguous ranges on the thread pool; inline when n·cost is small.
void parallelRanges(int64_t n, int64_t work, const std::function<void(int64_t, int64_t)>& fn) {
  ThreadPool& pool = ThreadPool::global();
  const int64_t tasks = work < (int64_t{1} << 16) ? 1 : std::min<int64_t>(n, 4 * pool.threads());
  if (tasks <= 1) {
    if (n > 0) fn(0, n);
    return;
  }
  const int64_t per = (n + tasks - 1) / tasks;
  pool.parallelFor((n + per - 1) / per, [&](int64_t i) { fn(i * per, std::min(n, (i + 1) * per)); });
}

// Elementwise over n f16 values: out[i] = f(a[i], b[i]) in f32, one rounding (b may be null).
template <typename F>
void elementwise(const h16* a, const h16* b, h16* out, int64_t n, F f) {
  parallelRanges(n, n, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) out[i] = static_cast<h16>(f(static_cast<float>(a[i]), b ? static_cast<float>(b[i]) : 0.f));
  });
}

// --- GEMM: out[M, N] = epi(x[M, K] @ w[K, N] (+ bias)) --------------------------------------------------------
// Tiles of 4 rows × 16 columns in f32 registers; K in order (sequential f32 accumulation). Tasks: 16-column strips
// (each strip of w, K × 32 bytes, stays in cache while every row block streams past it).

template <int MB>
void tile16(const h16* x, int64_t ldx, const h16* w, int64_t ldw, int64_t k, float32x4_t (&acc)[4][4]) {
  for (int r = 0; r < MB; ++r)
    for (int c = 0; c < 4; ++c) acc[r][c] = vdupq_n_f32(0.f);
  for (int64_t kk = 0; kk < k; ++kk) {
    float32x4_t w0, w1, w2, w3;
    load8(w + kk * ldw, w0, w1);
    load8(w + kk * ldw + 8, w2, w3);
    for (int r = 0; r < MB; ++r) {
      const float32x4_t xv = vdupq_n_f32(static_cast<float>(x[r * ldx + kk]));
      acc[r][0] = vfmaq_f32(acc[r][0], xv, w0);
      acc[r][1] = vfmaq_f32(acc[r][1], xv, w1);
      acc[r][2] = vfmaq_f32(acc[r][2], xv, w2);
      acc[r][3] = vfmaq_f32(acc[r][3], xv, w3);
    }
  }
}

inline float epilogue(float v, const h16* bias, int64_t n, bool gelu) {
  if (bias) v += static_cast<float>(bias[n]);
  return gelu ? geluErf(v) : v;
}

void gemmF16(const h16* x, const h16* w, const h16* bias, bool gelu, h16* out, int64_t m, int64_t k, int64_t n) {
  const int64_t strips = (n + 15) / 16;
  auto strip = [&](int64_t s) {
    const int64_t n0 = s * 16, nb = std::min<int64_t>(16, n - n0);
    for (int64_t m0 = 0; m0 < m; m0 += 4) {
      const int64_t mb = std::min<int64_t>(4, m - m0);
      const h16* xr = x + m0 * k;
      if (nb < 16) {  // ragged last strip: plain scalar dot products
        for (int64_t r = 0; r < mb; ++r)
          for (int64_t c = n0; c < n; ++c) {
            float acc = 0.f;
            for (int64_t kk = 0; kk < k; ++kk) acc += static_cast<float>(xr[r * k + kk]) * static_cast<float>(w[kk * n + c]);
            out[(m0 + r) * n + c] = static_cast<h16>(epilogue(acc, bias, c, gelu));
          }
        continue;
      }
      float32x4_t acc[4][4];
      switch (mb) {
        case 4: tile16<4>(xr, k, w + n0, n, k, acc); break;
        case 3: tile16<3>(xr, k, w + n0, n, k, acc); break;
        case 2: tile16<2>(xr, k, w + n0, n, k, acc); break;
        default: tile16<1>(xr, k, w + n0, n, k, acc); break;
      }
      for (int64_t r = 0; r < mb; ++r) {
        float v[16];
        for (int c = 0; c < 4; ++c) vst1q_f32(v + 4 * c, acc[r][c]);
        h16* o = out + (m0 + r) * n + n0;
        for (int c = 0; c < 16; ++c) o[c] = static_cast<h16>(epilogue(v[c], bias, n0 + c, gelu));
      }
    }
  };
  if (2 * m * n * k < (int64_t{1} << 21)) {
    for (int64_t s = 0; s < strips; ++s) strip(s);
  } else {
    ThreadPool::global().parallelFor(strips, strip);
  }
}

// ptrs: x, w, out. ints: x(M, K), w(K, N), out(M, N)
void runGemm(ExecCtx&, const KernelArgs& a) {
  gemmF16(H(a, 0), H(a, 1), nullptr, false, Hout(a, 2), a.ints[0], a.ints[1], a.ints[3]);
}

// ptrs: x, w, b, out. ints: x(M, K), w(K, N), b(N), out(M, N)
void runGemmBias(ExecCtx&, const KernelArgs& a) {
  gemmF16(H(a, 0), H(a, 1), H(a, 2), false, Hout(a, 3), a.ints[0], a.ints[1], a.ints[3]);
}

// --- dot products (LM head, attention) -------------------------------------------------------------------------

// Σ q[c]·k[c] over d (d % 8 == 0), q f32, k f16.
inline float dotF32F16(const float* q, const h16* k, int64_t d) {
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0;
  for (int64_t c = 0; c < d; c += 8) {
    float32x4_t lo, hi;
    load8(k + c, lo, hi);
    a0 = vfmaq_f32(a0, vld1q_f32(q + c), lo);
    a1 = vfmaq_f32(a1, vld1q_f32(q + c + 4), hi);
  }
  return vaddvq_f32(vaddq_f32(a0, a1));
}

// ptrs: x, table, out(f32). ints: x(T, D), table(V, D), out(1, V). out[v] = x[T-1] · table[v] in f32.
void runLmHeadLast(ExecCtx&, const KernelArgs& a) {
  const int64_t t = a.ints[0], d = a.ints[1], v = a.ints[2];
  std::vector<float> xf(static_cast<size_t>(d));
  const h16* xr = H(a, 0) + (t - 1) * d;
  for (int64_t c = 0; c < d; ++c) xf[c] = static_cast<float>(xr[c]);
  const h16* table = H(a, 1);
  float* out = static_cast<float*>(a.ptrs[2]);
  parallelRanges(v, v * d, [&](int64_t v0, int64_t v1) {
    for (int64_t r = v0; r < v1; ++r) {
      if (d % 8 == 0) {
        out[r] = dotF32F16(xf.data(), table + r * d, d);
      } else {
        float acc = 0.f;
        for (int64_t c = 0; c < d; ++c) acc += xf[c] * static_cast<float>(table[r * d + c]);
        out[r] = acc;
      }
    }
  });
}

// --- elementwise ------------------------------------------------------------------------------------------------

// ptrs: x, out. ints: x(R, C), out(R, C)
void runGeluErf(ExecCtx&, const KernelArgs& a) {
  elementwise(H(a, 0), nullptr, Hout(a, 1), a.ints[0] * a.ints[1], [](float x, float) { return geluErf(x); });
}

// ptrs: x, y, out. ints: x(R, C), y(R, C), out(R, C)
void runAdd(ExecCtx&, const KernelArgs& a) {
  elementwise(H(a, 0), H(a, 1), Hout(a, 2), a.ints[0] * a.ints[1], [](float x, float y) { return x + y; });
}

// ptrs: x, y, out. ints: x(R, C), y(R, C), out(R, C). floats: lim
void runAddClamp(ExecCtx&, const KernelArgs& a) {
  const float lim = static_cast<float>(a.floats[0]);
  elementwise(H(a, 0), H(a, 1), Hout(a, 2), a.ints[0] * a.ints[1],
              [lim](float x, float y) { return std::min(lim, std::max(-lim, x + y)); });
}

// ptrs: g, u, out. ints: g(R, C), u(R, C), out(R, C)
void runSiluMul(ExecCtx&, const KernelArgs& a) {
  elementwise(H(a, 0), H(a, 1), Hout(a, 2), a.ints[0] * a.ints[1],
              [](float g, float u) { return g / (1.f + std::exp(-g)) * u; });
}

// --- normalization ----------------------------------------------------------------------------------------------

// ptrs: x, g, b, out. ints: x(R, C), g(C), b(C), out(R, C). floats: eps.
// (x - mean) / sqrt(var + eps) · g + b; two-pass mean / variance in f32 over the f16 row.
void runLayerNorm(ExecCtx&, const KernelArgs& a) {
  const int64_t rows = a.ints[0], c = a.ints[1];
  const float eps = static_cast<float>(a.floats[0]);
  const h16 *x = H(a, 0), *g = H(a, 1), *b = H(a, 2);
  h16* out = Hout(a, 3);
  parallelRanges(rows, rows * c, [&](int64_t r0, int64_t r1) {
    std::vector<float> row(static_cast<size_t>(c));
    for (int64_t r = r0; r < r1; ++r) {
      const h16* xr = x + r * c;
      float sum = 0.f;
      for (int64_t j = 0; j < c; ++j) sum += (row[j] = static_cast<float>(xr[j]));
      const float mean = sum / static_cast<float>(c);
      float sq = 0.f;
      for (int64_t j = 0; j < c; ++j) sq += (row[j] - mean) * (row[j] - mean);
      const float rstd = 1.f / std::sqrt(sq / static_cast<float>(c) + eps);
      for (int64_t j = 0; j < c; ++j)
        out[r * c + j] = static_cast<h16>((row[j] - mean) * rstd * static_cast<float>(g[j]) + static_cast<float>(b[j]));
    }
  });
}

// HF Qwen3RMSNorm over consecutive groups of `width` values: h = f16(x · rsqrt(mean(x²) + eps)), y = f16(g · h).
void rmsGroups(const h16* x, const h16* g, h16* out, int64_t groups, int64_t width, float eps) {
  parallelRanges(groups, groups * width, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const h16* xr = x + i * width;
      float sq = 0.f;
      for (int64_t j = 0; j < width; ++j) sq += static_cast<float>(xr[j]) * static_cast<float>(xr[j]);
      const float r = 1.f / std::sqrt(sq / static_cast<float>(width) + eps);
      for (int64_t j = 0; j < width; ++j) {
        const h16 h = static_cast<h16>(static_cast<float>(xr[j]) * r);
        out[i * width + j] = static_cast<h16>(static_cast<float>(g[j]) * static_cast<float>(h));
      }
    }
  });
}

// ptrs: x, g, out. ints: x(R, C), g(C), out(R, C). floats: eps
void runRmsNorm(ExecCtx&, const KernelArgs& a) {
  rmsGroups(H(a, 0), H(a, 1), Hout(a, 2), a.ints[0], a.ints[1], static_cast<float>(a.floats[0]));
}

// ptrs: x, g, out. ints: x(R, C), g(hd), out(R, C). floats: eps. Each hd-wide head of every row normalized alone.
void runRmsNormHeads(ExecCtx&, const KernelArgs& a) {
  const int64_t hd = a.ints[2];
  rmsGroups(H(a, 0), H(a, 1), Hout(a, 2), a.ints[0] * a.ints[1] / hd, hd, static_cast<float>(a.floats[0]));
}

// --- RoPE -------------------------------------------------------------------------------------------------------

// ptrs: x, pos, out. ints: x(T, C), pos(1), out(T, C). floats: theta.
// rotate_half per 128-wide head at position p = pos + i, angles exactly as HF's Qwen3RotaryEmbedding (which the f64
// golden runs as well): inv_freq[j] = 1 / theta^(2j/128) in f32, angle = f32(p) · inv_freq[j] rounded to f32; cos / sin
// of that f32 angle evaluated in f64 and rounded to f32.
void runRope(ExecCtx& ctx, const KernelArgs& a) {
  constexpr int64_t kHd = 128, kHalf = 64;
  const int64_t t = a.ints[0], c = a.ints[1];
  const int64_t pos = I64(a, 1);
  if (pos < 0) {
    ctx.error = "rope: negative position";
    return;
  }
  const float theta = static_cast<float>(a.floats[0]);
  float invFreq[kHalf];
  for (int64_t j = 0; j < kHalf; ++j)
    invFreq[j] = 1.f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(kHd));
  const h16* x = H(a, 0);
  h16* out = Hout(a, 2);
  parallelRanges(t, t * c, [&](int64_t i0, int64_t i1) {
    float cs[kHalf], sn[kHalf];
    for (int64_t i = i0; i < i1; ++i) {
      const float p = static_cast<float>(pos + i);
      for (int64_t j = 0; j < kHalf; ++j) {
        const double ang = static_cast<double>(p * invFreq[j]);
        cs[j] = static_cast<float>(std::cos(ang));
        sn[j] = static_cast<float>(std::sin(ang));
      }
      for (int64_t h0 = 0; h0 < c; h0 += kHd) {
        const h16* xr = x + i * c + h0;
        h16* o = out + i * c + h0;
        for (int64_t j = 0; j < kHalf; ++j) {
          const float lo = static_cast<float>(xr[j]), hi = static_cast<float>(xr[j + kHalf]);
          o[j] = static_cast<h16>(lo * cs[j] - hi * sn[j]);
          o[j + kHalf] = static_cast<h16>(hi * cs[j] + lo * sn[j]);
        }
      }
    }
  });
}

// --- gathers / scatters -----------------------------------------------------------------------------------------

// ptrs: x, pos, out. ints: x(n, P, C), pos(P, C), out(rows, C), rows. y = (x + pos[None]).reshape(P·n, C)[:rows].
void runAddPosPrefix(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t n = a.ints[0], p = a.ints[1], c = a.ints[2], rows = a.ints[5];
  if (rows < 0 || rows > n * p) {
    ctx.error = "add_pos_prefix: rows exceeds 13 * n";
    return;
  }
  const h16 *x = H(a, 0), *pe = H(a, 1);
  h16* out = Hout(a, 2);
  parallelRanges(rows, rows * c, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const h16* pr = pe + (r % p) * c;
      for (int64_t j = 0; j < c; ++j)
        out[r * c + j] = static_cast<h16>(static_cast<float>(x[r * c + j]) + static_cast<float>(pr[j]));
    }
  });
}

// ptrs: ids, table, out. ints: ids(T), table(V, D), out(T, D)
void runEmbedding(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], v = a.ints[1], d = a.ints[2];
  const int64_t* ids = static_cast<const int64_t*>(a.ptrs[0]);
  for (int64_t i = 0; i < t; ++i)
    if (ids[i] < 0 || ids[i] >= v) {
      ctx.error = "embedding: token id out of range";
      return;
    }
  for (int64_t i = 0; i < t; ++i)
    std::memcpy(Hout(a, 2) + i * d, H(a, 1) + ids[i] * d, static_cast<size_t>(d) * sizeof(h16));
}

// ptrs: x, ids, audio, out. ints: x(T, D), ids(T), audio(A, D), out(T, D), pad_id.
// The j-th row with ids == pad_id (in order) becomes audio[j]; every other row is a copy of x.
void runScatterAudio(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], d = a.ints[1], na = a.ints[3], padId = a.ints[7];
  const int64_t* ids = static_cast<const int64_t*>(a.ptrs[1]);
  int64_t pads = 0;
  for (int64_t i = 0; i < t; ++i) pads += ids[i] == padId;
  if (pads > na) {
    ctx.error = "scatter_audio: more audio pad ids than audio rows";
    return;
  }
  int64_t j = 0;
  for (int64_t i = 0; i < t; ++i) {
    const h16* src = ids[i] == padId ? H(a, 2) + (j++) * d : H(a, 0) + i * d;
    std::memcpy(Hout(a, 3) + i * d, src, static_cast<size_t>(d) * sizeof(h16));
  }
}

// --- attention --------------------------------------------------------------------------------------------------

constexpr int64_t kMaxHd = 256;
constexpr int64_t kKeyBlock = 64;

// out[hd] = softmax(scale · q · Kᵀ) V over n keys (rows j of k / v at stride ld), streaming: blocks of 64 scores, a
// running max m and sum l (online softmax), the f32 accumulator rescaled when the max grows. hd % 8 == 0, n >= 1.
void attendRow(const float* q, const h16* k, const h16* v, int64_t ld, int64_t n, int64_t hd, float scale, h16* out) {
  float acc[kMaxHd] = {};
  float s[kKeyBlock];
  float m = -std::numeric_limits<float>::infinity(), l = 0.f;
  for (int64_t j0 = 0; j0 < n; j0 += kKeyBlock) {
    const int64_t nb = std::min(kKeyBlock, n - j0);
    float bm = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < nb; ++j) {
      s[j] = dotF32F16(q, k + (j0 + j) * ld, hd) * scale;
      bm = std::max(bm, s[j]);
    }
    if (bm > m) {
      const float corr = std::exp(m - bm);  // 0 for the first block
      l *= corr;
      for (int64_t c = 0; c < hd; ++c) acc[c] *= corr;
      m = bm;
    }
    for (int64_t j = 0; j < nb; ++j) {
      const float p = std::exp(s[j] - m);
      l += p;
      const float32x4_t pv = vdupq_n_f32(p);
      const h16* vr = v + (j0 + j) * ld;
      for (int64_t c = 0; c < hd; c += 8) {
        float32x4_t lo, hi;
        load8(vr + c, lo, hi);
        vst1q_f32(acc + c, vfmaq_f32(vld1q_f32(acc + c), pv, lo));
        vst1q_f32(acc + c + 4, vfmaq_f32(vld1q_f32(acc + c + 4), pv, hi));
      }
    }
  }
  const float32x4_t inv = vdupq_n_f32(1.f / l);
  for (int64_t c = 0; c < hd; c += 8) store8(out + c, vmulq_f32(vld1q_f32(acc + c), inv), vmulq_f32(vld1q_f32(acc + c + 4), inv));
}

inline void toF32(const h16* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] = static_cast<float>(x[i]);
}

// ptrs: q, k, v, out. ints: q(S, D), k(S, D), v(S, D), out(S, D), heads, window.
// Non-causal attention inside blocks of `window` rows: row i attends to rows [⌊i/w⌋·w, min(S, (⌊i/w⌋ + 1)·w)).
void runAttentionWindow(ExecCtx&, const KernelArgs& a) {
  const int64_t s = a.ints[0], d = a.ints[1], heads = a.ints[8], window = a.ints[9], hd = d / heads;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const h16 *q = H(a, 0), *k = H(a, 1), *v = H(a, 2);
  h16* out = Hout(a, 3);
  parallelRanges(s * heads, s * d * window, [&](int64_t t0, int64_t t1) {
    float qf[kMaxHd];
    for (int64_t task = t0; task < t1; ++task) {
      const int64_t i = task / heads, off = (task % heads) * hd;
      const int64_t w0 = i / window * window, w1 = std::min(s, w0 + window);
      toF32(q + i * d + off, qf, hd);
      attendRow(qf, k + w0 * d + off, v + w0 * d + off, d, w1 - w0, hd, scale, out + i * d + off);
    }
  });
}

// ptrs: kc, vc, k, v, pos. ints: kc(cap, D), vc(cap, D), k(T, D), v(T, D), pos(1). kc[pos:pos+T] = k, same for v.
void runKvWrite(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t cap = a.ints[0], d = a.ints[1], t = a.ints[4];
  const int64_t pos = I64(a, 4);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "kv_write: pos + T exceeds the cache";
    return;
  }
  const size_t bytes = static_cast<size_t>(t * d) * sizeof(h16);
  std::memcpy(static_cast<h16*>(a.ptrs[0]) + pos * d, H(a, 2), bytes);
  std::memcpy(static_cast<h16*>(a.ptrs[1]) + pos * d, H(a, 3), bytes);
}

// ptrs: q, kc, vc, pos, out. ints: q(T, Dq), kc(cap, Dk), vc(cap, Dk), pos(1), out(T, Dq), kv_heads, q_heads.
// Causal GQA over the cache: query row i (position pos + i) of q-head h attends to cache rows 0 .. pos + i of
// kv-head h / (q_heads / kv_heads).
void runAttentionGqa(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], dq = a.ints[1], cap = a.ints[2], dk = a.ints[3];
  const int64_t kvHeads = a.ints[9], qHeads = a.ints[10], hd = dq / qHeads, group = qHeads / kvHeads;
  const int64_t pos = I64(a, 3);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "attention_gqa: pos + T exceeds the cache";
    return;
  }
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const h16 *q = H(a, 0), *kc = H(a, 1), *vc = H(a, 2);
  h16* out = Hout(a, 4);
  // One task per (query row, q-head); the work of a row grows with its position.
  ThreadPool::global().parallelFor(t * qHeads, [&](int64_t task) {
    float qf[kMaxHd];
    const int64_t i = task / qHeads, h = task % qHeads, kvOff = (h / group) * hd;
    toF32(q + i * dq + h * hd, qf, hd);
    attendRow(qf, kc + kvOff, vc + kvOff, dk, pos + i + 1, hd, scale, out + i * dq + h * hd);
  });
}

// --- audio CNN front end ------------------------------------------------------------------------------------------

// im2col of a 3×3, stride 2, pad 1 conv over NHWC x[h, w, c]: rows (ho, wo), columns (kt, kf, ci); zeros outside.
void im2col3x3s2(const h16* x, int64_t h, int64_t w, int64_t c, h16* cols, int64_t ho, int64_t wo) {
  const size_t seg = static_cast<size_t>(c) * sizeof(h16);
  for (int64_t oy = 0; oy < ho; ++oy)
    for (int64_t ox = 0; ox < wo; ++ox) {
      h16* row = cols + (oy * wo + ox) * 9 * c;
      for (int64_t kt = 0; kt < 3; ++kt)
        for (int64_t kf = 0; kf < 3; ++kf) {
          const int64_t iy = 2 * oy - 1 + kt, ix = 2 * ox - 1 + kf;
          h16* dst = row + (kt * 3 + kf) * c;
          if (iy < 0 || iy >= h || ix < 0 || ix >= w)
            std::memset(dst, 0, seg);
          else
            std::memcpy(dst, x + (iy * w + ix) * c, seg);
        }
    }
}

// [Co, K] -> [K, Co] (conv weights [Co, kt, kf, Ci] as GEMM B operands).
std::vector<h16> transposed(const h16* w, int64_t co, int64_t k) {
  std::vector<h16> t(static_cast<size_t>(co * k));
  for (int64_t o = 0; o < co; ++o)
    for (int64_t j = 0; j < k; ++j) t[j * co + o] = w[o * k + j];
  return t;
}

// ptrs: features, w1, b1, w2, b2, w3, b3, w_out, out.
// ints: features(n, 128, 100), w1(480, 3, 3, 1), b1(480), w2(480, 3, 3, 480), b2(480), w3(480, 3, 3, 480), b3(480),
//       w_out(7680, 1024), out(n, 13, 1024).
// Per 1 s chunk: x = features[c]ᵀ as NHWC [100 time, 128 freq, 1] in f16; three conv 3×3 s2 p1 + bias + GELU(erf)
// layers (im2col + GEMM, each layer's output rounded to f16) -> [13, 16, 480] = [13, 7680] (freq, channel);
// @ w_out -> [13, 1024]. Scratch is per chunk (reused), never all chunks' intermediates.
void runAudioCnn(ExecCtx&, const KernelArgs& a) {
  constexpr int64_t kF = 128, kT = 100, kC = 480;
  constexpr int64_t kH1 = 50, kW1 = 64, kH2 = 25, kW2 = 32, kH3 = 13, kW3 = 16, kOut = 1024;
  const int64_t n = a.ints[0];
  const float* feats = static_cast<const float*>(a.ptrs[0]);
  const h16 *b1 = H(a, 2), *b2 = H(a, 4), *b3 = H(a, 6), *wOut = H(a, 7);
  h16* out = Hout(a, 8);
  const std::vector<h16> w1t = transposed(H(a, 1), kC, 9), w2t = transposed(H(a, 3), kC, 9 * kC),
                         w3t = transposed(H(a, 5), kC, 9 * kC);
  std::vector<h16> x0(kT * kF), cols(kH2 * kW2 * 9 * kC), a1(kH1 * kW1 * kC), a2(kH2 * kW2 * kC), a3(kH3 * kW3 * kC);
  for (int64_t ch = 0; ch < n; ++ch) {
    const float* f = feats + ch * kF * kT;
    for (int64_t t = 0; t < kT; ++t)
      for (int64_t q = 0; q < kF; ++q) x0[t * kF + q] = static_cast<h16>(f[q * kT + t]);
    im2col3x3s2(x0.data(), kT, kF, 1, cols.data(), kH1, kW1);
    gemmF16(cols.data(), w1t.data(), b1, true, a1.data(), kH1 * kW1, 9, kC);
    im2col3x3s2(a1.data(), kH1, kW1, kC, cols.data(), kH2, kW2);
    gemmF16(cols.data(), w2t.data(), b2, true, a2.data(), kH2 * kW2, 9 * kC, kC);
    im2col3x3s2(a2.data(), kH2, kW2, kC, cols.data(), kH3, kW3);
    gemmF16(cols.data(), w3t.data(), b3, true, a3.data(), kH3 * kW3, 9 * kC, kC);
    gemmF16(a3.data(), wOut, nullptr, false, out + ch * kH3 * kOut, kH3, kW3 * kC, kOut);
  }
}

}  // namespace

void registerCpuF16BasicKernels() {
  registerKernel({"audio_cnn_f16_cpu", Device::Cpu, nullptr, runAudioCnn, nullptr});
  registerKernel({"add_pos_prefix_f16_cpu", Device::Cpu, nullptr, runAddPosPrefix, nullptr});
  registerKernel({"gemm_f16_cpu", Device::Cpu, nullptr, runGemm, nullptr});
  registerKernel({"gemm_bias_f16_cpu", Device::Cpu, nullptr, runGemmBias, nullptr});
  registerKernel({"lm_head_last_f16_cpu", Device::Cpu, nullptr, runLmHeadLast, nullptr});
  registerKernel({"gelu_erf_f16_cpu", Device::Cpu, nullptr, runGeluErf, nullptr});
  registerKernel({"layernorm_f16_cpu", Device::Cpu, nullptr, runLayerNorm, nullptr});
  registerKernel({"rmsnorm_f16_cpu", Device::Cpu, nullptr, runRmsNorm, nullptr});
  registerKernel({"rmsnorm_heads_f16_cpu", Device::Cpu, nullptr, runRmsNormHeads, nullptr});
  registerKernel({"rope_f16_cpu", Device::Cpu, nullptr, runRope, nullptr});
  registerKernel({"add_f16_cpu", Device::Cpu, nullptr, runAdd, nullptr});
  registerKernel({"add_clamp_f16_cpu", Device::Cpu, nullptr, runAddClamp, nullptr});
  registerKernel({"silu_mul_f16_cpu", Device::Cpu, nullptr, runSiluMul, nullptr});
  registerKernel({"embedding_f16_cpu", Device::Cpu, nullptr, runEmbedding, nullptr});
  registerKernel({"scatter_audio_f16_cpu", Device::Cpu, nullptr, runScatterAudio, nullptr});
  registerKernel({"attention_window_f16_cpu", Device::Cpu, nullptr, runAttentionWindow, nullptr});
  registerKernel({"kv_write_f16_cpu", Device::Cpu, nullptr, runKvWrite, nullptr});
  registerKernel({"attention_gqa_f16_cpu", Device::Cpu, nullptr, runAttentionGqa, nullptr});
}

}  // namespace lk::rt
