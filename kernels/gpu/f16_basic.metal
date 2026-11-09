// f16 basic GPU kernels (Qwen3-ASR phase 1, docs/impl-notes/qwen3-asr-port-plan.md Q2): half storage, float
// accumulators; the reference oracle for the fused f16 kernels. Dispatched by kernels/gpu/f16_gpu.cpp.
// The op contract (signatures, semantics) is docs/impl-notes/qwen3-asr-f16-kernels.md.
//
// Every kernel loads half, computes and accumulates in float, and rounds once to half at its output (the
// exceptions are written by the contract: rmsnorm rounds the normalized value before the weight multiply, and
// audio_cnn rounds each conv layer's output, like the reference). Simple on purpose (one thread per output element,
// one simdgroup per attention row / LM-head row, a 16x16 threadgroup-tiled GEMM): no simdgroup matrices here.
//
//   lk_f16_add, lk_f16_add_clamp, lk_f16_silu_mul, lk_f16_gelu_erf   elementwise
//   lk_f16_add_pos_prefix                                             + per-chunk positions, first `rows` rows
//   lk_f16_gemm                                                       x @ w (+ bias), 16x16 tiles
//   lk_f16_lm_head_last                                               x[T-1] . table rows -> float
//   lk_f16_layernorm, lk_f16_rmsnorm (also per head)                  one threadgroup per row
//   lk_f16_rope                                                       rotate_half RoPE at positions pos + i
//   lk_f16_embedding, lk_f16_scatter_audio                            row gathers
//   lk_f16_attention_window, lk_f16_attention_gqa                     streaming (online) softmax, per (row, head)
//   lk_f16_kv_write                                                   cache rows pos .. pos + T - 1
//   lk_f16_conv1_feat, lk_f16_conv3x3s2_gelu                          audio_cnn's convs (one chunk per dispatch)
#include <metal_stdlib>
using namespace metal;

// IEEE semantics in this file: precise exp / sin / cos / rsqrt, infinities kept (the online softmax starts at
// -INFINITY), no reassociation, and no contraction of a * b + c into fma (RoPE's angle must be the rounded f32
// product; fma is only used where written).
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// ---- helpers ------------------------------------------------------------------------------------------------------

// erfc(z), relative error < 1.2e-7 for every z (Numerical Recipes' erfcc Chebyshev fit), so 1 + erf(z) = erfc(-z)
// keeps its relative accuracy in GELU's negative tail (1 + erf computed directly would cancel).
inline float lk_erfc(float z) {
  const float a = fabs(z);
  const float t = 1.0f / (1.0f + 0.5f * a);
  const float poly =
      -1.26551223f +
      t * (1.00002368f +
           t * (0.37409196f +
                t * (0.09678418f +
                     t * (-0.18628806f +
                          t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
  const float r = t * exp(-a * a + poly);
  return z >= 0.0f ? r : 2.0f - r;
}

// GELU(erf): 0.5 x (1 + erf(x / sqrt 2)).
inline float lk_gelu_erf(float x) { return 0.5f * x * lk_erfc(-x * 0.70710678118654752f); }

// Sum over the threadgroup (every thread gets the same value, summed in the same order). `sh` holds one float per
// simdgroup; reusable right after the call.
inline float lk_tg_sum(float v, threadgroup float* sh, uint lid, uint nthreads) {
  v = simd_sum(v);
  if (lid % 32 == 0) sh[lid / 32] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = 0.0f;
  for (uint s = 0; s < (nthreads + 31) / 32; ++s) r += sh[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// ---- elementwise --------------------------------------------------------------------------------------------------

struct LkF16EltParams {
  uint n;
  float lim;
};

kernel void lk_f16_add(const device half* x [[buffer(0)]], const device half* y [[buffer(1)]],
                       device half* out [[buffer(2)]], constant LkF16EltParams& p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  out[i] = half(float(x[i]) + float(y[i]));
}

// clamp(x + y, -lim, lim) in float, then rounded (lim = 64504 rounds to 64512, as torch's f16 clamp gives).
kernel void lk_f16_add_clamp(const device half* x [[buffer(0)]], const device half* y [[buffer(1)]],
                             device half* out [[buffer(2)]], constant LkF16EltParams& p [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  out[i] = half(clamp(float(x[i]) + float(y[i]), -p.lim, p.lim));
}

// g * sigmoid(g) * u.
kernel void lk_f16_silu_mul(const device half* g [[buffer(0)]], const device half* u [[buffer(1)]],
                            device half* out [[buffer(2)]], constant LkF16EltParams& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  const float gv = float(g[i]);
  out[i] = half(gv / (1.0f + exp(-gv)) * float(u[i]));
}

kernel void lk_f16_gelu_erf(const device half* x [[buffer(0)]], device half* out [[buffer(1)]],
                            constant LkF16EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  out[i] = half(lk_gelu_erf(float(x[i])));
}

// x [n, period, d] viewed as [n * period, d], pos [period, d]: out[r, c] = x[r, c] + pos[r % period, c] for the
// first `rows` rows. Grid (d, rows).
struct LkF16PosParams {
  uint rows, d, period;
};
kernel void lk_f16_add_pos_prefix(const device half* x [[buffer(0)]], const device half* pos [[buffer(1)]],
                                  device half* out [[buffer(2)]], constant LkF16PosParams& p [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.d || r >= p.rows) return;
  const size_t i = size_t(r) * p.d + c;
  out[i] = half(float(x[i]) + float(pos[size_t(r % p.period) * p.d + c]));
}

// ---- GEMM ---------------------------------------------------------------------------------------------------------

// a [m, k] @ b [k, n] (+ bias [n]) -> out [m, n]. One 16x16 threadgroup per 16x16 output tile, one thread per output
// element; K walked in 16-wide tiles staged (as float) in threadgroup memory; zero padding outside the matrices
// (adds exact zeros). Accumulation in k order, in float.
struct LkF16GemmParams {
  uint m, n, k, has_bias;
};
kernel void lk_f16_gemm(const device half* a [[buffer(0)]], const device half* b [[buffer(1)]],
                        const device half* bias [[buffer(2)]], device half* out [[buffer(3)]],
                        constant LkF16GemmParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]]) {
  constexpr uint TS = 16;
  threadgroup float As[TS][TS];
  threadgroup float Bs[TS][TS + 1];
  const uint row = tg.y * TS + lid.y, col = tg.x * TS + lid.x;
  float acc = 0.0f;
  for (uint k0 = 0; k0 < p.k; k0 += TS) {
    const uint ka = k0 + lid.x, kb = k0 + lid.y;
    As[lid.y][lid.x] = (row < p.m && ka < p.k) ? float(a[size_t(row) * p.k + ka]) : 0.0f;
    Bs[lid.y][lid.x] = (kb < p.k && col < p.n) ? float(b[size_t(kb) * p.n + col]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < TS; ++kk) acc += As[lid.y][kk] * Bs[kk][lid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row >= p.m || col >= p.n) return;
  if (p.has_bias) acc += float(bias[col]);
  out[size_t(row) * p.n + col] = half(acc);
}

// out[v] = x[t - 1] . table[v] in float (the tied LM head, f32 logits). One simdgroup per vocabulary row (lanes
// stride over d, simd_sum), 8 rows per 256-thread threadgroup.
struct LkF16LmHeadParams {
  uint t, d, v;
};
kernel void lk_f16_lm_head_last(const device half* x [[buffer(0)]], const device half* table [[buffer(1)]],
                                device float* out [[buffer(2)]], constant LkF16LmHeadParams& p [[buffer(3)]],
                                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const uint v = tg * 8 + sg;
  if (v >= p.v) return;
  const device half* xr = x + size_t(p.t - 1) * p.d;
  const device half* tr = table + size_t(v) * p.d;
  float acc = 0.0f;
  for (uint c = lane; c < p.d; c += 32) acc += float(xr[c]) * float(tr[c]);
  acc = simd_sum(acc);
  if (lane == 0) out[v] = acc;
}

// ---- norms (one threadgroup per row; threads stride over the columns) ----------------------------------------------

struct LkF16NormParams {
  uint rows, cols;
  float eps;
};

// (x - mean) / sqrt(var + eps) * g + b, biased variance from the centered values (two passes), all float.
kernel void lk_f16_layernorm(const device half* x [[buffer(0)]], const device half* g [[buffer(1)]],
                             const device half* b [[buffer(2)]], device half* out [[buffer(3)]],
                             constant LkF16NormParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint nthreads [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  const device half* xr = x + size_t(row) * p.cols;
  float s = 0.0f;
  for (uint c = lid; c < p.cols; c += nthreads) s += float(xr[c]);
  const float mean = lk_tg_sum(s, sh, lid, nthreads) / float(p.cols);
  float q = 0.0f;
  for (uint c = lid; c < p.cols; c += nthreads) {
    const float d = float(xr[c]) - mean;
    q += d * d;
  }
  const float rstd = rsqrt(lk_tg_sum(q, sh, lid, nthreads) / float(p.cols) + p.eps);
  device half* orow = out + size_t(row) * p.cols;
  for (uint c = lid; c < p.cols; c += nthreads) orow[c] = half((float(xr[c]) - mean) * rstd * float(g[c]) + float(b[c]));
}

// HF Qwen3RMSNorm over rows of `cols` (the whole row, or one 128-wide head for the QK-norm): h = half(x * rsqrt(mean(x^2)
// + eps)), out = half(g * h).
kernel void lk_f16_rmsnorm(const device half* x [[buffer(0)]], const device half* g [[buffer(1)]],
                           device half* out [[buffer(2)]], constant LkF16NormParams& p [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                           uint nthreads [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  const device half* xr = x + size_t(row) * p.cols;
  float s = 0.0f;
  for (uint c = lid; c < p.cols; c += nthreads) {
    const float v = float(xr[c]);
    s += v * v;
  }
  const float r = rsqrt(lk_tg_sum(s, sh, lid, nthreads) / float(p.cols) + p.eps);
  device half* orow = out + size_t(row) * p.cols;
  for (uint c = lid; c < p.cols; c += nthreads) orow[c] = half(float(g[c]) * float(half(float(xr[c]) * r)));
}

// ---- RoPE ---------------------------------------------------------------------------------------------------------

// x [t, heads * 128] -> rotate_half RoPE per head at positions pos + i. inv_freq[j] = theta^(-2j/128) comes from the
// CPU (correctly rounded to float); angle = float(pos + i) * inv_freq[j] rounded once to float (the contract's f32
// angle, as the reference computes it); cos / sin of that float angle are evaluated accurately: Cody-Waite reduction
// by 2 pi (two-part constant, fma) to [-pi, pi], then precise cos / sin. Grid (heads * 64, t): thread (h, j) of row i
// writes elements j and j + 64 of head h.
struct LkF16RopeParams {
  uint t, heads;
  float two_pi_hi, two_pi_lo;  // 2 pi = hi + lo (float(2 pi), float(2 pi - hi))
  float inv_freq[64];
};
kernel void lk_f16_rope(const device half* x [[buffer(0)]], const device long* pos [[buffer(1)]],
                        device half* out [[buffer(2)]], constant LkF16RopeParams& p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const uint h = gid.x / 64, j = gid.x % 64, i = gid.y;
  if (h >= p.heads || i >= p.t) return;
  const float position = float(pos[0] + long(i));  // exact below 2^24
  const float angle = position * p.inv_freq[j];
  const float k = rint(angle * 0.15915494309189535f);
  float r = fma(-k, p.two_pi_hi, angle);  // exact: |angle - k * hi| is small and a multiple of hi's ulp
  r = fma(-k, p.two_pi_lo, r);
  const float c = cos(r), s = sin(r);
  const size_t base = size_t(i) * p.heads * 128 + h * 128;
  const float x0 = float(x[base + j]), x1 = float(x[base + j + 64]);
  out[base + j] = half(x0 * c - x1 * s);
  out[base + j + 64] = half(x1 * c + x0 * s);
}

// ---- gathers ------------------------------------------------------------------------------------------------------

// out[i] = table[ids[i]]. Ids are validated on the CPU before the dispatch (ctx.error); an out-of-range id still
// never reads outside the table here (its row is zero-filled). Grid (d, t).
struct LkF16EmbParams {
  uint t, d, v;
};
kernel void lk_f16_embedding(const device long* ids [[buffer(0)]], const device half* table [[buffer(1)]],
                             device half* out [[buffer(2)]], constant LkF16EmbParams& p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, i = gid.y;
  if (c >= p.d || i >= p.t) return;
  const long id = ids[i];
  out[size_t(i) * p.d + c] = (id >= 0 && id < long(p.v)) ? table[size_t(id) * p.d + c] : half(0.0h);
}

// out = x, except that the j-th row (in order) whose id is pad_id becomes audio[j]. One threadgroup per row: the
// threads count the pad ids before the row (j), then copy the row. A row beyond audio's rows (validated on the CPU)
// keeps x.
struct LkF16ScatterParams {
  uint t, d, a;
  long pad_id;
};
kernel void lk_f16_scatter_audio(const device half* x [[buffer(0)]], const device long* ids [[buffer(1)]],
                                 const device half* audio [[buffer(2)]], device half* out [[buffer(3)]],
                                 constant LkF16ScatterParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]],
                                 uint nthreads [[threads_per_threadgroup]]) {
  threadgroup float sh[32];
  float n = 0.0f;  // exact (counts <= 2^24)
  for (uint r = lid; r < row; r += nthreads) n += ids[r] == p.pad_id ? 1.0f : 0.0f;
  const uint j = uint(lk_tg_sum(n, sh, lid, nthreads));
  const bool from_audio = ids[row] == p.pad_id && j < p.a;
  const device half* src = from_audio ? audio + size_t(j) * p.d : x + size_t(row) * p.d;
  device half* dst = out + size_t(row) * p.d;
  for (uint c = lid; c < p.d; c += nthreads) dst[c] = src[c];
}

// ---- attention (streaming softmax; one simdgroup per (query row, head), 4 rows per threadgroup) --------------------

// One query row of one head against key rows [j0, j1): lanes hold head dims lane, lane + 32, ... (head_dim / 32 <= 4
// of them), the score is a simd_sum, then the online softmax update (running max m, sum l, accumulator acc), all in
// float. Writes acc / l rounded to half.
inline void lk_f16_attend_row(const device half* qrow, const device half* kbase, const device half* vbase,
                              uint kv_stride, uint j0, uint j1, uint per, float scale, device half* orow, uint lane) {
  float q[4], acc[4];
  for (uint e = 0; e < 4; ++e) {
    q[e] = e < per ? float(qrow[lane + 32 * e]) * scale : 0.0f;
    acc[e] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (uint j = j0; j < j1; ++j) {
    const device half* kr = kbase + size_t(j) * kv_stride;
    const device half* vr = vbase + size_t(j) * kv_stride;
    float s = 0.0f;
    for (uint e = 0; e < 4; ++e)
      if (e < per) s += q[e] * float(kr[lane + 32 * e]);
    s = simd_sum(s);
    const float m_new = max(m, s);
    const float corr = exp(m - m_new), w = exp(s - m_new);
    l = l * corr + w;
    for (uint e = 0; e < 4; ++e)
      if (e < per) acc[e] = acc[e] * corr + w * float(vr[lane + 32 * e]);
    m = m_new;
  }
  for (uint e = 0; e < 4; ++e)
    if (e < per) orow[lane + 32 * e] = half(acc[e] / l);
}

struct LkF16AttnParams {
  uint t;         // query rows
  uint q_heads, kv_heads, head_dim;
  uint window;    // attention_window
  uint cap;       // attention_gqa: cache rows
  float scale;
};

// Encoder self-attention, non-causal within windows: row i attends to rows [floor(i / w) * w, min(t, that + w)).
// q, k, v, out [t, q_heads * head_dim]. Grid: threadgroups (ceil(t / 4), q_heads) of 128 threads.
kernel void lk_f16_attention_window(const device half* q [[buffer(0)]], const device half* k [[buffer(1)]],
                                    const device half* v [[buffer(2)]], device half* out [[buffer(3)]],
                                    constant LkF16AttnParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint i = tg.x * 4 + sg, h = tg.y;
  if (i >= p.t || h >= p.q_heads) return;
  const uint d = p.q_heads * p.head_dim, ho = h * p.head_dim;
  const uint j0 = i / p.window * p.window, j1 = min(p.t, j0 + p.window);
  lk_f16_attend_row(q + size_t(i) * d + ho, k + ho, v + ho, d, j0, j1, p.head_dim / 32, p.scale,
                    out + size_t(i) * d + ho, lane);
}

// Causal GQA over the caches: query row i (position pos + i) of q-head h attends to cache rows 0 .. pos + i of kv-head
// h / (q_heads / kv_heads). q, out [t, q_heads * head_dim]; kc, vc [cap, kv_heads * head_dim]. pos + t <= cap is
// checked on the CPU (ctx.error); here an invalid pos writes nothing. Grid as lk_f16_attention_window.
kernel void lk_f16_attention_gqa(const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                 const device half* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                 device half* out [[buffer(4)]], constant LkF16AttnParams& p [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint i = tg.x * 4 + sg, h = tg.y;
  const long pos0 = pos[0];
  if (i >= p.t || h >= p.q_heads || pos0 < 0 || pos0 + long(p.t) > long(p.cap)) return;
  const uint dq = p.q_heads * p.head_dim, dkv = p.kv_heads * p.head_dim;
  const uint kvo = h / (p.q_heads / p.kv_heads) * p.head_dim;
  lk_f16_attend_row(q + size_t(i) * dq + h * p.head_dim, kc + kvo, vc + kvo, dkv, 0, uint(pos0) + i + 1,
                    p.head_dim / 32, p.scale, out + size_t(i) * dq + h * p.head_dim, lane);
}

// kc[pos + i] = k[i], vc[pos + i] = v[i]. Grid (d, t); rows outside the cache are skipped (the CPU flags them).
struct LkF16KvParams {
  uint t, d, cap;
};
kernel void lk_f16_kv_write(device half* kc [[buffer(0)]], device half* vc [[buffer(1)]],
                            const device half* k [[buffer(2)]], const device half* v [[buffer(3)]],
                            const device long* pos [[buffer(4)]], constant LkF16KvParams& p [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const uint e = gid.x, i = gid.y;
  const long r = pos[0] + long(i);
  if (e >= p.d || i >= p.t || r < 0 || r >= long(p.cap)) return;
  kc[size_t(r) * p.d + e] = k[size_t(i) * p.d + e];
  vc[size_t(r) * p.d + e] = v[size_t(i) * p.d + e];
}

// ---- audio_cnn: conv2d 3x3, stride 2, pad 1, + bias, GELU(erf), NHWC (H = time, W = freq), one chunk --------------

struct LkF16ConvParams {
  uint h_in, w_in, h_out, w_out, ci, co;
};

// First layer, straight from the features: x[t, f] = half(features[f, t]) (the chunk's [freq, time] f32 block,
// transposed and cast), Ci = 1. w [co, 3, 3, 1], out [h_out, w_out, co]. Grid (co, w_out, h_out).
kernel void lk_f16_conv1_feat(const device float* feat [[buffer(0)]], const device half* w [[buffer(1)]],
                              const device half* bias [[buffer(2)]], device half* out [[buffer(3)]],
                              constant LkF16ConvParams& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
  const uint co = gid.x, ow = gid.y, oh = gid.z;
  if (co >= p.co || ow >= p.w_out || oh >= p.h_out) return;
  float acc = float(bias[co]);
  for (uint kt = 0; kt < 3; ++kt) {
    const int ih = int(oh * 2 + kt) - 1;
    if (ih < 0 || ih >= int(p.h_in)) continue;
    for (uint kf = 0; kf < 3; ++kf) {
      const int iw = int(ow * 2 + kf) - 1;
      if (iw < 0 || iw >= int(p.w_in)) continue;
      const float xv = float(half(feat[size_t(iw) * p.h_in + uint(ih)]));
      acc += xv * float(w[(co * 3 + kt) * 3 + kf]);
    }
  }
  out[(size_t(oh) * p.w_out + ow) * p.co + co] = half(lk_gelu_erf(acc));
}

// Later layers: in [h_in, w_in, ci] half, w [co, 3, 3, ci], out [h_out, w_out, co], as an implicit GEMM with the
// tiling of lk_f16_gemm: out[m, co] = sum_kk A[m, kk] * w[co, kk] for output position m = oh * w_out + ow and
// kk = (kt * 3 + kf) * ci + c (the weight's own flattening), A[m, kk] = in[2 oh + kt - 1, 2 ow + kf - 1, c] (0 in the
// padding). Grid: threadgroups (ceil(co / 16), ceil(h_out * w_out / 16)) of 16x16 threads.
kernel void lk_f16_conv3x3s2_gelu(const device half* in [[buffer(0)]], const device half* w [[buffer(1)]],
                                  const device half* bias [[buffer(2)]], device half* out [[buffer(3)]],
                                  constant LkF16ConvParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint2 lid [[thread_position_in_threadgroup]]) {
  constexpr uint TS = 16;
  threadgroup float As[TS][TS];
  threadgroup float Bs[TS][TS + 1];
  const uint npos = p.h_out * p.w_out, k = 9 * p.ci;
  const uint m = tg.y * TS + lid.y, co = tg.x * TS + lid.x;
  const uint am = tg.y * TS + lid.y;  // A tile row loaded by this thread
  const int aoh = int(am / p.w_out), aow = int(am % p.w_out);
  const uint bco = tg.x * TS + lid.y;  // B tile column loaded by this thread (k along lid.x: coalesced weight rows)
  float acc = 0.0f;
  for (uint k0 = 0; k0 < k; k0 += TS) {
    const uint ka = k0 + lid.x;
    float av = 0.0f;
    if (am < npos && ka < k) {
      const uint tap = ka / p.ci, c = ka % p.ci;
      const int ih = 2 * aoh + int(tap / 3) - 1, iw = 2 * aow + int(tap % 3) - 1;
      if (ih >= 0 && ih < int(p.h_in) && iw >= 0 && iw < int(p.w_in))
        av = float(in[(size_t(ih) * p.w_in + uint(iw)) * p.ci + c]);
    }
    As[lid.y][lid.x] = av;
    Bs[lid.x][lid.y] = (bco < p.co && ka < k) ? float(w[size_t(bco) * k + ka]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < TS; ++kk) acc += As[lid.y][kk] * Bs[kk][lid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (m >= npos || co >= p.co) return;
  out[size_t(m) * p.co + co] = half(lk_gelu_erf(acc + float(bias[co])));
}
