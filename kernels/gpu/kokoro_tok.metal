// Transformer GPU kernels, row-major [T, D] (ALBERT for Kokoro, GPT-2; GEMMs are MPS / steel / gemv4).
//
//   lk_attention_qkv_flash / lk_attention_qkv_sg{2,4,8}   attention over a fused QKV projection [T, 3D] (scalar flash /
//                      simdgroup-matrix flash, optional causal mask)
//   lk_attention_cache_flash / lk_attention_cache_sg{2,4,8}   chunked-prefill attention against a KV cache
//   lk_attention_decode, lk_attention_decode_kvw, lk_attention_decode_split{4,8,16,32}(f) + _combine   decode step
//   lk_kv_write, lk_kv_write_rows   KV-cache writes
//   lk_bias_gelu       gelu_tanh(x + bias[col])
//   lk_add_bias_ln, lk_add_bias_ln2(_sk), lk_layernorm_rows   residual add + bias + LayerNorm variants
//   lk_gemv, lk_gemv_rows, lk_gemv4_c{2,4,8,16}   decode GEMVs
#include <metal_stdlib>

using namespace metal;

struct TokAttnParams {
  uint t, heads, head_dim;
  float scale;
  uint causal;  // GPT-2: query i attends to keys 0..i
};

struct TokRowParams {
  uint rows, cols;
  float eps;
};

// x [R, C], bias [C] -> out [R, C] = gelu_tanh(x + bias) (torch's tanh approximation, same op order).
kernel void lk_bias_gelu(const device float* x [[buffer(0)]], const device float* bias [[buffer(1)]],
                         device float* out [[buffer(2)]], constant TokRowParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= p.rows * p.cols) return;
  const float v = x[gid] + bias[gid % p.cols];
  const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  out[gid] = 0.5f * v * (1.0f + precise::tanh(inner));
}

// a [R, C], bias [C], r [R, C], gamma [C], beta [C] -> out [R, C] = LN(a + bias + r) (biased var, eps).
// One threadgroup (256 threads) per row; mean then centered variance (two passes over the row in registers).
kernel void lk_add_bias_ln(const device float* a [[buffer(0)]], const device float* bias [[buffer(1)]],
                           const device float* r [[buffer(2)]], const device float* gamma [[buffer(3)]],
                           const device float* beta [[buffer(4)]], device float* out [[buffer(5)]],
                           constant TokRowParams& p [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_id [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float partial[8];
  const size_t base = static_cast<size_t>(row) * p.cols;
  constexpr uint kMaxPerThread = 8;  // cols <= 2048
  float v[kMaxPerThread];
  float s = 0.0f;
  uint n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) {
    v[n] = a[base + c] + bias[c] + r[base + c];
    s += v[n];
  }
  s = simd_sum(s);
  if (simd_lane == 0) partial[simd_id] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint k = 0; k < 8; ++k) tot += partial[k];
  const float mean = tot / float(p.cols);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (uint k = 0; k < n; ++k) {
    const float dv = v[k] - mean;
    q += dv * dv;
  }
  q = simd_sum(q);
  if (simd_lane == 0) partial[simd_id] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint k = 0; k < 8; ++k) var += partial[k];
  const float rstd = rsqrt(var / float(p.cols) + p.eps);
  n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) out[base + c] = (v[n] - mean) * rstd * gamma[c] + beta[c];
}

// a [R, C], bias [C], r [R, C], gamma, beta [C] -> sum [R, C] = a + bias + r and out [R, C] = LN(sum)
// (GPT-2's pre-LN blocks need both: the residual stream and the next sub-block's normalized input).
kernel void lk_add_bias_ln2(const device float* a [[buffer(0)]], const device float* bias [[buffer(1)]],
                            const device float* r [[buffer(2)]], const device float* gamma [[buffer(3)]],
                            const device float* beta [[buffer(4)]], device float* sum_out [[buffer(5)]],
                            device float* out [[buffer(6)]], constant TokRowParams& p [[buffer(7)]],
                            uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                            uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_id [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float partial[8];
  const size_t base = static_cast<size_t>(row) * p.cols;
  float v[8];
  float s = 0.0f;
  uint n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) {
    v[n] = a[base + c] + bias[c] + r[base + c];
    sum_out[base + c] = v[n];
    s += v[n];
  }
  s = simd_sum(s);
  if (simd_lane == 0) partial[simd_id] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint k = 0; k < 8; ++k) tot += partial[k];
  const float mean = tot / float(p.cols);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (uint k = 0; k < n; ++k) {
    const float dv = v[k] - mean;
    q += dv * dv;
  }
  q = simd_sum(q);
  if (simd_lane == 0) partial[simd_id] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint k = 0; k < 8; ++k) var += partial[k];
  const float rstd = rsqrt(var / float(p.cols) + p.eps);
  n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) out[base + c] = (v[n] - mean) * rstd * gamma[c] + beta[c];
}

// lk_add_bias_ln2 over split-K partials: a [S, R, C] (lk_gemm_splitk), sum = a[0] + ... + a[S-1] + bias + r.
struct TokRowSkParams {
  uint rows, cols;
  float eps;
  uint splits;
};

kernel void lk_add_bias_ln2_sk(const device float* a [[buffer(0)]], const device float* bias [[buffer(1)]],
                               const device float* r [[buffer(2)]], const device float* gamma [[buffer(3)]],
                               const device float* beta [[buffer(4)]], device float* sum_out [[buffer(5)]],
                               device float* out [[buffer(6)]], constant TokRowSkParams& p [[buffer(7)]],
                               uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_id [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float partial[8];
  const size_t base = static_cast<size_t>(row) * p.cols, plane = static_cast<size_t>(p.rows) * p.cols;
  float v[8];
  float s = 0.0f;
  uint n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) {
    float x = bias[c] + r[base + c];
    for (uint k = 0; k < p.splits; ++k) x += a[k * plane + base + c];
    v[n] = x;
    sum_out[base + c] = x;
    s += x;
  }
  s = simd_sum(s);
  if (simd_lane == 0) partial[simd_id] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint k = 0; k < 8; ++k) tot += partial[k];
  const float mean = tot / float(p.cols);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (uint k = 0; k < n; ++k) {
    const float dv = v[k] - mean;
    q += dv * dv;
  }
  q = simd_sum(q);
  if (simd_lane == 0) partial[simd_id] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint k = 0; k < 8; ++k) var += partial[k];
  const float rstd = rsqrt(var / float(p.cols) + p.eps);
  n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) out[base + c] = (v[n] - mean) * rstd * gamma[c] + beta[c];
}

// x [R, C], gamma, beta [C] -> LN(x) (one threadgroup of 256 per row, two-pass stats).
kernel void lk_layernorm_rows(const device float* x [[buffer(0)]], const device float* gamma [[buffer(1)]],
                              const device float* beta [[buffer(2)]], device float* out [[buffer(3)]],
                              constant TokRowParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_id [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float partial[8];
  const size_t base = static_cast<size_t>(row) * p.cols;
  float v[8];
  float s = 0.0f;
  uint n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) {
    v[n] = x[base + c];
    s += v[n];
  }
  s = simd_sum(s);
  if (simd_lane == 0) partial[simd_id] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint k = 0; k < 8; ++k) tot += partial[k];
  const float mean = tot / float(p.cols);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (uint k = 0; k < n; ++k) {
    const float dv = v[k] - mean;
    q += dv * dv;
  }
  q = simd_sum(q);
  if (simd_lane == 0) partial[simd_id] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint k = 0; k < 8; ++k) var += partial[k];
  const float rstd = rsqrt(var / float(p.cols) + p.eps);
  n = 0;
  for (uint c = lid; c < p.cols; c += 256, ++n) out[base + c] = (v[n] - mean) * rstd * gamma[c] + beta[c];
}

struct DecodeParams {
  uint cap, d, heads;
  float scale;
};

// kcache[pos] = qkv[D:2D], vcache[pos] = qkv[2D:3D] (+ biases, as the GPU QKV GEMM has none). One thread per element.
kernel void lk_kv_write(device float* kc [[buffer(0)]], device float* vc [[buffer(1)]],
                        const device float* qkv [[buffer(2)]], const device long* pos [[buffer(3)]],
                        const device float* bias [[buffer(4)]], constant DecodeParams& p [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
  const long ps = pos[0];
  if (gid >= p.d || ps < 0 || ps >= long(p.cap)) return;
  kc[size_t(ps) * p.d + gid] = qkv[p.d + gid] + bias[p.d + gid];
  vc[size_t(ps) * p.d + gid] = qkv[2 * p.d + gid] + bias[2 * p.d + gid];
}

// One query (q = qkv[0:D] + bias) against cache rows 0..pos; one 128-thread threadgroup per head, head_dim 64,
// cap <= 1024: scores into threadgroup memory, max / sum by SIMD reductions, then out[c] = sum_j p_j v_j[c] with the
// threads split over (c = tid % 64, key half = tid / 64).
kernel void lk_attention_decode(const device float* qkv [[buffer(0)]], const device float* kc [[buffer(1)]],
                                const device float* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                const device float* bias [[buffer(4)]], device float* out [[buffer(5)]],
                                constant DecodeParams& p [[buffer(6)]], uint h [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                                uint simd_id [[simdgroup_index_in_threadgroup]]) {
  constexpr uint HD = 64, NT = 128;
  const long ps = pos[0];
  if (h >= p.heads || ps < 0 || ps >= long(p.cap)) return;
  const uint n = uint(ps) + 1, off = h * HD;
  threadgroup float q[HD];
  threadgroup float sc[1024];
  threadgroup float red[4];
  threadgroup float half_acc[HD];
  if (tid < HD) q[tid] = (qkv[off + tid] + bias[off + tid]) * p.scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint j = tid; j < n; j += NT) {
    const device float* kr = kc + size_t(j) * p.d + off;
    float s = 0.0f;
    for (uint c = 0; c < HD; ++c) s += q[c] * kr[c];
    sc[j] = s;
    m = max(m, s);
  }
  m = simd_max(m);
  if (simd_lane == 0) red[simd_id] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(red[0], red[1]), max(red[2], red[3]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float l = 0.0f;
  for (uint j = tid; j < n; j += NT) {
    const float e = exp(sc[j] - m);
    sc[j] = e;
    l += e;
  }
  l = simd_sum(l);
  if (simd_lane == 0) red[simd_id] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  l = red[0] + red[1] + red[2] + red[3];
  const uint c = tid % HD, kh = tid / HD;
  float acc = 0.0f;
  for (uint j = kh; j < n; j += 2) acc += sc[j] * vc[size_t(j) * p.d + off + c];
  if (kh == 1) half_acc[c] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (kh == 0) out[off + c] = (acc + half_acc[c]) / l;
}

// Flash-style attention over a fused QKV projection (replaced a one-query-per-threadgroup scheme that re-read all
// K/V per query: memory-bound). One threadgroup (128 threads) per (32-query block, head): each
// 32-key K/V tile is loaded into threadgroup memory once (biases added) and reused by the 32 queries; 4 threads per
// query split the 64 dims (partial dots reduced with simd shuffles), online softmax per query; causal skips tiles
// past the block and masks the diagonal tile. qkv [T, 3D] (q | k | v, before bias), bias [3D], out [T, D].
kernel void lk_attention_qkv_flash(const device float* qkv [[buffer(0)]], const device float* bias [[buffer(1)]],
                                   device float* out [[buffer(2)]], constant TokAttnParams& p [[buffer(3)]],
                                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint HD = 64, BQ = 32, BKEY = 32, PART = 16;  // 4 threads x 16 dims per query
  const uint qb = tg.x, h = tg.y;
  if (h >= p.heads) return;
  const uint d = p.heads * HD, ld = 3 * d, qo = h * HD, ko = d + h * HD, vo = 2 * d + h * HD;
  const uint qi = tid / 4, part = tid % 4, i = qb * BQ + qi;
  const bool valid = i < p.t;
  threadgroup float Ks[BKEY][HD + 1];
  threadgroup float Vs[BKEY][HD + 1];
  float q[PART], acc[PART];
  for (uint c = 0; c < PART; ++c) {
    const uint col = part * PART + c;
    q[c] = valid ? (qkv[size_t(i) * ld + qo + col] + bias[qo + col]) * p.scale : 0.0f;
    acc[c] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const uint qend = min(p.t, (qb + 1) * BQ);  // keys needed by this block (causal) are < qend
  const uint kmax = p.causal ? qend : p.t;
  for (uint k0 = 0; k0 < kmax; k0 += BKEY) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BKEY * HD; e += 128) {
      const uint r = e / HD, cc = e % HD, j = k0 + r;
      Ks[r][cc] = j < p.t ? qkv[size_t(j) * ld + ko + cc] + bias[ko + cc] : 0.0f;
      Vs[r][cc] = j < p.t ? qkv[size_t(j) * ld + vo + cc] + bias[vo + cc] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[BKEY];
    float tmax = -INFINITY;
    for (uint r = 0; r < BKEY; ++r) {
      float part_dot = 0.0f;
      for (uint c = 0; c < PART; ++c) part_dot += q[c] * Ks[r][part * PART + c];
      part_dot += simd_shuffle_xor(part_dot, 1);
      part_dot += simd_shuffle_xor(part_dot, 2);
      const uint j = k0 + r;
      const bool ok = j < p.t && (!p.causal || j <= i);
      s[r] = ok ? part_dot : -INFINITY;
      tmax = max(tmax, s[r]);
    }
    const float mn = max(m, tmax);
    if (mn == -INFINITY) continue;  // nothing visible yet for this query (only possible for invalid rows)
    const float corr = exp(m - mn);
    l *= corr;
    for (uint c = 0; c < PART; ++c) acc[c] *= corr;
    for (uint r = 0; r < BKEY; ++r) {
      const float e = exp(s[r] - mn);
      l += e;
      for (uint c = 0; c < PART; ++c) acc[c] += e * Vs[r][part * PART + c];
    }
    m = mn;
  }
  if (valid)
    for (uint c = 0; c < PART; ++c) out[size_t(i) * d + qo + part * PART + c] = acc[c] / l;
}

struct GemvParams {
  uint k, n;
  uint has_bias, act, has_res;  // act: 0 none, 1 gelu_tanh
};

// Decode GEMV, weights [K, N] row-major: out[n] = act(sum_k x[k] W[k, n] + bias[n]) [+ res[n]].
// One threadgroup (256 = 8 simdgroups) per 32 output columns: lane = column (coalesced 128-byte reads of each W row),
// the 8 simdgroups split K, partial sums reduced through threadgroup memory. Bandwidth-bound by design (B = 1).
kernel void lk_gemv(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                    const device float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                    const device float* res [[buffer(4)]], constant GemvParams& p [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][32];
  const uint col = tg * 32 + simd_lane;
  float acc = 0.0f;
  if (col < p.n) {
    const uint chunk = (p.k + 7) / 8, k0 = simd_id * chunk, k1 = min(p.k, k0 + chunk);
    for (uint k = k0; k < k1; ++k) acc += x[k] * w[size_t(k) * p.n + col];
  }
  part[simd_id][simd_lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0 && col < p.n) {
    float s = 0.0f;
    for (uint i = 0; i < 8; ++i) s += part[i][simd_lane];
    if (p.has_bias) s += bias[col];
    if (p.act == 1) s = 0.5f * s * (1.0f + precise::tanh(0.7978845608028654f * (s + 0.044715f * s * s * s)));
    if (p.has_res) s += res[col];
    out[col] = s;
  }
}

// Decode GEMV against row-major rows: out[r] = W[r, :] . x (tied LM head, W = wte [V, K]).
// One simdgroup per row (8 rows per threadgroup), float4 loads, simd_sum.
kernel void lk_gemv_rows(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                         device float* out [[buffer(2)]], constant GemvParams& p [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint row = tg * 8 + simd_id;
  if (row >= p.n) return;
  const device float4* wr = reinterpret_cast<const device float4*>(w + size_t(row) * p.k);
  const device float4* x4 = reinterpret_cast<const device float4*>(x);
  float s = 0.0f;
  for (uint i = simd_lane; i < p.k / 4; i += 32) s += dot(wr[i], x4[i]);
  for (uint i = (p.k / 4) * 4 + simd_lane; i < p.k; i += 32) s += w[size_t(row) * p.k + i] * x[i];
  s = simd_sum(s);
  if (simd_lane == 0) out[row] = s;
}

struct CacheParams {
  uint t, cap, d, heads;
  float scale;
};

// kc[pos0 + i] = qkv[i, D:2D] + bias, vc[pos0 + i] = qkv[i, 2D:3D] + bias, i < T. One thread per (row, element).
kernel void lk_kv_write_rows(device float* kc [[buffer(0)]], device float* vc [[buffer(1)]],
                             const device float* qkv [[buffer(2)]], const device long* pos [[buffer(3)]],
                             const device float* bias [[buffer(4)]], constant CacheParams& p [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const uint e = gid.x, i = gid.y;
  const long r = pos[0] + long(i);
  if (e >= p.d || i >= p.t || r < 0 || r >= long(p.cap)) return;
  kc[size_t(r) * p.d + e] = qkv[size_t(i) * 3 * p.d + p.d + e] + bias[p.d + e];
  vc[size_t(r) * p.d + e] = qkv[size_t(i) * 3 * p.d + 2 * p.d + e] + bias[2 * p.d + e];
}

// Chunked prefill / decode attention: query i (position pos0 + i, q = qkv[i, 0:D] + bias) attends to cache rows
// 0 .. pos0 + i. Same scheme as lk_attention_qkv_flash: one 128-thread threadgroup per (32-query block, head),
// 32-row K/V cache tiles in threadgroup memory, 4 threads x 16 dims per query, online softmax. head_dim 64.
kernel void lk_attention_cache_flash(const device float* qkv [[buffer(0)]], const device float* kc [[buffer(1)]],
                                     const device float* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                     const device float* bias [[buffer(4)]], device float* out [[buffer(5)]],
                                     constant CacheParams& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint HD = 64, BQ = 32, BKEY = 32, PART = 16;
  const uint qb = tg.x, h = tg.y;
  const long pos0 = pos[0];
  if (h >= p.heads || pos0 < 0 || pos0 + long(p.t) > long(p.cap)) return;
  const uint d = p.d, qo = h * HD;
  const uint qi = tid / 4, part = tid % 4, i = qb * BQ + qi;
  const bool valid = i < p.t;
  const long qpos = pos0 + long(i);
  threadgroup float Ks[BKEY][HD + 1];
  threadgroup float Vs[BKEY][HD + 1];
  float q[PART], acc[PART];
  for (uint c = 0; c < PART; ++c) {
    const uint col = part * PART + c;
    q[c] = valid ? (qkv[size_t(i) * 3 * d + qo + col] + bias[qo + col]) * p.scale : 0.0f;
    acc[c] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const long kmax = pos0 + long(min(p.t, (qb + 1) * BQ));  // rows needed by this query block
  for (long k0 = 0; k0 < kmax; k0 += BKEY) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BKEY * HD; e += 128) {
      const uint r = e / HD, cc = e % HD;
      const long j = k0 + r;
      Ks[r][cc] = j < kmax ? kc[size_t(j) * d + qo + cc] : 0.0f;
      Vs[r][cc] = j < kmax ? vc[size_t(j) * d + qo + cc] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[BKEY];
    float tmax = -INFINITY;
    for (uint r = 0; r < BKEY; ++r) {
      float pd = 0.0f;
      for (uint c = 0; c < PART; ++c) pd += q[c] * Ks[r][part * PART + c];
      pd += simd_shuffle_xor(pd, 1);
      pd += simd_shuffle_xor(pd, 2);
      const long j = k0 + r;
      s[r] = (valid && j <= qpos) ? pd : -INFINITY;
      tmax = max(tmax, s[r]);
    }
    const float mn = max(m, tmax);
    if (mn == -INFINITY) continue;
    const float corr = exp(m - mn);
    l *= corr;
    for (uint c = 0; c < PART; ++c) acc[c] *= corr;
    for (uint r = 0; r < BKEY; ++r) {
      const float e = exp(s[r] - mn);
      l += e;
      for (uint c = 0; c < PART; ++c) acc[c] += e * Vs[r][part * PART + c];
    }
    m = mn;
  }
  if (valid)
    for (uint c = 0; c < PART; ++c) out[size_t(i) * d + qo + part * PART + c] = acc[c] / l;
}

// Shared body: queries = rows of qkv (q at column qo, + bias, at positions pos0 + i, i < T); keys/values = rows
// 0 .. nk-1 of kb / vb (row stride kld, already offset to the head; nk = pos0 + T). qkv mode: kb/vb point into qkv
// itself (pos0 = 0, k bias dropped, v bias added to the output); cache mode: kb/vb are the KV caches (biases already
// added when the rows were written), pos0 > 0 for later prefill chunks.
template <uint NSG>
METAL_FUNC void attn_sg_body(const device float* qkv, const device float* bias, device float* out,
                             const device float* kb, const device float* vb, uint kld, uint t, uint pos0, uint heads,
                             float scale, bool causal, const device float* vbias, threadgroup float* Ks,
                             threadgroup float* Vs, uint2 tg, uint tid, uint lane, uint sg) {
  constexpr uint HD = 64, BQ = 8 * NSG, BK = 32, LD = HD + 4, NT = 32 * NSG;
  const uint h = tg.y;
  const uint nqb = (t + BQ - 1) / BQ;
  const uint qb = causal ? nqb - 1 - tg.x : tg.x;
  const uint q0 = qb * BQ, r0 = q0 + sg * 8;
  const uint d = heads * HD, ld = 3 * d, qo = h * HD;
  const uint nk = pos0 + t;  // keys visible to the chunk
  // this lane's fragment coordinates (MLX BaseMMAFrag layout): row fm, columns fn, fn + 1
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  const uint i = r0 + fm, qpos = pos0 + i;
  const float qscale = scale * M_LOG2E_F;  // exp2 domain
  simdgroup_float8x8 Qf[8], Of[8];
  _Pragma("clang loop unroll(full)")
  for (uint c = 0; c < 8; ++c) {
    float2 v = 0.0f;
    if (i < t)
      v = (*reinterpret_cast<const device float2*>(qkv + size_t(i) * ld + qo + c * 8 + fn) +
           *reinterpret_cast<const device float2*>(bias + qo + c * 8 + fn)) * qscale;
    thread auto& qe = Qf[c].thread_elements();
    qe[0] = v.x;
    qe[1] = v.y;
    Of[c] = simdgroup_float8x8(0.0f);
  }
  float m = -INFINITY, l = 0.0f;
  const uint kmax = causal ? min(nk, pos0 + q0 + BQ) : nk;
  const uint kmine = causal ? min(nk, pos0 + r0 + 8) : nk;  // keys this simdgroup needs
  for (uint k0 = 0; k0 < kmax; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BK * HD / 4; e += NT) {
      const uint r = e / (HD / 4), c = (e % (HD / 4)) * 4, j = k0 + r;
      float4 kv = 0.0f, vv = 0.0f;
      if (j < nk) {
        kv = *reinterpret_cast<const device float4*>(kb + size_t(j) * kld + c);
        vv = *reinterpret_cast<const device float4*>(vb + size_t(j) * kld + c);
      }
      *reinterpret_cast<threadgroup float4*>(Ks + r * LD + c) = kv;
      *reinterpret_cast<threadgroup float4*>(Vs + r * LD + c) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 >= kmine) continue;  // causal: the tile is past this simdgroup's rows (still loaded for the others)
    simdgroup_float8x8 Sf[4];
    _Pragma("clang loop unroll(full)")
    for (uint kb_ = 0; kb_ < 4; ++kb_) Sf[kb_] = simdgroup_float8x8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < 8; ++c) {
      _Pragma("clang loop unroll(full)")
      for (uint kb_ = 0; kb_ < 4; ++kb_) {
        simdgroup_float8x8 Kt;
        simdgroup_load(Kt, Ks + kb_ * 8 * LD + c * 8, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Sf[kb_], Qf[c], Kt, Sf[kb_]);
      }
    }
    const bool edge = k0 + BK > nk || (causal && k0 + BK > pos0 + r0);
    float rmax = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (uint kb_ = 0; kb_ < 4; ++kb_) {
      thread auto& se = Sf[kb_].thread_elements();
      if (edge) {
        const uint j = k0 + kb_ * 8 + fn;
        if (j >= nk || (causal && j > qpos)) se[0] = -INFINITY;
        if (j + 1 >= nk || (causal && j + 1 > qpos)) se[1] = -INFINITY;
      }
      rmax = max(rmax, max(se[0], se[1]));
    }
    rmax = max(rmax, simd_shuffle_xor(rmax, 1));
    rmax = max(rmax, simd_shuffle_xor(rmax, 8));
    const float mn = max(m, rmax);
    const bool none = mn == -INFINITY;  // only rows past T with nothing visible yet
    const float corr = none ? 1.0f : exp2(m - mn);
    float ls = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint kb_ = 0; kb_ < 4; ++kb_) {
      thread auto& se = Sf[kb_].thread_elements();
      se[0] = none ? 0.0f : exp2(se[0] - mn);
      se[1] = none ? 0.0f : exp2(se[1] - mn);
      ls += se[0] + se[1];
    }
    l = l * corr + ls;
    m = mn;
    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < 8; ++c) {
      thread auto& oe = Of[c].thread_elements();
      oe[0] *= corr;
      oe[1] *= corr;
    }
    _Pragma("clang loop unroll(full)")
    for (uint kb_ = 0; kb_ < 4; ++kb_) {
      _Pragma("clang loop unroll(full)")
      for (uint c = 0; c < 8; ++c) {
        simdgroup_float8x8 Vf;
        simdgroup_load(Vf, Vs + kb_ * 8 * LD + c * 8, LD);
        simdgroup_multiply_accumulate(Of[c], Sf[kb_], Vf, Of[c]);
      }
    }
  }
  l += simd_shuffle_xor(l, 1);
  l += simd_shuffle_xor(l, 8);
  if (i < t) {
    const float inv = 1.0f / l;
    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < 8; ++c) {
      thread auto& oe = Of[c].thread_elements();
      float2 bv = 0.0f;
      if (vbias) bv = *reinterpret_cast<const device float2*>(vbias + c * 8 + fn);
      *reinterpret_cast<device float2*>(out + size_t(i) * d + qo + c * 8 + fn) = float2(oe[0], oe[1]) * inv + bv;
    }
  }
}

// Flash attention on simdgroup matrices (f32 8x8 MMA; no TF32 on Apple GPUs, so results stay f32 up to summation
// order). One threadgroup of NSG simdgroups per (8*NSG-query block, head); simdgroup s owns query rows 8s..8s+7, its
// Q fragments (bias + scale applied) stay in registers. Per 32-key tile, K and V go through threadgroup memory
// (shared by the simdgroups). Biases: k's is dropped (q.bk is constant over keys, softmax-invariant), v's is added to
// the output (the softmax weights sum to 1). S = Q K^T (4 independent accumulators, 8 MMAs each), online softmax in
// registers (each lane holds 2 adjacent columns of one row per 8x8 fragment; a row lives on lanes {x, x^1, x^8, x^9}),
// O = diag(corr) O + P V (8 independent accumulators). Row sums stay per lane until the end. Causal grids run the
// longest (last) query blocks first. qkv [T, 3D] (q | k | v, before bias), bias [3D], out [T, D]; head_dim 64.
template <uint NSG>
[[kernel]] void lk_attention_qkv_sg_t(const device float* qkv [[buffer(0)]], const device float* bias [[buffer(1)]],
                                      device float* out [[buffer(2)]], constant TokAttnParams& p [[buffer(3)]],
                                      uint2 tg [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint HD = 64, BK = 32, LD = HD + 4;
  if (tg.y >= p.heads) return;
  threadgroup float Ks[BK * LD];
  threadgroup float Vs[BK * LD];
  const uint d = p.heads * HD, qo = tg.y * HD;
  attn_sg_body<NSG>(qkv, bias, out, qkv + d + qo, qkv + 2 * d + qo, 3 * d, p.t, 0, p.heads, p.scale, p.causal != 0,
                    bias + 2 * d + qo, Ks, Vs, tg, tid, lane, sg);
}

struct CacheSgParams {
  uint t, cap, heads;
  float scale;
};

// Chunked-prefill attention on simdgroup matrices: the T queries of qkv (positions pos0 + i, q bias added) attend
// causally to KV-cache rows 0 .. pos0 + i (the chunk's own rows already written by lk_kv_write_rows, biases
// included). Same scheme / grid as lk_attention_qkv_sg; kc / vc [cap, D].
template <uint NSG>
[[kernel]] void lk_attention_cache_sg_t(const device float* qkv [[buffer(0)]], const device float* kc [[buffer(1)]],
                                        const device float* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                        const device float* bias [[buffer(4)]], device float* out [[buffer(5)]],
                                        constant CacheSgParams& p [[buffer(6)]],
                                        uint2 tg [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint HD = 64, BK = 32, LD = HD + 4;
  const long pos0 = pos[0];
  if (tg.y >= p.heads || pos0 < 0 || pos0 + long(p.t) > long(p.cap)) return;
  threadgroup float Ks[BK * LD];
  threadgroup float Vs[BK * LD];
  const uint d = p.heads * HD, qo = tg.y * HD;
  attn_sg_body<NSG>(qkv, bias, out, kc + qo, vc + qo, d, p.t, uint(pos0), p.heads, p.scale, true, nullptr, Ks, Vs,
                    tg, tid, lane, sg);
}

template [[host_name("lk_attention_cache_sg4")]] [[kernel]] decltype(lk_attention_cache_sg_t<4>) lk_attention_cache_sg_t<4>;
template [[host_name("lk_attention_cache_sg2")]] [[kernel]] decltype(lk_attention_cache_sg_t<2>) lk_attention_cache_sg_t<2>;
template [[host_name("lk_attention_cache_sg8")]] [[kernel]] decltype(lk_attention_cache_sg_t<8>) lk_attention_cache_sg_t<8>;

template [[host_name("lk_attention_qkv_sg4")]] [[kernel]] decltype(lk_attention_qkv_sg_t<4>) lk_attention_qkv_sg_t<4>;
template [[host_name("lk_attention_qkv_sg2")]] [[kernel]] decltype(lk_attention_qkv_sg_t<2>) lk_attention_qkv_sg_t<2>;
template [[host_name("lk_attention_qkv_sg8")]] [[kernel]] decltype(lk_attention_qkv_sg_t<8>) lk_attention_qkv_sg_t<8>;

// Decode attention fused with the KV-cache write: one query (q = qkv[0:D] + bias) against cache rows 0..pos-1 plus
// this step's own k/v (read from qkv, + bias, and written to cache row pos by the same threadgroup). One threadgroup of
// 1024 threads (32 simdgroups) per head, head_dim 64: lane = 2 dims (float2, coalesced 256-byte row reads), simdgroup
// sg takes keys sg, sg+32, ...; dots by simd_sum. Pass 1: scores -> threadgroup memory + per-simdgroup max; pass 2:
// p = exp2(s - max) and p*v partial sums per simdgroup, combined through threadgroup memory. cap <= 1024.
// UN keys per simdgroup per iteration (4; 8 measured slower on the M2 Max, 0.288 vs 0.258 ms per step, 2026-11-01).
template <uint UN>
[[kernel]] void lk_attention_decode_kvw_t(const device float* qkv [[buffer(0)]], device float* kc [[buffer(1)]],
                                    device float* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                    const device float* bias [[buffer(4)]], device float* out [[buffer(5)]],
                                    constant DecodeParams& p [[buffer(6)]], uint h [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint HD = 64, NSG = 32;
  const long ps = pos[0];
  if (h >= p.heads || ps < 0 || ps >= long(p.cap)) return;
  const uint n = uint(ps) + 1, d = p.d, off = h * HD + 2 * lane;
  threadgroup float sc[1024];
  threadgroup float red[NSG];
  threadgroup float lsum[NSG];
  threadgroup float2 part[NSG][32];
  const float2 q = (*reinterpret_cast<const device float2*>(qkv + off) +
                    *reinterpret_cast<const device float2*>(bias + off)) * (p.scale * M_LOG2E_F);
  const float2 knew = *reinterpret_cast<const device float2*>(qkv + d + off) +
                      *reinterpret_cast<const device float2*>(bias + d + off);
  const float2 vnew = *reinterpret_cast<const device float2*>(qkv + 2 * d + off) +
                      *reinterpret_cast<const device float2*>(bias + 2 * d + off);
  if (sg == 0) {
    *reinterpret_cast<device float2*>(kc + size_t(ps) * d + off) = knew;
    *reinterpret_cast<device float2*>(vc + size_t(ps) * d + off) = vnew;
  }
  const uint last = n - 1;  // = ps: its k/v come from qkv
  float m = -INFINITY;
  for (uint j0 = sg; j0 < n; j0 += UN * NSG) {
    float s[UN];
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const uint j = j0 + u * NSG;
      float2 k2 = 0.0f;
      if (j < last) k2 = *reinterpret_cast<const device float2*>(kc + size_t(j) * d + off);
      else if (j == last) k2 = knew;
      s[u] = dot(q, k2);
    }
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const uint j = j0 + u * NSG;
      const float t = simd_sum(s[u]);
      if (j < n) {
        if (lane == 0) sc[j] = t;
        m = max(m, t);
      }
    }
  }
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mx = simd_max(red[lane]);
  float l = 0.0f;
  float2 acc = 0.0f;
  for (uint j0 = sg; j0 < n; j0 += UN * NSG) {
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const uint j = j0 + u * NSG;
      if (j < n) {
        const float e = exp2(sc[j] - mx);
        const float2 v2 = j < last ? *reinterpret_cast<const device float2*>(vc + size_t(j) * d + off) : vnew;
        l += e;
        acc += e * v2;
      }
    }
  }
  part[sg][lane] = acc;
  if (lane == 0) lsum[sg] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float2 o = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < NSG; ++k) o += part[k][lane];
    const float L = simd_sum(lsum[lane]);
    *reinterpret_cast<device float2*>(out + off) = o / L;
  }
}

template [[host_name("lk_attention_decode_kvw")]] [[kernel]] decltype(lk_attention_decode_kvw_t<4>) lk_attention_decode_kvw_t<4>;

struct DecodeSplitParams {
  uint cap, d, heads;
  float scale;
  uint splits;
};

// lk_attention_decode_kvw with the keys split across `splits` threadgroups per head (one threadgroup per head used
// only 12 of the M2 Max's 30 cores): grid (heads, splits), NSG simdgroups each; split s takes a contiguous key range
// and writes its partial (max m, sum l, unnormalized p.v [64]) to part[h][s] (68 floats); lk_attention_decode_combine
// merges them. Split 0 writes this step's k/v to cache row pos (no other threadgroup reads that row: it uses knew /
// vnew). Scores in the exp2 domain, as in the fused kernel.
template <uint NSG, bool FUSED>
[[kernel]] void lk_attention_decode_split_t(const device float* qkv [[buffer(0)]], device float* kc [[buffer(1)]],
                                            device float* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                            const device float* bias [[buffer(4)]], device float* part [[buffer(5)]],
                                            constant DecodeSplitParams& p [[buffer(6)]],
                                            device float* out [[buffer(7)]],
                                            uint2 tg [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]],
                                            uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint HD = 64, UN = 4, PS = 68;
  const uint h = tg.x, s = tg.y;
  const long ps = pos[0];
  if (h >= p.heads || s >= p.splits || ps < 0 || ps >= long(p.cap)) return;
  const uint n = uint(ps) + 1, d = p.d, off = h * HD + 2 * lane;
  const uint chunk = (n + p.splits - 1) / p.splits, j_beg = min(n, s * chunk), j_end = min(n, j_beg + chunk);
  threadgroup float sc[1024];
  threadgroup float red[NSG];
  threadgroup float lsum[NSG];
  threadgroup float2 acc_tg[NSG][32];
  const float2 q = (*reinterpret_cast<const device float2*>(qkv + off) +
                    *reinterpret_cast<const device float2*>(bias + off)) * (p.scale * M_LOG2E_F);
  const float2 knew = *reinterpret_cast<const device float2*>(qkv + d + off) +
                      *reinterpret_cast<const device float2*>(bias + d + off);
  const float2 vnew = *reinterpret_cast<const device float2*>(qkv + 2 * d + off) +
                      *reinterpret_cast<const device float2*>(bias + 2 * d + off);
  if (s == 0 && sg == 0) {
    *reinterpret_cast<device float2*>(kc + size_t(ps) * d + off) = knew;
    *reinterpret_cast<device float2*>(vc + size_t(ps) * d + off) = vnew;
  }
  const uint last = n - 1;
  float m = -INFINITY;
  for (uint j0 = j_beg + sg; j0 < j_end; j0 += UN * NSG) {
    float sv[UN];
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const uint j = j0 + u * NSG;
      float2 k2 = 0.0f;
      if (j < last && j < j_end) k2 = *reinterpret_cast<const device float2*>(kc + size_t(j) * d + off);
      else if (j == last) k2 = knew;
      sv[u] = dot(q, k2);
    }
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const uint j = j0 + u * NSG;
      const float t = simd_sum(sv[u]);
      if (j < j_end) {
        if (lane == 0) sc[j - j_beg] = t;
        m = max(m, t);
      }
    }
  }
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mx = simd_max(lane < NSG ? red[lane] : -INFINITY);
  float l = 0.0f;
  float2 acc = 0.0f;
  for (uint j0 = j_beg + sg; j0 < j_end; j0 += UN * NSG) {
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const uint j = j0 + u * NSG;
      if (j < j_end) {
        const float e = exp2(sc[j - j_beg] - mx);
        const float2 v2 = j < last ? *reinterpret_cast<const device float2*>(vc + size_t(j) * d + off) : vnew;
        l += e;
        acc += e * v2;
      }
    }
  }
  acc_tg[sg][lane] = acc;
  if (lane == 0) lsum[sg] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float2 o = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < NSG; ++k) o += acc_tg[k][lane];
    const float L = simd_sum(lane < NSG ? lsum[lane] : 0.0f);
    device float* pp = part + (size_t(h) * p.splits + s) * PS;
    if (lane == 0) {
      pp[0] = mx;
      pp[1] = L;
    }
    *reinterpret_cast<device float2*>(pp + 4 + 2 * lane) = o;
  }
  if (!FUSED) return;
  // Single-dispatch combine: the last split of each head to finish (device-scope counter per head, reset by it)
  // merges the partials into out.
  threadgroup uint is_last;
  if (sg == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (lane == 0) {
      device atomic_uint* cnt = reinterpret_cast<device atomic_uint*>(part + size_t(p.heads) * p.splits * PS) + h;
      is_last = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == p.splits - 1 ? 1u : 0u;
      if (is_last) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!is_last || sg != 0) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  const device float* ph = part + size_t(h) * p.splits * PS;
  float M = -INFINITY;
  for (uint k = 0; k < p.splits; ++k) M = max(M, ph[k * PS]);
  float Lt = 0.0f;
  float2 ot = 0.0f;
  for (uint k = 0; k < p.splits; ++k) {
    const float ms = ph[k * PS];
    if (ms == -INFINITY) continue;
    const float w = exp2(ms - M);
    Lt += w * ph[k * PS + 1];
    ot += w * *reinterpret_cast<const device float2*>(ph + k * PS + 4 + 2 * lane);
  }
  *reinterpret_cast<device float2*>(out + off) = ot / Lt;
}


template [[host_name("lk_attention_decode_split4")]] [[kernel]] decltype(lk_attention_decode_split_t<4, false>) lk_attention_decode_split_t<4, false>;
template [[host_name("lk_attention_decode_splitf4")]] [[kernel]] decltype(lk_attention_decode_split_t<4, true>) lk_attention_decode_split_t<4, true>;
template [[host_name("lk_attention_decode_split8")]] [[kernel]] decltype(lk_attention_decode_split_t<8, false>) lk_attention_decode_split_t<8, false>;
template [[host_name("lk_attention_decode_splitf8")]] [[kernel]] decltype(lk_attention_decode_split_t<8, true>) lk_attention_decode_split_t<8, true>;
template [[host_name("lk_attention_decode_split16")]] [[kernel]] decltype(lk_attention_decode_split_t<16, false>) lk_attention_decode_split_t<16, false>;
template [[host_name("lk_attention_decode_splitf16")]] [[kernel]] decltype(lk_attention_decode_split_t<16, true>) lk_attention_decode_split_t<16, true>;
template [[host_name("lk_attention_decode_split32")]] [[kernel]] decltype(lk_attention_decode_split_t<32, false>) lk_attention_decode_split_t<32, false>;
template [[host_name("lk_attention_decode_splitf32")]] [[kernel]] decltype(lk_attention_decode_split_t<32, true>) lk_attention_decode_split_t<32, true>;

// Merges lk_attention_decode_split's partials: one simdgroup per head (lane = 2 dims),
// out = sum_s 2^(m_s - M) acc_s / sum_s 2^(m_s - M) l_s (empty splits have m = -inf, l = 0).
kernel void lk_attention_decode_combine(const device float* part [[buffer(0)]], device float* out [[buffer(1)]],
                                        constant DecodeSplitParams& p [[buffer(2)]],
                                        uint h [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint HD = 64, PS = 68;
  if (h >= p.heads) return;
  const device float* ph = part + size_t(h) * p.splits * PS;
  float M = -INFINITY;
  for (uint s = 0; s < p.splits; ++s) M = max(M, ph[s * PS]);
  float L = 0.0f;
  float2 o = 0.0f;
  for (uint s = 0; s < p.splits; ++s) {
    const float ms = ph[s * PS];
    if (ms == -INFINITY) continue;
    const float w = exp2(ms - M);
    L += w * ph[s * PS + 1];
    o += w * *reinterpret_cast<const device float2*>(ph + s * PS + 4 + 2 * lane);
  }
  *reinterpret_cast<device float2*>(out + h * HD + 2 * lane) = o / L;
}

// Decode GEMV v2, weights [K, N] row-major (N % 4 == 0): out[n] = act(sum_k x[k] W[k, n] + bias[n]) [+ res[n]].
// One 256-thread threadgroup per 4*CW output columns: thread = (float4 column c4 = tid % CW, K slice = tid / CW);
// slice s reads rows s, s + KS, ... (CW consecutive float4 per row, unrolled 4 deep for memory-level parallelism);
// partials reduced by simd shuffles then threadgroup memory; the epilogue (bias, gelu_tanh, residual) is fused.
template <uint CW>
[[kernel]] void lk_gemv4_t(const device float* x [[buffer(0)]], const device float4* w [[buffer(1)]],
                           const device float4* bias [[buffer(2)]], device float4* out [[buffer(3)]],
                           const device float4* res [[buffer(4)]], constant GemvParams& p [[buffer(5)]],
                           uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint NT = 256, KS = NT / CW, SPS = 32 / CW;  // K slices, slices per simdgroup
  const uint n4 = p.n / 4, c4 = tg * CW + tid % CW, s = tid / CW;
  float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  if (c4 < n4) {
    uint k = s;
    for (; k + 3 * KS < p.k; k += 4 * KS) {
      const float4 w0 = w[size_t(k) * n4 + c4], w1 = w[size_t(k + KS) * n4 + c4];
      const float4 w2 = w[size_t(k + 2 * KS) * n4 + c4], w3 = w[size_t(k + 3 * KS) * n4 + c4];
      acc0 += x[k] * w0;
      acc1 += x[k + KS] * w1;
      acc2 += x[k + 2 * KS] * w2;
      acc3 += x[k + 3 * KS] * w3;
    }
    for (; k < p.k; k += KS) acc0 += x[k] * w[size_t(k) * n4 + c4];
  }
  float4 acc = (acc0 + acc1) + (acc2 + acc3);
  _Pragma("clang loop unroll(full)")
  for (uint o = CW; o < 32; o *= 2) acc += simd_shuffle_xor(acc, o);  // the SPS slices of this simdgroup
  threadgroup float4 part[NT / 32][CW];
  if (lane < CW) part[sg][lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < CW && c4 < n4) {
    float4 r = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < NT / 32; ++i) r += part[i][tid];
    if (p.has_bias) r += bias[c4];
    if (p.act == 1) r = 0.5f * r * (1.0f + precise::tanh(0.7978845608028654f * (r + 0.044715f * r * r * r)));
    if (p.has_res) r += res[c4];
    out[c4] = r;
  }
  (void)SPS;
}

template [[host_name("lk_gemv4_c8")]] [[kernel]] decltype(lk_gemv4_t<8>) lk_gemv4_t<8>;
template [[host_name("lk_gemv4_c16")]] [[kernel]] decltype(lk_gemv4_t<16>) lk_gemv4_t<16>;
template [[host_name("lk_gemv4_c4")]] [[kernel]] decltype(lk_gemv4_t<4>) lk_gemv4_t<4>;
template [[host_name("lk_gemv4_c2")]] [[kernel]] decltype(lk_gemv4_t<2>) lk_gemv4_t<2>;
