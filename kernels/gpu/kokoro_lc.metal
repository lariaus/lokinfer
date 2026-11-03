// Kokoro frame-rate GPU kernels, channel-last [L, C] (M4; m2-conv-proof.md: the GPU path is channel-last).
//
//   lk_conv1d_lc        steel implicit-GEMM conv (MLX v0.32.4 steel_conv.h body, MIT, third_party/mlx_steel)
//                       with a fused epilogue: + bias[Co] [+ residual], * scale, [ReLU / ReLU6 / leaky 0.2]. Despite the name it is the
//                       general NHWC conv2d (conv1d = H 1); also used by ResNet-18 (kernels/gpu/vision.metal)
//   lk_act_lc           act((x - mean) * scale + beta) per channel: snake(alpha) or leaky(slope); affine optional
//   lk_stats_lc_tg      per-(128-row chunk, channel) (mean, M2) -> stats [S, C, 2]; 8 threads per (chunk, channel)
//   lk_adain_affine_lc  AdaIN affine [3, C] = (mean, (1 + g) / std, b) from the chunk stats
//   lk_convt_gather_lc  ConvTranspose1d overlap-add from y = x @ W ([Lin, K*Co], one GEMM), + bias [+ add],
//                       optional reflect-pad of one row on the left (Kokoro's ups[1] path)
//   lk_add3_scale       (a + b + c) * s
//   lk_lc_to_cf         first `cout` channels of [L, Cpad] -> channel-first [cout, L]
//   lk_concat_pad_lc    [a | b | 0...] per row (channel concat, zero-padded to a multiple of 16 for the steel loader)
//   lk_dwconvt_lc       depthwise ConvTranspose1d (AdainResBlk1d's upsample "pool")
//   lk_upsample2_lc     nearest x2 along L
//   lk_attention_qkv_flash, lk_istft_lc, lk_adain_affine_lc (one simdgroup per channel) — see their comments
// (the Kokoro generator's convs run as Winograd since 2026-11-01: kernels/gpu/winograd.metal)
//
// Buffer order follows lokinfer's kernel ABI (operands, outs, then params) except lk_conv1d_lc, whose fixed
// slots are documented on the kernel (dispatch: kernels/gpu/kokoro_lc.cpp).
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/conv/conv.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"

using namespace metal;

// ---- conv -------------------------------------------------------------------------------------------------

struct LcConvEpilogue {
  float scale;
  uint has_bias;
  uint has_res;
  uint act;  // 0 none, 1 ReLU, 2 ReLU6, 3 LeakyReLU(0.2)
};

struct LcLeakyOp {
  METAL_FUNC float apply(float a) const { return a >= 0.0f ? a : a * 0.2f; }
};

struct LcReluOp {
  METAL_FUNC float apply(float a) const { return max(a, 0.0f); }
};

struct LcRelu6Op {
  METAL_FUNC float apply(float a) const { return clamp(a, 0.0f, 6.0f); }
};

struct LcAddOp {
  METAL_FUNC float apply(float a, float b) const { return a + b; }
};

struct LcScaleOp {
  float s;
  METAL_FUNC float apply(float a) const { return a * s; }
};

// Slots: 0 x [L, Ci], 1 w [Co, K, Ci], 2 bias [Co], 3 out [Lo, Co], 4 residual [Lo, Co] (bound to out when
// unused), 5 MLXConvParams<2>, 6 ImplicitGemmConv2DParams, 7 LcConvEpilogue.
template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_conv1d_lc(
    const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device T* bias [[buffer(2)]],
    device T* C [[buffer(3)]], const device T* R [[buffer(4)]],
    const constant MLXConvParams<2>* params [[buffer(5)]],
    const constant ImplicitGemmConv2DParams* gemm_params [[buffer(6)]],
    const constant LcConvEpilogue* epi [[buffer(7)]], uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {
  using namespace mlx::steel;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);
  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;
  constexpr short tgp_size = WM * WN * 32;

  using loader_a_t = Conv2DInputBlockLoaderSmallFilter<T, BM, BN, BK, tgp_size, tgp_padding_a>;
  using loader_b_t = Conv2DWeightBlockLoader<T, BM, BN, BK, tgp_size, tgp_padding_b>;
  using mma_t = BlockMMA<T, T, BM, BN, BK, WM, WN, transpose_a, transpose_b, shape_a_cols, shape_b_cols>;

  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];

  const int tid_y = ((tid.y) << gemm_params->swizzle_log) + ((tid.x) & ((1 << gemm_params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;
  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) return;

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;
  const int N = gemm_params->N;

  B += c_col * K;
  C += static_cast<size_t>(c_row) * N + c_col;

  const int2 offsets_a(0, c_row);
  const int2 offsets_b(0, c_col);
  loader_a_t loader_a(A, As, offsets_a, params, gemm_params, simd_gid, simd_lid);
  loader_b_t loader_b(B, Bs, offsets_b, params, gemm_params, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < gemm_params->gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }
  threadgroup_barrier(mem_flags::mem_none);

  const short2 dims(short(min(BN, N - c_col)), short(min(BM, gemm_params->M - c_row)));
  if (epi->has_bias) mma_op.apply_epilogue_safe(bias + c_col, 0, 1, dims, LcAddOp());
  if (epi->has_res) mma_op.apply_epilogue_safe(R + static_cast<size_t>(c_row) * N + c_col, N, 1, dims, LcAddOp());
  if (epi->scale != 1.0f) mma_op.apply_epilogue(LcScaleOp{epi->scale});
  if (epi->act == 1) mma_op.apply_epilogue(LcReluOp());
  else if (epi->act == 2) mma_op.apply_epilogue(LcRelu6Op());
  else if (epi->act == 3) mma_op.apply_epilogue(LcLeakyOp());
  mma_op.store_result_safe(C, N, dims);
}

template [[host_name("lk_conv1d_lc_f32")]] [[kernel]] void lk_conv1d_lc<float, 64, 64, 16, 2, 2>(
    const device float*, const device float*, const device float*, device float*, const device float*,
    const constant MLXConvParams<2>*, const constant ImplicitGemmConv2DParams*, const constant LcConvEpilogue*,
    uint3, uint, uint);

// 32x32 tiles: for small outputs (e.g. ResNet's 7x7x512 at batch 1) where 64x64 tiles leave most GPU cores idle.
template [[host_name("lk_conv1d_lc_f32_bm32")]] [[kernel]] void lk_conv1d_lc<float, 32, 32, 16, 2, 2>(
    const device float*, const device float*, const device float*, device float*, const device float*,
    const constant MLXConvParams<2>*, const constant ImplicitGemmConv2DParams*, const constant LcConvEpilogue*,
    uint3, uint, uint);

// ---- elementwise / reductions ------------------------------------------------------------------------------

struct LcActParams {
  uint l, c;
  float slope;  // leaky
  uint act;     // 1 snake, 2 leaky
  uint has_affine;
};

// x [L, C], affine [3, C] (mean, scale, beta), alpha [C] (snake; bound to x when unused), out [L, C].
kernel void lk_act_lc(const device float* x [[buffer(0)]], const device float* af [[buffer(1)]],
                      const device float* alpha [[buffer(2)]], device float* out [[buffer(3)]],
                      constant LcActParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.l * p.c) return;
  const uint c = gid % p.c;
  float v = x[gid];
  if (p.has_affine) v = (v - af[c]) * af[p.c + c] + af[2 * p.c + c];
  if (p.act == 1) {
    const float a = alpha[c];
    const float s = precise::sin(a * v);
    v = v + (1.0f / a) * (s * s);
  } else if (p.act == 2) {
    v = v >= 0.0f ? v : v * p.slope;
  }
  out[gid] = v;
}

struct LcStatsParams {
  uint l, c, chunks;
};

// x [L, C] -> stats [S, C, 2], S = ceil(L / 128): per-(128-row chunk, channel) (mean, M2), two passes. A threadgroup (32 x 8) owns 32 channels of one 128-row
// chunk; thread (x, y) walks rows y*16 .. y*16+15 of channel x. Pass 1: partial sums -> threadgroup reduce -> mean;
// pass 2: partial M2 around that mean -> reduce. One thread per (chunk, channel) walking 128 dependent rows was
// latency-bound (~125 GB/s on the Mac, 2026-11-02).
kernel void lk_stats_lc_tg(const device float* x [[buffer(0)]], device float* st [[buffer(1)]],
                           constant LcStatsParams& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup float part[8][32];
  const uint c = tg.x * 32 + tid.x, s = tg.y;
  const uint t0 = s * 128, n = min(128u, p.l - t0);
  const uint r0 = min(tid.y * 16, n), r1 = min(r0 + 16, n);
  const bool ok = c < p.c;
  const device float* col = x + static_cast<size_t>(t0) * p.c + (ok ? c : 0);
  float sum = 0.0f;
  if (ok)
    for (uint i = r0; i < r1; ++i) sum += col[static_cast<size_t>(i) * p.c];
  part[tid.y][tid.x] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint j = 0; j < 8; ++j) tot += part[j][tid.x];
  const float mean = tot / float(n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m2 = 0.0f;
  if (ok)
    for (uint i = r0; i < r1; ++i) {
      const float d = col[static_cast<size_t>(i) * p.c] - mean;
      m2 += d * d;
    }
  part[tid.y][tid.x] = m2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid.y == 0 && ok) {
    float m2t = 0.0f;
    for (uint j = 0; j < 8; ++j) m2t += part[j][tid.x];
    st[(static_cast<size_t>(s) * p.c + c) * 2] = mean;
    st[(static_cast<size_t>(s) * p.c + c) * 2 + 1] = m2t;
  }
}

struct LcAdainParams {
  uint l, c, chunks;
  float eps;
  uint creal;  // channels >= creal are zero padding: affine (0, 0, 0) keeps them exactly 0
};

// stats [S, C, 2], gb [2 * creal] -> affine [3, C]. One simdgroup per channel (4 per threadgroup): lanes take
// chunks i, i + 32, ... with Kahan-compensated float sums, then a tree reduction (simd_sum) — the exact combine of
// the chunk partials, mean = Σ n_i m_i / N, M2 = Σ M2_i + Σ n_i (m_i - mean)² (no double on GPU). One thread per
// channel walking all chunks serially was latency-bound (Kokoro rand510 on the Mac: 58 calls, 12 ms).
kernel void lk_adain_affine_lc(const device float* st [[buffer(0)]], const device float* gb [[buffer(1)]],
                               device float* af [[buffer(2)]], constant LcAdainParams& p [[buffer(3)]],
                               uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg * 4 + sg;
  if (c >= p.c) return;
  if (c >= p.creal) {
    if (lane == 0) {
      af[c] = 0.0f;
      af[p.c + c] = 0.0f;
      af[2 * p.c + c] = 0.0f;
    }
    return;
  }
  float s = 0.0f, comp = 0.0f;
  for (uint i = lane; i < p.chunks; i += 32) {
    const float n = float(min(128u, p.l - i * 128));
    const float y = st[(static_cast<size_t>(i) * p.c + c) * 2] * n - comp;
    const float t = s + y;
    comp = (t - s) - y;
    s = t;
  }
  const float mean = simd_sum(s) / float(p.l);
  float m2 = 0.0f;
  comp = 0.0f;
  for (uint i = lane; i < p.chunks; i += 32) {
    const float n = float(min(128u, p.l - i * 128));
    const float dm = st[(static_cast<size_t>(i) * p.c + c) * 2] - mean;
    const float y = st[(static_cast<size_t>(i) * p.c + c) * 2 + 1] + n * dm * dm - comp;
    const float t = m2 + y;
    comp = (t - m2) - y;
    m2 = t;
  }
  m2 = simd_sum(m2);
  if (lane == 0) {
    const float var = m2 / float(p.l);
    af[c] = mean;
    af[p.c + c] = (1.0f + gb[c]) * rsqrt(var + p.eps);
    af[2 * p.c + c] = gb[p.creal + c];
  }
}

struct LcConvTParams {
  uint lin, lout, co, k, stride, pad, reflect_left, has_add;
};

// y [Lin, K*Co] (y[j, kk*Co + co] = Σ_ci x[j, ci] W[ci, co, kk]), bias [Co], add [Lout, Co] (bound to y when
// unused) -> out [Lout, Co]: out[t] = bias + Σ_{kk ≡ t+pad (mod s)} y[(t + pad - kk) / s, kk] (+ add[t]).
// reflect_left: out row 0 is the convT row 1 and row t is convT row t - 1 (F.pad(..., (1, 0), "reflect")).
kernel void lk_convt_gather_lc(const device float* y [[buffer(0)]], const device float* bias [[buffer(1)]],
                               const device float* add [[buffer(2)]], device float* out [[buffer(3)]],
                               constant LcConvTParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.lout * p.co) return;
  const uint tp = gid / p.co, co = gid % p.co;
  const int t = p.reflect_left ? (tp == 0 ? 1 : int(tp) - 1) : int(tp);
  float acc = bias[co];
  const int tpad = t + int(p.pad);
  const size_t ldy = static_cast<size_t>(p.k) * p.co;
  for (int kk = tpad % int(p.stride); kk < int(p.k); kk += int(p.stride)) {
    const int num = tpad - kk;
    if (num < 0) break;
    const int j = num / int(p.stride);
    if (j < int(p.lin)) acc += y[static_cast<size_t>(j) * ldy + static_cast<size_t>(kk) * p.co + co];
  }
  if (p.has_add) acc += add[gid];
  out[gid] = acc;
}

struct LcScaleParams {
  uint n;
  float s;
};

kernel void lk_add3_scale(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                          const device float* c [[buffer(2)]], device float* out [[buffer(3)]],
                          constant LcScaleParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  out[gid] = (a[gid] + b[gid] + c[gid]) * p.s;
}

struct LcToCfParams {
  uint l, cpad, cout;
};

// x [L, Cpad] -> out [cout, L] (first cout channels). One thread per output element; consecutive threads write
// consecutive t (coalesced writes; the strided reads are tiny next to the conv).
kernel void lk_lc_to_cf(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                        constant LcToCfParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint t = gid.x, c = gid.y;
  if (t >= p.l || c >= p.cout) return;
  out[static_cast<size_t>(c) * p.l + t] = x[static_cast<size_t>(t) * p.cpad + c];
}

struct LcConcatParams {
  uint l, ca, cb, cout;
};

// a [L, Ca], b [L, Cb] (bound to a when Cb = 0) -> out [L, Cout] = [a | b | 0 ...].
kernel void lk_concat_pad_lc(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                             device float* out [[buffer(2)]], constant LcConcatParams& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid >= p.l * p.cout) return;
  const uint t = gid / p.cout, c = gid % p.cout;
  float v = 0.0f;
  if (c < p.ca) v = a[static_cast<size_t>(t) * p.ca + c];
  else if (c < p.ca + p.cb) v = b[static_cast<size_t>(t) * p.cb + (c - p.ca)];
  out[gid] = v;
}

struct LcDwConvTParams {
  uint lin, lout, c, k, stride, pad;
};

// x [Lin, C], w [C, K], bias [C] -> out [Lout, C]: out[t, c] = bias[c] + Σ_{kk} x[(t + pad - kk) / s, c] w[c, kk].
kernel void lk_dwconvt_lc(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                          const device float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                          constant LcDwConvTParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.lout * p.c) return;
  const uint t = gid / p.c, c = gid % p.c;
  float acc = bias[c];
  const int tpad = int(t) + int(p.pad);
  for (int kk = tpad % int(p.stride); kk < int(p.k); kk += int(p.stride)) {
    const int num = tpad - kk;
    if (num < 0) break;
    const int j = num / int(p.stride);
    if (j < int(p.lin)) acc += x[static_cast<size_t>(j) * p.c + c] * w[c * p.k + kk];
  }
  out[gid] = acc;
}

struct LcUpParams {
  uint l, c;
};

// x [L, C] -> out [2L, C], out[t] = x[t / 2]
kernel void lk_upsample2_lc(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                            constant LcUpParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= 2 * p.l * p.c) return;
  const uint t = gid / p.c, c = gid % p.c;
  out[gid] = x[static_cast<size_t>(t / 2) * p.c + c];
}

struct IstftLcParams {
  uint l, cpad, nfft, hop, pad, len;
};

// iSTFT of the generator's logits straight from its channel-last output (no lc_to_cf, no CPU pass at the end of
// the graph): x [L, Cpad] with column k < K = nfft/2+1 the log-magnitude and K + k the phase logit of bin k ->
// out [hop * (L - 1)] = torch.istft(exp(mag) * e^{i sin(phase)}, periodic hann, center=True), the CPU kernel's math
// (istft_logits_f32): per frame the window-folded inverse real DFT, then overlap-add and the window-square envelope.
// One thread per output sample: it sums its <= ceil(nfft/hop) frames in frame order (the CPU's order), each frame
// summed over bins in bin order.
kernel void lk_istft_lc(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                        constant IstftLcParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.len) return;
  const uint bins = p.nfft / 2 + 1, pos = i + p.pad;
  const uint tlo = pos + 1 >= p.nfft ? (pos + 1 - p.nfft + p.hop - 1) / p.hop : 0u;
  const uint thi = min(p.l - 1, pos / p.hop);
  const float twopi_n = 2.0f * M_PI_F / float(p.nfft);
  float y = 0.0f, env = 0.0f;
  for (uint t = tlo; t <= thi; ++t) {
    const uint m = pos - t * p.hop;
    const float win = 0.5f - 0.5f * precise::cos(twopi_n * float(m));
    const device float* xt = x + size_t(t) * p.cpad;
    float v = 0.0f;
    for (uint k = 0; k < bins; ++k) {
      const bool edge = k == 0 || 2 * k == p.nfft;
      const float mag = precise::exp(xt[k]);
      const float ph = precise::sin(xt[bins + k]);
      const float re = mag * precise::cos(ph), im = mag * precise::sin(ph);
      const float c = (edge ? 1.0f : 2.0f) * win / float(p.nfft);
      const float ang = twopi_n * float((k * m) % p.nfft);
      v += (c * precise::cos(ang)) * re + (edge ? 0.0f : -c * precise::sin(ang)) * im;
    }
    y += v;
    env += win * win;
  }
  out[i] = y / env;
}
