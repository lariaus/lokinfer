// f16 optimized GPU kernels (Qwen3-ASR phase 1, docs/impl-notes/qwen3-asr-f16-opt-kernels.md, plan Q5). Dispatched by
// kernels/gpu/f16_gpu.cpp. half storage, float accumulators and epilogue math, one rounding at the output (the
// contract's exceptions: qknorm_rope_kv rounds the normalized value and the RoPE input like the basic composition,
// audio_cnn rounds each conv layer's output).
//
//   lk_f16o_gemm_{64x64,32x32}[_nt]     MLX steel GEMM (half in, float simdgroup-matrix accumulators) with a fused
//                                       epilogue: + bias, GELU(erf), + residual, clamp (gemm*_opt, audio_cnn convs)
//   lk_f16o_gemm_swiglu_{64x64,32x32}   gate and up tiles of w = [gate | up] in one threadgroup, silu(g) * u epilogue
//   lk_f16o_gemv_c{2,4,8,16}[_swiglu]   M = 1: bandwidth-bound GEMV (half4 columns, K split over the threadgroup)
//   lk_f16o_lm_head                     x[T-1] . table rows, 4 rows per simdgroup, 16-byte loads -> float logits
//   lk_f16o_qknorm_rope_kv              QK-norm + RoPE of q and k, k / v written to the caches, one simdgroup per head
//   lk_f16o_attn_window_hd{64,128}      simdgroup-matrix flash attention over packed qkv, non-causal windows
//   lk_f16o_attn_gqa_hd{64,128}         simdgroup-matrix flash attention, causal GQA over the caches (T > 1)
//   lk_f16o_attn_decode_hd{64,128}_g{1,2} + lk_f16o_attn_combine_hd{64,128}   T = 1: key-split decode attention
//   lk_f16o_conv1, lk_f16o_im2col3x3s2  audio_cnn: first conv (Ci = 1, from the f32 features), im2col of the others
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;
using namespace mlx::steel;

// IEEE semantics (as f16_basic.metal): precise exp / sin / cos / rsqrt, infinities kept (the online softmax starts at
// -INFINITY), no contraction of a * b + c (RoPE's angle is the rounded f32 product; hot loops write fma themselves).
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// ---- helpers ------------------------------------------------------------------------------------------------------

// erfc(z), relative error < 1.2e-7 (Numerical Recipes' erfcc), so GELU's negative tail keeps its relative accuracy.
// The same function as f16_basic.metal's lk_erfc.
inline float lko_erfc(float z) {
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
inline float lko_gelu(float x) { return 0.5f * x * lko_erfc(-x * 0.70710678118654752f); }
inline float lko_silu_mul(float g, float u) { return g / (1.0f + exp(-g)) * u; }

// ---- steel GEMM with fused epilogues ------------------------------------------------------------------------------

struct LkF16oGemmParams {
  int M, N, K, lda, ldb, ldd, ldr, tiles_n, tiles_m, swizzle_log, gemm_k_iterations;
  uint has_bias, act, has_res, has_clamp;  // act: 0 none, 1 gelu_erf (applied after the bias, before the residual)
  float lim;
};

struct LkoAdd {
  METAL_FUNC float apply(float a, half b) const { return a + float(b); }
};
struct LkoGelu {
  METAL_FUNC float apply(float v) const { return lko_gelu(v); }
};
struct LkoClamp {
  float lim;
  METAL_FUNC float apply(float v) const { return clamp(v, -lim, lim); }
};

// Runs steel's tile loop for the tile at (c_row, c_col) into mma_op (aligned fast paths per tile, safe loads at the
// edges; K's tail through the safe last iteration).
template <typename gk, int BM, int BN>
METAL_FUNC void lko_gemm_tile(const device half* A, const device half* B, int lda, int ldb, threadgroup half* As,
                              threadgroup half* Bs, thread typename gk::mma_t& mma_op, short tgp_bm, short tgp_bn,
                              int k_iters, short lbk, uint sg, uint lane) {
  thread typename gk::loader_a_t loader_a(A, lda, As, sg, lane);
  thread typename gk::loader_b_t loader_b(B, ldb, Bs, sg, lane);
  if (tgp_bm == BM && tgp_bn == BN)
    gk::template gemm_loop<true, true, false>(As, Bs, k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
  else if (tgp_bn == BN)
    gk::template gemm_loop<false, true, false>(As, Bs, k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
  else if (tgp_bm == BM)
    gk::template gemm_loop<true, false, false>(As, Bs, k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
  else
    gk::template gemm_loop<false, false, false>(As, Bs, k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
}

// D[M, N] = epilogue(A[M, K] @ op(B)), op(B) = B [K, N] or B^T for B [N, K] (TB). Epilogue on the float accumulators:
// + bias[col], GELU(erf), + R[row, col], clamp(+-lim); stored rounded once to half.
template <int BM, int BN, int BK, int WM, int WN, bool TB>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_f16o_gemm(
    const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], const device half* bias [[buffer(2)]],
    device half* D [[buffer(3)]], const device half* R [[buffer(4)]], const constant LkF16oGemmParams* p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using gk = GEMMKernel<half, half, BM, BN, BK, WM, WN, false, TB, false, false, float>;
  threadgroup half As[gk::tgp_mem_size_a];
  threadgroup half Bs[gk::tgp_mem_size_b];
  const int tid_y = ((tid.y) << p->swizzle_log) + ((tid.x) & ((1 << p->swizzle_log) - 1));
  const int tid_x = (tid.x) >> p->swizzle_log;
  if (p->tiles_n <= tid_x || p->tiles_m <= tid_y) return;
  const int c_row = tid_y * BM, c_col = tid_x * BN;
  const short tgp_bm = min(BM, p->M - c_row), tgp_bn = min(BN, p->N - c_col);
  const short lbk = p->K - p->gemm_k_iterations * BK;
  thread typename gk::mma_t mma_op(sg, lane);
  lko_gemm_tile<gk, BM, BN>(A + size_t(c_row) * p->lda, B + (TB ? size_t(c_col) * p->ldb : size_t(c_col)), p->lda, p->ldb, As,
                    Bs, mma_op, tgp_bm, tgp_bn, p->gemm_k_iterations, lbk, sg, lane);
  threadgroup_barrier(mem_flags::mem_none);
  const short2 dims(tgp_bn, tgp_bm);
  if (p->has_bias) mma_op.apply_epilogue_safe(bias + c_col, 0, 1, dims, LkoAdd());
  if (p->act == 1) mma_op.apply_epilogue(LkoGelu());
  if (p->has_res) mma_op.apply_epilogue_safe(R + size_t(c_row) * p->ldr + c_col, p->ldr, 1, dims, LkoAdd());
  if (p->has_clamp) mma_op.apply_epilogue(LkoClamp{p->lim});
  mma_op.store_result_safe(D + size_t(c_row) * p->ldd + c_col, p->ldd, dims);
}

#define LKO_GEMM(name, BM, BN, BK, WM, WN, TB)                                                                     \
  template [[host_name(name)]] [[kernel]] void lk_f16o_gemm<BM, BN, BK, WM, WN, TB>(                               \
      const device half*, const device half*, const device half*, device half*, const device half*,                \
      const constant LkF16oGemmParams*, uint, uint, uint3);

// BK = 32: M2 Max, [2048 x 2048] @ [2048 x 4096]: 4.53 ms vs 5.14 ms with BK = 16 (64x32 / 32x64 / WM 1 and a
// swizzle were not better). The 32x32 tiles serve small grids (more threadgroups), where the GEMM is bandwidth-bound.
LKO_GEMM("lk_f16o_gemm_64x64", 64, 64, 32, 2, 2, false)
LKO_GEMM("lk_f16o_gemm_32x32", 32, 32, 32, 2, 2, false)
LKO_GEMM("lk_f16o_gemm_64x64_nt", 64, 64, 32, 2, 2, true)
LKO_GEMM("lk_f16o_gemm_32x32_nt", 32, 32, 32, 2, 2, true)

// SwiGLU GEMM: B = [gate | up] [K, 2N] (N = p->N output columns, ldb = 2N): the threadgroup computes the gate tile
// (columns c_col..) and the up tile (columns N + c_col..) of the same rows, then D = silu(g) * u from the float
// accumulators (one rounding). A is read twice (the second pass mostly from cache).
template <int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_f16o_gemm_swiglu(
    const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], device half* D [[buffer(2)]],
    const constant LkF16oGemmParams* p [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint3 tid [[threadgroup_position_in_grid]]) {
  using gk = GEMMKernel<half, half, BM, BN, BK, WM, WN, false, false, false, false, float>;
  threadgroup half As[gk::tgp_mem_size_a];
  threadgroup half Bs[gk::tgp_mem_size_b];
  const int tid_y = ((tid.y) << p->swizzle_log) + ((tid.x) & ((1 << p->swizzle_log) - 1));
  const int tid_x = (tid.x) >> p->swizzle_log;
  if (p->tiles_n <= tid_x || p->tiles_m <= tid_y) return;
  const int c_row = tid_y * BM, c_col = tid_x * BN;
  const short tgp_bm = min(BM, p->M - c_row), tgp_bn = min(BN, p->N - c_col);
  const short lbk = p->K - p->gemm_k_iterations * BK;
  const device half* Ar = A + size_t(c_row) * p->lda;
  thread typename gk::mma_t gate(sg, lane);
  lko_gemm_tile<gk, BM, BN>(Ar, B + c_col, p->lda, p->ldb, As, Bs, gate, tgp_bm, tgp_bn, p->gemm_k_iterations, lbk, sg, lane);
  thread typename gk::mma_t up(sg, lane);
  lko_gemm_tile<gk, BM, BN>(Ar, B + p->N + c_col, p->lda, p->ldb, As, Bs, up, tgp_bm, tgp_bn, p->gemm_k_iterations, lbk, sg,
                    lane);
  threadgroup_barrier(mem_flags::mem_none);
  constexpr short NE = decltype(gate.Ctile)::kElemsPerTile;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < NE; i++) gate.Ctile.elems()[i] = lko_silu_mul(gate.Ctile.elems()[i], up.Ctile.elems()[i]);
  gate.store_result_safe(D + size_t(c_row) * p->ldd + c_col, p->ldd, short2(tgp_bn, tgp_bm));
}

#define LKO_GEMM_SWIGLU(name, BM, BN, BK, WM, WN)                                                                  \
  template [[host_name(name)]] [[kernel]] void lk_f16o_gemm_swiglu<BM, BN, BK, WM, WN>(                            \
      const device half*, const device half*, device half*, const constant LkF16oGemmParams*, uint, uint, uint3);

LKO_GEMM_SWIGLU("lk_f16o_gemm_swiglu_64x64", 64, 64, 32, 2, 2)
LKO_GEMM_SWIGLU("lk_f16o_gemm_swiglu_32x32", 32, 32, 32, 2, 2)

// ---- M = 1: GEMV ----------------------------------------------------------------------------------------------------

struct LkF16oGemvParams {
  uint k, n, ldw;  // n output columns; w row stride ldw (= n, or 2n for SwiGLU's [gate | up])
  uint has_bias, act, has_res, has_clamp;
  float lim;
};

// Partial sums of columns c .. c+3 over the K slice rows s, s + KS, ... (4 rows in flight per step). Vector loads when
// the row is 8-byte aligned and the 4 columns exist, else guarded scalar loads.
template <uint KS, typename XP>
METAL_FUNC float4 lko_gemv_cols(XP x, const device half* w, uint k, uint ldw, uint n, uint c, uint s,
                                bool vec) {
  float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  if (vec) {
    const device half* wc = w + c;
    uint r = s;
    for (; r + 7 * KS < k; r += 8 * KS) {
      half4 wv[8];
      _Pragma("clang loop unroll(full)")
      for (uint u = 0; u < 8; ++u) wv[u] = *reinterpret_cast<const device half4*>(wc + size_t(r + u * KS) * ldw);
      a0 = fma(float(x[r]), float4(wv[0]), a0);
      a1 = fma(float(x[r + KS]), float4(wv[1]), a1);
      a2 = fma(float(x[r + 2 * KS]), float4(wv[2]), a2);
      a3 = fma(float(x[r + 3 * KS]), float4(wv[3]), a3);
      a0 = fma(float(x[r + 4 * KS]), float4(wv[4]), a0);
      a1 = fma(float(x[r + 5 * KS]), float4(wv[5]), a1);
      a2 = fma(float(x[r + 6 * KS]), float4(wv[6]), a2);
      a3 = fma(float(x[r + 7 * KS]), float4(wv[7]), a3);
    }
    for (; r < k; r += KS) a0 = fma(float(x[r]), float4(*reinterpret_cast<const device half4*>(wc + size_t(r) * ldw)), a0);
  } else {
    for (uint r = s; r < k; r += KS) {
      const float xv = float(x[r]);
      const device half* wr = w + size_t(r) * ldw;
      for (uint e = 0; e < 4; ++e)
        if (c + e < n) a0[e] = fma(xv, float(wr[c + e]), a0[e]);
    }
  }
  return (a0 + a1) + (a2 + a3);
}

// Sums the partials of the KS slices of each column group: lanes of one simdgroup first (shuffles), then the 8
// simdgroups through threadgroup memory. Returns the total on threads tid < CW.
template <uint CW>
METAL_FUNC float4 lko_gemv_reduce(float4 acc, threadgroup float4 (*part)[CW], uint tid, uint lane, uint sg) {
  _Pragma("clang loop unroll(full)")
  for (uint o = CW; o < 32; o *= 2) acc += simd_shuffle_xor(acc, o);
  if (lane < CW) part[sg][lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 r = 0.0f;
  if (tid < CW) {
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 8; ++i) r += part[i][tid];
  }
  return r;
}

// out[n] = epilogue(sum_k x[k] w[k, n]) for x [1, K], w [K, ldw]. One 256-thread threadgroup per 4 * CW output
// columns: thread = (column group tid % CW, K slice tid / CW). SWIGLU: columns c of the gate half and n + c of the up
// half, out = silu(g) * u.
template <uint CW, bool SWIGLU>
[[kernel]] void lk_f16o_gemv(const device half* x [[buffer(0)]], const device half* w [[buffer(1)]],
                             const device half* bias [[buffer(2)]], device half* out [[buffer(3)]],
                             const device half* res [[buffer(4)]], constant LkF16oGemvParams& p [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint KS = 256 / CW;
  threadgroup float4 part[8][CW];
  const uint c = (tg * CW + tid % CW) * 4, s = tid / CW;
  const bool live = c < p.n;
  const bool vec = live && c + 4 <= p.n && (p.ldw % 4) == 0 && (!SWIGLU || (p.n % 4) == 0);
  float4 acc = live ? lko_gemv_cols<KS>(x, w, p.k, p.ldw, p.n, c, s, vec) : 0.0f;
  float4 g = lko_gemv_reduce<CW>(acc, part, tid, lane, sg);
  float4 u = 0.0f;
  if (SWIGLU) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc = live ? lko_gemv_cols<KS>(x, w + p.n, p.k, p.ldw, p.n, c, s, vec) : 0.0f;
    u = lko_gemv_reduce<CW>(acc, part, tid, lane, sg);
  }
  if (tid >= CW || !live) return;
  for (uint e = 0; e < 4; ++e) {
    if (c + e >= p.n) break;
    float r = g[e];
    if (SWIGLU) {
      r = lko_silu_mul(r, u[e]);
    } else {
      if (p.has_bias) r += float(bias[c + e]);
      if (p.act == 1) r = lko_gelu(r);
      if (p.has_res) r += float(res[c + e]);
      if (p.has_clamp) r = clamp(r, -p.lim, p.lim);
    }
    out[c + e] = half(r);
  }
}

#define LKO_GEMV(name, CW, SW)                                                                                     \
  template [[host_name(name)]] [[kernel]] void lk_f16o_gemv<CW, SW>(                                               \
      const device half*, const device half*, const device half*, device half*, const device half*,                \
      constant LkF16oGemvParams&, uint, uint, uint, uint);

LKO_GEMV("lk_f16o_gemv_c2", 2, false)
LKO_GEMV("lk_f16o_gemv_c4", 4, false)
LKO_GEMV("lk_f16o_gemv_c8", 8, false)
LKO_GEMV("lk_f16o_gemv_c16", 16, false)
LKO_GEMV("lk_f16o_gemv_c2_swiglu", 2, true)
LKO_GEMV("lk_f16o_gemv_c4_swiglu", 4, true)
LKO_GEMV("lk_f16o_gemv_c8_swiglu", 8, true)
LKO_GEMV("lk_f16o_gemv_c16_swiglu", 16, true)

// ---- rmsnorm + GEMV (M = 1): out = y(rmsnorm(x, gamma) @ w), y = identity or SwiGLU ---------------------------------
// The prologue is lk_f16_rmsnorm with 256 threads (same strided sums, simd_sum, then the 8 simdgroup partials in
// order: the same r, bit for bit), each threadgroup normalizing the row into threadgroup memory; then lk_f16o_gemv's
// K-split over it. Saves the rmsnorm dispatch before the decode step's qkv and gate|up GEMVs.
constant constexpr uint LKO_RMS_MAXK = 4096;
struct LkF16oRmsGemvParams {
  uint k, n, ldw;
  float eps;
};
template <uint CW, bool SWIGLU>
[[kernel]] void lk_f16o_rms_gemv(const device half* x [[buffer(0)]], const device half* gam [[buffer(1)]],
                                 const device half* w [[buffer(2)]], device half* out [[buffer(3)]],
                                 constant LkF16oRmsGemvParams& p [[buffer(4)]],
                                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint KS = 256 / CW;
  threadgroup half xs[LKO_RMS_MAXK];
  threadgroup float red[8];
  threadgroup float4 part[8][CW];
  float s = 0.0f;
  for (uint c = tid; c < p.k; c += 256) {
    const float v = float(x[c]);
    s += v * v;
  }
  s = simd_sum(s);
  if (lane == 0) red[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (uint i = 0; i < 8; ++i) tot += red[i];
  const float r = rsqrt(tot / float(p.k) + p.eps);
  for (uint c = tid; c < p.k; c += 256) xs[c] = half(float(gam[c]) * float(half(float(x[c]) * r)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup half* xv = xs;
  const uint c = (tg * CW + tid % CW) * 4, sl = tid / CW;
  const bool live = c < p.n;
  const bool vec = live && c + 4 <= p.n && (p.ldw % 4) == 0 && (!SWIGLU || (p.n % 4) == 0);
  float4 acc = live ? lko_gemv_cols<KS>(xv, w, p.k, p.ldw, p.n, c, sl, vec) : 0.0f;
  float4 g = lko_gemv_reduce<CW>(acc, part, tid, lane, sg);
  float4 u = 0.0f;
  if (SWIGLU) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc = live ? lko_gemv_cols<KS>(xv, w + p.n, p.k, p.ldw, p.n, c, sl, vec) : 0.0f;
    u = lko_gemv_reduce<CW>(acc, part, tid, lane, sg);
  }
  if (tid >= CW || !live) return;
  for (uint e = 0; e < 4; ++e) {
    if (c + e >= p.n) break;
    out[c + e] = half(SWIGLU ? lko_silu_mul(g[e], u[e]) : g[e]);
  }
}

#define LKO_RMS_GEMV(name, CW, SW)                                                                                 \
  template [[host_name(name)]] [[kernel]] void lk_f16o_rms_gemv<CW, SW>(                                           \
      const device half*, const device half*, const device half*, device half*, constant LkF16oRmsGemvParams&,     \
      uint, uint, uint, uint);

LKO_RMS_GEMV("lk_f16o_rms_gemv_c2", 2, false)
LKO_RMS_GEMV("lk_f16o_rms_gemv_c4", 4, false)
LKO_RMS_GEMV("lk_f16o_rms_gemv_c8", 8, false)
LKO_RMS_GEMV("lk_f16o_rms_gemv_c16", 16, false)
LKO_RMS_GEMV("lk_f16o_rms_gemv_c2_swiglu", 2, true)
LKO_RMS_GEMV("lk_f16o_rms_gemv_c4_swiglu", 4, true)
LKO_RMS_GEMV("lk_f16o_rms_gemv_c8_swiglu", 8, true)
LKO_RMS_GEMV("lk_f16o_rms_gemv_c16_swiglu", 16, true)

// ---- LM head: out[v] = x[t-1] . table[v] (float) --------------------------------------------------------------------

struct LkF16oLmHeadParams {
  uint t, d, v;
};

// 8 simdgroups per threadgroup, 4 table rows per simdgroup: lanes stride over d in 8-half (16-byte) chunks, four
// independent row streams in flight; simd_sum per row. Scalar path when d % 8 != 0.
kernel void lk_f16o_lm_head(const device half* x [[buffer(0)]], const device half* table [[buffer(1)]],
                            device float* out [[buffer(2)]], constant LkF16oLmHeadParams& p [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint R = 4;
  const uint v0 = (tg * 8 + sg) * R;
  if (v0 >= p.v) return;
  const device half* xr = x + size_t(p.t - 1) * p.d;
  float acc[R] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (p.d % 8 == 0) {
    for (uint c = lane * 8; c < p.d; c += 256) {
      const float4 x0 = float4(*reinterpret_cast<const device half4*>(xr + c));
      const float4 x1 = float4(*reinterpret_cast<const device half4*>(xr + c + 4));
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < R; ++r) {
        const uint v = min(v0 + r, p.v - 1);  // rows past the table re-read the last one (not stored)
        const device half* tr = table + size_t(v) * p.d + c;
        const float4 t0 = float4(*reinterpret_cast<const device half4*>(tr));
        const float4 t1 = float4(*reinterpret_cast<const device half4*>(tr + 4));
        acc[r] += dot(x0, t0) + dot(x1, t1);
      }
    }
  } else {
    for (uint c = lane; c < p.d; c += 32) {
      const float xv = float(xr[c]);
      for (uint r = 0; r < R; ++r) acc[r] = fma(xv, float(table[size_t(min(v0 + r, p.v - 1)) * p.d + c]), acc[r]);
    }
  }
  for (uint r = 0; r < R; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && v0 + r < p.v) out[v0 + r] = s;
  }
}

// ---- QK-norm + RoPE + KV write ----------------------------------------------------------------------------------

struct LkF16oQkrParams {
  uint t, q_heads, kv_heads, cap;
  float eps;
  float two_pi_hi, two_pi_lo;  // 2 pi = hi + lo
  float inv_freq[64];
};

// qkv [t, (q_heads + 2 kv_heads) * 128]. One 256-thread threadgroup per row i: the row's 64 RoPE cos / sin pairs are
// computed once into threadgroup memory (every head of the row shares them), then simdgroup sg takes head slots sg,
// sg + 8, ...: slots [0, q_heads) are q heads (rmsnorm with qg, RoPE, -> out [t, q_heads * 128]), [q_heads, q_heads +
// kv_heads) k heads (kg, RoPE, -> kc[pos + i]), the rest v heads (copied to vc[pos + i]). Lane l holds dims 2l, 2l + 1
// and 2l + 64, 2l + 65 (the RoPE pairs). The basic composition's roundings: h = half(x * rsqrt(mean(x^2) + eps)),
// y = half(g * h), out = half(rope(y)); the angle and cos / sin as lk_f16_rope. pos + t <= cap is checked on the CPU
// (here an invalid pos writes nothing).
kernel void lk_f16o_qknorm_rope_kv(const device half* qkv [[buffer(0)]], const device half* qg [[buffer(1)]],
                                   const device half* kg [[buffer(2)]], const device long* pos [[buffer(3)]],
                                   device half* kc [[buffer(4)]], device half* vc [[buffer(5)]],
                                   device half* out [[buffer(6)]], constant LkF16oQkrParams& p [[buffer(7)]],
                                   uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float cs[64], sn[64];
  const uint slots = p.q_heads + 2 * p.kv_heads;
  const long pos0 = pos[0];
  if (i >= p.t || pos0 < 0 || pos0 + long(p.t) > long(p.cap)) return;  // uniform over the threadgroup
  const long row = pos0 + long(i);
  if (tid < 64) {
    const float angle = float(row) * p.inv_freq[tid];  // the position is exact below 2^24
    const float k = rint(angle * 0.15915494309189535f);
    float rr = fma(-k, p.two_pi_hi, angle);
    rr = fma(-k, p.two_pi_lo, rr);
    cs[tid] = cos(rr);
    sn[tid] = sin(rr);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint kvd = p.kv_heads * 128, j = 2 * lane;
  const float2 c = float2(cs[j], cs[j + 1]), s = float2(sn[j], sn[j + 1]);
  for (uint slot = sg; slot < slots; slot += 8) {
    const device half* src = qkv + size_t(i) * slots * 128 + slot * 128;
    if (slot >= p.q_heads + p.kv_heads) {  // v: copy
      device half* dst = vc + size_t(row) * kvd + (slot - p.q_heads - p.kv_heads) * 128;
      *reinterpret_cast<device half2*>(dst + j) = *reinterpret_cast<const device half2*>(src + j);
      *reinterpret_cast<device half2*>(dst + j + 64) = *reinterpret_cast<const device half2*>(src + j + 64);
      continue;
    }
    const bool is_q = slot < p.q_heads;
    const device half* g = is_q ? qg : kg;
    device half* dst = is_q ? out + size_t(i) * p.q_heads * 128 + slot * 128
                            : kc + size_t(row) * kvd + (slot - p.q_heads) * 128;
    const float2 x0 = float2(*reinterpret_cast<const device half2*>(src + j));
    const float2 x1 = float2(*reinterpret_cast<const device half2*>(src + j + 64));
    const float ss = simd_sum(x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y);
    const float r = rsqrt(ss / 128.0f + p.eps);
    const float2 g0 = float2(*reinterpret_cast<const device half2*>(g + j));
    const float2 g1 = float2(*reinterpret_cast<const device half2*>(g + j + 64));
    const float2 y0 = float2(half2(g0 * float2(half2(x0 * r))));
    const float2 y1 = float2(half2(g1 * float2(half2(x1 * r))));
    *reinterpret_cast<device half2*>(dst + j) = half2(y0 * c - y1 * s);
    *reinterpret_cast<device half2*>(dst + j + 64) = half2(y1 * c + y0 * s);
  }
}

// ---- flash attention on simdgroup matrices --------------------------------------------------------------------------
// One simdgroup owns 8 query rows; its Q fragments (scaled into the exp2 domain) and O accumulators stay in registers
// as float 8x8 simdgroup matrices. Per 32-key tile, K and V go through threadgroup memory as half, shared by the
// threadgroup's simdgroups, and are multiplied as half fragments (mixed-precision MMA: half operands are exact in the
// float products, accumulation in float). S = Q K^T, online softmax in registers (lane = row fm, columns fn, fn + 1,
// MLX's BaseMMAFrag layout; a row lives on lanes {x, x^1, x^8, x^9}), O = diag(corr) O + P V with P float; out = O / l
// rounded once.

constexpr constant uint LKO_BK = 32;

struct LkoFlashArgs {
  const device half* q;   // row 0 of the queries, at the head's column
  const device half* kb;  // key row 0 (absolute index 0), at the head's column
  const device half* vb;
  device half* out;
  uint ldq, ldk, ldo;
  uint r0;            // this simdgroup's first query row
  uint r_end;         // query rows < r_end are valid
  uint k_beg, k_end;  // keys of the threadgroup's tiles: [k_beg, k_end)
  uint k_mine;        // keys this simdgroup needs (< k_end; causal)
  long qpos_base;     // causal: query row i is at position qpos_base + i
  bool causal;
  float scale;
};

template <uint HD>
METAL_FUNC void lko_flash(thread const LkoFlashArgs& a, threadgroup half* Ks, threadgroup half* Vs, uint tid,
                          uint nthreads, uint lane) {
  constexpr uint NC = HD / 8, LV = HD + 8, NKB = LKO_BK / 8;
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  const uint i = a.r0 + fm;
  const bool row_ok = i < a.r_end;
  const long qpos = a.qpos_base + long(i);
  const float qs = a.scale * M_LOG2E_F;
  simdgroup_float8x8 Qf[NC], Of[NC];
  _Pragma("clang loop unroll(full)")
  for (uint c = 0; c < NC; ++c) {
    float2 v = 0.0f;
    if (row_ok) v = float2(*reinterpret_cast<const device half2*>(a.q + size_t(i) * a.ldq + c * 8 + fn)) * qs;
    thread auto& qe = Qf[c].thread_elements();
    qe[0] = v.x;
    qe[1] = v.y;
    Of[c] = simdgroup_float8x8(0.0f);
  }
  float m = -INFINITY, l = 0.0f;
  for (uint k0 = a.k_beg; k0 < a.k_end; k0 += LKO_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < LKO_BK * HD / 4; e += nthreads) {
      const uint r = e / (HD / 4), c = (e % (HD / 4)) * 4, j = k0 + r;
      half4 kv = 0.0h, vv = 0.0h;
      if (j < a.k_end) {
        kv = *reinterpret_cast<const device half4*>(a.kb + size_t(j) * a.ldk + c);
        vv = *reinterpret_cast<const device half4*>(a.vb + size_t(j) * a.ldk + c);
      }
      *reinterpret_cast<threadgroup half4*>(Ks + r * LV + c) = kv;
      *reinterpret_cast<threadgroup half4*>(Vs + r * LV + c) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 >= a.k_mine) continue;  // causal: past this simdgroup's rows (still loaded for the others)
    simdgroup_float8x8 Sf[NKB];
    _Pragma("clang loop unroll(full)")
    for (uint kb = 0; kb < NKB; ++kb) Sf[kb] = simdgroup_float8x8(0.0f);
    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < NC; ++c) {
      _Pragma("clang loop unroll(full)")
      for (uint kb = 0; kb < NKB; ++kb) {
        simdgroup_half8x8 Kf;
        simdgroup_load(Kf, Ks + kb * 8 * LV + c * 8, LV, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Sf[kb], Qf[c], Kf, Sf[kb]);
      }
    }
    const bool edge = k0 + LKO_BK > a.k_end || (a.causal && long(k0 + LKO_BK) > a.qpos_base + long(a.r0));
    float rmax = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (uint kb = 0; kb < NKB; ++kb) {
      thread auto& se = Sf[kb].thread_elements();
      if (edge) {
        const uint j = k0 + kb * 8 + fn;
        if (j >= a.k_end || (a.causal && long(j) > qpos)) se[0] = -INFINITY;
        if (j + 1 >= a.k_end || (a.causal && long(j + 1) > qpos)) se[1] = -INFINITY;
      }
      rmax = max(rmax, max(se[0], se[1]));
    }
    rmax = max(rmax, simd_shuffle_xor(rmax, 1));
    rmax = max(rmax, simd_shuffle_xor(rmax, 8));
    const float mn = max(m, rmax);
    const bool none = mn == -INFINITY;  // only rows past r_end with nothing visible yet
    const float corr = none ? 1.0f : exp2(m - mn);
    float ls = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint kb = 0; kb < NKB; ++kb) {
      thread auto& se = Sf[kb].thread_elements();
      se[0] = none ? 0.0f : exp2(se[0] - mn);
      se[1] = none ? 0.0f : exp2(se[1] - mn);
      ls += se[0] + se[1];
    }
    l = l * corr + ls;
    m = mn;
    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < NC; ++c) {
      thread auto& oe = Of[c].thread_elements();
      oe[0] *= corr;
      oe[1] *= corr;
    }
    _Pragma("clang loop unroll(full)")
    for (uint kb = 0; kb < NKB; ++kb) {
      _Pragma("clang loop unroll(full)")
      for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 Vf;
        simdgroup_load(Vf, Vs + kb * 8 * LV + c * 8, LV);
        simdgroup_multiply_accumulate(Of[c], Sf[kb], Vf, Of[c]);
      }
    }
  }
  l += simd_shuffle_xor(l, 1);
  l += simd_shuffle_xor(l, 8);
  if (row_ok) {
    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < NC; ++c) {
      thread auto& oe = Of[c].thread_elements();
      *reinterpret_cast<device half2*>(a.out + size_t(i) * a.ldo + c * 8 + fn) = half2(oe[0] / l, oe[1] / l);
    }
  }
}

struct LkF16oWindowParams {
  uint s, heads, window, qblocks;  // qblocks: query blocks (of 8 * nsg rows) per window
  float scale;
};

// Encoder self-attention over packed qkv [s, 3 * heads * HD] (q | k | v), non-causal within windows. Grid
// (windows * qblocks, heads), nsg simdgroups per threadgroup (the host picks nsg so a 104-row window is one block).
template <uint HD>
[[kernel]] void lk_f16o_attn_window(const device half* qkv [[buffer(0)]], device half* out [[buffer(1)]],
                                    constant LkF16oWindowParams& p [[buffer(2)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup half Ks[LKO_BK * (HD + 8)];
  threadgroup half Vs[LKO_BK * (HD + 8)];
  const uint w = tg.x / p.qblocks, qb = tg.x % p.qblocks, h = tg.y;
  const uint w0 = w * p.window, w1 = min(p.s, w0 + p.window), q0 = w0 + qb * 8 * nsg;
  if (q0 >= w1 || h >= p.heads) return;  // uniform over the threadgroup
  const uint d = p.heads * HD;
  LkoFlashArgs a;
  a.q = qkv + h * HD;
  a.kb = qkv + d + h * HD;
  a.vb = qkv + 2 * d + h * HD;
  a.out = out + h * HD;
  a.ldq = 3 * d;
  a.ldk = 3 * d;
  a.ldo = d;
  a.r0 = q0 + sg * 8;
  a.r_end = min(w1, q0 + 8 * nsg);
  a.k_beg = w0;
  a.k_end = w1;
  a.k_mine = w1;
  a.qpos_base = 0;
  a.causal = false;
  a.scale = p.scale;
  lko_flash<HD>(a, Ks, Vs, tid, 32 * nsg, lane);
}

template [[host_name("lk_f16o_attn_window_hd64")]] [[kernel]] decltype(lk_f16o_attn_window<64>) lk_f16o_attn_window<64>;
template [[host_name("lk_f16o_attn_window_hd128")]] [[kernel]] decltype(lk_f16o_attn_window<128>) lk_f16o_attn_window<128>;

struct LkF16oGqaParams {
  uint t, q_heads, kv_heads, cap;
  float scale;
};

// Causal GQA over the caches (T > 1): query row i (position pos + i) of q-head h attends to cache rows 0 .. pos + i of
// kv-head h / (q_heads / kv_heads). q, out [t, q_heads * HD]; kc, vc [cap, kv_heads * HD]. Grid (query blocks of
// 8 * nsg rows, longest first; q_heads).
template <uint HD>
[[kernel]] void lk_f16o_attn_gqa(const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                 const device half* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                 device half* out [[buffer(4)]], constant LkF16oGqaParams& p [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup half Ks[LKO_BK * (HD + 8)];
  threadgroup half Vs[LKO_BK * (HD + 8)];
  const long pos0 = pos[0];
  const uint h = tg.y;
  if (h >= p.q_heads || pos0 < 0 || pos0 + long(p.t) > long(p.cap)) return;
  const uint bq = 8 * nsg, nqb = (p.t + bq - 1) / bq, q0 = (nqb - 1 - tg.x) * bq;
  const uint dq = p.q_heads * HD, dkv = p.kv_heads * HD, kvo = h / (p.q_heads / p.kv_heads) * HD;
  const uint p0 = uint(pos0);
  LkoFlashArgs a;
  a.q = q + h * HD;
  a.kb = kc + kvo;
  a.vb = vc + kvo;
  a.out = out + h * HD;
  a.ldq = dq;
  a.ldk = dkv;
  a.ldo = dq;
  a.r0 = q0 + sg * 8;
  a.r_end = p.t;
  a.k_beg = 0;
  a.k_end = p0 + min(p.t, q0 + bq);
  a.k_mine = p0 + min(p.t, a.r0 + 8);
  a.qpos_base = pos0;
  a.causal = true;
  a.scale = p.scale;
  lko_flash<HD>(a, Ks, Vs, tid, 32 * nsg, lane);
}

template [[host_name("lk_f16o_attn_gqa_hd64")]] [[kernel]] decltype(lk_f16o_attn_gqa<64>) lk_f16o_attn_gqa<64>;
template [[host_name("lk_f16o_attn_gqa_hd128")]] [[kernel]] decltype(lk_f16o_attn_gqa<128>) lk_f16o_attn_gqa<128>;

// ---- decode attention (T = 1): keys split over threadgroups --------------------------------------------------------

struct LkF16oDecodeParams {
  uint q_heads, kv_heads, cap, splits, chunk;  // split s takes keys [s * chunk, min(n, (s + 1) * chunk))
  float scale;
};

// One query (position pos) against cache rows 0 .. pos. Grid (q_heads / G, splits): the threadgroup takes G q heads of
// one kv head (they share every K / V row read) and one key range; lanes hold HD / 32 contiguous dims of a row, the
// dot products are simd_sums; simdgroup sg takes keys j_beg + sg + nsg * u, four at a time, with its own online
// softmax (exp2 domain), merged across the simdgroups through threadgroup memory. Writes the split's partial
// (max m, sum l, unnormalized acc [HD]) to part[hq][split] (HD + 4 floats); lk_f16o_attn_combine merges them.
template <uint HD, uint G>
[[kernel]] void lk_f16o_attn_decode(const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                    const device half* vc [[buffer(2)]], const device long* pos [[buffer(3)]],
                                    device float* part [[buffer(4)]], constant LkF16oDecodeParams& p [[buffer(5)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint nsg [[simdgroups_per_threadgroup]]) {
  constexpr uint PER = HD / 32, UN = 4, PS = HD + 4;
  threadgroup float tm[8][G];  // nsg <= 8
  threadgroup float tl[8][G];
  threadgroup float tacc[8][G][HD];
  const long pos0 = pos[0];
  const uint hq0 = tg.x * G, s = tg.y;
  if (hq0 >= p.q_heads || s >= p.splits || pos0 < 0 || pos0 >= long(p.cap)) return;
  const uint n = uint(pos0) + 1, j_beg = min(n, s * p.chunk), j_end = min(n, j_beg + p.chunk);
  const uint dkv = p.kv_heads * HD, kvo = hq0 / (p.q_heads / p.kv_heads) * HD + lane * PER;
  const float qs = p.scale * M_LOG2E_F;
  float qv[G][PER], acc[G][PER], m[G], l[G];
  for (uint g = 0; g < G; ++g) {
    for (uint e = 0; e < PER; ++e) {
      qv[g][e] = float(q[(hq0 + g) * HD + lane * PER + e]) * qs;
      acc[g][e] = 0.0f;
    }
    m[g] = -INFINITY;
    l[g] = 0.0f;
  }
  for (uint j0 = j_beg + sg; j0 < j_end; j0 += UN * nsg) {
    float kr[UN][PER], vr[UN][PER], sc[UN][G];
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const uint j = j0 + u * nsg;
      const bool ok = j < j_end;
      const device half* kp = kc + size_t(ok ? j : j_beg) * dkv + kvo;
      const device half* vp = vc + size_t(ok ? j : j_beg) * dkv + kvo;
      if (PER == 4) {
        const float4 k4 = float4(*reinterpret_cast<const device half4*>(kp));
        const float4 v4 = float4(*reinterpret_cast<const device half4*>(vp));
        for (uint e = 0; e < 4; ++e) kr[u][e] = k4[e], vr[u][e] = v4[e];
      } else {
        const float2 k2 = float2(*reinterpret_cast<const device half2*>(kp));
        const float2 v2 = float2(*reinterpret_cast<const device half2*>(vp));
        for (uint e = 0; e < 2; ++e) kr[u][e] = k2[e], vr[u][e] = v2[e];
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint u = 0; u < UN; ++u) {
      const bool ok = j0 + u * nsg < j_end;
      for (uint g = 0; g < G; ++g) {
        float d = 0.0f;
        for (uint e = 0; e < PER; ++e) d = fma(qv[g][e], kr[u][e], d);
        d = simd_sum(d);
        sc[u][g] = ok ? d : -INFINITY;
      }
    }
    for (uint g = 0; g < G; ++g) {
      float mx = m[g];
      for (uint u = 0; u < UN; ++u) mx = max(mx, sc[u][g]);  // finite: key j0 is always valid
      const float corr = exp2(m[g] - mx);
      l[g] *= corr;
      for (uint e = 0; e < PER; ++e) acc[g][e] *= corr;
      for (uint u = 0; u < UN; ++u) {
        const float w = exp2(sc[u][g] - mx);
        l[g] += w;
        for (uint e = 0; e < PER; ++e) acc[g][e] = fma(w, vr[u][e], acc[g][e]);
      }
      m[g] = mx;
    }
  }
  for (uint g = 0; g < G; ++g) {
    if (lane == 0) {
      tm[sg][g] = m[g];
      tl[sg][g] = l[g];
    }
    for (uint e = 0; e < PER; ++e) tacc[sg][g][lane * PER + e] = acc[g][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= G) return;
  const uint g = sg;  // simdgroup g merges q head hq0 + g
  float M = -INFINITY;
  for (uint k = 0; k < nsg; ++k) M = max(M, tm[k][g]);
  float L = 0.0f, o[PER];
  for (uint e = 0; e < PER; ++e) o[e] = 0.0f;
  if (M != -INFINITY) {
    for (uint k = 0; k < nsg; ++k) {
      const float mk = tm[k][g];
      if (mk == -INFINITY) continue;
      const float w = exp2(mk - M);
      L += w * tl[k][g];
      for (uint e = 0; e < PER; ++e) o[e] = fma(w, tacc[k][g][lane * PER + e], o[e]);
    }
  }
  device float* pp = part + (size_t(hq0 + g) * p.splits + s) * PS;
  if (lane == 0) {
    pp[0] = M;
    pp[1] = L;
  }
  for (uint e = 0; e < PER; ++e) pp[4 + lane * PER + e] = o[e];
}

template [[host_name("lk_f16o_attn_decode_hd64_g1")]] [[kernel]] decltype(lk_f16o_attn_decode<64, 1>) lk_f16o_attn_decode<64, 1>;
template [[host_name("lk_f16o_attn_decode_hd64_g2")]] [[kernel]] decltype(lk_f16o_attn_decode<64, 2>) lk_f16o_attn_decode<64, 2>;
template [[host_name("lk_f16o_attn_decode_hd128_g1")]] [[kernel]] decltype(lk_f16o_attn_decode<128, 1>) lk_f16o_attn_decode<128, 1>;
template [[host_name("lk_f16o_attn_decode_hd128_g2")]] [[kernel]] decltype(lk_f16o_attn_decode<128, 2>) lk_f16o_attn_decode<128, 2>;

// Merges the splits: one simdgroup per q head, out = sum_s 2^(m_s - M) acc_s / sum_s 2^(m_s - M) l_s (empty splits
// have m = -inf).
template <uint HD>
[[kernel]] void lk_f16o_attn_combine(const device float* part [[buffer(0)]], device half* out [[buffer(1)]],
                                     constant LkF16oDecodeParams& p [[buffer(2)]],
                                     uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint PER = HD / 32, PS = HD + 4;
  if (h >= p.q_heads) return;
  const device float* ph = part + size_t(h) * p.splits * PS;
  float M = -INFINITY;
  for (uint s = 0; s < p.splits; ++s) M = max(M, ph[s * PS]);
  float L = 0.0f, o[PER];
  for (uint e = 0; e < PER; ++e) o[e] = 0.0f;
  for (uint s = 0; s < p.splits; ++s) {
    const float ms = ph[s * PS];
    if (ms == -INFINITY) continue;
    const float w = exp2(ms - M);
    L += w * ph[s * PS + 1];
    for (uint e = 0; e < PER; ++e) o[e] = fma(w, ph[s * PS + 4 + lane * PER + e], o[e]);
  }
  for (uint e = 0; e < PER; ++e) out[h * HD + lane * PER + e] = half(o[e] / L);
}

template [[host_name("lk_f16o_attn_combine_hd64")]] [[kernel]] decltype(lk_f16o_attn_combine<64>) lk_f16o_attn_combine<64>;
template [[host_name("lk_f16o_attn_combine_hd128")]] [[kernel]] decltype(lk_f16o_attn_combine<128>) lk_f16o_attn_combine<128>;

// ---- audio_cnn ------------------------------------------------------------------------------------------------------

struct LkF16oConvParams {
  uint h_in, w_in, h_out, w_out, ci, co;
};

// First conv layer for a batch of chunks, straight from the features: x[t, f] = half(features[c, f, t]), Ci = 1,
// 3x3 stride 2 pad 1, + bias, GELU(erf), one rounding. w [co, 3, 3, 1], out [chunks, h_out, w_out, co]. Grid
// (co, w_out, chunks * h_out).
kernel void lk_f16o_conv1(const device float* feat [[buffer(0)]], const device half* w [[buffer(1)]],
                          const device half* bias [[buffer(2)]], device half* out [[buffer(3)]],
                          constant LkF16oConvParams& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
  const uint co = gid.x, ow = gid.y, ch = gid.z / p.h_out, oh = gid.z % p.h_out;
  if (co >= p.co || ow >= p.w_out) return;
  const device float* f = feat + size_t(ch) * p.w_in * p.h_in;  // [F = w_in, T = h_in]
  float acc = float(bias[co]);
  for (uint kt = 0; kt < 3; ++kt) {
    const int ih = int(oh * 2 + kt) - 1;
    if (ih < 0 || ih >= int(p.h_in)) continue;
    for (uint kf = 0; kf < 3; ++kf) {
      const int iw = int(ow * 2 + kf) - 1;
      if (iw < 0 || iw >= int(p.w_in)) continue;
      acc += float(half(f[size_t(iw) * p.h_in + uint(ih)])) * float(w[(co * 3 + kt) * 3 + kf]);
    }
  }
  out[((size_t(ch) * p.h_out + oh) * p.w_out + ow) * p.co + co] = half(lko_gelu(acc));
}

// im2col for a 3x3 stride-2 pad-1 conv over a batch of chunks: in [chunks, h_in, w_in, ci] -> A [chunks * h_out *
// w_out, 9 * ci], A[m, (kt * 3 + kf) * ci + c] = in[2 oh + kt - 1, 2 ow + kf - 1, c] (0 in the padding): the weight's
// own [co, 3, 3, ci] flattening, so the conv is A @ w^T. 4 channels per thread (ci % 4 == 0). Grid
// (9 * ci / 4, chunks * h_out * w_out).
kernel void lk_f16o_im2col3x3s2(const device half* in [[buffer(0)]], device half* a [[buffer(1)]],
                                constant LkF16oConvParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint kk = gid.x * 4, m = gid.y, k = 9 * p.ci;
  const uint npos = p.h_out * p.w_out;
  if (kk >= k) return;
  const uint ch = m / npos, r = m % npos, oh = r / p.w_out, ow = r % p.w_out;
  const uint tap = kk / p.ci, c = kk % p.ci;
  const int ih = int(2 * oh + tap / 3) - 1, iw = int(2 * ow + tap % 3) - 1;
  half4 v = 0.0h;
  if (ih >= 0 && ih < int(p.h_in) && iw >= 0 && iw < int(p.w_in))
    v = *reinterpret_cast<const device half4*>(in + ((size_t(ch) * p.h_in + uint(ih)) * p.w_in + uint(iw)) * p.ci + c);
  *reinterpret_cast<device half4*>(a + size_t(m) * k + kk) = v;
}
