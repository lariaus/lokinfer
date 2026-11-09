// f16 GEMMs on Metal 4 tensor ops (MetalPerformancePrimitives `matmul2d`): the M > 1 path of the Qwen3-ASR optimized
// GEMM ops (docs/impl-notes/qwen3-asr-f16-opt-kernels.md) on GPUs with neural accelerators (the iPhone 18 Pro Max's
// A20 Pro; dispatch rules in kernels/gpu/f16_gpu.cpp, docs/kernels/gemm.md "f16 tensor-op GEMM"). Same contract as
// the steel kernels of f16_fused.metal: half in, float accumulation (relaxed_precision = false), the fused epilogue
// (+ bias, GELU(erf), + residual, clamp, or SwiGLU) on the float accumulators, one rounding at the store.
//
// Structure after MLX's NAX GEMM (mlx 0.33, steel/gemm/nax.h + gemm_nax.h, MIT, (c) 2025 Apple Inc.): no threadgroup
// memory; each simdgroup owns an SM x SN block of the output, held in the float destination cooperative tensor of a
// matmul2d<(SM, SN, 32), execution_simdgroup> op across the whole K loop, and loads its A and B operands for each K
// step of 32 straight from device memory into the op's input cooperative tensors. A threadgroup barrier (mem_none)
// every BK keeps the simdgroups in lockstep (shared cache lines of A and B). Unlike MLX, which hard-codes the
// cooperative tensors' lane layout of the neural-accelerator GPUs, every element's position comes from the tensor's
// get_multidimensional_index (index 0 = the column / inner dimension, 1 = the row): the same kernel is correct on
// GPUs without neural accelerators (the M2 Max, where matmul2d runs on the shader ALUs with another layout: the MLX
// layout gave wrong results there, 2026-11-07).
//
//   lk_f16t_gemm_<BM>x<BN>[_nt]     D = epi(A @ op(B)), op(B) = B [K, N] or B^T for B [N, K] (audio_cnn's convs)
//   lk_f16t_gemm_swiglu_<BM>x<BN>   D = silu(A @ Bg) * (A @ Bu), B = [gate | up] [K, 2N]: both accumulators per
//                                   simdgroup share the A operand
//   lk_f16t_gemm[_swiglu]_nax_*     the same with MLX's fixed neural-accelerator fragment layout (Apple10+ only; opt-in
//                                   until validated on the phone: LOKINFER_F16T_KERNEL=nax), lk_f16t_gemm_naxr_128x128
//                                   with relaxed_precision (measurement only)
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// As f16_fused.metal: precise exp (GELU, SiLU), no contraction of a * b + c.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

#define LKT_UNROLL _Pragma("clang loop unroll(full)")
#define LKT_NO_UNROLL _Pragma("clang loop unroll(disable)")

// erfc / GELU / SiLU: the same functions as f16_fused.metal (lko_erfc: Numerical Recipes' erfcc, rel. err < 1.2e-7).
inline float lkt_erfc(float z) {
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
inline float lkt_gelu(float x) { return 0.5f * x * lkt_erfc(-x * 0.70710678118654752f); }
inline float lkt_silu_mul(float g, float u) { return g / (1.0f + exp(-g)) * u; }

// The steel kernels' parameter block (LkF16oGemmParams), so the host fills both the same way.
struct LkF16tGemmParams {
  int M, N, K, lda, ldb, ldd, ldr, tiles_n, tiles_m, swizzle_log, gemm_k_iterations;
  uint has_bias, act, has_res, has_clamp;
  float lim;
};

// ---- the simdgroup's K loop ----------------------------------------------------------------------------------------

constant constexpr short LKT_SK = 32;  // K per step

template <short SM, short SN, bool TB>
struct LktOp {
  using op_t = mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(
                                             SM, SN, LKT_SK, false, TB, false,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                                         execution_simdgroup>;
  using a_t = typename op_t::template cooperative_tensor_left_input_t<half, half, float>;
  using b_t = typename op_t::template cooperative_tensor_right_input_t<half, half, float>;
  using c_t = typename op_t::template cooperative_tensor_destination_t<a_t, b_t, float>;
};

// The op's left operand (A [SM rows, SK] at A, leading dimension lda) into a; SAFE: rows >= sm or columns >= sk read 0.
template <class CT, bool SAFE>
inline void lkt_fill(thread CT& t, const device half* p, int ld, short rows, short cols) {
  LKT_UNROLL
  for (ushort i = 0; i < t.get_capacity(); i++) {
    if (!t.is_valid_element(i)) continue;
    const auto ix = t.get_multidimensional_index(i);  // [0] = column, [1] = row
    if (SAFE)
      t[i] = (ix[1] < rows && ix[0] < cols) ? p[ix[1] * ld + ix[0]] : half(0);
    else
      t[i] = p[ix[1] * ld + ix[0]];
  }
}

// One K step of sk (<= SK) at (A, B): C[nb] += A @ op(B + nb * boff) for NB column blocks (SwiGLU: gate, up).
template <short SM, short SN, short NB, bool TB, bool SAFE>
inline void lkt_step(thread typename LktOp<SM, SN, TB>::c_t& C0, thread typename LktOp<SM, SN, TB>::c_t& C1,
                     const device half* A, const device half* B,
                     int lda, int ldb, int boff, short sm, short sn, short sk) {
  using O = LktOp<SM, SN, TB>;
  typename O::op_t op;
  auto a = op.template get_left_input_cooperative_tensor<half, half, float>();
  lkt_fill<decltype(a), SAFE>(a, A, lda, sm, sk);
  LKT_UNROLL
  for (short nb = 0; nb < NB; nb++) {
    auto b = op.template get_right_input_cooperative_tensor<half, half, float>();
    if (TB)  // B [N, K]: rows n, columns k
      lkt_fill<decltype(b), SAFE>(b, B + nb * boff, ldb, sn, sk);
    else  // B [K, N]: rows k, columns n
      lkt_fill<decltype(b), SAFE>(b, B + nb * boff, ldb, sk, sn);
    if (nb == 0)
      op.run(a, b, C0);
    else
      op.run(a, b, C1);
  }
}

// C[nb] = A [sm rows, K] @ op(B) for the simdgroup's block. Full SK steps (a mem_none threadgroup barrier every BK),
// then K's tail (< SK) with K-guarded loads. ALIGNED: the block has SM rows and SN columns (unguarded loads in the full
// steps). All simdgroups run the barriers, those without output (sm or sn <= 0) skip the work.
template <short SM, short SN, short NB, int BK, bool TB, bool ALIGNED>
inline void lkt_kloop(thread typename LktOp<SM, SN, TB>::c_t& C0, thread typename LktOp<SM, SN, TB>::c_t& C1,
                      const device half* A, const device half* B,
                      int lda, int ldb, int boff, int K, short sm, short sn) {
  const bool has_output = sm > 0 && sn > 0;
  const int a_step = LKT_SK, b_step = TB ? LKT_SK : LKT_SK * ldb;
  // One flat loop over the full K steps (nesting the BK loop around the SK loop lost the accumulator between
  // iterations on the M2's matmul2d: only the last K step survived, 2026-11-07), the barrier every BK / SK steps.
  constexpr int per_barrier = BK / LKT_SK;
  const int full = K / LKT_SK, barrier_steps = K / BK * per_barrier;
  LKT_NO_UNROLL
  for (int s = 0; s < full; s++) {
    if (s < barrier_steps && s % per_barrier == 0) threadgroup_barrier(mem_flags::mem_none);
    if (!ALIGNED && !has_output) continue;
    lkt_step<SM, SN, NB, TB, !ALIGNED>(C0, C1, A, B, lda, ldb, boff, sm, sn, LKT_SK);
    A += a_step;
    B += b_step;
  }
  if (!ALIGNED && !has_output) return;
  const int rest = K - full * LKT_SK;
  const short tail = short(rest);
  if (tail > 0) lkt_step<SM, SN, NB, TB, true>(C0, C1, A, B, lda, ldb, boff, sm, sn, tail);
}

template <class CT>
inline void lkt_clear(thread CT& C) {
  LKT_UNROLL
  for (ushort i = 0; i < C.get_capacity(); i++)
    if (C.is_valid_element(i)) C[i] = 0.0f;
}

// ---- kernels -------------------------------------------------------------------------------------------------------

// Tile (tid) -> (column block, row block); MLX's swizzle (swizzle_log 0: plain).
inline int2 lkt_tile(uint3 tid, int swizzle_log) {
  const int ty = (int(tid.y) << swizzle_log) + (int(tid.x) & ((1 << swizzle_log) - 1));
  const int tx = int(tid.x) >> swizzle_log;
  return int2(tx, ty);
}

// D[M, N] = epilogue(A[M, K] @ op(B)): + bias[col], GELU(erf), + R[row, col], clamp(+-lim), each on the float
// accumulator (the steel kernel's order), rounded once to half.
template <int BM, int BN, int WM, int WN, int BK, bool TB>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_f16t_gemm(
    const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], const device half* bias [[buffer(2)]],
    device half* D [[buffer(3)]], const device half* R [[buffer(4)]], const constant LkF16tGemmParams* p [[buffer(5)]],
    ushort sg [[simdgroup_index_in_threadgroup]], uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = BM / WM, SN = BN / WN;
  const int2 t = lkt_tile(tid, p->swizzle_log);
  if (p->tiles_n <= t.x || p->tiles_m <= t.y) return;
  const int row0 = t.y * BM + SM * (sg / WN), col0 = t.x * BN + SN * (sg % WN);
  const short sm = short(clamp(p->M - row0, -1, int(SM))), sn = short(clamp(p->N - col0, -1, int(SN)));
  const device half* Ab = A + size_t(row0) * p->lda;
  const device half* Bb = B + (TB ? size_t(col0) * p->ldb : size_t(col0));
  typename LktOp<SM, SN, TB>::c_t C;
  lkt_clear(C);
  if (sm == SM && sn == SN)
    lkt_kloop<SM, SN, 1, BK, TB, true>(C, C, Ab, Bb, p->lda, p->ldb, 0, p->K, sm, sn);
  else
    lkt_kloop<SM, SN, 1, BK, TB, false>(C, C, Ab, Bb, p->lda, p->ldb, 0, p->K, sm, sn);
  if (sm <= 0 || sn <= 0) return;
  LKT_UNROLL
  for (ushort i = 0; i < C.get_capacity(); i++) {
    if (!C.is_valid_element(i)) continue;
    const auto ix = C.get_multidimensional_index(i);
    if (ix[1] >= sm || ix[0] >= sn) continue;
    const int row = row0 + ix[1], col = col0 + ix[0];
    float v = C[i];
    if (p->has_bias) v = v + float(bias[col]);
    if (p->act == 1) v = lkt_gelu(v);
    if (p->has_res) v = v + float(R[size_t(row) * p->ldr + col]);
    if (p->has_clamp) v = clamp(v, -p->lim, p->lim);
    D[size_t(row) * p->ldd + col] = half(v);
  }
}

#define LKT_GEMM(name, BM, BN, WM, WN, BK, TB)                                                                     \
  template [[host_name(name)]] [[kernel]] void lk_f16t_gemm<BM, BN, WM, WN, BK, TB>(                               \
      const device half*, const device half*, const device half*, device half*, const device half*,                \
      const constant LkF16tGemmParams*, ushort, uint3);

// 128 x 128 with 4 x 4 simdgroups and BK 512: MLX's iPhone configuration; 64 x 128 / 64 x 64 for smaller grids.
LKT_GEMM("lk_f16t_gemm_128x128", 128, 128, 4, 4, 512, false)
LKT_GEMM("lk_f16t_gemm_64x128", 64, 128, 2, 4, 512, false)
LKT_GEMM("lk_f16t_gemm_64x64", 64, 64, 2, 2, 512, false)
LKT_GEMM("lk_f16t_gemm_128x128_nt", 128, 128, 4, 4, 512, true)
LKT_GEMM("lk_f16t_gemm_64x128_nt", 64, 128, 2, 4, 512, true)
LKT_GEMM("lk_f16t_gemm_64x64_nt", 64, 64, 2, 2, 512, true)

// SwiGLU: B = [gate | up] [K, 2N] (ldb = 2N, N = p->N output columns): each simdgroup accumulates its gate block
// (columns col0..) and up block (N + col0..) from the same A operand, then D = silu(g) * u (one rounding).
template <int BM, int BN, int WM, int WN, int BK>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_f16t_gemm_swiglu(
    const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], device half* D [[buffer(2)]],
    const constant LkF16tGemmParams* p [[buffer(3)]], ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = BM / WM, SN = BN / WN;
  const int2 t = lkt_tile(tid, p->swizzle_log);
  if (p->tiles_n <= t.x || p->tiles_m <= t.y) return;
  const int row0 = t.y * BM + SM * (sg / WN), col0 = t.x * BN + SN * (sg % WN);
  const short sm = short(clamp(p->M - row0, -1, int(SM))), sn = short(clamp(p->N - col0, -1, int(SN)));
  const device half* Ab = A + size_t(row0) * p->lda;
  const device half* Bb = B + size_t(col0);
  typename LktOp<SM, SN, false>::c_t G, U;
  lkt_clear(G);
  lkt_clear(U);
  if (sm == SM && sn == SN)
    lkt_kloop<SM, SN, 2, BK, false, true>(G, U, Ab, Bb, p->lda, p->ldb, p->N, p->K, sm, sn);
  else
    lkt_kloop<SM, SN, 2, BK, false, false>(G, U, Ab, Bb, p->lda, p->ldb, p->N, p->K, sm, sn);
  if (sm <= 0 || sn <= 0) return;
  LKT_UNROLL
  for (ushort i = 0; i < G.get_capacity(); i++) {
    if (!G.is_valid_element(i)) continue;
    const auto ix = G.get_multidimensional_index(i);
    if (ix[1] >= sm || ix[0] >= sn) continue;
    D[size_t(row0 + ix[1]) * p->ldd + col0 + ix[0]] = half(lkt_silu_mul(G[i], U[i]));
  }
}

#define LKT_GEMM_SWIGLU(name, BM, BN, WM, WN, BK)                                                                  \
  template [[host_name(name)]] [[kernel]] void lk_f16t_gemm_swiglu<BM, BN, WM, WN, BK>(                            \
      const device half*, const device half*, device half*, const constant LkF16tGemmParams*, ushort, uint3);

LKT_GEMM_SWIGLU("lk_f16t_gemm_swiglu_128x128", 128, 128, 4, 4, 512)
LKT_GEMM_SWIGLU("lk_f16t_gemm_swiglu_64x128", 64, 128, 2, 4, 512)
LKT_GEMM_SWIGLU("lk_f16t_gemm_swiglu_64x64", 64, 64, 2, 2, 512)

// ---- neural-accelerator layout variant (lk_f16t_*_nax) ---------------------------------------------------------------
// MLX's scheme exactly: the accumulators live in 16 x 16 float register fragments, operands are loaded as 16 x 16 half
// fragments in the lane layout of the neural accelerators' matmul2d (16, 32, 16) cooperative tensors (lane l: rows fm,
// fm + 8, columns fn .. fn + 3; 4 contiguous halves per row: vector loads), copied into the op's cooperative tensors
// for each run. Only valid where that layout holds (Apple10+ GPUs; wrong on the M2): tested on the phone.

typedef vec<half, 8> lkt_hfrag;
typedef vec<float, 8> lkt_ffrag;

inline short2 lkt_coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2(fn, fm);
}
inline void lkt_fload(thread lkt_hfrag& f, const device half* src, int ld, short2 sc) {
  src += sc.y * ld + sc.x;
  LKT_UNROLL
  for (short i = 0; i < 2; i++) {
    LKT_UNROLL
    for (short j = 0; j < 4; j++) f[i * 4 + j] = src[i * 8 * ld + j];
  }
}
inline void lkt_fload_safe(thread lkt_hfrag& f, const device half* src, int ld, short2 sc, short rows, short cols) {
  src += sc.y * ld + sc.x;
  const short lr = rows - sc.y, lc = cols - sc.x;
  LKT_UNROLL
  for (short i = 0; i < 2; i++) {
    LKT_UNROLL
    for (short j = 0; j < 4; j++) f[i * 4 + j] = (i * 8 < lr && j < lc) ? src[i * 8 * ld + j] : half(0);
  }
}

template <bool TB, bool RELAXED>
inline void lkt_fmma(thread lkt_ffrag& c0, thread lkt_ffrag& c1, thread const lkt_hfrag& a, thread const lkt_hfrag& b0,
                     thread const lkt_hfrag& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, TB, RELAXED, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto cc = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(ca)>,
                                                           remove_addrspace_t<decltype(cb)>, float>();
  LKT_UNROLL
  for (short i = 0; i < 8; i++) {
    ca[i] = a[i];
    cb[i] = b0[i];
    cb[8 + i] = b1[i];
    cc[i] = c0[i];
    cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
  LKT_UNROLL
  for (short i = 0; i < 8; i++) {
    c0[i] = cc[i];
    c1[i] = cc[8 + i];
  }
}

// One K step (sk <= 32) for an SM x SN block, NB column blocks (B + nb * boff).
template <short TM, short TN, short NB, bool TB, bool SAFE, bool RELAXED>
inline void lkt_fstep(thread lkt_ffrag (&C)[NB][TM][TN], const device half* A, const device half* B, int lda, int ldb,
                      int boff, short2 sc, short sm, short sn, short sk) {
  lkt_hfrag af[TM][2];
  LKT_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    LKT_UNROLL
    for (short kk = 0; kk < 2; kk++) {
      const device half* p = A + (mm * 16) * lda + kk * 16;
      if (SAFE)
        lkt_fload_safe(af[mm][kk], p, lda, sc, short(sm - mm * 16), short(sk - kk * 16));
      else
        lkt_fload(af[mm][kk], p, lda, sc);
    }
  }
  LKT_UNROLL
  for (short nb = 0; nb < NB; nb++) {
    lkt_hfrag bf[2][TN];
    LKT_UNROLL
    for (short kk = 0; kk < 2; kk++) {
      LKT_UNROLL
      for (short nn = 0; nn < TN; nn++) {
        if (TB) {
          const device half* p = B + nb * boff + (nn * 16) * ldb + kk * 16;
          if (SAFE)
            lkt_fload_safe(bf[kk][nn], p, ldb, sc, short(sn - nn * 16), short(sk - kk * 16));
          else
            lkt_fload(bf[kk][nn], p, ldb, sc);
        } else {
          const device half* p = B + nb * boff + (kk * 16) * ldb + nn * 16;
          if (SAFE)
            lkt_fload_safe(bf[kk][nn], p, ldb, sc, short(sk - kk * 16), short(sn - nn * 16));
          else
            lkt_fload(bf[kk][nn], p, ldb, sc);
        }
      }
    }
    LKT_UNROLL
    for (short mm = 0; mm < TM; mm++) {
      LKT_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        LKT_UNROLL
        for (short kk = 0; kk < 2; kk++)
          lkt_fmma<TB, RELAXED>(C[nb][mm][nn], C[nb][mm][nn + 1], af[mm][kk], bf[kk][nn], bf[kk][nn + 1]);
      }
    }
  }
}

template <short TM, short TN, short NB, int BK, bool TB, bool ALIGNED, bool RELAXED>
inline void lkt_fkloop(thread lkt_ffrag (&C)[NB][TM][TN], const device half* A, const device half* B, int lda, int ldb,
                       int boff, int K, short2 sc, short sm, short sn) {
  const bool has_output = sm > 0 && sn > 0;
  const int a_step = LKT_SK, b_step = TB ? LKT_SK : LKT_SK * ldb;
  constexpr int per_barrier = BK / LKT_SK;
  const int full = K / LKT_SK, barrier_steps = K / BK * per_barrier;
  LKT_NO_UNROLL
  for (int s = 0; s < full; s++) {
    if (s < barrier_steps && s % per_barrier == 0) threadgroup_barrier(mem_flags::mem_none);
    if (!ALIGNED && !has_output) continue;
    lkt_fstep<TM, TN, NB, TB, !ALIGNED, RELAXED>(C, A, B, lda, ldb, boff, sc, sm, sn, LKT_SK);
    A += a_step;
    B += b_step;
  }
  if (!ALIGNED && !has_output) return;
  const short tail = short(K - full * LKT_SK);
  if (tail > 0) lkt_fstep<TM, TN, NB, TB, true, RELAXED>(C, A, B, lda, ldb, boff, sc, sm, sn, tail);
}

template <int BM, int BN, int WM, int WN, int BK, bool TB, bool RELAXED>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_f16t_gemm_nax(
    const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], const device half* bias [[buffer(2)]],
    device half* D [[buffer(3)]], const device half* R [[buffer(4)]], const constant LkF16tGemmParams* p [[buffer(5)]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = BM / WM, SN = BN / WN, TM = SM / 16, TN = SN / 16;
  const int2 t = lkt_tile(tid, p->swizzle_log);
  if (p->tiles_n <= t.x || p->tiles_m <= t.y) return;
  const int row0 = t.y * BM + SM * (sg / WN), col0 = t.x * BN + SN * (sg % WN);
  const short sm = short(clamp(p->M - row0, -1, int(SM))), sn = short(clamp(p->N - col0, -1, int(SN)));
  const short2 sc = lkt_coord(lane);
  const device half* Ab = A + size_t(row0) * p->lda;
  const device half* Bb = B + (TB ? size_t(col0) * p->ldb : size_t(col0));
  lkt_ffrag C[1][TM][TN];
  LKT_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    LKT_UNROLL
    for (short nn = 0; nn < TN; nn++) C[0][mm][nn] = 0.0f;
  }
  if (sm == SM && sn == SN)
    lkt_fkloop<TM, TN, 1, BK, TB, true, RELAXED>(C, Ab, Bb, p->lda, p->ldb, 0, p->K, sc, sm, sn);
  else
    lkt_fkloop<TM, TN, 1, BK, TB, false, RELAXED>(C, Ab, Bb, p->lda, p->ldb, 0, p->K, sc, sm, sn);
  if (sm <= 0 || sn <= 0) return;
  LKT_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    LKT_UNROLL
    for (short nn = 0; nn < TN; nn++) {
      LKT_UNROLL
      for (short i = 0; i < 8; i++) {
        const short r = mm * 16 + sc.y + (i >> 2) * 8, c = nn * 16 + sc.x + (i & 3);
        if (r >= sm || c >= sn) continue;
        const int row = row0 + r, col = col0 + c;
        float v = C[0][mm][nn][i];
        if (p->has_bias) v = v + float(bias[col]);
        if (p->act == 1) v = lkt_gelu(v);
        if (p->has_res) v = v + float(R[size_t(row) * p->ldr + col]);
        if (p->has_clamp) v = clamp(v, -p->lim, p->lim);
        D[size_t(row) * p->ldd + col] = half(v);
      }
    }
  }
}

#define LKT_GEMM_NAX(name, BM, BN, WM, WN, BK, TB, RELAXED)                                                        \
  template [[host_name(name)]] [[kernel]] void lk_f16t_gemm_nax<BM, BN, WM, WN, BK, TB, RELAXED>(                  \
      const device half*, const device half*, const device half*, device half*, const device half*,                \
      const constant LkF16tGemmParams*, ushort, ushort, uint3);

LKT_GEMM_NAX("lk_f16t_gemm_nax_128x128", 128, 128, 4, 4, 512, false, false)
LKT_GEMM_NAX("lk_f16t_gemm_nax_64x128", 64, 128, 2, 4, 512, false, false)
LKT_GEMM_NAX("lk_f16t_gemm_nax_64x64", 64, 64, 2, 2, 512, false, false)
LKT_GEMM_NAX("lk_f16t_gemm_nax_128x128_nt", 128, 128, 4, 4, 512, true, false)
LKT_GEMM_NAX("lk_f16t_gemm_nax_64x128_nt", 64, 128, 2, 4, 512, true, false)
LKT_GEMM_NAX("lk_f16t_gemm_nax_64x64_nt", 64, 64, 2, 2, 512, true, false)
LKT_GEMM_NAX("lk_f16t_gemm_naxr_128x128", 128, 128, 4, 4, 512, false, true)  // relaxed_precision: measurement only

template <int BM, int BN, int WM, int WN, int BK>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_f16t_gemm_swiglu_nax(
    const device half* A [[buffer(0)]], const device half* B [[buffer(1)]], device half* D [[buffer(2)]],
    const constant LkF16tGemmParams* p [[buffer(3)]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr short SM = BM / WM, SN = BN / WN, TM = SM / 16, TN = SN / 16;
  const int2 t = lkt_tile(tid, p->swizzle_log);
  if (p->tiles_n <= t.x || p->tiles_m <= t.y) return;
  const int row0 = t.y * BM + SM * (sg / WN), col0 = t.x * BN + SN * (sg % WN);
  const short sm = short(clamp(p->M - row0, -1, int(SM))), sn = short(clamp(p->N - col0, -1, int(SN)));
  const short2 sc = lkt_coord(lane);
  const device half* Ab = A + size_t(row0) * p->lda;
  const device half* Bb = B + size_t(col0);
  lkt_ffrag C[2][TM][TN];
  LKT_UNROLL
  for (short nb = 0; nb < 2; nb++) {
    LKT_UNROLL
    for (short mm = 0; mm < TM; mm++) {
      LKT_UNROLL
      for (short nn = 0; nn < TN; nn++) C[nb][mm][nn] = 0.0f;
    }
  }
  if (sm == SM && sn == SN)
    lkt_fkloop<TM, TN, 2, BK, false, true, false>(C, Ab, Bb, p->lda, p->ldb, p->N, p->K, sc, sm, sn);
  else
    lkt_fkloop<TM, TN, 2, BK, false, false, false>(C, Ab, Bb, p->lda, p->ldb, p->N, p->K, sc, sm, sn);
  if (sm <= 0 || sn <= 0) return;
  LKT_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    LKT_UNROLL
    for (short nn = 0; nn < TN; nn++) {
      LKT_UNROLL
      for (short i = 0; i < 8; i++) {
        const short r = mm * 16 + sc.y + (i >> 2) * 8, c = nn * 16 + sc.x + (i & 3);
        if (r >= sm || c >= sn) continue;
        D[size_t(row0 + r) * p->ldd + col0 + c] = half(lkt_silu_mul(C[0][mm][nn][i], C[1][mm][nn][i]));
      }
    }
  }
}

#define LKT_GEMM_SWIGLU_NAX(name, BM, BN, WM, WN, BK)                                                              \
  template [[host_name(name)]] [[kernel]] void lk_f16t_gemm_swiglu_nax<BM, BN, WM, WN, BK>(                        \
      const device half*, const device half*, device half*, const constant LkF16tGemmParams*, ushort, ushort, uint3);

LKT_GEMM_SWIGLU_NAX("lk_f16t_gemm_swiglu_nax_128x128", 128, 128, 4, 4, 512)
LKT_GEMM_SWIGLU_NAX("lk_f16t_gemm_swiglu_nax_64x128", 64, 128, 2, 4, 512)
LKT_GEMM_SWIGLU_NAX("lk_f16t_gemm_swiglu_nax_64x64", 64, 64, 2, 2, 512)
