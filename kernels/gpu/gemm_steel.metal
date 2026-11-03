// lokinfer's GPU GEMM on MLX's steel building blocks (BlockLoader + BlockMMA, MIT, third_party/mlx_steel):
//   D[M, N] = act(A[M, K] @ op(B) + bias [+ residual])     op(B) = B [K, N] or B^T for B [N, K] (TB)
// The tile loop mirrors steel's GEMMKernel::run (aligned fast path per tile, safe loads at the edges) without its
// function constants; the epilogue (bias, gelu_tanh, residual) is applied to the accumulators before the store.
// M2: steel's GEMM measured 8.7 TF through lokinfer's runtime vs ~1.5 TF for the old simple kernel.
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;
using namespace mlx::steel;

struct LkGemmParams {
  int M, N, K, lda, ldb, ldd, tiles_n, tiles_m, swizzle_log, gemm_k_iterations;
  uint has_bias, act, has_res;  // act: 0 none, 1 gelu_tanh, 2 relu
};

struct LkAdd {
  METAL_FUNC float apply(float a, float b) const { return a + b; }
};

struct LkRelu {
  METAL_FUNC float apply(float v) const { return max(v, 0.0f); }
};

struct LkGeluTanh {
  METAL_FUNC float apply(float v) const {
    return 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
  }
};

template <int BM, int BN, int BK, int WM, int WN, bool TB>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_gemm(
    const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], const device float* bias [[buffer(2)]],
    device float* D [[buffer(3)]], const device float* R [[buffer(4)]], const constant LkGemmParams* p [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using gk = GEMMKernel<float, float, BM, BN, BK, WM, WN, false, TB, false, false, float>;
  threadgroup float As[gk::tgp_mem_size_a];
  threadgroup float Bs[gk::tgp_mem_size_b];

  const int tid_y = ((tid.y) << p->swizzle_log) + ((tid.x) & ((1 << p->swizzle_log) - 1));
  const int tid_x = (tid.x) >> p->swizzle_log;
  if (p->tiles_n <= tid_x || p->tiles_m <= tid_y) return;

  const int c_row = tid_y * BM, c_col = tid_x * BN;
  A += size_t(c_row) * p->lda;
  B += TB ? size_t(c_col) * p->ldb : size_t(c_col);
  D += size_t(c_row) * p->ldd + c_col;

  thread typename gk::loader_a_t loader_a(A, p->lda, As, simd_group_id, simd_lane_id);
  thread typename gk::loader_b_t loader_b(B, p->ldb, Bs, simd_group_id, simd_lane_id);
  thread typename gk::mma_t mma_op(simd_group_id, simd_lane_id);

  const short tgp_bm = min(BM, p->M - c_row), tgp_bn = min(BN, p->N - c_col);
  const short lbk = p->K - p->gemm_k_iterations * BK;
  if (tgp_bm == BM && tgp_bn == BN)
    gk::template gemm_loop<true, true, false>(As, Bs, p->gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm,
                                              tgp_bn, lbk);
  else if (tgp_bn == BN)
    gk::template gemm_loop<false, true, false>(As, Bs, p->gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm,
                                               tgp_bn, lbk);
  else if (tgp_bm == BM)
    gk::template gemm_loop<true, false, false>(As, Bs, p->gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm,
                                               tgp_bn, lbk);
  else
    gk::template gemm_loop<false, false, false>(As, Bs, p->gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm,
                                                tgp_bn, lbk);

  threadgroup_barrier(mem_flags::mem_none);
  const short2 dims(tgp_bn, tgp_bm);
  if (p->has_bias) mma_op.apply_epilogue_safe(bias + c_col, 0, 1, dims, LkAdd());
  if (p->act == 1) mma_op.apply_epilogue(LkGeluTanh());
  else if (p->act == 2) mma_op.apply_epilogue(LkRelu());
  if (p->has_res) mma_op.apply_epilogue_safe(R + size_t(c_row) * p->ldd + c_col, p->ldd, 1, dims, LkAdd());
  mma_op.store_result_safe(D, p->ldd, dims);
}

#define LK_GEMM(name, BM, BN, BK, WM, WN, TB)                                                                      \
  template [[host_name(name)]] [[kernel]] void lk_gemm<BM, BN, BK, WM, WN, TB>(                                    \
      const device float*, const device float*, const device float*, device float*, const device float*,          \
      const constant LkGemmParams*, uint, uint, uint3);

LK_GEMM("lk_gemm_f32_64x64", 64, 64, 16, 2, 2, false)
LK_GEMM("lk_gemm_f32_32x32", 32, 32, 16, 2, 2, false)
LK_GEMM("lk_gemm_nt_f32_64x64", 64, 64, 16, 2, 2, true)
LK_GEMM("lk_gemm_nt_f32_32x32", 32, 32, 16, 2, 2, true)

// Split-K partials (GPT-2's N = 768 GEMMs at small M leave most cores idle: 2 x 12 tiles of 64x64 at M = 128 on 30
// cores): grid z = split s computes A[:, s*kc:(s+1)*kc] @ B[s*kc:(s+1)*kc, :] into D[s] ([S, M, N], no epilogue);
// the consumer (lk_add_bias_ln2_sk) sums the S partials while it loads them. kc % BK == 0 (checked on the host).
struct LkSplitKParams {
  int M, N, K, lda, ldb, ldd, tiles_n, tiles_m, k_iters, kc;
};

template <int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void lk_gemm_splitk(
    const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], device float* D [[buffer(2)]],
    const constant LkSplitKParams* p [[buffer(3)]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]], uint3 tid [[threadgroup_position_in_grid]]) {
  using gk = GEMMKernel<float, float, BM, BN, BK, WM, WN, false, false, false, false, float>;
  threadgroup float As[gk::tgp_mem_size_a];
  threadgroup float Bs[gk::tgp_mem_size_b];
  const int tid_y = tid.y, tid_x = tid.x;
  if (p->tiles_n <= tid_x || p->tiles_m <= tid_y) return;
  const int c_row = tid_y * BM, c_col = tid_x * BN, k0 = int(tid.z) * p->kc;
  A += size_t(c_row) * p->lda + k0;
  B += size_t(k0) * p->ldb + c_col;
  D += size_t(tid.z) * size_t(p->M) * p->ldd + size_t(c_row) * p->ldd + c_col;

  thread typename gk::loader_a_t loader_a(A, p->lda, As, simd_group_id, simd_lane_id);
  thread typename gk::loader_b_t loader_b(B, p->ldb, Bs, simd_group_id, simd_lane_id);
  thread typename gk::mma_t mma_op(simd_group_id, simd_lane_id);
  const short tgp_bm = min(BM, p->M - c_row), tgp_bn = min(BN, p->N - c_col);
  if (tgp_bm == BM && tgp_bn == BN)
    gk::template gemm_loop<true, true, false>(As, Bs, p->k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, 0);
  else if (tgp_bn == BN)
    gk::template gemm_loop<false, true, false>(As, Bs, p->k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, 0);
  else if (tgp_bm == BM)
    gk::template gemm_loop<true, false, false>(As, Bs, p->k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, 0);
  else
    gk::template gemm_loop<false, false, false>(As, Bs, p->k_iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, 0);
  threadgroup_barrier(mem_flags::mem_none);
  mma_op.store_result_safe(D, p->ldd, short2(tgp_bn, tgp_bm));
}

#define LK_GEMM_SK(name, BM, BN, BK, WM, WN)                                                                       \
  template [[host_name(name)]] [[kernel]] void lk_gemm_splitk<BM, BN, BK, WM, WN>(                                 \
      const device float*, const device float*, device float*, const constant LkSplitKParams*, uint, uint, uint3);

LK_GEMM_SK("lk_gemm_splitk_f32_64x64", 64, 64, 16, 2, 2)
LK_GEMM_SK("lk_gemm_splitk_f32_32x32", 32, 32, 16, 2, 2)
