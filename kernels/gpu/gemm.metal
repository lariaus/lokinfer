// Baseline GPU GEMM: out[M,N] = epi(a[M,K] @ b[K,N]), row-major f32.
// simdgroup_matrix 8x8 tiles: each threadgroup (4 simdgroups) computes a
// 32x32 output tile, each simdgroup a 16x16 block (2x2 8x8 accumulators).
// Bounds are handled by staging through threadgroup memory with zero fill.
// research-kernels.md has the steel-style tuning plan.
//
// Buffer order follows the kernel ABI (operands, outs, then params):
//   gemm_f32:            a, b, out, params
//   gemm_bias[_relu]_f32: a, b, bias, out, params
//   gemm_bias[_relu]_smallm_f32, gemm_bias[_relu]_sk_f32_*: a, b, bias, out, params, split-K scratch
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

struct GemmParams {
  uint m, n, k;
};

constant constexpr uint BM = 32, BN = 32, BK = 16;

// Epilogues: 0 none, 1 +bias[N], 2 +bias then ReLU.
template <int EPI>
inline float epilogue(float v, device const float* bias, uint col) {
  if (EPI >= 1) v += bias[col];
  if (EPI == 2) v = max(v, 0.0f);
  return v;
}

template <int EPI>
inline void gemm_body(device const float* a, device const float* b, device const float* bias, device float* out,
                      constant GemmParams& p, threadgroup float* As, threadgroup float* Bs, threadgroup float* Cs,
                      uint2 tg, uint sg, uint tid) {
  const uint row0 = tg.y * BM, col0 = tg.x * BN;
  const uint sr = (sg / 2) * 16, sc = (sg % 2) * 16;  // simdgroup sub-tile
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

  for (uint k0 = 0; k0 < p.k; k0 += BK) {
    // 128 threads load 32x16 of A and 16x32 of B (4 elements each).
    for (uint e = tid; e < BM * BK; e += 128) {
      uint r = e / BK, c = e % BK;
      uint gr = row0 + r, gc = k0 + c;
      As[e] = (gr < p.m && gc < p.k) ? a[gr * p.k + gc] : 0.0f;
    }
    for (uint e = tid; e < BK * BN; e += 128) {
      uint r = e / BN, c = e % BN;
      uint gr = k0 + r, gc = col0 + c;
      Bs[e] = (gr < p.k && gc < p.n) ? b[gr * p.n + gc] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[2], bm[2];
      for (uint i = 0; i < 2; ++i) simdgroup_load(am[i], As + (sr + i * 8) * BK + kk, BK);
      for (uint j = 0; j < 2; ++j) simdgroup_load(bm[j], Bs + kk * BN + sc + j * 8, BN);
      for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * BN + sc + j * 8, BN);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < BM * BN; e += 128) {
    uint r = e / BN, c = e % BN;
    uint gr = row0 + r, gc = col0 + c;
    if (gr < p.m && gc < p.n) out[gr * p.n + gc] = epilogue<EPI>(Cs[e], bias, gc);
  }
}

kernel void gemm_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                     device float* out [[buffer(2)]], constant GemmParams& p [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float As[BM * BK], Bs[BK * BN], Cs[BM * BN];
  gemm_body<0>(a, b, a, out, p, As, Bs, Cs, tg, sg, tid);
}

kernel void gemm_bias_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                          device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                          constant GemmParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float As[BM * BK], Bs[BK * BN], Cs[BM * BN];
  gemm_body<1>(a, b, bias, out, p, As, Bs, Cs, tg, sg, tid);
}

kernel void gemm_bias_relu_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                               device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                               constant GemmParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float As[BM * BK], Bs[BK * BN], Cs[BM * BN];
  gemm_body<2>(a, b, bias, out, p, As, Bs, Cs, tg, sg, tid);
}

// Small-M GEMM (M <= 8 per row block, e.g. batch-1 MLP / classifier heads): the 32x32 tile kernel above gives such a
// GEMM only ceil(N/32) threadgroups that each walk all of K, leaving the GPU idle (MNIST 784->128 at M = 1: 192 us).
// Here a threadgroup owns 32 columns and splits K over 16 slices (512 threads); each thread accumulates every row for
// one column over its K slice (B reads coalesced across the simdgroup, A reads broadcast), then the slices are summed
// through threadgroup memory; K is also split over grid z (see GemmSkParams). Buffers: a, b, bias, out, params,
// scratch.
constant constexpr uint SM_COLS = 32, SM_SLICES = 16, SM_MAXM = 8;

// Split-K completion shared by the split kernels below: every threadgroup of a tile has stored its partial; the
// last one to arrive (device-scope counter per tile, reset by it, as in lk_attention_decode_splitf) returns true and
// then sees every partial. All threads of the threadgroup must call it.
inline bool sk_last_arrival(device atomic_uint* cnt, uint splits, uint tid, threadgroup uint& flag) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    flag = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == splits - 1 ? 1u : 0u;
    if (flag) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool last = flag != 0;
  if (last) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return last;
}

// Split-K scratch (one per op instance, zero-initialized): `cnt_cap` uint counters, then the partials [splits, M, N].
struct GemmSkParams {
  uint m, n, k;
  uint kc;       // K per split (a multiple of 16; the last split takes the rest)
  uint splits;   // grid z; 1 = no scratch use, the epilogue is applied directly
  uint tiles_n;  // grid x (counter index = y * tiles_n + x)
  uint cnt_cap;
};

// Rows come in blocks of SM_MAXM: threadgroup (x, y, z) owns columns [32x, 32x + 32) of rows [8y, 8y + 8) over the
// K range of split z (iPhone MNIST 784->128 at M = 1: 4 threadgroups for the whole GPU without the split).
template <int EPI>
inline void gemm_smallm_body(device const float* a, device const float* b, device const float* bias,
                             device float* out, device float* part, constant GemmSkParams& p, threadgroup float* red,
                             threadgroup uint& flag, uint3 tgp, uint2 tid) {
  const uint col = tgp.x * SM_COLS + tid.x, slice = tid.y, row0 = tgp.y * SM_MAXM;
  const uint mrows = min(SM_MAXM, p.m - row0);
  const uint k0 = tgp.z * p.kc, k1 = min(p.k, k0 + p.kc);
  a += size_t(row0) * p.k;
  float acc[SM_MAXM];
  for (uint r = 0; r < SM_MAXM; ++r) acc[r] = 0.0f;
  if (col < p.n)
    for (uint kk = k0 + slice; kk < k1; kk += SM_SLICES) {
      const float bv = b[kk * p.n + col];
      for (uint r = 0; r < SM_MAXM; ++r)
        if (r < mrows) acc[r] = fma(a[r * p.k + kk], bv, acc[r]);
    }
  for (uint r = 0; r < SM_MAXM; ++r) red[(r * SM_SLICES + slice) * SM_COLS + tid.x] = acc[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Thread (x, y) finishes row y (y < M): sum the 16 slices of column x.
  const bool mine = slice < mrows && col < p.n;
  float s = 0.0f;
  if (mine)
    for (uint j = 0; j < SM_SLICES; ++j) s += red[(slice * SM_SLICES + j) * SM_COLS + tid.x];
  const size_t off = size_t(row0 + slice) * p.n + col;
  if (p.splits == 1) {
    if (mine) out[off] = epilogue<EPI>(s, bias, col);
    return;
  }
  device float* parts = part + p.cnt_cap;
  const size_t mn = size_t(p.m) * p.n;
  if (mine) parts[tgp.z * mn + off] = s;
  device atomic_uint* cnt = reinterpret_cast<device atomic_uint*>(part) + tgp.y * p.tiles_n + tgp.x;
  if (!sk_last_arrival(cnt, p.splits, tid.y * SM_COLS + tid.x, flag) || !mine) return;
  float t = 0.0f;
  for (uint z = 0; z < p.splits; ++z) t += parts[z * mn + off];
  out[off] = epilogue<EPI>(t, bias, col);
}

kernel void gemm_bias_smallm_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                                 device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                                 constant GemmSkParams& p [[buffer(4)]], device float* part [[buffer(5)]],
                                 uint3 tg [[threadgroup_position_in_grid]],
                                 uint3 tid [[thread_position_in_threadgroup]]) {
  threadgroup float red[SM_MAXM * SM_SLICES * SM_COLS];
  threadgroup uint flag;
  gemm_smallm_body<1>(a, b, bias, out, part, p, red, flag, tg, tid.xy);
}

kernel void gemm_bias_relu_smallm_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                                      device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                                      constant GemmSkParams& p [[buffer(4)]], device float* part [[buffer(5)]],
                                      uint3 tg [[threadgroup_position_in_grid]],
                                      uint3 tid [[thread_position_in_threadgroup]]) {
  threadgroup float red[SM_MAXM * SM_SLICES * SM_COLS];
  threadgroup uint flag;
  gemm_smallm_body<2>(a, b, bias, out, part, p, red, flag, tg, tid.xy);
}

// Steel-tile GEMM + bias [+ ReLU] with optional split-K in ONE dispatch (MLX steel BlockLoader / BlockMMA,
// third_party/mlx_steel): threadgroup (x, y, z) computes tile (y, x) over the K range of split z. splits = 1: the
// epilogue runs on the accumulators (a plain steel GEMM, any tile shape below). splits > 1: partials go to the
// scratch, the last split of each tile sums them and applies the epilogue (medium M: MNIST B=64 784->128 is 8 tiles
// of 32x32 without the split).
struct SkAdd {
  METAL_FUNC float apply(float a, float b) const { return a + b; }
};
struct SkRelu {
  METAL_FUNC float apply(float v) const { return max(v, 0.0f); }
};

template <int TBM, int TBN, int TBK, int WM, int WN, int EPI>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm_bias_sk(
    const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], const device float* bias [[buffer(2)]],
    device float* D [[buffer(3)]], constant GemmSkParams& p [[buffer(4)]], device float* part [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]], uint li [[thread_index_in_threadgroup]]) {
  using gk = mlx::steel::GEMMKernel<float, float, TBM, TBN, TBK, WM, WN, false, false, false, false, float>;
  threadgroup float As[gk::tgp_mem_size_a];
  threadgroup float Bs[gk::tgp_mem_size_b];
  threadgroup uint flag;
  const int M = p.m, N = p.n, K = p.k;
  const int c_row = int(tid.y) * TBM, c_col = int(tid.x) * TBN, k0 = int(tid.z * p.kc);
  const int klen = min(int(p.kc), K - k0);
  thread typename gk::loader_a_t loader_a(A + size_t(c_row) * K + k0, K, As, sg, lane);
  thread typename gk::loader_b_t loader_b(B + size_t(k0) * N + c_col, N, Bs, sg, lane);
  thread typename gk::mma_t mma_op(sg, lane);
  const short tgp_bm = min(TBM, M - c_row), tgp_bn = min(TBN, N - c_col);
  const int iters = klen / TBK;
  const short lbk = klen - iters * TBK;
  if (tgp_bm == TBM && tgp_bn == TBN)
    gk::template gemm_loop<true, true, false>(As, Bs, iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
  else if (tgp_bn == TBN)
    gk::template gemm_loop<false, true, false>(As, Bs, iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
  else if (tgp_bm == TBM)
    gk::template gemm_loop<true, false, false>(As, Bs, iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
  else
    gk::template gemm_loop<false, false, false>(As, Bs, iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
  threadgroup_barrier(mem_flags::mem_none);
  const short2 dims(tgp_bn, tgp_bm);
  const size_t off0 = size_t(c_row) * N + c_col;
  if (p.splits == 1) {
    if (EPI >= 1) mma_op.apply_epilogue_safe(bias + c_col, 0, 1, dims, SkAdd());
    if (EPI == 2) mma_op.apply_epilogue(SkRelu());
    mma_op.store_result_safe(D + off0, N, dims);
    return;
  }
  device float* parts = part + p.cnt_cap;
  const size_t mn = size_t(M) * N;
  mma_op.store_result_safe(parts + tid.z * mn + off0, N, dims);
  device atomic_uint* cnt = reinterpret_cast<device atomic_uint*>(part) + tid.y * p.tiles_n + tid.x;
  if (!sk_last_arrival(cnt, p.splits, li, flag)) return;
  for (int e = int(li); e < TBM * TBN; e += WM * WN * 32) {
    const int r = e / TBN, c = e % TBN;
    if (r >= tgp_bm || c >= tgp_bn) continue;
    const size_t off = off0 + size_t(r) * N + c;
    float t = 0.0f;
    for (uint z = 0; z < p.splits; ++z) t += parts[z * mn + off];
    D[off] = epilogue<EPI>(t, bias, c_col + c);
  }
}

#define LK_GEMM_SK(BM_, BN_, BK_, WM_, WN_)                                                                         \
  template [[host_name("gemm_bias_sk_f32_" #BM_ "x" #BN_ "x" #BK_)]] [[kernel]] void                              \
  gemm_bias_sk<BM_, BN_, BK_, WM_, WN_, 1>(const device float*, const device float*, const device float*,          \
                                           device float*, constant GemmSkParams&, device float*, uint, uint, uint3, \
                                           uint);                                                                  \
  template [[host_name("gemm_bias_relu_sk_f32_" #BM_ "x" #BN_ "x" #BK_)]] [[kernel]] void                         \
  gemm_bias_sk<BM_, BN_, BK_, WM_, WN_, 2>(const device float*, const device float*, const device float*,          \
                                           device float*, constant GemmSkParams&, device float*, uint, uint, uint3, \
                                           uint);

LK_GEMM_SK(32, 32, 16, 2, 2)
LK_GEMM_SK(64, 64, 16, 2, 2)
LK_GEMM_SK(64, 32, 16, 2, 2)
LK_GEMM_SK(64, 32, 32, 2, 2)
LK_GEMM_SK(32, 64, 16, 2, 2)
LK_GEMM_SK(32, 32, 32, 2, 2)
LK_GEMM_SK(16, 32, 16, 2, 2)
LK_GEMM_SK(16, 64, 16, 2, 2)

// 32x32 tile, BK = 32, float4 global loads when the whole tile is in range and rows are 16-byte multiples (scalar
// zero-filled loads at the edges). Same math as gemm_body (2x2 8x8 accumulators per simdgroup); half the barriers.
template <int EPI>
inline void gemm_body_k32(device const float* a, device const float* b, device const float* bias, device float* out,
                          constant GemmParams& p, threadgroup float* As, threadgroup float* Bs, threadgroup float* Cs,
                          uint2 tg, uint sg, uint tid) {
  constexpr uint TK = 32;
  const uint row0 = tg.y * BM, col0 = tg.x * BN;
  const uint sr = (sg / 2) * 16, sc = (sg % 2) * 16;
  const bool vec_ok = (p.k % 4 == 0) && (p.n % 4 == 0) && row0 + BM <= p.m && col0 + BN <= p.n;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint k0 = 0; k0 < p.k; k0 += TK) {
    if (vec_ok && k0 + TK <= p.k) {
      for (uint e = tid; e < BM * TK / 4; e += 128) {  // A: 32 rows x 8 float4
        const uint r = e / (TK / 4), c = (e % (TK / 4)) * 4;
        *reinterpret_cast<threadgroup float4*>(As + r * TK + c) =
            *reinterpret_cast<device const float4*>(a + size_t(row0 + r) * p.k + k0 + c);
      }
      for (uint e = tid; e < TK * BN / 4; e += 128) {  // B: 32 rows x 8 float4
        const uint r = e / (BN / 4), c = (e % (BN / 4)) * 4;
        *reinterpret_cast<threadgroup float4*>(Bs + r * BN + c) =
            *reinterpret_cast<device const float4*>(b + size_t(k0 + r) * p.n + col0 + c);
      }
    } else {
      for (uint e = tid; e < BM * TK; e += 128) {
        const uint r = e / TK, c = e % TK, gr = row0 + r, gc = k0 + c;
        As[e] = (gr < p.m && gc < p.k) ? a[size_t(gr) * p.k + gc] : 0.0f;
      }
      for (uint e = tid; e < TK * BN; e += 128) {
        const uint r = e / BN, c = e % BN, gr = k0 + r, gc = col0 + c;
        Bs[e] = (gr < p.k && gc < p.n) ? b[size_t(gr) * p.n + gc] : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < TK; kk += 8) {
      simdgroup_float8x8 am[2], bm[2];
      for (uint i = 0; i < 2; ++i) simdgroup_load(am[i], As + (sr + i * 8) * TK + kk, TK);
      for (uint j = 0; j < 2; ++j) simdgroup_load(bm[j], Bs + kk * BN + sc + j * 8, BN);
      for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * BN + sc + j * 8, BN);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < BM * BN; e += 128) {
    const uint r = e / BN, c = e % BN, gr = row0 + r, gc = col0 + c;
    if (gr < p.m && gc < p.n) out[size_t(gr) * p.n + gc] = epilogue<EPI>(Cs[e], bias, gc);
  }
}

kernel void gemm_bias_k32_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                              device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                              constant GemmParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float As[BM * 32], Bs[32 * BN], Cs[BM * BN];
  gemm_body_k32<1>(a, b, bias, out, p, As, Bs, Cs, tg, sg, tid);
}

kernel void gemm_bias_relu_k32_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                                   device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                                   constant GemmParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float As[BM * 32], Bs[32 * BN], Cs[BM * BN];
  gemm_body_k32<2>(a, b, bias, out, p, As, Bs, Cs, tg, sg, tid);
}
