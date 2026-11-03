// M2 conv proof (GPU): implicit-GEMM conv1d, f32, channel-first [C, L].
// out[co, t] = sum_{ci, k} W[co, ci*K + k] * x[ci, t + k*d - pad]   (GEMM M=Co, N=Lo, K=Ci*K)
// The im2col happens inside the B-tile load: no column buffer in memory.
// Tile 64x64x16, 4 simdgroups (2x2), each 32x32 = 4x4 simdgroup_float8x8 accumulators.
#include <metal_stdlib>
using namespace metal;

struct ConvParams {
  uint ci, co, l, lo, k, d, pad;
};

constant constexpr uint BM = 64, BN = 64;

template <uint BK, uint PAD>
[[kernel, max_total_threads_per_threadgroup(128)]] void conv1d_igemm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                         device float* out [[buffer(2)]], constant ConvParams& p [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint tid [[thread_index_in_threadgroup]]) {
  // Padded strides (PAD floats) avoid threadgroup-memory bank conflicts in simdgroup_load.
  constexpr uint SA = BK + PAD, SB = BN + PAD;
  threadgroup float As[BM * SA];
  threadgroup float Bs[BK * SB];
  threadgroup float edge[4 * 64];  // per-simdgroup 8x8 staging for partial edge blocks only
  const uint row0 = tg.y * BM, col0 = tg.x * BN;
  const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
  const uint kTotal = p.ci * p.k;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

  for (uint k0 = 0; k0 < kTotal; k0 += BK) {
    // A tile: W[row0 .. +64, k0 .. +32]. Thread -> row tid/2, 16 consecutive k (contiguous reads).
    {
      constexpr uint per = BK / 2;  // 128 threads: 2 per row of 64 rows
      uint r = tid / 2, c0 = (tid % 2) * per, gr = row0 + r;
      device const float* src = w + gr * kTotal + k0 + c0;
      for (uint c = 0; c < per; ++c) {
        uint gk = k0 + c0 + c;
        As[r * SA + c0 + c] = (gr < p.co && gk < kTotal) ? src[c] : 0.0f;
      }
    }
    // B tile: im2col[k0 .. +32, col0 .. +64] straight from x. Thread -> 2 rows (one (ci, tap) pair
    // each: the divide happens once per row), 8 consecutive output columns (coalesced).
    for (uint hb = 0; hb < (BK + 15) / 16; ++hb) {
      uint r = tid / 8 + hb * 16, c0 = (tid % 8) * 8, gk = k0 + r;
      if (r >= BK) break;
      bool rowOk = gk < kTotal;
      uint ci = rowOk ? gk / p.k : 0, tap = rowOk ? gk % p.k : 0;
      int t0 = int(col0 + c0) + int(tap * p.d) - int(p.pad);
      device const float* src = x + ci * p.l;
      for (uint c = 0; c < 8; ++c) {
        int t = t0 + int(c);
        bool ok = rowOk && (col0 + c0 + c) < p.lo && t >= 0 && t < int(p.l);
        Bs[r * SB + c0 + c] = ok ? src[t] : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[4], bm[4];
      for (uint i = 0; i < 4; ++i) simdgroup_load(am[i], As + (sr + i * 8) * SA + kk, SA);
      for (uint j = 0; j < 4; ++j) simdgroup_load(bm[j], Bs + kk * SB + sc + j * 8, SB);
      for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Store each 8x8 accumulator straight to memory when fully in range (the
  // common case); partial edge blocks go through a small per-simdgroup scratch.
  const uint lane = tid % 32;
  threadgroup float* mine = edge + sg * 64;
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) {
      uint r0 = row0 + sr + i * 8, c0 = col0 + sc + j * 8;
      if (r0 + 8 <= p.co && c0 + 8 <= p.lo) {
        simdgroup_store(acc[i][j], out + r0 * p.lo + c0, p.lo);
      } else if (r0 < p.co && c0 < p.lo) {
        simdgroup_store(acc[i][j], mine, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
          uint rr = e / 8, cc = e % 8;
          if (r0 + rr < p.co && c0 + cc < p.lo) out[(r0 + rr) * p.lo + c0 + cc] = mine[e];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
}

template [[host_name("conv1d_igemm_bk8_p0")]] [[kernel]] void conv1d_igemm<8, 0>(device const float*,
    device const float*, device float*, constant ConvParams&, uint2, uint, uint);
template [[host_name("conv1d_igemm_bk8_p4")]] [[kernel]] void conv1d_igemm<8, 4>(device const float*,
    device const float*, device float*, constant ConvParams&, uint2, uint, uint);
template [[host_name("conv1d_igemm_bk16_p0")]] [[kernel]] void conv1d_igemm<16, 0>(device const float*,
    device const float*, device float*, constant ConvParams&, uint2, uint, uint);
template [[host_name("conv1d_igemm_bk16_p4")]] [[kernel]] void conv1d_igemm<16, 4>(device const float*,
    device const float*, device float*, constant ConvParams&, uint2, uint, uint);

// ---------------------------------------------------------------------------
// Direct variant: no threadgroup memory, no barriers. Fragments are loaded
// straight from device memory by simdgroup_load:
//   K is ordered (tap, ci): for a fixed tap, 8 consecutive input channels are
//   8 rows of xp at stride LP, so a B fragment is one strided simdgroup_load.
//   Weights are pre-reordered to wr[Co, K, Ci] (once, at load).
//   xp is zero-padded in memory ([Ci, LP], LP >= LOP + (K-1)*d), so every read
//   is in bounds; out is [Co, LOP] with LOP a multiple of the tile width.
// Requirements: Ci % 8 == 0, Co % (2*32) == 0, LOP % 64 == 0.
struct DirectParams {
  uint ci, co, lp, lop, k, d;
};

[[kernel, max_total_threads_per_threadgroup(128)]] void conv1d_direct_f32(device const float* xp [[buffer(0)]], device const float* wr [[buffer(1)]],
                              device float* out [[buffer(2)]], constant DirectParams& p [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint row0 = tg.y * 64 + (sg / 2) * 32, col0 = tg.x * 64 + (sg % 2) * 32;
  const uint kci = p.k * p.ci;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  for (uint tap = 0; tap < p.k; ++tap) {
    device const float* xb = xp + col0 + tap * p.d;
    device const float* wb = wr + row0 * kci + tap * p.ci;
    for (uint c = 0; c < p.ci; c += 8) {
      simdgroup_float8x8 am[4], bm[4];
      for (uint i = 0; i < 4; ++i) simdgroup_load(am[i], wb + i * 8 * kci + c, kci);
      for (uint j = 0; j < 4; ++j) simdgroup_load(bm[j], xb + c * p.lp + j * 8, p.lp);
      for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
    }
  }
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], out + (row0 + i * 8) * p.lop + col0 + j * 8, p.lop);
}

// Ceiling probe: MMAs on register fragments only (no memory traffic in the loop).
kernel void mma_peak_f32(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                         uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  simdgroup_float8x8 a[4], b[4], acc[4][4];
  for (uint i = 0; i < 4; ++i) {
    a[i] = simdgroup_float8x8(1.0f + i * 1e-3f);
    b[i] = simdgroup_float8x8(1.0f - i * 1e-3f);
    for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  }
  for (uint it = 0; it < iters; ++it)
    for (uint i = 0; i < 4; ++i)
      for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  threadgroup float scratch[4 * 64];
  float s = 0;
  for (uint i = 0; i < 4; ++i)
    for (uint j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], scratch + sg * 64, 8);
      s += scratch[sg * 64];
    }
  if (s == 12345.f) out[gid] = s;  // keep the work alive
}
