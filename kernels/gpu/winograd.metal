// Winograd / Toom-Cook 1-D convolution, channel-last, in two kernels (docs/impl-notes/winograd-conv.md):
//
//   lk_wino_tf_lc   input transform (optionally of snake(AdaIN-affine(x)), the activation every Kokoro generator
//                   conv applies to its input): V[rho, xi, r, ci] = sum_j BT[xi, j] h[rho + d (r M + j - pad), ci]
//   lk_wino_mm_lc   per point xi a GEMM over (group, ci) on V, folded into the output rows by AT in registers, + bias
//                   [+ residual] * scale:  y[o, co] = (b[co] + sum_k sum_ci w[co, ci, k] h[o + d k - pad, ci] [+ r]) s
//
// Math (per dilation residue class rho: the decimated sequences h[rho + d q] are an undilated conv, pad % d == 0).
// Inner Toom-Cook F(M, r) with A = M + r - 1 points: V = BT h_tile (A inputs at stride M), y_tile = AT ((G w) . V).
// K > r taps: "nested" groups of r = M taps at offsets g M (g < N): the transformed tile of group g is V[t + g],
// so all groups share the same transformed input and
//   Y[t, :, co] = sum_xi AT_{:, xi} M_xi[t, co],   M_xi[t, co] = sum_{g, ci} V_xi[t + g, ci] U[xi, g ci, co]
// with U = G w precomputed on the host ([A, N Ci, Co]).
//
// Why two kernels: fusing the input transform into the GEMM's loader (built first, A/B'd) either redoes it per point
// (A x the input loads: ~2 TF/s of real MMA work on the M2 Max) or keeps all A points' accumulators live (8 %
// occupancy). With V precomputed, the GEMM's A loader is one float4 load per element, like steel's.
//
// Register discipline (a first version spilled: 42 % "buffer store" utilization in the GPU counters): accumulators
// are plain float2 arrays (the fragment's thread elements, as in MLX steel's BaseMMAFrag) and every loop is unrolled.
#include <metal_stdlib>

using namespace metal;

#define LK_UNROLL _Pragma("clang loop unroll(full)")

struct WinoTfParams {
  uint l, c, d, pad, rmax, act;  // act: 0 none, 1 snake(affine), 2 leaky(affine)
  float slope;
};

// x [L, C], tr [BT (A x A) | AT (M x A)], affine [3, C] (act 1, 2), alpha [C] (act 1; bound to x when unused),
// V [d, A, rmax, C].
// Grid (C / 4, ceil(rmax / RT), d): each thread transforms RT consecutive tiles, sharing the overlapping inputs
// (RT M + A - M loads and activations instead of RT A).
template <int M, int A, int RT>
kernel void lk_wino_tf_lc(const device float* x [[buffer(0)]], const device float* tr [[buffer(1)]],
                          const device float* af [[buffer(2)]], const device float* alpha [[buffer(3)]],
                          device float* v [[buffer(4)]], constant WinoTfParams& p [[buffer(5)]],
                          uint3 gid [[thread_position_in_grid]]) {
  constexpr int NR = RT * M + A - M;
  const uint c4 = gid.x, r0 = gid.y * RT, rho = gid.z, d = p.d, c = p.c;
  if (c4 * 4 >= c || r0 >= p.rmax) return;
  const int lr = rho < p.l ? int((p.l - rho + d - 1) / d) : 0;
  const int q0 = int(r0) * M - int(p.pad);
  float4 mean = 0.0f, scale = 1.0f, beta = 0.0f, al = 1.0f, inv = 1.0f;
  if (p.act != 0) {
    mean = *reinterpret_cast<const device float4*>(af + c4 * 4);
    scale = *reinterpret_cast<const device float4*>(af + c + c4 * 4);
    beta = *reinterpret_cast<const device float4*>(af + 2 * c + c4 * 4);
  }
  if (p.act == 1) {
    al = *reinterpret_cast<const device float4*>(alpha + c4 * 4);
    inv = 1.0f / al;
  }
  float4 raw[NR];
  LK_UNROLL for (int j = 0; j < NR; ++j) {
    const int q = q0 + j;
    float4 h = 0.0f;
    if (q >= 0 && q < lr) {
      h = *reinterpret_cast<const device float4*>(x + size_t(rho + uint(q) * d) * c + c4 * 4);
      if (p.act == 1) {  // same math as lk_act_lc (kokoro_lc.metal)
        h = (h - mean) * scale + beta;
        const float4 s = precise::sin(al * h);
        h = h + inv * (s * s);
      } else if (p.act == 2) {
        h = (h - mean) * scale + beta;
        h = select(h * p.slope, h, h >= 0.0f);
      }
    }
    raw[j] = h;
  }
  LK_UNROLL for (int i = 0; i < RT; ++i) {
    const uint r = r0 + uint(i);
    if (r >= p.rmax) break;
    LK_UNROLL for (int xi = 0; xi < A; ++xi) {
      float4 o = 0.0f;
      LK_UNROLL for (int j = 0; j < A; ++j) o += tr[xi * A + j] * raw[i * M + j];
      *reinterpret_cast<device float4*>(v + ((size_t(rho) * A + xi) * p.rmax + r) * c + c4 * 4) = o;
    }
  }
}

template [[host_name("lk_wino_tf_lc_m4a6")]] kernel void lk_wino_tf_lc<4, 6, 1>(
    const device float*, const device float*, const device float*, const device float*, device float*,
    constant WinoTfParams&, uint3);
template [[host_name("lk_wino_tf_lc_m2a4")]] kernel void lk_wino_tf_lc<2, 4, 1>(
    const device float*, const device float*, const device float*, const device float*, device float*,
    constant WinoTfParams&, uint3);
template [[host_name("lk_wino_tf_lc_m4a7")]] kernel void lk_wino_tf_lc<4, 7, 1>(
    const device float*, const device float*, const device float*, const device float*, device float*,
    constant WinoTfParams&, uint3);

struct WinoMmParams {
  uint lo, ci, co, d, rmax, has_res;
  float scale;
  // 2-D (D2): tiles = n * th * tw of 2x2 outputs, output [n, ho, wo, co]; act 1 = ReLU after the residual
  uint th, tw, ho, wo, act;
  uint chunks;  // STATS: chunk rows per residue class (= the grid's y size)
  uint splits;  // D2: the A points are split over grid z; > 1: partial sums atomically added to a zeroed out
};

constant constexpr int kAPad = 4;  // threadgroup row padding (floats) against bank conflicts
constant constexpr int kBPad = 4;

// Buffers: 0 V [d, A, rmax, Ci], 1 U [A, N Ci, Co], 2 bias [Co], 3 res [Lo, Co] (bound to out when unused),
// 4 out [Lo, Co], 5 tr, 6 WinoMmParams. Grid (Co / BN, ceil(T_0 / BT), d), 128 threads (2 x 2 simdgroups).
// rmax >= ceil(T_0 / BT) BT + N - 1 (rows past a class's tiles are zero), so the A loader needs no bounds checks.
template <int M, int A, int N, int BT, int BN, int BK, bool D2 = false, bool STATS = false>
[[kernel, max_total_threads_per_threadgroup(128)]] void lk_wino_mm_lc(
    const device float* v [[buffer(0)]], const device float* u [[buffer(1)]], const device float* bias [[buffer(2)]],
    const device float* res [[buffer(3)]], device float* out [[buffer(4)]], const device float* tr [[buffer(5)]],
    constant WinoMmParams& p [[buffer(6)]], device float* st [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = BT + N - 1;  // transformed rows per point (tile shift by group)
  constexpr int FR = BT / 16;       // 8x8 fragments per simdgroup along tiles (2 simdgroup rows)
  constexpr int FC = BN / 16;       // ... along output channels (2 simdgroup columns)
  constexpr int AS = BK + kAPad;
  constexpr int BS = BN + kBPad;
  constexpr int ATASKS = ROWS * (BK / 4);
  constexpr int BTASKS = N * BK * (BN / 4);
  threadgroup float As[ROWS * AS];
  threadgroup float Bs[N * BK * BS];

  const uint rho = D2 ? 0 : tg.z, d = p.d, ci = p.ci, co = p.co;
  const int xi0 = D2 ? int(tg.z) * A / int(p.splits) : 0, xi1 = D2 ? (int(tg.z) + 1) * A / int(p.splits) : A;
  // 1-D: output rows in this class; 2-D: lo = number of tiles (x M so the same test works)
  const int lor = D2 ? int(p.lo) * M : rho < p.lo ? int((p.lo - rho + d - 1) / d) : 0;
  const int t0 = int(tg.y) * BT;
  if (t0 * M >= lor) return;
  const uint c0 = tg.x * BN;
  const int sr = int(sg) / 2, sc = int(sg) % 2;

  // Fragment element coordinates (MLX steel BaseMMAFrag::get_coord): elements (fm, fn) and (fm, fn + 1).
  const short qid = short(sl / 4);
  const short fm = (qid & 4) + short((sl / 2) % 4);
  const short fn = (qid & 2) * 2 + short(sl % 2) * 2;

  float2 y[M][FR][FC];
  LK_UNROLL for (int j = 0; j < M; ++j) LK_UNROLL for (int i = 0; i < FR; ++i) LK_UNROLL for (int k = 0; k < FC; ++k)
      y[j][i][k] = 0.0f;

  const int ksteps = int(ci) / BK;
  const uint nci = uint(N) * ci;
  for (int xi = xi0; xi < xi1; ++xi) {
    float2 acc[FR][FC];
    LK_UNROLL for (int i = 0; i < FR; ++i) LK_UNROLL for (int k = 0; k < FC; ++k) acc[i][k] = 0.0f;
    const device float* vx = v + ((size_t(rho) * A + xi) * p.rmax + t0) * ci;
    const device float* ux = u + size_t(xi) * nci * co + c0;
    for (int kb = 0; kb < ksteps; ++kb) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      LK_UNROLL for (int t = 0; t < (ATASKS + 127) / 128; ++t) {
        const int task = int(lid) + t * 128;
        if (ATASKS % 128 == 0 || task < ATASKS) {
          const int r = task / (BK / 4), c4 = task % (BK / 4);
          *reinterpret_cast<threadgroup float4*>(As + r * AS + c4 * 4) =
              *reinterpret_cast<const device float4*>(vx + size_t(r) * ci + kb * BK + c4 * 4);
        }
      }
      LK_UNROLL for (int t = 0; t < (BTASKS + 127) / 128; ++t) {
        const int task = int(lid) + t * 128;
        if (BTASKS % 128 == 0 || task < BTASKS) {
          const int c4 = task % (BN / 4), gk = task / (BN / 4), g = gk / BK, k = gk % BK;
          *reinterpret_cast<threadgroup float4*>(Bs + gk * BS + c4 * 4) =
              *reinterpret_cast<const device float4*>(ux + size_t(g * int(ci) + kb * BK + k) * co + c4 * 4);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      LK_UNROLL for (int g = 0; g < N; ++g) {
        LK_UNROLL for (int kk = 0; kk < BK; kk += 8) {
          float2 a[FR], b[FC];
          LK_UNROLL for (int i = 0; i < FR; ++i) {
            const threadgroup float* ap = As + (sr * (BT / 2) + i * 8 + g + fm) * AS + kk + fn;
            a[i] = float2(ap[0], ap[1]);
          }
          LK_UNROLL for (int k = 0; k < FC; ++k) {
            const threadgroup float* bp = Bs + (g * BK + kk + fm) * BS + sc * (BN / 2) + k * 8 + fn;
            b[k] = float2(bp[0], bp[1]);
          }
          LK_UNROLL for (int i = 0; i < FR; ++i) LK_UNROLL for (int k = 0; k < FC; ++k) {
            simdgroup_float8x8 ma, mb, mc;
            ma.thread_elements()[0] = a[i][0];
            ma.thread_elements()[1] = a[i][1];
            mb.thread_elements()[0] = b[k][0];
            mb.thread_elements()[1] = b[k][1];
            mc.thread_elements()[0] = acc[i][k][0];
            mc.thread_elements()[1] = acc[i][k][1];
            simdgroup_multiply_accumulate(mc, ma, mb, mc);
            acc[i][k] = float2(mc.thread_elements()[0], mc.thread_elements()[1]);
          }
        }
      }
    }
    // Output transform: Y_j += AT[j, xi] M_xi.
    LK_UNROLL for (int j = 0; j < M; ++j) {
      const float at = tr[A * A + j * A + xi];
      LK_UNROLL for (int i = 0; i < FR; ++i) LK_UNROLL for (int k = 0; k < FC; ++k) y[j][i][k] += at * acc[i][k];
    }
  }

  LK_UNROLL for (int i = 0; i < FR; ++i) {
    const int t = t0 + sr * (BT / 2) + i * 8 + fm;
    uint tn = 0, ty = 0, tx = 0;
    if (D2) {
      tn = uint(t) / (p.th * p.tw);
      ty = (uint(t) / p.tw) % p.th;
      tx = uint(t) % p.tw;
    }
    LK_UNROLL for (int j = 0; j < M; ++j) {
      size_t row;
      if (D2) {  // output (2 ty + j / 2, 2 tx + j % 2) of image tn
        const uint oy = 2 * ty + uint(j / 2), ox = 2 * tx + uint(j % 2);
        if (t * M >= lor || oy >= p.ho || ox >= p.wo) continue;
        row = (size_t(tn * p.ho + oy) * p.wo + ox) * co;
      } else {
        const int uo = t * M + j;
        if (uo >= lor) continue;
        row = size_t(rho + uint(uo) * d) * co;
      }
      LK_UNROLL for (int k = 0; k < FC; ++k) {
        const uint col = c0 + uint(sc * (BN / 2) + k * 8 + fn);
        if (D2 && p.splits > 1) {  // partial over this split's points; lk_wino2d_finalize adds the epilogue
          device atomic_float* ao = reinterpret_cast<device atomic_float*>(out + row + col);
          atomic_fetch_add_explicit(ao, y[j][i][k][0], memory_order_relaxed);
          atomic_fetch_add_explicit(ao + 1, y[j][i][k][1], memory_order_relaxed);
          continue;
        }
        float2 o = y[j][i][k] + *reinterpret_cast<const device float2*>(bias + col);
        if (p.has_res) o += *reinterpret_cast<const device float2*>(res + row + col);
        o *= p.scale;
        if (D2 && p.act == 1) o = max(o, 0.0f);
        *reinterpret_cast<device float2*>(out + row + col) = o;
        if (STATS) y[j][i][k] = o;
      }
    }
  }

  if (STATS) {
    // Channel stats of this threadgroup's output rows (one chunk = BT * M rows of residue class rho; for d = 1
    // exactly lk_stats_lc's 128-row chunk tg.y): (mean, M2) per column -> st[rho * chunks + tg.y, C, 2]. Rows past
    // the class end are masked. Two passes over the registers (mean, then M2), each reduced over the lanes holding
    // other rows (lane bits 1, 2, 4) and over the two simdgroup rows (threadgroup memory).
    threadgroup float red[2][BN];
    threadgroup float meanv[BN];
    const int nrows = min(BT * M, lor - t0 * M);
    LK_UNROLL for (int pass = 0; pass < 2; ++pass) {
      LK_UNROLL for (int k = 0; k < FC; ++k) {  // one column pair at a time: few live registers
        const int cl = sc * (BN / 2) + k * 8 + fn;
        const float2 mk = pass == 0 ? float2(0.0f) : float2(meanv[cl], meanv[cl + 1]);
        float2 part = 0.0f;
        LK_UNROLL for (int i = 0; i < FR; ++i) {
          const int t = t0 + sr * (BT / 2) + i * 8 + fm;
          LK_UNROLL for (int j = 0; j < M; ++j) {
            if (t * M + j >= lor) continue;
            const float2 yv = y[j][i][k];
            const float2 dv = yv - mk;
            part += pass == 0 ? dv : dv * dv;
          }
        }
        part += simd_shuffle_xor(part, ushort(2));
        part += simd_shuffle_xor(part, ushort(4));
        part += simd_shuffle_xor(part, ushort(16));
        if (fm == 0) {
          red[sr][cl] = part[0];
          red[sr][cl + 1] = part[1];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (pass == 0) {
        if (lid < uint(BN)) meanv[lid] = (red[0][lid] + red[1][lid]) / float(nrows);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      } else if (lid < uint(BN)) {
        const size_t o = ((size_t(rho) * p.chunks + tg.y) * co + c0 + lid) * 2;
        st[o] = meanv[lid];
        st[o + 1] = red[0][lid] + red[1][lid];
      }
    }
  }
}

#define LK_WINO_MM(NAME, M, A, N, BT, BN, BK)                                                                       \
  template [[host_name(NAME)]] [[kernel]] void lk_wino_mm_lc<M, A, N, BT, BN, BK>(                                \
      const device float*, const device float*, const device float*, const device float*, device float*,          \
      const device float*, constant WinoMmParams&, device float*, uint3, uint, uint, uint);

LK_WINO_MM("lk_wino_mm_lc_m4a6n1", 4, 6, 1, 32, 64, 16)  // F(4,3): K = 3
LK_WINO_MM("lk_wino_mm_lc_m4a7n2", 4, 7, 2, 32, 64, 16)  // nested F(4,4), K = 4..8
LK_WINO_MM("lk_wino_mm_lc_m4a7n3", 4, 7, 3, 32, 64, 16)  // nested F(4,4), K = 9..12
LK_WINO_MM("lk_wino_mm_lc_m2a4n1", 2, 4, 1, 64, 64, 16)  // F(2,3), K = 3: 32x32 per simdgroup like steel
LK_WINO_MM("lk_wino_mm_lc_m2a4n1_bt32", 2, 4, 1, 32, 64, 16)
// 16-tile threadgroups for small grids (short sequences: more threadgroups to fill the GPU)
LK_WINO_MM("lk_wino_mm_lc_m4a6n1_bt16", 4, 6, 1, 16, 64, 16)
LK_WINO_MM("lk_wino_mm_lc_m4a7n2_bt16", 4, 7, 2, 16, 64, 16)
LK_WINO_MM("lk_wino_mm_lc_m4a7n3_bt16", 4, 7, 3, 16, 64, 16)

// ---- 2-D: F(2x2, 3x3), stride 1, padding 1, NHWC (ResNet-18's 3x3 convs) -----------------------------------------
// Same GEMM kernel (D2 = true): A = 16 points (4x4), M = 4 outputs (2x2) per tile, AT2 = AT (x) AT given in `tr`.

struct WinoTf2Params {
  uint n, h, w, c, th, tw, rmax;
};

// x [N, H, W, C], bt = F(2,3)'s BT [4x4], V [16, rmax, C]. Grid (C / 4, rmax). (The GEMM gets tr = [BT2 | AT2].)
// V[(a, b), t] = sum_{i,k} BT[a, i] BT[b, k] x[2 ty + i - 1, 2 tx + k - 1] (separable: rows, then columns).
kernel void lk_wino_tf2d_nhwc(const device float* x [[buffer(0)]], const device float* tr [[buffer(1)]],
                              device float* v [[buffer(2)]], constant WinoTf2Params& p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
  const uint c4 = gid.x, t = gid.y, c = p.c;
  if (c4 * 4 >= c || t >= p.rmax) return;
  const uint tiles = p.n * p.th * p.tw;
  float4 in[4][4];
  if (t < tiles) {
    const uint tn = t / (p.th * p.tw), ty = (t / p.tw) % p.th, tx = t % p.tw;
    LK_UNROLL for (int i = 0; i < 4; ++i) LK_UNROLL for (int k = 0; k < 4; ++k) {
      const int yy = int(2 * ty) + i - 1, xx = int(2 * tx) + k - 1;
      in[i][k] = (yy >= 0 && yy < int(p.h) && xx >= 0 && xx < int(p.w))
                     ? *reinterpret_cast<const device float4*>(x + ((size_t(tn) * p.h + uint(yy)) * p.w + uint(xx)) * c + c4 * 4)
                     : float4(0.0f);
    }
  } else {
    LK_UNROLL for (int i = 0; i < 4; ++i) LK_UNROLL for (int k = 0; k < 4; ++k) in[i][k] = 0.0f;
  }
  float4 tmp[4][4];
  LK_UNROLL for (int a = 0; a < 4; ++a) LK_UNROLL for (int k = 0; k < 4; ++k) {
    float4 s = 0.0f;
    LK_UNROLL for (int i = 0; i < 4; ++i) s += tr[a * 4 + i] * in[i][k];
    tmp[a][k] = s;
  }
  LK_UNROLL for (int a = 0; a < 4; ++a) LK_UNROLL for (int b = 0; b < 4; ++b) {
    float4 s = 0.0f;
    LK_UNROLL for (int k = 0; k < 4; ++k) s += tr[b * 4 + k] * tmp[a][k];
    *reinterpret_cast<device float4*>(v + (size_t(a * 4 + b) * p.rmax + t) * c + c4 * 4) = s;
  }
}

#define LK_WINO_MMS(NAME, M, A, N)                                                                                  \
  template [[host_name(NAME)]] [[kernel]] void lk_wino_mm_lc<M, A, N, 32, 64, 16, false, true>(            \
      const device float*, const device float*, const device float*, const device float*, device float*,          \
      const device float*, constant WinoMmParams&, device float*, uint3, uint, uint, uint);

// + channel stats of the output in the epilogue (always 32-tile threadgroups: one 128-row chunk each)
LK_WINO_MMS("lk_wino_mm_lc_m4a6n1_stats", 4, 6, 1)
LK_WINO_MMS("lk_wino_mm_lc_m4a7n2_stats", 4, 7, 2)
LK_WINO_MMS("lk_wino_mm_lc_m4a7n3_stats", 4, 7, 3)
template [[host_name("lk_wino_mm_lc_m2a4n1_stats")]] [[kernel]] void lk_wino_mm_lc<2, 4, 1, 64, 64, 16, false, true>(
    const device float*, const device float*, const device float*, const device float*, device float*,
    const device float*, constant WinoMmParams&, device float*, uint3, uint, uint, uint);

// stats [S, C, 2] from lk_wino_mm_lc_*_stats (chunk i = (rho, y) = (i / chunks, i % chunks), rows
// clamp(lor_rho - 128 y, 0, 128)), gb [2 creal] -> AdaIN affine [3, C] = (mean, (1 + g) / std, b). Same combine as
// lk_adain_affine_lc (kokoro_lc.metal): one simdgroup per channel, Kahan-compensated sums, exact chunk merge.
struct WinoAffineParams {
  uint l, c, d, chunks, rows;  // rows: output rows per chunk (128)
  float eps;
  uint creal;
};

kernel void lk_wino_adain_affine_lc(const device float* st [[buffer(0)]], const device float* gb [[buffer(1)]],
                                    device float* af [[buffer(2)]], constant WinoAffineParams& p [[buffer(3)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg * 4 + sg;
  if (c >= p.c) return;
  if (c >= p.creal) {
    if (lane == 0) af[c] = af[p.c + c] = af[2 * p.c + c] = 0.0f;
    return;
  }
  const uint total = p.d * p.chunks;
  auto count = [&](uint i) {
    const uint rho = i / p.chunks, yy = i % p.chunks;
    const int lr = rho < p.l ? int((p.l - rho + p.d - 1) / p.d) : 0;
    return float(clamp(lr - int(yy * p.rows), 0, int(p.rows)));
  };
  float s = 0.0f, comp = 0.0f;
  for (uint i = lane; i < total; i += 32) {
    const float n = count(i);
    if (n == 0.0f) continue;
    const float yv = st[(size_t(i) * p.c + c) * 2] * n - comp;
    const float t = s + yv;
    comp = (t - s) - yv;
    s = t;
  }
  const float mean = simd_sum(s) / float(p.l);
  float m2 = 0.0f;
  comp = 0.0f;
  for (uint i = lane; i < total; i += 32) {
    const float n = count(i);
    if (n == 0.0f) continue;
    const float dm = st[(size_t(i) * p.c + c) * 2] - mean;
    const float yv = st[(size_t(i) * p.c + c) * 2 + 1] + n * dm * dm - comp;
    const float t = m2 + yv;
    comp = (t - m2) - yv;
    m2 = t;
  }
  m2 = simd_sum(m2);
  if (lane == 0) {
    af[c] = mean;
    af[p.c + c] = (1.0f + gb[c]) * rsqrt(m2 / float(p.l) + p.eps);
    af[2 * p.c + c] = gb[p.creal + c];
  }
}

#define LK_WINO_MM2(NAME, BT)                                                                                       \
  template [[host_name(NAME)]] [[kernel]] void lk_wino_mm_lc<4, 16, 1, BT, 64, 16, true>(                         \
      const device float*, const device float*, const device float*, const device float*, device float*,          \
      const device float*, constant WinoMmParams&, device float*, uint3, uint, uint, uint);

LK_WINO_MM2("lk_wino_mm2d_nhwc", 32)
LK_WINO_MM2("lk_wino_mm2d_nhwc_bt16", 16)



// Split-point 2-D path: out = 0 first (lk_wino_fill_zero), then the partial GEMMs, then this epilogue in place.
struct Wino2dFinParams {
  uint count, co, has_res, act;
};
kernel void lk_wino_fill_zero(device float4* out [[buffer(0)]], constant uint& n4 [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid < n4) out[gid] = 0.0f;
}
kernel void lk_wino2d_finalize(device float4* out [[buffer(0)]], const device float4* bias [[buffer(1)]],
                               const device float4* res [[buffer(2)]], constant Wino2dFinParams& p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  float4 o = out[gid] + bias[gid % (p.co / 4)];
  if (p.has_res) o += res[gid];
  if (p.act == 1) o = max(o, 0.0f);
  out[gid] = o;
}
