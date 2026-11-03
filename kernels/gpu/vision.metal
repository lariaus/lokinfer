// 2-D vision helper kernels, NHWC (M6: ResNet-18). The conv itself is lk_conv1d_lc (kokoro_lc.metal), the
// general NHWC steel conv with a bias / residual / ReLU epilogue.
#include <metal_stdlib>

using namespace metal;

struct NhwcPadParams {
  uint n, c, hw, cpad;
};

// x [N, C, H, W] -> out [N, H, W, Cpad], channels >= C zero (the steel loader needs Ci % 16 == 0).
kernel void lk_nchw_to_nhwc_pad(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                                constant NhwcPadParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n * p.hw * p.cpad) return;
  const uint c = gid % p.cpad, pix = (gid / p.cpad) % p.hw, b = gid / (p.cpad * p.hw);
  out[gid] = c < p.c ? x[(static_cast<size_t>(b) * p.c + c) * p.hw + pix] : 0.0f;
}

// Same, one thread per pixel (Cpad % 4 == 0, C <= 16): coalesced plane reads, one float4 store per 4 channels.
// The per-element kernel above reads 3 planes in a scattered pattern per simdgroup (MobileNetV2 N=8 stem 0.26 ms).
kernel void lk_nchw_to_nhwc_pad_pix(const device float* x [[buffer(0)]], device float4* out [[buffer(1)]],
                                    constant NhwcPadParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n * p.hw) return;
  const uint pix = gid % p.hw, b = gid / p.hw;
  float v[16];  // compile-time indices only (unrolled), so it stays in registers
#pragma unroll
  for (uint c = 0; c < 16; ++c) v[c] = c < p.c ? x[(size_t(b) * p.c + c) * p.hw + pix] : 0.0f;
  device float4* dst = out + size_t(gid) * (p.cpad / 4);
#pragma unroll
  for (uint q = 0; q < 4; ++q)
    if (q < p.cpad / 4) dst[q] = float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
  for (uint q = 4; q < p.cpad / 4; ++q) dst[q] = 0.0f;
}

struct PoolParams {
  uint n, h, w, c, ho, wo, k, pad, stride;
};

// x [N, H, W, C] -> out [N, Ho, Wo, C]; padding acts as -inf.
kernel void lk_maxpool_nhwc(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                            constant PoolParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n * p.ho * p.wo * p.c) return;
  const uint c = gid % p.c, ow = (gid / p.c) % p.wo, oh = (gid / (p.c * p.wo)) % p.ho, b = gid / (p.c * p.wo * p.ho);
  float m = -INFINITY;
  for (uint y = 0; y < p.k; ++y) {
    const int ih = int(oh * p.stride + y) - int(p.pad);
    if (ih < 0 || ih >= int(p.h)) continue;
    for (uint xk = 0; xk < p.k; ++xk) {
      const int iw = int(ow * p.stride + xk) - int(p.pad);
      if (iw < 0 || iw >= int(p.w)) continue;
      m = max(m, x[((static_cast<size_t>(b) * p.h + ih) * p.w + iw) * p.c + c]);
    }
  }
  out[gid] = m;
}

struct GapParams {
  uint n, hw, c;
};

// x [N, H, W, C] -> out [N, C] (mean over H, W).
kernel void lk_global_avgpool_nhwc(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                                   constant GapParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n * p.c) return;
  const uint c = gid % p.c, b = gid / p.c;
  float s = 0.0f;
  for (uint i = 0; i < p.hw; ++i) s += x[(static_cast<size_t>(b) * p.hw + i) * p.c + c];
  out[gid] = s / float(p.hw);
}

struct DwParams {
  uint n, h, w, c, kh, kw, ho, wo, stride, pad, act;
};

// Depthwise conv: x [N, H, W, C], w [KH, KW, C], bias [C] -> out [N, Ho, Wo, C], act 0 none / 1 ReLU / 2 ReLU6.
kernel void lk_dwconv2d_nhwc(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                             const device float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                             constant DwParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n * p.ho * p.wo * p.c) return;
  const uint c = gid % p.c, ow = (gid / p.c) % p.wo, oh = (gid / (p.c * p.wo)) % p.ho, b = gid / (p.c * p.wo * p.ho);
  float acc = bias[c];
  for (uint y = 0; y < p.kh; ++y) {
    const int ih = int(oh * p.stride + y) - int(p.pad);
    if (ih < 0 || ih >= int(p.h)) continue;
    for (uint xk = 0; xk < p.kw; ++xk) {
      const int iw = int(ow * p.stride + xk) - int(p.pad);
      if (iw < 0 || iw >= int(p.w)) continue;
      acc += x[((static_cast<size_t>(b) * p.h + ih) * p.w + iw) * p.c + c] * w[(y * p.kw + xk) * p.c + c];
    }
  }
  if (p.act == 1) acc = max(acc, 0.0f);
  else if (p.act == 2) acc = clamp(acc, 0.0f, 6.0f);
  out[gid] = acc;
}

constant constexpr uint DW_OT = 4;

// Stride is a template parameter so every array index is compile-time and the loops fully unroll (with a runtime
// stride the arrays went to stack memory and the kernel was slower than the scalar one).
template <uint S>
inline void dw3c4(const device float4* x, const device float4* w, const device float4* bias, device float4* out,
                  constant DwParams& p, uint gid) {
  const uint c4n = p.c / 4, wt = (p.wo + DW_OT - 1) / DW_OT;
  if (gid >= p.n * p.ho * wt * c4n) return;
  const uint c4 = gid % c4n, owt = (gid / c4n) % wt, oh = (gid / (c4n * wt)) % p.ho, b = gid / (c4n * wt * p.ho);
  const uint ow0 = owt * DW_OT;
  constexpr uint NC = (DW_OT - 1) * S + 3;
  const float4 bv = bias[c4];
  float4 acc[DW_OT];
#pragma unroll
  for (uint o = 0; o < DW_OT; ++o) acc[o] = bv;
  const int iw0 = int(ow0 * S) - int(p.pad);
#pragma unroll
  for (uint y = 0; y < 3; ++y) {
    const int ih = int(oh * S + y) - int(p.pad);
    if (ih < 0 || ih >= int(p.h)) continue;
    const device float4* row = x + (size_t(b) * p.h + uint(ih)) * p.w * c4n + c4;
    const float4 w0 = w[(y * 3 + 0) * c4n + c4], w1 = w[(y * 3 + 1) * c4n + c4], w2 = w[(y * 3 + 2) * c4n + c4];
    float4 in[NC];
#pragma unroll
    for (uint j = 0; j < NC; ++j) {
      const int iw = iw0 + int(j);
      in[j] = (iw >= 0 && iw < int(p.w)) ? row[size_t(iw) * c4n] : float4(0.0f);
    }
#pragma unroll
    for (uint o = 0; o < DW_OT; ++o) acc[o] = fma(in[o * S + 2], w2, fma(in[o * S + 1], w1, fma(in[o * S], w0, acc[o])));
  }
#pragma unroll
  for (uint o = 0; o < DW_OT; ++o) {
    if (ow0 + o >= p.wo) break;
    float4 v = acc[o];
    if (p.act == 1) v = max(v, 0.0f);
    else if (p.act == 2) v = clamp(v, 0.0f, 6.0f);
    out[((size_t(b) * p.ho + oh) * p.wo + ow0 + o) * c4n + c4] = v;
  }
}

// 3x3 depthwise, C % 4 == 0, stride 1 / 2 (all of MobileNetV2): one thread = 4 channels (float4) x DW_OT consecutive
// output columns, so each input row segment is read once for the DW_OT outputs.
kernel void lk_dwconv2d_nhwc_k3s1c4(const device float4* x [[buffer(0)]], const device float4* w [[buffer(1)]],
                                    const device float4* bias [[buffer(2)]], device float4* out [[buffer(3)]],
                                    constant DwParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  dw3c4<1>(x, w, bias, out, p, gid);
}

kernel void lk_dwconv2d_nhwc_k3s2c4(const device float4* x [[buffer(0)]], const device float4* w [[buffer(1)]],
                                    const device float4* bias [[buffer(2)]], device float4* out [[buffer(3)]],
                                    constant DwParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  dw3c4<2>(x, w, bias, out, p, gid);
}

struct ConvT2dParams {
  uint b, h, w, co, k, ho, wo, stride, pad, act;  // act: 1 ReLU, 4 tanh
};

// ConvTranspose2d overlap-add (DCGAN G): y [B, H, W, K*K*Co] = x @ W (one GEMM), bias [Co] -> out [B, Ho, Wo, Co].
kernel void lk_convt2d_gather_nhwc(const device float* y [[buffer(0)]], const device float* bias [[buffer(1)]],
                                   device float* out [[buffer(2)]], constant ConvT2dParams& p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.b * p.ho * p.wo * p.co) return;
  const uint c = gid % p.co, ow = (gid / p.co) % p.wo, oh = (gid / (p.co * p.wo)) % p.ho,
             bi = gid / (p.co * p.wo * p.ho);
  const size_t ldy = static_cast<size_t>(p.k) * p.k * p.co;
  float acc = bias[c];
  for (uint kh = (oh + p.pad) % p.stride; kh < p.k; kh += p.stride) {
    const int nh = int(oh + p.pad) - int(kh);
    if (nh < 0) break;
    const uint ih = uint(nh) / p.stride;
    if (ih >= p.h) continue;
    for (uint kw = (ow + p.pad) % p.stride; kw < p.k; kw += p.stride) {
      const int nw = int(ow + p.pad) - int(kw);
      if (nw < 0) break;
      const uint iw = uint(nw) / p.stride;
      if (iw >= p.w) continue;
      acc += y[((static_cast<size_t>(bi) * p.h + ih) * p.w + iw) * ldy + (kh * p.k + kw) * p.co + c];
    }
  }
  out[gid] = p.act == 1 ? max(acc, 0.0f) : (p.act == 4 ? precise::tanh(acc) : acc);
}

struct ToNchwParams {
  uint n, hw, c;
};

// x [N, H, W, C] -> out [N, C, H, W]
kernel void lk_nhwc_to_nchw(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                            constant ToNchwParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n * p.hw * p.c) return;
  const uint pix = gid % p.hw, c = (gid / p.hw) % p.c, b = gid / (p.hw * p.c);
  out[gid] = x[(static_cast<size_t>(b) * p.hw + pix) * p.c + c];
}

kernel void lk_sigmoid(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) out[gid] = 1.0f / (1.0f + precise::exp(-x[gid]));
}

struct Im2colParams {
  uint n, c, h, w, kh, kw, ho, wo, stride, pad, kpad;
};

// Stem im2col straight from the NCHW graph input: out [N*Ho*Wo, Kpad], column k = (y*KW + x)*C + c (the GEMM weight's
// row order), zero for padding pixels and for k >= KH*KW*C. Lets a 3-channel stem conv run as one GEMM with K = 147
// (padded to 160) instead of the steel conv with channels padded to 16 (K = 784).
kernel void lk_im2col_nchw(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                           constant Im2colParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x, pix = gid.y;
  if (k >= p.kpad || pix >= p.n * p.ho * p.wo) return;
  float v = 0.0f;
  const uint kreal = p.kh * p.kw * p.c;
  if (k < kreal) {
    const uint c = k % p.c, tap = k / p.c, xk = tap % p.kw, y = tap / p.kw;
    const uint ow = pix % p.wo, oh = (pix / p.wo) % p.ho, b = pix / (p.wo * p.ho);
    const int ih = int(oh * p.stride + y) - int(p.pad), iw = int(ow * p.stride + xk) - int(p.pad);
    if (ih >= 0 && ih < int(p.h) && iw >= 0 && iw < int(p.w))
      v = x[((size_t(b) * p.c + c) * p.h + ih) * p.w + iw];
  }
  out[size_t(pix) * p.kpad + k] = v;
}

// Tiled version: one threadgroup (256 threads) = 32 consecutive output pixels of one output row. It stages the
// input patch they need (C x KH x ((32-1)*stride + KW)) in threadgroup memory with coalesced row reads, then writes
// the 32 x Kpad output block contiguously. The per-element kernel above reads across channel planes per simdgroup
// (ResNet-18 N=8 stem: 1.22 ms, ~50 GB/s). Used when the patch fits IM2COL_PATCH floats.
constant constexpr uint IM2COL_OW = 32, IM2COL_PATCH = 4096;

kernel void lk_im2col_nchw_tile(const device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                                constant Im2colParams& p [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float patch[IM2COL_PATCH];
  threadgroup int koff[256];  // column k -> patch offset (-1 = zero padding column); Kpad <= 256
  const uint ow0 = tg.x * IM2COL_OW, oh = tg.y, b = tg.z;
  const uint pw = (IM2COL_OW - 1) * p.stride + p.kw, plane = p.kh * pw;
  for (uint e = tid; e < p.c * plane; e += 256) {
    const uint c = e / plane, r = e % plane, y = r / pw, j = r % pw;
    const int ih = int(oh * p.stride + y) - int(p.pad), iw = int(ow0 * p.stride + j) - int(p.pad);
    patch[e] = (ih >= 0 && ih < int(p.h) && iw >= 0 && iw < int(p.w))
                   ? x[((size_t(b) * p.c + c) * p.h + uint(ih)) * p.w + uint(iw)] : 0.0f;
  }
  const uint kreal = p.kh * p.kw * p.c, nout = min(IM2COL_OW, p.wo - ow0);
  if (tid < p.kpad) {
    const uint c = tid % p.c, tap = tid / p.c;
    koff[tid] = tid < kreal ? int(c * plane + (tap / p.kw) * pw + tap % p.kw) : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* dst = out + (size_t(b * p.ho + oh) * p.wo + ow0) * p.kpad;
  // e = o * Kpad + k, walked incrementally (no division per element).
  uint o = tid / p.kpad, k = tid % p.kpad;
  const uint step_o = 256 / p.kpad, step_k = 256 % p.kpad;
  for (uint e = tid; e < nout * p.kpad; e += 256) {
    const int off = koff[k];
    dst[e] = off >= 0 ? patch[off + o * p.stride] : 0.0f;
    o += step_o;
    k += step_k;
    if (k >= p.kpad) { k -= p.kpad; ++o; }
  }
}

struct RowDotParams {
  uint rows, k;
  uint act;  // 0 none, 1 sigmoid
};

// out[r] = act(x[r, :] . w + b[0]): x [R, K], w [K], b [1] -> out [R]. One threadgroup (256) per row, SIMD reduction.
// GEMV with a single output column (DCGAN D's last layer) — a tiled GEMM would leave the GPU almost idle.
kernel void lk_rowdot(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                      const device float* b [[buffer(2)]], device float* out [[buffer(3)]],
                      constant RowDotParams& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                      uint simd_id [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float partial[8];
  const device float* xr = x + size_t(row) * p.k;
  float s = 0.0f;
  for (uint i = lid; i < p.k; i += 256) s += xr[i] * w[i];
  s = simd_sum(s);
  if (simd_lane == 0) partial[simd_id] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float t = b[0];
    for (uint i = 0; i < 8; ++i) t += partial[i];
    out[row] = p.act == 1 ? 1.0f / (1.0f + precise::exp(-t)) : t;
  }
}
