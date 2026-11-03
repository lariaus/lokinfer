// 2-D vision kernels, NHWC (channels-last), f32 (M6: ResNet-18 / MobileNetV2; research-models.md §2).
//
//   conv2d_nhwc      out = act(conv2d(x, w) + bias [+ residual]), act ∈ {none, ReLU, ReLU6}, w as [KH, KW, Ci, Co]
//                    (= im2col K-major rows).
//                    Pixel tiles over the runtime pool: per tile, im2col into a ~3 MB block [P, KH*KW*Ci] and one
//                    single-threaded sgemm [P, K] x [K, Co] (the Kokoro conv recipe, M2/M3); the epilogue adds
//                    bias / residual and applies ReLU while the tile is in cache. 1x1 stride-1 convs skip im2col.
//   dwconv2d_nhwc    depthwise k x k conv (MobileNetV2), + bias, act; NEON over channels, rows on the pool
//   maxpool2d_nhwc   k x k, stride s, padding p (padding = -inf)
//   global_avgpool_nhwc   [N, H, W, C] -> [N, C]
//   nchw_to_nhwc / nhwc_to_nchw   layout conversions at the graph boundary (PyTorch's layout)
//   convt2d_gather_nhwc   ConvTranspose2d overlap-add from y = x @ W ([B, H, W, K*K*Co], one GEMM) + bias, act
//                    (DCGAN G); a gather (no atomics): each output reads its <= ceil(K/s)^2 contributions
//   sigmoid          elementwise
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "kokoro_common.h"
#include "sme2_gemm.h"
#include "lokinfer/runtime/thread_pool.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

namespace {

constexpr int64_t kColBudget = int64_t{3} << 20;  // im2col block bytes per task
// Inline (calling thread) below this many MACs, pool above. One value for every device: the "per-device threshold"
// of the threading design has not found a device that wants another value yet. M2 Max A/B 2026-11-02 (interleaved, 2
// rounds): 1e6 -> ResNet-18 N=1 4.3 -> 4.8 ms; 1.6e7 -> MobileNetV2 N=1 1.5 -> 2.6 ms; DCGAN / N=8 rows within noise.
// The iPhone was tuned around this value (2026-11-01: serial pointwise and a 1e6 depthwise threshold both slower);
// re-check 2e6 / 8e6 there when the phone is free.
constexpr double kParallelMacs = 4e6;
// Parallel convs: ~3 tasks per pool thread (the iPhone's E-cores are slower; tried 1 / 3 / 8 on the M2 Max: within
// noise, 3 best on MobileNetV2 N=8) of >= 0.5 M MACs each (below that the per-sgemm fixed cost dominates).
constexpr int64_t kTilesPerThread = 3;
constexpr double kMinTileMacs = 5e5;

enum class Act { None, Relu, Relu6, Leaky, Tanh };  // Leaky: slope 0.2 (DCGAN D)

struct Conv2dArgs {
  const float* x;
  const float* w;
  const float* bias;
  const float* res;
  float* out;
  int64_t n, h, wd, ci, kh, kw, co, ho, wo, stride, pad;
  Act act;
};

constexpr float kLeakySlope = 0.2f;
// (Tried: MobileNetV2's small-map blocks (<= 1600 input pixels) and small pointwise convs as single-stream SME2 GEMMs
// with the weights as the in-place B operand: iPhone N = 1 1.62 -> 1.99-2.07 ms, N = 8 8.28 -> 8.44-8.66. Dropped.)

inline float actS(float v, Act act) {
  switch (act) {
    case Act::None: return v;
    case Act::Relu: return std::max(v, 0.f);
    case Act::Relu6: return std::min(std::max(v, 0.f), 6.f);
    case Act::Leaky: return v >= 0.f ? v : v * kLeakySlope;
    case Act::Tanh: return std::tanh(v);
  }
  return v;
}

inline float32x4_t actV(float32x4_t v, Act act) {
  switch (act) {
    case Act::None: return v;
    case Act::Relu: return vmaxq_f32(v, vdupq_n_f32(0.f));
    case Act::Relu6: return vminq_f32(vmaxq_f32(v, vdupq_n_f32(0.f)), vdupq_n_f32(6.f));
    case Act::Leaky: return vbslq_f32(vcgeq_f32(v, vdupq_n_f32(0.f)), v, vmulq_f32(v, vdupq_n_f32(kLeakySlope)));
    case Act::Tanh: {
      float t[4];
      vst1q_f32(t, v);
      for (float& e : t) e = std::tanh(e);
      return vld1q_f32(t);
    }
  }
  return v;
}

// o[r, c] = act(o[r, c] + bias[c] (+ res[r, c])) for r < rows, c < nc; o and res have row stride ld.
void epilogue(float* o, int64_t rows, int64_t nc, int64_t ld, const float* bias, const float* res, Act act) {
  for (int64_t r = 0; r < rows; ++r) {
    float* orow = o + r * ld;
    const float* rrow = res ? res + r * ld : nullptr;
    int64_t c = 0;
    for (; c + 4 <= nc; c += 4) {
      float32x4_t v = vaddq_f32(vld1q_f32(orow + c), vld1q_f32(bias + c));
      if (rrow) v = vaddq_f32(v, vld1q_f32(rrow + c));
      vst1q_f32(orow + c, actV(v, act));
    }
    for (; c < nc; ++c) orow[c] = actS(orow[c] + bias[c] + (rrow ? rrow[c] : 0.f), act);
  }
}

// im2col rows [p0, p0 + np) into dst [np, KH*KW*Ci].
void im2col(const Conv2dArgs& a, int64_t p0, int64_t np, float* dst0) {
  const int64_t k = a.kh * a.kw * a.ci;
  for (int64_t q = 0; q < np; ++q) {
    const int64_t p = p0 + q, ni = p / (a.ho * a.wo), rem = p % (a.ho * a.wo), oh = rem / a.wo, ow = rem % a.wo;
    float* dst = dst0 + q * k;
    const int64_t iwLo = ow * a.stride - a.pad;
    const bool colsIn = iwLo >= 0 && iwLo + a.kw <= a.wd;
    for (int64_t y = 0; y < a.kh; ++y) {
      const int64_t ih = oh * a.stride - a.pad + y;
      if (colsIn && ih >= 0 && ih < a.h) {  // the kernel row's kw taps are kw*ci contiguous floats in NHWC
        const float* src = a.x + ((ni * a.h + ih) * a.wd + iwLo) * a.ci;
        float* d = dst + y * a.kw * a.ci;
        const int64_t len = a.kw * a.ci;
        int64_t q2 = 0;
        for (; q2 + 4 <= len; q2 += 4) vst1q_f32(d + q2, vld1q_f32(src + q2));
        for (; q2 < len; ++q2) d[q2] = src[q2];
        continue;
      }
      for (int64_t xk = 0; xk < a.kw; ++xk) {
        const int64_t iw = ow * a.stride - a.pad + xk;
        float* d = dst + (y * a.kw + xk) * a.ci;
        if (ih < 0 || ih >= a.h || iw < 0 || iw >= a.wd) {
          for (int64_t q2 = 0; q2 < a.ci; ++q2) d[q2] = 0.f;
        } else if (a.ci < 16) {  // tiny channel counts (3-channel stems): a loop beats a memcpy call per tap
          const float* src = a.x + ((ni * a.h + ih) * a.wd + iw) * a.ci;
          for (int64_t q2 = 0; q2 < a.ci; ++q2) d[q2] = src[q2];
        } else {
          std::memcpy(d, a.x + ((ni * a.h + ih) * a.wd + iw) * a.ci, static_cast<size_t>(a.ci) * sizeof(float));
        }
      }
    }
  }
}

const float* zeroRow(int64_t n);

// SME2 engine (A19 / M4+; sme2_gemm.cpp), OPT-IN (LOKINFER_SME2=1), needs iPhone validation of the selection rule
// below. One FMOPA stream on the calling thread; the SME units are per cluster, so more threads don't add throughput
// (iPhone microbench 2026-11-02: 2 or 6 pool tasks were 5-60 % slower than one stream, which reaches ~1.8 TF/s on
// ResNet-18's 3x3 shapes with A pre-packed vs Accelerate's 1.2-1.6). The im2col is implicit: the pixel tile's taps are
// gathered straight into the packed A panels (ZA transposes). In-model (iPhone CPU profile, all convs on SME vs the
// Accelerate grid below): the packing makes it lose on big maps (56x56: 215 -> 240-300 us; 7x7 stem 378 -> 733 us,
// single-threaded im2col) and on the 9.4 MB-weight 7x7x512 layers (330-380 -> 420 us), and win on 28x28 / 14x14 maps
// (14x14x256: 266-294 -> 204-210 us; strided 1x1 downsamples 70 -> 26 us); e2e ResNet-18 N=1 4.18 -> 3.88 ms, N=8
// 23.3 -> 23.6, MobileNetV2 N=1 1.65 -> 2.28 (small pointwise convs). smeWins() keeps only the winning class.
bool sme2On() {
  static const bool on = [] {
    const char* e = std::getenv("LOKINFER_SME2");
    return sme2::available() && e && e[0] == '1';
  }();
  return on;
}

bool smeWins(const Conv2dArgs& a) {
  const int64_t pixels = a.n * a.ho * a.wo, k = a.kh * a.kw * a.ci;
  const bool pointwise = a.kh == 1 && a.kw == 1 && a.stride == 1 && a.pad == 0;
  return !pointwise && a.ci % 16 == 0 && pixels <= 1024 && k * a.co * 4 <= (int64_t{8} << 20) &&
         (k >= 1024 || a.kh * a.kw == 1);
}

constexpr int64_t kSmePackBytes = int64_t{2} << 20;  // packed A per pixel tile (L2-resident)

// Packs pixels [p0, p0 + np) of the conv's implicit im2col matrix into A panels.
void smePack(const Conv2dArgs& a, int64_t p0, int64_t np, float* ap) {
  const int64_t k = a.kh * a.kw * a.ci, taps = a.kh * a.kw;
  if (a.kh == 1 && a.kw == 1 && a.stride == 1 && a.pad == 0) return sme2::packPanels(a.x + p0 * a.ci, a.ci, np, k, ap);
  if (a.ci % 16 == 0) {
    const float* zero = zeroRow(a.ci);
    thread_local std::vector<const float*> ptrs;
    ptrs.resize(static_cast<size_t>(np * taps));
    for (int64_t q = 0; q < np; ++q) {
      const int64_t p = p0 + q, ni = p / (a.ho * a.wo), rem = p % (a.ho * a.wo), oh = rem / a.wo, ow = rem % a.wo;
      const float** row = ptrs.data() + q * taps;
      for (int64_t y = 0; y < a.kh; ++y) {
        const int64_t ih = oh * a.stride - a.pad + y;
        for (int64_t xk = 0; xk < a.kw; ++xk) {
          const int64_t iw = ow * a.stride - a.pad + xk;
          row[y * a.kw + xk] = ih < 0 || ih >= a.h || iw < 0 || iw >= a.wd
                                   ? zero : a.x + ((ni * a.h + ih) * a.wd + iw) * a.ci;
        }
      }
    }
    return sme2::packPanelsGather(ptrs.data(), np, taps, a.ci, ap);
  }
  float* col = detail::scratch(static_cast<size_t>(np * k), 2);
  im2col(a, p0, np, col);
  sme2::packPanels(col, k, np, k, ap);
}

// C rows [p0, p0 + np) x columns [c0, c0 + nc) from packed panels ap (row p0 = panel row 0).
void smeGemm(const Conv2dArgs& a, const float* ap, int64_t p0, int64_t np, int64_t c0, int64_t nc) {
  const int64_t k = a.kh * a.kw * a.ci, co = a.co;
  sme2::Epilogue e;
  e.bias = a.bias + c0;
  e.res = a.res ? a.res + p0 * co + c0 : nullptr;
  e.ldres = co;
  e.act = a.act == Act::Relu ? sme2::Act::Relu : a.act == Act::Relu6 ? sme2::Act::Relu6
        : a.act == Act::Leaky ? sme2::Act::Leaky : sme2::Act::None;
  e.leaky = kLeakySlope;
  sme2::gemm(ap, np, k, {a.w + c0, co}, nc, a.out + p0 * co + c0, co, e);
}

void conv2dSme(const Conv2dArgs& a) {
  const int64_t pixels = a.n * a.ho * a.wo, k = a.kh * a.kw * a.ci, co = a.co;
  // Whole problem in one tile when its packed A fits (small maps: the weights then stream once); else ~2 MiB tiles.
  const int64_t tile = pixels * k * 4 <= 2 * kSmePackBytes ? pixels
                                                           : std::max<int64_t>(32, kSmePackBytes / (4 * k) / 32 * 32);
  float* ap = detail::scratch(sme2::packedSize(std::min(tile, pixels), k), 3);
  for (int64_t p0 = 0; p0 < pixels; p0 += tile) {
    const int64_t np = std::min(tile, pixels - p0);
    smePack(a, p0, np, ap);
    smeGemm(a, ap, p0, np, 0, co);
  }
}

// conv2d as [P, K] x [K, Co] GEMMs (P = output pixels, K = KH*KW*Ci).
// Small convs (< kParallel MACs): one thread, ~3 MB im2col tiles. Larger ones run a grid of (pixel tile x Co chunk)
// tasks on the pool, single-threaded sgemm per task (one AMX/SME stream per core). The grid shape minimizes the bytes
// the tasks read: every pixel tile streams the whole weight chunk, every Co chunk streams its pixel tiles' im2col rows.
// Big feature maps split pixels (each tile im2col'd in cache right before its sgemm); small maps with big weights
// (ResNet-18's 7x7x512 layers: 9.4 MB of weights for 49 pixels) split Co, after one shared im2col pass.
void conv2d(const Conv2dArgs& a) {
  if (sme2On() && a.act != Act::Tanh && smeWins(a)) return conv2dSme(a);
  const int64_t pixels = a.n * a.ho * a.wo, k = a.kh * a.kw * a.ci, co = a.co;
  const bool pointwise = a.kh == 1 && a.kw == 1 && a.stride == 1 && a.pad == 0;
  const double macs = static_cast<double>(pixels) * static_cast<double>(k) * static_cast<double>(co);
  // (Serial pointwise up to 2e7 / 6e7 MACs measured slower on the M2 Max: MobileNetV2 N=1 2.2 -> 3.3 / 3.6 ms.)
  const double parMacs = kParallelMacs;
  auto gemmTile = [&](const float* col, int64_t ldcol, int64_t p0, int64_t np, int64_t c0, int64_t nc) {
    float* o = a.out + p0 * co + c0;
    detail::sgemm(false, false, np, nc, k, 1.f, col, ldcol, a.w + c0, co, 0.f, o, co);
    epilogue(o, np, nc, co, a.bias + c0, a.res ? a.res + p0 * co + c0 : nullptr, a.act);
  };
  auto tileFused = [&](int64_t p0, int64_t np) {  // im2col of this pixel tile in cache, then its sgemm over all Co
    if (pointwise) return gemmTile(a.x + p0 * a.ci, k, p0, np, 0, co);
    float* buf = detail::scratch(static_cast<size_t>(np * k), 2);
    im2col(a, p0, np, buf);
    gemmTile(buf, k, p0, np, 0, co);
  };
  const int64_t budgetTile = std::max<int64_t>(8, kColBudget / (4 * k)) / 8 * 8;
  if (macs < parMacs) {
    for (int64_t p0 = 0; p0 < pixels; p0 += budgetTile) tileFused(p0, std::min(budgetTile, pixels - p0));
    return;
  }
  // One im2col tile and a big GEMM (small maps, big weights: ResNet-18's 14x14 / 7x7 stages, DCGAN D): one sgemm with
  // Accelerate's own threading beats our task grid (measured iPhone + M2 Max: it splits the work across the AMX / SME
  // units without every task re-streaming a weight slice). Exception: weights > 8 MiB (ResNet-18's 3x3x512x512, 9.4 MB
  // per layer, 33 MB for the stage) don't stay cached between calls, the GEMM is DRAM-bound, and two K halves
  // (contiguous weight rows) on two threads stream faster: iPhone 430 -> 345 us per layer (K in 6 parts: 365 us).
  // DCGAN D's 8 MiB layer stays cached and is 2x slower split. (Tried: weights pre-transposed to [Co, K] for these
  // layers, iPhone 320 -> 310 us per layer, e2e within noise; not kept.)
  if (!pointwise && k >= 1024 && pixels <= budgetTile && 2.0 * macs >= detail::kSmallGemmFlops) {
    constexpr int64_t kKSplit = 2;
    if (static_cast<double>(k) * static_cast<double>(co) * 4.0 <= 8.0 * 1024 * 1024) return tileFused(0, pixels);
    float* col = detail::scratch(static_cast<size_t>(pixels * k), 1);
    im2col(a, 0, pixels, col);
    const int64_t kc = (k + kKSplit - 1) / kKSplit, nk = (k + kc - 1) / kc;
    float* part = detail::scratch(static_cast<size_t>((nk - 1) * pixels * co), 0);
    detail::parallelForSt(nk, [&](int64_t t) {
      const int64_t k0 = t * kc, kn = std::min(kc, k - k0);
      float* dst = t == 0 ? a.out : part + (t - 1) * pixels * co;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, detail::li(pixels), detail::li(co), detail::li(kn), 1.f,
                  col + k0, detail::li(k), a.w + k0 * co, detail::li(co), 0.f, dst, detail::li(co));
    });
    const int64_t total = pixels * co;
    for (int64_t t = 1; t < nk; ++t) {
      const float* p = part + (t - 1) * total;
      int64_t i = 0;
      for (; i + 4 <= total; i += 4) vst1q_f32(a.out + i, vaddq_f32(vld1q_f32(a.out + i), vld1q_f32(p + i)));
      for (; i < total; ++i) a.out[i] += p[i];
    }
    epilogue(a.out, pixels, co, co, a.bias, a.res, a.act);
    return;
  }
  // Grid choice: tp pixel tiles x tc Co chunks ~ kTilesPerThread * threads tasks of >= kMinTileMacs each.
  const int64_t threads = rt::ThreadPool::global().threads();
  const int64_t want = std::max<int64_t>(1, std::min<int64_t>(kTilesPerThread * threads,
                                                               static_cast<int64_t>(macs / kMinTileMacs)));
  int64_t bestTp = 1, bestTc = 1;
  double bestCost = 1e300;
  for (int64_t tc = 1; tc <= want && co / tc >= 16; ++tc) {
    const int64_t tp = std::min<int64_t>((want + tc - 1) / tc, (pixels + 7) / 8);
    // bytes read: weights once per pixel tile, activations once per Co chunk (+ one im2col write/read if shared)
    const double cost = static_cast<double>(tp) * static_cast<double>(k * co) +
                        static_cast<double>(tc) * static_cast<double>(pixels * k) +
                        (tc > 1 && !pointwise ? 2.0 * static_cast<double>(pixels * k) : 0.0);
    if (cost < bestCost * 0.97) bestCost = cost, bestTp = tp, bestTc = tc;
  }
  int64_t tile = std::min(budgetTile, std::max<int64_t>(8, ((pixels + bestTp - 1) / bestTp + 7) / 8 * 8));
  tile = std::min(tile, std::max<int64_t>(8, pixels));
  const int64_t tp = (pixels + tile - 1) / tile;
  if (bestTc == 1) {
    if (tp == 1) return tileFused(0, pixels);
    detail::parallelForSt(tp, [&](int64_t ti) { tileFused(ti * tile, std::min(tile, pixels - ti * tile)); });
    return;
  }
  const int64_t chunk = ((co + bestTc - 1) / bestTc + 15) / 16 * 16, tc = (co + chunk - 1) / chunk;
  const float* col = a.x;
  if (!pointwise) {  // shared im2col (small maps only: tc > 1 implies the activations are small next to the weights)
    float* buf = detail::scratch(static_cast<size_t>(pixels * k), 1);
    if (tp == 1) {
      im2col(a, 0, pixels, buf);
    } else {
      rt::ThreadPool::global().parallelFor(tp, [&](int64_t ti) {
        im2col(a, ti * tile, std::min(tile, pixels - ti * tile), buf + ti * tile * k);
      });
    }
    col = buf;
  }
  detail::parallelForSt(tp * tc, [&](int64_t t) {
    const int64_t ti = t / tc, ci = t % tc, p0 = ti * tile, c0 = ci * chunk;
    gemmTile(col + p0 * k, k, p0, std::min(tile, pixels - p0), c0, std::min(chunk, co - c0));
  });
}

// Depthwise conv: x [N, H, W, C], w [KH, KW, C], bias [C] -> out [N, Ho, Wo, C].
// Tasks = (output row, channel chunk). Channel blocks of 8 (two NEON vectors) are the outer loop of a task: the block's
// weight vectors and bias stay in registers while the row's pixels stream through (XNNPACK-style). 3x3 (stride 1 / 2,
// every MobileNetV2 dw layer) is fully specialized: kernel size, stride and activation are compile-time, so the 18
// weight vectors really live in registers (the runtime-tap version spilled them and ran ~15 GFLOP/s per core on the
// M2 Max); padded rows read a zero row instead of branching; only the first / last output columns check bounds.
const float* zeroRow(int64_t n) {
  thread_local std::vector<float> z;
  if (static_cast<int64_t>(z.size()) < n) z.assign(static_cast<size_t>(n), 0.f);
  return z.data();
}

template <int K, int S, Act A>
void dwRowK(const float* const* rp, const float* w, const float* bias, float* orow, int64_t wd, int64_t c, int64_t c0,
            int64_t c1, int64_t wo, int64_t pad) {
  const int64_t owLo = std::min<int64_t>(wo, (pad + S - 1) / S);
  const int64_t owHi = std::max<int64_t>(owLo, std::min<int64_t>(wo, (wd - K + pad) / S + 1));
  int64_t cb = c0;
  for (; cb + 8 <= c1; cb += 8) {
    float32x4_t w0[K * K], w1[K * K];
    for (int t = 0; t < K * K; ++t) {
      w0[t] = vld1q_f32(w + t * c + cb);
      w1[t] = vld1q_f32(w + t * c + cb + 4);
    }
    const float32x4_t b0 = vld1q_f32(bias + cb), b1 = vld1q_f32(bias + cb + 4);
    auto edge = [&](int64_t ow) {
      float32x4_t a0 = b0, a1 = b1;
      const int64_t iw0 = ow * S - pad;
      for (int y = 0; y < K; ++y)
        for (int xk = 0; xk < K; ++xk) {
          const int64_t iw = iw0 + xk;
          if (iw < 0 || iw >= wd) continue;
          const float* s = rp[y] + iw * c + cb;
          a0 = vfmaq_f32(a0, vld1q_f32(s), w0[y * K + xk]);
          a1 = vfmaq_f32(a1, vld1q_f32(s + 4), w1[y * K + xk]);
        }
      vst1q_f32(orow + ow * c + cb, actV(a0, A));
      vst1q_f32(orow + ow * c + cb + 4, actV(a1, A));
    };
    for (int64_t ow = 0; ow < owLo; ++ow) edge(ow);
    const float* r[K];
    for (int y = 0; y < K; ++y) r[y] = rp[y] + (owLo * S - pad) * c + cb;
    float* o = orow + owLo * c + cb;
    int64_t ow = owLo;
    for (; ow + 2 <= owHi; ow += 2) {  // two pixels: independent FMA chains (+ shared columns when S == 1)
      float32x4_t a0 = b0, a1 = b1, p0 = b0, p1 = b1;
      for (int y = 0; y < K; ++y)
        for (int xk = 0; xk < K; ++xk) {
          const float* s = r[y] + xk * c;
          a0 = vfmaq_f32(a0, vld1q_f32(s), w0[y * K + xk]);
          a1 = vfmaq_f32(a1, vld1q_f32(s + 4), w1[y * K + xk]);
          p0 = vfmaq_f32(p0, vld1q_f32(s + S * c), w0[y * K + xk]);
          p1 = vfmaq_f32(p1, vld1q_f32(s + S * c + 4), w1[y * K + xk]);
        }
      vst1q_f32(o, actV(a0, A));
      vst1q_f32(o + 4, actV(a1, A));
      vst1q_f32(o + c, actV(p0, A));
      vst1q_f32(o + c + 4, actV(p1, A));
      for (int y = 0; y < K; ++y) r[y] += 2 * S * c;
      o += 2 * c;
    }
    for (; ow < owHi; ++ow) {
      float32x4_t a0 = b0, a1 = b1;
      for (int y = 0; y < K; ++y)
        for (int xk = 0; xk < K; ++xk) {
          a0 = vfmaq_f32(a0, vld1q_f32(r[y] + xk * c), w0[y * K + xk]);
          a1 = vfmaq_f32(a1, vld1q_f32(r[y] + xk * c + 4), w1[y * K + xk]);
        }
      vst1q_f32(o, actV(a0, A));
      vst1q_f32(o + 4, actV(a1, A));
      for (int y = 0; y < K; ++y) r[y] += S * c;
      o += c;
    }
    for (int64_t ow = owHi; ow < wo; ++ow) edge(ow);
  }
  for (; cb < c1; ++cb)  // channel tail (C % 8)
    for (int64_t ow = 0; ow < wo; ++ow) {
      float acc = bias[cb];
      for (int y = 0; y < K; ++y)
        for (int xk = 0; xk < K; ++xk) {
          const int64_t iw = ow * S - pad + xk;
          if (iw >= 0 && iw < wd) acc += rp[y][iw * c + cb] * w[(y * K + xk) * c + cb];
        }
      orow[ow * c + cb] = actS(acc, A);
    }
}

// Generic k x k, any stride (scalar per channel; no model uses it today).
void dwRowGeneric(const float* xb, const float* w, const float* bias, float* orow, int64_t h, int64_t wd, int64_t c,
                  int64_t c0, int64_t c1, int64_t kh, int64_t kw, int64_t wo, int64_t ih0, int64_t stride, int64_t pad,
                  Act act) {
  for (int64_t cb = c0; cb < c1; ++cb)
    for (int64_t ow = 0; ow < wo; ++ow) {
      float acc = bias[cb];
      for (int64_t y = 0; y < kh; ++y) {
        const int64_t ih = ih0 + y;
        if (ih < 0 || ih >= h) continue;
        for (int64_t xk = 0; xk < kw; ++xk) {
          const int64_t iw = ow * stride - pad + xk;
          if (iw < 0 || iw >= wd) continue;
          acc += xb[(ih * wd + iw) * c + cb] * w[(y * kw + xk) * c + cb];
        }
      }
      orow[ow * c + cb] = actS(acc, act);
    }
}

template <Act A>
void dwconv2dT(const float* x, const float* w, const float* bias, float* out, int64_t n, int64_t h, int64_t wd,
               int64_t c, int64_t kh, int64_t kw, int64_t ho, int64_t wo, int64_t stride, int64_t pad) {
  const bool k3 = kh == 3 && kw == 3 && (stride == 1 || stride == 2);
  const int64_t rows = n * ho;
  const double work = static_cast<double>(rows) * static_cast<double>(wo * c * kh * kw);
  constexpr double kDwSerial = 2e5;  // (1e6 measured slower on the iPhone)
  const bool par = work >= kDwSerial;
  // Channel chunks (multiples of 8) so that small feature maps (7x7, 14x14) still give every thread several tasks.
  int64_t chunks = 1;
  if (par) {
    const int64_t threads = rt::ThreadPool::global().threads();
    constexpr int64_t kDwTasksPerThread = 4;
    const int64_t want = (kDwTasksPerThread * threads + rows - 1) / rows;
    chunks = std::max<int64_t>(1, std::min(want, c / 32));
  }
  const int64_t blocks8 = (c + 7) / 8, perChunk = (blocks8 + chunks - 1) / chunks * 8;
  chunks = (c + perChunk - 1) / perChunk;
  auto task = [&](int64_t t) {
    const int64_t r = t / chunks, ch = t % chunks;
    const int64_t c0 = ch * perChunk, c1 = std::min(c, c0 + perChunk);
    const int64_t b = r / ho, oh = r % ho, ih0 = oh * stride - pad;
    float* orow = out + (b * ho + oh) * wo * c;
    const float* xb = x + b * h * wd * c;
    if (!k3) {
      dwRowGeneric(xb, w, bias, orow, h, wd, c, c0, c1, kh, kw, wo, ih0, stride, pad, A);
      return;
    }
    const float* rp[3];
    for (int y = 0; y < 3; ++y) {
      const int64_t ih = ih0 + y;
      rp[y] = ih >= 0 && ih < h ? xb + ih * wd * c : zeroRow(wd * c);
    }
    if (stride == 1)
      dwRowK<3, 1, A>(rp, w, bias, orow, wd, c, c0, c1, wo, pad);
    else
      dwRowK<3, 2, A>(rp, w, bias, orow, wd, c, c0, c1, wo, pad);
  };
  const int64_t tasks = rows * chunks;
  if (!par) {
    for (int64_t t = 0; t < tasks; ++t) task(t);
  } else {
    rt::ThreadPool::global().parallelFor(tasks, task);
  }
}

void dwconv2d(const float* x, const float* w, const float* bias, float* out, int64_t n, int64_t h, int64_t wd,
              int64_t c, int64_t kh, int64_t kw, int64_t ho, int64_t wo, int64_t stride, int64_t pad, Act act) {
  switch (act) {
    case Act::Relu6: return dwconv2dT<Act::Relu6>(x, w, bias, out, n, h, wd, c, kh, kw, ho, wo, stride, pad);
    case Act::Relu: return dwconv2dT<Act::Relu>(x, w, bias, out, n, h, wd, c, kh, kw, ho, wo, stride, pad);
    default: return dwconv2dT<Act::None>(x, w, bias, out, n, h, wd, c, kh, kw, ho, wo, stride, pad);
  }
}

// MobileNetV2 inverted-residual block in one kernel: x [N, H, W, Ci] -> 1x1 expand (we [Ci, Ce], be) + ReLU6 ->
// depthwise 3x3 stride s, pad 1 (wd [3, 3, Ce], bd) + ReLU6 -> 1x1 project (wp [Ce, Co], bp) [+ x] -> out [N, Ho, Wo, Co].
// Fused path: one task per (image, band of R output rows): the band's expanded input rows (R*s + 2 - s with the halo)
// and its dw output stay in per-thread scratch (cache) instead of making two round trips through memory as full
// [N, H, W, Ce] tensors (112x112x96 at N=8: 38 MB each way). The halo rows are expanded twice, so small maps (where
// R would be 1-2 rows and the expanded tensor fits in cache anyway) run the three steps unfused, still in this kernel.
constexpr int64_t kBandBytes = int64_t{384} << 10;  // expanded band budget per task (L2-resident)

void invertedResidual(const float* x, const float* we, const float* be, const float* wdw, const float* bdw,
                      const float* wp, const float* bp, float* out, int64_t n, int64_t h, int64_t w, int64_t ci,
                      int64_t ce, int64_t co, int64_t ho, int64_t wo, int64_t s, bool res) {
  const int64_t threads = rt::ThreadPool::global().threads();
  // Band height: >= kTilesPerThread * threads tasks, expanded band within kBandBytes.
  const int64_t rows = n * ho;
  int64_t r = std::max<int64_t>(1, rows / (kTilesPerThread * threads));
  r = std::min(r, std::max<int64_t>(1, (kBandBytes / (4 * w * ce) - 2 + s) / s));
  r = std::min(r, ho);
  const double halo = static_cast<double>((r - 1) * s + 3) / static_cast<double>(r * s);
  if (halo > 1.5 || static_cast<double>(n * h * w) * static_cast<double>(ci * ce) < kParallelMacs) {
    // Unfused: expand -> dw -> project with the standalone kernels, intermediates in this thread's scratch.
    thread_local std::vector<float> eBuf, dBuf;
    if (eBuf.size() < static_cast<size_t>(n * h * w * ce)) eBuf.resize(static_cast<size_t>(n * h * w * ce));
    if (dBuf.size() < static_cast<size_t>(n * ho * wo * ce)) dBuf.resize(static_cast<size_t>(n * ho * wo * ce));
    conv2d({x, we, be, nullptr, eBuf.data(), n, h, w, ci, 1, 1, ce, h, w, 1, 0, Act::Relu6});
    dwconv2dT<Act::Relu6>(eBuf.data(), wdw, bdw, dBuf.data(), n, h, w, ce, 3, 3, ho, wo, s, 1);
    conv2d({dBuf.data(), wp, bp, res ? x : nullptr, out, n, ho, wo, ce, 1, 1, co, ho, wo, 1, 0, Act::None});
    return;
  }
  const int64_t bands = (ho + r - 1) / r, maxIn = (r - 1) * s + 3;
  detail::parallelForSt(n * bands, [&](int64_t t) {
    const int64_t b = t / bands, oh0 = (t % bands) * r, nr = std::min(r, ho - oh0);
    const int64_t ihA = oh0 * s - 1, ihB = (oh0 + nr - 1) * s + 1;  // input rows [ihA, ihB] incl. halo
    const int64_t v0 = std::max<int64_t>(0, ihA), v1 = std::min<int64_t>(h - 1, ihB);
    float* e = detail::scratch(static_cast<size_t>(maxIn * w * ce), 0);
    float* d = detail::scratch(static_cast<size_t>(r * wo * ce), 1);
    // expand the valid input rows (contiguous in NHWC) into e rows [v0 - ihA, ...]
    float* ev = e + (v0 - ihA) * w * ce;
    const int64_t m = (v1 - v0 + 1) * w;
    detail::sgemm(false, false, m, ce, ci, 1.f, x + ((b * h + v0) * w) * ci, ci, we, ce, 0.f, ev, ce);
    epilogue(ev, m, ce, ce, be, nullptr, Act::Relu6);
    const float* zero = zeroRow(w * ce);
    for (int64_t q = 0; q < nr; ++q) {
      const float* rp[3];
      for (int y = 0; y < 3; ++y) {
        const int64_t ih = (oh0 + q) * s - 1 + y;
        rp[y] = ih >= 0 && ih < h ? e + (ih - ihA) * w * ce : zero;
      }
      if (s == 1)
        dwRowK<3, 1, Act::Relu6>(rp, wdw, bdw, d + q * wo * ce, w, ce, 0, ce, wo, 1);
      else
        dwRowK<3, 2, Act::Relu6>(rp, wdw, bdw, d + q * wo * ce, w, ce, 0, ce, wo, 1);
    }
    const int64_t p0 = (b * ho + oh0) * wo, np = nr * wo;
    float* o = out + p0 * co;
    detail::sgemm(false, false, np, co, ce, 1.f, d, ce, wp, co, 0.f, o, co);
    epilogue(o, np, co, co, bp, res ? x + p0 * ci : nullptr, Act::None);
  });
}

// out[b, oh, ow, co] = bias[co] + Σ y[b, (oh+p-kh)/s, (ow+p-kw)/s, (kh*K + kw)*Co + co] over the taps whose
// division is exact and in range, then act. y [B, H, W, K*K*Co], out [B, Ho, Wo, Co].
void convt2dGather(const float* y, const float* bias, float* out, int64_t b, int64_t h, int64_t w, int64_t co,
                   int64_t k, int64_t ho, int64_t wo, int64_t s, int64_t p, Act act) {
  const int64_t ldy = k * k * co;
  auto row = [&](int64_t r) {
    const int64_t bi = r / ho, oh = r % ho;
    for (int64_t ow = 0; ow < wo; ++ow) {
      float* o = out + ((bi * ho + oh) * wo + ow) * co;
      std::memcpy(o, bias, static_cast<size_t>(co) * sizeof(float));
      for (int64_t kh = (oh + p) % s; kh < k; kh += s) {
        const int64_t ih = (oh + p - kh) / s;
        if (oh + p - kh < 0 || ih >= h) continue;
        for (int64_t kw = (ow + p) % s; kw < k; kw += s) {
          const int64_t iw = (ow + p - kw) / s;
          if (ow + p - kw < 0 || iw >= w) continue;
          const float* src = y + ((bi * h + ih) * w + iw) * ldy + (kh * k + kw) * co;
          int64_t c = 0;
          for (; c + 4 <= co; c += 4) vst1q_f32(o + c, vaddq_f32(vld1q_f32(o + c), vld1q_f32(src + c)));
          for (; c < co; ++c) o[c] += src[c];
        }
      }
      if (act != Act::Tanh) {
        int64_t c = 0;
        for (; c + 4 <= co; c += 4) vst1q_f32(o + c, actV(vld1q_f32(o + c), act));
        for (; c < co; ++c) o[c] = actS(o[c], act);
      }
    }
    if (act == Act::Tanh) {  // the whole output row at once through vvtanhf (scalar tanh dominated small-Co rows)
      float* orow = out + (bi * ho + oh) * wo * co;
      const int n = static_cast<int>(wo * co);
      vvtanhf(orow, orow, &n);
    }
  };
  const int64_t rows = b * ho;
  // Work per output pixel ~ (gather of ceil(k/s)^2 taps x co) + fixed per-pixel overhead: parallelize on pixels, not
  // on co (co = 3 at DCGAN G's last layer ran single-threaded with a co-based estimate).
  if (static_cast<double>(rows) * static_cast<double>(wo) * static_cast<double>(16 + co * 4) < kParallelMacs) {
    for (int64_t r = 0; r < rows; ++r) row(r);
  } else {
    rt::ThreadPool::global().parallelFor(rows, row);
  }
}


// ---- Direct NEON conv for few input channels, NCHW in -> NHWC out (network stems: MobileNetV2 3x3 s2 3->32, DCGAN D
// 4x4 s2 3->64) ------------------------------------------------------------------------------------------------------
// out[n, oy, ox, co] = act(b[co] + Σ w[ky, kx, c, co] · x[n, c, oy·s - p + ky, ox·s - p + kx]). Tasks = output rows. Per
// task the Ci·KH input rows it needs are copied zero-padded into scratch, then 4 output pixels x 16 output channels are
// accumulated in 16 registers, one FMLA-by-lane per (tap, pixel, channel vector): the input vector holds the tap's value
// for the 4 pixels (stride-2 deinterleave via vld2), the 4 weight vectors come from L1. Replaces im2col (K = 27 / 48)
// + a short-K sgemm, and the NCHW -> NHWC pass in front of it. Co must be a multiple of 16.
struct SmallCiConv {
  const float* x;  // [N, Ci, H, W]
  const float* w;  // [KH, KW, Ci, Co]
  const float* bias;
  float* out;  // [N, Ho, Wo, Co]
  int64_t n, ci, h, wd, kh, kw, co, ho, wo, stride, pad;
  Act act;
};

template <int S>
inline float32x4_t loadTap(const float* r) {
  if constexpr (S == 2) return vld2q_f32(r).val[0];
  return vld1q_f32(r);
}

template <int S>
void smallCiRow(const SmallCiConv& p, const float* const* rows, float* orow) {
  const int64_t taps = p.kh * p.kw;
  for (int64_t c0 = 0; c0 < p.co; c0 += 16) {
    const float32x4_t b0 = vld1q_f32(p.bias + c0), b1 = vld1q_f32(p.bias + c0 + 4), b2 = vld1q_f32(p.bias + c0 + 8),
                      b3 = vld1q_f32(p.bias + c0 + 12);
    for (int64_t ox = 0; ox < p.wo; ox += 4) {  // rows are padded to a multiple of 4 output pixels
      float32x4_t a[4][4];
      for (int q = 0; q < 4; ++q) a[q][0] = b0, a[q][1] = b1, a[q][2] = b2, a[q][3] = b3;
      for (int64_t c = 0; c < p.ci; ++c)
        for (int64_t ky = 0; ky < p.kh; ++ky) {
          const float* r = rows[c * p.kh + ky] + ox * S;
          const float* wr = p.w + (ky * p.kw * p.ci + c) * p.co + c0;
          for (int64_t kx = 0; kx < p.kw; ++kx, wr += p.ci * p.co) {
            const float32x4_t in = loadTap<S>(r + kx);
            const float32x4_t w0 = vld1q_f32(wr), w1 = vld1q_f32(wr + 4), w2 = vld1q_f32(wr + 8),
                              w3 = vld1q_f32(wr + 12);
#define LK_FMA_LANE(L)                               \
  a[L][0] = vfmaq_laneq_f32(a[L][0], w0, in, L); \
  a[L][1] = vfmaq_laneq_f32(a[L][1], w1, in, L); \
  a[L][2] = vfmaq_laneq_f32(a[L][2], w2, in, L); \
  a[L][3] = vfmaq_laneq_f32(a[L][3], w3, in, L);
            LK_FMA_LANE(0) LK_FMA_LANE(1) LK_FMA_LANE(2) LK_FMA_LANE(3)
#undef LK_FMA_LANE
          }
        }
      (void)taps;
      const int64_t cnt = std::min<int64_t>(4, p.wo - ox);
      for (int q = 0; q < cnt; ++q) {
        float* o = orow + (ox + q) * p.co + c0;
        for (int v = 0; v < 4; ++v) vst1q_f32(o + 4 * v, actV(a[q][v], p.act));
      }
    }
  }
}

void smallCiConv(const SmallCiConv& p) {
  // Padded row: p.pad zeros, the W inputs, zeros up to the last tap of the last (rounded-up) pixel group + 8 (vld2).
  const int64_t wo4 = (p.wo + 3) / 4 * 4;
  const int64_t rowLen = std::max<int64_t>(p.pad + p.wd, (wo4 - 1) * p.stride + p.kw + 8) + 8;
  auto task = [&](int64_t t) {
    const int64_t img = t / p.ho, oy = t % p.ho;
    float* buf = detail::scratch(static_cast<size_t>(p.ci * p.kh * rowLen), 2);
    const float* zero = zeroRow(rowLen);
    const float* rows[64];
    for (int64_t c = 0; c < p.ci; ++c)
      for (int64_t ky = 0; ky < p.kh; ++ky) {
        const int64_t y = oy * p.stride - p.pad + ky;
        if (y < 0 || y >= p.h) {
          rows[c * p.kh + ky] = zero;
          continue;
        }
        float* d = buf + (c * p.kh + ky) * rowLen;
        std::memset(d, 0, static_cast<size_t>(p.pad) * sizeof(float));
        std::memcpy(d + p.pad, p.x + ((img * p.ci + c) * p.h + y) * p.wd, static_cast<size_t>(p.wd) * sizeof(float));
        std::memset(d + p.pad + p.wd, 0, static_cast<size_t>(rowLen - p.pad - p.wd) * sizeof(float));
        rows[c * p.kh + ky] = d;
      }
    float* orow = p.out + (img * p.ho + oy) * p.wo * p.co;
    if (p.stride == 2)
      smallCiRow<2>(p, rows, orow);
    else
      smallCiRow<1>(p, rows, orow);
  };
  const int64_t rowsTotal = p.n * p.ho;
  const double macs = static_cast<double>(rowsTotal * p.wo * p.co) * static_cast<double>(p.ci * p.kh * p.kw);
  if (macs < 2e6) {
    for (int64_t t = 0; t < rowsTotal; ++t) task(t);
  } else {
    rt::ThreadPool::global().parallelFor(rowsTotal, task);
  }
}

// ---- Channel-planar padded layout ("cnhw", the iPhone SME2 path of ResNet-18; docs/models/resnet18.md) ----------------
// A tensor [C, N, Hp, Wp] holds image n's channel c at plane c, rows / cols 1..H / 1..W (border 1, plus `extra` more
// zero rows / cols at the bottom / right); every cell outside the interior is zero. A 3x3 / 1x1 stride-1 conv is then a
// GEMM C[Co, columns] = W^T[Co, K] · B[K, columns] whose B row kk = (tap, c) is the input plane at a constant shift,
// read in place (no im2col, no per-call packing): out flat index o reads x + rowOff[kk] + o. Columns that land on
// border cells are computed and zeroed by a column mask. Stride 2 reads 2x2 phase planes (phases[a][b][i][j] =
// padded input [2i + a][2j + b]) laid out in the output geometry, so it is again a constant shift per tap.
struct PlanarConv {
  const float* x;   // s1: [Ci][N][Hpo][Wpo]; s2: [4][Ci][N][Hpo][Wpo]
  const float* wp;  // [ceil(Co / 32)][K][32], kk = (ky * KW + kx) * Ci + c
  const float* bias;
  const float* res;  // [Co][N][Hpo][Wpo] or null
  float* out;        // [Co][N][Hpo][Wpo]
  int64_t ci, co, n, hpo, wpo, extra, kh, kw, pad, pin, stride;
  Act act;
};

// 0/1 per column o0 + j (interior cells of the output geometry), cached per geometry on this thread (a model cycles
// through a few geometries; recomputing it per call cost ~3 ns per column, ~0.3 ms for ResNet-18's N = 8 stem).
const float* interiorMask(int64_t n, int64_t hpo, int64_t wpo, int64_t extra, int64_t o0, int64_t cols) {
  struct Entry {
    int64_t key[6];
    std::vector<float> m;
  };
  thread_local std::vector<Entry> cache;
  const int64_t key[6] = {n, hpo, wpo, extra, o0, cols};
  for (const Entry& e : cache)
    if (std::equal(key, key + 6, e.key)) return e.m.data();
  if (cache.size() >= 16) cache.erase(cache.begin());
  Entry e;
  std::copy(key, key + 6, e.key);
  e.m.assign(static_cast<size_t>(cols), 0.f);
  const int64_t ho = hpo - 2 - extra, wo = wpo - 2 - extra, img = hpo * wpo;
  for (int64_t j = 0; j < cols; ++j) {
    const int64_t o = o0 + j, r = (o % img) / wpo, c = o % wpo;
    e.m[static_cast<size_t>(j)] = r >= 1 && r <= ho && c >= 1 && c <= wo ? 1.f : 0.f;
  }
  cache.push_back(std::move(e));
  return cache.back().m.data();
}

void planarConv(const PlanarConv& p) {
  const int64_t img = p.hpo * p.wpo, plane = p.n * img, k = p.kh * p.kw * p.ci;
  const int64_t ho = p.hpo - 2 - p.extra, wo = p.wpo - 2 - p.extra;
  const int64_t o0 = p.wpo + 1, o1 = (p.n - 1) * img + ho * p.wpo + wo + 1, cols = o1 - o0;
  // Input planes whose stride is a multiple of 2 KB (ResNet-18's 14x14 stage at N = 8: 8 x 16 x 16 floats) put every B
  // row of a column block at the same cache-set offset: those convs ran ~5x slower than their FLOPs (iPhone, 3.7 ms
  // for the stride-2 conv into the stage). Such inputs are first copied to planes padded by 48 floats.
  const float* x = p.x;
  int64_t ps = plane;
  if (plane % 512 == 0 && k >= 256) {
    const int64_t np = (p.stride == 1 ? 1 : 4) * p.ci;
    ps = plane + 48;
    float* buf = detail::scratch(static_cast<size_t>(np * ps), 7);
    auto cp = [&](int64_t i) {
      std::memcpy(buf + i * ps, p.x + i * plane, static_cast<size_t>(plane) * sizeof(float));
    };
    if (np * plane >= (int64_t{1} << 18)) {
      rt::ThreadPool::global().parallelFor(np, cp);
    } else {
      for (int64_t i = 0; i < np; ++i) cp(i);
    }
    x = buf;
  }
  thread_local std::vector<int64_t> off;
  off.resize(static_cast<size_t>(k));
  for (int64_t ky = 0; ky < p.kh; ++ky)
    for (int64_t kx = 0; kx < p.kw; ++kx)
      for (int64_t c = 0; c < p.ci; ++c) {
        int64_t v;
        if (p.stride == 1) {
          v = c * ps + (ky - p.pad) * p.wpo + (kx - p.pad);
        } else {
          const int64_t r = ky - p.pad + p.pin, q = kx - p.pad + p.pin;
          v = ((r % 2) * 2 + q % 2) * p.ci * ps + c * ps + (r / 2 - 1) * p.wpo + (q / 2 - 1);
        }
        off[static_cast<size_t>((ky * p.kw + kx) * p.ci + c)] = v;
      }
  const float* mask = interiorMask(p.n, p.hpo, p.wpo, p.extra, o0, cols);
  for (int64_t c = 0; c < p.co; ++c) {  // cells outside [o0, o1) are never computed: zero them
    float* pl = p.out + c * plane;
    std::memset(pl, 0, static_cast<size_t>(o0) * sizeof(float));
    std::memset(pl + o1, 0, static_cast<size_t>(plane - o1) * sizeof(float));
  }
  // (Tried for the 7x7 stem, K = 147: B gathered into aligned column chunks by the pool, pipelined with plain mode-0
  // SME2 GEMMs on a P core: iPhone N = 1 390 vs 288 us, N = 8 3.2 vs 2.5 ms with the row-offset B below. Dropped.)
  if (sme2::available()) {
    sme2::BOperand bo{x + o0, 0};
    if (k > 1 && p.kh * p.kw == 1)  // 1x1: B rows are evenly spaced planes: the plain row-major operand (no offset table)
      bo = {x + o0 + off[0], off[1] - off[0]};
    else
      bo.rowOff = off.data();
    sme2::Epilogue e;
    e.bias = p.bias;
    e.biasPerRow = true;
    e.res = p.res ? p.res + o0 : nullptr;
    e.ldres = plane;
    e.act = p.act == Act::Relu ? sme2::Act::Relu : p.act == Act::Relu6 ? sme2::Act::Relu6
          : p.act == Act::Leaky ? sme2::Act::Leaky : sme2::Act::None;
    e.leaky = kLeakySlope;
    e.colMask = mask;
    sme2::gemm(p.wp, p.co, k, bo, cols, p.out + o0, plane, e);
    return;
  }
  // Accelerate (no SME2: Mac, tests; on the iPhone this grid of single-threaded sgemms was 2-2.5x slower than the one
  // SME2 stream on every ResNet-18 layer, 2026-11-02): tasks = (column chunk, group of 32-row weight panels)
  // on the pool, each gathers its B block [K, chunk] and runs single-threaded sgemms (one AMX / SME stream per core).
  const int64_t threads = rt::ThreadPool::global().threads();
  const int64_t panels = (p.co + 31) / 32;
  int64_t chunk = std::max<int64_t>(64, (cols / (3 * threads) + 31) / 32 * 32);
  chunk = std::min<int64_t>(chunk, std::max<int64_t>(64, (int64_t{1} << 20) / (4 * k) / 32 * 32));  // B block <= 1 MB
  const int64_t ncc = (cols + chunk - 1) / chunk;
  const int64_t ngrp = std::min(panels, std::max<int64_t>(1, (2 * threads + ncc - 1) / ncc));
  const int64_t ppg = (panels + ngrp - 1) / ngrp;
  const int64_t* offp = off.data();  // (off is thread_local: the workers must not name it)
  auto task = [&](int64_t t) {
    const int64_t j0 = (t / ngrp) * chunk, nc = std::min(chunk, cols - j0), g0 = (t % ngrp) * ppg;
    float* bm = detail::scratch(static_cast<size_t>(k * nc), 2);
    for (int64_t kk = 0; kk < k; ++kk)
      std::memcpy(bm + kk * nc, x + o0 + j0 + offp[kk], static_cast<size_t>(nc) * sizeof(float));
    for (int64_t pb = g0; pb < std::min(panels, g0 + ppg); ++pb) {
      const int64_t rows = std::min<int64_t>(32, p.co - pb * 32);
      float* c = p.out + pb * 32 * plane + o0 + j0;
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, detail::li(rows), detail::li(nc), detail::li(k), 1.f,
                  p.wp + pb * 32 * k, 32, bm, detail::li(nc), 0.f, c, detail::li(plane));
      for (int64_t r = 0; r < rows; ++r) {
        float* cr = c + r * plane;
        const float* rr = p.res ? p.res + (pb * 32 + r) * plane + o0 + j0 : nullptr;
        const float b = p.bias[pb * 32 + r];
        for (int64_t j = 0; j < nc; ++j) cr[j] = actS(cr[j] + b + (rr ? rr[j] : 0.f), p.act) * mask[j0 + j];
      }
    }
  };
  if (ncc * ngrp == 1) return task(0);
  detail::parallelForSt(ncc * ngrp, task);
}

// ---- 2-D Winograd F(2x2, 3x3) in the cnhw layout (3x3 stride-1 pad-1 convs of the planar ResNet-18 build) -----------
// An output tile (2x2 pixels at (2ty, 2tx)) = AT · M · A with M_p[co, tile] = Σ_c U_p[co, c] · V_p[c, tile] over the 16
// points p = 4a + b; V = BT · d · B of the tile's 4x4 input patch d (padded rows / cols 2ty..2ty+3 / 2tx..2tx+3: the cnhw
// zero border is the conv's padding, so the transform never branches); U = G · w · GT (ops.pack_wino_weights_cnhw:
// [16][ceil(Co / 32)][Ci][32], the SME2 A panels of each point). 16 GEMMs [Co, Ci] x [Ci, tiles] (B row-major, read in
// place) replace the 9-tap GEMM: 2.25x fewer multiplications. BT = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]],
// AT = [[1,1,1,0],[0,1,-1,1]] (= winograd.transforms(2, 3, [0, 1, -1])).
// Tiles are processed in chunks through a ring of buffers, three stages per chunk: 1) input transform (items = channel
// blocks), 2) the 16 GEMMs (one SME2 stream: the P cluster has one SME unit, so one thread at a time), 3) output transform
// + bias / residual / activation into the cnhw output, border included (items = output-channel blocks). Every pool thread
// loops over the stages, P cores preferring the GEMM, so the transforms of chunks c - 1 / c + 1 overlap the GEMM of c.
// Tried F(4x4, 3x3) (36 points; per-layer and run-time F(2)/F(4) selection): its GEMMs take half the time, but the 6x6
// transforms cost ~1.8x more per output on the pool (E cores): N = 8 64ch 970 -> 945 us, 128ch 690 -> 830, 256ch
// 730 -> 957; N = 1 256ch 162 -> 137 us per conv alone, but in-model N = 1 3.06 -> 3.16 ms (the 2.25x larger weights
// stream from DRAM). Dropped (iPhone, 2026-11-02).
// (iPhone, 256ch 14x14 N = 8: pipelined 1.6 ms vs 2.0 with parallel transforms around a GEMM on the caller per chunk,
// 3.3 all on one thread; thermal serious, 2026-11-02.)
// Winograd from 512 2x2 tiles, or from 64 in batches of >= 4 images. iPhone per-conv A/B (2026-11-02, nominal, direct
// planar SME2 vs Winograd, us, weights hot): N = 1: 64ch 56x56 148 -> 130, 128ch 28x28 158 -> 134, 256ch 14x14
// 162 -> 145, 512ch 7x7 (16 tiles) 147-183 -> 207; N = 8: 64ch 1450 -> 960, 128ch 1260 -> 690, 256ch 1550 -> 730,
// 512ch 1500-1750 -> 890. But in-model at N = 1 Winograd on the 28x28 / 14x14 stages too was slower (2.79 -> 2.99 ms):
// U is 1.8x the direct weights and streams from DRAM each call, so N = 1 keeps it to the 56x56 stage.
constexpr int64_t kWinoMinTiles = 512;

struct WinoPlanar {
  const float* x;  // [Ci][N][Hp][Wp] (border 1, no extra)
  const float* u;
  const float* bias;
  const float* res;  // like out, or null
  float* out;        // [Co][N][Hp][Wp]
  int64_t ci, co, n, hp, wp;
  Act act;
};

struct WinoGeom2 {
  int64_t h, w, th, tw, img, plane, tiles;
  explicit WinoGeom2(const WinoPlanar& p)
      : h(p.hp - 2), w(p.wp - 2), th((h + 1) / 2), tw((w + 1) / 2), img(p.hp * p.wp), plane(p.n * p.hp * p.wp),
        tiles(p.n * th * tw) {}
};

// V_p (4 tiles in the lanes) from the patch d[row][col], stored at out + p * pst.
inline void winoInV(const float32x4_t (&d)[4][4], float* out, int64_t pst) {
  float32x4_t t[4][4];
  for (int j = 0; j < 4; ++j) {
    t[0][j] = vsubq_f32(d[0][j], d[2][j]);
    t[1][j] = vaddq_f32(d[1][j], d[2][j]);
    t[2][j] = vsubq_f32(d[2][j], d[1][j]);
    t[3][j] = vsubq_f32(d[3][j], d[1][j]);
  }
  for (int a = 0; a < 4; ++a) {
    float* o = out + (4 * a) * pst;
    vst1q_f32(o, vsubq_f32(t[a][0], t[a][2]));
    vst1q_f32(o + pst, vaddq_f32(t[a][1], t[a][2]));
    vst1q_f32(o + 2 * pst, vsubq_f32(t[a][2], t[a][1]));
    vst1q_f32(o + 3 * pst, vsubq_f32(t[a][3], t[a][1]));
  }
}

// Input transform of channel c for tiles [t0, t0 + tc): V_p[c][t - t0] at v + p * pst + (t - t0).
void winoIn2(const WinoPlanar& p, const WinoGeom2& g, int64_t c, int64_t t0, int64_t tc, float* v, int64_t pst) {
  const float* zero = zeroRow(p.wp + 16);
  const float* xc = p.x + c * g.plane;
  const int64_t per = g.th * g.tw;
  for (int64_t t = t0; t < t0 + tc;) {
    const int64_t im = t / per, rem = t % per, ty = rem / g.tw, tx0 = rem % g.tw;
    const int64_t tx1 = std::min(g.tw, tx0 + (t0 + tc - t));
    const float* rows[4];
    for (int r = 0; r < 4; ++r) {
      const int64_t y = 2 * ty + r;
      rows[r] = y < p.hp ? xc + im * g.img + y * p.wp : zero;
    }
    float* vj = v + (t - t0);
    int64_t tx = tx0;
    float32x4_t d[4][4];
    for (; tx + 4 <= tx1 && 2 * tx + 10 <= p.wp; tx += 4, vj += 4) {
      for (int r = 0; r < 4; ++r) {
        const float32x4x2_t a = vld2q_f32(rows[r] + 2 * tx), b = vld2q_f32(rows[r] + 2 * tx + 2);
        d[r][0] = a.val[0], d[r][1] = a.val[1], d[r][2] = b.val[0], d[r][3] = b.val[1];
      }
      winoInV(d, vj, pst);
    }
    for (; tx < tx1; tx += 4, vj += 4) {
      // Tail (row end / small maps): the 4 rows' columns 2tx .. 2tx + 9 copied zero-padded, then the vector path. All 4
      // lanes are stored: lanes past the segment land in V's padding columns or are overwritten by the next segment.
      float buf[4][12];
      const int64_t c0 = 2 * tx, nc = std::min<int64_t>(10, p.wp - c0);
      for (int r = 0; r < 4; ++r) {
        for (int64_t j = 0; j < 12; ++j) buf[r][j] = j < nc ? rows[r][c0 + j] : 0.f;
        const float32x4x2_t a = vld2q_f32(buf[r]), b = vld2q_f32(buf[r] + 2);
        d[r][0] = a.val[0], d[r][1] = a.val[1], d[r][2] = b.val[0], d[r][3] = b.val[1];
      }
      winoInV(d, vj, pst);
    }
    t += tx1 - tx0;
  }
}

// Y (4 tiles in the lanes) from M_p at m + p * pst: y[i][j] = output pixel (2ty + i, 2tx + j) before the epilogue.
inline void winoOutY(const float* m, int64_t pst, float32x4_t (&y)[2][2]) {
  float32x4_t r0[4], r1[4];
  for (int b = 0; b < 4; ++b) {
    const float32x4_t m0 = vld1q_f32(m + b * pst), m1 = vld1q_f32(m + (4 + b) * pst);
    const float32x4_t m2 = vld1q_f32(m + (8 + b) * pst), m3 = vld1q_f32(m + (12 + b) * pst);
    r0[b] = vaddq_f32(vaddq_f32(m0, m1), m2);
    r1[b] = vaddq_f32(vsubq_f32(m1, m2), m3);
  }
  y[0][0] = vaddq_f32(vaddq_f32(r0[0], r0[1]), r0[2]);
  y[0][1] = vaddq_f32(vsubq_f32(r0[1], r0[2]), r0[3]);
  y[1][0] = vaddq_f32(vaddq_f32(r1[0], r1[1]), r1[2]);
  y[1][1] = vaddq_f32(vsubq_f32(r1[1], r1[2]), r1[3]);
}

// Output transform + epilogue of output channel c for tiles [t0, t0 + tc) (M_p[c][t - t0] at m + p * pst + (t - t0)),
// plus the zero border cells these tiles own (row 0 / the bottom rows with the first tile of a tile row, col 0 / the
// right cols with its first / last tile).
void winoOut2(const WinoPlanar& p, const WinoGeom2& g, int64_t c, int64_t t0, int64_t tc, const float* m, int64_t pst) {
  float* oc = p.out + c * g.plane;
  const float* rc = p.res ? p.res + c * g.plane : nullptr;
  const float bias = p.bias[c];
  const float32x4_t bv = vdupq_n_f32(bias);
  const int64_t per = g.th * g.tw, wp = p.wp;
  for (int64_t t = t0; t < t0 + tc;) {
    const int64_t im = t / per, rem = t % per, ty = rem / g.tw, tx0 = rem % g.tw;
    const int64_t tx1 = std::min(g.tw, tx0 + (t0 + tc - t));
    float* o = oc + im * g.img;
    const float* rr = rc ? rc + im * g.img : nullptr;
    const int64_t y0 = 1 + 2 * ty;
    const bool two = 2 * ty + 1 < g.h;  // second output row inside the image
    if (tx0 == 0) {
      if (ty == 0) std::memset(o, 0, static_cast<size_t>(wp) * sizeof(float));
      if (ty == g.th - 1) {
        const int64_t r0 = two ? y0 + 2 : y0 + 1;
        std::memset(o + r0 * wp, 0, static_cast<size_t>((p.hp - r0) * wp) * sizeof(float));
      }
      o[y0 * wp] = 0.f;
      if (two) o[(y0 + 1) * wp] = 0.f;
    }
    if (tx1 == g.tw) {
      for (int64_t j = 1 + g.w; j < wp; ++j) {
        o[y0 * wp + j] = 0.f;
        if (two) o[(y0 + 1) * wp + j] = 0.f;
      }
    }
    const float* mj = m + (t - t0);
    int64_t tx = tx0;
    float32x4_t y[2][2];
    for (; tx + 4 <= tx1 && 2 * tx + 8 <= g.w; tx += 4, mj += 4) {
      winoOutY(mj, pst, y);
      for (int i = 0; i < (two ? 2 : 1); ++i) {
        const int64_t off = (y0 + i) * wp + 1 + 2 * tx;
        float32x4_t a = vaddq_f32(y[i][0], bv), b = vaddq_f32(y[i][1], bv);
        if (rr) {
          const float32x4x2_t r2 = vld2q_f32(rr + off);
          a = vaddq_f32(a, r2.val[0]), b = vaddq_f32(b, r2.val[1]);
        }
        float32x4x2_t st;
        st.val[0] = actV(a, p.act), st.val[1] = actV(b, p.act);
        vst2q_f32(o + off, st);
      }
    }
    for (; tx < tx1; tx += 4, mj += 4) {  // tail: M read in place (lanes past the chunk read its padding columns)
      const int64_t cnt = std::min<int64_t>(4, tx1 - tx);
      winoOutY(mj, pst, y);
      float ys[2][2][4];
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) vst1q_f32(ys[i][j], y[i][j]);
      for (int l = 0; l < cnt; ++l)
        for (int i = 0; i < (two ? 2 : 1); ++i)
          for (int j = 0; j < 2; ++j) {
            const int64_t col = 2 * (tx + l) + j;
            if (col >= g.w) continue;
            const int64_t off = (y0 + i) * wp + 1 + col;
            o[off] = actS(ys[i][j][l] + bias + (rr ? rr[off] : 0.f), p.act);
          }
    }
    t += tx1 - tx0;
  }
}

// M_p[co][0, tc) (row stride ld) = U_p · V_p for the 16 points, output rows [r0, r0 + rows) (r0 a multiple of 32).
void winoGemm2(const WinoPlanar& p, const float* v, float* m, int64_t ld, int64_t tc, int64_t r0, int64_t rows) {
  const int64_t cop = (p.co + 31) / 32 * 32;
  for (int64_t pt = 0; pt < 16; ++pt) {
    const float* up = p.u + pt * cop * p.ci + r0 * p.ci;
    const float* vp = v + pt * p.ci * ld;
    float* mp = m + pt * p.co * ld + r0 * ld;
    if (sme2::available()) {
      sme2::gemm(up, rows, p.ci, {vp, ld}, tc, mp, ld);
      continue;
    }
    for (int64_t pb = 0; pb * 32 < rows; ++pb)  // no SME2 (Mac, tests): one GEMM per 32-row panel ([Ci][32] = A^T)
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, detail::li(std::min<int64_t>(32, rows - pb * 32)),
                  detail::li(tc), detail::li(p.ci), 1.f, up + pb * 32 * p.ci, 32, vp, detail::li(ld), 0.f,
                  mp + pb * 32 * ld, detail::li(ld));
  }
}

void winoPlanarConv(const WinoPlanar& p) {
  const WinoGeom2 g(p);
  const int64_t threads = rt::ThreadPool::global().threads();
  // Chunk: all tiles when V + M fit in ~6 MB or are at most twice U (the weights then stream once: 512 channels at
  // N = 8, U = 16.8 MB), else ~2 MB chunks (multiple of 32 tiles).
  const int64_t perTile = 16 * (p.ci + p.co) * 4;
  const int64_t ubytes = 16 * p.ci * p.co * 4;
  int64_t tc = g.tiles * perTile <= std::max<int64_t>(int64_t{6} << 20, 2 * ubytes)
                   ? g.tiles
                   : std::max<int64_t>(32, (int64_t{2} << 20) / perTile / 32 * 32);
  // Still split a single chunk into ~2 (multiples of 16 tiles) so the transforms overlap the GEMMs, even though the
  // weights then stream twice (iPhone, us: 512ch 7x7 N = 8 980 -> 890, 256ch 14x14 N = 1 165 -> 145, 128ch 28x28
  // N = 1 161 -> 134; 512ch N = 2 (32 tiles) 272 -> 405, hence >= 48 tiles).
  if (tc == g.tiles && g.tiles >= 48) tc = std::max<int64_t>(16, (g.tiles + 1) / 2 / 16 * 16);
  const int64_t nchunks = (g.tiles + tc - 1) / tc;
  // Row stride of V / M: an odd multiple of 16 floats (power-of-two strides alias in the caches).
  const int64_t ld = (tc + 31) / 32 * 32 + 16;
  const int64_t R = std::min<int64_t>(nchunks, 3);
  const size_t vsz = static_cast<size_t>(16 * p.ci * ld), msz = static_cast<size_t>(16 * p.co * ld);
  float* vring = detail::scratch(vsz * R, 5);
  float* mring = detail::scratch(msz * R, 6);
  const int64_t nb1 = std::min<int64_t>(p.ci, 2 * threads), cb1 = (p.ci + nb1 - 1) / nb1;
  // One chunk (no pipelining over tiles): the GEMMs run in G groups of output rows, so the output transform of a group
  // overlaps the GEMMs of the next (512 channels at N = 8, the N = 1 layers).
  const int64_t panels = (p.co + 31) / 32;
  const int64_t G = nchunks == 1 ? std::min<int64_t>(panels, 4) : 1, cg = (panels + G - 1) / G * 32;
  const int64_t S = std::max<int64_t>(1, 2 * threads / G);  // output-transform items per group
  const int64_t nb3 = G * S;
  const int64_t items1 = nchunks * nb1, items2 = nchunks * G, items3 = nchunks * nb3;
  auto chunkTiles = [&](int64_t ch) { return std::min(tc, g.tiles - ch * tc); };
  auto stage1 = [&](int64_t it) {
    const int64_t ch = it / nb1, b = it % nb1, t0 = ch * tc, n = chunkTiles(ch);
    float* v = vring + (ch % R) * vsz;
    for (int64_t c = b * cb1; c < std::min(p.ci, (b + 1) * cb1); ++c) winoIn2(p, g, c, t0, n, v + c * ld, p.ci * ld);
  };
  auto stage2 = [&](int64_t it) {
    const int64_t ch = it / G, gi = it % G, r0 = gi * cg;
    if (r0 < p.co)
      winoGemm2(p, vring + (ch % R) * vsz, mring + (ch % R) * msz, ld, chunkTiles(ch), r0, std::min(cg, p.co - r0));
  };
  auto stage3 = [&](int64_t it) {
    const int64_t ch = it / nb3, gi = (it % nb3) / S, sb = it % S, t0 = ch * tc, n = chunkTiles(ch);
    const int64_t g0 = gi * cg, g1 = std::min(p.co, g0 + cg), per = (g1 - g0 + S - 1) / S;
    const float* m = mring + (ch % R) * msz;
    for (int64_t c = g0 + sb * per; c < std::min(g1, g0 + (sb + 1) * per); ++c)
      winoOut2(p, g, c, t0, n, m + c * ld, p.co * ld);
  };
  std::unique_ptr<std::atomic<int>[]> done1(new std::atomic<int>[nchunks]), done2(new std::atomic<int>[items2]),
      done3(new std::atomic<int>[nchunks]);
  for (int64_t i = 0; i < nchunks; ++i) done1[i] = 0, done3[i] = 0;
  for (int64_t i = 0; i < items2; ++i) done2[i] = 0;
  std::atomic<int64_t> next1{0}, next2{0}, next3{0}, finished{0};
  std::atomic<bool> busy2{false};
  auto claim = [](std::atomic<int64_t>& c, int64_t t) { return c.compare_exchange_strong(t, t + 1); };
  auto try1 = [&]() {
    const int64_t it = next1.load(std::memory_order_acquire), ch = it / nb1;
    if (it >= items1 || (ch >= R && done3[ch - R].load(std::memory_order_acquire) < nb3) || !claim(next1, it))
      return false;
    stage1(it);
    done1[ch].fetch_add(1, std::memory_order_acq_rel);
    return true;
  };
  auto try2 = [&]() {
    const int64_t it = next2.load(std::memory_order_acquire), ch = it / G;
    if (it >= items2 || done1[ch].load(std::memory_order_acquire) < nb1) return false;
    bool idle = false;
    if (!busy2.compare_exchange_strong(idle, true)) return false;
    if (!claim(next2, it)) {
      busy2.store(false, std::memory_order_release);
      return false;
    }
    stage2(it);
    done2[it].store(1, std::memory_order_release);
    busy2.store(false, std::memory_order_release);
    return true;
  };
  auto try3 = [&]() {
    const int64_t it = next3.load(std::memory_order_acquire), ch = it / nb3, gi = (it % nb3) / S;
    if (it >= items3 || done2[ch * G + gi].load(std::memory_order_acquire) == 0 || !claim(next3, it)) return false;
    stage3(it);
    done3[ch].fetch_add(1, std::memory_order_acq_rel);
    finished.fetch_add(1, std::memory_order_acq_rel);
    return true;
  };
  detail::parallelForSt(threads, [&](int64_t) {
    int64_t idleSince = -1;
    while (finished.load(std::memory_order_acquire) < items3) {
      bool did;
      if (detail::onPerfCore()) {
        did = try2() || try3() || try1();
      } else {
        // E cores (their SME unit runs these GEMMs ~9x slower) take a GEMM only when it sat unclaimed for 1 ms.
        const bool stale = idleSince >= 0 &&
                           static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) - idleSince > 1000000;
        did = try3() || try1() || (stale && try2());
      }
      if (did) {
        idleSince = -1;
      } else if (idleSince < 0) {
        idleSince = static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
      }
    }
  });
}

// Small layout passes run on the calling thread: their consumer is an SME2 GEMM on a P core, and data written by the
// pool's efficiency cores is slow to pull across clusters (iPhone ResNet-18 N = 1, 2026-11-02: the stem 301 -> 254 us
// and the 1x1 downsamples 71 / 57 / 71 -> 43 / 28 / 22 us when their phase planes come from the P core; e2e 2.96 ->
// 2.84 ms). Above ~160 K output floats the pool's bandwidth wins (64ch 58x58 phases: 18 us pooled vs 77 serial; N = 8
// layers 3 / 4: about even).
inline void layoutFor(int64_t n, int64_t outFloats, const std::function<void(int64_t)>& fn) {
  if (outFloats <= 160000) {
    for (int64_t i = 0; i < n; ++i) fn(i);
    return;
  }
  rt::ThreadPool::global().parallelFor(n, fn);
}

// Phase planes for a stride-2 conv, in the output geometry [4][C][N][Hpo][Wpo]: phases[a * 2 + b][c][n][i][j] =
// src(2i + a, 2j + b), where src is either a cnhw tensor (stored coordinates, border included) or an NCHW image with a
// virtual zero border of `pad` (the stem).
void phasesFromCnhw(const float* x, float* out, int64_t c, int64_t n, int64_t hp, int64_t wp, int64_t hpo,
                    int64_t wpo) {
  const int64_t img = hpo * wpo, plane = n * img;
  auto task = [&](int64_t t) {  // one (channel, image)
    const int64_t ch = t / n, b = t % n;
    const float* src = x + (ch * n + b) * hp * wp;
    for (int64_t a = 0; a < 2; ++a)
      for (int64_t i = 0; i < hpo; ++i) {
        float* d0 = out + ((a * 2) * c + ch) * plane + b * img + i * wpo;
        float* d1 = out + ((a * 2 + 1) * c + ch) * plane + b * img + i * wpo;
        const int64_t y = 2 * i + a;
        if (y >= hp) {
          std::memset(d0, 0, static_cast<size_t>(wpo) * sizeof(float));
          std::memset(d1, 0, static_cast<size_t>(wpo) * sizeof(float));
          continue;
        }
        const float* row = src + y * wp;
        int64_t j = 0;
        for (; j + 4 <= wpo && 2 * j + 8 <= wp; j += 4) {
          const float32x4x2_t v = vld2q_f32(row + 2 * j);
          vst1q_f32(d0 + j, v.val[0]);
          vst1q_f32(d1 + j, v.val[1]);
        }
        for (; j < wpo; ++j) {
          d0[j] = 2 * j < wp ? row[2 * j] : 0.f;
          d1[j] = 2 * j + 1 < wp ? row[2 * j + 1] : 0.f;
        }
      }
  };
  if (c * n * hpo * wpo * 4 < 65536) {
    for (int64_t t = 0; t < c * n; ++t) task(t);
  } else {
    layoutFor(c * n, 4 * c * plane, task);
  }
}

void phasesFromNchw(const float* x, float* out, int64_t n, int64_t c, int64_t h, int64_t w, int64_t pad,
                    int64_t hpo, int64_t wpo) {
  // Tasks = (channel, image, phase row a, block of output rows): each source row is copied once into a zero-padded
  // scratch row (pad + 1 zeros in front, so both column phases are an aligned vld2 deinterleave), then split into the
  // b = 0 / 1 phase rows. (Was a per-element loop over (channel, image) tasks: 77 us at N = 1 on the iPhone.)
  const int64_t img = hpo * wpo, plane = n * img;
  const int64_t lead = pad + 1;  // tmp[k] = src[k - lead]
  const int64_t rowLen = lead + 2 * wpo + 8;
  constexpr int64_t kRows = 16;
  const int64_t rblocks = (hpo + kRows - 1) / kRows;
  auto task = [&](int64_t t) {
    const int64_t rb = t % rblocks, a = (t / rblocks) % 2, cb = t / rblocks / 2, ch = cb / n, b = cb % n;
    const float* src = x + (b * c + ch) * h * w;
    float* tmp = detail::scratch(static_cast<size_t>(rowLen), 1);
    float* d0 = out + ((a * 2) * c + ch) * plane + b * img;
    float* d1 = out + ((a * 2 + 1) * c + ch) * plane + b * img;
    for (int64_t i = rb * kRows; i < std::min(hpo, (rb + 1) * kRows); ++i) {
      const int64_t y = 2 * i + a - pad;
      if (y < 0 || y >= h) {
        std::memset(d0 + i * wpo, 0, static_cast<size_t>(wpo) * sizeof(float));
        std::memset(d1 + i * wpo, 0, static_cast<size_t>(wpo) * sizeof(float));
        continue;
      }
      std::memset(tmp, 0, static_cast<size_t>(lead) * sizeof(float));
      const int64_t nw = std::min(w, rowLen - lead);
      std::memcpy(tmp + lead, src + y * w, static_cast<size_t>(nw) * sizeof(float));
      std::memset(tmp + lead + nw, 0, static_cast<size_t>(rowLen - lead - nw) * sizeof(float));
      // phase b column j = src[2j + b - pad] = tmp[2j + b + 1]: b = 0 -> odd lanes of vld2 at tmp, b = 1 -> even lanes
      // of vld2 at tmp + 2.
      const float* p0 = tmp + (lead - pad - 1);
      const float* p1 = tmp + (lead - pad + 1);
      float* r0 = d0 + i * wpo;
      float* r1 = d1 + i * wpo;
      int64_t j = 0;
      for (; j + 4 <= wpo; j += 4) {
        vst1q_f32(r0 + j, vld2q_f32(p0 + 2 * j).val[1]);
        vst1q_f32(r1 + j, vld2q_f32(p1 + 2 * j).val[0]);
      }
      for (; j < wpo; ++j) r0[j] = p0[2 * j + 1], r1[j] = p1[2 * j];
    }
  };
  // Always on the calling thread: the stem GEMM that reads it runs ~20 % faster on P-core-written planes (N = 8: stem
  // 3.03 -> 2.48 ms for 207 -> 317 us here).
  for (int64_t t = 0; t < c * n * 2 * rblocks; ++t) task(t);
}

// 3x3 stride-2 pad-1 max pool, cnhw -> cnhw (border 1). The input must be >= 0 (post-ReLU): its zero border then acts
// as the -inf padding.
void maxpoolCnhw(const float* x, float* out, int64_t c, int64_t n, int64_t hp, int64_t wp, int64_t hpo, int64_t wpo) {
  const int64_t ho = hpo - 2, wo = wpo - 2;
  auto task = [&](int64_t t) {
    const float* src = x + t * hp * wp;
    float* dst = out + t * hpo * wpo;
    std::memset(dst, 0, static_cast<size_t>(wpo) * sizeof(float));
    std::memset(dst + (ho + 1) * wpo, 0, static_cast<size_t>(wpo) * sizeof(float));
    for (int64_t oh = 0; oh < ho; ++oh) {
      const float *r0 = src + 2 * oh * wp, *r1 = r0 + wp, *r2 = r1 + wp;
      float* d = dst + (oh + 1) * wpo;
      d[0] = 0.f;
      int64_t ow = 0;
      for (; ow + 4 <= wo && 2 * ow + 10 <= wp; ow += 4) {
        const float32x4x2_t a0 = vld2q_f32(r0 + 2 * ow), a1 = vld2q_f32(r1 + 2 * ow), a2 = vld2q_f32(r2 + 2 * ow);
        const float32x4x2_t b0 = vld2q_f32(r0 + 2 * ow + 2), b1 = vld2q_f32(r1 + 2 * ow + 2),
                            b2 = vld2q_f32(r2 + 2 * ow + 2);
        const float32x4_t ev = vmaxq_f32(vmaxq_f32(a0.val[0], a1.val[0]), a2.val[0]);
        const float32x4_t od = vmaxq_f32(vmaxq_f32(a0.val[1], a1.val[1]), a2.val[1]);
        const float32x4_t e2 = vmaxq_f32(vmaxq_f32(b0.val[0], b1.val[0]), b2.val[0]);
        vst1q_f32(d + 1 + ow, vmaxq_f32(vmaxq_f32(ev, od), e2));
      }
      for (; ow < wo; ++ow) {
        float m = 0.f;
        for (int64_t y = 0; y < 3; ++y)
          for (int64_t xx = 0; xx < 3; ++xx) m = std::max(m, src[(2 * oh + y) * wp + 2 * ow + xx]);
        d[1 + ow] = m;
      }
      for (int64_t j = wo + 1; j < wpo; ++j) d[j] = 0.f;
    }
  };
  if (c * n * ho * wo < 50000) {
    for (int64_t t = 0; t < c * n; ++t) task(t);
  } else {
    rt::ThreadPool::global().parallelFor(c * n, task);
  }
}

}  // namespace

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

const float* F(const KernelArgs& a, int i) { return static_cast<const float*>(a.ptrs[i]); }
float* O(const KernelArgs& a, int i) { return static_cast<float*>(a.ptrs[i]); }

using kernels::cpu::Act;

// conv2d_nhwc[_res][_relu|_relu6]_f32_cpu(x, w, b, [res]) outs(out)
// ints: x(N, H, W, Ci), w(KH, KW, Ci, Co), b(Co), [res(N, Ho, Wo, Co)], out(N, Ho, Wo, Co), padding, stride
template <bool Res, Act A>
void runConv2d(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  const int o = Res ? 13 : 9;
  kernels::cpu::Conv2dArgs c{F(a, 0), F(a, 1), F(a, 2), Res ? F(a, 3) : nullptr, O(a, Res ? 4 : 3),
                             n[0], n[1], n[2], n[3], n[4], n[5], n[7], n[o + 1], n[o + 2], n[o + 5], n[o + 4], A};
  kernels::cpu::conv2d(c);
}

// dwconv2d_nhwc[_relu6]_f32_cpu(x, w, b) outs(out)
// ints: x(N, H, W, C), w(KH, KW, C), b(C), out(N, Ho, Wo, C), padding, stride
template <Act A>
void runDwConv2d(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::dwconv2d(F(a, 0), F(a, 1), F(a, 2), O(a, 3), n[0], n[1], n[2], n[3], n[4], n[5], n[9], n[10],
                         n[13], n[12], A);
}

// ints: x(N, H, W, C), out(N, Ho, Wo, C), kernel, padding, stride
void runMaxPool(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  const int64_t N = n[0], H = n[1], W = n[2], C = n[3], Ho = n[5], Wo = n[6], k = n[8], pad = n[9], s = n[10];
  const float* x = F(a, 0);
  float* out = O(a, 1);
  auto row = [&](int64_t r) {  // one output row; NEON over channels, the window in registers per 16-channel block
    const int64_t b = r / Ho, oh = r % Ho;
    for (int64_t ow = 0; ow < Wo; ++ow) {
      float* o = out + ((b * Ho + oh) * Wo + ow) * C;
      const int64_t y0 = std::max<int64_t>(0, oh * s - pad), y1 = std::min<int64_t>(H, oh * s - pad + k);
      const int64_t x0 = std::max<int64_t>(0, ow * s - pad), x1 = std::min<int64_t>(W, ow * s - pad + k);
      int64_t c = 0;
      for (; c + 16 <= C; c += 16) {
        const float32x4_t ninf = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        float32x4_t m0 = ninf, m1 = ninf, m2 = ninf, m3 = ninf;
        for (int64_t ih = y0; ih < y1; ++ih)
          for (int64_t iw = x0; iw < x1; ++iw) {
            const float* src = x + ((b * H + ih) * W + iw) * C + c;
            m0 = vmaxq_f32(m0, vld1q_f32(src));
            m1 = vmaxq_f32(m1, vld1q_f32(src + 4));
            m2 = vmaxq_f32(m2, vld1q_f32(src + 8));
            m3 = vmaxq_f32(m3, vld1q_f32(src + 12));
          }
        vst1q_f32(o + c, m0);
        vst1q_f32(o + c + 4, m1);
        vst1q_f32(o + c + 8, m2);
        vst1q_f32(o + c + 12, m3);
      }
      for (; c < C; ++c) {
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t ih = y0; ih < y1; ++ih)
          for (int64_t iw = x0; iw < x1; ++iw) m = std::max(m, x[((b * H + ih) * W + iw) * C + c]);
        o[c] = m;
      }
    }
  };
  const int64_t rows = N * Ho;
  if (static_cast<double>(rows * Wo * C * k * k) < 2e5) {
    for (int64_t r = 0; r < rows; ++r) row(r);
  } else {
    rt::ThreadPool::global().parallelFor(rows, row);
  }
}

// ints: x(N, H, W, C), out(N, C)
void runGlobalAvgPool(ExecCtx&, const KernelArgs& a) {
  const int64_t N = a.ints[0], HW = a.ints[1] * a.ints[2], C = a.ints[3];
  const float* x = F(a, 0);
  float* out = O(a, 1);
  for (int64_t b = 0; b < N; ++b) {
    float* o = out + b * C;
    std::fill(o, o + C, 0.f);
    for (int64_t p = 0; p < HW; ++p) {
      const float* src = x + (b * HW + p) * C;
      for (int64_t c = 0; c < C; ++c) o[c] += src[c];
    }
    const float inv = 1.f / static_cast<float>(HW);
    for (int64_t c = 0; c < C; ++c) o[c] *= inv;
  }
}

// convt2d_gather_nhwc[_relu|_tanh]_f32_cpu(y, b) outs(out)
// ints: y(B, H, W, K*K*Co), b(Co), out(B, Ho, Wo, Co), kernel, padding, stride
template <Act A>
void runConvT2dGather(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::convt2dGather(F(a, 0), F(a, 1), O(a, 2), n[0], n[1], n[2], n[4], n[9], n[6], n[7], n[11], n[10], A);
}

// ints: x(N, H, W, C), out(N, C, H, W)
void runNhwcToNchw(ExecCtx&, const KernelArgs& a) {
  const int64_t N = a.ints[0], HW = a.ints[1] * a.ints[2], C = a.ints[3];
  const float* x = F(a, 0);
  float* out = O(a, 1);
  for (int64_t b = 0; b < N; ++b)
    for (int64_t c = 0; c < C; ++c) {
      float* dst = out + (b * C + c) * HW;
      const float* src = x + b * HW * C + c;
      for (int64_t p = 0; p < HW; ++p) dst[p] = src[p * C];
    }
}

// Same shape in/out; rank = numInts / 2.
void runSigmoid(ExecCtx&, const KernelArgs& a) {
  int64_t n = 1;
  for (uint32_t i = 0; i < a.numInts / 2; ++i) n *= a.ints[i];
  const float* x = F(a, 0);
  float* out = O(a, 1);
  for (int64_t i = 0; i < n; ++i) out[i] = 1.f / (1.f + std::exp(-x[i]));
}

// ints: x(N, C, H, W), out(N, H, W, C)
void runNchwToNhwc(ExecCtx&, const KernelArgs& a) {
  const int64_t N = a.ints[0], C = a.ints[1], HW = a.ints[2] * a.ints[3];
  const float* x = F(a, 0);
  float* out = O(a, 1);
  if (C == 3) {  // image inputs: NEON interleaving stores, 4 pixels per vst3q, chunks of pixels on the pool
    constexpr int64_t kChunk = 4096;
    const int64_t chunks = (HW + kChunk - 1) / kChunk;
    auto task = [&](int64_t t) {
      const int64_t b = t / chunks, p0 = (t % chunks) * kChunk, p1 = std::min(HW, p0 + kChunk);
      const float* s0 = x + b * 3 * HW;
      const float *s1 = s0 + HW, *s2 = s1 + HW;
      float* d = out + b * HW * 3;
      int64_t p = p0;
      for (; p + 4 <= p1; p += 4) vst3q_f32(d + p * 3, (float32x4x3_t{{vld1q_f32(s0 + p), vld1q_f32(s1 + p), vld1q_f32(s2 + p)}}));
      for (; p < p1; ++p) d[p * 3] = s0[p], d[p * 3 + 1] = s1[p], d[p * 3 + 2] = s2[p];
    };
    if (N * HW < 32768) {
      for (int64_t t = 0; t < N * chunks; ++t) task(t);
    } else {
      rt::ThreadPool::global().parallelFor(N * chunks, task);
    }
    return;
  }
  for (int64_t b = 0; b < N; ++b)
    for (int64_t c = 0; c < C; ++c) {
      const float* src = x + (b * C + c) * HW;
      float* dst = out + b * HW * C + c;
      for (int64_t p = 0; p < HW; ++p) dst[p * C] = src[p];
    }
}

// conv2d_cnhw[_res][_relu]_f32_cpu(x, wp, b, [res]) outs(out); x [Ci, N, Hp, Wp] (stride 1) or [4, Ci, N, Hpo, Wpo]
// (stride 2 phases), wp packed weights f32[P], out [Co, N, Hpo, Wpo]. ints: shapes, then extra, kh, kw, pad, pin, stride.
template <bool Res, Act A>
void runConvCnhw(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  const int64_t* at = n + a.numInts - 6;
  const int64_t* o = at - 4;
  const int64_t stride = at[5];
  kernels::cpu::PlanarConv p{F(a, 0), F(a, 1), F(a, 2), Res ? F(a, 3) : nullptr, O(a, Res ? 4 : 3),
                             stride == 1 ? n[0] : n[1], o[0], o[1], o[2], o[3], at[0], at[1], at[2], at[3], at[4],
                             stride, A};
  kernels::cpu::planarConv(p);
}

// conv2d_wino_cnhw[_res]_relu_f32_cpu(x, wp, u, b, [res]) outs(out): 3x3 stride-1 pad-1, x [Ci, N, Hp, Wp] (no extra),
// wp = ops.pack_conv_weights_cnhw(w), u = ops.pack_wino_weights_cnhw(w), out [Co, N, Hp, Wp]. ints: shapes only.
// Winograd from kWinoMinTiles 2x2 tiles up, the direct planar GEMM below (N is only known at run time).
template <bool Res, Act A>
void runConvWinoCnhw(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  const int64_t* o = n + a.numInts - 4;
  const float* res = Res ? F(a, 4) : nullptr;
  float* out = O(a, Res ? 5 : 4);
  kernels::cpu::WinoPlanar p{F(a, 0), F(a, 2), F(a, 3), res, out, n[0], o[0], o[1], o[2], o[3], A};
  static const int64_t minTiles = [] { const char* e = std::getenv("LOKINFER_WINO_MIN_TILES"); return e ? std::atoll(e) : kernels::cpu::kWinoMinTiles; }();
  const int64_t tiles = o[1] * ((o[2] - 1) / 2) * ((o[3] - 1) / 2);
  if (tiles >= minTiles || (o[1] >= 4 && tiles >= 64)) return kernels::cpu::winoPlanarConv(p);
  kernels::cpu::PlanarConv d{F(a, 0), F(a, 1), F(a, 3), res, out, n[0], o[0], o[1], o[2], o[3], 0, 3, 3, 1, 1, 1, A};
  kernels::cpu::planarConv(d);
}

// conv2d_smallci_nchw[_act]_f32_cpu(x, w, b) outs(out): x(N, Ci, H, W) NCHW, w(KH, KW, Ci, Co), b(Co),
// out(N, Ho, Wo, Co) NHWC; padding, stride (1 or 2). Ci * KH <= 64, Co % 16 == 0.
template <Act A>
void runSmallCi(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::SmallCiConv p{F(a, 0), F(a, 1), F(a, 2), O(a, 3), n[0], n[1], n[2], n[3], n[4], n[5], n[7],
                              n[10], n[11], n[14], n[13], A};
  kernels::cpu::smallCiConv(p);
}

// nchw_to_phases_f32_cpu(x) outs(out): x(N, C, H, W), out(4, C, N, Hpo, Wpo), extra, kernel, pad
void runNchwToPhases(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::phasesFromNchw(F(a, 0), O(a, 1), n[0], n[1], n[2], n[3], n[11], n[7], n[8]);
}

// cnhw_to_phases_f32_cpu(x) outs(out): x(C, N, Hp, Wp), out(4, C, N, Hpo, Wpo)
void runCnhwToPhases(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::phasesFromCnhw(F(a, 0), O(a, 1), n[0], n[1], n[2], n[3], n[7], n[8]);
}

// maxpool_cnhw_f32_cpu(x) outs(out): x(C, N, Hp, Wp), out(C, N, Hpo, Wpo), extra
void runMaxPoolCnhw(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::maxpoolCnhw(F(a, 0), O(a, 1), n[0], n[1], n[2], n[3], n[6], n[7]);
}

// global_avgpool_cnhw_f32_cpu(x) outs(out): x(C, N, Hp, Wp) (border 1), out(N, C)
void runAvgPoolCnhw(ExecCtx&, const KernelArgs& a) {
  const int64_t C = a.ints[0], N = a.ints[1], hp = a.ints[2], wp = a.ints[3];
  const float* x = F(a, 0);
  float* out = O(a, 1);
  const float inv = 1.f / static_cast<float>((hp - 2) * (wp - 2));
  for (int64_t c = 0; c < C; ++c)
    for (int64_t b = 0; b < N; ++b) {
      const float* pl = x + (c * N + b) * hp * wp;
      float s = 0.f;
      for (int64_t y = 1; y + 1 < hp; ++y)
        for (int64_t xx = 1; xx + 1 < wp; ++xx) s += pl[y * wp + xx];
      out[b * C + c] = s * inv;
    }
}

// mbv2_block_nhwc[_res]_f32_cpu(x, we, be, wd, bd, wp, bp) outs(out)
// ints: x(N, H, W, Ci), we(Ci, Ce), be(Ce), wd(3, 3, Ce), bd(Ce), wp(Ce, Co), bp(Co), out(N, Ho, Wo, Co), stride
template <bool Res>
void runInvertedResidual(ExecCtx&, const KernelArgs& a) {
  const int64_t* n = a.ints;
  kernels::cpu::invertedResidual(F(a, 0), F(a, 1), F(a, 2), F(a, 3), F(a, 4), F(a, 5), F(a, 6), O(a, 7), n[0], n[1],
                                 n[2], n[3], n[5], n[12], n[15], n[16], n[18], Res);
}

}  // namespace


void registerCpuVisionKernels() {
  registerKernel({"conv2d_nhwc_f32_cpu", Device::Cpu, nullptr, runConv2d<false, Act::None>, nullptr});
  registerKernel({"conv2d_nhwc_relu_f32_cpu", Device::Cpu, nullptr, runConv2d<false, Act::Relu>, nullptr});
  registerKernel({"conv2d_nhwc_relu6_f32_cpu", Device::Cpu, nullptr, runConv2d<false, Act::Relu6>, nullptr});
  registerKernel({"conv2d_nhwc_res_f32_cpu", Device::Cpu, nullptr, runConv2d<true, Act::None>, nullptr});
  registerKernel({"conv2d_nhwc_res_relu_f32_cpu", Device::Cpu, nullptr, runConv2d<true, Act::Relu>, nullptr});
  registerKernel({"conv2d_nhwc_leaky_f32_cpu", Device::Cpu, nullptr, runConv2d<false, Act::Leaky>, nullptr});
  registerKernel({"convt2d_gather_nhwc_relu_f32_cpu", Device::Cpu, nullptr, runConvT2dGather<Act::Relu>, nullptr});
  registerKernel({"convt2d_gather_nhwc_tanh_f32_cpu", Device::Cpu, nullptr, runConvT2dGather<Act::Tanh>, nullptr});
  registerKernel({"nhwc_to_nchw_f32_cpu", Device::Cpu, nullptr, runNhwcToNchw, nullptr});
  registerKernel({"sigmoid_f32_cpu", Device::Cpu, nullptr, runSigmoid, nullptr});
  registerKernel({"dwconv2d_nhwc_f32_cpu", Device::Cpu, nullptr, runDwConv2d<Act::None>, nullptr});
  registerKernel({"dwconv2d_nhwc_relu6_f32_cpu", Device::Cpu, nullptr, runDwConv2d<Act::Relu6>, nullptr});
  registerKernel({"mbv2_block_nhwc_f32_cpu", Device::Cpu, nullptr, runInvertedResidual<false>, nullptr});
  registerKernel({"mbv2_block_nhwc_res_f32_cpu", Device::Cpu, nullptr, runInvertedResidual<true>, nullptr});
  registerKernel({"maxpool2d_nhwc_f32_cpu", Device::Cpu, nullptr, runMaxPool, nullptr});
  registerKernel({"global_avgpool_nhwc_f32_cpu", Device::Cpu, nullptr, runGlobalAvgPool, nullptr});
  registerKernel({"nchw_to_nhwc_f32_cpu", Device::Cpu, nullptr, runNchwToNhwc, nullptr});
  registerKernel({"conv2d_cnhw_f32_cpu", Device::Cpu, nullptr, runConvCnhw<false, Act::None>, nullptr});
  registerKernel({"conv2d_cnhw_relu_f32_cpu", Device::Cpu, nullptr, runConvCnhw<false, Act::Relu>, nullptr});
  registerKernel({"conv2d_cnhw_res_relu_f32_cpu", Device::Cpu, nullptr, runConvCnhw<true, Act::Relu>, nullptr});
  registerKernel({"conv2d_wino_cnhw_relu_f32_cpu", Device::Cpu, nullptr, runConvWinoCnhw<false, Act::Relu>, nullptr});
  registerKernel({"conv2d_wino_cnhw_res_relu_f32_cpu", Device::Cpu, nullptr, runConvWinoCnhw<true, Act::Relu>, nullptr});
  registerKernel({"conv2d_smallci_nchw_relu6_f32_cpu", Device::Cpu, nullptr, runSmallCi<Act::Relu6>, nullptr});
  registerKernel({"conv2d_smallci_nchw_leaky_f32_cpu", Device::Cpu, nullptr, runSmallCi<Act::Leaky>, nullptr});
  registerKernel({"nchw_to_phases_f32_cpu", Device::Cpu, nullptr, runNchwToPhases, nullptr});
  registerKernel({"cnhw_to_phases_f32_cpu", Device::Cpu, nullptr, runCnhwToPhases, nullptr});
  registerKernel({"maxpool_cnhw_f32_cpu", Device::Cpu, nullptr, runMaxPoolCnhw, nullptr});
  registerKernel({"global_avgpool_cnhw_f32_cpu", Device::Cpu, nullptr, runAvgPoolCnhw, nullptr});
}

}  // namespace lk::rt
