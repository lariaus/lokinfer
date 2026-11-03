// Dispatch for the Kokoro channel-last GPU kernels (kernels/gpu/kokoro_lc.metal), M4.
#include <Metal/Metal.hpp>
#include <TargetConditionals.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../../src/runtime/gpu_internal.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

namespace {

struct PsoState {
  MTL::ComputePipelineState* pso;
};

template <const char* Function>
Status preparePso(PrepareCtx& pc, const Op&, void** state) {
  MTL::ComputePipelineState* pso = pc.gpu->pipeline(Function);
  if (!pso) return Status::error(std::string("missing GPU kernel function ") + Function);
  *state = new PsoState{pso};
  return Status::ok();
}

void destroyPso(void* s) { delete static_cast<PsoState*>(s); }

MTL::ComputePipelineState* psoOf(const KernelArgs& a) { return static_cast<const PsoState*>(a.state)->pso; }

void bindArg(GpuContext& g, uint32_t slot, const KernelArgs& a, uint32_t i) {
  g.bind(slot, static_cast<MTL::Buffer*>(a.bufs[i].mtlBuffer), a.bufs[i].offset);
}

uint32_t u32(int64_t v) { return static_cast<uint32_t>(v); }

void dispatch1d(GpuContext& g, const KernelArgs& a, uint64_t n) { g.dispatch(psoOf(a), n, 1, 1, 256, 1, 1); }

// ---- conv (steel) -----------------------------------------------------------------------------------------

// Two tile configs of the same steel conv: 64x64 (default) and 32x32. The 2-D path (runConv2dNhwc) picks 32x32 per
// call when the 64x64 grid would have fewer threadgroups than kSmallGrid (too few to fill the GPU's cores); the 1-D
// Kokoro path (runConvLc) always uses 64x64.
struct ConvPsos {
  MTL::ComputePipelineState* bm64;
  MTL::ComputePipelineState* bm32;
};
constexpr int kSmallGrid = 128;  // iPhone A/B with 32 (2026-11-01): no gain on MNIST / ResNet-18 / MobileNetV2

Status prepareConv(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new ConvPsos{pc.gpu->pipeline("lk_conv1d_lc_f32"), pc.gpu->pipeline("lk_conv1d_lc_f32_bm32")};
  if (!s->bm64 || !s->bm32) {
    delete s;
    return Status::error("missing GPU kernel function lk_conv1d_lc_f32[_bm32]");
  }
  *state = s;
  return Status::ok();
}
void destroyConv(void* s) { delete static_cast<ConvPsos*>(s); }

// Mirrors MLXConvParams<2> / ImplicitGemmConv2DParams (third_party/mlx_steel/.../steel/conv/params.h).
struct MLXConvParams2 {
  int N, C, O, iS[2], wS[2], oS[2], str[2], pad[2], kdil[2], idil[2];
  int64_t in_strides[4], wt_strides[4], out_strides[4];
  int groups;
  bool flip;
};
struct ImplicitGemmConv2DParams {
  int M, N, K, gemm_k_iterations, inp_jump_w, inp_jump_h, inp_jump_c, tiles_n, tiles_m, swizzle_log;
};
struct LcConvEpilogue {
  float scale;
  uint32_t has_bias, has_res, act;  // act: 0 none, 1 ReLU, 2 ReLU6, 3 leaky 0.2
};

// ptrs: x, w, b, [res], out. ints: x(L, Ci), w(Co, K, Ci), b(Co), [res(Lo, Co)], out(Lo, Co), dilation, padding,
// stride; floats: [oscale].
template <bool Res>
void runConvLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const int L = int(n[0]), Ci = int(n[1]), Co = int(n[2]), K = int(n[3]);
  const int outPtr = Res ? 4 : 3, outInt = Res ? 8 : 6;
  const int lo = int(n[outInt]), d = int(n[outInt + 2]), pad = int(n[outInt + 3]), s = int(n[outInt + 4]);
  MLXConvParams2 cp{1, Ci, Co, {1, L}, {1, K}, {1, lo}, {1, s}, {0, pad}, {1, d}, {1, 1},
                    {int64_t(L) * Ci, int64_t(L) * Ci, Ci, 1}, {int64_t(K) * Ci, int64_t(K) * Ci, Ci, 1},
                    {int64_t(lo) * Co, int64_t(lo) * Co, Co, 1}, 1, false};
  const auto* cps = static_cast<const ConvPsos*>(a.state);
  // 64x64x16 tiles: on Kokoro rand510 (Mac GPU) 64x64x32, 128x64 and 64x128 were 4-8 % slower E2E (prof-lokinfer
  // A/B 2026-11-01), despite the ~22 % occupancy of 64x64x16.
  constexpr int bm = 64, bn = 64, bk = 16;
  const int tn = (Co + bn - 1) / bn, tm = (lo + bm - 1) / bm, ijw = Ci * d;  // Kokoro shapes: always 64x64 by default
  ImplicitGemmConv2DParams gp{lo, Co, K * Ci, K * ((Ci + bk - 1) / bk), ijw, L * Ci - (K - 1) * ijw,
                              bk - (K - 1) * ijw, tn, tm, 0};
  LcConvEpilogue epi{Res ? static_cast<float>(a.floats[0]) : 1.f, 1u, Res ? 1u : 0u, 0u};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, 2);
  bindArg(g, 3, a, outPtr);
  bindArg(g, 4, a, Res ? 3 : outPtr);
  g.bindParams(5, &cp, sizeof cp);
  g.bindParams(6, &gp, sizeof gp);
  g.bindParams(7, &epi, sizeof epi);
  g.dispatchThreadgroups(cps->bm64, tn, tm, 1, 32, 2, 2);
}

// ptrs: x, w, b, [res], out. ints: x(N, H, W, Ci), w(Co, KH, KW, Ci), b(Co), [res(N, Ho, Wo, Co)],
// out(N, Ho, Wo, Co), padding, stride. NHWC conv2d through the same steel kernel (ResNet-18).
template <bool Res, uint32_t Act>
void runConv2dNhwc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const int N = int(n[0]), H = int(n[1]), W = int(n[2]), Ci = int(n[3]), Co = int(n[4]), KH = int(n[5]),
            KW = int(n[6]);
  const int o = Res ? 13 : 9, outPtr = Res ? 4 : 3;
  const int Ho = int(n[o + 1]), Wo = int(n[o + 2]), pad = int(n[o + 4]), s = int(n[o + 5]);
  MLXConvParams2 cp{N, Ci, Co, {H, W}, {KH, KW}, {Ho, Wo}, {s, s}, {pad, pad}, {1, 1}, {1, 1},
                    {int64_t(H) * W * Ci, int64_t(W) * Ci, Ci, 1}, {int64_t(KH) * KW * Ci, int64_t(KW) * Ci, Ci, 1},
                    {int64_t(Ho) * Wo * Co, int64_t(Wo) * Co, Co, 1}, 1, false};
  constexpr int bk = 16;
  const int M = N * Ho * Wo;
  const auto* psos = static_cast<const ConvPsos*>(a.state);
  const bool small = ((Co + 63) / 64) * ((M + 63) / 64) < kSmallGrid;
  const int bm = small ? 32 : 64, bn = bm;
  const int tn = (Co + bn - 1) / bn, tm = (M + bm - 1) / bm;
  const int ijw = Ci, ijh = W * Ci - (KW - 1) * ijw, ijc = bk - (KH - 1) * W * Ci - (KW - 1) * ijw;
  ImplicitGemmConv2DParams gp{M, Co, KH * KW * Ci, KH * KW * ((Ci + bk - 1) / bk), ijw, ijh, ijc, tn, tm, 0};
  LcConvEpilogue epi{1.f, 1u, Res ? 1u : 0u, Act};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, 2);
  bindArg(g, 3, a, outPtr);
  bindArg(g, 4, a, Res ? 3 : outPtr);
  g.bindParams(5, &cp, sizeof cp);
  g.bindParams(6, &gp, sizeof gp);
  g.bindParams(7, &epi, sizeof epi);
  g.dispatchThreadgroups(small ? psos->bm32 : psos->bm64, tn, tm, 1, 32, 2, 2);
}

// ints: x(N, H, W, C), w(KH, KW, C), b(C), out(N, Ho, Wo, C), padding, stride
// Depthwise: the register-blocked 3x3 float4 kernel when it applies, else the generic one (vision.metal).
struct DwPsos {
  MTL::ComputePipelineState* generic;
  MTL::ComputePipelineState* k3s1;
  MTL::ComputePipelineState* k3s2;
};

Status prepareDw(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new DwPsos{pc.gpu->pipeline("lk_dwconv2d_nhwc"), pc.gpu->pipeline("lk_dwconv2d_nhwc_k3s1c4"),
                       pc.gpu->pipeline("lk_dwconv2d_nhwc_k3s2c4")};
  if (!s->generic || !s->k3s1 || !s->k3s2) {
    delete s;
    return Status::error("missing GPU kernel function lk_dwconv2d_nhwc[_k3s{1,2}c4]");
  }
  *state = s;
  return Status::ok();
}
void destroyDw(void* s) { delete static_cast<DwPsos*>(s); }

template <uint32_t Act>
void runDwConv2dNhwc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  struct {
    uint32_t n, h, w, c, kh, kw, ho, wo, stride, pad, act;
  } p{u32(n[0]), u32(n[1]), u32(n[2]), u32(n[3]), u32(n[4]), u32(n[5]), u32(n[9]), u32(n[10]), u32(n[13]),
      u32(n[12]), Act};
  const auto* ps = static_cast<const DwPsos*>(a.state);
  const bool k3 = p.kh == 3 && p.kw == 3 && p.c % 4 == 0 && (p.stride == 1 || p.stride == 2);
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  if (k3) {
    const uint64_t threads = uint64_t(p.n) * p.ho * ((p.wo + 3) / 4) * (p.c / 4);
    g.dispatch(p.stride == 1 ? ps->k3s1 : ps->k3s2, threads, 1, 1, 256, 1, 1);
  } else {
    g.dispatch(ps->generic, uint64_t(p.n) * p.ho * p.wo * p.c, 1, 1, 256, 1, 1);
  }
}

// ints: y(B, H, W, K*K*Co), b(Co), out(B, Ho, Wo, Co), kernel, padding, stride
template <uint32_t Act>
void runConvT2dGatherNhwc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  struct {
    uint32_t b, h, w, co, k, ho, wo, stride, pad, act;
  } p{u32(n[0]), u32(n[1]), u32(n[2]), u32(n[4]), u32(n[9]), u32(n[6]), u32(n[7]), u32(n[11]), u32(n[10]), Act};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.b) * p.ho * p.wo * p.co);
}

// ints: x(N, H, W, C), out(N, C, H, W)
void runNhwcToNchwGpu(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t n, hw, c;
  } p{u32(a.ints[0]), u32(a.ints[1] * a.ints[2]), u32(a.ints[3])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.n) * p.hw * p.c);
}

// Same shape in/out; rank = numInts / 2.
void runSigmoidGpu(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  uint64_t count = 1;
  for (uint32_t i = 0; i < a.numInts / 2; ++i) count *= uint64_t(a.ints[i]);
  const uint32_t n32 = u32(int64_t(count));
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &n32, sizeof n32);
  dispatch1d(g, a, count);
}

// ptrs: x, w, [bias], [res], out. ints: x(1, K), w(K, N), [bias(N)], [res(1, N)], out(1, N)
template <bool Bias, uint32_t Act, bool Res>
void runGemv(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t k, n, has_bias, act, has_res;
  } p{u32(a.ints[1]), u32(a.ints[3]), Bias ? 1u : 0u, Act, Res ? 1u : 0u};
  const uint32_t outPtr = 2 + (Bias ? 1 : 0) + (Res ? 1 : 0);
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, Bias ? 2 : outPtr);
  bindArg(g, 3, a, outPtr);
  bindArg(g, 4, a, Res ? (Bias ? 3 : 2) : outPtr);
  g.bindParams(5, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (p.n + 31) / 32, 1, 1, 256, 1, 1);
}

// ints: x(1, K), w(N, K), out(1, N)
void runGemvRows(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t k, n, has_bias, act, has_res;
  } p{u32(a.ints[1]), u32(a.ints[2]), 0u, 0u, 0u};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (p.n + 7) / 8, 1, 1, 256, 1, 1);
}

// ints: x(R, K), w(K, 1), b(1), out(R, 1)
template <uint32_t Act>
void runRowDot(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, k, act;
  } p{u32(a.ints[0]), u32(a.ints[1]), Act};
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.rows, 1, 1, 256, 1, 1);
}

// ints: x(N, C, H, W), out(N*Ho*Wo, Kpad), then attrs by name: kernel, kpad, padding, stride
struct Im2colPsos {
  MTL::ComputePipelineState* elem;
  MTL::ComputePipelineState* tile;
};

Status prepareIm2col(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new Im2colPsos{pc.gpu->pipeline("lk_im2col_nchw"), pc.gpu->pipeline("lk_im2col_nchw_tile")};
  if (!s->elem || !s->tile) {
    delete s;
    return Status::error("missing GPU kernel function lk_im2col_nchw[_tile]");
  }
  *state = s;
  return Status::ok();
}
void destroyIm2col(void* s) { delete static_cast<Im2colPsos*>(s); }

void runIm2colNchw(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const uint32_t N = u32(n[0]), C = u32(n[1]), H = u32(n[2]), W = u32(n[3]), kpad = u32(n[5]);
  const uint32_t k = u32(n[6]), pad = u32(n[8]), s = u32(n[9]);
  struct {
    uint32_t n, c, h, w, kh, kw, ho, wo, stride, pad, kpad;
  } p{N, C, H, W, k, k, (H + 2 * pad - k) / s + 1, (W + 2 * pad - k) / s + 1, s, pad, kpad};
  const auto* ps = static_cast<const Im2colPsos*>(a.state);
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  if (C * k * (31 * s + k) <= 4096 && kpad <= 256)  // fits lk_im2col_nchw_tile's threadgroup buffers
    g.dispatchThreadgroups(ps->tile, (p.wo + 31) / 32, p.ho, N, 256, 1, 1);
  else
    g.dispatch(ps->elem, kpad, uint64_t(N) * p.ho * p.wo, 1, 32, 8, 1);
}

struct NhwcPadPsos {
  MTL::ComputePipelineState* elem;
  MTL::ComputePipelineState* pix;
};

Status prepareNhwcPad(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new NhwcPadPsos{pc.gpu->pipeline("lk_nchw_to_nhwc_pad"), pc.gpu->pipeline("lk_nchw_to_nhwc_pad_pix")};
  if (!s->elem || !s->pix) {
    delete s;
    return Status::error("missing GPU kernel function lk_nchw_to_nhwc_pad[_pix]");
  }
  *state = s;
  return Status::ok();
}
void destroyNhwcPad(void* s) { delete static_cast<NhwcPadPsos*>(s); }

// ints: x(N, C, H, W), out(N, H, W, Cpad)
void runNchwToNhwcPad(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t n, c, hw, cpad;
  } p{u32(a.ints[0]), u32(a.ints[1]), u32(a.ints[2] * a.ints[3]), u32(a.ints[7])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  const auto* ps = static_cast<const NhwcPadPsos*>(a.state);
  if (p.cpad % 4 == 0 && p.c <= 16)
    g.dispatch(ps->pix, uint64_t(p.n) * p.hw, 1, 1, 256, 1, 1);
  else
    g.dispatch(ps->elem, uint64_t(p.n) * p.hw * p.cpad, 1, 1, 256, 1, 1);
}

// ints: x(N, H, W, C), out(N, Ho, Wo, C), kernel, padding, stride
void runMaxPoolNhwc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  struct {
    uint32_t n, h, w, c, ho, wo, k, pad, stride;
  } p{u32(n[0]), u32(n[1]), u32(n[2]), u32(n[3]), u32(n[5]), u32(n[6]), u32(n[8]), u32(n[9]), u32(n[10])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.n) * p.ho * p.wo * p.c);
}

// ints: x(N, H, W, C), out(N, C)
void runGlobalAvgPoolNhwc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t n, hw, c;
  } p{u32(a.ints[0]), u32(a.ints[1] * a.ints[2]), u32(a.ints[3])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.n) * p.c);
}

// ---- GEMM (steel, kernels/gpu/gemm_steel.metal) ------------------------------------------------------------

struct GemmPsos {
  MTL::ComputePipelineState* t64;
  MTL::ComputePipelineState* t32;
};

template <bool TB>
Status prepareGemmSteel(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new GemmPsos{pc.gpu->pipeline(TB ? "lk_gemm_nt_f32_64x64" : "lk_gemm_f32_64x64"),
                         pc.gpu->pipeline(TB ? "lk_gemm_nt_f32_32x32" : "lk_gemm_f32_32x32")};
  if (!s->t64 || !s->t32) {
    delete s;
    return Status::error("missing GPU kernel function lk_gemm_*");
  }
  *state = s;
  return Status::ok();
}
void destroyGemmSteel(void* s) { delete static_cast<GemmPsos*>(s); }

struct LkGemmParams {
  int M, N, K, lda, ldb, ldd, tiles_n, tiles_m, swizzle_log, gemm_k_iterations;
  uint32_t has_bias, act, has_res;
};

// ptrs: a, b, [bias], [res], out. ints: a(M, K), b(K, N) or b(N, K) (TB), [bias(N)], [res(M, N)], out(M, N).
template <bool TB, bool Bias, uint32_t Act, bool Res>
void runGemmSteel(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const int M = int(n[0]), K = int(n[1]), N = int(TB ? n[2] : n[3]);
  const int biasPtr = 2, resPtr = Bias ? 3 : 2, outPtr = 2 + (Bias ? 1 : 0) + (Res ? 1 : 0);
  const bool small = ((N + 63) / 64) * ((M + 63) / 64) < kSmallGrid;
  const int bm = small ? 32 : 64, tn = (N + bm - 1) / bm, tm = (M + bm - 1) / bm;
  LkGemmParams p{M, N, K, K, TB ? K : N, N, tn, tm, 0, K / 16, Bias ? 1u : 0u, Act, Res ? 1u : 0u};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, Bias ? biasPtr : outPtr);
  bindArg(g, 3, a, outPtr);
  bindArg(g, 4, a, Res ? resPtr : outPtr);
  g.bindParams(5, &p, sizeof p);
  const auto* ps = static_cast<const GemmPsos*>(a.state);
  g.dispatchThreadgroups(small ? ps->t32 : ps->t64, tn, tm, 1, 32, 2, 2);
}

// gemm_bias[_relu]_f32_gpu (the generic op). Kernels (gemm.metal): M <= 8 -> split-K small-M kernel (the tiled
// kernels leave the GPU idle: MNIST 784->128 at M = 1 took 192 us); 8 < M <= skMaxM -> steel tiles with split-K, the
// last split of each tile applying the epilogue (one dispatch; lk_gemm when nothing splits); above -> one kernel
// without split (kGemmBigCfg). The splits are chosen so that the grid has about `targetTgs` threadgroups. Env knobs
// (A/B runs, read at prepare): LOKINFER_GEMMB_LEGACY=1 (the 2026-11-01 dispatch), LOKINFER_GEMMB_TGS,
// LOKINFER_GEMMB_SKMAX, LOKINFER_GEMMB_MID / _BIG (index into kSkCfgs, or -1 lk_gemm, -3 the 32x32 BK 16 tiled
// kernel, -5 its BK 32 float4-load variant), _MINKC, _SMALLM (small-M kernel threshold).
struct SkCfg {
  const char* name;
  int bm, bn, bk;
};
constexpr SkCfg kSkCfgs[] = {{"32x32x16", 32, 32, 16}, {"64x64x16", 64, 64, 16}, {"64x32x16", 64, 32, 16},
                             {"64x32x32", 64, 32, 32}, {"32x64x16", 32, 64, 16}, {"32x32x32", 32, 32, 32},
                             {"16x32x16", 16, 32, 16}, {"16x64x16", 16, 64, 16}};
constexpr int kNumSkCfgs = int(sizeof kSkCfgs / sizeof kSkCfgs[0]);
// Defaults from interleaved A/B runs (2026-11-02, MNIST e2e medians). iPhone: target 32 (B=64 0.153-0.157 ms vs
// 0.159-0.164 at 64, 0.171 at 128), big M = the BK 32 tiled kernel (B=1024 0.248-0.269 vs lk_gemm 0.258-0.324, the
// BK 16 tiled kernel 0.273-0.275, 64x32 / 32x32 sk tiles 0.29-0.33, MPS 0.44). Mac: target 64 (128 / 256 no better),
// big M = lk_gemm (0.37 ms vs 0.48 BK 32 tiled, 0.66 BK 16 tiled).
#if TARGET_OS_IPHONE
constexpr int kGemmTargetTgs = 32, kGemmBigCfg = -5;
#else
constexpr int kGemmTargetTgs = 64, kGemmBigCfg = -1;  // -1: lk_gemm 64x64 / 32x32 (runGemmSteel)
#endif
// Split-K scratch per op instance: counters, then partials [splits, M, N] (a call that would need more uses fewer
// splits).
constexpr uint32_t kSkCounters = 4096;
constexpr size_t kSkPartFloats = size_t(1) << 18;

struct GemmBiasPsos {
  GemmPsos steel;
  MTL::ComputePipelineState* smallM;
  MTL::ComputePipelineState* tiled;
  MTL::ComputePipelineState* tiledK32 = nullptr;
  MTL::ComputePipelineState* sk[kNumSkCfgs];
  MTL::Buffer* scratch = nullptr;
  void* mem = nullptr;
  bool legacy = false;
  int targetTgs = kGemmTargetTgs, skMaxM = 256, midCfg = 0, bigCfg = kGemmBigCfg, minKc = 64, smallMaxM = 8;
};
// Row blocks of 8 handle any M, but the steel / tiled kernels win from M = 16 (Mac) and at M = 64 (iPhone:
// 0.19 vs 0.34 ms for MNIST B=64 with the small-M kernel), A/B 2026-11-01.
constexpr int kSmallM = 8;

int envInt(const char* name, int def) {
  const char* v = std::getenv(name);
  return v && *v ? std::atoi(v) : def;
}

template <uint32_t Act>
Status prepareGemmBias(PrepareCtx& pc, const Op& op, void** state) {
  void* st = nullptr;
  LK_RETURN_IF_ERROR(prepareGemmSteel<false>(pc, op, &st));
  auto* s = new GemmBiasPsos{*static_cast<GemmPsos*>(st),
                             pc.gpu->pipeline(Act ? "gemm_bias_relu_smallm_f32" : "gemm_bias_smallm_f32"),
                             pc.gpu->pipeline(Act ? "gemm_bias_relu_f32" : "gemm_bias_f32"),
                             {}};
  destroyGemmSteel(st);
  s->tiledK32 = pc.gpu->pipeline(Act ? "gemm_bias_relu_k32_f32" : "gemm_bias_k32_f32");
  bool ok = s->smallM && s->tiled && s->tiledK32;
  for (int i = 0; i < kNumSkCfgs; ++i) {
    s->sk[i] = pc.gpu->pipeline(std::string(Act ? "gemm_bias_relu_sk_f32_" : "gemm_bias_sk_f32_") + kSkCfgs[i].name);
    ok = ok && s->sk[i];
  }
  const size_t bytes = kSkCounters * sizeof(uint32_t) + kSkPartFloats * sizeof(float);
  if (ok && posix_memalign(&s->mem, 16384, bytes) == 0) {
    std::memset(s->mem, 0, bytes);
    s->scratch = pc.gpu->wrapNoCopy(s->mem, bytes);
  }
  if (!ok || !s->scratch) {
    if (s->scratch) pc.gpu->releaseBuffer(s->scratch);
    free(s->mem);
    delete s;
    return Status::error("missing GPU kernel function gemm_bias[_relu]_{smallm,sk}_f32* (or scratch alloc failed)");
  }
  s->legacy = envInt("LOKINFER_GEMMB_LEGACY", 0) != 0;
  s->targetTgs = envInt("LOKINFER_GEMMB_TGS", s->targetTgs);
  s->skMaxM = envInt("LOKINFER_GEMMB_SKMAX", s->skMaxM);
  s->midCfg = std::min(envInt("LOKINFER_GEMMB_MID", s->midCfg), kNumSkCfgs - 1);
  s->bigCfg = std::min(envInt("LOKINFER_GEMMB_BIG", s->bigCfg), kNumSkCfgs - 1);
  s->minKc = std::max(16, envInt("LOKINFER_GEMMB_MINKC", s->minKc));
  s->smallMaxM = envInt("LOKINFER_GEMMB_SMALLM", s->smallMaxM);

  *state = s;
  return Status::ok();
}
void destroyGemmBias(void* p) {
  auto* s = static_cast<GemmBiasPsos*>(p);
  if (GpuContext* g = GpuContext::existing()) {
    g->commitAndWait();  // the scratch may be referenced by a pending segment
    g->releaseBuffer(s->scratch);
  }
  free(s->mem);
  delete s;
}

struct GemmSkParams {
  uint32_t m, n, k, kc, splits, tiles_n, cnt_cap;
};

// Splits of K for a grid of `tiles` threadgroups: about targetTgs threadgroups in all, >= minKc of K per split
// (a multiple of 16), and the partials must fit the scratch.
GemmSkParams skParams(const GemmBiasPsos& s, uint32_t m, uint32_t n, uint32_t k, uint32_t tilesN, uint32_t tiles) {
  GemmSkParams p{m, n, k, k, 1, tilesN, kSkCounters};
  uint32_t want = tiles >= uint32_t(s.targetTgs) || tiles > kSkCounters ? 1 : uint32_t(s.targetTgs) / tiles;
  want = std::min<uint32_t>(want, std::max<uint32_t>(1, k / uint32_t(s.minKc)));
  while (want > 1 && size_t(want) * m * n > kSkPartFloats) --want;
  if (want > 1) {
    p.kc = (((k + want - 1) / want) + 15) / 16 * 16;
    p.splits = (k + p.kc - 1) / p.kc;
  }
  return p;
}

// ptrs: a, b, bias, out. ints: a(M, K), b(K, N), bias(N), out(M, N).
template <uint32_t Act>
void runGemmBias(ExecCtx& ctx, const KernelArgs& a) {
  const auto* ps = static_cast<const GemmBiasPsos*>(a.state);
  const int64_t M = a.ints[0];
  const bool smallM = M <= (ps->legacy ? kSmallM : ps->smallMaxM);
  int cfg = -2;  // -2 small-M, -1 runGemmSteel, -3 tiled, -5 tiled BK 32, else kSkCfgs index
  if (ps->legacy) {
#if TARGET_OS_IPHONE
    cfg = smallM ? -2 : M <= 256 ? -1 : -3;
#else
    cfg = smallM ? -2 : -1;
#endif
  } else if (!smallM) {
    cfg = M <= ps->skMaxM ? ps->midCfg : ps->bigCfg;
  }
  if (cfg >= 0 && !ps->legacy && M <= ps->skMaxM) {  // medium M with nothing to split: the plain steel GEMM (as before)
    const SkCfg& c = kSkCfgs[cfg];
    const uint32_t tn = u32((a.ints[3] + c.bn - 1) / c.bn), tm = u32((M + c.bm - 1) / c.bm);
    if (skParams(*ps, u32(M), u32(a.ints[3]), u32(a.ints[1]), tn, tn * tm).splits == 1) cfg = -1;
  }
  if (cfg == -1) {
    KernelArgs sa = a;
    sa.state = &ps->steel;
    runGemmSteel<false, true, Act, false>(ctx, sa);
    return;
  }
  GpuContext& g = *ctx.gpu;
  const uint32_t m = u32(M), n = u32(a.ints[3]), k = u32(a.ints[1]);
  g.beginDispatch();
  for (int i = 0; i < 4; ++i) bindArg(g, i, a, i);
  if (cfg == -3 || cfg == -5) {
    struct {
      uint32_t m, n, k;
    } p{m, n, k};
    g.bindParams(4, &p, sizeof p);
    g.dispatchThreadgroups(cfg == -3 ? ps->tiled : ps->tiledK32, (n + 31) / 32, (m + 31) / 32, 1, 128, 1, 1);
    return;
  }
  g.bind(5, ps->scratch, 0);
  if (cfg == -2) {
    const uint32_t tn = (n + 31) / 32, tm = (m + 7) / 8;
    GemmSkParams p = ps->legacy ? GemmSkParams{m, n, k, k, 1, tn, kSkCounters} : skParams(*ps, m, n, k, tn, tn * tm);
    g.bindParams(4, &p, sizeof p);
    g.dispatchThreadgroups(ps->smallM, tn, tm, p.splits, 32, 16, 1);
    return;
  }
  const SkCfg& c = kSkCfgs[cfg];
  const uint32_t tn = (n + c.bn - 1) / c.bn, tm = (m + c.bm - 1) / c.bm;
  GemmSkParams p = skParams(*ps, m, n, k, tn, tn * tm);
  if (p.splits > 1 && p.kc % c.bk) {  // keep each split a whole number of K tiles
    p.kc = (p.kc + c.bk - 1) / c.bk * c.bk;
    p.splits = (k + p.kc - 1) / p.kc;
  }
  g.bindParams(4, &p, sizeof p);
  g.dispatchThreadgroups(ps->sk[cfg], tn, tm, p.splits, 128, 1, 1);
}

// ---- elementwise / reductions -----------------------------------------------------------------------------

struct LcActParams {
  uint32_t l, c;
  float slope;
  uint32_t act, has_affine;
};

// Act: 1 snake (x, affine, alpha), 2 leaky with affine (x, affine), 3 leaky (x). ints start with x(L, C).
template <int Act>
void runActLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t l = u32(a.ints[0]), c = u32(a.ints[1]);
  LcActParams p{l, c, Act == 1 ? 0.f : static_cast<float>(a.floats[0]), Act == 1 ? 1u : 2u, Act == 3 ? 0u : 1u};
  const uint32_t outIdx = Act == 1 ? 3 : Act == 2 ? 2 : 1;
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, Act == 3 ? 0 : 1);
  bindArg(g, 2, a, Act == 1 ? 2 : 0);
  bindArg(g, 3, a, outIdx);
  g.bindParams(4, &p, sizeof p);
  dispatch1d(g, a, uint64_t(l) * c);
}

// ints: x(L, C), stats(S, C, 2)
void runStatsLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t l, c, chunks;
  } p{u32(a.ints[0]), u32(a.ints[1]), u32(a.ints[2])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (p.c + 31) / 32, p.chunks, 1, 32, 8, 1);  // lk_stats_lc_tg
}

// ints: stats(S, C, 2), gb(2 Creal), affine(3, C), length; floats: eps. Creal = gb / 2 (channels past it are
// zero padding).
void runAdainAffineLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t l, c, chunks;
    float eps;
    uint32_t creal;
  } p{u32(a.ints[6]), u32(a.ints[1]), u32(a.ints[0]), static_cast<float>(a.floats[0]), u32(a.ints[3] / 2)};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (p.c + 3) / 4, 1, 1, 128, 1, 1);  // one simdgroup per channel
}

// ptrs: y, b, [add], out. ints: y(Lin, K*Co), b(Co), [add(Lout, Co)], out(Lout, Co), kernel, padding,
// reflect_left, stride.
template <bool Add>
void runConvTGatherLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const int o = Add ? 5 : 3;
  struct {
    uint32_t lin, lout, co, k, stride, pad, reflect_left, has_add;
  } p{u32(n[0]), u32(n[o]), u32(n[o + 1]), u32(n[o + 2]), u32(n[o + 5]), u32(n[o + 3]), u32(n[o + 4]), Add ? 1u : 0u};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, Add ? 2 : 0);
  bindArg(g, 3, a, Add ? 3 : 2);
  g.bindParams(4, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.lout) * p.co);
}

// Same-shape a, b, c, out (rank = numInts / 4); floats: scale
void runAdd3Scale(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t rank = a.numInts / 4;
  int64_t count = 1;
  for (int64_t i = 0; i < rank; ++i) count *= a.ints[i];
  struct {
    uint32_t n;
    float s;
  } p{u32(count), static_cast<float>(a.floats[0])};
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  dispatch1d(g, a, p.n);
}

// ints: x(L, Cpad), out(cout, L), channels
void runLcToCf(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t l, cpad, cout;
  } p{u32(a.ints[0]), u32(a.ints[1]), u32(a.ints[4])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  g.dispatch(psoOf(a), p.l, p.cout, 1, 256, 1, 1);
}

// ptrs: x, out. ints: x(L, Cpad), out(len), hop, n_fft
void runIstftLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t nfft = u32(a.ints[4]);
  struct {
    uint32_t l, cpad, nfft, hop, pad, len;
  } p{u32(a.ints[0]), u32(a.ints[1]), nfft, u32(a.ints[3]), nfft / 2, u32(a.ints[2])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  g.dispatch(psoOf(a), p.len, 1, 1, 256, 1, 1);
}

// ptrs: a, [b], out. ints: a(L, Ca), [b(L, Cb)], out(L, Cout)
template <bool B>
void runConcatPadLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  struct {
    uint32_t l, ca, cb, cout;
  } p{u32(n[0]), u32(n[1]), B ? u32(n[3]) : 0u, u32(n[B ? 5 : 3])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, B ? 1 : 0);
  bindArg(g, 2, a, B ? 2 : 1);
  g.bindParams(3, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.l) * p.cout);
}

// ints: x(Lin, C), w(C, K), b(C), out(Lout, C), padding, stride
void runDwConvTLc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  struct {
    uint32_t lin, lout, c, k, stride, pad;
  } p{u32(n[0]), u32(n[5]), u32(n[1]), u32(n[3]), u32(n[8]), u32(n[7])};
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.lout) * p.c);
}

// ints: x(L, C), out(2L, C)
void runUpsample2Lc(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t l, c;
  } p{u32(a.ints[0]), u32(a.ints[1])};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  dispatch1d(g, a, 2 * uint64_t(p.l) * p.c);
}

// ---- token rate (kernels/gpu/kokoro_tok.metal) -------------------------------------------------------------

// ints: qkv(T, 3D), bias(3D), out(T, D), heads
template <bool Causal>
void runAttentionQkv(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t t = u32(a.ints[0]), d = u32(a.ints[4]), heads = u32(a.ints[5]);
  struct {
    uint32_t t, heads, head_dim;
    float scale;
    uint32_t causal;
  } p{t, heads, d / heads, 1.0f / std::sqrt(static_cast<float>(d / heads)), Causal ? 1u : 0u};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (t + 31) / 32, heads, 1, 128, 1, 1);  // lk_attention_qkv_flash: 32-query blocks
}

// ints: x(R, C), bias(C), out(R, C)
void runBiasGelu(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, cols;
    float eps;
  } p{u32(a.ints[0]), u32(a.ints[1]), 0.f};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  dispatch1d(g, a, uint64_t(p.rows) * p.cols);
}

// ints: a(R, C), bias(C), r(R, C), gamma(C), beta(C), out(R, C); floats: eps
void runAddBiasLn(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, cols;
    float eps;
  } p{u32(a.ints[0]), u32(a.ints[1]), static_cast<float>(a.floats[0])};
  if (p.cols > 2048) {  // the LayerNorm kernels keep <= 8 columns per thread x 256 threads in registers
    ctx.error = "GPU LayerNorm needs C <= 2048";
    return;
  }
  g.beginDispatch();
  for (uint32_t i = 0; i < 6; ++i) bindArg(g, i, a, i);
  g.bindParams(6, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.rows, 1, 1, 256, 1, 1);
}

// ints: a(R, C), bias(C), r(R, C), gamma(C), beta(C), sum(R, C), out(R, C); floats: eps
void runAddBiasLn2(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, cols;
    float eps;
  } p{u32(a.ints[0]), u32(a.ints[1]), static_cast<float>(a.floats[0])};
  if (p.cols > 2048) {  // the LayerNorm kernels keep <= 8 columns per thread x 256 threads in registers
    ctx.error = "GPU LayerNorm needs C <= 2048";
    return;
  }
  g.beginDispatch();
  for (uint32_t i = 0; i < 7; ++i) bindArg(g, i, a, i);
  g.bindParams(7, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.rows, 1, 1, 256, 1, 1);
}

// ints: x(R, C), gamma(C), beta(C), out(R, C); floats: eps
void runLayerNormRows(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, cols;
    float eps;
  } p{u32(a.ints[0]), u32(a.ints[1]), static_cast<float>(a.floats[0])};
  if (p.cols > 2048) {  // the LayerNorm kernels keep <= 8 columns per thread x 256 threads in registers
    ctx.error = "GPU LayerNorm needs C <= 2048";
    return;
  }
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.rows, 1, 1, 256, 1, 1);
}

struct CacheParams {
  uint32_t t, cap, d, heads;
  float scale;
};

// ptrs: kc, vc, qkv, pos, bias. ints: kc(P, D), vc(P, D), qkv(T, 3D), pos(1), bias(3D)
void runKvWriteRowsGpu(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  CacheParams p{u32(a.ints[4]), u32(a.ints[0]), u32(a.ints[1]), 0u, 0.f};
  g.beginDispatch();
  for (uint32_t i = 0; i < 5; ++i) bindArg(g, i, a, i);
  g.bindParams(5, &p, sizeof p);
  g.dispatch(psoOf(a), p.d, p.t, 1, 256, 1, 1);
}

// ptrs: qkv, kc, vc, pos, bias, out. ints: qkv(T, 3D), kc(P, D), vc(P, D), pos(1), bias(3D), out(T, D), heads
void runAttentionCacheGpu(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t t = u32(a.ints[0]), d = u32(a.ints[3]), heads = u32(a.ints[10]);
  CacheParams p{t, u32(a.ints[2]), d, heads, 1.0f / std::sqrt(static_cast<float>(d / heads))};
  g.beginDispatch();
  for (uint32_t i = 0; i < 6; ++i) bindArg(g, i, a, i);
  g.bindParams(6, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (t + 31) / 32, heads, 1, 128, 1, 1);
}

// ptrs: kc, vc, qkv, pos, bias. ints: kc(P, D), vc(P, D), qkv(1, 3D), pos(1), bias(3D)
void runKvWriteGpu(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t cap, d, heads;
    float scale;
  } p{u32(a.ints[0]), u32(a.ints[1]), 0u, 0.f};
  g.beginDispatch();
  for (uint32_t i = 0; i < 5; ++i) bindArg(g, i, a, i);
  g.bindParams(5, &p, sizeof p);
  dispatch1d(g, a, p.d);
}

// ptrs: qkv, kc, vc, pos, bias, out. ints: qkv(1, 3D), kc(P, D), vc(P, D), pos(1), bias(3D), out(1, D), heads
void runAttentionDecodeGpu(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t d = u32(a.ints[3]), heads = u32(a.ints[10]);
  struct {
    uint32_t cap, d, heads;
    float scale;
  } p{u32(a.ints[2]), d, heads, 1.0f / std::sqrt(static_cast<float>(d / heads))};
  g.beginDispatch();
  for (uint32_t i = 0; i < 6; ++i) bindArg(g, i, a, i);
  g.bindParams(6, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), heads, 1, 1, 128, 1, 1);
}

constexpr char kConv[] = "lk_conv1d_lc_f32";
constexpr char kAct[] = "lk_act_lc";
constexpr char kStats[] = "lk_stats_lc_tg";
constexpr char kAdain[] = "lk_adain_affine_lc";
constexpr char kConvT[] = "lk_convt_gather_lc";
constexpr char kAdd3[] = "lk_add3_scale";
constexpr char kToCf[] = "lk_lc_to_cf";
constexpr char kIstftLc[] = "lk_istft_lc";
constexpr char kConcat[] = "lk_concat_pad_lc";
constexpr char kDwConvT[] = "lk_dwconvt_lc";
constexpr char kUp2[] = "lk_upsample2_lc";
constexpr char kAttn[] = "lk_attention_qkv_flash";
constexpr char kKvWrite[] = "lk_kv_write";
constexpr char kKvWriteRows[] = "lk_kv_write_rows";
constexpr char kAttnCache[] = "lk_attention_cache_flash";
constexpr char kAttnDecode[] = "lk_attention_decode";
constexpr char kAddLn2[] = "lk_add_bias_ln2";
constexpr char kLnRows[] = "lk_layernorm_rows";
constexpr char kRowDot[] = "lk_rowdot";
constexpr char kGemv[] = "lk_gemv";
constexpr char kGemvRows[] = "lk_gemv_rows";
constexpr char kMaxPoolNhwc[] = "lk_maxpool_nhwc";
constexpr char kGapNhwc[] = "lk_global_avgpool_nhwc";
constexpr char kConvT2d[] = "lk_convt2d_gather_nhwc";
constexpr char kToNchw[] = "lk_nhwc_to_nchw";
constexpr char kSigmoid[] = "lk_sigmoid";
constexpr char kBiasGelu[] = "lk_bias_gelu";
constexpr char kAddLn[] = "lk_add_bias_ln";

}  // namespace

void registerGpuKokoroKernels() {
  registerKernel({"conv1d_lc_f32_gpu", Device::Gpu, prepareConv, runConvLc<false>, destroyConv});
  registerKernel({"conv1d_lc_res_f32_gpu", Device::Gpu, prepareConv, runConvLc<true>, destroyConv});
  registerKernel({"snake_affine_lc_f32_gpu", Device::Gpu, preparePso<kAct>, runActLc<1>, destroyPso});
  registerKernel({"leaky_affine_lc_f32_gpu", Device::Gpu, preparePso<kAct>, runActLc<2>, destroyPso});
  registerKernel({"leaky_lc_f32_gpu", Device::Gpu, preparePso<kAct>, runActLc<3>, destroyPso});
  registerKernel({"channel_stats_lc_f32_gpu", Device::Gpu, preparePso<kStats>, runStatsLc, destroyPso});
  registerKernel({"adain_affine_lc_f32_gpu", Device::Gpu, preparePso<kAdain>, runAdainAffineLc, destroyPso});
  registerKernel({"convt_gather_lc_f32_gpu", Device::Gpu, preparePso<kConvT>, runConvTGatherLc<false>, destroyPso});
  registerKernel({"convt_gather_add_lc_f32_gpu", Device::Gpu, preparePso<kConvT>, runConvTGatherLc<true>, destroyPso});
  registerKernel({"add3_scale_f32_gpu", Device::Gpu, preparePso<kAdd3>, runAdd3Scale, destroyPso});
  registerKernel({"lc_to_cf_f32_gpu", Device::Gpu, preparePso<kToCf>, runLcToCf, destroyPso});
  registerKernel({"istft_logits_lc_f32_gpu", Device::Gpu, preparePso<kIstftLc>, runIstftLc, destroyPso});
  registerKernel({"pad_cols_lc_f32_gpu", Device::Gpu, preparePso<kConcat>, runConcatPadLc<false>, destroyPso});
  registerKernel({"concat_pad_lc_f32_gpu", Device::Gpu, preparePso<kConcat>, runConcatPadLc<true>, destroyPso});
  registerKernel({"dwconvt_lc_f32_gpu", Device::Gpu, preparePso<kDwConvT>, runDwConvTLc, destroyPso});
  registerKernel({"upsample2_lc_f32_gpu", Device::Gpu, preparePso<kUp2>, runUpsample2Lc, destroyPso});
  registerKernel({"gemm_bias_f32_gpu", Device::Gpu, prepareGemmBias<0>, runGemmBias<0>, destroyGemmBias});
  registerKernel({"gemm_bias_relu_f32_gpu", Device::Gpu, prepareGemmBias<2>, runGemmBias<2>, destroyGemmBias});
  registerKernel({"gemm_steel_f32_gpu", Device::Gpu, prepareGemmSteel<false>, runGemmSteel<false, false, 0, false>,
                  destroyGemmSteel});
  registerKernel({"gemm_steel_bias_f32_gpu", Device::Gpu, prepareGemmSteel<false>, runGemmSteel<false, true, 0, false>,
                  destroyGemmSteel});
  registerKernel({"gemm_steel_bias_gelu_f32_gpu", Device::Gpu, prepareGemmSteel<false>,
                  runGemmSteel<false, true, 1, false>, destroyGemmSteel});
  registerKernel({"gemm_steel_bias_res_f32_gpu", Device::Gpu, prepareGemmSteel<false>,
                  runGemmSteel<false, true, 0, true>, destroyGemmSteel});
  registerKernel({"gemm_steel_bias_relu_f32_gpu", Device::Gpu, prepareGemmSteel<false>,
                  runGemmSteel<false, true, 2, false>, destroyGemmSteel});
  registerKernel({"gemv_f32_gpu", Device::Gpu, preparePso<kGemv>, runGemv<false, 0, false>, destroyPso});
  registerKernel({"gemv_bias_f32_gpu", Device::Gpu, preparePso<kGemv>, runGemv<true, 0, false>, destroyPso});
  registerKernel({"gemv_bias_gelu_f32_gpu", Device::Gpu, preparePso<kGemv>, runGemv<true, 1, false>, destroyPso});
  registerKernel({"gemv_rows_f32_gpu", Device::Gpu, preparePso<kGemvRows>, runGemvRows, destroyPso});
  registerKernel({"gemv1_bias_f32_gpu", Device::Gpu, preparePso<kRowDot>, runRowDot<0>, destroyPso});
  registerKernel({"gemv1_bias_sigmoid_f32_gpu", Device::Gpu, preparePso<kRowDot>, runRowDot<1>, destroyPso});
  registerKernel({"im2col_nchw_f32_gpu", Device::Gpu, prepareIm2col, runIm2colNchw, destroyIm2col});
  registerKernel({"gemm_nt_steel_f32_gpu", Device::Gpu, prepareGemmSteel<true>, runGemmSteel<true, false, 0, false>,
                  destroyGemmSteel});
  registerKernel({"conv2d_nhwc_f32_gpu", Device::Gpu, prepareConv, runConv2dNhwc<false, 0>, destroyConv});
  registerKernel({"conv2d_nhwc_relu_f32_gpu", Device::Gpu, prepareConv, runConv2dNhwc<false, 1>, destroyConv});
  registerKernel({"conv2d_nhwc_relu6_f32_gpu", Device::Gpu, prepareConv, runConv2dNhwc<false, 2>, destroyConv});
  registerKernel({"conv2d_nhwc_res_f32_gpu", Device::Gpu, prepareConv, runConv2dNhwc<true, 0>, destroyConv});
  registerKernel({"conv2d_nhwc_res_relu_f32_gpu", Device::Gpu, prepareConv, runConv2dNhwc<true, 1>, destroyConv});
  registerKernel({"conv2d_nhwc_leaky_f32_gpu", Device::Gpu, prepareConv, runConv2dNhwc<false, 3>, destroyConv});
  registerKernel({"convt2d_gather_nhwc_relu_f32_gpu", Device::Gpu, preparePso<kConvT2d>, runConvT2dGatherNhwc<1>,
                  destroyPso});
  registerKernel({"convt2d_gather_nhwc_tanh_f32_gpu", Device::Gpu, preparePso<kConvT2d>, runConvT2dGatherNhwc<4>,
                  destroyPso});
  registerKernel({"nhwc_to_nchw_f32_gpu", Device::Gpu, preparePso<kToNchw>, runNhwcToNchwGpu, destroyPso});
  registerKernel({"sigmoid_f32_gpu", Device::Gpu, preparePso<kSigmoid>, runSigmoidGpu, destroyPso});
  registerKernel({"dwconv2d_nhwc_f32_gpu", Device::Gpu, prepareDw, runDwConv2dNhwc<0>, destroyDw});
  registerKernel({"dwconv2d_nhwc_relu6_f32_gpu", Device::Gpu, prepareDw, runDwConv2dNhwc<2>, destroyDw});
  registerKernel({"nchw_to_nhwc_pad_f32_gpu", Device::Gpu, prepareNhwcPad, runNchwToNhwcPad, destroyNhwcPad});
  registerKernel({"maxpool2d_nhwc_f32_gpu", Device::Gpu, preparePso<kMaxPoolNhwc>, runMaxPoolNhwc, destroyPso});
  registerKernel({"global_avgpool_nhwc_f32_gpu", Device::Gpu, preparePso<kGapNhwc>, runGlobalAvgPoolNhwc,
                  destroyPso});
  registerKernel({"attention_qkv_f32_gpu", Device::Gpu, preparePso<kAttn>, runAttentionQkv<false>, destroyPso});
  registerKernel({"attention_qkv_causal_f32_gpu", Device::Gpu, preparePso<kAttn>, runAttentionQkv<true>, destroyPso});
  registerKernel({"add_bias_layernorm2_f32_gpu", Device::Gpu, preparePso<kAddLn2>, runAddBiasLn2, destroyPso});
  registerKernel({"kv_write_rows_f32_gpu", Device::Gpu, preparePso<kKvWriteRows>, runKvWriteRowsGpu, destroyPso});
  registerKernel({"attention_cache_f32_gpu", Device::Gpu, preparePso<kAttnCache>, runAttentionCacheGpu, destroyPso});
  registerKernel({"kv_write_f32_gpu", Device::Gpu, preparePso<kKvWrite>, runKvWriteGpu, destroyPso});
  registerKernel({"attention_decode_f32_gpu", Device::Gpu, preparePso<kAttnDecode>, runAttentionDecodeGpu, destroyPso});
  registerKernel({"layernorm_f32_gpu", Device::Gpu, preparePso<kLnRows>, runLayerNormRows, destroyPso});
  registerKernel({"bias_gelu_f32_gpu", Device::Gpu, preparePso<kBiasGelu>, runBiasGelu, destroyPso});
  registerKernel({"add_bias_layernorm_f32_gpu", Device::Gpu, preparePso<kAddLn>, runAddBiasLn, destroyPso});
}

}  // namespace lk::rt
