// Dispatch for the Winograd 1-D conv (kernels/gpu/winograd.metal, docs/impl-notes/winograd-conv.md):
// wino_tf[_snake]_lc_f32_gpu (input transform) + wino_mm[_res]_lc_f32_gpu (GEMMs + output transform + epilogue).
#include <Metal/Metal.hpp>

#include <cstdlib>
#include <string>

#include "../../src/runtime/gpu_internal.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

namespace {

constexpr uint32_t kTile = 4;  // M: outputs per Winograd tile
constexpr uint32_t kBT = 32;   // tiles per threadgroup (lk_wino_mm_lc, tile 4); rmax is a multiple of it (+ N - 1)
constexpr uint32_t kRowsTg = 128;  // output rows per threadgroup (and per stats chunk), any tile size
constexpr uint32_t kBN = 64;   // output channels per threadgroup
// Fewer 32-tile threadgroups than this -> 16-tile ones. 1-D, M2 Max, Kokoro word/app shapes (L = 1100): 5-15 %
// faster, neutral above (2026-11-01).
constexpr int64_t kSmallGrid = 128;
constexpr int64_t kSmallGrid2d = 128;
// 2-D: split the 16 points over grid z until the grid has this many threadgroups. M2 Max sweep (2026-11-02, ResNet
// 3x3 s1 + ReLU, splits 1..16): best splits were N=8 14x14x256: 1 (100 TGs), 7x7x512: 8; N=1 14x14x256: 8,
// 7x7x512: 16, 28x28x128: 4; target 100 picks all of them but N=8 7x7x512 (2: 1.11x instead of 1.20x).
// Needs iPhone validation (6 GPU cores).
constexpr int64_t kSplitTarget2d = 100;  // 2-D: fewer 32-tile threadgroups than this -> 16-tile ones

void bindArg(GpuContext& g, uint32_t slot, const KernelArgs& a, uint32_t i) {
  g.bind(slot, static_cast<MTL::Buffer*>(a.bufs[i].mtlBuffer), a.bufs[i].offset);
}

// Index 0-2: F(4,3) / nested F(4,4) with 2 / 3 groups; 3: F(2,3) (tile 2, 64-tile threadgroups).
struct Psos {
  MTL::ComputePipelineState* pso[4];
  MTL::ComputePipelineState* small[4];  // half-size threadgroups (lk_wino_mm_lc_*_bt16 / _bt32)
  MTL::ComputePipelineState* stats[4];  // + output channel stats (lk_wino_mm_lc_*_stats)
};

Status makePsos(PrepareCtx& pc, void** state, const char* a, const char* b, const char* c) {
  auto* s = new Psos{{pc.gpu->pipeline(a), pc.gpu->pipeline(b), c ? pc.gpu->pipeline(c) : nullptr}, {}, {}};
  if (c)
    for (int i = 0; i < 3; ++i) {
      s->small[i] = pc.gpu->pipeline(std::string(i == 0 ? a : i == 1 ? b : c) + "_bt16");
      s->stats[i] = pc.gpu->pipeline(std::string(i == 0 ? a : i == 1 ? b : c) + "_stats");
    }
  if (!s->pso[0] || !s->pso[1] ||
      (c && (!s->pso[2] || !s->small[0] || !s->small[1] || !s->small[2] || !s->stats[0] || !s->stats[1] ||
             !s->stats[2]))) {
    delete s;
    return Status::error(std::string("missing GPU kernel function ") + a + " / " + b);
  }
  *state = s;
  return Status::ok();
}
Status prepareTf(PrepareCtx& pc, const Op&, void** state) {
  LK_RETURN_IF_ERROR(makePsos(pc, state, "lk_wino_tf_lc_m4a6", "lk_wino_tf_lc_m4a7", nullptr));
  auto* s = static_cast<Psos*>(*state);
  s->pso[3] = pc.gpu->pipeline("lk_wino_tf_lc_m2a4");
  return s->pso[3] ? Status::ok() : Status::error("missing GPU kernel function lk_wino_tf_lc_m2a4");
}
Status prepareMm(PrepareCtx& pc, const Op&, void** state) {
  LK_RETURN_IF_ERROR(makePsos(pc, state, "lk_wino_mm_lc_m4a6n1", "lk_wino_mm_lc_m4a7n2", "lk_wino_mm_lc_m4a7n3"));
  auto* s = static_cast<Psos*>(*state);
  s->pso[3] = pc.gpu->pipeline("lk_wino_mm_lc_m2a4n1");
  s->small[3] = pc.gpu->pipeline("lk_wino_mm_lc_m2a4n1_bt32");
  s->stats[3] = pc.gpu->pipeline("lk_wino_mm_lc_m2a4n1_stats");
  return s->pso[3] && s->small[3] && s->stats[3] ? Status::ok()
                                                 : Status::error("missing GPU kernel function lk_wino_mm_lc_m2a4n1*");
}
void destroyPsos(void* s) { delete static_cast<Psos*>(s); }

struct TfParams {
  uint32_t l, c, d, pad, rmax, act;
  float slope;
};

// Act 0: ptrs x, tr, v. 1 (snake): x, tr, affine, alpha, v. 2 (leaky): x, tr, affine, v.
// ints: x(L, C), tr(T), [affine(3, C)], [alpha(C)], v(d*A*rmax, C), dilation, groups, kernel, padding, tile;
// floats (act 2): slope.
template <int Act>
void runTf(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const uint32_t L = uint32_t(n[0]), C = uint32_t(n[1]);
  constexpr bool Snake = Act == 1;
  const int o = Act == 1 ? 6 : Act == 2 ? 5 : 3;  // v dims
  const uint32_t R = uint32_t(n[o]), d = uint32_t(n[o + 2]), groups = uint32_t(n[o + 3]),
                 K = uint32_t(n[o + 4]), pad = uint32_t(n[o + 5]), tile = uint32_t(n[o + 6]);
  const uint32_t A = tile == 2 ? 4 : K == 3 ? 6 : 7;
  (void)groups;
  TfParams p{L, C, d, pad / d, R / (d * A), uint32_t(Act), Act == 2 ? static_cast<float>(a.floats[0]) : 0.f};
  const uint32_t vPtr = Act == 1 ? 4 : Act == 2 ? 3 : 2;
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, Act ? 2 : 0);
  bindArg(g, 3, a, Snake ? 3 : 0);
  bindArg(g, 4, a, vPtr);
  g.bindParams(5, &p, sizeof p);
  constexpr uint32_t kRT = 1;  // tiles per thread (lk_wino_tf_lc's RT; 4 was 2x slower on Kokoro rand510)
  g.dispatch(static_cast<const Psos*>(a.state)->pso[A == 4 ? 3 : A == 6 ? 0 : 1], C / 4, (p.rmax + kRT - 1) / kRT, d, 32, 4, 1);
}

struct MmParams {
  uint32_t lo, ci, co, d, rmax, has_res;
  float scale;
  uint32_t th, tw, ho, wo, act, chunks, splits;
};

// ptrs: v, u, b, tr, x, [res], out, [stats]. ints: v(R, Ci), u(A, N*Ci, Co), b(Co), tr(T), x(L, Ci),
// [res(Lo, Co)], out(Lo, Co), [stats(d*chunks, Co, 2)], dilation, groups, kernel, padding, tile; floats: [oscale].
// x only carries the input length.
template <bool Res, bool Stats>
void runMm(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const uint32_t R = uint32_t(n[0]), Ci = uint32_t(n[1]), A = uint32_t(n[2]), Co = uint32_t(n[4]);
  const int o = Res ? 11 : 9;          // out dims
  const int at = o + 2 + (Stats ? 3 : 0);  // attrs
  const uint32_t Lo = uint32_t(n[o]), d = uint32_t(n[at]), groups = uint32_t(n[at + 1]);
  const uint32_t tile = uint32_t(n[at + 4]), tbt = kRowsTg / tile;  // tiles per threadgroup (128 output rows)
  const uint32_t tiles = ((Lo + d - 1) / d + tile - 1) / tile;  // residue class 0 is the longest
  const auto* ps = static_cast<const Psos*>(a.state);
  const int k = A == 4 ? 3 : A == 6 ? 0 : int(groups) - 1;
  const bool small = !Stats && int64_t(Co / kBN) * ((tiles + tbt - 1) / tbt) * d < kSmallGrid;
  const uint32_t bt = small ? tbt / 2 : tbt, gy = (tiles + bt - 1) / bt;
  MmParams p{Lo, Ci, Co, d, R / (d * A), Res ? 1u : 0u, Res ? static_cast<float>(a.floats[0]) : 1.f, 0, 0, 0, 0, 0,
             gy, 1};
  const uint32_t outPtr = Res ? 6 : 5;
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, 2);
  bindArg(g, 3, a, Res ? 5 : outPtr);
  bindArg(g, 4, a, outPtr);
  bindArg(g, 5, a, 3);
  g.bindParams(6, &p, sizeof p);
  bindArg(g, 7, a, Stats ? outPtr + 1 : outPtr);
  MTL::ComputePipelineState* pso = Stats ? ps->stats[k] : small ? ps->small[k] : ps->pso[k];
  g.dispatchThreadgroups(pso, Co / kBN, gy, d, 128, 1, 1);
}

// ptrs: stats, gb, af. ints: stats(S, C, 2), gb(2 creal), af(3, C), dilation, length; floats: eps.
void runAffine(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const uint32_t S = uint32_t(n[0]), C = uint32_t(n[1]), d = uint32_t(n[6]), L = uint32_t(n[7]);
  struct {
    uint32_t l, c, d, chunks, rows;
    float eps;
    uint32_t creal;
  } p{L, C, d, S / d, kBT * kTile, static_cast<float>(a.floats[0]), uint32_t(n[3] / 2)};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(static_cast<const Psos*>(a.state)->pso[0], (C + 3) / 4, 1, 1, 128, 1, 1);
}
Status prepareAffine(PrepareCtx& pc, const Op&, void** state) {
  return makePsos(pc, state, "lk_wino_adain_affine_lc", "lk_wino_adain_affine_lc", nullptr);
}

// ---- 2-D F(2x2, 3x3), stride 1, padding 1, NHWC ----------------------------------------------------------------

Status prepareTf2(PrepareCtx& pc, const Op&, void** state) {
  return makePsos(pc, state, "lk_wino_tf2d_nhwc", "lk_wino_tf2d_nhwc", nullptr);
}
Status prepareMm2(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new Psos{{pc.gpu->pipeline("lk_wino_mm2d_nhwc"), nullptr, nullptr},
                     {pc.gpu->pipeline("lk_wino_mm2d_nhwc_bt16"), nullptr, nullptr},
                     {pc.gpu->pipeline("lk_wino_fill_zero"), pc.gpu->pipeline("lk_wino2d_finalize")}};
  if (!s->pso[0] || !s->small[0] || !s->stats[0] || !s->stats[1]) {
    delete s;
    return Status::error("missing GPU kernel function lk_wino_mm2d_nhwc[_bt16]");
  }
  *state = s;
  return Status::ok();
}

// ptrs: x, bt, v. ints: x(N, H, W, C), bt(16), v(16*rmax, C)
void runTf2(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  struct {
    uint32_t n, h, w, c, th, tw, rmax;
  } p{uint32_t(n[0]), uint32_t(n[1]), uint32_t(n[2]), uint32_t(n[3]), uint32_t((n[1] + 1) / 2),
      uint32_t((n[2] + 1) / 2), uint32_t(n[5] / 16)};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatch(static_cast<const Psos*>(a.state)->pso[0], p.c / 4, p.rmax, 1, 32, 4, 1);
}

// ptrs: v, u, b, tr, x, [res], out. ints: v(16*rmax, Ci), u(16, Ci, Co), b(Co), tr(320), x(N, H, W, Ci),
// [res(N, H, W, Co)], out(N, H, W, Co)
template <bool Res, bool Relu>
void runMm2(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int64_t* n = a.ints;
  const uint32_t R = uint32_t(n[0]), Ci = uint32_t(n[1]), Co = uint32_t(n[4]);
  const uint32_t N = uint32_t(n[7]), H = uint32_t(n[8]), W = uint32_t(n[9]);
  const uint32_t th = (H + 1) / 2, tw = (W + 1) / 2, tiles = N * th * tw;
  const auto* ps = static_cast<const Psos*>(a.state);
  const int64_t grid32 = int64_t(Co / kBN) * ((tiles + kBT - 1) / kBT);
  const bool small = grid32 < kSmallGrid2d;
  const uint32_t bt = small ? 16 : kBT, gy = (tiles + bt - 1) / bt;
  // Small grids: split the 16 points over grid z (partial sums atomically added into a zeroed output).
  uint32_t splits = 1;
  while (splits < 16 && int64_t(Co / kBN) * gy * splits < kSplitTarget2d) splits *= 2;
  MmParams p{tiles, Ci, Co, 1, R / 16, Res ? 1u : 0u, 1.f, th, tw, H, W, Relu ? 1u : 0u, 0, splits};
  const uint32_t outPtr = Res ? 6 : 5;
  const uint32_t count = N * H * W * Co;
  if (splits > 1) {
    const uint32_t n4 = count / 4;
    g.beginDispatch();
    bindArg(g, 0, a, outPtr);
    g.bindParams(1, &n4, sizeof n4);
    g.dispatch(ps->stats[0], n4, 1, 1, 256, 1, 1);
  }
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  bindArg(g, 2, a, 2);
  bindArg(g, 3, a, Res ? 5 : outPtr);
  bindArg(g, 4, a, outPtr);
  bindArg(g, 5, a, 3);
  g.bindParams(6, &p, sizeof p);
  bindArg(g, 7, a, outPtr);
  g.dispatchThreadgroups(small ? ps->small[0] : ps->pso[0], Co / kBN, gy, splits, 128, 1, 1);
  if (splits > 1) {
    struct {
      uint32_t count, co, has_res, act;
    } f{count / 4, Co, Res ? 1u : 0u, Relu ? 1u : 0u};
    g.beginDispatch();
    bindArg(g, 0, a, outPtr);
    bindArg(g, 1, a, 2);
    bindArg(g, 2, a, Res ? 5 : outPtr);
    g.bindParams(3, &f, sizeof f);
    g.dispatch(ps->stats[1], f.count, 1, 1, 256, 1, 1);
  }
}

}  // namespace

void registerGpuWinogradKernels() {
  registerKernel({"wino_tf2d_nhwc_f32_gpu", Device::Gpu, prepareTf2, runTf2, destroyPsos});
  registerKernel({"wino_mm2d_nhwc_f32_gpu", Device::Gpu, prepareMm2, runMm2<false, false>, destroyPsos});
  registerKernel({"wino_mm2d_nhwc_relu_f32_gpu", Device::Gpu, prepareMm2, runMm2<false, true>, destroyPsos});
  registerKernel({"wino_mm2d_nhwc_res_f32_gpu", Device::Gpu, prepareMm2, runMm2<true, false>, destroyPsos});
  registerKernel({"wino_mm2d_nhwc_res_relu_f32_gpu", Device::Gpu, prepareMm2, runMm2<true, true>, destroyPsos});
  registerKernel({"wino_tf_lc_f32_gpu", Device::Gpu, prepareTf, runTf<0>, destroyPsos});
  registerKernel({"wino_tf_snake_lc_f32_gpu", Device::Gpu, prepareTf, runTf<1>, destroyPsos});
  registerKernel({"wino_tf_leaky_lc_f32_gpu", Device::Gpu, prepareTf, runTf<2>, destroyPsos});
  registerKernel({"wino_mm_lc_f32_gpu", Device::Gpu, prepareMm, runMm<false, false>, destroyPsos});
  registerKernel({"wino_mm_res_lc_f32_gpu", Device::Gpu, prepareMm, runMm<true, false>, destroyPsos});
  registerKernel({"wino_mm_stats_lc_f32_gpu", Device::Gpu, prepareMm, runMm<false, true>, destroyPsos});
  registerKernel({"wino_mm_res_stats_lc_f32_gpu", Device::Gpu, prepareMm, runMm<true, true>, destroyPsos});
  registerKernel({"wino_adain_affine_lc_f32_gpu", Device::Gpu, prepareAffine, runAffine, destroyPsos});
}

}  // namespace lk::rt
