// Dispatch for lokinfer's transformer GPU kernels (GPT-2; kernels/gpu/kokoro_tok.metal) that are not shared with
// Kokoro's dispatch file (kernels/gpu/kokoro_lc.cpp).
#include <Metal/Metal.hpp>
#include <TargetConditionals.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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

// Kernels with tile variants picked per op by an int attribute (attention: simdgroups per threadgroup, "nsg";
// gemv4: float4 columns per threadgroup, "cw"): every variant's pipeline is prepared, the run picks one.
struct VariantPsos {
  MTL::ComputePipelineState* pso[17] = {};  // indexed by the variant value (1..16)
};

Status prepareVariants(PrepareCtx& pc, void** state, const std::string& prefix, std::initializer_list<uint32_t> vs) {
  auto* s = new VariantPsos();
  for (uint32_t v : vs) {
    const std::string fn = prefix + std::to_string(v);
    s->pso[v] = pc.gpu->pipeline(fn);
    if (!s->pso[v]) {
      delete s;
      return Status::error("missing GPU kernel function " + fn);
    }
  }
  *state = s;
  return Status::ok();
}
Status prepareAttnSg(PrepareCtx& pc, const Op&, void** state) {
  return prepareVariants(pc, state, "lk_attention_qkv_sg", {2, 4, 8});
}
Status prepareGemv4(PrepareCtx& pc, const Op&, void** state) {
  return prepareVariants(pc, state, "lk_gemv4_c", {2, 4, 8, 16});
}
void destroyVariants(void* s) { delete static_cast<VariantPsos*>(s); }
MTL::ComputePipelineState* variantPso(const KernelArgs& a, int64_t v) {
  return v >= 1 && v <= 16 ? static_cast<const VariantPsos*>(a.state)->pso[v] : nullptr;
}

void destroyPso(void* s) { delete static_cast<PsoState*>(s); }

MTL::ComputePipelineState* psoOf(const KernelArgs& a) { return static_cast<const PsoState*>(a.state)->pso; }

void bindArg(GpuContext& g, uint32_t slot, const KernelArgs& a, uint32_t i) {
  g.bind(slot, static_cast<MTL::Buffer*>(a.bufs[i].mtlBuffer), a.bufs[i].offset);
}

uint32_t u32(int64_t v) { return static_cast<uint32_t>(v); }

// ints: qkv(T, 3D), bias(3D), out(T, D), heads, nsg
template <bool Causal>
void runAttentionQkvSg(ExecCtx& ctx, const KernelArgs& a) {
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
  const uint32_t nsg = u32(a.ints[6]), bq = 8 * nsg;
  g.dispatchThreadgroups(variantPso(a, nsg), (t + bq - 1) / bq, heads, 1, 32 * nsg, 1, 1);
}


// ptrs: qkv, kc, vc, pos, bias, out. ints: qkv(1, 3D), kc(P, D), vc(P, D), pos(1), bias(3D), out(1, D), heads
void runAttentionDecodeKvw(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t d = u32(a.ints[3]), heads = u32(a.ints[10]);
  struct {
    uint32_t cap, d, heads;
    float scale;
  } p{u32(a.ints[2]), d, heads, 1.0f / std::sqrt(static_cast<float>(d / heads))};
  g.beginDispatch();
  for (uint32_t i = 0; i < 6; ++i) bindArg(g, i, a, i);
  g.bindParams(6, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), heads, 1, 1, 1024, 1, 1);
}

// ptrs: qkv, kc, vc, pos, bias, out. ints: qkv(T, 3D), kc(P, D), vc(P, D), pos(1), bias(3D), out(T, D), heads, nsg
// (lk_attention_cache_sg: chunked-prefill queries at pos.. against the KV cache, simdgroup-matrix flash).
Status prepareCacheSg(PrepareCtx& pc, const Op&, void** state) {
  return prepareVariants(pc, state, "lk_attention_cache_sg", {2, 4, 8});
}
void runAttentionCacheSg(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t t = u32(a.ints[0]), d = u32(a.ints[3]), heads = u32(a.ints[10]);
  struct {
    uint32_t t, cap, heads;
    float scale;
  } p{t, u32(a.ints[2]), heads, 1.0f / std::sqrt(static_cast<float>(d / heads))};
  g.beginDispatch();
  for (uint32_t i = 0; i < 6; ++i) bindArg(g, i, a, i);
  g.bindParams(6, &p, sizeof p);
  const uint32_t nsg = u32(a.ints[11]), bq = 8 * nsg;
  g.dispatchThreadgroups(variantPso(a, nsg), (t + bq - 1) / bq, heads, 1, 32 * nsg, 1, 1);
}

// Decode attention with the keys split over `splits` threadgroups per head + a combine dispatch
// (lk_attention_decode_split{nsg} / lk_attention_decode_combine). The partials (heads x splits x 68 floats) live in
// a scratch buffer owned by the op instance (one per layer, so no cross-layer reuse hazard).
constexpr uint32_t kDecodeSplitMaxHeads = 16, kDecodeSplitMax = 32, kDecodePartFloats = 68;
// splits = 0 (auto): below kDecodeSplitMinKeys cached keys (pos + 1, read at encode time) the single-threadgroup
// lk_attention_decode_kvw, above it `kDecodeSplitAuto` splits of kDecodeSplitNsgAuto simdgroups. A/B 2026-11-02
// (GPT-2 decode e2e, M2 Max): P=16 kvw 1.87 vs split 4x16 1.92 ms; P=512 2.02 vs 2.02; P=1000 2.23-2.29 vs
// 2.17-2.25 (8x8 / 4x8 / 8x16 / 16x4 never better; a second combine dispatch costs the same as the in-kernel one).
#if TARGET_OS_IPHONE
constexpr int64_t kDecodeSplitAuto = 4, kDecodeSplitNsgAuto = 16, kDecodeSplitMinKeys = 1 << 30;
#else
constexpr int64_t kDecodeSplitAuto = 4, kDecodeSplitNsgAuto = 16, kDecodeSplitMinKeys = 768;
#endif
struct DecodeSplitState {
  MTL::ComputePipelineState* split[33] = {};  // by nsg
  MTL::ComputePipelineState* fused[33] = {};  // by nsg: single dispatch, last split per head combines
  MTL::ComputePipelineState* combine = nullptr;
  MTL::ComputePipelineState* kvw = nullptr;  // auto policy, short contexts
  MTL::Buffer* part = nullptr;
  void* mem = nullptr;
};
Status prepareDecodeSplit(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new DecodeSplitState();
  for (uint32_t v : {4u, 8u, 16u, 32u}) {
    s->split[v] = pc.gpu->pipeline("lk_attention_decode_split" + std::to_string(v));
    s->fused[v] = pc.gpu->pipeline("lk_attention_decode_splitf" + std::to_string(v));
  }
  s->combine = pc.gpu->pipeline("lk_attention_decode_combine");
  s->kvw = pc.gpu->pipeline("lk_attention_decode_kvw");
  // partials + one zero-initialized counter per head (fused mode; the last split resets it)
  const size_t bytes = (size_t(kDecodeSplitMaxHeads) * kDecodeSplitMax * kDecodePartFloats + kDecodeSplitMaxHeads) *
                       sizeof(float);
  if (posix_memalign(&s->mem, 16384, bytes) == 0) {
    std::memset(s->mem, 0, bytes);
    s->part = pc.gpu->wrapNoCopy(s->mem, bytes);
  }
  // nsg 32 = 1024 threads: register pressure can cap the pipeline below that; the run then uses 16.
  if (s->split[32] && s->split[32]->maxTotalThreadsPerThreadgroup() < 1024) s->split[32] = nullptr;
  if (s->fused[32] && s->fused[32]->maxTotalThreadsPerThreadgroup() < 1024) s->fused[32] = nullptr;
  if (!s->split[4] || !s->split[8] || !s->split[16] || !s->fused[4] || !s->fused[8] || !s->fused[16] ||
      !s->combine || !s->kvw || !s->part) {
    if (s->part) pc.gpu->releaseBuffer(s->part);
    free(s->mem);
    delete s;
    return Status::error("missing GPU kernel function lk_attention_decode_split* / combine (or scratch alloc failed)");
  }
  *state = s;
  return Status::ok();
}
void destroyDecodeSplit(void* p) {
  auto* s = static_cast<DecodeSplitState*>(p);
  if (GpuContext* g = GpuContext::existing()) {
    g->commitAndWait();  // the scratch may be referenced by a pending segment
    g->releaseBuffer(s->part);
  }
  free(s->mem);
  delete s;
}

// ptrs: qkv, kc, vc, pos, bias, out. ints: qkv(1, 3D), kc(P, D), vc(P, D), pos(1), bias(3D), out(1, D), heads, nsg,
// single, splits
void runAttentionDecodeSplit(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const auto* s = static_cast<const DecodeSplitState*>(a.state);
  const uint32_t d = u32(a.ints[3]), heads = u32(a.ints[10]);
  int64_t nsg = a.ints[11], splits = a.ints[13];
  if (nsg == 0) nsg = kDecodeSplitNsgAuto;
  if (splits == 0) {
    if (*static_cast<const int64_t*>(a.ptrs[3]) + 1 < kDecodeSplitMinKeys) {  // short context: one threadgroup / head
      struct {
        uint32_t cap, d, heads;
        float scale;
      } q{u32(a.ints[2]), d, heads, 1.0f / std::sqrt(static_cast<float>(d / heads))};
      g.beginDispatch();
      for (uint32_t i = 0; i < 6; ++i) bindArg(g, i, a, i);
      g.bindParams(6, &q, sizeof q);
      g.dispatchThreadgroups(s->kvw, heads, 1, 1, 1024, 1, 1);
      return;
    }
    splits = kDecodeSplitAuto;
  }
  if (!s->split[nsg]) nsg = 16;
  struct {
    uint32_t cap, d, heads;
    float scale;
    uint32_t splits;
  } p{u32(a.ints[2]), d, heads, 1.0f / std::sqrt(static_cast<float>(d / heads)), u32(splits)};
  const bool fused = a.ints[12] != 0;
  g.beginDispatch();
  for (uint32_t i = 0; i < 5; ++i) bindArg(g, i, a, i);
  g.bind(5, s->part, 0);
  g.bindParams(6, &p, sizeof p);
  bindArg(g, 7, a, 5);
  g.dispatchThreadgroups((fused ? s->fused : s->split)[nsg], heads, u32(splits), 1, 32 * u32(nsg), 1, 1);
  if (fused) return;
  g.beginDispatch();  // a later dispatch of the same kernel: barrier by default
  g.bind(0, s->part, 0);
  bindArg(g, 1, a, 5);
  g.bindParams(2, &p, sizeof p);
  g.dispatchThreadgroups(s->combine, heads, 1, 1, 32, 1, 1);
}

// cw = 0 (auto): per-platform default, A/B 2026-11-01 (GPT-2 decode P=512): M2 Max cw 4 (2.2 ms vs 2.6 for 8/16,
// MPS 2.8), iPhone 18 Pro Max cw 8 (5.60 ms vs 6.01 for 4, MPS 6.91).
#if TARGET_OS_IPHONE
constexpr int64_t kGemv4AutoCw = 8;
#else
constexpr int64_t kGemv4AutoCw = 4;
#endif

// ptrs: x, w, [bias], [res], out. ints: x(1, K), w(K, N), ..., cw (last)
template <bool Bias, uint32_t Act, bool Res>
void runGemv4(ExecCtx& ctx, const KernelArgs& a) {
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
  int64_t cw = a.ints[a.numInts - 1];
  if (cw == 0) cw = kGemv4AutoCw;
  const uint32_t cols = 4 * u32(cw);
  g.dispatchThreadgroups(variantPso(a, cw), (p.n + cols - 1) / cols, 1, 1, 256, 1, 1);
}

// ---- split-K GEMM partials (gemm_steel.metal lk_gemm_splitk) + their consumer (kokoro_tok.metal) -----------------

// Split-K only pays while the output tiles underfill the GPU (Mac GPT-2 T = 128: 24 tiles of 64x64 on 30 cores, the
// two N = 768 GEMMs per block 2.0 -> 1.55 ms in total; T = 512: 96 tiles, split-K steel is slower than MPS). Above
// the threshold the op runs MPS into plane 0 and the consumer reads one plane. Rule shared by both kernels (the
// consumer only knows M and N). A/B 2026-11-01; re-checked 2026-11-02 with the threshold forced off at T = 512:
// 64x64 split-K 17.8 ms vs MPS 17.2-17.5, 32x32 19.0, sk (2, 2) 18.4 -> threshold kept.
#if TARGET_OS_IPHONE
constexpr int64_t kSplitKMaxTiles = 0;  // iPhone 18 Pro Max (6 GPU cores): MPS at every M (T = 128: 11.43 vs 11.89 ms)
#else
constexpr int64_t kSplitKMaxTiles = 48;
#endif
// M = 1 (decode GEMV) always writes one plane: build_generate's gemm_auto_sk runs gemv4 there.
int64_t splitKEffective(int64_t m, int64_t n, int64_t smax) {
  return m > 1 && ((m + 63) / 64) * ((n + 63) / 64) < kSplitKMaxTiles ? smax : 1;
}

struct SplitKPsos {
  MTL::ComputePipelineState* t64;
  MTL::ComputePipelineState* t32;
  const KernelEntry* mps;
  void* mpsState;
};
Status prepareSplitK(PrepareCtx& pc, const Op& op, void** state) {
  auto* s = new SplitKPsos{pc.gpu->pipeline("lk_gemm_splitk_f32_64x64"), pc.gpu->pipeline("lk_gemm_splitk_f32_32x32"),
                           findKernel("gemm_f32_gpu_mps"), nullptr};
  if (!s->t64 || !s->t32 || !s->mps) {
    delete s;
    return Status::error("missing GPU kernel function lk_gemm_splitk_f32_* or gemm_f32_gpu_mps");
  }
  if (s->mps->prepare) {
    Status st = s->mps->prepare(pc, op, &s->mpsState);
    if (!st.isOk()) {
      delete s;
      return st;
    }
  }
  *state = s;
  return Status::ok();
}
void destroySplitK(void* p) {
  auto* s = static_cast<SplitKPsos*>(p);
  if (s->mps && s->mps->destroy && s->mpsState) s->mps->destroy(s->mpsState);
  delete s;
}

// ptrs: a, b, out. ints: a(M, K), b(K, N), out(S, M, N), splits, tile (64 or 32)
void runGemmSplitK(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const int M = int(a.ints[0]), K = int(a.ints[1]), N = int(a.ints[3]);
  const auto* ps = static_cast<const SplitKPsos*>(a.state);
  const int S = int(splitKEffective(M, N, a.ints[7])), bm = a.ints[8] == 32 ? 32 : 64;
  if (S == 1 && a.ints[7] > 1) {  // big M: MPS into plane 0 (its ints a(M, K), b(K, N) come first, as here)
    KernelArgs ma = a;
    ma.state = ps->mpsState;
    ps->mps->run(ctx, ma);
    return;
  }
  const int kc = K / S, tn = (N + bm - 1) / bm, tm = (M + bm - 1) / bm;
  struct {
    int M, N, K, lda, ldb, ldd, tiles_n, tiles_m, k_iters, kc;
  } p{M, N, K, K, N, N, tn, tm, kc / 16, kc};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(bm == 32 ? ps->t32 : ps->t64, u32(tn), u32(tm), u32(S), 32, 2, 2);
}

// ptrs: parts, bias, r, gamma, beta, sum, out. ints: parts(S, R, C), ... ; floats: eps
void runAddBiasLn2Sk(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, cols;
    float eps;
    uint32_t splits;
  } p{u32(a.ints[1]), u32(a.ints[2]), static_cast<float>(a.floats[0]),
      u32(splitKEffective(a.ints[1], a.ints[2], a.ints[0]))};
  g.beginDispatch();
  for (uint32_t i = 0; i < 7; ++i) bindArg(g, i, a, i);
  g.bindParams(7, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.rows, 1, 1, 256, 1, 1);
}

constexpr char kAddLn2Sk[] = "lk_add_bias_ln2_sk";
// ---- runtime-dispatched ops for build_generate (one graph, T symbolic: T = 1 decode steps, T > 1 prefill) -------
// Each composite op prepares the registered kernels it delegates to and picks per call from the runtime T (M):
// T = 1 -> the decode kernels (gemv4_*, attention_decode_kvw), T > 1 -> the prefill ones (MPS, kv_write_rows +
// attention_cache).

struct SubKernel {
  const KernelEntry* e = nullptr;
  void* st = nullptr;
};
struct SubKernels {
  SubKernel k[4];
};
Status prepareSubs(PrepareCtx& pc, const Op& op, void** state, std::initializer_list<const char*> names) {
  auto* s = new SubKernels();
  int i = 0;
  for (const char* n : names) {
    SubKernel& sk = s->k[i++];
    sk.e = findKernel(n);
    Status st = sk.e ? (sk.e->prepare ? sk.e->prepare(pc, op, &sk.st) : Status::ok())
                     : Status::error(std::string("missing kernel ") + n);
    if (!st.isOk()) {
      for (auto& o : s->k)
        if (o.e && o.e->destroy && o.st) o.e->destroy(o.st);
      delete s;
      return st;
    }
  }
  *state = s;
  return Status::ok();
}
void destroySubs(void* p) {
  auto* s = static_cast<SubKernels*>(p);
  for (auto& o : s->k)
    if (o.e && o.e->destroy && o.st) o.e->destroy(o.st);
  delete s;
}
void runSub(ExecCtx& ctx, const SubKernel& sk, const KernelArgs& base, std::initializer_list<int> argIdx,
            std::initializer_list<int64_t> ints) {
  void* ptrs[8];
  BufRef bufs[8];
  int64_t iv[16];
  uint32_t np = 0, ni = 0;
  for (int i : argIdx) ptrs[np] = base.ptrs[i], bufs[np] = base.bufs[i], ++np;
  for (int64_t v : ints) iv[ni++] = v;
  KernelArgs a{ptrs, bufs, iv, base.floats, sk.st, np, ni};
  sk.e->run(ctx, a);
}

Status prepareGemmAuto(PrepareCtx& pc, const Op& op, void** state) {
  return prepareSubs(pc, op, state, {"gemv4_f32_gpu", "gemm_f32_gpu_mps"});
}
Status prepareGemmAutoGelu(PrepareCtx& pc, const Op& op, void** state) {
  return prepareSubs(pc, op, state, {"gemv4_bias_gelu_f32_gpu", "gemm_f32_gpu_mps", "bias_gelu_f32_gpu"});
}

// ptrs: x, w, out. ints: x(M, K), w(K, N), out(M, N)
void runGemmAuto(ExecCtx& ctx, const KernelArgs& a) {
  const auto* s = static_cast<const SubKernels*>(a.state);
  const int64_t m = a.ints[0], k = a.ints[1], n = a.ints[3];
  if (m == 1) return runSub(ctx, s->k[0], a, {0, 1, 2}, {1, k, k, n, 1, n, 0});
  runSub(ctx, s->k[1], a, {0, 1, 2}, {m, k, k, n, m, n});
}

// ptrs: x, w, bias, out. ints: x(M, K), w(K, N), bias(N), out(M, N)
void runGemmAutoGelu(ExecCtx& ctx, const KernelArgs& a) {
  const auto* s = static_cast<const SubKernels*>(a.state);
  const int64_t m = a.ints[0], k = a.ints[1], n = a.ints[3];
  if (m == 1) return runSub(ctx, s->k[0], a, {0, 1, 2, 3}, {1, k, k, n, n, 1, n, 0});
  runSub(ctx, s->k[1], a, {0, 1, 3}, {m, k, k, n, m, n});
  runSub(ctx, s->k[2], a, {3, 2, 3}, {m, n, n, m, n});  // in place: out = gelu(out + bias)
}

Status prepareGemmAutoSk(PrepareCtx& pc, const Op& op, void** state) {
  return prepareSubs(pc, op, state, {"gemv4_f32_gpu", "gemm_splitk_f32_gpu"});
}

// build_generate's N = 768 projections: M = 1 -> gemv4 into plane 0; M > 1 -> split-K partials (MPS into plane 0
// once the tiles fill the GPU); the consumer add_bias_layernorm2_sk reads splitKEffective(M, N, S) planes.
// ptrs: x, w, out. ints: x(M, K), w(K, N), out(S, M, N), splits, tile
void runGemmAutoSk(ExecCtx& ctx, const KernelArgs& a) {
  const auto* s = static_cast<const SubKernels*>(a.state);
  const int64_t m = a.ints[0], k = a.ints[1], n = a.ints[3];
  if (m == 1) return runSub(ctx, s->k[0], a, {0, 1, 2}, {1, k, k, n, 1, n, 0});
  runSub(ctx, s->k[1], a, {0, 1, 2}, {m, k, k, n, a.ints[4], m, n, a.ints[7], a.ints[8]});
}

Status prepareAttnGen(PrepareCtx& pc, const Op& op, void** state) {
  return prepareSubs(pc, op, state, {"attention_decode_kvw_f32_gpu", "kv_write_rows_f32_gpu", "attention_cache_f32_gpu",
                                     "attention_qkv_sg_causal_f32_gpu"});
}
Status prepareAttnGen2(PrepareCtx& pc, const Op& op, void** state) {
  return prepareSubs(pc, op, state, {"attention_decode_split_f32_gpu", "kv_write_rows_f32_gpu",
                                     "attention_cache_sg_f32_gpu", "attention_qkv_sg_causal_f32_gpu"});
}

// ptrs: qkv, kc, vc, pos, bias, out. ints: qkv(T, 3D), kc(P, D), vc(P, D), pos(1), bias(3D), out(T, D), heads
void runAttnGen(ExecCtx& ctx, const KernelArgs& a) {
  const auto* s = static_cast<const SubKernels*>(a.state);
  const int64_t* n = a.ints;
  if (n[0] == 1) return runSub(ctx, s->k[0], a, {0, 1, 2, 3, 4, 5}, {n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7],
                                                                      n[8], n[9], n[10]});
  runSub(ctx, s->k[1], a, {1, 2, 0, 3, 4}, {n[2], n[3], n[4], n[5], n[0], n[1], n[6], n[7]});
  // A prompt prefilled from position 0 (pos is a graph input, already written at encode time) only attends to its
  // own rows: the simdgroup-matrix flash attention over qkv (build()'s prefill kernel) instead of the cache scan.
  if (*static_cast<const int64_t*>(a.ptrs[3]) == 0) {
    return runSub(ctx, s->k[3], a, {0, 4, 5}, {n[0], n[1], n[7], n[8], n[9], n[10], 4});
  }
  runSub(ctx, s->k[2], a, {0, 1, 2, 3, 4, 5}, {n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9], n[10]});
}

// attention_gen2 (default of build_generate): T = 1 -> the key-split decode attention (+ combine), T > 1 ->
// kv_write_rows, then the simdgroup-matrix flash attention: over qkv when the chunk starts at pos 0, else over the
// cache rows (lk_attention_cache_sg, later prefill chunks).
void runAttnGen2(ExecCtx& ctx, const KernelArgs& a) {
  const auto* s = static_cast<const SubKernels*>(a.state);
  const int64_t* n = a.ints;
  if (n[0] == 1) return runSub(ctx, s->k[0], a, {0, 1, 2, 3, 4, 5}, {n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7],
                                                                      n[8], n[9], n[10], 0, 1, 0});
  runSub(ctx, s->k[1], a, {1, 2, 0, 3, 4}, {n[2], n[3], n[4], n[5], n[0], n[1], n[6], n[7]});
  if (*static_cast<const int64_t*>(a.ptrs[3]) == 0) {
    return runSub(ctx, s->k[3], a, {0, 4, 5}, {n[0], n[1], n[7], n[8], n[9], n[10], 4});
  }
  runSub(ctx, s->k[2], a, {0, 1, 2, 3, 4, 5}, {n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9], n[10], 4});
}

constexpr char kAttnDecodeKvw[] = "lk_attention_decode_kvw";

}  // namespace

void registerGpuTransformerKernels() {
  registerKernel({"gemv4_f32_gpu", Device::Gpu, prepareGemv4, runGemv4<false, 0, false>, destroyVariants});
  registerKernel({"gemv4_bias_f32_gpu", Device::Gpu, prepareGemv4, runGemv4<true, 0, false>, destroyVariants});
  registerKernel({"gemv4_bias_gelu_f32_gpu", Device::Gpu, prepareGemv4, runGemv4<true, 1, false>, destroyVariants});
  registerKernel({"gemv4_bias_res_f32_gpu", Device::Gpu, prepareGemv4, runGemv4<true, 0, true>, destroyVariants});
  registerKernel({"attention_decode_kvw_f32_gpu", Device::Gpu, preparePso<kAttnDecodeKvw>, runAttentionDecodeKvw,
                  destroyPso});
  registerKernel({"gemm_auto_f32_gpu", Device::Gpu, prepareGemmAuto, runGemmAuto, destroySubs});
  registerKernel({"gemm_auto_bias_gelu_f32_gpu", Device::Gpu, prepareGemmAutoGelu, runGemmAutoGelu, destroySubs});
  registerKernel({"attention_gen_f32_gpu", Device::Gpu, prepareAttnGen, runAttnGen, destroySubs});
  registerKernel({"gemm_auto_sk_f32_gpu", Device::Gpu, prepareGemmAutoSk, runGemmAutoSk, destroySubs});
  registerKernel({"attention_gen2_f32_gpu", Device::Gpu, prepareAttnGen2, runAttnGen2, destroySubs});
  registerKernel({"gemm_splitk_f32_gpu", Device::Gpu, prepareSplitK, runGemmSplitK, destroySplitK});
  registerKernel({"add_bias_layernorm2_sk_f32_gpu", Device::Gpu, preparePso<kAddLn2Sk>, runAddBiasLn2Sk, destroyPso});
  registerKernel({"attention_cache_sg_f32_gpu", Device::Gpu, prepareCacheSg, runAttentionCacheSg, destroyVariants});
  registerKernel({"attention_decode_split_f32_gpu", Device::Gpu, prepareDecodeSplit, runAttentionDecodeSplit,
                  destroyDecodeSplit});
  registerKernel({"attention_qkv_sg_f32_gpu", Device::Gpu, prepareAttnSg, runAttentionQkvSg<false>, destroyVariants});
  registerKernel({"attention_qkv_sg_causal_f32_gpu", Device::Gpu, prepareAttnSg, runAttentionQkvSg<true>,
                  destroyVariants});
}

}  // namespace lk::rt
