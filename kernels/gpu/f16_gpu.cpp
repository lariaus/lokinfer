// f16 GPU kernels (Qwen3-ASR phase 1, docs/impl-notes/qwen3-asr-port-plan.md Q2/Q5): dispatch of
// kernels/gpu/f16_basic.metal and of the Q5 optimized kernels in f16_fused.metal. half storage, float accumulators.
// The op contracts are docs/impl-notes/qwen3-asr-f16-kernels.md and qwen3-asr-f16-opt-kernels.md; the Python ops are
// python/lokinfer/ops/f16_gpu.py.
//
// Data-dependent errors (an id out of range, more pad ids than audio rows, a KV write past the cache) are detected
// on the CPU at encode time from the operand's memory (unified): those operands (ids, pos) are graph inputs, written
// before the call, as runAttnGen in transformer_gpu.cpp also assumes for pos. The kernels never read or write out of
// bounds even when the check would have failed.
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

void bindArg(GpuContext& g, uint32_t slot, const KernelArgs& a, uint32_t i, uint64_t extraOffset = 0) {
  g.bind(slot, static_cast<MTL::Buffer*>(a.bufs[i].mtlBuffer), a.bufs[i].offset + extraOffset);
}
uint32_t u32(int64_t v) { return static_cast<uint32_t>(v); }
int64_t readI64(const KernelArgs& a, uint32_t i, int64_t k = 0) { return static_cast<const int64_t*>(a.ptrs[i])[k]; }

// ---- elementwise: ptrs x, y, out (or x, out). ints x(R, C), ... ; floats [lim] ---------------------------------

struct EltParams {
  uint32_t n;
  float lim;
};

template <uint32_t Operands>
void runElt(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  EltParams p{u32(a.ints[0] * a.ints[1]), 0.f};
  if (p.n == 0) return;
  g.beginDispatch();
  for (uint32_t i = 0; i <= Operands; ++i) bindArg(g, i, a, i);
  g.bindParams(Operands + 1, &p, sizeof p);
  g.dispatch(psoOf(a), p.n, 1, 1, 256, 1, 1);
}

void runAddClamp(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  EltParams p{u32(a.ints[0] * a.ints[1]), static_cast<float>(a.floats[0])};
  if (p.n == 0) return;
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatch(psoOf(a), p.n, 1, 1, 256, 1, 1);
}

// ---- add_pos_prefix: ptrs x, pos, out. ints x(n, P, D), pos(P, D), out(rows, D), rows ---------------------------

void runAddPosPrefix(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t n = a.ints[0], period = a.ints[1], d = a.ints[2], rows = a.ints[7];
  if (rows > n * period) {
    ctx.error = "add_pos_prefix_f16_gpu: rows exceeds chunks * positions";
    return;
  }
  if (rows == 0) return;
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t rows, d, period;
  } p{u32(rows), u32(d), u32(period)};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatch(psoOf(a), p.d, p.rows, 1, 256, 1, 1);
}

// ---- GEMM -------------------------------------------------------------------------------------------------------

struct GemmParams {
  uint32_t m, n, k, has_bias;
};

// Encodes out = a @ b (+ bias) with lk_f16_gemm; the buffers are given as (buffer, offset) pairs.
void encodeGemm(GpuContext& g, MTL::ComputePipelineState* pso, const BufRef& a, uint64_t aOff, const BufRef& b,
                const BufRef* bias, const BufRef& out, uint64_t outOff, uint32_t m, uint32_t n, uint32_t k) {
  GemmParams p{m, n, k, bias ? 1u : 0u};
  g.beginDispatch();
  g.bind(0, static_cast<MTL::Buffer*>(a.mtlBuffer), a.offset + aOff);
  g.bind(1, static_cast<MTL::Buffer*>(b.mtlBuffer), b.offset);
  const BufRef& bb = bias ? *bias : out;  // unused when has_bias = 0
  g.bind(2, static_cast<MTL::Buffer*>(bb.mtlBuffer), bb.offset);
  g.bind(3, static_cast<MTL::Buffer*>(out.mtlBuffer), out.offset + outOff);
  g.bindParams(4, &p, sizeof p);
  g.dispatchThreadgroups(pso, (n + 15) / 16, (m + 15) / 16, 1, 16, 16, 1);
}

// ptrs: x, w, [bias], out. ints: x(M, K), w(K, N), [bias(N)], out(M, N)
template <bool Bias>
void runGemm(ExecCtx& ctx, const KernelArgs& a) {
  const uint32_t m = u32(a.ints[0]), k = u32(a.ints[1]), n = u32(a.ints[3]);
  if (m == 0 || n == 0) return;
  encodeGemm(*ctx.gpu, psoOf(a), a.bufs[0], 0, a.bufs[1], Bias ? &a.bufs[2] : nullptr, a.bufs[Bias ? 3 : 2], 0, m, n,
             k);
}

// ptrs: x, table, out. ints: x(T, D), table(V, D), out(1, V)
void runLmHeadLast(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t t, d, v;
  } p{u32(a.ints[0]), u32(a.ints[1]), u32(a.ints[2])};
  if (p.t == 0 || p.v == 0) return;
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (p.v + 7) / 8, 1, 1, 256, 1, 1);
}

// ---- norms ------------------------------------------------------------------------------------------------------

struct NormParams {
  uint32_t rows, cols;
  float eps;
};
uint32_t normThreads(uint32_t cols) { return std::min<uint32_t>(256, std::max<uint32_t>(32, (cols + 31) / 32 * 32)); }

// ptrs: x, g, b, out. ints: x(R, C), g(C), b(C), out(R, C); floats: eps
void runLayerNorm(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  NormParams p{u32(a.ints[0]), u32(a.ints[1]), static_cast<float>(a.floats[0])};
  if (p.rows == 0) return;
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.rows, 1, 1, normThreads(p.cols), 1, 1);
}

// ptrs: x, g, out. ints: x(R, C), g(W), out(R, C); floats: eps. W = C (rmsnorm) or 128 (rmsnorm_heads: the rows of
// the [R * C / 128, 128] view).
void runRmsNorm(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const uint32_t w = u32(a.ints[2]);
  NormParams p{u32(a.ints[0] * a.ints[1] / w), w, static_cast<float>(a.floats[0])};
  if (p.rows == 0) return;
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.rows, 1, 1, normThreads(p.cols), 1, 1);
}

// ---- RoPE: ptrs x, pos, out. ints x(T, H*128), pos(1), out(T, H*128); floats theta ------------------------------

struct RopeParams {
  uint32_t t, heads;
  float twoPiHi, twoPiLo;
  float invFreq[64];
};
void runRope(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const double theta = a.floats[0], twoPi = 2.0 * M_PI;
  RopeParams p{u32(a.ints[0]), u32(a.ints[1] / 128), static_cast<float>(twoPi), 0.f, {}};
  p.twoPiLo = static_cast<float>(twoPi - double(p.twoPiHi));
  for (int j = 0; j < 64; ++j) p.invFreq[j] = static_cast<float>(std::pow(theta, -2.0 * j / 128.0));
  if (p.t == 0) return;
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatch(psoOf(a), uint64_t(p.heads) * 64, p.t, 1, 64, 4, 1);
}

// ---- gathers ----------------------------------------------------------------------------------------------------

// ptrs: ids, table, out. ints: ids(T), table(V, D), out(T, D)
void runEmbedding(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], v = a.ints[1];
  for (int64_t i = 0; i < t; ++i) {
    const int64_t id = readI64(a, 0, i);
    if (id < 0 || id >= v) {
      ctx.error = "embedding_f16_gpu: id out of range";
      return;
    }
  }
  if (t == 0) return;
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t t, d, v;
  } p{u32(t), u32(a.ints[2]), u32(v)};
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatch(psoOf(a), p.d, p.t, 1, 256, 1, 1);
}

// ptrs: x, ids, audio, out. ints: x(T, D), ids(T), audio(A, D), out(T, D), pad_id
void runScatterAudio(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], rowsA = a.ints[3], padId = a.ints[7];
  int64_t pads = 0;
  for (int64_t i = 0; i < t; ++i) pads += readI64(a, 1, i) == padId;
  if (pads > rowsA) {
    ctx.error = "scatter_audio_f16_gpu: more pad ids than audio rows";
    return;
  }
  if (t == 0) return;
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t t, d, a;
    int64_t padId;
  } p{u32(t), u32(a.ints[1]), u32(rowsA), padId};
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.t, 1, 1, 256, 1, 1);
}

// ---- attention --------------------------------------------------------------------------------------------------

struct AttnParams {
  uint32_t t, qHeads, kvHeads, headDim, window, cap;
  float scale;
};

// ptrs: q, k, v, out. ints: q(S, C), k(S, C), v(S, C), out(S, C), heads, window
void runAttentionWindow(ExecCtx& ctx, const KernelArgs& a) {
  const uint32_t s = u32(a.ints[0]), c = u32(a.ints[1]), heads = u32(a.ints[8]);
  if (s == 0) return;
  GpuContext& g = *ctx.gpu;
  const uint32_t hd = c / heads;
  AttnParams p{s, heads, heads, hd, u32(a.ints[9]), 0, 1.0f / std::sqrt(static_cast<float>(hd))};
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bindParams(4, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (s + 3) / 4, heads, 1, 128, 1, 1);
}

// ptrs: q, kc, vc, pos, out. ints: q(T, Hq*hd), kc(cap, Hkv*hd), vc(cap, Hkv*hd), pos(1), out(T, Hq*hd), kv_heads,
// q_heads
void runAttentionGqa(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], cap = a.ints[2], pos = readI64(a, 3);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "attention_gqa_f16_gpu: pos + T exceeds the cache";
    return;
  }
  if (t == 0) return;
  GpuContext& g = *ctx.gpu;
  const uint32_t kvHeads = u32(a.ints[9]), qHeads = u32(a.ints[10]), hd = u32(a.ints[1]) / qHeads;
  AttnParams p{u32(t), qHeads, kvHeads, hd, 0, u32(cap), 1.0f / std::sqrt(static_cast<float>(hd))};
  g.beginDispatch();
  for (uint32_t i = 0; i < 5; ++i) bindArg(g, i, a, i);
  g.bindParams(5, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), u32((t + 3) / 4), qHeads, 1, 128, 1, 1);
}

// ptrs: kc, vc, k, v, pos. ints: kc(cap, D), vc(cap, D), k(T, D), v(T, D), pos(1)
void runKvWrite(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t cap = a.ints[0], t = a.ints[4], pos = readI64(a, 4);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "kv_write_f16_gpu: pos + T exceeds the cache";
    return;
  }
  if (t == 0) return;
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t t, d, cap;
  } p{u32(t), u32(a.ints[1]), u32(cap)};
  g.beginDispatch();
  for (uint32_t i = 0; i < 5; ++i) bindArg(g, i, a, i);
  g.bindParams(5, &p, sizeof p);
  g.dispatch(psoOf(a), p.d, p.t, 1, 256, 1, 1);
}

// ---- audio_cnn --------------------------------------------------------------------------------------------------
// One chunk at a time: per chunk c, conv1 (from features[c]) -> s1, conv2 -> s2, conv3 -> s3, s3 viewed as
// [H3, W3 * Co] @ w_out -> out[c]. s1..s3 are one chunk's intermediates in a scratch buffer owned by the op instance
// (allocated at the first call: the per-chunk sizes are static), reused by every chunk; consecutive dispatches are
// separated by barriers (the context's default for a kernel's later dispatches; forced for the first one, which
// overwrites the scratch a previous call of this op may still be reading in the same segment).

struct CnnState {
  MTL::ComputePipelineState* conv1;
  MTL::ComputePipelineState* conv;
  MTL::ComputePipelineState* gemm;
  void* mem = nullptr;
  MTL::Buffer* scratch = nullptr;
  size_t bytes = 0;
};

Status prepareCnn(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new CnnState{pc.gpu->pipeline("lk_f16_conv1_feat"), pc.gpu->pipeline("lk_f16_conv3x3s2_gelu"),
                         pc.gpu->pipeline("lk_f16_gemm")};
  if (!s->conv1 || !s->conv || !s->gemm) {
    delete s;
    return Status::error("missing GPU kernel function lk_f16_conv1_feat / lk_f16_conv3x3s2_gelu / lk_f16_gemm");
  }
  *state = s;
  return Status::ok();
}
void destroyCnn(void* p) {
  auto* s = static_cast<CnnState*>(p);
  if (s->scratch) {
    if (GpuContext* g = GpuContext::existing()) {
      g->commitAndWait();  // the scratch may be referenced by a pending segment
      g->releaseBuffer(s->scratch);
    }
  }
  free(s->mem);
  delete s;
}

uint32_t convOut(uint32_t n) { return (n + 2 - 3) / 2 + 1; }  // 3x3, stride 2, pad 1

struct ConvParams {
  uint32_t hIn, wIn, hOut, wOut, ci, co;
};

// ptrs: features, w1, b1, w2, b2, w3, b3, w_out, out. ints: features(n, F, Tm), w1(Co, 3, 3, 1), b1(Co),
// w2(Co, 3, 3, Co), b2(Co), w3(Co, 3, 3, Co), b3(Co), w_out(W3 * Co, D), out(n, H3, D)
void runAudioCnn(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  auto* s = const_cast<CnnState*>(static_cast<const CnnState*>(a.state));
  const uint32_t n = u32(a.ints[0]), f = u32(a.ints[1]), tm = u32(a.ints[2]), co = u32(a.ints[3]);
  const uint32_t d = u32(a.ints[19]);
  const uint32_t h1 = convOut(tm), w1 = convOut(f), h2 = convOut(h1), w2 = convOut(w1), h3 = convOut(h2),
                 w3 = convOut(w2);
  const size_t s1 = size_t(h1) * w1 * co * 2, s2 = size_t(h2) * w2 * co * 2, s3 = size_t(h3) * w3 * co * 2;
  const size_t off2 = (s1 + 255) / 256 * 256, off3 = off2 + (s2 + 255) / 256 * 256;
  const size_t need = (off3 + s3 + 16383) / 16384 * 16384;
  if (n == 0) return;
  if (s->bytes < need) {  // first call (sizes are static per op, so this never grows later)
    if (s->scratch) {
      g.commitAndWait();
      g.releaseBuffer(s->scratch);
      free(s->mem);
      s->scratch = nullptr;
      s->mem = nullptr;
      s->bytes = 0;
    }
    if (posix_memalign(&s->mem, 16384, need) != 0) {
      s->mem = nullptr;
      ctx.error = "audio_cnn_f16_gpu: scratch allocation failed";
      return;
    }
    s->scratch = g.wrapNoCopy(s->mem, need);
    s->bytes = need;
  }
  const BufRef scratch{s->scratch, 0};
  const size_t featChunk = size_t(f) * tm * sizeof(float), outChunk = size_t(h3) * d * 2;
  const ConvParams p1{tm, f, h1, w1, 1, co}, p2{h1, w1, h2, w2, co, co}, p3{h2, w2, h3, w3, co, co};
  for (uint32_t c = 0; c < n; ++c) {
    if (c == 0) g.setNextBarrier(true);
    g.beginDispatch();
    bindArg(g, 0, a, 0, c * featChunk);
    bindArg(g, 1, a, 1);
    bindArg(g, 2, a, 2);
    g.bind(3, s->scratch, 0);
    g.bindParams(4, &p1, sizeof p1);
    g.dispatch(s->conv1, co, w1, h1, 32, 4, 2);

    g.beginDispatch();
    g.bind(0, s->scratch, 0);
    bindArg(g, 1, a, 3);
    bindArg(g, 2, a, 4);
    g.bind(3, s->scratch, off2);
    g.bindParams(4, &p2, sizeof p2);
    g.dispatchThreadgroups(s->conv, (co + 15) / 16, (h2 * w2 + 15) / 16, 1, 16, 16, 1);

    g.beginDispatch();
    g.bind(0, s->scratch, off2);
    bindArg(g, 1, a, 5);
    bindArg(g, 2, a, 6);
    g.bind(3, s->scratch, off3);
    g.bindParams(4, &p3, sizeof p3);
    g.dispatchThreadgroups(s->conv, (co + 15) / 16, (h3 * w3 + 15) / 16, 1, 16, 16, 1);

    encodeGemm(g, s->gemm, scratch, off3, a.bufs[7], nullptr, a.bufs[8], c * outChunk, h3, d, w3 * co);
  }
}

// =================================================================================================================
// Q5 optimized kernels (kernels/gpu/f16_fused.metal; contract docs/impl-notes/qwen3-asr-f16-opt-kernels.md)
// =================================================================================================================

// Tuning constants (A/B'd on the M2 Max and the iPhone 18 Pro Max; numbers in docs/models/qwen3-asr.md, "Q5 GPU
// kernels"). Steel 64x64 tiles once the grid has at least kOptMinTiles64 of them, else 32x32 (more threadgroups for
// small M). GEMV: the widest column block (4 * cw columns per threadgroup, 16 B ... 128 B per weight row) that still
// gives at least kOptGemvGroups threadgroups. Decode attention: one key split per kOptKeysPerSplit keys (at most
// kOptMaxSplits). audio_cnn: kOptCnnBatch chunks per batch (scratch ~11 MB per chunk).
#if TARGET_OS_IPHONE
constexpr uint64_t kOptMinTiles64 = 24;
constexpr int kOptMin64Rows = 0;
constexpr uint32_t kOptGemvGroups = 16;
constexpr uint32_t kOptCnnBatch = 4;
#else
constexpr uint64_t kOptMinTiles64 = 128;
// 64x64 tiles only from this many rows (in-model A/B 2026-11-05, Qwen3-ASR GPU: 32x32 everywhere vs the tile-count
// rule alone: prefill 10 s (M = 149) 77.5 -> 65.9 ms, 30 s (M = 401) 161.9 -> 144.3, enc 30 s 56.7 -> 54.2; a
// 2048-row prefill chunk 736 (64) vs 826 ms (32): the M padding of 64-row tiles costs more than the tile size gains
// for small M; prefill 60 s (M = 789), alternating runs: 32x32 278-294 ms vs 64x64 291-335)
constexpr int kOptMin64Rows = 1024;
constexpr uint32_t kOptGemvGroups = 128;  // in-model A/B 2026-11-05, Qwen3-ASR decode C=480 (DRAM-bound): >= 32 groups 12.74 ms, 64 12.25, 128 12.14, 256 13.02 (the kernel bench reuses its weights from the SLC and showed no difference)
constexpr uint32_t kOptCnnBatch = 8;
#endif
// (2026-11-05, decode C = 15,600, alternating runs: <= 32 splits of >= 256 keys 6.0-6.7 ms of attention per step;
// <= 64: 6.4; <= 128 of 128 keys: 7.5)
constexpr uint32_t kOptKeysPerSplit = 256, kOptMaxSplits = 32;
constexpr int kOptGemmBk = 32;  // the steel instantiations' K tile (f16_fused.metal)

// Tensor-op GEMMs (f16_tensor.metal, Metal 4 matmul2d) for M > 1: meant for GPUs with neural accelerators (Apple10
// family and later: A19 / M5; the iPhone 18 Pro Max), on by default there since 2026-11-07 21:17 (validated on the
// phone: test_f16_fused_gpu.py + test_f16_rms_gemm.py --run-slow 193 passed, Qwen model tests --run-slow CPU + GPU 56
// passed, transcripts golden; the M2 Max keeps steel: tensor ops are slower there). LOKINFER_F16_GEMM=tensor | steel
// overrides (tests of the tensor kernels on the Mac, A/B); LOKINFER_F16T_TILE=128x128 | 64x128 | 64x64 forces one tile shape (tuning).
// Read when a kernel is prepared (model load).
enum TensorTile { kT128x128 = 0, kT64x128 = 1, kT64x64 = 2, kTensorTiles = 3 };
constexpr int kTensorTileBm[kTensorTiles] = {128, 64, 64}, kTensorTileBn[kTensorTiles] = {128, 128, 64};
constexpr int kTensorTileThreads[kTensorTiles] = {512, 256, 128};

struct OptGemmPsos {
  MTL::ComputePipelineState* gemm[2][2] = {};  // [tile 64 / 32][B transposed]
  MTL::ComputePipelineState* swiglu[2] = {};   // [tile 64 / 32]
  MTL::ComputePipelineState* gemv[17] = {};    // by cw
  MTL::ComputePipelineState* gemvSw[17] = {};
  bool tensor = false;                                             // M > 1 on the tensor-op kernels
  int tensorTile = -1;                                             // forced tile (LOKINFER_F16T_TILE), else -1
  MTL::ComputePipelineState* tgemm[kTensorTiles][2] = {};          // [tile][B transposed]
  MTL::ComputePipelineState* tswiglu[kTensorTiles] = {};
};

constexpr bool kTensorGemmDefault = true;  // on for Apple10+ (validated on the phone 2026-11-07)

bool tensorGemmEnabled(PrepareCtx& pc) {
  if (const char* e = std::getenv("LOKINFER_F16_GEMM")) return std::strcmp(e, "tensor") == 0;
  return kTensorGemmDefault && pc.gpu->device()->supportsFamily(MTL::GPUFamilyApple10);
}

Status loadGemmPsos(PrepareCtx& pc, OptGemmPsos& s) {
  const char* gemm[2][2] = {{"lk_f16o_gemm_64x64", "lk_f16o_gemm_64x64_nt"}, {"lk_f16o_gemm_32x32", "lk_f16o_gemm_32x32_nt"}};
  const char* sw[2] = {"lk_f16o_gemm_swiglu_64x64", "lk_f16o_gemm_swiglu_32x32"};
  for (int t = 0; t < 2; ++t) {
    for (int tb = 0; tb < 2; ++tb)
      if (!(s.gemm[t][tb] = pc.gpu->pipeline(gemm[t][tb]))) return Status::error(std::string("missing ") + gemm[t][tb]);
    if (!(s.swiglu[t] = pc.gpu->pipeline(sw[t]))) return Status::error(std::string("missing ") + sw[t]);
  }
  for (uint32_t cw : {2u, 4u, 8u, 16u}) {
    const std::string fn = "lk_f16o_gemv_c" + std::to_string(cw);
    if (!(s.gemv[cw] = pc.gpu->pipeline(fn))) return Status::error("missing " + fn);
    if (!(s.gemvSw[cw] = pc.gpu->pipeline(fn + "_swiglu"))) return Status::error("missing " + fn + "_swiglu");
  }
  s.tensor = tensorGemmEnabled(pc);
  if (!s.tensor) return Status::ok();
  // Kernel flavour: "_nax" (MLX's register fragments in the neural accelerators' lane layout: valid on Apple10+ only;
  // iPhone 18 Pro Max, 2026-11-07: ~2x the layout-agnostic kernels, kernel slow tier + Qwen model tests passed) on
  // Apple10+ GPUs, else the layout-agnostic kernels (correct on any GPU: the Mac's forced runs). LOKINFER_F16T_KERNEL=
  // generic | nax | naxr overrides (naxr: the 128x128 plain GEMM with relaxed_precision, a measurement only: no gain).
  std::string flavour = pc.gpu->device()->supportsFamily(MTL::GPUFamilyApple10) ? "nax" : "generic";
  if (const char* e = std::getenv("LOKINFER_F16T_KERNEL")) flavour = e;
  const std::string pre = flavour == "generic" ? "" : "nax_";
  if (const char* e = std::getenv("LOKINFER_F16T_TILE")) {
    for (int t = 0; t < kTensorTiles; ++t)
      if (std::to_string(kTensorTileBm[t]) + "x" + std::to_string(kTensorTileBn[t]) == e) s.tensorTile = t;
  }
  for (int t = 0; t < kTensorTiles; ++t) {
    const std::string tile = std::to_string(kTensorTileBm[t]) + "x" + std::to_string(kTensorTileBn[t]);
    for (int tb = 0; tb < 2; ++tb) {
      std::string fn = "lk_f16t_gemm_" + pre + tile + (tb ? "_nt" : "");
      if (flavour == "naxr" && t == kT128x128 && !tb) fn = "lk_f16t_gemm_naxr_128x128";
      if (!(s.tgemm[t][tb] = pc.gpu->pipeline(fn))) return Status::error("missing " + fn);
    }
    const std::string fn = "lk_f16t_gemm_swiglu_" + pre + tile;
    if (!(s.tswiglu[t] = pc.gpu->pipeline(fn))) return Status::error("missing " + fn);
  }
  return Status::ok();
}

Status prepareOptGemm(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new OptGemmPsos();
  Status st = loadGemmPsos(pc, *s);
  if (!st.isOk()) {
    delete s;
    return st;
  }
  *state = s;
  return Status::ok();
}
void destroyOptGemm(void* s) { delete static_cast<OptGemmPsos*>(s); }

struct OptGemmParams {
  int32_t m, n, k, lda, ldb, ldd, ldr, tilesN, tilesM, swizzleLog, kIters;
  uint32_t hasBias, act, hasRes, hasClamp;
  float lim;
};

struct Epi {
  const BufRef* bias = nullptr;
  const BufRef* res = nullptr;  // [M, N] row-major, row stride N
  bool gelu = false, clamp = false;
  float lim = 0.f;
};

// Tensor-op tile for an [m, n] output: 64x128 when its grid has kTensorMinTiles threadgroups, else 64x64. 128x128
// (MLX's iPhone tile) only when forced: in-model A/B on the iPhone 2026-11-07 (--quick, forced tile vs a rule that
// started at 128x128): prefill 30 s 150.6 vs 165.3 ms, 300 s 1922 vs 2064, enc 30 s 59.9 vs 62.0, enc 300 s 565 vs
// 590 with 64x128.
constexpr uint64_t kTensorMinTiles = 32;
int tensorTile(const OptGemmPsos& s, int m, int n) {
  if (s.tensorTile >= 0) return s.tensorTile;
  for (int t = kT64x128; t < kTensorTiles - 1; ++t) {
    const uint64_t tiles = uint64_t((m + kTensorTileBm[t] - 1) / kTensorTileBm[t]) *
                           uint64_t((n + kTensorTileBn[t] - 1) / kTensorTileBn[t]);
    if (tiles >= kTensorMinTiles) return t;
  }
  return kTensorTiles - 1;
}

// The tensor-op GEMM (f16_tensor.metal): same buffers and parameter block as the steel kernels (swiglu: a, b, out,
// params at 0..3; else a, b, bias, out, res, params at 0..5). MLX's swizzle for the phone: 1 once the grid has > 3
// row blocks.
void encodeTensor(GpuContext& g, const OptGemmPsos& s, const BufRef& a, uint64_t aOff, const BufRef& b, bool tb,
                  const Epi& e, bool swiglu, const BufRef& out, uint64_t outOff, int m, int n, int k) {
  const int t = tensorTile(s, m, n), bm = kTensorTileBm[t], bn = kTensorTileBn[t];
  const int tilesM = (m + bm - 1) / bm, tilesN = (n + bn - 1) / bn, swz = tilesM <= 3 ? 0 : 1;
  OptGemmParams p{m, n, k, k, swiglu ? 2 * n : (tb ? k : n), n, n, tilesN, tilesM, swz, 0,
                  e.bias ? 1u : 0u, e.gelu ? 1u : 0u, e.res ? 1u : 0u, e.clamp ? 1u : 0u, e.lim};
  g.beginDispatch();
  g.bind(0, static_cast<MTL::Buffer*>(a.mtlBuffer), a.offset + aOff);
  g.bind(1, static_cast<MTL::Buffer*>(b.mtlBuffer), b.offset);
  if (swiglu) {
    g.bind(2, static_cast<MTL::Buffer*>(out.mtlBuffer), out.offset + outOff);
    g.bindParams(3, &p, sizeof p);
  } else {
    const BufRef& bb = e.bias ? *e.bias : out;
    g.bind(2, static_cast<MTL::Buffer*>(bb.mtlBuffer), bb.offset);
    g.bind(3, static_cast<MTL::Buffer*>(out.mtlBuffer), out.offset + outOff);
    const BufRef& rr = e.res ? *e.res : out;
    g.bind(4, static_cast<MTL::Buffer*>(rr.mtlBuffer), rr.offset);
    g.bindParams(5, &p, sizeof p);
  }
  const uint32_t gx = u32(tilesN) << swz, gy = u32((tilesM + (1 << swz) - 1) >> swz);
  g.dispatchThreadgroups(swiglu ? s.tswiglu[t] : s.tgemm[t][tb ? 1 : 0], gx, gy, 1, u32(kTensorTileThreads[t]), 1, 1);
  g.addWork((swiglu ? 4.0 : 2.0) * m * n * k);
}

// out (+outOff) [m, n] = epi(a (+aOff) [m, k] @ op(b)), b [k, n] or (tb) [n, k]; steel tiles, 64x64 or 32x32 (or
// the tensor-op kernels when enabled).
void encodeSteel(GpuContext& g, const OptGemmPsos& s, const BufRef& a, uint64_t aOff, const BufRef& b, bool tb,
                 const Epi& e, const BufRef& out, uint64_t outOff, int m, int n, int k) {
  if (s.tensor) return encodeTensor(g, s, a, aOff, b, tb, e, false, out, outOff, m, n, k);
  const bool big = m >= kOptMin64Rows && uint64_t((m + 63) / 64) * uint64_t((n + 63) / 64) >= kOptMinTiles64;
  const int bm = big ? 64 : 32;
  OptGemmParams p{m, n, k, k, tb ? k : n, n, n, (n + bm - 1) / bm, (m + bm - 1) / bm, 0, k / kOptGemmBk,
                  e.bias ? 1u : 0u, e.gelu ? 1u : 0u, e.res ? 1u : 0u, e.clamp ? 1u : 0u, e.lim};
  g.beginDispatch();
  g.bind(0, static_cast<MTL::Buffer*>(a.mtlBuffer), a.offset + aOff);
  g.bind(1, static_cast<MTL::Buffer*>(b.mtlBuffer), b.offset);
  const BufRef& bb = e.bias ? *e.bias : out;
  g.bind(2, static_cast<MTL::Buffer*>(bb.mtlBuffer), bb.offset);
  g.bind(3, static_cast<MTL::Buffer*>(out.mtlBuffer), out.offset + outOff);
  const BufRef& rr = e.res ? *e.res : out;
  g.bind(4, static_cast<MTL::Buffer*>(rr.mtlBuffer), rr.offset);
  g.bindParams(5, &p, sizeof p);
  g.dispatchThreadgroups(s.gemm[big ? 0 : 1][tb ? 1 : 0], u32(p.tilesN), u32(p.tilesM), 1, 128, 1, 1);
  g.addWork(2.0 * m * n * k);
}

void encodeSteelSwiglu(GpuContext& g, const OptGemmPsos& s, const BufRef& a, const BufRef& b, const BufRef& out, int m,
                       int n, int k) {
  if (s.tensor) return encodeTensor(g, s, a, 0, b, false, Epi{}, true, out, 0, m, n, k);
  const bool big = m >= kOptMin64Rows && uint64_t((m + 63) / 64) * uint64_t((n + 63) / 64) >= kOptMinTiles64;
  const int bm = big ? 64 : 32;
  OptGemmParams p{m, n, k, k, 2 * n, n, n, (n + bm - 1) / bm, (m + bm - 1) / bm, 0, k / kOptGemmBk, 0, 0, 0, 0, 0.f};
  g.beginDispatch();
  g.bind(0, static_cast<MTL::Buffer*>(a.mtlBuffer), a.offset);
  g.bind(1, static_cast<MTL::Buffer*>(b.mtlBuffer), b.offset);
  g.bind(2, static_cast<MTL::Buffer*>(out.mtlBuffer), out.offset);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(s.swiglu[big ? 0 : 1], u32(p.tilesN), u32(p.tilesM), 1, 128, 1, 1);
  g.addWork(4.0 * m * n * k);  // gate and up
}

uint32_t gemvCw(uint32_t n) {
  for (uint32_t cw : {16u, 8u, 4u}) 
    if ((n + 4 * cw - 1) / (4 * cw) >= kOptGemvGroups) return cw;
  return 2;
}

// M = 1: out [1, n] = epi(x [1, k] @ w [k, ldw]) (ldw = n, or 2n for SwiGLU).
void encodeGemv(GpuContext& g, const OptGemmPsos& s, const BufRef& x, const BufRef& w, const Epi& e, bool swiglu,
                const BufRef& out, uint32_t n, uint32_t k) {
  struct {
    uint32_t k, n, ldw, hasBias, act, hasRes, hasClamp;
    float lim;
  } p{k, n, swiglu ? 2 * n : n, e.bias ? 1u : 0u, e.gelu ? 1u : 0u, e.res ? 1u : 0u, e.clamp ? 1u : 0u, e.lim};
  const uint32_t cw = gemvCw(n);
  g.beginDispatch();
  g.bind(0, static_cast<MTL::Buffer*>(x.mtlBuffer), x.offset);
  g.bind(1, static_cast<MTL::Buffer*>(w.mtlBuffer), w.offset);
  const BufRef& bb = e.bias ? *e.bias : out;
  g.bind(2, static_cast<MTL::Buffer*>(bb.mtlBuffer), bb.offset);
  g.bind(3, static_cast<MTL::Buffer*>(out.mtlBuffer), out.offset);
  const BufRef& rr = e.res ? *e.res : out;
  g.bind(4, static_cast<MTL::Buffer*>(rr.mtlBuffer), rr.offset);
  g.bindParams(5, &p, sizeof p);
  g.dispatchThreadgroups((swiglu ? s.gemvSw : s.gemv)[cw], (n + 4 * cw - 1) / (4 * cw), 1, 1, 256, 1, 1);
}

// gemm*_opt: ptrs x, w, [b], [res], out. ints x(M, K), w(K, Nw), [b(N)], [res(M, N)], out(M, N); floats [lim].
// SwiGLU: Nw = 2N.
template <bool Bias, bool Gelu, bool Res, bool Clamp, bool Swiglu>
void runOptGemm(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const auto& s = *static_cast<const OptGemmPsos*>(a.state);
  const int64_t m = a.ints[0], k = a.ints[1], nw = a.ints[3], n = Swiglu ? nw / 2 : nw;
  if (m == 0 || n == 0) return;
  constexpr uint32_t bi = 2, ri = 2 + (Bias ? 1 : 0), oi = ri + (Res ? 1 : 0);
  Epi e;
  if (Bias) e.bias = &a.bufs[bi];
  if (Res) e.res = &a.bufs[ri];
  e.gelu = Gelu;
  e.clamp = Clamp;
  e.lim = Clamp ? static_cast<float>(a.floats[0]) : 0.f;
  if (m == 1) return encodeGemv(g, s, a.bufs[0], a.bufs[1], e, Swiglu, a.bufs[oi], u32(n), u32(k));
  if (Swiglu) return encodeSteelSwiglu(g, s, a.bufs[0], a.bufs[1], a.bufs[oi], int(m), int(n), int(k));
  encodeSteel(g, s, a.bufs[0], 0, a.bufs[1], false, e, a.bufs[oi], 0, int(m), int(n), int(k));
}

// rmsnorm_gemm[_swiglu]_opt: ptrs x, g, w, out, ws. ints x(M, K), g(K), w(K, Nw), out(M, N), ws(M, K); floats eps.
// M = 1 (decode) and K <= 4096: one fused dispatch (lk_f16o_rms_gemv: each threadgroup normalizes the row itself; ws
// is not written). Else rmsnorm into the workspace ws, then the optimized GEMM / GEMV on it (a kernel's later dispatch
// gets a barrier by default).
struct RmsGemmPsos {
  OptGemmPsos gemm;
  MTL::ComputePipelineState* rms = nullptr;
  MTL::ComputePipelineState* fused[17] = {};  // by cw
  MTL::ComputePipelineState* fusedSw[17] = {};
};
constexpr uint32_t kRmsGemvMaxK = 4096;  // LKO_RMS_MAXK (f16_fused.metal)

Status prepareRmsGemm(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new RmsGemmPsos();
  Status st = loadGemmPsos(pc, s->gemm);
  if (st.isOk() && !(s->rms = pc.gpu->pipeline("lk_f16_rmsnorm"))) st = Status::error("missing lk_f16_rmsnorm");
  for (uint32_t cw : {2u, 4u, 8u, 16u}) {
    if (!st.isOk()) break;
    const std::string fn = "lk_f16o_rms_gemv_c" + std::to_string(cw);
    if (!(s->fused[cw] = pc.gpu->pipeline(fn))) st = Status::error("missing " + fn);
    else if (!(s->fusedSw[cw] = pc.gpu->pipeline(fn + "_swiglu"))) st = Status::error("missing " + fn + "_swiglu");
  }
  if (!st.isOk()) {
    delete s;
    return st;
  }
  *state = s;
  return Status::ok();
}
void destroyRmsGemm(void* s) { delete static_cast<RmsGemmPsos*>(s); }

template <bool Swiglu>
void runRmsGemm(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  const auto& s = *static_cast<const RmsGemmPsos*>(a.state);
  const int64_t m = a.ints[0], k = a.ints[1], nw = a.ints[4], n = Swiglu ? nw / 2 : nw;
  if (m == 0 || n == 0) return;
  const float eps = static_cast<float>(a.floats[0]);
  const BufRef &x = a.bufs[0], &gam = a.bufs[1], &w = a.bufs[2], &out = a.bufs[3], &ws = a.bufs[4];
  if (m == 1 && k <= kRmsGemvMaxK) {
    struct {
      uint32_t k, n, ldw;
      float eps;
    } p{u32(k), u32(n), u32(Swiglu ? 2 * n : n), eps};
    const uint32_t cw = gemvCw(u32(n));
    g.beginDispatch();
    g.bind(0, static_cast<MTL::Buffer*>(x.mtlBuffer), x.offset);
    g.bind(1, static_cast<MTL::Buffer*>(gam.mtlBuffer), gam.offset);
    g.bind(2, static_cast<MTL::Buffer*>(w.mtlBuffer), w.offset);
    g.bind(3, static_cast<MTL::Buffer*>(out.mtlBuffer), out.offset);
    g.bindParams(4, &p, sizeof p);
    g.dispatchThreadgroups((Swiglu ? s.fusedSw : s.fused)[cw], (u32(n) + 4 * cw - 1) / (4 * cw), 1, 1, 256, 1, 1);
    return;
  }
  NormParams np{u32(m), u32(k), eps};
  g.beginDispatch();
  g.bind(0, static_cast<MTL::Buffer*>(x.mtlBuffer), x.offset);
  g.bind(1, static_cast<MTL::Buffer*>(gam.mtlBuffer), gam.offset);
  g.bind(2, static_cast<MTL::Buffer*>(ws.mtlBuffer), ws.offset);
  g.bindParams(3, &np, sizeof np);
  g.dispatchThreadgroups(s.rms, np.rows, 1, 1, normThreads(np.cols), 1, 1);
  const Epi e;
  if (m == 1) return encodeGemv(g, s.gemm, ws, w, e, Swiglu, out, u32(n), u32(k));
  if (Swiglu) return encodeSteelSwiglu(g, s.gemm, ws, w, out, int(m), int(n), int(k));
  encodeSteel(g, s.gemm, ws, 0, w, false, e, out, 0, int(m), int(n), int(k));
}

// lm_head_last_opt: ptrs x, table, out. ints x(T, D), table(V, D), out(1, V)
void runOptLmHead(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t t, d, v;
  } p{u32(a.ints[0]), u32(a.ints[1]), u32(a.ints[2])};
  if (p.t == 0 || p.v == 0) return;
  g.beginDispatch();
  for (uint32_t i = 0; i < 3; ++i) bindArg(g, i, a, i);
  g.bindParams(3, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), (p.v + 31) / 32, 1, 1, 256, 1, 1);
}

// qknorm_rope_kv_opt: ptrs qkv, qg, kg, pos, kc, vc, out. ints qkv(T, C), qg(128), kg(128), pos(1), kc(P, W),
// vc(P, W), out(T, Cq); floats eps, theta.
void runOptQkr(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], c = a.ints[1], cap = a.ints[5], w = a.ints[6], pos = readI64(a, 3);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "qknorm_rope_kv_opt_f16_gpu: pos + T exceeds the cache";
    return;
  }
  if (t == 0) return;
  GpuContext& g = *ctx.gpu;
  struct {
    uint32_t t, qHeads, kvHeads, cap;
    float eps, twoPiHi, twoPiLo;
    float invFreq[64];
  } p{u32(t), u32((c - 2 * w) / 128), u32(w / 128), u32(cap), static_cast<float>(a.floats[0]), 0.f, 0.f, {}};
  const double theta = a.floats[1], twoPi = 2.0 * M_PI;
  p.twoPiHi = static_cast<float>(twoPi);
  p.twoPiLo = static_cast<float>(twoPi - double(p.twoPiHi));
  for (int j = 0; j < 64; ++j) p.invFreq[j] = static_cast<float>(std::pow(theta, -2.0 * j / 128.0));
  g.beginDispatch();
  for (uint32_t i = 0; i < 7; ++i) bindArg(g, i, a, i);
  g.bindParams(7, &p, sizeof p);
  g.dispatchThreadgroups(psoOf(a), p.t, 1, 1, 256, 1, 1);
}

// attention_window_qkv_opt: ptrs qkv, out. ints qkv(S, 3C), out(S, C), heads, window
struct WindowPsos {
  MTL::ComputePipelineState* hd64;
  MTL::ComputePipelineState* hd128;
};
Status prepareOptWindow(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new WindowPsos{pc.gpu->pipeline("lk_f16o_attn_window_hd64"), pc.gpu->pipeline("lk_f16o_attn_window_hd128")};
  if (!s->hd64 || !s->hd128) {
    delete s;
    return Status::error("missing GPU kernel function lk_f16o_attn_window_hd*");
  }
  *state = s;
  return Status::ok();
}
void destroyOptWindow(void* s) { delete static_cast<WindowPsos*>(s); }

void runOptWindow(ExecCtx& ctx, const KernelArgs& a) {
  const uint32_t s = u32(a.ints[0]), c = u32(a.ints[3]), heads = u32(a.ints[4]), window = u32(a.ints[5]);
  if (s == 0) return;
  GpuContext& g = *ctx.gpu;
  const auto* st = static_cast<const WindowPsos*>(a.state);
  const uint32_t hd = c / heads;
  MTL::ComputePipelineState* pso = hd == 64 ? st->hd64 : st->hd128;
  // One threadgroup per window when it fits (104 rows: 13 simdgroups), K / V tiles loaded once per window.
  const uint32_t maxSg = std::min<uint32_t>(16, u32(pso->maxTotalThreadsPerThreadgroup()) / 32);
  const uint32_t nsg = std::max<uint32_t>(1, std::min<uint32_t>(maxSg, (std::min(window, s) + 7) / 8));
  const uint32_t qblocks = (std::min(window, s) + 8 * nsg - 1) / (8 * nsg), nwin = (s + window - 1) / window;
  struct {
    uint32_t s, heads, window, qblocks;
    float scale;
  } p{s, heads, window, qblocks, 1.0f / std::sqrt(static_cast<float>(hd))};
  g.beginDispatch();
  bindArg(g, 0, a, 0);
  bindArg(g, 1, a, 1);
  g.bindParams(2, &p, sizeof p);
  g.dispatchThreadgroups(pso, nwin * qblocks, heads, 1, 32 * nsg, 1, 1);
  g.addWork(4.0 * double(s) * double(std::min(window, s)) * double(hd) * double(heads));
}

// attention_gqa_opt: ptrs q, kc, vc, pos, out. ints q(T, Cq), kc(P, W), vc(P, W), pos(1), out(T, Cq), kv_heads,
// q_heads. T = 1: key-split decode (+ combine; partials in a scratch buffer owned by the op instance); T > 1: flash.
struct GqaState {
  MTL::ComputePipelineState* flash[2] = {};      // hd 64 / 128
  MTL::ComputePipelineState* decode[2][2] = {};  // [hd 64 / 128][G 1 / 2]
  MTL::ComputePipelineState* combine[2] = {};
  void* mem = nullptr;
  MTL::Buffer* part = nullptr;
  size_t bytes = 0;
};
Status prepareOptGqa(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new GqaState();
  const char* hds[2] = {"64", "128"};
  bool ok = true;
  for (int h = 0; h < 2; ++h) {
    ok &= (s->flash[h] = pc.gpu->pipeline(std::string("lk_f16o_attn_gqa_hd") + hds[h])) != nullptr;
    ok &= (s->combine[h] = pc.gpu->pipeline(std::string("lk_f16o_attn_combine_hd") + hds[h])) != nullptr;
    for (int gi = 0; gi < 2; ++gi)
      ok &= (s->decode[h][gi] = pc.gpu->pipeline(std::string("lk_f16o_attn_decode_hd") + hds[h] + "_g" +
                                                 std::to_string(gi + 1))) != nullptr;
  }
  if (!ok) {
    delete s;
    return Status::error("missing GPU kernel function lk_f16o_attn_gqa_* / decode / combine");
  }
  *state = s;
  return Status::ok();
}
void destroyOptGqa(void* p) {
  auto* s = static_cast<GqaState*>(p);
  if (s->part) {
    if (GpuContext* g = GpuContext::existing()) {
      g->commitAndWait();  // the scratch may be referenced by a pending segment
      g->releaseBuffer(s->part);
    }
  }
  free(s->mem);
  delete s;
}

// Grows an op-owned scratch buffer to `need` bytes (sizes are static per op, so this happens on the first call).
bool ensureScratch(GpuContext& g, void*& mem, MTL::Buffer*& buf, size_t& bytes, size_t need) {
  if (bytes >= need) return true;
  if (buf) {
    g.commitAndWait();
    g.releaseBuffer(buf);
    free(mem);
    buf = nullptr;
    mem = nullptr;
    bytes = 0;
  }
  need = (need + 16383) / 16384 * 16384;
  if (posix_memalign(&mem, 16384, need) != 0) {
    mem = nullptr;
    return false;
  }
  buf = g.wrapNoCopy(mem, need);
  bytes = need;
  return true;
}

void runOptGqa(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], cap = a.ints[2], pos = readI64(a, 3);
  if (pos < 0 || pos + t > cap) {
    ctx.error = "attention_gqa_opt_f16_gpu: pos + T exceeds the cache";
    return;
  }
  if (t == 0) return;
  GpuContext& g = *ctx.gpu;
  auto* s = const_cast<GqaState*>(static_cast<const GqaState*>(a.state));
  const uint32_t kvHeads = u32(a.ints[9]), qHeads = u32(a.ints[10]), hd = u32(a.ints[1]) / qHeads;
  const int hi = hd == 64 ? 0 : 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  if (t > 1) {
    struct {
      uint32_t t, qHeads, kvHeads, cap;
      float scale;
    } p{u32(t), qHeads, kvHeads, u32(cap), scale};
    constexpr uint32_t nsg = 4;  // 32-row query blocks (8 simdgroups: T = 2048 at pos 13570 60.4 vs 45.7 ms)
    g.beginDispatch();
    for (uint32_t i = 0; i < 5; ++i) bindArg(g, i, a, i);
    g.bindParams(5, &p, sizeof p);
    g.dispatchThreadgroups(s->flash[hi], (u32(t) + 8 * nsg - 1) / (8 * nsg), qHeads, 1, 32 * nsg, 1, 1);
    g.addWork(4.0 * double(t) * double(pos + (t + 1) / 2) * double(hd) * double(qHeads));  // causal QK + PV
    return;
  }
  const uint32_t n = u32(pos) + 1;
  const uint32_t splits = std::min(kOptMaxSplits, std::max<uint32_t>(1, (n + kOptKeysPerSplit - 1) / kOptKeysPerSplit));
  const uint32_t chunk = (n + splits - 1) / splits;
  if (!ensureScratch(g, s->mem, s->part, s->bytes, size_t(qHeads) * kOptMaxSplits * (hd + 4) * sizeof(float))) {
    ctx.error = "attention_gqa_opt_f16_gpu: scratch allocation failed";
    return;
  }
  struct {
    uint32_t qHeads, kvHeads, cap, splits, chunk;
    float scale;
  } p{qHeads, kvHeads, u32(cap), splits, chunk, scale};
  const uint32_t G = (qHeads / kvHeads) % 2 == 0 ? 2 : 1;
  g.setNextBarrier(true);  // the partials scratch is not tracked by the executable's hazard ranges
  g.beginDispatch();
  for (uint32_t i = 0; i < 4; ++i) bindArg(g, i, a, i);
  g.bind(4, s->part, 0);
  g.bindParams(5, &p, sizeof p);
  g.dispatchThreadgroups(s->decode[hi][G - 1], qHeads / G, splits, 1, 256, 1, 1);
  g.beginDispatch();  // a later dispatch of the kernel: barrier by default
  g.bind(0, s->part, 0);
  bindArg(g, 1, a, 4);
  g.bindParams(2, &p, sizeof p);
  g.dispatchThreadgroups(s->combine[hi], qHeads, 1, 1, 32, 1, 1);
}

// audio_cnn_opt: the basic op's signature and math; per batch of up to kOptCnnBatch chunks: conv1 (direct, Ci = 1),
// then for conv2 / conv3 im2col + the steel GEMM (w [Co, 9 Ci] as B^T) with the bias + GELU epilogue, then conv_out
// as one GEMM over the batch's [chunks * H3, W3 * Co] rows. Scratch (op-owned, reused by every batch): s1 (also s3),
// the im2col matrix, s2.
struct OptCnnState {
  OptGemmPsos gemm;
  MTL::ComputePipelineState* conv1 = nullptr;
  MTL::ComputePipelineState* im2col = nullptr;
  void* mem = nullptr;
  MTL::Buffer* scratch = nullptr;
  size_t bytes = 0;
};
Status prepareOptCnn(PrepareCtx& pc, const Op&, void** state) {
  auto* s = new OptCnnState();
  Status st = loadGemmPsos(pc, s->gemm);
  s->conv1 = pc.gpu->pipeline("lk_f16o_conv1");
  s->im2col = pc.gpu->pipeline("lk_f16o_im2col3x3s2");
  if (st.isOk() && (!s->conv1 || !s->im2col)) st = Status::error("missing lk_f16o_conv1 / lk_f16o_im2col3x3s2");
  if (!st.isOk()) {
    delete s;
    return st;
  }
  *state = s;
  return Status::ok();
}
void destroyOptCnn(void* p) {
  auto* s = static_cast<OptCnnState*>(p);
  if (s->scratch) {
    if (GpuContext* g = GpuContext::existing()) {
      g->commitAndWait();
      g->releaseBuffer(s->scratch);
    }
  }
  free(s->mem);
  delete s;
}

size_t align256(size_t v) { return (v + 255) / 256 * 256; }

// ptrs/ints as audio_cnn_f16_gpu
void runOptCnn(ExecCtx& ctx, const KernelArgs& a) {
  GpuContext& g = *ctx.gpu;
  auto* s = const_cast<OptCnnState*>(static_cast<const OptCnnState*>(a.state));
  const uint32_t n = u32(a.ints[0]), f = u32(a.ints[1]), tm = u32(a.ints[2]), co = u32(a.ints[3]);
  const uint32_t d = u32(a.ints[19]);
  const uint32_t h1 = convOut(tm), w1 = convOut(f), h2 = convOut(h1), w2 = convOut(w1), h3 = convOut(h2),
                 w3 = convOut(w2);
  if (n == 0) return;
  const uint32_t cb = std::min(n, kOptCnnBatch);
  const size_t s1 = align256(size_t(cb) * h1 * w1 * co * 2);
  const size_t am = align256(size_t(cb) * std::max(h2 * w2, h3 * w3) * 9 * co * 2);
  const size_t s2 = align256(size_t(cb) * h2 * w2 * co * 2);
  if (!ensureScratch(g, s->mem, s->scratch, s->bytes, s1 + am + s2)) {
    ctx.error = "audio_cnn_opt_f16_gpu: scratch allocation failed";
    return;
  }
  const BufRef scr{s->scratch, 0};
  const uint64_t offS1 = 0, offA = s1, offS2 = s1 + am;
  const size_t featChunk = size_t(f) * tm * sizeof(float), outChunk = size_t(h3) * d * 2;
  const ConvParams p1{tm, f, h1, w1, 1, co}, p2{h1, w1, h2, w2, co, co}, p3{h2, w2, h3, w3, co, co};
  for (uint32_t c0 = 0; c0 < n; c0 += cb) {
    const uint32_t nb = std::min(cb, n - c0);
    if (c0 == 0) g.setNextBarrier(true);  // the scratch may still be read by a previous call of this op
    g.beginDispatch();
    bindArg(g, 0, a, 0, c0 * featChunk);
    bindArg(g, 1, a, 1);
    bindArg(g, 2, a, 2);
    g.bind(3, s->scratch, offS1);
    g.bindParams(4, &p1, sizeof p1);
    g.dispatch(s->conv1, co, w1, uint64_t(nb) * h1, 32, 4, 2);
    // conv2: im2col(s1) -> A, A @ w2^T + b2, GELU -> s2
    g.beginDispatch();
    g.bind(0, s->scratch, offS1);
    g.bind(1, s->scratch, offA);
    g.bindParams(2, &p2, sizeof p2);
    g.dispatch(s->im2col, 9 * co / 4, uint64_t(nb) * h2 * w2, 1, 64, 4, 1);
    Epi e;
    e.bias = &a.bufs[4];
    e.gelu = true;
    encodeSteel(g, s->gemm, scr, offA, a.bufs[3], true, e, scr, offS2, int(nb * h2 * w2), int(co), int(9 * co));
    // conv3: im2col(s2) -> A, A @ w3^T + b3, GELU -> s3 (in s1's place)
    g.beginDispatch();
    g.bind(0, s->scratch, offS2);
    g.bind(1, s->scratch, offA);
    g.bindParams(2, &p3, sizeof p3);
    g.dispatch(s->im2col, 9 * co / 4, uint64_t(nb) * h3 * w3, 1, 64, 4, 1);
    e.bias = &a.bufs[6];
    encodeSteel(g, s->gemm, scr, offA, a.bufs[5], true, e, scr, offS1, int(nb * h3 * w3), int(co), int(9 * co));
    // conv_out: s3 as [nb * H3, W3 * Co] @ w_out -> out rows of chunks c0 ..
    encodeSteel(g, s->gemm, scr, offS1, a.bufs[7], false, Epi{}, a.bufs[8], c0 * outChunk, int(nb * h3), int(d),
                int(w3 * co));
  }
}

constexpr char kAdd[] = "lk_f16_add";
constexpr char kAddClamp[] = "lk_f16_add_clamp";
constexpr char kSiluMul[] = "lk_f16_silu_mul";
constexpr char kGelu[] = "lk_f16_gelu_erf";
constexpr char kAddPos[] = "lk_f16_add_pos_prefix";
constexpr char kGemm[] = "lk_f16_gemm";
constexpr char kLmHead[] = "lk_f16_lm_head_last";
constexpr char kLayerNorm[] = "lk_f16_layernorm";
constexpr char kRmsNorm[] = "lk_f16_rmsnorm";
constexpr char kRope[] = "lk_f16_rope";
constexpr char kEmbedding[] = "lk_f16_embedding";
constexpr char kScatter[] = "lk_f16_scatter_audio";
constexpr char kAttnWindow[] = "lk_f16_attention_window";
constexpr char kAttnGqa[] = "lk_f16_attention_gqa";
constexpr char kKvWrite[] = "lk_f16_kv_write";
constexpr char kOptLmHead[] = "lk_f16o_lm_head";
constexpr char kOptQkr[] = "lk_f16o_qknorm_rope_kv";

}  // namespace

void registerGpuF16Kernels() {
  registerKernel({"audio_cnn_f16_gpu", Device::Gpu, prepareCnn, runAudioCnn, destroyCnn});
  registerKernel({"add_pos_prefix_f16_gpu", Device::Gpu, preparePso<kAddPos>, runAddPosPrefix, destroyPso});
  registerKernel({"gemm_f16_gpu", Device::Gpu, preparePso<kGemm>, runGemm<false>, destroyPso});
  registerKernel({"gemm_bias_f16_gpu", Device::Gpu, preparePso<kGemm>, runGemm<true>, destroyPso});
  registerKernel({"lm_head_last_f16_gpu", Device::Gpu, preparePso<kLmHead>, runLmHeadLast, destroyPso});
  registerKernel({"gelu_erf_f16_gpu", Device::Gpu, preparePso<kGelu>, runElt<1>, destroyPso});
  registerKernel({"layernorm_f16_gpu", Device::Gpu, preparePso<kLayerNorm>, runLayerNorm, destroyPso});
  registerKernel({"rmsnorm_f16_gpu", Device::Gpu, preparePso<kRmsNorm>, runRmsNorm, destroyPso});
  registerKernel({"rmsnorm_heads_f16_gpu", Device::Gpu, preparePso<kRmsNorm>, runRmsNorm, destroyPso});
  registerKernel({"rope_f16_gpu", Device::Gpu, preparePso<kRope>, runRope, destroyPso});
  registerKernel({"add_f16_gpu", Device::Gpu, preparePso<kAdd>, runElt<2>, destroyPso});
  registerKernel({"add_clamp_f16_gpu", Device::Gpu, preparePso<kAddClamp>, runAddClamp, destroyPso});
  registerKernel({"silu_mul_f16_gpu", Device::Gpu, preparePso<kSiluMul>, runElt<2>, destroyPso});
  registerKernel({"embedding_f16_gpu", Device::Gpu, preparePso<kEmbedding>, runEmbedding, destroyPso});
  registerKernel({"scatter_audio_f16_gpu", Device::Gpu, preparePso<kScatter>, runScatterAudio, destroyPso});
  registerKernel({"attention_window_f16_gpu", Device::Gpu, preparePso<kAttnWindow>, runAttentionWindow, destroyPso});
  registerKernel({"kv_write_f16_gpu", Device::Gpu, preparePso<kKvWrite>, runKvWrite, destroyPso});
  registerKernel({"attention_gqa_f16_gpu", Device::Gpu, preparePso<kAttnGqa>, runAttentionGqa, destroyPso});
  // Q5 optimized kernels (f16_fused.metal)
  registerKernel({"gemm_opt_f16_gpu", Device::Gpu, prepareOptGemm, runOptGemm<false, false, false, false, false>,
                  destroyOptGemm});
  registerKernel({"gemm_bias_opt_f16_gpu", Device::Gpu, prepareOptGemm, runOptGemm<true, false, false, false, false>,
                  destroyOptGemm});
  registerKernel({"gemm_bias_gelu_opt_f16_gpu", Device::Gpu, prepareOptGemm,
                  runOptGemm<true, true, false, false, false>, destroyOptGemm});
  registerKernel({"gemm_add_opt_f16_gpu", Device::Gpu, prepareOptGemm, runOptGemm<false, false, true, false, false>,
                  destroyOptGemm});
  registerKernel({"gemm_bias_add_opt_f16_gpu", Device::Gpu, prepareOptGemm, runOptGemm<true, false, true, false, false>,
                  destroyOptGemm});
  registerKernel({"gemm_bias_add_clamp_opt_f16_gpu", Device::Gpu, prepareOptGemm,
                  runOptGemm<true, false, true, true, false>, destroyOptGemm});
  registerKernel({"gemm_swiglu_opt_f16_gpu", Device::Gpu, prepareOptGemm, runOptGemm<false, false, false, false, true>,
                  destroyOptGemm});
  registerKernel({"rmsnorm_gemm_opt_f16_gpu", Device::Gpu, prepareRmsGemm, runRmsGemm<false>, destroyRmsGemm});
  registerKernel({"rmsnorm_gemm_swiglu_opt_f16_gpu", Device::Gpu, prepareRmsGemm, runRmsGemm<true>, destroyRmsGemm});
  registerKernel({"lm_head_last_opt_f16_gpu", Device::Gpu, preparePso<kOptLmHead>, runOptLmHead, destroyPso});
  registerKernel({"qknorm_rope_kv_opt_f16_gpu", Device::Gpu, preparePso<kOptQkr>, runOptQkr, destroyPso});
  registerKernel({"attention_window_qkv_opt_f16_gpu", Device::Gpu, prepareOptWindow, runOptWindow, destroyOptWindow});
  registerKernel({"attention_gqa_opt_f16_gpu", Device::Gpu, prepareOptGqa, runOptGqa, destroyOptGqa});
  registerKernel({"audio_cnn_opt_f16_gpu", Device::Gpu, prepareOptCnn, runOptCnn, destroyOptCnn});
}

}  // namespace lk::rt
