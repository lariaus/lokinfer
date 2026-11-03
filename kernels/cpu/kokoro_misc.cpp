// Kokoro token-rate and elementwise kernels: embedding, attention, gelu_tanh, leaky_relu, scale, add3_scale,
// snake, concat_rows/cols, broadcast_rows, durations, repeat_interleave.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "kokoro_common.h"
#include "neon_math.h"
#include "lokinfer/kernels/cpu_kokoro.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::kernels::cpu {

bool embedding_f32(const float* table, const int64_t* ids, float* out, int64_t t, int64_t d, int64_t v) {
  for (int64_t i = 0; i < t; ++i)
    if (ids[i] < 0 || ids[i] >= v) return false;
  for (int64_t i = 0; i < t; ++i)
    std::memcpy(out + i * d, table + ids[i] * d, static_cast<size_t>(d) * sizeof(float));
  return true;
}

void attention_f32(const float* q, const float* k, const float* v, float* out, int64_t t, int64_t heads,
                   int64_t headDim) {
  const int64_t d = heads * headDim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
  const int tn = static_cast<int>(t);
  auto head = [&](int64_t h) {
    float* s = detail::scratch(static_cast<size_t>(t * t), 1);
    const int64_t off = h * headDim;
    // s = (q_h @ k_h^T) * scale, softmax per row (vectorized max / exp / sum / scale), out_h = s @ v_h
    detail::sgemm(false, true, t, t, headDim, scale, q + off, d, k + off, d, 0.f, s, t);
    for (int64_t i = 0; i < t; ++i) {
      float* row = s + i * t;
      float m, sum;
      vDSP_maxv(row, 1, &m, static_cast<vDSP_Length>(t));
      m = -m;
      vDSP_vsadd(row, 1, &m, row, 1, static_cast<vDSP_Length>(t));
      vvexpf(row, row, &tn);
      vDSP_sve(row, 1, &sum, static_cast<vDSP_Length>(t));
      const float inv = 1.f / sum;
      vDSP_vsmul(row, 1, &inv, row, 1, static_cast<vDSP_Length>(t));
    }
    detail::sgemm(false, false, t, headDim, t, 1.f, s, t, v + off, d, 0.f, out + off, d);
  };
  // Heads are independent: one per pool task (single-threaded BLAS each) once the work is worth it.
  if (t * t * headDim * heads < (int64_t{1} << 21)) {
    for (int64_t h = 0; h < heads; ++h) head(h);
  } else {
    detail::parallelForSt(heads, head);
  }
}

// Attention over a fused QKV projection qkv [T, 3D] (q | k | v, biases already added) -> out [T, D]; optional
// causal mask (GPT-2). Heads on the pool like attention_f32.
void attention_qkv_f32(const float* qkv, float* out, int64_t t, int64_t heads, int64_t headDim, bool causal) {
  const int64_t d = heads * headDim, ld = 3 * d;
  const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
  auto head = [&](int64_t h) {
    float* s = detail::scratch(static_cast<size_t>(t * t), 1);
    const int64_t off = h * headDim;
    detail::sgemm(false, true, t, t, headDim, scale, qkv + off, ld, qkv + d + off, ld, 0.f, s, t);
    for (int64_t i = 0; i < t; ++i) {
      float* row = s + i * t;
      const int64_t n = causal ? i + 1 : t;
      const int ni = static_cast<int>(n);
      float m, sum;
      vDSP_maxv(row, 1, &m, static_cast<vDSP_Length>(n));
      m = -m;
      vDSP_vsadd(row, 1, &m, row, 1, static_cast<vDSP_Length>(n));
      vvexpf(row, row, &ni);
      vDSP_sve(row, 1, &sum, static_cast<vDSP_Length>(n));
      const float inv = 1.f / sum;
      vDSP_vsmul(row, 1, &inv, row, 1, static_cast<vDSP_Length>(n));
      if (n < t) std::fill(row + n, row + t, 0.f);
    }
    detail::sgemm(false, false, t, headDim, t, 1.f, s, t, qkv + 2 * d + off, ld, 0.f, out + off, d);
  };
  if (t * t * headDim * heads < (int64_t{1} << 21)) {
    for (int64_t h = 0; h < heads; ++h) head(h);
  } else {
    detail::parallelForSt(heads, head);
  }
}

namespace {

// row[0:vis] = exp(row - max) (NEON exp), row[vis:n] = 0; returns 1 / sum.
inline float softmaxPrefix(float* row, int64_t vis, int64_t n) {
  float32x4_t vm = vdupq_n_f32(-INFINITY);
  int64_t j = 0;
  for (; j + 4 <= vis; j += 4) vm = vmaxq_f32(vm, vld1q_f32(row + j));
  float m = vmaxvq_f32(vm);
  for (; j < vis; ++j) m = std::max(m, row[j]);
  const float32x4_t mv = vdupq_n_f32(m);
  float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = acc0;
  j = 0;
  for (; j + 8 <= vis; j += 8) {
    const float32x4_t e0 = detail::exp4(vsubq_f32(vld1q_f32(row + j), mv));
    const float32x4_t e1 = detail::exp4(vsubq_f32(vld1q_f32(row + j + 4), mv));
    vst1q_f32(row + j, e0);
    vst1q_f32(row + j + 4, e1);
    acc0 = vaddq_f32(acc0, e0);
    acc1 = vaddq_f32(acc1, e1);
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; j < vis; ++j) sum += (row[j] = std::exp(row[j] - m));
  if (vis < n) std::fill(row + vis, row + n, 0.f);
  return 1.f / sum;
}

}  // namespace

// Causal attention over a fused QKV projection, work items (head, query block of qb rows) with the keys trimmed to
// the block's causal extent (half the work of the full T x T scores) and a NEON softmax whose 1/sum is applied to the
// 64-wide output rows. mode 0: items on the pool (largest first, single-threaded BLAS each); mode 1: sequential.
void attention_qkv_causal_blk_f32(const float* qkv, float* out, int64_t t, int64_t heads, int64_t headDim, int64_t qb,
                                  int mode) {
  const int64_t d = heads * headDim, ld = 3 * d;
  const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
  qb = std::min<int64_t>(std::max<int64_t>(qb, 1), 256);
  const int64_t nqb = (t + qb - 1) / qb;
  auto item = [&](int64_t i) {
    const int64_t blk = nqb - 1 - i / heads, h = i % heads, off = h * headDim;
    const int64_t i0 = blk * qb, i1 = std::min(t, i0 + qb), rows = i1 - i0, n = i1;
    float* s = detail::scratch(static_cast<size_t>(rows * n), 1);
    detail::sgemm(false, true, rows, n, headDim, scale, qkv + i0 * ld + off, ld, qkv + d + off, ld, 0.f, s, n);
    float inv[256];
    for (int64_t r = 0; r < rows; ++r) inv[r] = softmaxPrefix(s + r * n, i0 + r + 1, n);
    float* o = out + i0 * d + off;
    detail::sgemm(false, false, rows, headDim, n, 1.f, s, n, qkv + 2 * d + off, ld, 0.f, o, d);
    for (int64_t r = 0; r < rows; ++r) {
      const float32x4_t vi = vdupq_n_f32(inv[r]);
      float* orow = o + r * d;
      int64_t c = 0;
      for (; c + 4 <= headDim; c += 4) vst1q_f32(orow + c, vmulq_f32(vld1q_f32(orow + c), vi));
      for (; c < headDim; ++c) orow[c] *= inv[r];
    }
  };
  const int64_t items = heads * nqb;
  if (mode == 1 || items == 1) {
    for (int64_t i = 0; i < items; ++i) item(i);
  } else {
    detail::parallelForSt(items, item);
  }
}

void gelu_tanh_f32(const float* x, float* out, int64_t n) {
  // 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))), same op order as torch's tanh approximation; tanh from
  // Accelerate's vectorized vvtanhf over L1-sized chunks.
  constexpr float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float kKappa = 0.044715f;
  constexpr int64_t kChunk = 2048;
  auto chunk = [&](int64_t c) {
    float buf[kChunk];
    const int64_t i0 = c * kChunk;
    const int m = static_cast<int>(std::min(kChunk, n - i0));
    const float* xs = x + i0;
    for (int i = 0; i < m; ++i) buf[i] = kBeta * (xs[i] + kKappa * xs[i] * xs[i] * xs[i]);
    vvtanhf(buf, buf, &m);
    for (int i = 0; i < m; ++i) out[i0 + i] = 0.5f * xs[i] * (1.f + buf[i]);
  };
  // Chunks are independent: on the pool above ~32 chunks (ALBERT's FFN [T, 2048]: 1.6 ms of Kokoro app59f on one
  // thread, Mac CPU 2026-11-01).
  const int64_t chunks = (n + kChunk - 1) / kChunk;
  if (chunks >= 32) {
    rt::ThreadPool::global().parallelFor(chunks, chunk);
  } else {
    for (int64_t c = 0; c < chunks; ++c) chunk(c);
  }
}

void leaky_relu_f32(const float* x, float* out, int64_t n, float slope) {
  float32x4_t vs = vdupq_n_f32(slope), z = vdupq_n_f32(0.f);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    vst1q_f32(out + i, vbslq_f32(vcgtq_f32(v, z), v, vmulq_f32(v, vs)));
  }
  for (; i < n; ++i) out[i] = x[i] > 0.f ? x[i] : x[i] * slope;
}

void scale_f32(const float* x, float* out, int64_t n, float s) {
  float32x4_t vs = vdupq_n_f32(s);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vmulq_f32(vld1q_f32(x + i), vs));
  for (; i < n; ++i) out[i] = x[i] * s;
}

void add3_scale_f32(const float* a, const float* b, const float* c, float* out, int64_t n, float s) {
  float32x4_t vs = vdupq_n_f32(s);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(out + i, vmulq_f32(vaddq_f32(vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)), vld1q_f32(c + i)), vs));
  for (; i < n; ++i) out[i] = (a[i] + b[i] + c[i]) * s;
}

void snake_f32(const float* x, const float* alpha, float* out, int64_t c, int64_t l) {
  // Same formula and op order as torch (x + (1/a)·sin(a·x)²); the sin is
  // Accelerate's vectorized vvsinf over L1-sized chunks (the scalar std::sin
  // loop took 113 ms on one Kokoro stage-1 tensor).
  constexpr int64_t kChunk = 2048;
  float buf[kChunk];
  for (int64_t ch = 0; ch < c; ++ch) {
    const float a = alpha[ch], inv = 1.f / a;
    const float* xr = x + ch * l;
    float* o = out + ch * l;
    for (int64_t i0 = 0; i0 < l; i0 += kChunk) {
      const int n = static_cast<int>(std::min(kChunk, l - i0));
      for (int i = 0; i < n; ++i) buf[i] = a * xr[i0 + i];
      vvsinf(buf, buf, &n);
      for (int i = 0; i < n; ++i) o[i0 + i] = xr[i0 + i] + inv * (buf[i] * buf[i]);
    }
  }
}

void broadcast_rows_f32(const float* v, float* out, int64_t rows, int64_t cols) {
  for (int64_t r = 0; r < rows; ++r) std::memcpy(out + r * cols, v, static_cast<size_t>(cols) * sizeof(float));
}

void concat_rows_f32(const float* const* xs, const int64_t* rows, int64_t n, float* out, int64_t cols) {
  for (int64_t i = 0; i < n; ++i) {
    size_t sz = static_cast<size_t>(rows[i] * cols);
    std::memcpy(out, xs[i], sz * sizeof(float));
    out += sz;
  }
}

void concat_cols_f32(const float* const* xs, const int64_t* cols, int64_t n, float* out, int64_t rows) {
  int64_t total = 0;
  for (int64_t i = 0; i < n; ++i) total += cols[i];
  for (int64_t r = 0; r < rows; ++r) {
    float* o = out + r * total;
    for (int64_t i = 0; i < n; ++i) {
      std::memcpy(o, xs[i] + r * cols[i], static_cast<size_t>(cols[i]) * sizeof(float));
      o += cols[i];
    }
  }
}

void durations_f32(const float* x, float speed, int64_t* dur, int64_t* total, int64_t t, int64_t n) {
  int64_t sum = 0;
  for (int64_t i = 0; i < t; ++i) {
    float s = 0.f;
    for (int64_t j = 0; j < n; ++j) s += 1.f / (1.f + std::exp(-x[i * n + j]));
    float r = std::nearbyint(s / speed);  // round half to even (default FP rounding mode), like torch.round
    int64_t d = std::max<int64_t>(1, static_cast<int64_t>(r));
    dur[i] = d;
    sum += d;
  }
  total[0] = sum;
}

bool repeat_interleave_f32(const float* x, const int64_t* dur, float* out, int64_t t, int64_t c, int64_t frames) {
  int64_t total = 0;
  for (int64_t i = 0; i < t; ++i) {
    if (dur[i] < 0) return false;
    total += dur[i];
  }
  if (total != frames) return false;  // never truncate or pad silently
  int64_t f = 0;
  for (int64_t i = 0; i < t; ++i)
    for (int64_t r = 0; r < dur[i]; ++r, ++f)
      std::memcpy(out + f * c, x + i * c, static_cast<size_t>(c) * sizeof(float));
  return true;
}

}  // namespace lk::kernels::cpu

namespace lk::rt {

namespace {

const float* F(const KernelArgs& a, int i) { return static_cast<const float*>(a.ptrs[i]); }
float* O(const KernelArgs& a, int i) { return static_cast<float*>(a.ptrs[i]); }

// Unary elementwise, any rank: ints = x dims then out dims (no int attrs).
int64_t numelHalf(const KernelArgs& a, uint32_t numTensors) {
  int64_t n = 1;
  for (uint32_t i = 0; i < a.numInts / numTensors; ++i) n *= a.ints[i];
  return n;
}

// ints: table(V, D), ids(T), out(T, D)
void runEmbedding(ExecCtx& ctx, const KernelArgs& a) {
  if (!kernels::cpu::embedding_f32(F(a, 0), static_cast<const int64_t*>(a.ptrs[1]), O(a, 2), a.ints[2], a.ints[1],
                                   a.ints[0]))
    ctx.error = "id out of range of the embedding table";
}

// ints: q(T, D), k(T, D), v(T, D), out(T, D), heads
void runAttention(ExecCtx&, const KernelArgs& a) {
  int64_t heads = a.ints[8];
  kernels::cpu::attention_f32(F(a, 0), F(a, 1), F(a, 2), O(a, 3), a.ints[0], heads, a.ints[1] / heads);
}

// ints: qkv(T, 3D), out(T, D), heads
template <bool Causal>
void runAttentionQkv(ExecCtx&, const KernelArgs& a) {
  const int64_t heads = a.ints[4];
  kernels::cpu::attention_qkv_f32(F(a, 0), O(a, 1), a.ints[0], heads, a.ints[3] / heads, Causal);
}

// ints: qkv(T, 3D), out(T, D), heads, mode, qb
void runAttentionQkvCausalBlk(ExecCtx&, const KernelArgs& a) {
  const int64_t heads = a.ints[4];
  kernels::cpu::attention_qkv_causal_blk_f32(F(a, 0), O(a, 1), a.ints[0], heads, a.ints[3] / heads, a.ints[6],
                                             static_cast<int>(a.ints[5]));
}

void runGelu(ExecCtx&, const KernelArgs& a) { kernels::cpu::gelu_tanh_f32(F(a, 0), O(a, 1), numelHalf(a, 2)); }
void runLeakyRelu(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::leaky_relu_f32(F(a, 0), O(a, 1), numelHalf(a, 2), static_cast<float>(a.floats[0]));
}
void runScale(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::scale_f32(F(a, 0), O(a, 1), numelHalf(a, 2), static_cast<float>(a.floats[0]));
}
void runAdd3Scale(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::add3_scale_f32(F(a, 0), F(a, 1), F(a, 2), O(a, 3), numelHalf(a, 4), static_cast<float>(a.floats[0]));
}

// ints: x(C, L), alpha(C), out(C, L)
void runSnake(ExecCtx&, const KernelArgs& a) { kernels::cpu::snake_f32(F(a, 0), F(a, 1), O(a, 2), a.ints[0], a.ints[1]); }

// ints: v(C), out(R, C), rows
void runBroadcastRows(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::broadcast_rows_f32(F(a, 0), O(a, 1), a.ints[1], a.ints[2]);
}

// ints: x_i(R_i, C) for each input, out(sum R, C)
void runConcatRows(ExecCtx&, const KernelArgs& a) {
  const int64_t n = a.numPtrs - 1;
  const float* xs[8];
  int64_t rows[8];
  for (int64_t i = 0; i < n; ++i) {
    xs[i] = F(a, static_cast<int>(i));
    rows[i] = a.ints[2 * i];
  }
  kernels::cpu::concat_rows_f32(xs, rows, n, O(a, static_cast<int>(n)), a.ints[1]);
}

// ints: x_i(R, C_i) for each input, out(R, sum C)
void runConcatCols(ExecCtx&, const KernelArgs& a) {
  const int64_t n = a.numPtrs - 1;
  const float* xs[8];
  int64_t cols[8];
  for (int64_t i = 0; i < n; ++i) {
    xs[i] = F(a, static_cast<int>(i));
    cols[i] = a.ints[2 * i + 1];
  }
  kernels::cpu::concat_cols_f32(xs, cols, n, O(a, static_cast<int>(n)), a.ints[0]);
}

// ptrs: x, speed (scalar slot), dur, total; ints: x(T, N), dur(T), total(1)
void runDurations(ExecCtx&, const KernelArgs& a) {
  kernels::cpu::durations_f32(F(a, 0), *F(a, 1), static_cast<int64_t*>(a.ptrs[2]), static_cast<int64_t*>(a.ptrs[3]),
                              a.ints[0], a.ints[1]);
}

// ints: x(T, C), dur(T), out(F, C), frames
void runRepeatInterleave(ExecCtx& ctx, const KernelArgs& a) {
  if (!kernels::cpu::repeat_interleave_f32(F(a, 0), static_cast<const int64_t*>(a.ptrs[1]), O(a, 2), a.ints[0],
                                           a.ints[1], a.ints[5]))
    ctx.error = "durations don't sum to the frame count (or are negative)";
}

}  // namespace

void registerCpuKokoroMiscKernels() {
  registerKernel({"embedding_f32_cpu", Device::Cpu, nullptr, runEmbedding, nullptr});
  registerKernel({"attention_f32_cpu", Device::Cpu, nullptr, runAttention, nullptr});
  registerKernel({"attention_qkv_f32_cpu", Device::Cpu, nullptr, runAttentionQkv<false>, nullptr});
  registerKernel({"attention_qkv_causal_f32_cpu", Device::Cpu, nullptr, runAttentionQkv<true>, nullptr});
  registerKernel({"attention_qkv_causal_blk_f32_cpu", Device::Cpu, nullptr, runAttentionQkvCausalBlk, nullptr});
  registerKernel({"gelu_tanh_f32_cpu", Device::Cpu, nullptr, runGelu, nullptr});
  registerKernel({"leaky_relu_f32_cpu", Device::Cpu, nullptr, runLeakyRelu, nullptr});
  registerKernel({"scale_f32_cpu", Device::Cpu, nullptr, runScale, nullptr});
  registerKernel({"add3_scale_f32_cpu", Device::Cpu, nullptr, runAdd3Scale, nullptr});
  registerKernel({"snake_f32_cpu", Device::Cpu, nullptr, runSnake, nullptr});
  registerKernel({"broadcast_rows_f32_cpu", Device::Cpu, nullptr, runBroadcastRows, nullptr});
  registerKernel({"concat_rows_f32_cpu", Device::Cpu, nullptr, runConcatRows, nullptr});
  registerKernel({"concat_cols_f32_cpu", Device::Cpu, nullptr, runConcatCols, nullptr});
  registerKernel({"durations_f32_cpu", Device::Cpu, nullptr, runDurations, nullptr});
  registerKernel({"repeat_interleave_f32_cpu", Device::Cpu, nullptr, runRepeatInterleave, nullptr});
}

}  // namespace lk::rt
