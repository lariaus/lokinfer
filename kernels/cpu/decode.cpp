// Single-token transformer decode with a persistent KV cache (M6: GPT-2 decode; research-models.md §5.3).
//
//   kv_write          kcache[pos] = k, vcache[pos] = v, read from this step's fused QKV row qkv [1, 3D]
//                     (inout: the caches are persistent buffers, written in place)
//   attention_decode  one query (qkv's q part) against cache rows 0..pos: out [1, D]
//   kv_write_rows / attention_cache   the chunked form: T new tokens at pos0 .. pos0+T-1 (prefill and decode in one
//                     compiled model sharing its caches; T = 1 is a decode step)
#include <algorithm>
#include <cmath>
#include <cstring>

#include "kokoro_common.h"
#include "neon_math.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

namespace {

const float* F(const KernelArgs& a, int i) { return static_cast<const float*>(a.ptrs[i]); }

// ptrs: kc, vc, qkv, pos. ints: kc(P, D), vc(P, D), qkv(1, 3D), pos(1)
void runKvWrite(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t cap = a.ints[0], d = a.ints[1];
  const int64_t pos = *static_cast<const int64_t*>(a.ptrs[3]);
  if (pos < 0 || pos >= cap) {
    ctx.error = "kv_write: pos out of the cache range";
    return;
  }
  const float* qkv = F(a, 2);
  std::memcpy(static_cast<float*>(a.ptrs[0]) + pos * d, qkv + d, static_cast<size_t>(d) * sizeof(float));
  std::memcpy(static_cast<float*>(a.ptrs[1]) + pos * d, qkv + 2 * d, static_cast<size_t>(d) * sizeof(float));
}

// ptrs: qkv, kc, vc, pos, out. ints: qkv(1, 3D), kc(P, D), vc(P, D), pos(1), out(1, D), heads
void runAttentionDecode(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t d = a.ints[3], cap = a.ints[2], heads = a.ints[9], hd = d / heads;
  const int64_t pos = *static_cast<const int64_t*>(a.ptrs[3]);
  if (pos < 0 || pos >= cap) {
    ctx.error = "attention_decode: pos out of the cache range";
    return;
  }
  const float* q = F(a, 0);
  const float* kc = F(a, 1);
  const float* vc = F(a, 2);
  float* out = static_cast<float*>(a.ptrs[4]);
  const int64_t n = pos + 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  // One head per pool task, plain NEON (the old per-head cblas_sgemv + vDSP loop took 0.96 ms for 12 layers at
  // P = 512 on the iPhone, single-threaded): scores = K_h q_h, softmax, out_h = V_h^T p.
  auto head = [&](int64_t h) {
    const int64_t off = h * hd;
    float* sc = kernels::cpu::detail::scratch(static_cast<size_t>(n), 1);
    const float* qh = q + off;
    if (hd % 16 == 0) {
      for (int64_t j = 0; j < n; ++j) {
        const float* kr = kc + j * d + off;
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
        for (int64_t c = 0; c < hd; c += 16) {
          a0 = vfmaq_f32(a0, vld1q_f32(kr + c), vld1q_f32(qh + c));
          a1 = vfmaq_f32(a1, vld1q_f32(kr + c + 4), vld1q_f32(qh + c + 4));
          a2 = vfmaq_f32(a2, vld1q_f32(kr + c + 8), vld1q_f32(qh + c + 8));
          a3 = vfmaq_f32(a3, vld1q_f32(kr + c + 12), vld1q_f32(qh + c + 12));
        }
        sc[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * scale;
      }
    } else {
      for (int64_t j = 0; j < n; ++j) {
        float acc = 0.f;
        for (int64_t c = 0; c < hd; ++c) acc += kc[j * d + off + c] * qh[c];
        sc[j] = acc * scale;
      }
    }
    float m = sc[0];
    for (int64_t j = 1; j < n; ++j) m = std::max(m, sc[j]);
    float sum = 0.f;
    int64_t j = 0;
    const float32x4_t mv = vdupq_n_f32(m);
    float32x4_t sv = vdupq_n_f32(0.f);
    for (; j + 4 <= n; j += 4) {
      const float32x4_t e = kernels::cpu::detail::exp4(vsubq_f32(vld1q_f32(sc + j), mv));
      vst1q_f32(sc + j, e);
      sv = vaddq_f32(sv, e);
    }
    sum = vaddvq_f32(sv);
    for (; j < n; ++j) sum += (sc[j] = std::exp(sc[j] - m));
    const float inv = 1.f / sum;
    float* oh = out + off;
    if (hd == 64) {
      float32x4_t acc[16];
      for (auto& v : acc) v = vdupq_n_f32(0.f);
      for (int64_t r = 0; r < n; ++r) {
        const float32x4_t pv = vdupq_n_f32(sc[r]);
        const float* vr = vc + r * d + off;
        for (int c = 0; c < 16; ++c) acc[c] = vfmaq_f32(acc[c], pv, vld1q_f32(vr + 4 * c));
      }
      const float32x4_t iv = vdupq_n_f32(inv);
      for (int c = 0; c < 16; ++c) vst1q_f32(oh + 4 * c, vmulq_f32(acc[c], iv));
    } else {
      for (int64_t c = 0; c < hd; ++c) oh[c] = 0.f;
      for (int64_t r = 0; r < n; ++r)
        for (int64_t c = 0; c < hd; ++c) oh[c] += sc[r] * vc[r * d + off + c];
      for (int64_t c = 0; c < hd; ++c) oh[c] *= inv;
    }
  };
  if (n * d < 16384) {
    for (int64_t h = 0; h < heads; ++h) head(h);
  } else {
    rt::ThreadPool::global().parallelFor(heads, head);
  }
}

// Chunked prefill / decode (T new tokens at positions pos0 .. pos0+T-1):
// ptrs: kc, vc, qkv, pos. ints: kc(P, D), vc(P, D), qkv(T, 3D), pos(1). kc[pos0 + t] = k_t, vc[pos0 + t] = v_t.
void runKvWriteRows(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t cap = a.ints[0], d = a.ints[1], t = a.ints[4];
  const int64_t pos0 = *static_cast<const int64_t*>(a.ptrs[3]);
  if (pos0 < 0 || pos0 + t > cap) {
    ctx.error = "kv_write_rows: pos + T exceeds the cache";
    return;
  }
  const float* qkv = F(a, 2);
  for (int64_t i = 0; i < t; ++i) {
    std::memcpy(static_cast<float*>(a.ptrs[0]) + (pos0 + i) * d, qkv + i * 3 * d + d, static_cast<size_t>(d) * 4);
    std::memcpy(static_cast<float*>(a.ptrs[1]) + (pos0 + i) * d, qkv + i * 3 * d + 2 * d, static_cast<size_t>(d) * 4);
  }
}

// Queries qkv[t, 0:D] at positions pos0 + t attend to cache rows 0 .. pos0 + t (causal).
// ptrs: qkv, kc, vc, pos, out. ints: qkv(T, 3D), kc(P, D), vc(P, D), pos(1), out(T, D), heads
void runAttentionCache(ExecCtx& ctx, const KernelArgs& a) {
  const int64_t t = a.ints[0], cap = a.ints[2], d = a.ints[3], heads = a.ints[9], hd = d / heads;
  const int64_t pos0 = *static_cast<const int64_t*>(a.ptrs[3]);
  if (pos0 < 0 || pos0 + t > cap) {
    ctx.error = "attention_cache: pos + T exceeds the cache";
    return;
  }
  const float* qkv = F(a, 0);
  const float* kc = F(a, 1);
  const float* vc = F(a, 2);
  float* out = static_cast<float*>(a.ptrs[4]);
  const int64_t n = pos0 + t;  // cache rows visible to the last query
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  auto head = [&](int64_t h) {
    float* s = kernels::cpu::detail::scratch(static_cast<size_t>(t * n), 1);
    const int64_t off = h * hd;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<__LAPACK_int>(t), static_cast<__LAPACK_int>(n),
                static_cast<__LAPACK_int>(hd), scale, qkv + off, static_cast<__LAPACK_int>(3 * d), kc + off,
                static_cast<__LAPACK_int>(d), 0.f, s, static_cast<__LAPACK_int>(n));
    for (int64_t i = 0; i < t; ++i) {
      float* row = s + i * n;
      const int64_t vis = pos0 + i + 1;
      const int vi = static_cast<int>(vis);
      float m, sum;
      vDSP_maxv(row, 1, &m, static_cast<vDSP_Length>(vis));
      m = -m;
      vDSP_vsadd(row, 1, &m, row, 1, static_cast<vDSP_Length>(vis));
      vvexpf(row, row, &vi);
      vDSP_sve(row, 1, &sum, static_cast<vDSP_Length>(vis));
      const float inv = 1.f / sum;
      vDSP_vsmul(row, 1, &inv, row, 1, static_cast<vDSP_Length>(vis));
      if (vis < n) std::fill(row + vis, row + n, 0.f);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<__LAPACK_int>(t), static_cast<__LAPACK_int>(hd),
                static_cast<__LAPACK_int>(n), 1.f, s, static_cast<__LAPACK_int>(n), vc + off,
                static_cast<__LAPACK_int>(d), 0.f, out + off, static_cast<__LAPACK_int>(d));
  };
  if (t * n * hd * heads < (int64_t{1} << 21)) {
    for (int64_t h = 0; h < heads; ++h) head(h);
  } else {
    kernels::cpu::detail::parallelForSt(heads, head);
  }
}

// (s, LN(s)) with s = o + bias + r over rows of [R, C]: the residual stream after a sub-block and the next sub-block's
// LayerNorm input in one pass (CPU twin of add_bias_layernorm2_f32_gpu). Row blocks on the pool when large.
// ptrs: o, bias, r, gamma, beta, s, ln. ints: o(R, C), bias(C), r(R, C), gamma(C), beta(C), s(R, C), ln(R, C).
void runAddBiasLayerNorm2(ExecCtx&, const KernelArgs& a) {
  const int64_t rows = a.ints[0], c = a.ints[1];
  const float eps = a.floats[0];
  const float *o = F(a, 0), *bias = F(a, 1), *res = F(a, 2), *g = F(a, 3), *b = F(a, 4);
  float* s = static_cast<float*>(a.ptrs[5]);
  float* ln = static_cast<float*>(a.ptrs[6]);
  auto rowsFn = [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float *orow = o + r * c, *rrow = res + r * c;
      float* srow = s + r * c;
      float* lrow = ln + r * c;
      int64_t j = 0;
      for (; j + 4 <= c; j += 4)
        vst1q_f32(srow + j, vaddq_f32(vaddq_f32(vld1q_f32(orow + j), vld1q_f32(bias + j)), vld1q_f32(rrow + j)));
      for (; j < c; ++j) srow[j] = orow[j] + bias[j] + rrow[j];
      float mean, var;
      kernels::cpu::detail::meanVar(srow, c, mean, var);
      const float rstd = 1.0f / std::sqrt(var + eps);
      const float32x4_t vm = vdupq_n_f32(mean), vr = vdupq_n_f32(rstd);
      for (j = 0; j + 4 <= c; j += 4)
        vst1q_f32(lrow + j, vfmaq_f32(vld1q_f32(b + j), vmulq_f32(vsubq_f32(vld1q_f32(srow + j), vm), vr),
                                      vld1q_f32(g + j)));
      for (; j < c; ++j) lrow[j] = (srow[j] - mean) * rstd * g[j] + b[j];
    }
  };
  const int64_t tasks = rows * c >= 32768 ? std::min<int64_t>(rows, 4 * rt::ThreadPool::global().threads()) : 1;
  if (tasks <= 1) return rowsFn(0, rows);
  const int64_t rb = (rows + tasks - 1) / tasks;
  rt::ThreadPool::global().parallelFor((rows + rb - 1) / rb,
                                       [&](int64_t i) { rowsFn(i * rb, std::min(rows, (i + 1) * rb)); });
}

// out[i] = pos[0] + i (position ids of T new tokens). ints: pos(1), out(T), length
void runArangeOffset(ExecCtx&, const KernelArgs& a) {
  const int64_t base = *static_cast<const int64_t*>(a.ptrs[0]), t = a.ints[1];
  int64_t* out = static_cast<int64_t*>(a.ptrs[1]);
  for (int64_t i = 0; i < t; ++i) out[i] = base + i;
}

}  // namespace

void registerCpuDecodeKernels() {
  registerKernel({"add_bias_layernorm2_f32_cpu", Device::Cpu, nullptr, runAddBiasLayerNorm2, nullptr});
  registerKernel({"arange_offset_i64_cpu", Device::Cpu, nullptr, runArangeOffset, nullptr});
  registerKernel({"kv_write_rows_f32_cpu", Device::Cpu, nullptr, runKvWriteRows, nullptr});
  registerKernel({"attention_cache_f32_cpu", Device::Cpu, nullptr, runAttentionCache, nullptr});
  registerKernel({"kv_write_f32_cpu", Device::Cpu, nullptr, runKvWrite, nullptr});
  registerKernel({"attention_decode_f32_cpu", Device::Cpu, nullptr, runAttentionDecode, nullptr});
}

}  // namespace lk::rt
