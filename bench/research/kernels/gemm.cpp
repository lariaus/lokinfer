// CPU GEMM/GEMV microbench: Accelerate cblas vs naive vs BLIS-style NEON 8x12 f32 kernel, + NEON GEMV f32/f16.
// Build: clang++ -O3 -std=c++20 -march=armv8.6-a+fp16 -DACCELERATE_NEW_LAPACK gemm.cpp -framework Accelerate -o gemm
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/qos.h>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class F> double median_time(F f, double min_total = 0.3) {
  f();
  double t0 = now(); f(); double one = now() - t0;
  int reps = std::max(5, (int)(min_total / std::max(one, 1e-7)));
  reps = std::min(reps, 20000);
  std::vector<double> ts;
  int batches = 7; int per = std::max(1, reps / batches);
  for (int b = 0; b < batches; b++) {
    double t = now(); for (int i = 0; i < per; i++) f(); ts.push_back((now() - t) / per);
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

static void* amalloc(size_t n) { void* p; posix_memalign(&p, 128, n); return p; }

// ---------- naive (compiler autovectorized) ----------
static void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; i++) {
    float* c = C + (size_t)i * N; memset(c, 0, N * 4);
    for (int k = 0; k < K; k++) { float a = A[(size_t)i * K + k]; const float* b = B + (size_t)k * N; for (int j = 0; j < N; j++) c[j] += a * b[j]; }
  }
}

// ---------- BLIS-style: pack A (MR=8) and B (NR=12), 8x12 micro-kernel ----------
enum { MR = 8, NR = 12, KC = 256, MC = 128, NC = 3072 };
static inline void ukernel_8x12(int kc, const float* __restrict a, const float* __restrict b, float* __restrict C, int ldc, bool accum) {
  float32x4_t c[8][3];
  if (accum) { for (int i = 0; i < 8; i++) for (int j = 0; j < 3; j++) c[i][j] = vld1q_f32(C + i * ldc + 4 * j); }
  else { for (int i = 0; i < 8; i++) for (int j = 0; j < 3; j++) c[i][j] = vdupq_n_f32(0); }
  for (int k = 0; k < kc; k++) {
    float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
    float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
    c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a0, 0); c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a0, 0); c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a0, 0);
    c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a0, 1); c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a0, 1); c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a0, 1);
    c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a0, 2); c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a0, 2); c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a0, 2);
    c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a0, 3); c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a0, 3); c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a0, 3);
    c[4][0] = vfmaq_laneq_f32(c[4][0], b0, a1, 0); c[4][1] = vfmaq_laneq_f32(c[4][1], b1, a1, 0); c[4][2] = vfmaq_laneq_f32(c[4][2], b2, a1, 0);
    c[5][0] = vfmaq_laneq_f32(c[5][0], b0, a1, 1); c[5][1] = vfmaq_laneq_f32(c[5][1], b1, a1, 1); c[5][2] = vfmaq_laneq_f32(c[5][2], b2, a1, 1);
    c[6][0] = vfmaq_laneq_f32(c[6][0], b0, a1, 2); c[6][1] = vfmaq_laneq_f32(c[6][1], b1, a1, 2); c[6][2] = vfmaq_laneq_f32(c[6][2], b2, a1, 2);
    c[7][0] = vfmaq_laneq_f32(c[7][0], b0, a1, 3); c[7][1] = vfmaq_laneq_f32(c[7][1], b1, a1, 3); c[7][2] = vfmaq_laneq_f32(c[7][2], b2, a1, 3);
    a += 8; b += 12;
  }
  for (int i = 0; i < 8; i++) for (int j = 0; j < 3; j++) vst1q_f32(C + i * ldc + 4 * j, c[i][j]);
}
// requires M%8==0, N%12==0 for brevity (shapes chosen accordingly)
static void gemm_blis(int M, int N, int K, const float* A, const float* B, float* C, float* packA, float* packB) {
  for (int jc = 0; jc < N; jc += NC) {
    int nc = std::min((int)NC, N - jc);
    for (int pc = 0; pc < K; pc += KC) {
      int kc = std::min((int)KC, K - pc);
      // pack B[pc:pc+kc, jc:jc+nc] into NR panels
      for (int jr = 0; jr < nc; jr += NR) {
        float* dst = packB + (size_t)jr * kc;
        for (int k = 0; k < kc; k++) { const float* src = B + (size_t)(pc + k) * N + jc + jr; memcpy(dst + k * NR, src, NR * 4); }
      }
      for (int ic = 0; ic < M; ic += MC) {
        int mc = std::min((int)MC, M - ic);
        for (int ir = 0; ir < mc; ir += MR) {
          float* dst = packA + (size_t)ir * kc;
          for (int k = 0; k < kc; k++) for (int i = 0; i < MR; i++) dst[k * MR + i] = A[(size_t)(ic + ir + i) * K + pc + k];
        }
        for (int jr = 0; jr < nc; jr += NR)
          for (int ir = 0; ir < mc; ir += MR)
            ukernel_8x12(kc, packA + (size_t)ir * kc, packB + (size_t)jr * kc, C + (size_t)(ic + ir) * N + jc + jr, N, pc > 0);
      }
    }
  }
}

// ---------- GEMV: y[N] = W[N,K] x[K] (row-major W, i.e. nn.Linear layout) ----------
static void gemv_f32(int N, int K, const float* W, const float* x, float* y) {
  for (int n = 0; n < N; n += 4) {
    float32x4_t acc[4][2] = {};
    for (int k = 0; k < K; k += 8) {
      float32x4_t x0 = vld1q_f32(x + k), x1 = vld1q_f32(x + k + 4);
      for (int r = 0; r < 4; r++) {
        const float* w = W + (size_t)(n + r) * K + k;
        acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(w), x0);
        acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(w + 4), x1);
      }
    }
    for (int r = 0; r < 4; r++) y[n + r] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
  }
}
// f16 weights, f32 activations/accum (convert on the fly)
static void gemv_f16w(int N, int K, const __fp16* W, const float* x, float* y) {
  for (int n = 0; n < N; n += 4) {
    float32x4_t acc[4][2] = {};
    for (int k = 0; k < K; k += 8) {
      float32x4_t x0 = vld1q_f32(x + k), x1 = vld1q_f32(x + k + 4);
      for (int r = 0; r < 4; r++) {
        float16x8_t w = vld1q_f16(W + (size_t)(n + r) * K + k);
        acc[r][0] = vfmaq_f32(acc[r][0], vcvt_f32_f16(vget_low_f16(w)), x0);
        acc[r][1] = vfmaq_f32(acc[r][1], vcvt_high_f32_f16(w), x1);
      }
    }
    for (int r = 0; r < 4; r++) y[n + r] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
  }
}

static float maxerr(const float* a, const float* b, size_t n) { float m = 0; for (size_t i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i])); return m; }

int main(int argc, char** argv) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  const char* vt = getenv("VECLIB_MAXIMUM_THREADS");
  printf("VECLIB_MAXIMUM_THREADS=%s\n", vt ? vt : "(default)");
  struct S { int M, N, K; const char* what; };
  S shapes[] = {
      {512, 768, 768, "ALBERT-ish proj, 512 tok"},
      {64, 768, 768, "ALBERT proj, 64 tok"},
      {512, 3072, 768, "FFN up, 512 tok"},
      {200, 1024, 512, "LSTM input proj (T=200, 4H=1024)"},
      {512, 516, 1536, "conv1d k3 512ch im2col (T=516)"},
      {1024, 1020, 1024, "square-ish 1k"},
  };
  printf("\n== GEMM f32 (C[M,N] = A[M,K] B[K,N]) ==\n%-36s %6s %6s %6s | %9s %9s %9s\n", "shape", "M", "N", "K", "cblas", "blis8x12", "naive");
  for (auto s : shapes) {
    int M = s.M, N = s.N, K = s.K;
    float* A = (float*)amalloc((size_t)M * K * 4); float* B = (float*)amalloc((size_t)K * N * 4);
    float* C = (float*)amalloc((size_t)M * N * 4); float* C2 = (float*)amalloc((size_t)M * N * 4);
    float* pA = (float*)amalloc((size_t)MC * KC * 4); float* pB = (float*)amalloc((size_t)NC * KC * 4);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (rand() % 1000) / 1000.f - .5f;
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (rand() % 1000) / 1000.f - .5f;
    double flop = 2.0 * M * N * K;
    double tc = median_time([&] { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N); });
    double tb = -1, tn = -1;
    if (M % MR == 0 && N % NR == 0) {
      tb = median_time([&] { gemm_blis(M, N, K, A, B, C2, pA, pB); });
      float e = maxerr(C, C2, (size_t)M * N); if (e > 1e-2) printf("  blis err %g\n", e);
    }
    if (flop < 2e9) tn = median_time([&] { gemm_naive(M, N, K, A, B, C2); }, 0.2);
    printf("%-36s %6d %6d %6d | %7.1f GF %7.1f GF %7.1f GF\n", s.what, M, N, K, flop / tc / 1e9, tb > 0 ? flop / tb / 1e9 : 0, tn > 0 ? flop / tn / 1e9 : 0);
    free(A); free(B); free(C); free(C2); free(pA); free(pB);
  }

  printf("\n== GEMV batch=1 (y[N] = W[N,K] x) ==\n%-28s %6s %6s | %16s %16s %16s\n", "shape", "N", "K", "cblas_sgemv", "neon f32", "neon f16-weights");
  S gv[] = {{1, 768, 768, "ALBERT proj"}, {1, 3072, 768, "FFN up"}, {1, 1024, 512, "LSTM gates (H=256 bi? x2)"},
            {1, 1024, 256, "LSTM recurrent 4H x H"}, {1, 2304, 768, "GPT-2 qkv"}, {1, 50256, 768, "GPT-2 lm_head"}};
  for (auto s : gv) {
    int N = s.N, K = s.K;
    float* W = (float*)amalloc((size_t)N * K * 4); __fp16* Wh = (__fp16*)amalloc((size_t)N * K * 2);
    float* x = (float*)amalloc(K * 4); float* y = (float*)amalloc(N * 4); float* y2 = (float*)amalloc(N * 4);
    for (size_t i = 0; i < (size_t)N * K; i++) { W[i] = (rand() % 1000) / 1000.f - .5f; Wh[i] = (__fp16)W[i]; }
    for (int i = 0; i < K; i++) x[i] = (rand() % 1000) / 1000.f - .5f;
    double tc = median_time([&] { cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1, W, K, x, 1, 0, y, 1); });
    double tn = median_time([&] { gemv_f32(N, K, W, x, y2); });
    float e = maxerr(y, y2, N); if (e > 1e-2) printf("  gemv err %g\n", e);
    double th = median_time([&] { gemv_f16w(N, K, Wh, x, y2); });
    double bytes = (double)N * K * 4;
    printf("%-28s %6d %6d | %5.1fus %5.0fGB/s %5.1fus %5.0fGB/s %5.1fus %5.0fGB/s\n", s.what, N, K, tc * 1e6, bytes / tc / 1e9, tn * 1e6, bytes / tn / 1e9,
           th * 1e6, bytes / 2 / th / 1e9);
    free(W); free(Wh); free(x); free(y); free(y2);
  }
}
