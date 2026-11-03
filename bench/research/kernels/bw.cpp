// Memory bandwidth (read) per working-set size, 1..N threads; fork-join latency of dispatch_apply vs spin pool;
// better GEMV (8 rows x 2 acc, k unroll 8) single/multi-thread.
// Build: clang++ -O3 -std=c++20 -march=armv8.6-a+fp16 bw.cpp -o bw
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/qos.h>
#include <thread>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class F> double median_time(F f, double min_total = 0.3) {
  f(); double t0 = now(); f(); double one = now() - t0;
  int reps = std::clamp((int)(min_total / std::max(one, 1e-7)), 5, 200000);
  std::vector<double> ts; int per = std::max(1, reps / 7);
  for (int b = 0; b < 7; b++) { double t = now(); for (int i = 0; i < per; i++) f(); ts.push_back((now() - t) / per); }
  std::sort(ts.begin(), ts.end()); return ts[3];
}

// --- simple spin thread pool (workers spin on a generation counter) ---
struct SpinPool {
  int n; std::vector<std::thread> th; std::atomic<long> gen{0}; std::atomic<int> done{0}; std::atomic<bool> quit{false};
  std::function<void(int)> job;
  SpinPool(int n_, qos_class_t q) : n(n_) {
    for (int i = 1; i < n; i++) th.emplace_back([this, i, q] {
      pthread_set_qos_class_self_np(q, 0);
      long seen = 0;
      while (true) {
        long g; while ((g = gen.load(std::memory_order_acquire)) == seen) { if (quit) return; __builtin_arm_yield(); }
        seen = g; if (quit) return; job(i); done.fetch_add(1, std::memory_order_acq_rel);
      }
    });
  }
  void run(std::function<void(int)> f) {
    job = std::move(f); done.store(0); gen.fetch_add(1, std::memory_order_acq_rel);
    job(0);
    while (done.load(std::memory_order_acquire) != n - 1) __builtin_arm_yield();
  }
  ~SpinPool() { quit = true; gen.fetch_add(1); for (auto& t : th) t.join(); }
};

__attribute__((noinline)) float sum_range(const float* p, size_t n) {
  float32x4_t a[8] = {};
  for (size_t i = 0; i < n; i += 32)
    for (int j = 0; j < 8; j++) a[j] = vaddq_f32(a[j], vld1q_f32(p + i + 4 * j));
  for (int j = 1; j < 8; j++) a[0] = vaddq_f32(a[0], a[j]);
  return vaddvq_f32(a[0]);
}

// 8 rows at a time, 2 accumulators each (16 acc), loads 8 W vectors + 2 x per step
static void gemv_rows(int n0, int n1, int K, const float* W, const float* x, float* y) {
  for (int n = n0; n < n1; n += 8) {
    float32x4_t acc[8][2] = {};
    for (int k = 0; k < K; k += 8) {
      float32x4_t x0 = vld1q_f32(x + k), x1 = vld1q_f32(x + k + 4);
#pragma unroll
      for (int r = 0; r < 8; r++) {
        const float* w = W + (size_t)(n + r) * K + k;
        acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(w), x0);
        acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(w + 4), x1);
      }
    }
    for (int r = 0; r < 8; r++) y[n + r] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
  }
}
// f16 weights + f16 x, FMLAL widening into f32 (fp16fml): no conversion instructions
static void gemv_rows_f16(int n0, int n1, int K, const __fp16* W, const __fp16* x, float* y) {
  for (int n = n0; n < n1; n += 8) {
    float32x4_t acc[8][2] = {};
    for (int k = 0; k < K; k += 8) {
      float16x8_t xv = vld1q_f16(x + k);
#pragma unroll
      for (int r = 0; r < 8; r++) {
        float16x8_t w = vld1q_f16(W + (size_t)(n + r) * K + k);
        acc[r][0] = vfmlalq_low_f16(acc[r][0], w, xv);
        acc[r][1] = vfmlalq_high_f16(acc[r][1], w, xv);
      }
    }
    for (int r = 0; r < 8; r++) y[n + r] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
  }
}

volatile float sink;
int main() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  size_t maxb = 1ull << 30; float* buf; posix_memalign((void**)&buf, 16384, maxb);
  for (size_t i = 0; i < maxb / 4; i++) buf[i] = 1.0f;

  printf("== Read bandwidth (GB/s), QoS user-interactive ==\n%10s", "size");
  int thr[] = {1, 2, 4, 8, 12};
  for (int t : thr) printf(" %7dT", t);
  printf("\n");
  size_t sizes[] = {64 << 10, 1 << 20, 8 << 20, 32 << 20, 128 << 20, 1ull << 30};
  for (size_t sz : sizes) {
    printf("%9zuK", sz >> 10);
    for (int t : thr) {
      SpinPool pool(t, QOS_CLASS_USER_INTERACTIVE);
      size_t n = sz / 4, chunk = (n / t) & ~(size_t)31;
      double tm = median_time([&] { pool.run([&](int i) { sink = sum_range(buf + i * chunk, chunk); }); }, 0.3);
      printf(" %8.1f", (double)chunk * t * 4 / tm / 1e9);
    }
    printf("\n");
  }

  printf("\n== Fork-join latency (empty job) ==\n");
  for (int t : {2, 4, 8, 12}) {
    SpinPool pool(t, QOS_CLASS_USER_INTERACTIVE);
    double ts = median_time([&] { pool.run([](int) {}); }, 0.2);
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    double td = median_time([&] { dispatch_apply(t, q, ^(size_t) {}); }, 0.2);
    printf("  %2d threads: spin-pool %6.2f us   dispatch_apply %6.2f us\n", t, ts * 1e6, td * 1e6);
  }

  printf("\n== GEMV 8-row kernel, f32 W / f16 W (FMLAL), threads ==\n");
  struct { int N, K; const char* w; } gv[] = {{768, 768, "768x768"}, {3072, 768, "3072x768"}, {1024, 256, "1024x256 (LSTM rec)"}, {2304, 768, "2304x768"}, {50256, 768, "50256x768 lm_head"}};
  for (auto s : gv) {
    int N = s.N, K = s.K;
    float* W; posix_memalign((void**)&W, 128, (size_t)N * K * 4); __fp16* Wh; posix_memalign((void**)&Wh, 128, (size_t)N * K * 2);
    float x[4096]; __fp16 xh[4096]; float* y = (float*)malloc(N * 4);
    for (size_t i = 0; i < (size_t)N * K; i++) { W[i] = (i % 7) * 0.01f; Wh[i] = (__fp16)W[i]; }
    for (int i = 0; i < K; i++) { x[i] = 0.5f; xh[i] = 0.5; }
    printf("%-22s", s.w);
    for (int t : {1, 2, 4, 8}) {
      SpinPool pool(t, QOS_CLASS_USER_INTERACTIVE);
      int rows = (N / 8 + t - 1) / t * 8;
      double tf = median_time([&] { pool.run([&](int i) { int a = i * rows, b = std::min(N, a + rows); if (a < b) gemv_rows(a, b, K, W, x, y); }); });
      double th = median_time([&] { pool.run([&](int i) { int a = i * rows, b = std::min(N, a + rows); if (a < b) gemv_rows_f16(a, b, K, Wh, xh, y); }); });
      printf(" | %dT f32 %6.1fus %4.0fGB/s f16 %6.1fus %4.0fGB/s", t, tf * 1e6, (double)N * K * 4 / tf / 1e9, th * 1e6, (double)N * K * 2 / th / 1e9);
    }
    printf("\n");
    free(W); free(Wh); free(y);
  }
}
