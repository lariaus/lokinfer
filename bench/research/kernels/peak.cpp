// Per-core instruction-throughput peaks: FMLA f32/f16, BFMMLA, SDOT, SMMLA, FMLAL.
// Build: clang++ -O3 -std=c++20 -march=armv8.6-a+fp16+bf16+i8mm+fp16fml peak.cpp -o peak
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <pthread.h>
#include <sys/qos.h>
#include <thread>
#include <vector>
#include <atomic>

using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

#define N_ACC 16
static const long ITERS = 50'000'000;

template <class F> double bench(F f) {
  f(1000000);
  double best = 1e30;
  for (int r = 0; r < 3; r++) { double t = now(); f(ITERS); best = std::min(best, now() - t); }
  return best;
}

__attribute__((noinline)) float32x4_t fma32(long n) {
  float32x4_t a[N_ACC]; for (int i = 0; i < N_ACC; i++) a[i] = vdupq_n_f32(i * 0.001f);
  float32x4_t b = vdupq_n_f32(0.999f), c = vdupq_n_f32(0.0001f);
  for (long k = 0; k < n; k++) {
#pragma unroll
    for (int i = 0; i < N_ACC; i++) a[i] = vfmaq_f32(a[i], b, c);
    asm volatile("" : "+w"(b));
  }
  float32x4_t s = a[0]; for (int i = 1; i < N_ACC; i++) s = vaddq_f32(s, a[i]); return s;
}
__attribute__((noinline)) float16x8_t fma16(long n) {
  float16x8_t a[N_ACC]; for (int i = 0; i < N_ACC; i++) a[i] = vdupq_n_f16((__fp16)(i * 0.001f));
  float16x8_t b = vdupq_n_f16(0.999), c = vdupq_n_f16(0.0001);
  for (long k = 0; k < n; k++) {
#pragma unroll
    for (int i = 0; i < N_ACC; i++) a[i] = vfmaq_f16(a[i], b, c);
    asm volatile("" : "+w"(b));
  }
  float16x8_t s = a[0]; for (int i = 1; i < N_ACC; i++) s = vaddq_f16(s, a[i]); return s;
}
__attribute__((noinline)) float32x4_t fmlal(long n) {
  float32x4_t a[N_ACC]; for (int i = 0; i < N_ACC; i++) a[i] = vdupq_n_f32(i * 0.001f);
  float16x8_t b = vdupq_n_f16(0.999), c = vdupq_n_f16(0.0001);
  for (long k = 0; k < n; k++) {
#pragma unroll
    for (int i = 0; i < N_ACC; i++) a[i] = vfmlalq_low_f16(a[i], b, c);
    asm volatile("" : "+w"(b));
  }
  float32x4_t s = a[0]; for (int i = 1; i < N_ACC; i++) s = vaddq_f32(s, a[i]); return s;
}
__attribute__((noinline)) float32x4_t bfmmla(long n) {
  float32x4_t a[N_ACC]; for (int i = 0; i < N_ACC; i++) a[i] = vdupq_n_f32(i * 0.001f);
  bfloat16x8_t b = vreinterpretq_bf16_u16(vdupq_n_u16(0x3f80)), c = b;
  for (long k = 0; k < n; k++) {
#pragma unroll
    for (int i = 0; i < N_ACC; i++) a[i] = vbfmmlaq_f32(a[i], b, c);
    asm volatile("" : "+w"(b));
  }
  float32x4_t s = a[0]; for (int i = 1; i < N_ACC; i++) s = vaddq_f32(s, a[i]); return s;
}
__attribute__((noinline)) float32x4_t bfdot(long n) {
  float32x4_t a[N_ACC]; for (int i = 0; i < N_ACC; i++) a[i] = vdupq_n_f32(i * 0.001f);
  bfloat16x8_t b = vreinterpretq_bf16_u16(vdupq_n_u16(0x3f80)), c = b;
  for (long k = 0; k < n; k++) {
#pragma unroll
    for (int i = 0; i < N_ACC; i++) a[i] = vbfdotq_f32(a[i], b, c);
    asm volatile("" : "+w"(b));
  }
  float32x4_t s = a[0]; for (int i = 1; i < N_ACC; i++) s = vaddq_f32(s, a[i]); return s;
}
__attribute__((noinline)) int32x4_t sdot(long n) {
  int32x4_t a[N_ACC]; for (int i = 0; i < N_ACC; i++) a[i] = vdupq_n_s32(i);
  int8x16_t b = vdupq_n_s8(1), c = vdupq_n_s8(2);
  for (long k = 0; k < n; k++) {
#pragma unroll
    for (int i = 0; i < N_ACC; i++) a[i] = vdotq_s32(a[i], b, c);
    asm volatile("" : "+w"(b));
  }
  int32x4_t s = a[0]; for (int i = 1; i < N_ACC; i++) s = vaddq_s32(s, a[i]); return s;
}
__attribute__((noinline)) int32x4_t smmla(long n) {
  int32x4_t a[N_ACC]; for (int i = 0; i < N_ACC; i++) a[i] = vdupq_n_s32(i);
  int8x16_t b = vdupq_n_s8(1), c = vdupq_n_s8(2);
  for (long k = 0; k < n; k++) {
#pragma unroll
    for (int i = 0; i < N_ACC; i++) a[i] = vmmlaq_s32(a[i], b, c);
    asm volatile("" : "+w"(b));
  }
  int32x4_t s = a[0]; for (int i = 1; i < N_ACC; i++) s = vaddq_s32(s, a[i]); return s;
}
// latency: dependent chain
__attribute__((noinline)) float32x4_t fma32_lat(long n) {
  float32x4_t a = vdupq_n_f32(0.1f), b = vdupq_n_f32(0.999f), c = vdupq_n_f32(0.0001f);
  for (long k = 0; k < n; k++) { a = vfmaq_f32(a, b, c); a = vfmaq_f32(a, b, c); a = vfmaq_f32(a, b, c); a = vfmaq_f32(a, b, c); }
  return a;
}

volatile float sink;
int main(int argc, char** argv) {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  bool ecore = argc > 1;
  if (ecore) pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
  double ops = (double)ITERS * N_ACC;  // instructions
  struct { const char* name; double flop_per_inst; double t; } r[8];
  r[0] = {"FMLA f32 (4 lanes, 8 flop)", 8, bench([](long n){ sink = vgetq_lane_f32(fma32(n),0); })};
  r[1] = {"FMLA f16 (8 lanes, 16 flop)", 16, bench([](long n){ sink = (float)vgetq_lane_f16(fma16(n),0); })};
  r[2] = {"FMLAL f16->f32 (4 lanes, 8 flop)", 8, bench([](long n){ sink = vgetq_lane_f32(fmlal(n),0); })};
  r[3] = {"BFMMLA (2x4x2, 32 flop)", 32, bench([](long n){ sink = vgetq_lane_f32(bfmmla(n),0); })};
  r[4] = {"BFDOT (4x2, 16 flop)", 16, bench([](long n){ sink = vgetq_lane_f32(bfdot(n),0); })};
  r[5] = {"SDOT int8 (4x4, 32 op)", 32, bench([](long n){ sink = vgetq_lane_s32(sdot(n),0); })};
  r[6] = {"SMMLA int8 (2x8x2, 64 op)", 64, bench([](long n){ sink = vgetq_lane_s32(smmla(n),0); })};
  for (int i = 0; i < 7; i++)
    printf("%-36s %6.2f Ginst/s  %7.1f GFLOP(OP)/s/core\n", r[i].name, ops / r[i].t / 1e9, ops * r[i].flop_per_inst / r[i].t / 1e9);
  double tl = bench([](long n){ sink = vgetq_lane_f32(fma32_lat(n/4),0); });
  printf("FMLA f32 dependent chain: %.3f ns/inst\n", tl / (ITERS) * 1e9);
}
