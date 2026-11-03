#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <pthread.h>
#include <sys/qos.h>
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
// integer add chain: 1 cycle latency -> measures clock
__attribute__((noinline)) long addchain(long n){ long a=0; for(long k=0;k<n;k++){ asm volatile("add %0,%0,#1\n add %0,%0,#1\n add %0,%0,#1\n add %0,%0,#1\n add %0,%0,#1\n add %0,%0,#1\n add %0,%0,#1\n add %0,%0,#1":"+r"(a)); } return a;}
__attribute__((noinline)) float32x4_t lat(long n){ float32x4_t a=vdupq_n_f32(0.1f),b=vdupq_n_f32(0.999f),c=vdupq_n_f32(1e-4f);
 for(long k=0;k<n;k++){ asm volatile("fmla %0.4s,%1.4s,%2.4s\n fmla %0.4s,%1.4s,%2.4s\n fmla %0.4s,%1.4s,%2.4s\n fmla %0.4s,%1.4s,%2.4s":"+w"(a):"w"(b),"w"(c)); } return a;}
template<int NA> __attribute__((noinline)) float tp(long n){ float32x4_t a[NA]; for(int i=0;i<NA;i++)a[i]=vdupq_n_f32(i);
 float32x4_t b=vdupq_n_f32(0.999f),c=vdupq_n_f32(1e-4f);
 for(long k=0;k<n;k++){
#pragma unroll
 for(int i=0;i<NA;i++) asm volatile("fmla %0.4s,%1.4s,%2.4s":"+w"(a[i]):"w"(b),"w"(c)); }
 float s=0; for(int i=0;i<NA;i++) s+=vgetq_lane_f32(a[i],0); return s;}
volatile float sink; volatile long lsink;
int main(){ pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
 long N=100000000; addchain(N/10);
 double t=now(); lsink=addchain(N); t=now()-t; double ghz=N*8/t/1e9; printf("int add chain: %.3f GHz (1-cycle latency)\n",ghz);
 t=now(); sink=vgetq_lane_f32(lat(N/4),0); t=now()-t; printf("FMLA latency: %.2f cycles\n", t/N*ghz*1e9);
 #define TP(K) t=now(); sink=tp<K>(N/K*4); t=now()-t; printf("FMLA indep acc=%2d: %.2f inst/cycle\n",K,(double)(N/K*4)*K/t/1e9/ghz);
 TP(8) TP(12) TP(16) TP(20) TP(24) TP(28)
}
