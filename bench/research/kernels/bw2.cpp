#include <arm_neon.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <sys/qos.h>
#include <thread>
#include <vector>
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
__attribute__((noinline)) float sum_range(const float* p, size_t n){ float32x4_t a[8]={};
 for(size_t i=0;i<n;i+=32) for(int j=0;j<8;j++) a[j]=vaddq_f32(a[j],vld1q_f32(p+i+4*j));
 for(int j=1;j<8;j++) a[0]=vaddq_f32(a[0],a[j]); return vaddvq_f32(a[0]);}
volatile float sink;
int main(){
 size_t maxb=512ull<<20; float* buf; posix_memalign((void**)&buf,16384,maxb); for(size_t i=0;i<maxb/4;i++) buf[i]=1.f;
 printf("%10s","size"); int thr[]={1,2,4,6,8,12}; for(int t:thr) printf(" %6dT",t); printf("   (GB/s, best of 5)\n");
 for(size_t sz: {64ull<<10, 1ull<<20, 4ull<<20, 12ull<<20, 32ull<<20, 96ull<<20, 512ull<<20}){
  printf("%9zuK",sz>>10);
  for(int t:thr){
   size_t chunk=(sz/4/t)&~(size_t)31; long reps=std::max<long>(1,(long)((2ull<<30)/sz)); double best=0;
   for(int trial=0;trial<5;trial++){
    std::atomic<int> ready{0}; std::atomic<bool> go{false}; std::vector<double> el(t);
    std::vector<std::thread> ths;
    for(int i=0;i<t;i++) ths.emplace_back([&,i]{ pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
      sink=sum_range(buf+i*chunk,chunk); ready++; while(!go) ; double s=now(); for(long r=0;r<reps;r++) sink=sum_range(buf+i*chunk,chunk); el[i]=now()-s;});
    while(ready<t); go=true; for(auto&th:ths) th.join();
    double mx=*std::max_element(el.begin(),el.end()); best=std::max(best,(double)chunk*4*t*reps/mx/1e9);
   }
   printf(" %7.1f",best);
  }
  printf("\n");
 }
}
