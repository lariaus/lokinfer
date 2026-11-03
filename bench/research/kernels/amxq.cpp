#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <sys/qos.h>
#include <thread>
#include <vector>
#include <algorithm>
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
double run(int M,int N,int K){ std::vector<float> A(M*K,0.1f),B(K*N,0.2f),C(M*N);
 for(int i=0;i<3;i++) cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,A.data(),K,B.data(),N,0,C.data(),N);
 std::vector<double> ts; for(int r=0;r<9;r++){double t=now(); for(int i=0;i<10;i++) cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,A.data(),K,B.data(),N,0,C.data(),N); ts.push_back((now()-t)/10);}
 std::sort(ts.begin(),ts.end()); return 2.0*M*N*K/ts[4]/1e9;}
int main(){
 int sh[][3]={{512,768,768},{1024,1024,1024},{8,768,768},{1,768,768}};
 for(auto q: {QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_BACKGROUND}){
  std::thread th([&]{ pthread_set_qos_class_self_np(q,0);
   for(auto&s:sh) printf("%s %4dx%4dx%4d: %7.1f GFLOPS\n", q==QOS_CLASS_BACKGROUND?"bg(E-core)":"UI(P-core)", s[0],s[1],s[2], run(s[0],s[1],s[2])); });
  th.join(); }
 // two independent sgemms concurrently on 2 threads (2 P-clusters -> 2 AMX units?)
 for (int nt : {1,2,3,4}) {
  std::vector<std::thread> ths; std::vector<double> g(nt); double t0=now();
  for(int i=0;i<nt;i++) ths.emplace_back([&,i]{ pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0); g[i]=run(512,768,768);});
  for(auto&t:ths) t.join(); double s=0; for(double x:g) s+=x; printf("%d concurrent sgemm 512x768x768 (VECLIB threads env as set): aggregate %.0f GFLOPS\n", nt, s);
 }
}
