#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <vector>
#include <algorithm>
static const char* src = R"(
#include <metal_stdlib>
using namespace metal;
template<typename T> kernel void k(device T* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  T a[16]; for (int j=0;j<16;j++) a[j] = T(gid*1e-6f + j);
  T b = T(0.999f), c = T(1e-4f);
  for (uint i = 0; i < iters; i++) {
    #pragma unroll
    for (int u=0;u<4;u++) {
    #pragma unroll
    for (int j=0;j<16;j++) a[j] = fma(a[j], b, c);
    }
  }
  T s = 0; for (int j=0;j<16;j++) s += a[j]; out[gid] = s;
}
template [[host_name("f32")]] kernel void k<float>(device float*, constant uint&, uint);
template [[host_name("f16")]] kernel void k<half>(device half*, constant uint&, uint);
template [[host_name("f16x2")]] kernel void k<half2>(device half2*, constant uint&, uint);
)";
int main(){ @autoreleasepool {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[d newCommandQueue]; NSError* e=nil;
  MTLCompileOptions* o=[MTLCompileOptions new]; o.mathMode=MTLMathModeSafe;
  id<MTLLibrary> l=[d newLibraryWithSource:@(src) options:o error:&e]; if(!l){printf("%s\n",e.description.UTF8String);return 1;}
  for (const char* n : {"f32","f16","f16x2"}) {
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@(n)] error:&e];
    uint iters=1024; size_t th=1<<20; id<MTLBuffer> b=[d newBufferWithLength:th*8 options:0];
    std::vector<double> ts;
    for(int r=0;r<12;r++){ id<MTLCommandBuffer> cb=[q commandBuffer]; auto en=[cb computeCommandEncoder]; [en setComputePipelineState:p]; [en setBuffer:b offset:0 atIndex:0]; [en setBytes:&iters length:4 atIndex:1];
      [en dispatchThreads:MTLSizeMake(th,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; if(r>1) ts.push_back(cb.GPUEndTime-cb.GPUStartTime);}
    std::sort(ts.begin(),ts.end()); double w = (n[3]=='x')?2:1;
    printf("%-6s FMA: %.2f TFLOPS\n", n, 2.0*64*iters*th*w/ts[ts.size()/2]/1e12);
  }
}}
