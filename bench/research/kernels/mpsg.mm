#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <chrono>
#include <cstdio>
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(){ @autoreleasepool {
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
 int shapes[][3]={{512,768,768},{64,768,768},{512,3072,768},{512,512,1536},{2048,2048,2048}};
 for (int f16=0; f16<2; f16++) for (auto& s: shapes){ int M=s[0],N=s[1],K=s[2]; MPSDataType dt=f16?MPSDataTypeFloat16:MPSDataTypeFloat32; int es=f16?2:4;
  id<MTLBuffer> A=[dev newBufferWithLength:M*K*es options:0], B=[dev newBufferWithLength:K*N*es options:0], C=[dev newBufferWithLength:M*N*es options:0];
  MPSGraph* g=[MPSGraph new]; auto ta=[g placeholderWithShape:@[@(M),@(K)] dataType:dt name:nil]; auto tb=[g placeholderWithShape:@[@(K),@(N)] dataType:dt name:nil];
  auto tc=[g matrixMultiplicationWithPrimaryTensor:ta secondaryTensor:tb name:nil];
  auto da=[[MPSGraphTensorData alloc] initWithMTLBuffer:A shape:@[@(M),@(K)] dataType:dt]; auto db=[[MPSGraphTensorData alloc] initWithMTLBuffer:B shape:@[@(K),@(N)] dataType:dt];
  auto dc=[[MPSGraphTensorData alloc] initWithMTLBuffer:C shape:@[@(M),@(N)] dataType:dt];
  auto ex=[g compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:dev] feeds:@{ta:[[MPSGraphShapedType alloc] initWithShape:@[@(M),@(K)] dataType:dt], tb:[[MPSGraphShapedType alloc] initWithShape:@[@(K),@(N)] dataType:dt]} targetTensors:@[tc] targetOperations:nil compilationDescriptor:nil];
  for(int w=0;w<3;w++) [ex runWithMTLCommandQueue:q inputsArray:@[da,db] resultsArray:@[dc] executionDescriptor:nil];
  int R=20; double t0=now();
  for(int r=0;r<R;r++){ MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; [ex encodeToCommandBuffer:cb inputsArray:@[da,db] resultsArray:@[dc] executionDescriptor:nil]; [cb commit]; if(r==R-1)[cb waitUntilCompleted]; }
  double t=(now()-t0)/R; double one; {double a=now(); [ex runWithMTLCommandQueue:q inputsArray:@[da,db] resultsArray:@[dc] executionDescriptor:nil]; one=now()-a;}
  printf("MPSGraph %s %4dx%4dx%4d: pipelined %8.1f us/iter (%5.2f TF), single sync run %8.1f us\n", f16?"f16":"f32",M,N,K,t*1e6,2.0*M*N*K/t/1e12, one*1e6);
 }
}}
