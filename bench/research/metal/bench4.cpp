#include "common.inc"
static const char* kInc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void inc(device float* a [[buffer(0)]], device const uint* n [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i < n[0]) a[i] += 1.0f; }
)MSL";
int main() {
  mach_timebase_info_data_t tb; mach_timebase_info(&tb); g_tb = (double)tb.numer / tb.denom;
  auto top = NS::AutoreleasePool::alloc()->init();
  auto dev = MTL::CreateSystemDefaultDevice(); NS::Error* err = nullptr;
  auto lib = dev->newLibrary(NS::String::string(kInc, NS::UTF8StringEncoding), MTL::CompileOptions::alloc()->init(), &err);
  auto pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("inc", NS::UTF8StringEncoding)), &err);
  auto opt = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;
  auto X = dev->newBuffer(1024, opt), Y = dev->newBuffer(1024, opt), Nb = dev->newBuffer(16, opt);
  memset(X->contents(), 0, 1024); memset(Y->contents(), 0, 1024); ((uint32_t*)Nb->contents())[0] = 64;
  auto rs = dev->newResidencySet(MTL::ResidencySetDescriptor::alloc()->init(), &err);
  rs->addAllocation(X); rs->addAllocation(Y); rs->addAllocation(Nb); rs->commit();
  auto q4 = dev->newMTL4CommandQueue(); q4->addResidencySet(rs);
  auto atd = MTL4::ArgumentTableDescriptor::alloc()->init(); atd->setMaxBufferBindCount(2);
  auto at = dev->newArgumentTable(atd, &err);
  auto al = dev->newCommandAllocator(); auto cb = dev->newCommandBuffer();
  auto ev = dev->newSharedEvent(); uint64_t v = 0;
  cb->beginCommandBuffer(al);
  auto e = cb->computeCommandEncoder(); e->setArgumentTable(at); e->setComputePipelineState(pso);
  at->setAddress(X->gpuAddress(), 0); at->setAddress(Nb->gpuAddress(), 1);
  e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(64, 1, 1));
  at->setAddress(Y->gpuAddress(), 0);   // change binding after first dispatch
  e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(64, 1, 1));
  e->endEncoding(); cb->endCommandBuffer();
  const MTL4::CommandBuffer* arr[1] = {cb};
  q4->commit(arr, 1); q4->signalEvent(ev, ++v); ev->waitUntilSignaledValue(v, 10000);
  printf("argtable snapshot test: X[0]=%.0f Y[0]=%.0f  (1,1 => bindings snapshotted per dispatch)\n", ((float*)X->contents())[0], ((float*)Y->contents())[0]);
  // re-commit same encoded command buffer without re-encoding
  std::vector<double> lat;
  for (int r = 0; r < 200; r++) {
    uint64_t t0 = now_ticks();
    q4->commit(arr, 1); q4->signalEvent(ev, ++v);
    uint64_t t1 = now_ticks();
    ev->waitUntilSignaledValue(v, 10000);
    lat.push_back(ticks_to_us(t1 - t0));
  }
  printf("after 200 re-commits of the same MTL4 cb: X[0]=%.0f Y[0]=%.0f (201,201 => re-commit replays)\n", ((float*)X->contents())[0], ((float*)Y->contents())[0]);
  report("MTL4 re-commit of pre-encoded cb: CPU commit+signal cost", lat);
  // Does changing the referenced buffer contents (params) between re-commits take effect? (it must: data, not commands)
  ((uint32_t*)Nb->contents())[0] = 32;
  q4->commit(arr, 1); q4->signalEvent(ev, ++v); ev->waitUntilSignaledValue(v, 10000);
  printf("after n=32 recommit: X[0]=%.0f X[40]=%.0f\n", ((float*)X->contents())[0], ((float*)X->contents())[40]);
  top->release();
}
