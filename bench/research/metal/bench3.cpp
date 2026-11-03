#include "common.inc"
int main() {
  mach_timebase_info_data_t tb; mach_timebase_info(&tb); g_tb = (double)tb.numer / tb.denom;
  auto top = NS::AutoreleasePool::alloc()->init();
  auto dev = MTL::CreateSystemDefaultDevice(); auto q = dev->newCommandQueue(); NS::Error* err = nullptr;
  auto lib = dev->newLibrary(NS::String::string(kSrc, NS::UTF8StringEncoding), MTL::CompileOptions::alloc()->init(), &err);
  auto fn = lib->newFunction(NS::String::string("copy4", NS::UTF8StringEncoding));
  auto copy4 = dev->newComputePipelineState(fn, &err);
  auto opt = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;
  auto b1 = dev->newBuffer(64 << 20, opt), b2 = dev->newBuffer(64 << 20, opt);
  MTL::Timestamp c0, g0, c1, g1; dev->sampleTimestamps(&c0, &g0); usleep(100000); dev->sampleTimestamps(&c1, &g1);
  printf("sampleTimestamps: cpu delta %llu (mach ticks), gpu delta %llu over 100ms -> gpu ticks/s %.0f\n",
         (unsigned long long)(c1 - c0), (unsigned long long)(g1 - g0), (double)(g1 - g0) / ((c1 - c0) * g_tb / 1e9));
  MTL::CounterSet* tsSet = (MTL::CounterSet*)dev->counterSets()->object(0);
  auto sd = MTL::CounterSampleBufferDescriptor::alloc()->init();
  sd->setCounterSet(tsSet); sd->setStorageMode(MTL::StorageModeShared); sd->setSampleCount(16);
  auto sb = dev->newCounterSampleBuffer(sd, &err);
  auto c = q->commandBuffer();
  for (int k = 0; k < 8; k++) {
    auto pd = MTL::ComputePassDescriptor::computePassDescriptor();
    auto att = pd->sampleBufferAttachments()->object(0);
    att->setSampleBuffer(sb); att->setStartOfEncoderSampleIndex(2 * k); att->setEndOfEncoderSampleIndex(2 * k + 1);
    auto e = c->computeCommandEncoder(pd);
    e->setComputePipelineState(copy4); e->setBuffer(b1, 0, 0); e->setBuffer(b2, 0, 1);
    e->dispatchThreads(MTL::Size((64 << 20) / 16, 1, 1), MTL::Size(256, 1, 1)); e->endEncoding();
  }
  c->commit(); c->waitUntilCompleted();
  auto d = sb->resolveCounterRange(NS::Range(0, 16));
  const uint64_t* ts = (const uint64_t*)d->bytes();
  printf("classic stage-boundary raw deltas (8 x 64MB copy, expect ~400us each):");
  for (int k = 0; k < 8; k++) printf(" %llu", (unsigned long long)(ts[2 * k + 1] - ts[2 * k]));
  printf("\ncb GPU time %.1f us total\n", (c->GPUEndTime() - c->GPUStartTime()) * 1e6);
  top->release();
}
