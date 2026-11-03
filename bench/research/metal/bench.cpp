// lokinfer Metal runtime micro-benchmarks (metal-cpp, classic Metal + Metal 4).
// Build: see build.sh. CLI only; no GUI.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

static double g_tb = 1.0;
static inline uint64_t now_ticks() { return mach_absolute_time(); }
static inline double ticks_to_us(uint64_t t) { return (double)t * g_tb / 1000.0; }

struct Stats { double med, p10, p90, mn; };
static Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1)))]; };
  return {at(0.5), at(0.1), at(0.9), v.front()};
}
static void report(const char* name, const std::vector<double>& us, const char* unit = "us", double div = 1.0) {
  std::vector<double> d(us);
  for (auto& x : d) x /= div;
  Stats s = stats(d);
  printf("%-62s med %9.3f %s  p10 %9.3f  p90 %9.3f  min %9.3f\n", name, s.med, unit, s.p10, s.p90, s.mn);
}

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void tiny(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                 constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) a[i] = a[i] * 0.5f + b[i];
}
// Same as tiny but n read from a device buffer (MTL4 has no setBytes).
kernel void tiny_dev(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                     device const uint* n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n[0]) a[i] = a[i] * 0.5f + b[i];
}
// Data-dependent size producer: writes indirect threadgroup counts + n.
kernel void produce(device const uint* in [[buffer(0)]], device uint* args [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
  if (i == 0) { uint n = in[0]; args[0] = (n + 255) / 256; args[1] = 1; args[2] = 1; args[3] = n; }
}
kernel void consume(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                    device const uint* args [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < args[3]) a[i] = a[i] * 0.5f + b[i];
}
kernel void copy4(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
  dst[i] = src[i] * 1.0001f;
}
kernel void read4(device const float4* src [[buffer(0)]], device float* dst [[buffer(1)]],
                  uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
  // Strided-sum read; one write per 64 threads to keep writes negligible.
  float4 v = src[i];
  float s = v.x + v.y + v.z + v.w;
  s = simd_sum(s);
  if ((i & 63) == 0) dst[i >> 6] = s;
}
)MSL";

#define CHECK(x, msg) do { if (!(x)) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); } } while (0)

struct Ctx {
  MTL::Device* dev;
  MTL::CommandQueue* q;
  MTL::Library* lib;
  MTL::ComputePipelineState *tiny, *tiny_dev, *produce, *consume, *copy4, *read4, *tiny_icb;
};

static MTL::ComputePipelineState* pso(Ctx& c, const char* name, bool icb = false) {
  NS::Error* err = nullptr;
  auto fn = c.lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  CHECK(fn, name);
  auto d = MTL::ComputePipelineDescriptor::alloc()->init();
  d->setComputeFunction(fn);
  d->setSupportIndirectCommandBuffers(icb);
  auto p = c.dev->newComputePipelineState(d, MTL::PipelineOptionNone, nullptr, &err);
  CHECK(p, name);
  d->release();
  fn->release();
  return p;
}

static void wait_spin(MTL::CommandBuffer* cb) {
  while (cb->status() < MTL::CommandBufferStatusCompleted) {}
}

int main(int argc, char** argv) {
  mach_timebase_info_data_t tb; mach_timebase_info(&tb); g_tb = (double)tb.numer / tb.denom;
  NS::AutoreleasePool* top = NS::AutoreleasePool::alloc()->init();
  Ctx c{};
  c.dev = MTL::CreateSystemDefaultDevice();
  CHECK(c.dev, "device");
  printf("Device: %s\n", c.dev->name()->utf8String());
  printf("supportsFamily: Apple7 %d Apple8 %d Apple9 %d Apple10 %d Apple11 %d Metal3 %d Metal4 %d\n",
         c.dev->supportsFamily(MTL::GPUFamilyApple7), c.dev->supportsFamily(MTL::GPUFamilyApple8),
         c.dev->supportsFamily(MTL::GPUFamilyApple9), c.dev->supportsFamily(MTL::GPUFamilyApple10),
         c.dev->supportsFamily(MTL::GPUFamilyApple11), c.dev->supportsFamily(MTL::GPUFamilyMetal3),
         c.dev->supportsFamily(MTL::GPUFamilyMetal4));
  printf("counterSampling: stage %d draw %d dispatch %d tileDispatch %d blit %d\n",
         c.dev->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary),
         c.dev->supportsCounterSampling(MTL::CounterSamplingPointAtDrawBoundary),
         c.dev->supportsCounterSampling(MTL::CounterSamplingPointAtDispatchBoundary),
         c.dev->supportsCounterSampling(MTL::CounterSamplingPointAtTileDispatchBoundary),
         c.dev->supportsCounterSampling(MTL::CounterSamplingPointAtBlitBoundary));
  printf("timestampFrequency %llu Hz, maxBufferLength %.1f GB, recommendedMaxWorkingSet %.1f GB, hasUnifiedMemory %d\n",
         (unsigned long long)c.dev->queryTimestampFrequency(), c.dev->maxBufferLength() / 1e9,
         c.dev->recommendedMaxWorkingSetSize() / 1e9, c.dev->hasUnifiedMemory());
  printf("timestamp entry size %lu\n", (unsigned long)c.dev->sizeOfCounterHeapEntry(MTL4::CounterHeapTypeTimestamp));

  c.q = c.dev->newCommandQueue();
  NS::Error* err = nullptr;
  auto opts = MTL::CompileOptions::alloc()->init();
  c.lib = c.dev->newLibrary(NS::String::string(kSrc, NS::UTF8StringEncoding), opts, &err);
  if (!c.lib) { fprintf(stderr, "%s\n", err->localizedDescription()->utf8String()); return 1; }
  c.tiny = pso(c, "tiny"); c.tiny_dev = pso(c, "tiny_dev"); c.produce = pso(c, "produce");
  c.consume = pso(c, "consume"); c.copy4 = pso(c, "copy4"); c.read4 = pso(c, "read4");
  c.tiny_icb = pso(c, "tiny_dev", true);

  const int NMAX = 4096;
  const uint32_t TG = 256;  // one threadgroup per tiny kernel
  auto A = c.dev->newBuffer(NMAX * TG * 4, MTL::ResourceStorageModeShared);
  auto B = c.dev->newBuffer(NMAX * TG * 4, MTL::ResourceStorageModeShared);
  auto Au = c.dev->newBuffer(NMAX * TG * 4, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
  auto Bu = c.dev->newBuffer(NMAX * TG * 4, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
  auto P = c.dev->newBuffer(NMAX * 16, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
  memset(A->contents(), 0, A->length()); memset(B->contents(), 0, B->length());
  memset(Au->contents(), 0, Au->length()); memset(Bu->contents(), 0, Bu->length());
  for (int i = 0; i < NMAX; i++) ((uint32_t*)P->contents())[i * 4] = TG;
  uint32_t n = TG;

  // Residency set holding everything (used by Metal 4 queue and classic queue).
  auto rsd = MTL::ResidencySetDescriptor::alloc()->init();
  rsd->setInitialCapacity(64);
  auto rs = c.dev->newResidencySet(rsd, &err);
  CHECK(rs, "residency set");
  rs->addAllocation(A); rs->addAllocation(B); rs->addAllocation(Au); rs->addAllocation(Bu); rs->addAllocation(P);

  const int WARM = 50;

  // ---------------------------------------------------------------- 1. round trip
  printf("\n== 1. Submit -> completion round trip (classic MTLCommandQueue) ==\n");
  for (int mode = 0; mode < 4; mode++) {
    std::vector<double> v;
    for (int r = 0; r < 2000 + WARM; r++) {
      auto pool = NS::AutoreleasePool::alloc()->init();
      uint64_t t0 = now_ticks();
      auto cb = (mode >= 2) ? c.q->commandBufferWithUnretainedReferences() : c.q->commandBuffer();
      if (mode & 1) {
        auto e = cb->computeCommandEncoder();
        e->setComputePipelineState(c.tiny); e->setBuffer(A, 0, 0); e->setBuffer(B, 0, 1); e->setBytes(&n, 4, 2);
        e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1)); e->endEncoding();
      }
      cb->commit();
      cb->waitUntilCompleted();
      uint64_t t1 = now_ticks();
      if (r >= WARM) v.push_back(ticks_to_us(t1 - t0));
      pool->release();
    }
    const char* names[] = {"empty cb, commit+waitUntilCompleted", "1 tiny dispatch, commit+waitUntilCompleted",
                           "empty cb (unretained refs), commit+wait", "1 tiny dispatch (unretained), commit+wait"};
    report(names[mode], v);
  }
  {
    std::vector<double> v;
    for (int r = 0; r < 2000 + WARM; r++) {
      auto pool = NS::AutoreleasePool::alloc()->init();
      uint64_t t0 = now_ticks();
      auto cb = c.q->commandBufferWithUnretainedReferences();
      auto e = cb->computeCommandEncoder();
      e->setComputePipelineState(c.tiny); e->setBuffer(A, 0, 0); e->setBuffer(B, 0, 1); e->setBytes(&n, 4, 2);
      e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1)); e->endEncoding();
      cb->commit();
      wait_spin(cb);
      uint64_t t1 = now_ticks();
      if (r >= WARM) v.push_back(ticks_to_us(t1 - t0));
      pool->release();
    }
    report("1 tiny dispatch, commit + spin on cb.status", v);
  }
  auto ev = c.dev->newSharedEvent();
  uint64_t evv = 0;
  {
    std::vector<double> vb, vs;
    for (int spin = 0; spin < 2; spin++)
      for (int r = 0; r < 2000 + WARM; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        uint64_t t0 = now_ticks();
        auto cb = c.q->commandBufferWithUnretainedReferences();
        auto e = cb->computeCommandEncoder();
        e->setComputePipelineState(c.tiny); e->setBuffer(A, 0, 0); e->setBuffer(B, 0, 1); e->setBytes(&n, 4, 2);
        e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1)); e->endEncoding();
        cb->encodeSignalEvent(ev, ++evv);
        cb->commit();
        if (spin) { while (ev->signaledValue() < evv) {} }
        else ev->waitUntilSignaledValue(evv, 10000);
        uint64_t t1 = now_ticks();
        if (r >= WARM) (spin ? vs : vb).push_back(ticks_to_us(t1 - t0));
        pool->release();
      }
    report("1 tiny dispatch, MTLSharedEvent waitUntilSignaledValue", vb);
    report("1 tiny dispatch, MTLSharedEvent spin on signaledValue", vs);
  }

  // ---------------------------------------------------------------- Metal 4 setup
  bool has4 = c.dev->supportsFamily(MTL::GPUFamilyMetal4);
  MTL4::CommandQueue* q4 = nullptr; MTL4::CommandAllocator* alloc4 = nullptr; MTL4::CommandBuffer* cb4 = nullptr;
  MTL4::ArgumentTable* at = nullptr;
  if (has4) {
    q4 = c.dev->newMTL4CommandQueue();
    alloc4 = c.dev->newCommandAllocator();
    cb4 = c.dev->newCommandBuffer();
    auto atd = MTL4::ArgumentTableDescriptor::alloc()->init();
    atd->setMaxBufferBindCount(4);
    at = c.dev->newArgumentTable(atd, &err);
    CHECK(q4 && alloc4 && cb4 && at, "metal4 objects");
    rs->commit();
    rs->requestResidency();
    q4->addResidencySet(rs);
  } else {
    rs->commit();
  }
  c.q->addResidencySet(rs);

  if (has4) {
    printf("\n== 1b. Submit -> completion round trip (MTL4CommandQueue) ==\n");
    for (int mode = 0; mode < 4; mode++) {
      std::vector<double> v;
      for (int r = 0; r < 2000 + WARM; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        uint64_t t0 = now_ticks();
        alloc4->reset();
        cb4->beginCommandBuffer(alloc4);
        if (mode & 1) {
          auto e = cb4->computeCommandEncoder();
          e->setComputePipelineState(c.tiny_dev);
          e->setArgumentTable(at);
          at->setAddress(A->gpuAddress(), 0); at->setAddress(B->gpuAddress(), 1); at->setAddress(P->gpuAddress(), 2);
          e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
          e->endEncoding();
        }
        cb4->endCommandBuffer();
        const MTL4::CommandBuffer* cbs[1] = {cb4};
        q4->commit(cbs, 1);
        q4->signalEvent(ev, ++evv);
        if (mode & 2) { while (ev->signaledValue() < evv) {} }
        else ev->waitUntilSignaledValue(evv, 10000);
        uint64_t t1 = now_ticks();
        if (r >= WARM) v.push_back(ticks_to_us(t1 - t0));
        pool->release();
      }
      const char* names[] = {"MTL4 empty cb, signalEvent + waitUntilSignaledValue", "MTL4 1 tiny dispatch, waitUntilSignaledValue",
                             "MTL4 empty cb, spin on signaledValue", "MTL4 1 tiny dispatch, spin on signaledValue"};
      report(names[mode], v);
    }
  }

  // ---------------------------------------------------------------- 2. encode cost + GPU time for N dispatches
  printf("\n== 2. N tiny dispatches (256 threads each): CPU encode cost per dispatch, and GPU time ==\n");
  printf("   (cpu = cb creation .. commit returns; gpu = GPUEndTime-GPUStartTime; wall = cpu start .. completion)\n");
  // Variants (classic):
  //  0 serial encoder, tracked buffers, dependent chain (same A)
  //  1 serial encoder, untracked buffers, dependent chain        -> Metal still serializes serial encoders
  //  2 concurrent encoder, untracked, independent (no barriers)
  //  3 concurrent encoder, untracked, dependent chain + memoryBarrier(buffers) between each
  //  4 serial encoder, untracked, one encoder per dispatch (encoder churn)
  const char* vnames[] = {"serial enc, tracked, chain", "serial enc, untracked, chain", "concurrent enc, untracked, independent",
                          "concurrent enc, untracked, chain+memoryBarrier", "1 encoder per dispatch, untracked"};
  int Ns[] = {10, 100, 1000};
  for (int N : Ns) {
    for (int var = 0; var < 5; var++) {
      std::vector<double> cpu, gpu, wall;
      int reps = N >= 1000 ? 200 : 500;
      for (int r = 0; r < reps + 20; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        MTL::Buffer* a = var == 0 ? A : Au; MTL::Buffer* b = var == 0 ? B : Bu;
        uint64_t t0 = now_ticks();
        auto cb = c.q->commandBufferWithUnretainedReferences();
        MTL::ComputeCommandEncoder* e = nullptr;
        if (var != 4) e = (var >= 2) ? cb->computeCommandEncoder(MTL::DispatchTypeConcurrent) : cb->computeCommandEncoder();
        for (int k = 0; k < N; k++) {
          if (var == 4) e = cb->computeCommandEncoder();
          size_t off = (var == 2) ? (size_t)k * TG * 4 : 0;
          if (var == 3 && k) e->memoryBarrier(MTL::BarrierScopeBuffers);
          e->setComputePipelineState(c.tiny);
          e->setBuffer(a, off, 0); e->setBuffer(b, off, 1); e->setBytes(&n, 4, 2);
          e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
          if (var == 4) e->endEncoding();
        }
        if (var != 4) e->endEncoding();
        cb->commit();
        uint64_t t1 = now_ticks();
        cb->waitUntilCompleted();
        uint64_t t2 = now_ticks();
        if (r >= 20) {
          cpu.push_back(ticks_to_us(t1 - t0));
          gpu.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e6);
          wall.push_back(ticks_to_us(t2 - t0));
        }
        pool->release();
      }
      char nm[160];
      snprintf(nm, sizeof nm, "N=%4d %-44s cpu/disp", N, vnames[var]); report(nm, cpu, "us", N);
      snprintf(nm, sizeof nm, "N=%4d %-44s gpu/disp", N, vnames[var]); report(nm, gpu, "us", N);
      snprintf(nm, sizeof nm, "N=%4d %-44s wall tot", N, vnames[var]); report(nm, wall, "us");
    }
    if (has4) {
      // Metal 4 variants: 0 independent no barriers, 1 chain + barrierAfterEncoderStages(dispatch->dispatch)
      const char* v4names[] = {"MTL4 independent, no barriers", "MTL4 chain + intra-encoder barrier"};
      for (int var = 0; var < 2; var++) {
        std::vector<double> cpu, gpu, wall;
        int reps = N >= 1000 ? 200 : 500;
        for (int r = 0; r < reps + 20; r++) {
          auto pool = NS::AutoreleasePool::alloc()->init();
          std::atomic<int> done{0};
          double gs = 0, ge = 0;
          uint64_t t0 = now_ticks();
          alloc4->reset();
          cb4->beginCommandBuffer(alloc4);
          auto e = cb4->computeCommandEncoder();
          e->setArgumentTable(at);
          uint64_t aa = Au->gpuAddress(), ba = Bu->gpuAddress(), pa = P->gpuAddress();
          for (int k = 0; k < N; k++) {
            size_t off = (var == 0) ? (size_t)k * TG * 4 : 0;
            if (var == 1 && k) e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
            e->setComputePipelineState(c.tiny_dev);
            at->setAddress(aa + off, 0); at->setAddress(ba + off, 1); at->setAddress(pa + k * 16, 2);
            e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
          }
          e->endEncoding();
          cb4->endCommandBuffer();
          auto co = MTL4::CommitOptions::alloc()->init();
          co->addFeedbackHandler([&](MTL4::CommitFeedback* fb) { gs = fb->GPUStartTime(); ge = fb->GPUEndTime(); done.store(1); });
          const MTL4::CommandBuffer* cbs[1] = {cb4};
          q4->commit(cbs, 1, co);
          q4->signalEvent(ev, ++evv);
          uint64_t t1 = now_ticks();
          ev->waitUntilSignaledValue(evv, 10000);
          uint64_t t2 = now_ticks();
          while (!done.load()) {}
          co->release();
          if (r >= 20) {
            cpu.push_back(ticks_to_us(t1 - t0));
            gpu.push_back((ge - gs) * 1e6);
            wall.push_back(ticks_to_us(t2 - t0));
          }
          pool->release();
        }
        char nm[160];
        snprintf(nm, sizeof nm, "N=%4d %-44s cpu/disp", N, v4names[var]); report(nm, cpu, "us", N);
        snprintf(nm, sizeof nm, "N=%4d %-44s gpu/disp", N, v4names[var]); report(nm, gpu, "us", N);
        snprintf(nm, sizeof nm, "N=%4d %-44s wall tot", N, v4names[var]); report(nm, wall, "us");
      }
    }
  }

  // ---------------------------------------------------------------- 3. ICB record / replay
  printf("\n== 3. Indirect command buffer: record once, replay ==\n");
  for (int N : Ns) {
    for (int barrier = 0; barrier < 2; barrier++) {
      auto icbd = MTL::IndirectCommandBufferDescriptor::alloc()->init();
      icbd->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
      icbd->setInheritBuffers(false);
      icbd->setInheritPipelineState(false);
      icbd->setMaxKernelBufferBindCount(3);
      auto icb = c.dev->newIndirectCommandBuffer(icbd, N, MTL::ResourceStorageModeShared);
      CHECK(icb, "icb");
      uint64_t r0 = now_ticks();
      for (int k = 0; k < N; k++) {
        auto cmd = icb->indirectComputeCommand(k);
        size_t off = barrier ? 0 : (size_t)k * TG * 4;
        cmd->setComputePipelineState(c.tiny_icb);
        cmd->setKernelBuffer(Au, off, 0); cmd->setKernelBuffer(Bu, off, 1); cmd->setKernelBuffer(P, k * 16, 2);
        cmd->concurrentDispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
        if (barrier) cmd->setBarrier();
      }
      uint64_t r1 = now_ticks();
      printf("N=%4d %s: ICB record (CPU) total %.1f us = %.3f us/cmd\n", N, barrier ? "chain(setBarrier)" : "independent",
             ticks_to_us(r1 - r0), ticks_to_us(r1 - r0) / N);
      // single-binding update cost
      {
        std::vector<double> v;
        for (int r = 0; r < 2000; r++) {
          uint64_t t0 = now_ticks();
          icb->indirectComputeCommand(r % N)->setKernelBuffer(Au, 0, 0);
          v.push_back(ticks_to_us(now_ticks() - t0));
        }
        report("   ICB per-command single setKernelBuffer update (CPU)", v);
      }
      // restore
      for (int k = 0; k < N; k++) {
        size_t off = barrier ? 0 : (size_t)k * TG * 4;
        icb->indirectComputeCommand(k)->setKernelBuffer(Au, off, 0);
      }
      rs->addAllocation(icb); rs->commit();
      std::vector<double> cpu, gpu, wall;
      for (int r = 0; r < 300 + 20; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        uint64_t t0 = now_ticks();
        auto cb = c.q->commandBufferWithUnretainedReferences();
        auto e = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
        e->executeCommandsInBuffer(icb, NS::Range(0, N));
        e->endEncoding();
        cb->commit();
        uint64_t t1 = now_ticks();
        cb->waitUntilCompleted();
        uint64_t t2 = now_ticks();
        if (r >= 20) { cpu.push_back(ticks_to_us(t1 - t0)); gpu.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e6); wall.push_back(ticks_to_us(t2 - t0)); }
        pool->release();
      }
      char nm[160];
      snprintf(nm, sizeof nm, "N=%4d classic ICB replay %-18s cpu total", N, barrier ? "chain" : "indep"); report(nm, cpu);
      snprintf(nm, sizeof nm, "N=%4d classic ICB replay %-18s gpu/disp", N, barrier ? "chain" : "indep"); report(nm, gpu, "us", N);
      snprintf(nm, sizeof nm, "N=%4d classic ICB replay %-18s wall tot", N, barrier ? "chain" : "indep"); report(nm, wall);
      if (has4) {
        std::vector<double> cpu4, gpu4, wall4;
        for (int r = 0; r < 300 + 20; r++) {
          auto pool = NS::AutoreleasePool::alloc()->init();
          std::atomic<int> done{0}; double gs = 0, ge = 0;
          uint64_t t0 = now_ticks();
          alloc4->reset();
          cb4->beginCommandBuffer(alloc4);
          auto e = cb4->computeCommandEncoder();
          e->executeCommandsInBuffer(icb, NS::Range(0, N));
          e->endEncoding();
          cb4->endCommandBuffer();
          auto co = MTL4::CommitOptions::alloc()->init();
          co->addFeedbackHandler([&](MTL4::CommitFeedback* fb) { gs = fb->GPUStartTime(); ge = fb->GPUEndTime(); done.store(1); });
          const MTL4::CommandBuffer* cbs[1] = {cb4};
          q4->commit(cbs, 1, co);
          q4->signalEvent(ev, ++evv);
          uint64_t t1 = now_ticks();
          ev->waitUntilSignaledValue(evv, 10000);
          uint64_t t2 = now_ticks();
          while (!done.load()) {}
          co->release();
          if (r >= 20) { cpu4.push_back(ticks_to_us(t1 - t0)); gpu4.push_back((ge - gs) * 1e6); wall4.push_back(ticks_to_us(t2 - t0)); }
          pool->release();
        }
        snprintf(nm, sizeof nm, "N=%4d MTL4 ICB replay %-21s cpu total", N, barrier ? "chain" : "indep"); report(nm, cpu4);
        snprintf(nm, sizeof nm, "N=%4d MTL4 ICB replay %-21s gpu/disp", N, barrier ? "chain" : "indep"); report(nm, gpu4, "us", N);
        snprintf(nm, sizeof nm, "N=%4d MTL4 ICB replay %-21s wall tot", N, barrier ? "chain" : "indep"); report(nm, wall4);
      }
      rs->removeAllocation(icb); rs->commit();
      icb->release(); icbd->release();
    }
  }

  // ---------------------------------------------------------------- 4. data-dependent size
  printf("\n== 4. Data-dependent size: GPU-driven indirect dispatch vs CPU readback vs upper-bound grid ==\n");
  {
    auto In = c.dev->newBuffer(16, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    auto Args = c.dev->newBuffer(16, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    ((uint32_t*)In->contents())[0] = 1000;  // ~4 threadgroups
    for (int mode = 0; mode < 3; mode++) {
      std::vector<double> v;
      for (int r = 0; r < 1000 + WARM; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        uint64_t t0 = now_ticks();
        auto cb = c.q->commandBufferWithUnretainedReferences();
        auto e = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
        e->setComputePipelineState(c.produce); e->setBuffer(In, 0, 0); e->setBuffer(Args, 0, 1);
        e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
        if (mode == 0) {  // GPU-driven
          e->memoryBarrier(MTL::BarrierScopeBuffers);
          e->setComputePipelineState(c.consume); e->setBuffer(Au, 0, 0); e->setBuffer(Bu, 0, 1); e->setBuffer(Args, 0, 2);
          e->dispatchThreadgroups(Args, 0, MTL::Size(TG, 1, 1));
          e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        } else if (mode == 1) {  // readback
          e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
          uint32_t nn = ((uint32_t*)Args->contents())[3];
          auto cb2 = c.q->commandBufferWithUnretainedReferences();
          auto e2 = cb2->computeCommandEncoder(MTL::DispatchTypeConcurrent);
          e2->setComputePipelineState(c.consume); e2->setBuffer(Au, 0, 0); e2->setBuffer(Bu, 0, 1); e2->setBuffer(Args, 0, 2);
          e2->dispatchThreadgroups(MTL::Size((nn + TG - 1) / TG, 1, 1), MTL::Size(TG, 1, 1));
          e2->endEncoding(); cb2->commit(); cb2->waitUntilCompleted();
        } else {  // upper bound grid (4096 elems), kernel early-exits
          e->memoryBarrier(MTL::BarrierScopeBuffers);
          e->setComputePipelineState(c.consume); e->setBuffer(Au, 0, 0); e->setBuffer(Bu, 0, 1); e->setBuffer(Args, 0, 2);
          e->dispatchThreadgroups(MTL::Size(16, 1, 1), MTL::Size(TG, 1, 1));
          e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        }
        uint64_t t1 = now_ticks();
        if (r >= WARM) v.push_back(ticks_to_us(t1 - t0));
        pool->release();
      }
      const char* names[] = {"produce -> indirect consume, 1 cb (GPU-driven)", "produce, commit+wait, read n, consume cb (readback)",
                             "produce -> upper-bound grid consume, 1 cb"};
      report(names[mode], v);
    }
    // Pure added cost of one mid-graph readback inside a longer graph: 200 dispatches, split in 2 cbs w/ wait vs 1 cb.
    for (int split = 0; split < 2; split++) {
      std::vector<double> v;
      for (int r = 0; r < 300 + 20; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        uint64_t t0 = now_ticks();
        for (int part = 0; part < 2; part++) {
          auto cb = c.q->commandBufferWithUnretainedReferences();
          auto e = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
          for (int k = 0; k < 100; k++) {
            if (k) e->memoryBarrier(MTL::BarrierScopeBuffers);
            e->setComputePipelineState(c.tiny); e->setBuffer(Au, 0, 0); e->setBuffer(Bu, 0, 1); e->setBytes(&n, 4, 2);
            e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
          }
          e->endEncoding(); cb->commit();
          if (split || part == 1) cb->waitUntilCompleted();
        }
        uint64_t t1 = now_ticks();
        if (r >= 20) v.push_back(ticks_to_us(t1 - t0));
        pool->release();
      }
      report(split ? "200-dispatch chain, 2 cbs with CPU wait in the middle" : "200-dispatch chain, 2 cbs pipelined (no mid wait)", v);
    }
  }

  // ---------------------------------------------------------------- 5. CPU->GPU event latency
  printf("\n== 5. CPU signal -> GPU wake (cb waits on MTLSharedEvent set by CPU) ==\n");
  {
    auto ev2 = c.dev->newSharedEvent();
    uint64_t v2 = 0;
    std::vector<double> v;
    for (int r = 0; r < 1000 + WARM; r++) {
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto cb = c.q->commandBufferWithUnretainedReferences();
      cb->encodeWait(ev2, ++v2);
      auto e = cb->computeCommandEncoder();
      e->setComputePipelineState(c.tiny); e->setBuffer(Au, 0, 0); e->setBuffer(Bu, 0, 1); e->setBytes(&n, 4, 2);
      e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1)); e->endEncoding();
      cb->encodeSignalEvent(ev, ++evv);
      cb->commit();
      usleep(200);  // let the GPU reach the wait
      uint64_t t0 = now_ticks();
      ev2->setSignaledValue(v2);
      while (ev->signaledValue() < evv) {}
      uint64_t t1 = now_ticks();
      if (r >= WARM) v.push_back(ticks_to_us(t1 - t0));
      pool->release();
    }
    report("CPU setSignaledValue -> GPU runs 1 tiny dispatch -> CPU sees signal (spin)", v);
  }

  // ---------------------------------------------------------------- 6. timestamps
  if (has4) {
    printf("\n== 6. MTL4 counter heap timestamps around each dispatch ==\n");
    const int N = 100;
    auto chd = MTL4::CounterHeapDescriptor::alloc()->init();
    chd->setType(MTL4::CounterHeapTypeTimestamp);
    chd->setCount(2 * N + 2);
    auto heap = c.dev->newCounterHeap(chd, &err);
    if (!heap) printf("counter heap creation failed: %s\n", err ? err->localizedDescription()->utf8String() : "?");
    double freq = (double)c.dev->queryTimestampFrequency();
    for (int gran = 0; heap && gran < 2; gran++) {
      std::vector<double> per, tot, wall, cbgpu;
      for (int r = 0; r < 100 + 10; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        std::atomic<int> done{0}; double gs = 0, ge = 0;
        uint64_t t0 = now_ticks();
        alloc4->reset();
        cb4->beginCommandBuffer(alloc4);
        auto e = cb4->computeCommandEncoder();
        e->setArgumentTable(at);
        for (int k = 0; k < N; k++) {
          if (k) e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
          e->writeTimestamp((MTL4::TimestampGranularity)gran, heap, 2 * k);
          e->setComputePipelineState(c.tiny_dev);
          at->setAddress(Au->gpuAddress(), 0); at->setAddress(Bu->gpuAddress(), 1); at->setAddress(P->gpuAddress(), 2);
          e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
          e->writeTimestamp((MTL4::TimestampGranularity)gran, heap, 2 * k + 1);
        }
        e->endEncoding();
        cb4->endCommandBuffer();
        auto co = MTL4::CommitOptions::alloc()->init();
        co->addFeedbackHandler([&](MTL4::CommitFeedback* fb) { gs = fb->GPUStartTime(); ge = fb->GPUEndTime(); done.store(1); });
        const MTL4::CommandBuffer* cbs[1] = {cb4};
        q4->commit(cbs, 1, co);
        q4->signalEvent(ev, ++evv);
        ev->waitUntilSignaledValue(evv, 10000);
        uint64_t t1 = now_ticks();
        while (!done.load()) {}
        co->release();
        auto data = heap->resolveCounterRange(NS::Range(0, 2 * N));
        if (r >= 10 && data) {
          const uint64_t* ts = (const uint64_t*)data->bytes();
          double sum = 0;
          for (int k = 0; k < N; k++) { double d = (double)(ts[2 * k + 1] - ts[2 * k]) / freq * 1e6; per.push_back(d); sum += d; }
          tot.push_back((double)(ts[2 * N - 1] - ts[0]) / freq * 1e6);
          wall.push_back(ticks_to_us(t1 - t0));
          cbgpu.push_back((ge - gs) * 1e6);
          if (r == 10) printf("   first ts values: %llu %llu %llu (entry stride assumed 8B; data len %lu)\n",
                              (unsigned long long)ts[0], (unsigned long long)ts[1], (unsigned long long)ts[2], (unsigned long)data->length());
        }
        pool->release();
      }
      char nm[160];
      snprintf(nm, sizeof nm, "%s: per-dispatch ts delta (tiny kernel)", gran ? "Precise" : "Relaxed"); if (!per.empty()) report(nm, per);
      snprintf(nm, sizeof nm, "%s: first..last ts span, %d dispatches", gran ? "Precise" : "Relaxed", N); if (!tot.empty()) report(nm, tot);
      snprintf(nm, sizeof nm, "%s: cb GPUEnd-GPUStart (feedback)", gran ? "Precise" : "Relaxed"); if (!cbgpu.empty()) report(nm, cbgpu);
    }
  }

  // ---------------------------------------------------------------- 7. memory: bandwidth shared/private/nocopy-mmap
  printf("\n== 7. Memory: bandwidth of shared vs private vs no-copy mmapped buffers (copy & read kernels) ==\n");
  {
    const size_t SZ = 256ull << 20;
    auto run_copy = [&](MTL::Buffer* src, MTL::Buffer* dst, const char* name, bool readonly) {
      std::vector<double> v;
      for (int r = 0; r < 30 + 3; r++) {
        auto pool = NS::AutoreleasePool::alloc()->init();
        auto cb = c.q->commandBufferWithUnretainedReferences();
        auto e = cb->computeCommandEncoder();
        e->setComputePipelineState(readonly ? c.read4 : c.copy4);
        e->setBuffer(src, 0, 0); e->setBuffer(dst, 0, 1);
        e->dispatchThreads(MTL::Size(SZ / 16, 1, 1), MTL::Size(256, 1, 1));
        e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        double s = cb->GPUEndTime() - cb->GPUStartTime();
        double bytes = readonly ? (double)SZ : 2.0 * SZ;
        if (r == 0) printf("   %-52s first run %.1f GB/s\n", name, bytes / s / 1e9);
        if (r >= 3) v.push_back(bytes / s / 1e9);
        pool->release();
      }
      report(name, v, "GB/s");
    };
    auto sh1 = c.dev->newBuffer(SZ, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    auto sh2 = c.dev->newBuffer(SZ, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    auto pr1 = c.dev->newBuffer(SZ, MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeUntracked);
    auto pr2 = c.dev->newBuffer(SZ, MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeUntracked);
    memset(sh1->contents(), 1, SZ); memset(sh2->contents(), 1, SZ);
    run_copy(sh1, sh2, "copy  shared  -> shared  (r+w GB/s)", false);
    run_copy(pr1, pr2, "copy  private -> private (r+w GB/s)", false);
    run_copy(sh1, sh2, "read  shared             (GB/s)", true);
    run_copy(pr1, pr2, "read  private            (GB/s)", true);
    // mmap a file and wrap without copying
    std::string path = std::string(argc > 1 ? argv[1] : "/tmp") + "/weights.bin";
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0, "open weights file");
    CHECK(ftruncate(fd, SZ) == 0, "ftruncate");
    {
      std::vector<char> chunk(1 << 20, 1);
      for (size_t o = 0; o < SZ; o += chunk.size()) pwrite(fd, chunk.data(), chunk.size(), o);
      fsync(fd);
    }
    for (int prot = 0; prot < 2; prot++) {
      void* p = mmap(nullptr, SZ, prot ? PROT_READ : (PROT_READ | PROT_WRITE), MAP_PRIVATE, fd, 0);
      CHECK(p != MAP_FAILED, "mmap");
      uint64_t t0 = now_ticks();
      auto nb = c.dev->newBuffer(p, SZ, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked, nullptr);
      uint64_t t1 = now_ticks();
      printf("   newBufferWithBytesNoCopy on %s MAP_PRIVATE file mmap (256MB): %s, %.1f us, contents()==ptr %d\n",
             prot ? "PROT_READ" : "PROT_READ|WRITE", nb ? "OK" : "FAILED", ticks_to_us(t1 - t0), nb ? nb->contents() == p : 0);
      if (nb) {
        run_copy(nb, sh2, prot ? "read  nocopy mmap (PROT_READ)  (GB/s)" : "read  nocopy mmap (RW)         (GB/s)", true);
        nb->release();
      }
      munmap(p, SZ);
    }
    close(fd);
    unlink(path.c_str());
  }

  // ---------------------------------------------------------------- 8. heap placement / aliasing check
  printf("\n== 8. MTLHeap placement + aliasing sanity ==\n");
  {
    auto hd = MTL::HeapDescriptor::alloc()->init();
    hd->setType(MTL::HeapTypePlacement);
    hd->setSize(64 << 20);
    hd->setStorageMode(MTL::StorageModeShared);
    hd->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    auto heap = c.dev->newHeap(hd);
    printf("   placement heap 64MB: %s\n", heap ? "OK" : "FAILED");
    if (heap) {
      auto b1 = heap->newBuffer(1 << 20, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked, 0);
      auto b2 = heap->newBuffer(1 << 20, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked, 0);
      printf("   two placement buffers at offset 0: b1 %p b2 %p, gpuAddress equal %d, contents equal %d\n", (void*)b1, (void*)b2,
             b1 && b2 && b1->gpuAddress() == b2->gpuAddress(), b1 && b2 && b1->contents() == b2->contents());
      auto sa = c.dev->heapBufferSizeAndAlign(1 << 20, MTL::ResourceStorageModeShared);
      printf("   heapBufferSizeAndAlign(1MB): size %lu align %lu\n", (unsigned long)sa.size, (unsigned long)sa.align);
    }
  }

  top->release();
  printf("\ndone\n");
  return 0;
}
