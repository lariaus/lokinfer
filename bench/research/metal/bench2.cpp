// lokinfer Metal bench part 2: latency decomposition, warm steady-state, mid-graph sync cost, timestamps.
#include "common.inc"

static double host_s() { return (double)mach_absolute_time() * g_tb / 1e9; }

static MTL::ComputePipelineState* mk(MTL::Device* d, MTL::Library* lib, const char* name, bool icb = false) {
  NS::Error* err = nullptr;
  auto fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  auto desc = MTL::ComputePipelineDescriptor::alloc()->init();
  desc->setComputeFunction(fn);
  desc->setSupportIndirectCommandBuffers(icb);
  auto p = d->newComputePipelineState(desc, MTL::PipelineOptionNone, nullptr, &err);
  CHECK(p, name);
  return p;
}

int main() {
  mach_timebase_info_data_t tb; mach_timebase_info(&tb); g_tb = (double)tb.numer / tb.denom;
  auto top = NS::AutoreleasePool::alloc()->init();
  auto dev = MTL::CreateSystemDefaultDevice();
  auto q = dev->newCommandQueue();
  NS::Error* err = nullptr;
  auto lib = dev->newLibrary(NS::String::string(kSrc, NS::UTF8StringEncoding), MTL::CompileOptions::alloc()->init(), &err);
  CHECK(lib, "lib");
  auto tiny = mk(dev, lib, "tiny"), tiny_dev = mk(dev, lib, "tiny_dev"), tiny_icb = mk(dev, lib, "tiny_dev", true);
  auto copy4 = mk(dev, lib, "copy4");
  const uint32_t TG = 256; uint32_t n = TG;
  auto opt = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;
  auto A = dev->newBuffer(4096 * TG * 4, opt), B = dev->newBuffer(4096 * TG * 4, opt), P = dev->newBuffer(4096 * 16, opt);
  memset(A->contents(), 0, A->length()); memset(B->contents(), 0, B->length());
  for (int i = 0; i < 4096; i++) ((uint32_t*)P->contents())[i * 4] = TG;
  auto big1 = dev->newBuffer(64 << 20, opt), big2 = dev->newBuffer(64 << 20, opt);
  auto rsd = MTL::ResidencySetDescriptor::alloc()->init();
  auto rs = dev->newResidencySet(rsd, &err);
  rs->addAllocation(A); rs->addAllocation(B); rs->addAllocation(P); rs->addAllocation(big1); rs->addAllocation(big2);
  rs->commit(); rs->requestResidency();
  q->addResidencySet(rs);
  auto q4 = dev->newMTL4CommandQueue();
  q4->addResidencySet(rs);
  auto ev = dev->newSharedEvent(); uint64_t evv = 0;

  auto enc_chain = [&](MTL::CommandBuffer* cb, int N, bool chain) {
    auto e = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    for (int k = 0; k < N; k++) {
      if (chain && k) e->memoryBarrier(MTL::BarrierScopeBuffers);
      size_t off = chain ? 0 : (size_t)(k % 4096) * TG * 4;
      e->setComputePipelineState(tiny); e->setBuffer(A, off, 0); e->setBuffer(B, off, 1); e->setBytes(&n, 4, 2);
      e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
    }
    e->endEncoding();
  };
  auto heavy = [&](MTL::CommandBuffer* cb) {  // ~0.4 ms of bandwidth work to keep clocks up
    auto e = cb->computeCommandEncoder();
    e->setComputePipelineState(copy4); e->setBuffer(big1, 0, 0); e->setBuffer(big2, 0, 1);
    e->dispatchThreads(MTL::Size((64 << 20) / 16, 1, 1), MTL::Size(256, 1, 1));
    e->endEncoding();
  };

  // ------------------------------------------------------------ A. latency decomposition
  printf("== A. 1-dispatch cb latency decomposition (us): commit->GPUStart, GPU, GPUEnd->CPU wake ==\n");
  for (int cond = 0; cond < 3; cond++) {
    std::vector<double> s2g, gdur, g2w, tot;
    for (int r = 0; r < 600; r++) {
      auto pool = NS::AutoreleasePool::alloc()->init();
      if (cond == 0) usleep(3000);  // GPU goes idle
      if (cond == 2) { auto h = q->commandBufferWithUnretainedReferences(); heavy(h); h->commit(); h->waitUntilCompleted(); }
      auto cb = q->commandBufferWithUnretainedReferences();
      enc_chain(cb, 1, false);
      cb->encodeSignalEvent(ev, ++evv);
      double tc = host_s();
      cb->commit();
      while (ev->signaledValue() < evv) {}
      double tw = host_s();
      cb->waitUntilCompleted();
      if (r >= 20) {
        s2g.push_back((cb->GPUStartTime() - tc) * 1e6); gdur.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e6);
        g2w.push_back((tw - cb->GPUEndTime()) * 1e6); tot.push_back((tw - tc) * 1e6);
      }
      pool->release();
    }
    const char* cn[] = {"idle 3ms before", "back-to-back (prev rep just finished)", "right after 64MB copy kernel (GPU hot)"};
    printf(" -- %s\n", cn[cond]);
    report("    commit -> GPUStartTime", s2g); report("    GPU duration", gdur);
    report("    GPUEndTime -> CPU sees event (spin)", g2w); report("    total commit -> CPU wake", tot);
  }

  // ------------------------------------------------------------ B. warm steady-state per-dispatch GPU cost
  printf("\n== B. Warm steady state: 40 back-to-back cbs of 200 tiny dispatches; per-dispatch GPU time ==\n");
  {
    auto icbd = MTL::IndirectCommandBufferDescriptor::alloc()->init();
    icbd->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
    icbd->setInheritBuffers(false); icbd->setInheritPipelineState(false); icbd->setMaxKernelBufferBindCount(3);
    auto icbC = dev->newIndirectCommandBuffer(icbd, 200, MTL::ResourceStorageModeShared);
    auto icbI = dev->newIndirectCommandBuffer(icbd, 200, MTL::ResourceStorageModeShared);
    for (int k = 0; k < 200; k++) {
      for (int w = 0; w < 2; w++) {
        auto cmd = (w ? icbI : icbC)->indirectComputeCommand(k);
        size_t off = w ? (size_t)k * TG * 4 : 0;
        cmd->setComputePipelineState(tiny_icb); cmd->setKernelBuffer(A, off, 0); cmd->setKernelBuffer(B, off, 1); cmd->setKernelBuffer(P, 0, 2);
        cmd->concurrentDispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
        if (!w) cmd->setBarrier();
      }
    }
    rs->addAllocation(icbC); rs->addAllocation(icbI); rs->commit();
    const char* vn[] = {"classic concurrent enc, chain + memoryBarrier", "classic concurrent enc, independent",
                        "classic serial enc, chain (implicit)", "classic ICB replay, chain (setBarrier)", "classic ICB replay, independent",
                        "classic 1 encoder per dispatch, chain"};
    for (int var = 0; var < 6; var++) {
      for (int rep = 0; rep < 3; rep++) {
        std::vector<MTL::CommandBuffer*> cbs;
        uint64_t t0 = now_ticks();
        for (int i = 0; i < 40; i++) {
          auto cb = q->commandBuffer();
          if (var == 0) enc_chain(cb, 200, true);
          else if (var == 1) enc_chain(cb, 200, false);
          else if (var == 2 || var == 5) {
            auto e = cb->computeCommandEncoder();
            for (int k = 0; k < 200; k++) {
              if (var == 5 && k) { e->endEncoding(); e = cb->computeCommandEncoder(); }
              e->setComputePipelineState(tiny); e->setBuffer(A, 0, 0); e->setBuffer(B, 0, 1); e->setBytes(&n, 4, 2);
              e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
            }
            e->endEncoding();
          } else {
            auto e = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
            e->executeCommandsInBuffer(var == 3 ? icbC : icbI, NS::Range(0, 200));
            e->endEncoding();
          }
          cb->commit();
          cbs.push_back(cb);
        }
        uint64_t t1 = now_ticks();
        cbs.back()->waitUntilCompleted();
        uint64_t t2 = now_ticks();
        if (rep < 2) continue;  // warm-up reps
        std::vector<double> g;
        for (size_t i = 5; i < cbs.size(); i++) g.push_back((cbs[i]->GPUEndTime() - cbs[i]->GPUStartTime()) * 1e6);
        char nm[160]; snprintf(nm, sizeof nm, "%-48s gpu/disp", vn[var]); report(nm, g, "us", 200);
        printf("%-62s cpu encode %.3f us/disp, wall %.3f us/disp (8000 disp)\n", "", ticks_to_us(t1 - t0) / 8000, ticks_to_us(t2 - t0) / 8000);
      }
    }
    // MTL4
    auto at = dev->newArgumentTable([] { auto d = MTL4::ArgumentTableDescriptor::alloc()->init(); d->setMaxBufferBindCount(4); return d; }(), &err);
    std::vector<MTL4::CommandAllocator*> allocs; std::vector<MTL4::CommandBuffer*> cb4s;
    for (int i = 0; i < 40; i++) { allocs.push_back(dev->newCommandAllocator()); cb4s.push_back(dev->newCommandBuffer()); }
    const char* vn4[] = {"MTL4 chain + barrierAfterEncoderStages", "MTL4 independent", "MTL4 ICB replay chain", "MTL4 1 encoder per dispatch + queue barrier"};
    for (int var = 0; var < 4; var++) {
      for (int rep = 0; rep < 3; rep++) {
        std::vector<double> gs(40), ge(40);
        std::atomic<int> done{0};
        uint64_t t0 = now_ticks();
        for (int i = 0; i < 40; i++) {
          allocs[i]->reset();
          auto cb = cb4s[i];
          cb->beginCommandBuffer(allocs[i]);
          auto e = cb->computeCommandEncoder();
          e->setArgumentTable(at);
          if (var == 2) e->executeCommandsInBuffer(icbC, NS::Range(0, 200));
          else for (int k = 0; k < 200; k++) {
            if (var == 0 && k) e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
            if (var == 3 && k) { e->endEncoding(); e = cb->computeCommandEncoder(); e->setArgumentTable(at);
                                 e->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone); }
            size_t off = var == 1 ? (size_t)k * TG * 4 : 0;
            e->setComputePipelineState(tiny_dev);
            at->setAddress(A->gpuAddress() + off, 0); at->setAddress(B->gpuAddress() + off, 1); at->setAddress(P->gpuAddress(), 2);
            e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
          }
          e->endEncoding();
          cb->endCommandBuffer();
          auto co = MTL4::CommitOptions::alloc()->init();
          co->addFeedbackHandler([&, i](MTL4::CommitFeedback* fb) { gs[i] = fb->GPUStartTime(); ge[i] = fb->GPUEndTime(); done++; });
          const MTL4::CommandBuffer* arr[1] = {cb};
          q4->commit(arr, 1, co);
          co->release();
        }
        q4->signalEvent(ev, ++evv);
        uint64_t t1 = now_ticks();
        ev->waitUntilSignaledValue(evv, 100000);
        uint64_t t2 = now_ticks();
        while (done.load() < 40) {}
        if (rep < 2) continue;
        std::vector<double> g;
        for (int i = 5; i < 40; i++) g.push_back((ge[i] - gs[i]) * 1e6);
        char nm[160]; snprintf(nm, sizeof nm, "%-48s gpu/disp", vn4[var]); report(nm, g, "us", 200);
        printf("%-62s cpu encode %.3f us/disp, wall %.3f us/disp (8000 disp)\n", "", ticks_to_us(t1 - t0) / 8000, ticks_to_us(t2 - t0) / 8000);
      }
    }
  }

  // ------------------------------------------------------------ C. mid-graph sync cost, warm
  printf("\n== C. Mid-graph CPU sync cost (warm): 30 segments x 100 chained dispatches ==\n");
  for (int mode = 0; mode < 5; mode++) {
    std::vector<double> v;
    for (int rep = 0; rep < 12; rep++) {
      auto pool = NS::AutoreleasePool::alloc()->init();
      uint64_t t0 = now_ticks();
      MTL::CommandBuffer* last = nullptr;
      MTL::CommandBuffer* pre = nullptr;
      for (int s = 0; s < 30; s++) {
        MTL::CommandBuffer* cb;
        if (mode == 3 && pre) cb = pre; else { cb = q->commandBufferWithUnretainedReferences(); enc_chain(cb, 100, true); }
        if (mode >= 1) cb->encodeSignalEvent(ev, ++evv);
        cb->commit();
        last = cb;
        if (mode == 3) { pre = q->commandBufferWithUnretainedReferences(); enc_chain(pre, 100, true); }  // pre-encode next while GPU runs
        if (mode == 1 || mode == 3) { while (ev->signaledValue() < evv) {} }
        if (mode == 2) ev->waitUntilSignaledValue(evv, 100000);
        if (mode == 4) cb->waitUntilCompleted();
      }
      last->waitUntilCompleted();
      uint64_t t1 = now_ticks();
      if (rep >= 2) v.push_back(ticks_to_us(t1 - t0) / 30);
      pool->release();
    }
    const char* mn[] = {"no sync (pipelined)", "sync each segment: event spin, encode after", "sync each segment: event waitUntilSignaledValue",
                        "sync each segment: event spin, next cb pre-encoded", "sync each segment: waitUntilCompleted"};
    char nm[160]; snprintf(nm, sizeof nm, "%-50s per seg", mn[mode]); report(nm, v);
  }

  // ------------------------------------------------------------ D. timestamps
  printf("\n== D. Timestamps ==\n");
  {
    // MTL4 relaxed: dump values
    auto chd = MTL4::CounterHeapDescriptor::alloc()->init();
    chd->setType(MTL4::CounterHeapTypeTimestamp); chd->setCount(64);
    auto heap = dev->newCounterHeap(chd, &err);
    auto at = dev->newArgumentTable([] { auto d = MTL4::ArgumentTableDescriptor::alloc()->init(); d->setMaxBufferBindCount(4); return d; }(), &err);
    auto al = dev->newCommandAllocator(); auto cb = dev->newCommandBuffer();
    double freq = (double)dev->queryTimestampFrequency();
    for (int gran = 0; gran < 2; gran++) {
      heap->invalidateCounterRange(NS::Range(0, 64));
      al->reset(); cb->beginCommandBuffer(al);
      auto e = cb->computeCommandEncoder(); e->setArgumentTable(at);
      for (int k = 0; k < 8; k++) {
        if (k) e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
        e->writeTimestamp((MTL4::TimestampGranularity)gran, heap, 2 * k);
        e->setComputePipelineState(copy4); at->setAddress(big1->gpuAddress(), 0); at->setAddress(big2->gpuAddress(), 1);
        e->dispatchThreads(MTL::Size((64 << 20) / 16, 1, 1), MTL::Size(256, 1, 1));
        e->writeTimestamp((MTL4::TimestampGranularity)gran, heap, 2 * k + 1);
      }
      e->endEncoding();
      cb->writeTimestampIntoHeap(heap, 16);
      cb->endCommandBuffer();
      const MTL4::CommandBuffer* arr[1] = {cb};
      q4->commit(arr, 1); q4->signalEvent(ev, ++evv); ev->waitUntilSignaledValue(evv, 100000);
      auto data = heap->resolveCounterRange(NS::Range(0, 17));
      const uint64_t* ts = (const uint64_t*)data->bytes();
      printf(" MTL4 %s, 8 x 64MB copy kernels (each ~0.4ms): per-kernel us:", gran ? "Precise" : "Relaxed");
      for (int k = 0; k < 8; k++) printf(" %.1f", ((double)ts[2 * k + 1] - (double)ts[2 * k]) / freq * 1e6);
      printf("\n   raw:"); for (int k = 0; k < 17; k++) printf(" %llu", (unsigned long long)ts[k]); printf("\n");
    }
    // Classic: stage-boundary sampling, one encoder per kernel.
    auto sets = dev->counterSets();
    MTL::CounterSet* tsSet = nullptr;
    for (NS::UInteger i = 0; i < sets->count(); i++) {
      auto s = (MTL::CounterSet*)sets->object(i);
      printf(" counter set: %s\n", s->name()->utf8String());
      if (s->name()->isEqualToString(MTL::CommonCounterSetTimestamp)) tsSet = s;
    }
    if (tsSet) {
      const int N = 100;
      auto sd = MTL::CounterSampleBufferDescriptor::alloc()->init();
      sd->setCounterSet(tsSet); sd->setStorageMode(MTL::StorageModeShared); sd->setSampleCount(2 * N);
      auto sb = dev->newCounterSampleBuffer(sd, &err);
      CHECK(sb, "counter sample buffer");
      for (int withTs = 0; withTs < 2; withTs++) {
        std::vector<double> per, cbt;
        for (int rep = 0; rep < 60; rep++) {
          auto pool = NS::AutoreleasePool::alloc()->init();
          auto c = q->commandBufferWithUnretainedReferences();
          for (int k = 0; k < N; k++) {
            auto pd = MTL::ComputePassDescriptor::computePassDescriptor();
            if (withTs) {
              auto att = pd->sampleBufferAttachments()->object(0);
              att->setSampleBuffer(sb); att->setStartOfEncoderSampleIndex(2 * k); att->setEndOfEncoderSampleIndex(2 * k + 1);
            }
            auto e = c->computeCommandEncoder(pd);
            e->setComputePipelineState(tiny); e->setBuffer(A, 0, 0); e->setBuffer(B, 0, 1); e->setBytes(&n, 4, 2);
            e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
            e->endEncoding();
          }
          c->commit(); c->waitUntilCompleted();
          if (rep >= 10) {
            cbt.push_back((c->GPUEndTime() - c->GPUStartTime()) * 1e6 / N);
            if (withTs) {
              auto d = sb->resolveCounterRange(NS::Range(0, 2 * N));
              const uint64_t* ts = (const uint64_t*)d->bytes();
              for (int k = 0; k < N; k++) per.push_back(((double)ts[2 * k + 1] - (double)ts[2 * k]) / freq * 1e6);
            }
          }
          pool->release();
        }
        report(withTs ? " classic 1 enc/kernel WITH stage-boundary ts: cb gpu/disp" : " classic 1 enc/kernel, no ts: cb gpu/disp", cbt);
        if (withTs) report(" classic stage-boundary ts: per-encoder delta (tiny)", per);
      }
    }
  }
  top->release();
  printf("\ndone\n");
}
