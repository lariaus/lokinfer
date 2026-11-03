// metal-cpp implementation of GpuContext (the only TU defining the
// metal-cpp private implementation symbols).
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <CoreFoundation/CoreFoundation.h>
#include <TargetConditionals.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>

#include "gpu_internal.h"
#include "gpu_replay.h"

namespace lk::rt {

namespace {

std::string& metallibPath() {
  static std::string p = [] {
    if (const char* env = std::getenv("LOKINFER_METALLIB")) return std::string(env);
#ifdef LOKINFER_DEFAULT_METALLIB
    return std::string(LOKINFER_DEFAULT_METALLIB);
#else
    return std::string("lokinfer.metallib");
#endif
  }();
  return p;
}

constexpr size_t kParamsCapacity = 1 << 20;
constexpr uint32_t kMaxBindings = 16;

}  // namespace

void setMetallibPath(std::string path) { metallibPath() = std::move(path); }

GpuContext::~GpuContext() {
  commitAndWait();
  for (auto& [n, p] : pipelines_) p->release();
  if (counterHeap_) static_cast<MTL4::CounterHeap*>(counterHeap_)->release();
  if (commitOptions_) commitOptions_->release();
  for (SegSlot& sl : slots_) {
    sl.params->release();
    sl.cmd->release();
    sl.allocator->release();
  }
  if (table_) table_->release();
  if (queue_) queue_->release();
  if (event_) event_->release();
  if (residency_) residency_->release();
  if (library_) library_->release();
  if (device_) device_->release();
}

Expected<std::unique_ptr<GpuContext>> GpuContext::create() {
  std::unique_ptr<GpuContext> g(new GpuContext());
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  g->device_ = MTL::CreateSystemDefaultDevice();
  if (!g->device_) {
    pool->release();
    return Status::error("no Metal device");
  }
  if (!g->device_->supportsFamily(MTL::GPUFamilyMetal4)) {
    pool->release();
    return Status::error("Metal 4 not supported on this GPU");
  }
  NS::Error* err = nullptr;
  NS::String* path = NS::String::string(metallibPath().c_str(), NS::UTF8StringEncoding);
  g->library_ = g->device_->newLibrary(NS::URL::fileURLWithPath(path), &err);
  if (!g->library_) {
    std::string msg = "cannot load metallib '" + metallibPath() + "'";
    if (err) msg += std::string(": ") + err->localizedDescription()->utf8String();
    pool->release();
    return Status::error(msg);
  }
  g->queue_ = g->device_->newMTL4CommandQueue();
  auto* atd = MTL4::ArgumentTableDescriptor::alloc()->init();
  atd->setMaxBufferBindCount(kMaxBindings);
  g->table_ = g->device_->newArgumentTable(atd, &err);
  atd->release();
  auto* rsd = MTL::ResidencySetDescriptor::alloc()->init();
  g->residency_ = g->device_->newResidencySet(rsd, &err);
  rsd->release();
  g->event_ = g->device_->newSharedEvent();
  if (!g->queue_ || !g->table_ || !g->residency_ || !g->event_) {
    pool->release();
    return Status::error("failed to create Metal 4 objects");
  }
  g->queue_->addResidencySet(g->residency_);
  // Commit feedback: a failed command buffer (GPU fault / timeout / OOM) is recorded so waits don't hang on an event
  // it will never signal, and the call returns the error.
  g->commitOptions_ = MTL4::CommitOptions::alloc()->init();
  GpuContext* self = g.get();
  g->commitOptions_->addFeedbackHandler([self](MTL4::CommitFeedback* fb) {
    NS::Error* e = fb ? fb->error() : nullptr;
    if (!e) return;
    std::lock_guard<std::mutex> lock(self->gpuErrorMu_);
    if (self->gpuError_.empty()) {
      const char* d = e->localizedDescription() ? e->localizedDescription()->utf8String() : nullptr;
      self->gpuError_ = std::string("GPU command buffer failed: ") + (d ? d : "unknown error");
    }
    self->gpuFailed_ = true;
  });
  g->paramsCapacity_ = kParamsCapacity;
  // Command buffers that may exist at once (docs/runtime.md). macOS: 2 (one in flight while the next is encoded;
  // interleaved A/B 2026-11-02: Kokoro app59f GPU 50.7 -> 50.1 ms, GPT-2 prefill T=128 5.04 -> 4.90, decode
  // 2.17 -> 2.03; 3 no better). iOS: 2 as well since the slot-reuse fix + early commit (interleaved A/B x2,
  // 2026-11-02: GPT-2 prefill T=128 11.44 -> 11.29 / 11.30 ms (no bimodality), decode 5.58 -> 5.56, Kokoro app59f
  // 65.3 -> 64.6); before them it was 1 (T=128 bimodal 11.3 / 12.9 ms at 2).
#if TARGET_OS_IPHONE
  int segments = 2;
#else
  int segments = 2;
#endif
  if (const char* e = std::getenv("LOKINFER_GPU_INFLIGHT")) segments = std::atoi(e);
  segments = std::min(std::max(segments, 1), 4);
  for (int i = 0; i < segments; ++i) {
    SegSlot sl{g->device_->newCommandAllocator(), g->device_->newCommandBuffer(),
               g->device_->newBuffer(kParamsCapacity, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked)};
    if (!sl.allocator || !sl.cmd || !sl.params) {
      pool->release();
      return Status::error("failed to create Metal 4 command buffers");
    }
    g->residency_->addAllocation(sl.params);
    g->slots_.push_back(sl);
  }
  g->allocator_ = g->slots_[0].allocator;
  g->cmd_ = g->slots_[0].cmd;
  g->params_ = g->slots_[0].params;
  g->residencyDirty_ = true;
  pool->release();
  return g;
}

MTL::ComputePipelineState* GpuContext::pipeline(const std::string& function) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  auto it = pipelines_.find(function);
  if (it != pipelines_.end()) return it->second;
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState* pso = nullptr;
  NS::String* nsName = NS::String::string(function.c_str(), NS::UTF8StringEncoding);  // nil for invalid UTF-8
  MTL::Function* fn = nsName ? library_->newFunction(nsName) : nullptr;
  if (fn) {
    NS::Error* err = nullptr;
    pso = device_->newComputePipelineState(fn, &err);
    if (!pso && err) std::fprintf(stderr, "lokinfer: pipeline %s: %s\n", function.c_str(), err->localizedDescription()->utf8String());
    fn->release();
  } else {
    std::fprintf(stderr, "lokinfer: kernel function '%s' not in metallib\n", function.c_str());
  }
  pool->release();
  if (pso) {
    pipelines_[function] = pso;
    psoNames_[pso] = function;
  }
  return pso;
}

void GpuContext::commitResidency() {
  if (residencyDirty_) {
    residency_->commit();
    residencyDirty_ = false;
  }
}

MTL4::ComputeCommandEncoder* GpuContext::beginDispatch() {
  if (!cmdOpen_) {
    // The next slot's allocator, command buffer and params ring are reused: wait until it is free (with one slot,
    // until the previous segment completed).
    while (inflight_.size() >= slots_.size()) waitOldest();
    // Nothing in flight: reuse the last slot (warm allocator / command buffer / params ring). Rotating through the
    // slots on every call made single-segment models slower on the iPhone (GPT-2 prefill T=128 GPU 11.6 -> 12.9 ms).
    if (inflight_.empty()) nextSlot_ = lastSlot_;
    lastSlot_ = nextSlot_;
    const SegSlot& sl = slots_[nextSlot_];
    nextSlot_ = (nextSlot_ + 1) % slots_.size();
    allocator_ = sl.allocator;
    cmd_ = sl.cmd;
    params_ = sl.params;
    queueBarrier_ = !inflight_.empty();
    // One autorelease pool spans every open / in-flight segment and drains when the GPU is idle (commitAndWait):
    // pools are LIFO, so per-segment pools can't retire out of order with several segments in flight.
    if (!segmentPool_) segmentPool_ = NS::AutoreleasePool::alloc()->init();
    allocator_->reset();
    cmd_->beginCommandBuffer(allocator_);
    cmdOpen_ = true;
    paramsOffset_ = 0;
  }
  if (!encoder_) {
    encoder_ = cmd_->computeCommandEncoder();
    encoder_->setArgumentTable(table_);
    dispatchesInEncoder_ = 0;
    if (queueBarrier_) {  // order after the in-flight segments' dispatches (Metal 4 doesn't across command buffers)
      encoder_->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
      queueBarrier_ = false;
    }
  }
  if (tableDirty_) {
    encoder_->setArgumentTable(table_);
    tableDirty_ = false;
  }
  if (dispatchesInEncoder_ > 0 && nextBarrier_)
    encoder_->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionNone);
  lastBarrier_ = nextBarrier_;
  nextBarrier_ = true;
  if (profBeginPending_) {  // profiling: the kernel's begin timestamp, after its barrier
    profBeginPending_ = false;
    profileStamp(profTag_, true);
  }
  return encoder_;
}

void GpuContext::bind(uint32_t slot, MTL::Buffer* buffer, uint64_t offset) {
  table_->setAddress(buffer->gpuAddress() + offset, slot);
  if (recorder_) recorder_->onBind(slot, buffer, offset);
}

void GpuContext::bindParams(uint32_t slot, const void* data, size_t size) {
  size_t off = (paramsOffset_ + 15) & ~size_t(15);
  if (off + size > paramsCapacity_) {
    // Ring full mid-segment: flush and start over.
    if (recorder_) recorderFail("params ring flushed mid-call");
    commitAndWait();
    beginDispatch();
    off = 0;
  }
  std::memcpy(static_cast<char*>(params_->contents()) + off, data, size);
  paramsOffset_ = off + size;
  table_->setAddress(params_->gpuAddress() + off, slot);
  if (recorder_) recorder_->onParams(slot, data, size);
}

void GpuContext::dispatch(MTL::ComputePipelineState* pso, uint64_t gx, uint64_t gy, uint64_t gz, uint64_t tx,
                          uint64_t ty, uint64_t tz) {
  encoder_->setComputePipelineState(pso);
  encoder_->dispatchThreads(MTL::Size(gx, gy, gz), MTL::Size(tx, ty, tz));
  ++dispatchesInEncoder_;
  if (recorder_) recorder_->onDispatch(pso, gx, gy, gz, tx, ty, tz, false, lastBarrier_);
  if (profiling_) profileLogDispatch(pso, gx, gy, gz, tx, ty, tz, false);
}

void GpuContext::dispatchThreadgroups(MTL::ComputePipelineState* pso, uint64_t gx, uint64_t gy, uint64_t gz,
                                      uint64_t tx, uint64_t ty, uint64_t tz) {
  encoder_->setComputePipelineState(pso);
  encoder_->dispatchThreadgroups(MTL::Size(gx, gy, gz), MTL::Size(tx, ty, tz));
  ++dispatchesInEncoder_;
  if (recorder_) recorder_->onDispatch(pso, gx, gy, gz, tx, ty, tz, true, lastBarrier_);
  if (profiling_) profileLogDispatch(pso, gx, gy, gz, tx, ty, tz, true);
}

void GpuContext::endAndCommit() {
  if (encoder_) {
    encoder_->endEncoding();
    encoder_ = nullptr;
  }
  cmd_->endCommandBuffer();
  commitResidency();
  const MTL4::CommandBuffer* cbs[1] = {cmd_};
  queue_->commit(cbs, 1, commitOptions_);
  queue_->signalEvent(event_, ++eventValue_);
  cmdOpen_ = false;
  ++segmentId_;
}

void GpuContext::commitAsync() {
  if (!cmdOpen_) return;
  endAndCommit();
  inflight_.push_back(Inflight{eventValue_, std::move(keepAlive_), std::move(pendingRelease_), nullptr});
  keepAlive_.clear();
  pendingRelease_.clear();
}

// Spin on the event's value for up to 2 ms before falling back to the blocking wait: the blocking path adds ~100 us of
// wake-up latency per call (M2 Max: wait 180 -> 75 us for a 1-kernel graph), which dominated small models
// (MNIST, DCGAN / MobileNetV2 at batch 1).
void GpuContext::waitEvent(uint64_t value) {
  if (value <= skipUpTo_) return;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
  while (event_->signaledValue() < value) {
    if (gpuFailed_) {  // a failed command buffer may never signal: don't wait for it (takeGpuError reports it)
      skipUpTo_ = eventValue_;
      return;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      // Blocking wait in 100 ms slices so a failure reported meanwhile ends it.
      while (!event_->waitUntilSignaledValue(value, 100)) {
        if (gpuFailed_) {
          skipUpTo_ = eventValue_;
          return;
        }
      }
      return;
    }
  }
}

std::string GpuContext::takeGpuError() {
  if (!gpuFailed_) return {};
  std::lock_guard<std::mutex> lock(gpuErrorMu_);
  std::string e;
  e.swap(gpuError_);
  gpuFailed_ = false;
  return e.empty() ? "GPU command buffer failed" : e;
}

void GpuContext::retire(Inflight& f) {
  for (MTL::Allocation* b : f.release) {
    residency_->removeAllocation(b);
    residencyDirty_ = true;
    b->release();
  }
  for (void* o : f.keepAlive) CFRelease(o);
  if (f.pool) static_cast<NS::AutoreleasePool*>(f.pool)->release();
}

void GpuContext::setMaxSegments(size_t n) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  n = std::min<size_t>(std::max<size_t>(n, 1), 4);
  commitAndWait();
  while (slots_.size() > n) {
    SegSlot sl = slots_.back();
    slots_.pop_back();
    residency_->removeAllocation(sl.params);
    residencyDirty_ = true;
    sl.params->release();
    sl.cmd->release();
    sl.allocator->release();
  }
  while (slots_.size() < n) {
    SegSlot sl{device_->newCommandAllocator(), device_->newCommandBuffer(),
               device_->newBuffer(kParamsCapacity, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked)};
    residency_->addAllocation(sl.params);
    residencyDirty_ = true;
    slots_.push_back(sl);
  }
  nextSlot_ = 0;
  lastSlot_ = 0;
}

// Waits for the oldest in-flight segment and releases what it held.
void GpuContext::waitOldest() {
  if (inflight_.empty()) return;
  const auto t0 = profiling_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  waitEvent(inflight_.front().value);
  if (profiling_)
    profWaitNs_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  retire(inflight_.front());
  inflight_.erase(inflight_.begin());
}

// Waits for every in-flight segment (the queue signals in commit order, so the newest value covers all).
void GpuContext::waitInflight() {
  if (inflight_.empty()) return;
  const auto t0 = profiling_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  waitEvent(inflight_.back().value);
  if (profiling_)
    profWaitNs_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  for (Inflight& f : inflight_) retire(f);
  inflight_.clear();
}

void GpuContext::commitAndWait() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (!cmdOpen_) {
    waitInflight();
    if (segmentPool_) {
      static_cast<NS::AutoreleasePool*>(segmentPool_)->release();
      segmentPool_ = nullptr;
    }
    return;
  }
  endAndCommit();
  waitEvent(eventValue_);
  waitInflight();  // everything committed earlier completed too
  for (MTL::Allocation* b : pendingRelease_) {
    residency_->removeAllocation(b);
    residencyDirty_ = true;
    b->release();
  }
  pendingRelease_.clear();
  for (void* o : keepAlive_) CFRelease(o);
  keepAlive_.clear();
  if (segmentPool_) {
    static_cast<NS::AutoreleasePool*>(segmentPool_)->release();
    segmentPool_ = nullptr;
  }
}

MTL::Buffer* GpuContext::wrapNoCopy(void* data, size_t size) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  MTL::Buffer* b = device_->newBuffer(data, size,
                                      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked,
                                      nullptr);
  if (b) {
    residency_->addAllocation(b);
    residencyDirty_ = true;
  }
  return b;
}

void GpuContext::releaseBuffer(MTL::Buffer* b) { releaseAllocation(b); }

void GpuContext::releaseAllocation(MTL::Allocation* b) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  // An open or in-flight segment may still reference it: release after that segment
  // completes instead of forcing a commit + wait mid-call.
  if (cmdOpen_) {
    pendingRelease_.push_back(b);
    return;
  }
  if (!inflight_.empty()) {
    inflight_.back().release.push_back(b);
    return;
  }
  residency_->removeAllocation(b);
  residencyDirty_ = true;
  b->release();
}


// ---- record/replay (gpu_replay.cpp) -------------------------------------------------------------------------------

void GpuContext::recorderFail(const char* why) {
  GpuReplay* r = recorder_;
  recorder_ = nullptr;
  r->fail(why);
}

void GpuContext::makeResident(MTL::Allocation* a) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  residency_->addAllocation(a);
  residencyDirty_ = true;
}

MTL::ComputePipelineState* GpuContext::icbPipeline(MTL::ComputePipelineState* pso) {
  if (auto it = icbPipelines_.find(pso); it != icbPipelines_.end()) return it->second;
  auto name = psoNames_.find(pso);
  if (name == psoNames_.end()) return nullptr;
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState* out = nullptr;
  NS::String* nsName = NS::String::string(name->second.c_str(), NS::UTF8StringEncoding);
  if (MTL::Function* fn = nsName ? library_->newFunction(nsName) : nullptr) {
    auto* d = MTL::ComputePipelineDescriptor::alloc()->init();
    d->setComputeFunction(fn);
    d->setSupportIndirectCommandBuffers(true);
    NS::Error* err = nullptr;
    out = device_->newComputePipelineState(d, MTL::PipelineOptionNone, nullptr, &err);
    if (!out && err)
      std::fprintf(stderr, "lokinfer: ICB pipeline %s: %s\n", name->second.c_str(), err->localizedDescription()->utf8String());
    d->release();
    fn->release();
  }
  pool->release();
  if (out) {
    makeResident(out);  // Metal 4: pipelines an ICB references must be resident
    icbPipelines_[pso] = out;
    psoNames_[out] = name->second;
  }
  return out;
}

void GpuContext::executeIndirect(MTL::IndirectCommandBuffer* icb, uint32_t count) {
  nextBarrier_ = true;  // the replayed commands depend on whatever came before in this encoder
  beginDispatch();
  encoder_->executeCommandsInBuffer(icb, NS::Range(0, count));
  ++dispatchesInEncoder_;
}

// ---- profiling ---------------------------------------------------------------------------------------------------

bool GpuContext::profileStart(uint32_t maxStamps) {
  profiling_ = true;
  profWaitNs_ = 0;
  profBeginPending_ = false;
  profTag_ = -1;
  stampCount_ = 0;
  stamps_.clear();
  profDispatches_.clear();
  if (counterCapacity_ < maxStamps) {
    if (counterHeap_) static_cast<MTL4::CounterHeap*>(counterHeap_)->release();
    counterHeap_ = nullptr;
    counterCapacity_ = 0;
    auto* d = MTL4::CounterHeapDescriptor::alloc()->init();
    d->setType(MTL4::CounterHeapTypeTimestamp);
    d->setCount(maxStamps);
    NS::Error* err = nullptr;
    counterHeap_ = device_->newCounterHeap(d, &err);
    d->release();
    if (counterHeap_) counterCapacity_ = maxStamps;
  }
  return counterHeap_ != nullptr;
}

// Writes a timestamp for kernel `tag` into the counter heap. "Precise" granularity: the GPU writes it exactly
// between the surrounding dispatches (Metal may serialize there), so begin/end bracket the kernel's own work.
// LOKINFER_PROFILE_STAMPS=relaxed lets Metal coalesce them (cheaper, coarser).
void GpuContext::profileStamp(int32_t tag, bool begin) {
  static const auto granularity = [] {
    const char* e = std::getenv("LOKINFER_PROFILE_STAMPS");
    return e && std::strcmp(e, "relaxed") == 0 ? MTL4::TimestampGranularityRelaxed : MTL4::TimestampGranularityPrecise;
  }();
  if (!counterHeap_ || !encoder_) return;
  if (begin) {
    if (stampCount_ + 2 > counterCapacity_) return;  // full: the rest of the call has no GPU times
    encoder_->writeTimestamp(granularity, static_cast<MTL4::CounterHeap*>(counterHeap_), stampCount_);
    stamps_.push_back({tag, stampCount_, ~0u});
    stampCount_ += 2;  // the end stamp goes right after
    profKernelStamped_ = true;
    return;
  }
  StampSlots& s = stamps_.back();
  encoder_->writeTimestamp(granularity, static_cast<MTL4::CounterHeap*>(counterHeap_), s.begin + 1);
  s.end = s.begin + 1;
}

void GpuContext::profileKernelEnd() {
  if (!profiling_) return;
  profBeginPending_ = false;
  if (profKernelStamped_ && !stamps_.empty() && stamps_.back().tag == profTag_) profileStamp(profTag_, false);
  profKernelStamped_ = false;
  profTag_ = -1;
}

void GpuContext::profileLogDispatch(MTL::ComputePipelineState* pso, uint64_t gx, uint64_t gy, uint64_t gz,
                                    uint64_t tx, uint64_t ty, uint64_t tz, bool groups) {
  ProfDispatch d{profTag_, "(MPS)", {uint32_t(gx), uint32_t(gy), uint32_t(gz)}, {uint32_t(tx), uint32_t(ty), uint32_t(tz)},
                 groups};
  if (pso) {
    auto it = psoNames_.find(pso);
    d.function = it != psoNames_.end() ? it->second : "(unknown pipeline)";
  }
  profDispatches_.push_back(std::move(d));
}

void GpuContext::profileStop(std::vector<ProfStamp>& stamps, std::vector<ProfDispatch>& dispatches) {
  profiling_ = false;
  profBeginPending_ = false;
  profKernelStamped_ = false;
  stamps.clear();
  dispatches.swap(profDispatches_);
  profDispatches_.clear();
  if (!counterHeap_ || stampCount_ == 0) return;
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  NS::Data* data = static_cast<MTL4::CounterHeap*>(counterHeap_)->resolveCounterRange(NS::Range(0, stampCount_));
  // Counter-heap timestamps tick at queryTimestampFrequency(); sampleTimestamps() pairs the GPU clock with the CPU's
  // (CLOCK_UPTIME_RAW ns), which puts both on one timeline.
  MTL::Timestamp cpuNow = 0, gpuNow = 0;
  device_->sampleTimestamps(&cpuNow, &gpuNow);
  const double freq = static_cast<double>(device_->queryTimestampFrequency());
  const auto* raw = data ? static_cast<const uint64_t*>(data->bytes()) : nullptr;
  const size_t n = data ? data->length() / sizeof(uint64_t) : 0;
  // Apple GPUs write heap timestamps in mach_absolute_time ticks (queryTimestampFrequency() = 24 MHz), while
  // sampleTimestamps() returns nanoseconds; accept either domain, whichever lands near the sampled GPU time.
  const double gpuNowNs = static_cast<double>(gpuNow), shift = static_cast<double>(cpuNow) - gpuNowNs;
  auto toNs = [&](uint32_t i) -> int64_t {
    if (!raw || i >= n || raw[i] == 0 || raw[i] == ~0ull) return -1;
    const double asTicks = static_cast<double>(raw[i]) * 1e9 / freq, asNs = static_cast<double>(raw[i]);
    if (std::fabs(asTicks - gpuNowNs) < 600e9) return static_cast<int64_t>(asTicks + shift);
    if (std::fabs(asNs - gpuNowNs) < 600e9) return static_cast<int64_t>(asNs + shift);
    return -1;
  };
  for (const StampSlots& s : stamps_)
    stamps.push_back({s.tag, toNs(s.begin), s.end == ~0u ? -1 : toNs(s.end)});
  stamps_.clear();
  stampCount_ = 0;
  pool->release();
}

namespace {
std::string jsonStr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += c;
    else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
    else o += c;
  }
  return o + "\"";
}
}  // namespace

std::string GpuContext::inspectJson(const std::vector<std::string>& functions, const std::string& archivePath) {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  std::string o = "{\"device\":{";
  o += "\"name\":" + jsonStr(device_->name()->utf8String());
  if (auto* a = device_->architecture()) o += ",\"architecture\":" + jsonStr(a->name()->utf8String());
  o += ",\"max_threadgroup_memory\":" + std::to_string(device_->maxThreadgroupMemoryLength());
  MTL::Size mt = device_->maxThreadsPerThreadgroup();
  o += ",\"max_threads_per_threadgroup\":[" + std::to_string(mt.width) + "," + std::to_string(mt.height) + "," +
       std::to_string(mt.depth) + "]";
  o += ",\"recommended_max_working_set\":" + std::to_string(device_->recommendedMaxWorkingSetSize());
  o += ",\"unified_memory\":" + std::string(device_->hasUnifiedMemory() ? "true" : "false");
  o += ",\"timestamp_frequency\":" + std::to_string(device_->queryTimestampFrequency());
  o += ",\"metallib\":" + jsonStr(metallibPath());
  o += "},\"functions\":[";
  std::vector<std::string> names = functions;
  if (names.empty()) {
    NS::Array* all = library_->functionNames();
    for (NS::UInteger i = 0; i < all->count(); ++i) names.push_back(all->object<NS::String>(i)->utf8String());
  }
  MTL::BinaryArchive* archive = nullptr;
  if (!archivePath.empty()) {
    auto* bd = MTL::BinaryArchiveDescriptor::alloc()->init();
    NS::Error* err = nullptr;
    archive = device_->newBinaryArchive(bd, &err);
    bd->release();
  }
  bool first = true;
  for (const std::string& name : names) {
    NS::String* nsName = NS::String::string(name.c_str(), NS::UTF8StringEncoding);  // nil for invalid UTF-8
    MTL::Function* fn = nsName ? library_->newFunction(nsName) : nullptr;
    if (!first) o += ",";
    first = false;
    o += "{\"name\":" + jsonStr(name);
    if (!fn) {
      o += ",\"error\":\"not in metallib\"}";
      continue;
    }
    NS::Error* err = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    MTL::ComputePipelineState* pso = device_->newComputePipelineState(fn, &err);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!pso) {
      o += ",\"error\":" + jsonStr(err ? err->localizedDescription()->utf8String() : "pipeline creation failed") + "}";
      fn->release();
      continue;
    }
    o += ",\"max_total_threads_per_threadgroup\":" + std::to_string(pso->maxTotalThreadsPerThreadgroup());
    o += ",\"thread_execution_width\":" + std::to_string(pso->threadExecutionWidth());
    o += ",\"static_threadgroup_memory\":" + std::to_string(pso->staticThreadgroupMemoryLength());
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3f", ms);
    o += ",\"pipeline_create_ms\":" + std::string(buf);
    if (archive) {
      auto* cd = MTL::ComputePipelineDescriptor::alloc()->init();
      cd->setComputeFunction(fn);
      NS::Error* aerr = nullptr;
      bool ok = archive->addComputePipelineFunctions(cd, &aerr);
      cd->release();
      if (!ok) o += ",\"archive_error\":" + jsonStr(aerr ? aerr->localizedDescription()->utf8String() : "failed");
    }
    o += "}";
    pso->release();
    fn->release();
  }
  o += "]";
  if (archive) {
    NS::Error* err = nullptr;
    NS::String* path = NS::String::string(archivePath.c_str(), NS::UTF8StringEncoding);
    bool ok = archive->serializeToURL(NS::URL::fileURLWithPath(path), &err);
    o += ",\"archive\":" + (ok ? jsonStr(archivePath)
                               : "{\"error\":" + jsonStr(err ? err->localizedDescription()->utf8String() : "failed") + "}");
    archive->release();
  }
  o += "}";
  pool->release();
  return o;
}

namespace {
std::unique_ptr<GpuContext>& sharedSlot() {
  // Intentionally leaked at exit: Metal objects must not be torn down during
  // static destruction (Storage destructors may still run after).
  static auto* slot = new std::unique_ptr<GpuContext>();
  return *slot;
}
}  // namespace

Expected<GpuContext*> GpuContext::shared() {
  auto& slot = sharedSlot();
  if (!slot) {
    auto g = create();
    if (!g) return g.status();
    slot = g.take();
  }
  return slot.get();
}

GpuContext* GpuContext::existing() { return sharedSlot().get(); }

void GpuContext::adopt(MTL::Buffer* b) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (adopted_[b]++ == 0) {
    residency_->addAllocation(b);
    residencyDirty_ = true;
  }
}

void* GpuContext::contents(void* mtlBuffer) { return static_cast<MTL::Buffer*>(mtlBuffer)->contents(); }

void* gpuWrapNoCopy(GpuContext* gpu, void* data, size_t size) { return gpu->wrapNoCopy(data, size); }

void gpuReleaseBuffer(void* mtlBuffer) {
  if (GpuContext* g = GpuContext::existing()) g->releaseBuffer(static_cast<MTL::Buffer*>(mtlBuffer));
  else static_cast<MTL::Buffer*>(mtlBuffer)->release();
}

void gpuSync(GpuContext& gpu, SyncKind kind) {
  if (kind == SyncKind::GpuToCpu) gpu.commitAndWait();
}

void gpuFinish(GpuContext& gpu) { gpu.commitAndWait(); }

uint64_t gpuAllocatedBytes() {
  GpuContext* g = GpuContext::existing();
  return g ? g->device()->currentAllocatedSize() : 0;
}

}  // namespace lk::rt
