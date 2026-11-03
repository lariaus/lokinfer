// GPU record/replay through a Metal indirect command buffer (see gpu_replay.h).
#include "gpu_replay.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace lk::rt {

namespace {
bool verbose() {
  static const bool v = [] {
    const char* e = std::getenv("LOKINFER_GPU_REPLAY");
    return e && std::strcmp(e, "verbose") == 0;
  }();
  return v;
}
}  // namespace

GpuReplay::~GpuReplay() {
  if (gpu_ && gpu_->recorder() == this) gpu_->setRecorder(nullptr);
  if (icb_) gpu_->releaseAllocation(icb_);
  if (paramsBuf_) gpu_->releaseBuffer(paramsBuf_);
}

void GpuReplay::beginRecord() {
  if (state_ != State::Idle) return;
  state_ = State::Recording;
  for (Binding& b : table_) b = Binding{};
  cmds_.clear();
  params_.clear();
  kernels_.clear();
  gpu_->setRecorder(this);
}

void GpuReplay::fail(const char* why) {
  if (state_ == State::Off) return;
  state_ = State::Off;
  offReason_ = why;
  if (verbose()) std::fprintf(stderr, "lokinfer: GPU replay off: %s\n", why);
  if (gpu_->recorder() == this) gpu_->setRecorder(nullptr);
  cmds_.clear();
  params_.clear();
  kernels_.clear();
}

void GpuReplay::kernelBegin(uint32_t instr, const BufRef* bufs, uint32_t n) {
  if (state_ != State::Recording) return;
  if (kernels_.size() <= instr) kernels_.resize(instr + 1);
  KernelRec& k = kernels_[instr];
  if (k.firstCmd >= 0 || !k.bufs.empty()) return fail("a GPU kernel ran twice in one call");
  k.bufs.assign(bufs, bufs + n);
  k.patches.assign(n, {});
  curInstr_ = instr;
  curBufs_.assign(bufs, bufs + n);
}

void GpuReplay::onBind(uint32_t slot, MTL::Buffer* buf, uint64_t off) {
  if (slot >= kSlots) return fail("binding slot >= 16");
  Binding b{buf, off, false, -1, -1, 0};
  // Which argument of the current kernel is this? Exact (buffer, offset) first, else the same buffer at the nearest
  // lower offset (a kernel binding a sub-range of an argument).
  int best = -1;
  for (size_t i = 0; i < curBufs_.size(); ++i) {
    if (curBufs_[i].mtlBuffer != buf || curBufs_[i].offset > off) continue;
    if (best < 0 || curBufs_[i].offset > curBufs_[best].offset) best = static_cast<int>(i);
  }
  if (best >= 0) {
    b.instr = static_cast<int32_t>(curInstr_);
    b.arg = best;
    b.delta = off - curBufs_[best].offset;
  }
  table_[slot] = b;
}

void GpuReplay::onParams(uint32_t slot, const void* data, size_t size) {
  if (slot >= kSlots) return fail("binding slot >= 16");
  size_t at = (params_.size() + 15) & ~size_t(15);
  params_.resize(at + size);
  std::memcpy(params_.data() + at, data, size);
  table_[slot] = Binding{nullptr, at, true, -1, -1, 0};
}

void GpuReplay::onDispatch(MTL::ComputePipelineState* pso, uint64_t gx, uint64_t gy, uint64_t gz, uint64_t tx,
                           uint64_t ty, uint64_t tz, bool threadgroups, bool barrier) {
  if (state_ != State::Recording) return;
  if (curInstr_ == ~0u) return fail("dispatch outside a kernel");
  Cmd c{};
  c.pso = pso;
  for (uint32_t s = 0; s < kSlots; ++s) {
    c.b[s] = table_[s];
    if (table_[s].buf || table_[s].params) c.mask |= 1u << s;
  }
  c.grid[0] = gx, c.grid[1] = gy, c.grid[2] = gz;
  c.group[0] = tx, c.group[1] = ty, c.group[2] = tz;
  c.threadgroups = threadgroups;
  c.barrier = barrier && !cmds_.empty();
  KernelRec& k = kernels_[curInstr_];
  if (k.firstCmd < 0) {
    k.firstCmd = static_cast<int32_t>(cmds_.size());
    k.firstHasBarrier = c.barrier;
  }
  cmds_.push_back(c);
}

void GpuReplay::endRecord(bool callOk) {
  if (gpu_->recorder() == this) gpu_->setRecorder(nullptr);
  if (state_ != State::Recording) return;
  if (!callOk) return fail("the recording call failed");
  if (cmds_.empty()) return fail("no GPU dispatches");
  if (!build()) return;
  state_ = State::Ready;
  if (verbose()) std::fprintf(stderr, "lokinfer: GPU replay recorded %zu commands\n", cmds_.size());
}

bool GpuReplay::build() {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* dev = gpu_->device();
  if (!params_.empty()) {
    paramsBuf_ = dev->newBuffer(params_.size(), MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    if (!paramsBuf_) {
      pool->release();
      fail("cannot allocate the params buffer");
      return false;
    }
    std::memcpy(paramsBuf_->contents(), params_.data(), params_.size());
    gpu_->makeResident(paramsBuf_);
  }
  auto* d = MTL::IndirectCommandBufferDescriptor::alloc()->init();
  d->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch | MTL::IndirectCommandTypeConcurrentDispatchThreads);
  d->setInheritPipelineState(false);
  d->setInheritBuffers(false);
  d->setMaxKernelBufferBindCount(kSlots);
  icb_ = dev->newIndirectCommandBuffer(d, cmds_.size(), MTL::ResourceStorageModeShared);
  d->release();
  if (!icb_) {
    pool->release();
    fail("cannot create the indirect command buffer");
    return false;
  }
  gpu_->makeResident(icb_);
  for (size_t i = 0; i < cmds_.size(); ++i) {
    const Cmd& c = cmds_[i];
    MTL::ComputePipelineState* pso = gpu_->icbPipeline(c.pso);
    if (!pso) {
      pool->release();
      fail("cannot create an ICB-capable pipeline");
      return false;
    }
    MTL::IndirectComputeCommand* ic = icb_->indirectComputeCommand(i);
    ic->setComputePipelineState(pso);
    for (uint32_t s = 0; s < kSlots; ++s) {
      if (!(c.mask & (1u << s))) continue;
      const Binding& b = c.b[s];
      ic->setKernelBuffer(b.params ? paramsBuf_ : b.buf, b.off, s);
      if (b.arg >= 0) kernels_[b.instr].patches[b.arg].push_back({static_cast<uint32_t>(i), s, b.delta});
    }
    const MTL::Size grid(c.grid[0], c.grid[1], c.grid[2]), group(c.group[0], c.group[1], c.group[2]);
    if (c.threadgroups) ic->concurrentDispatchThreadgroups(grid, group);
    else ic->concurrentDispatchThreads(grid, group);
    if (c.barrier) ic->setBarrier();
  }
  pool->release();
  params_.clear();
  params_.shrink_to_fit();
  return true;
}

void GpuReplay::replayKernel(uint32_t instr, const BufRef* bufs, uint32_t n, bool needBarrier) {
  if (instr >= kernels_.size()) return;
  KernelRec& k = kernels_[instr];
  if (k.firstCmd < 0) return;  // recorded no dispatch
  for (uint32_t i = 0; i < n && i < k.bufs.size(); ++i) {
    if (bufs[i].mtlBuffer == k.bufs[i].mtlBuffer && bufs[i].offset == k.bufs[i].offset) continue;
    k.bufs[i] = bufs[i];
    for (const Patch& p : k.patches[i])
      icb_->indirectComputeCommand(p.cmd)->setKernelBuffer(static_cast<MTL::Buffer*>(bufs[i].mtlBuffer),
                                                          bufs[i].offset + p.delta, p.slot);
  }
  if (needBarrier && !k.firstHasBarrier && k.firstCmd > 0) {  // buffers now alias differently than when recorded
    icb_->indirectComputeCommand(k.firstCmd)->setBarrier();
    k.firstHasBarrier = true;
  }
}

void GpuReplay::submit() { gpu_->executeIndirect(icb_, static_cast<uint32_t>(cmds_.size())); }

}  // namespace lk::rt
