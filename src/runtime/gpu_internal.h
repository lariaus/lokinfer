// Metal 4 GPU context (internal to the runtime and GPU kernels).
//
// One command queue, a reused allocator/command buffer, one compute encoder
// per GPU segment, an argument table and a residency set holding every
// buffer the runtime creates. Kernel parameters go in a per-segment ring
// buffer (Metal 4 has no setBytes). Baseline barrier policy: a dispatch->
// dispatch barrier before every dispatch after the first in an encoder
// (correct for any dependency; dependency-level barriers come later).
#pragma once

#include <cstddef>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"

namespace MTL {
class Device;
class Buffer;
class ComputePipelineState;
class ResidencySet;
class SharedEvent;
class Library;
class IndirectCommandBuffer;
class Allocation;
}  // namespace MTL
namespace MTL4 {
class CommandQueue;
class CommandAllocator;
class CommandBuffer;
class CommitOptions;
class CommitFeedback;
class ComputeCommandEncoder;
class ArgumentTable;
}  // namespace MTL4

namespace lk::rt {

class GpuReplay;  // gpu_replay.h

// Overrides where the kernel metallib is loaded from (default: build-time
// path, or $LOKINFER_METALLIB).
void setMetallibPath(std::string path);

class GpuContext {
 public:
  ~GpuContext();
  // Process-wide context (one device, queue and residency set shared by every
  // executable and tensor). Created on first use.
  static Expected<GpuContext*> shared();
  // Whether a shared context already exists (without creating one).
  static GpuContext* existing();

  MTL::Device* device() const { return device_; }
  // Cached pipeline for a kernel function in the metallib (null + error log if missing).
  MTL::ComputePipelineState* pipeline(const std::string& function);

  // ---- encoding (used by GPU kernel run functions)
  // Opens the command buffer / encoder if needed and inserts the barrier policy.
  MTL4::ComputeCommandEncoder* beginDispatch();
  // Binds buffer index `slot` to a GPU address.
  void bind(uint32_t slot, MTL::Buffer* buffer, uint64_t offset);
  // Copies `size` bytes of parameters into the ring buffer and binds them at `slot`.
  void bindParams(uint32_t slot, const void* data, size_t size);
  // Dispatches with the current argument table.
  void dispatch(MTL::ComputePipelineState* pso, uint64_t gridX, uint64_t gridY, uint64_t gridZ, uint64_t tgX,
                uint64_t tgY, uint64_t tgZ);
  void dispatchThreadgroups(MTL::ComputePipelineState* pso, uint64_t groupsX, uint64_t groupsY, uint64_t groupsZ,
                            uint64_t tgX, uint64_t tgY, uint64_t tgZ);

  // Barrier policy for the next dispatch: the executable clears it when the
  // next kernel's buffer ranges don't conflict with any dispatch since the
  // last barrier. Every later dispatch (e.g. a kernel's 2nd internal
  // dispatch) defaults back to a barrier.
  void setNextBarrier(bool needed) { nextBarrier_ = needed; }
  // Incremented whenever a segment is committed (hazard tracking resets).
  uint64_t segmentId() const { return segmentId_; }

  // Library kernels (MPS) that encode their own dispatch call this afterwards;
  // our argument table is re-bound before the next dispatch.
  void noteExternalDispatch() {
    ++dispatchesInEncoder_;
    tableDirty_ = true;
    if (profiling_) profileLogDispatch(nullptr, 0, 0, 0, 0, 0, 0, false);
    if (recorder_) recorderFail("a kernel encoded its own GPU work (MPS)");
  }

  // ---- record/replay (gpu_replay.h). While a recorder is set, every dispatch is also teed into it.
  void setRecorder(GpuReplay* r) { recorder_ = r; }
  GpuReplay* recorder() const { return recorder_; }
  // The hazard tracker's decision for the next kernel (the executable sets it before each GPU kernel).
  bool nextBarrierPending() const { return nextBarrier_; }
  // Encodes `count` commands of an indirect command buffer into the current segment, after a barrier when the
  // encoder already holds work; later dispatches get a barrier against it.
  void executeIndirect(MTL::IndirectCommandBuffer* icb, uint32_t count);
  // `pso`'s function compiled with supportIndirectCommandBuffers (cached; made resident, as ICBs require).
  MTL::ComputePipelineState* icbPipeline(MTL::ComputePipelineState* pso);
  // Adds an allocation (ICB, buffer) to the residency set.
  void makeResident(MTL::Allocation* a);

  // Takes ownership of one +1 reference to an Objective-C object (CFRelease'd
  // after the current segment completes).
  void keepAlive(void* retainedObject) { keepAlive_.push_back(retainedObject); }

  // Ends the segment, commits and waits for completion of everything committed (no-op if nothing encoded or
  // in flight).
  void commitAndWait();
  // commitAndWait, then commits pending residency-set removals now. The set keeps every allocation it holds alive
  // until it is committed, and it was only committed at the next encode: a freed model's weights (3.6 GB for
  // Qwen3-ASR) stayed allocated until the next GPU call (2026-11-06; on the iPhone, through a following CPU-only
  // session). Call after freeing big buffers outside a run.
  void trim();
  // The context is one shared encoder: every thread that encodes, commits or waits holds this (recursive: kernels
  // commit inside a run). Executable::run holds it for the whole call; the entry points below lock it themselves.
  std::recursive_mutex& mutex() { return mu_; }
  // Ends the segment and commits it without waiting, so the CPU can work while the GPU runs it. At most one
  // segment is in flight: the next segment's first dispatch (or commitAndWait) waits for it, then releases
  // its deferred buffers / objects.
  void commitAsync();
  // Estimated GPU work (FLOPs) a kernel just encoded. With a work budget (iOS), the open command buffer is committed
  // once the work encoded into it passes the budget: iOS aborts a command buffer that runs too long ("GPU timeout",
  // MTL4CommandQueueErrorDomain 1), e.g. a 1473-row Qwen3-ASR prefill, ~0.6 s of GPU work per buffer when only the
  // params ring split them. Decode GEMVs report ~no work: no extra commits.
  void addWork(double flops);
  bool hasOpenSegment() const { return cmdOpen_; }
  // A committed command buffer failed (GPU fault, timeout, out of memory; reported by Metal's commit feedback):
  // returns the error once and clears it. Waits stop blocking while an error is pending (a failed command buffer
  // may never signal its event, which used to hang the caller forever).
  std::string takeGpuError();
  // Command buffers that may exist at once (open + in flight); see slots_.
  size_t maxSegments() const { return slots_.size(); }
  // Changes it (1-4; drains the GPU first). Tests and A/B runs; the default comes from $LOKINFER_GPU_INFLIGHT.
  void setMaxSegments(size_t n);

  // ---- memory
  MTL::Buffer* wrapNoCopy(void* data, size_t size);  // retained; added to the residency set
  void releaseBuffer(MTL::Buffer* b);
  // Same deferred release for any resident allocation (indirect command buffers).
  void releaseAllocation(MTL::Allocation* a);
  // Adopts a foreign MTLBuffer (e.g. a torch MPS tensor's) into the residency set.
  void adopt(MTL::Buffer* b);
  static void* contents(void* mtlBuffer);

  // ---- profiling (prof-lokinfer, docs/profiling.md). Inert unless an executable records a profile: the only
  // cost when off is one predictable branch per dispatch.
  // Starts recording a call: per-kernel GPU timestamps in a Metal 4 counter heap (room for `maxStamps`) and a
  // log of every dispatch. Returns false when timestamps are unavailable (the dispatch log still works).
  bool profileStart(uint32_t maxStamps);
  // Tags the following dispatches with kernel `tag`; the kernel's first dispatch writes its begin timestamp right
  // after its barrier, profileKernelEnd() the end timestamp after its last dispatch.
  void profileKernelBegin(int32_t tag) {
    profTag_ = tag;
    profBeginPending_ = profiling_;
  }
  void profileKernelEnd();
  // CPU time spent waiting for the in-flight segment (waitInflight) since the last call: the part of a GPU kernel's
  // "encode" wall time that is really the CPU blocked on the previous segment (one segment in flight at a time).
  int64_t profileTakeWaitNs() {
    int64_t w = profWaitNs_;
    profWaitNs_ = 0;
    return w;
  }
  struct ProfStamp {
    int32_t tag;
    int64_t beginNs, endNs;  // CLOCK_UPTIME_RAW nanoseconds; -1 when missing
  };
  struct ProfDispatch {
    int32_t tag;
    std::string function;  // metallib function, or "(MPS)" for library-encoded work
    uint32_t grid[3], group[3];
    bool threadgroups;  // grid counts threadgroups (dispatchThreadgroups) instead of threads
  };
  // Ends the recording (call after gpuFinish): resolves the timestamps.
  void profileStop(std::vector<ProfStamp>& stamps, std::vector<ProfDispatch>& dispatches);
  // Device info + pipeline statistics of metallib functions (all when `functions` is empty) as JSON; with
  // `archivePath`, also serializes a binary archive holding their native GPU code.
  std::string inspectJson(const std::vector<std::string>& functions, const std::string& archivePath);

 private:
  GpuContext() = default;
  static Expected<std::unique_ptr<GpuContext>> create();
  void commitResidency();
  void endAndCommit();
  double workBudget_ = 0;  // FLOPs per command buffer (0 = no budget); see addWork
  double workEncoded_ = 0;
  void waitInflight();
  void waitEvent(uint64_t value);

  MTL::Device* device_ = nullptr;
  MTL::Library* library_ = nullptr;
  MTL4::CommandQueue* queue_ = nullptr;
  MTL4::CommandAllocator* allocator_ = nullptr;
  MTL4::CommandBuffer* cmd_ = nullptr;
  MTL4::ComputeCommandEncoder* encoder_ = nullptr;
  MTL4::ArgumentTable* table_ = nullptr;
  MTL::ResidencySet* residency_ = nullptr;
  MTL::SharedEvent* event_ = nullptr;
  uint64_t eventValue_ = 0;
  bool residencyDirty_ = false;
  bool cmdOpen_ = false;
  bool tableDirty_ = false;
  bool nextBarrier_ = true;
  bool lastBarrier_ = false;  // the barrier decision of the latest beginDispatch (recorded with the dispatch)
  uint64_t segmentId_ = 0;
  uint32_t dispatchesInEncoder_ = 0;
  // Parameter ring (shared memory, reset per segment).
  MTL::Buffer* params_ = nullptr;
  size_t paramsOffset_ = 0;
  size_t paramsCapacity_ = 0;
  std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
  std::unordered_map<MTL::Buffer*, int> adopted_;
  std::vector<void*> keepAlive_;
  std::vector<MTL::Allocation*> pendingRelease_;
  void* segmentPool_ = nullptr;  // NS::AutoreleasePool* spanning one segment
  // The committed-but-unfinished segment (commitAsync) and what to release once it completes.
  // Committed-but-unfinished segments, oldest first (commitAsync), each with what to release once it completes.
  // At most `slots_.size()` command buffers exist at a time (open + in flight): $LOKINFER_GPU_INFLIGHT, default 2
  // (1 = the open segment waits for the previous one); each slot owns an allocator, a command buffer and a params ring.
  struct Inflight {
    uint64_t value;  // event value signaled when it completes
    std::vector<void*> keepAlive;
    std::vector<MTL::Allocation*> release;
    void* pool;
  };
  std::vector<Inflight> inflight_;
  struct SegSlot {
    MTL4::CommandAllocator* allocator;
    MTL4::CommandBuffer* cmd;
    MTL::Buffer* params;
  };
  std::vector<SegSlot> slots_;
  size_t nextSlot_ = 0;
  size_t lastSlot_ = 0;  // slot of the most recent segment (reused while nothing is in flight)
  bool queueBarrier_ = false;
  std::function<void(MTL4::CommitFeedback*)> onFeedback_;  // attached to every commit's options (endAndCommit)
  std::atomic<bool> gpuFailed_{false};
  std::mutex gpuErrorMu_;
  // Commit feedback arrives on Metal's thread, possibly after the event signal a wait returned on: commitAndWait waits
  // until every commit so far has its feedback, so a failure is reported by the call that caused it (never a later
  // call, never lost). Guarded by gpuErrorMu_.
  std::atomic<uint64_t> commits_{0};
  std::atomic<uint64_t> feedbacks_{0};
  std::condition_variable feedbackCv_;
  void waitFeedback();
  // Sentinel: a 1-thread dispatch at the end of every command buffer copies req[slot] to res[slot] (one shared
  // buffer: 4096 request words, then 4096 result words). After the completion event, res == req for every committed
  // segment means success with no feedback wait; a missing marker = a failed buffer: wait for its feedback (error).
  static constexpr uint32_t kSentinelSlots = 4096;
  MTL::Buffer* sentinel_ = nullptr;
  MTL::ComputePipelineState* sentinelPso_ = nullptr;
  std::vector<std::pair<uint32_t, uint32_t>> pendingSentinels_;  // (slot, marker) committed since the last check
  void encodeSentinel();
  void checkSentinels();
  std::recursive_mutex mu_;
  std::string gpuError_;
  uint64_t skipUpTo_ = 0;  // events up to this value belong to failed work: never wait for them  // the open segment must wait for the queue's earlier (in-flight) work
  void waitOldest();
  void retire(Inflight& f);
  // Profiling state (see profileStart).
  void profileStamp(int32_t tag, bool begin);
  void profileLogDispatch(MTL::ComputePipelineState* pso, uint64_t gx, uint64_t gy, uint64_t gz, uint64_t tx,
                          uint64_t ty, uint64_t tz, bool groups);
  bool profiling_ = false;
  int64_t profWaitNs_ = 0;
  bool profBeginPending_ = false;
  bool profKernelStamped_ = false;
  int32_t profTag_ = -1;
  void* counterHeap_ = nullptr;  // MTL4::CounterHeap*
  uint32_t counterCapacity_ = 0;
  uint32_t stampCount_ = 0;
  struct StampSlots {
    int32_t tag;
    uint32_t begin, end;  // counter heap indices; ~0u when not written
  };
  std::vector<StampSlots> stamps_;
  std::vector<ProfDispatch> profDispatches_;
  std::unordered_map<const void*, std::string> psoNames_;
  // record/replay
  void recorderFail(const char* why);
  GpuReplay* recorder_ = nullptr;
  std::unordered_map<const void*, MTL::ComputePipelineState*> icbPipelines_;
};

// Hooks used by the executable.
void* gpuWrapNoCopy(GpuContext* gpu, void* data, size_t size);
void gpuReleaseBuffer(void* mtlBuffer);
void gpuSync(GpuContext& gpu, SyncKind kind);
void gpuFinish(GpuContext& gpu);
// MTLDevice.currentAllocatedSize of the shared context (0 when none exists yet).
uint64_t gpuAllocatedBytes();

}  // namespace lk::rt
