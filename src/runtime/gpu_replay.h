// GPU record/replay with a Metal indirect command buffer (ICB): an opt-in per-executable strategy for static-shape
// GPU models (docs/runtime.md "Record / replay").
//
// The first exec call runs normally while GpuContext tees every dispatch (pipeline, bound buffers, parameters,
// grid, barrier) into a GpuRecorder. Afterwards the commands are written once into an ICB (parameters copied into
// a buffer the replay owns). Every later call runs the interpreter as usual for non-GPU work (views, allocations,
// syncs), but a GPU kernel only compares its argument buffers with the recorded ones and re-binds the ICB commands
// that use an argument that moved (inputs, fresh outputs: ~0.05 us per binding); the last GPU kernel encodes a
// single executeCommandsInBuffer. Any recorded dispatch the replay can't reproduce (MPS encodes its own work, a
// params-ring flush) turns the strategy off for that executable.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu_internal.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

class GpuReplay {
 public:
  explicit GpuReplay(GpuContext* gpu) : gpu_(gpu) {}
  ~GpuReplay();

  bool ready() const { return state_ == State::Ready; }
  bool off() const { return state_ == State::Off; }
  const std::string& offReason() const { return offReason_; }
  uint32_t numCommands() const { return static_cast<uint32_t>(cmds_.size()); }

  // Recording (the first exec call): tees the GpuContext's dispatches; `kernelBegin` before each GPU kernel runs.
  void beginRecord();
  void kernelBegin(uint32_t instr, const BufRef* bufs, uint32_t n);
  // Ends the recording; on success builds the ICB (ready() afterwards), else turns the strategy off.
  void endRecord(bool callOk);

  // Replay: instead of running GPU kernel `instr`, patch the commands whose argument buffers moved. `needBarrier`:
  // the hazard tracker wants a barrier before this kernel (added to its first command if the recording had none).
  void replayKernel(uint32_t instr, const BufRef* bufs, uint32_t n, bool needBarrier);
  // Encodes the whole recorded command list into the current segment.
  void submit();

  // GpuContext tee hooks (recording only).
  void onBind(uint32_t slot, MTL::Buffer* buf, uint64_t off);
  void onParams(uint32_t slot, const void* data, size_t size);
  void onDispatch(MTL::ComputePipelineState* pso, uint64_t gx, uint64_t gy, uint64_t gz, uint64_t tx, uint64_t ty,
                  uint64_t tz, bool threadgroups, bool barrier);
  void fail(const char* why);

 private:
  enum class State { Idle, Recording, Ready, Off };
  struct Binding {
    MTL::Buffer* buf = nullptr;
    uint64_t off = 0;
    bool params = false;  // `off` is into the replay's params buffer
    int32_t instr = -1, arg = -1;  // the kernel argument this binding came from (-1: fixed, e.g. kernel state)
    uint64_t delta = 0;            // binding offset - argument offset
  };
  static constexpr uint32_t kSlots = 16;
  struct Cmd {
    MTL::ComputePipelineState* pso;
    Binding b[kSlots];
    uint32_t mask = 0;
    uint64_t grid[3], group[3];
    bool threadgroups, barrier;
  };
  struct Patch {
    uint32_t cmd, slot;
    uint64_t delta;
  };
  struct KernelRec {
    std::vector<BufRef> bufs;                  // argument buffers at the last call
    std::vector<std::vector<Patch>> patches;   // per argument
    int32_t firstCmd = -1;
    bool firstHasBarrier = false;
  };
  bool build();

  GpuContext* gpu_;
  State state_ = State::Idle;
  std::string offReason_;
  // recording
  Binding table_[kSlots];
  uint32_t curInstr_ = ~0u;
  std::vector<BufRef> curBufs_;
  std::vector<Cmd> cmds_;
  std::vector<uint8_t> params_;
  std::vector<KernelRec> kernels_;  // by instruction index
  // replay
  MTL::Buffer* paramsBuf_ = nullptr;
  MTL::IndirectCommandBuffer* icb_ = nullptr;
};

}  // namespace lk::rt
