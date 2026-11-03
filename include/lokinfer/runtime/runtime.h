// Runtime: lowers a compiled Model into an Executable and runs it
// (see docs/impl-notes/design-runtime-and-kernels.md §3).
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"
#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

// Page-aligned (16 KiB) host memory; GPU-visible when a GpuContext wraps it
// as a no-copy shared MTLBuffer.
struct Storage {
  void* data = nullptr;
  size_t size = 0;
  void* mtlBuffer = nullptr;  // MTL::Buffer*, owned (released in destructor)
  bool ownsData = true;       // false: wraps memory owned by `keep` (mmapped weights, caller arrays)
  std::shared_ptr<const void> keep;
  ~Storage();
};
std::shared_ptr<Storage> allocateStorage(size_t bytes, GpuContext* gpu);
// GPU-visible view of existing memory without copying: wraps the page range
// enclosing [data, data + bytes) (unaligned data is fine; the slot uses an
// offset). `keep` must keep the memory alive; nullptr when it can't be wrapped
// (no GPU, or no owner to keep the pages mapped). `offset` receives
// data - storage->data.
std::shared_ptr<Storage> wrapStorage(void* data, size_t bytes, std::shared_ptr<const void> keep, GpuContext* gpu,
                                     uint64_t& offset);

// A borrowed tensor (inputs, external constants). Scalars: empty shape.
struct TensorView {
  void* data = nullptr;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  void* mtlBuffer = nullptr;  // optional; GPU kernels need it (or a copy is made)
  uint64_t mtlOffset = 0;
  // Keeps `data` alive. With an owner, GPU models wrap caller memory with no
  // copy; without one they must copy (the runtime can't keep it mapped).
  std::shared_ptr<const void> owner;
};

struct OwnedTensor {
  std::shared_ptr<Storage> storage;
  size_t byteOffset = 0;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  void* data() const { return static_cast<char*>(storage->data) + byteOffset; }
};

struct ExternalBinding {
  std::string name;
  TensorView view;
  std::shared_ptr<void> keepAlive;  // keeps the owner (e.g. a Python tensor) alive
};

struct RuntimeOptions {
  bool enableGpu = true;  // create a GpuContext if the model has GPU kernels
  // GPU record/replay through an indirect command buffer (docs/runtime.md): -1 = $LOKINFER_GPU_REPLAY (default off),
  // 0 off, 1 on for eligible (static-shape, single-segment) models.
  int gpuReplay = -1;
};

// Structured profiling (prof-lokinfer; docs/profiling.md). Off by default; when off it costs one branch per
// instruction.
struct ProfileOptions {
  int level = 0;              // 0 off; 1 records every exec call (kernels, GPU timestamps, dispatches, syncs, arenas)
  bool gpuSerialize = false;  // commit + wait after every GPU kernel (no CPU/GPU overlap; the GPU times stay exact)
};

// GPU device info and pipeline statistics of metallib functions (all when `functions` is empty), as JSON; with
// `archivePath`, also writes a binary archive holding their native GPU code. {"error": ...} without a GPU.
std::string gpuInspectJson(const std::vector<std::string>& functions, const std::string& archivePath);

class Executable {
 public:
  ~Executable();
  // `model` must be in final (post-pipeline) buffer form. Runs init.
  static Expected<std::unique_ptr<Executable>> create(std::shared_ptr<const Model> model,
                                                      std::vector<ExternalBinding> externals,
                                                      const RuntimeOptions& opts = {});

  // Synchronous call. Inputs are checked against the exec signature.
  Status run(std::span<const TensorView> inputs, std::vector<OwnedTensor>& outputs);
  // Same, writing output k into callerOutputs[k] when its `data` is set (empty span or null data: allocate as usual).
  // The model computes straight into the caller's memory (no copy) when output k is (a zero-offset view of) one of
  // the call's own output allocations, and the caller's memory is GPU-visible or can be wrapped (it has an owner);
  // otherwise the result is copied into it. dtype and shape must match the output exactly. outputs[k].storage is
  // null for those outputs (the data is in the caller's memory).
  Status run(std::span<const TensorView> inputs, std::vector<OwnedTensor>& outputs,
             std::span<const TensorView> callerOutputs);

  bool usesGpu() const;

  // mem.plan: results are cached per plan, keyed by the values of the symbols the plan depends on (a hit skips size
  // evaluation and planning); the plans of a call share one arena per pool (0: CPU-touched, 1: gpu_only).
  struct MemPlanStats {
    int64_t hits = 0, misses = 0, overflows = 0;  // overflow: a later plan didn't fit the pool's arena this call
    size_t arenaBytes[2] = {0, 0};
  };
  MemPlanStats memPlanStats() const;

  // Host pointer to exec state value i (an init result) after init ran (bake-init reads it; GPU work is finished).
  const void* stateData(size_t i) const;
  // "none", "direct", "direct (replay: <why not>)" or "replay (...)": how GPU work is submitted.
  std::string gpuStrategy() const;
  const Model& model() const { return *model_; }

  // Profiling: see ProfileOptions. takeProfileJson returns the calls recorded since the last take (and forgets
  // them): {"version", "ops": static kernel table, "calls": [...]} (schema in docs/profiling.md).
  void setProfile(const ProfileOptions& opts);
  std::string takeProfileJson();

 private:
  struct Impl;
  explicit Executable(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  std::shared_ptr<const Model> model_;
};

}  // namespace lk::rt
