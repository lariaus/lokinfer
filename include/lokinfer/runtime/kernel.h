// Kernel ABI + registry (see docs/impl-notes/design-runtime-and-kernels.md §1).
//
// Argument convention for every kernel (no per-kernel signature in C++):
//   ptrs / bufs : operands in order, then outs in order (scalar operands
//                 point at their 8-byte host slot)
//   ints        : every operand's shape dims, then every out's shape dims,
//                 then integer/bool/int-list/dim attrs in attribute-name order
//                 (structural attrs - nohoist, inplace, output, arena, device,
//                 persistent, inout, gpu_only - are never passed; see
//                 isStructuralAttr in lokinfer/ir/ir.h)
//   floats      : float attrs in attribute-name order
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"

namespace lk::rt {

enum class Device : uint8_t { Cpu, Gpu };

struct BufRef {
  void* mtlBuffer;  // MTL::Buffer* (GPU kernels), nullptr for CPU-only storage
  uint64_t offset;  // bytes into mtlBuffer
};

struct KernelArgs {
  void* const* ptrs;      // CPU-visible pointers (unified memory: valid for GPU storage too)
  const BufRef* bufs;     // GPU kernels: buffer + offset
  const int64_t* ints;
  const double* floats;
  const void* state;      // from prepare()
  uint32_t numPtrs;
  uint32_t numInts;
};

class GpuContext;  // runtime/gpu.h

struct ExecCtx {
  GpuContext* gpu = nullptr;
  // A kernel that detects invalid input (data-dependent, e.g. an id out of
  // range) sets this instead of producing garbage; the executable turns it
  // into an error Status right after the kernel returns. Static string only.
  const char* error = nullptr;
};

struct PrepareCtx {
  const Model* model;
  GpuContext* gpu = nullptr;
};

using PrepareFn = Status (*)(PrepareCtx&, const Op&, void** state);
using RunFn = void (*)(ExecCtx&, const KernelArgs&);
using DestroyFn = void (*)(void* state);

struct KernelEntry {
  const char* name;
  Device device;
  PrepareFn prepare;  // may be null
  RunFn run;
  DestroyFn destroy;  // may be null
};

// Registers every built-in kernel (idempotent). Explicit rather than static
// constructors: static libraries drop unreferenced registration objects.
void registerBuiltinKernels();
void registerKernel(const KernelEntry& e);
const KernelEntry* findKernel(std::string_view name);
// Names of every registered kernel (sorted).
std::vector<std::string> registeredKernels();

}  // namespace lk::rt
