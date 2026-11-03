// IR verifier (MLIR-style: not a pass). Runs when a model is loaded/built and after every pass (runPasses); it
// never changes the IR. No shape inference; see design-compiler-passes.md.
#pragma once

#include <vector>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"

namespace lk {

struct VerifyOptions {
  // Require buffer form (no tensor-typed values left) even if the model doesn't look bufferized; the executor sets
  // it. A model already in buffer form (a kernel with outs, a buffer.free or a mem.plan) always gets this check.
  bool requireBuffers = false;
};

// Structural checks (SSA order, op shapes, symbol binding, buffer form) + device boundaries (verifyDevices).
Status verifyModel(const Model& m, const VerifyOptions& opts = {});

// Explicit CPU/GPU boundaries only (src/ir/verify_devices.cpp, docs/verifier.md).
Status verifyDevices(const Model& m);

// Appends the ids of all symbols referenced by `d` (with duplicates).
void collectSymbols(Dim d, std::vector<int64_t>& out);

}  // namespace lk
