// Shared analyses for buffer-form regions (internal to the compiler).
#pragma once

#include <string_view>
#include <utility>
#include <vector>

#include "lokinfer/ir/ir.h"

namespace lk {

// isGpuKernel (lokinfer/ir/ir.h) under the name the passes use.
inline bool isGpuKernelName(std::string_view name) { return isGpuKernel(name); }

// view result -> alias root (non-views map to themselves). Dense, indexed by value id.
struct Aliases {
  std::vector<ValueId> root;  // kNoValue = own root
  ValueId rootOf(ValueId v) const { return root[v] == kNoValue ? v : root[v]; }
};
Aliases computeAliases(const Model& m, const Region& r);

// Ops of a buffer-form region that can run once in init. Two notions:
// - memory-hoistable values: constants, static non-output allocs, constant-
//   offset views of those (the *memory* can be set up once);
// - constant contents: a const, or a buffer all of whose writers are
//   hoisted kernels.
// A kernel is hoisted iff it's not `nohoist`, all operands have constant
// contents and all outs are memory-hoistable buffers that no per-call kernel
// writes (fixpoint). Scratch buffers written per call keep a hoistable alloc
// (pure memory) but their writers/readers stay in exec.
struct Hoistability {
  std::vector<bool> op;                   // per op index
  std::vector<uint8_t> values;            // per value id: result of a hoisted op
  std::vector<uint8_t> initWrittenRoots;  // per value id: buffer fully computed in init
};
Hoistability computeHoistability(const Model& m, const Region& r);

// Rewrites an op list in place: drops ops with drop[i], inserts `prologue` first and each insertAfter[k].second right
// after op insertAfter[k].first (sorted by that index). Each op moves at most once and no second op list is built
// (Ops are big; the pipeline must stay well under 1 ms): output that overtakes the input displaces the few unread
// ops it would overwrite into a small queue.
void rewriteOps(std::vector<Op>& ops, const std::vector<uint8_t>& drop, std::vector<Op>& prologue,
                std::vector<std::pair<size_t, Op>>& insertAfter);

}  // namespace lk
