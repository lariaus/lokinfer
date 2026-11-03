// dce: dead-code elimination on the exec region (before bufferize).
//
// Backward sweep: a value is live if `return`, or an op that is kept, uses it. Kept: `return`; `dim.read` (it
// defines symbols that later shapes depend on); kernels with an `inout` attr or without results (side effects:
// they write persistent buffers in place); any other op only if one of its results is live. Unused constants,
// views, syncs, kernels (incl. unused RNG / `nohoist` ones) and allocs disappear.
#include <cstddef>
#include <vector>

#include "lokinfer/compiler/passes.h"

namespace lk {

Status dce(Model& m) {
  Region& r = m.exec;
  std::vector<uint8_t> live(m.numValues(), 0);
  std::vector<bool> keep(r.ops.size(), false);
  for (size_t i = r.ops.size(); i-- > 0;) {
    const Op& op = r.ops[i];
    bool k = op.kind == OpKind::Return || op.kind == OpKind::DimRead;
    if (op.kind == OpKind::Kernel && (op.attr("inout") || (op.results.empty() && op.outs.empty()))) k = true;
    for (ValueId v : op.results) k = k || live[v];
    for (ValueId v : op.outs) k = k || live[v];
    if (!k) continue;
    keep[i] = true;
    for (ValueId v : op.operands) live[v] = 1;
    for (ValueId v : op.outs) live[v] = 1;
  }
  // Compact in place (no new op list: Ops are big and the common case removes nothing or little).
  size_t w = 0;
  for (size_t i = 0; i < r.ops.size(); ++i) {
    if (!keep[i]) continue;
    if (w != i) r.ops[w] = std::move(r.ops[i]);
    ++w;
  }
  r.ops.erase(r.ops.begin() + static_cast<std::ptrdiff_t>(w), r.ops.end());
  return Status::ok();
}

}  // namespace lk
