// hoist-init: move everything that doesn't depend on exec inputs into init.
//
// Hoistable ops come from computeHoistability (constants, static allocs incl.
// the static arena, constant views, kernels on constants). After moving:
// - hoisted values still used by exec become state (init returns them,
//   exec receives them as `state(...)` args under fresh value ids);
// - a `buffer.free` of a hoisted value moves to init if nothing in exec uses
//   the value anymore (init-only temporaries), otherwise it's dropped (the
//   buffer lives as long as the model).
#include <algorithm>
#include <vector>

#include "analysis.h"
#include "lokinfer/compiler/passes.h"

namespace lk {

Status hoistInit(Model& m) {
  Region& ex = m.exec;
  Hoistability h = computeHoistability(m, ex);

  // Values used by ops that stay in exec (frees excluded).
  std::vector<uint8_t> usedByExec(m.numValues(), 0);
  for (size_t i = 0; i < ex.ops.size(); ++i) {
    const Op& op = ex.ops[i];
    if (h.op[i] || op.kind == OpKind::BufferFree) continue;
    for (ValueId v : op.operands) usedByExec[v] = 1;
    for (ValueId v : op.outs) usedByExec[v] = 1;
  }

  Op initReturn = std::move(m.init.ops.back());
  m.init.ops.pop_back();
  size_t numHoisted = 0;
  for (bool b : h.op) numHoisted += b;
  m.init.ops.reserve(m.init.ops.size() + numHoisted + ex.ops.size() / 8 + 1);  // + init-only frees
  // Ops staying in exec are compacted in place (no second op list).
  size_t w = 0;
  for (size_t i = 0; i < ex.ops.size(); ++i) {
    Op& op = ex.ops[i];
    if (h.op[i]) {
      m.init.ops.push_back(std::move(op));
    } else if (op.kind == OpKind::BufferFree && h.values[op.operands[0]]) {
      if (!usedByExec[op.operands[0]]) m.init.ops.push_back(std::move(op));
      // else: dropped, the buffer is model-lifetime state.
    } else {
      if (w != i) ex.ops[w] = std::move(op);
      ++w;
    }
  }
  ex.ops.erase(ex.ops.begin() + static_cast<std::ptrdiff_t>(w), ex.ops.end());
  std::vector<Op>& keep = ex.ops;

  // Hoisted values used in exec become state, in first-use order.
  std::vector<ValueId> toState(m.numValues(), kNoValue);  // (new state values are never hoisted themselves)
  auto mapValue = [&](ValueId& v) {
    if (v >= h.values.size() || !h.values[v]) return;
    if (toState[v] == kNoValue) {
      toState[v] = m.newValue(m.typeOf(v), m.nameOf(v));
      initReturn.operands.push_back(v);
      ex.state.push_back(toState[v]);
    }
    v = toState[v];
  };
  for (Op& op : keep) {
    for (ValueId& v : op.operands) mapValue(v);
    for (ValueId& v : op.outs) mapValue(v);
  }
  m.init.ops.push_back(std::move(initReturn));
  return Status::ok();
}

}  // namespace lk
