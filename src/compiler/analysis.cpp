#include "analysis.h"

#include <cstddef>
#include <deque>

namespace lk {

Aliases computeAliases(const Model& m, const Region& r) {
  Aliases a;
  a.root.assign(m.numValues(), kNoValue);
  for (const Op& op : r.ops)
    if ((op.kind == OpKind::View || op.kind == OpKind::Sync) && !op.results.empty())
      a.root[op.results[0]] = a.rootOf(op.operands[0]);
  return a;
}

namespace {

bool dimIsConst(Dim d) { return d->isConst(); }

bool isStatic(Type t) {
  for (Dim d : t->shape)
    if (!d->isConst()) return false;
  return true;
}

}  // namespace

Hoistability computeHoistability(const Model& m, const Region& r) {
  Aliases aliases = computeAliases(m, r);
  Hoistability h;
  const size_t nv = m.numValues();
  h.op.assign(r.ops.size(), false);
  h.values.assign(nv, 0);
  h.initWrittenRoots.assign(nv, 0);

  // 1. Memory-hoistable values: constants, static non-output allocs and
  //    constant-offset views of memory-hoistable values.
  std::vector<uint8_t> constRoots(nv, 0);  // results of const.* ops
  for (size_t i = 0; i < r.ops.size(); ++i) {
    const Op& op = r.ops[i];
    bool ok = false;
    switch (op.kind) {
      case OpKind::ConstInline:
      case OpKind::ConstInternal:
      case OpKind::ConstExternal:
        ok = true;
        constRoots[op.results[0]] = 1;
        break;
      case OpKind::BufferAlloc:
        ok = !op.hasFlag("output") && isStatic(m.typeOf(op.results[0]));
        break;
      case OpKind::View:
        // The result type must be static too: a view typed with a per-call
        // symbol (e.g. pos_table[0:T]) can't exist at init time.
        ok = h.values[op.operands[0]] && op.dims[0]->isConst() && isStatic(m.typeOf(op.results[0]));
        break;
      default:
        break;
    }
    h.op[i] = ok;
    if (ok) h.values[op.results[0]] = 1;
  }

  // 2. Kernels: start optimistic, demote until stable. A kernel is hoistable
  //    iff it's not `nohoist`, has outs, static results, every operand's
  //    contents are constant (a const, or a buffer whose writers are all
  //    hoistable), and every out is memory-hoistable.
  std::vector<bool> kernelOk(r.ops.size(), false);
  for (size_t i = 0; i < r.ops.size(); ++i) {
    const Op& op = r.ops[i];
    if (op.kind != OpKind::Kernel) continue;
    bool ok = !op.hasFlag("nohoist") && !op.outs.empty();
    // Dim attrs evaluated at runtime must not depend on per-call symbols.
    for (const NamedAttr& a : op.attrs)
      if (auto* d = std::get_if<Dim>(&a.attr.value)) ok = ok && dimIsConst(*d);
    for (ValueId v : op.results) ok = ok && isStatic(m.typeOf(v));
    for (ValueId v : op.outs) ok = ok && h.values[v];
    kernelOk[i] = ok;
  }
  // Per root: -1 no kernel writes it, 0 some writer isn't hoistable, 1 all writers are hoistable.
  std::vector<int8_t> rootAllWritersOk(nv, -1);
  while (true) {
    std::fill(rootAllWritersOk.begin(), rootAllWritersOk.end(), int8_t(-1));
    for (size_t i = 0; i < r.ops.size(); ++i) {
      const Op& op = r.ops[i];
      if (op.kind != OpKind::Kernel) continue;
      for (ValueId v : op.outs) {
        int8_t& w = rootAllWritersOk[aliases.rootOf(v)];
        w = (w != 0 && kernelOk[i]) ? 1 : 0;
      }
    }
    auto contentConst = [&](ValueId v) {
      ValueId root = aliases.rootOf(v);
      if (constRoots[root]) return true;
      return rootAllWritersOk[root] == 1 && h.values[root];
    };
    bool changed = false;
    for (size_t i = 0; i < r.ops.size(); ++i) {
      if (!kernelOk[i]) continue;
      const Op& op = r.ops[i];
      bool ok = true;
      for (ValueId v : op.operands) ok = ok && contentConst(v);
      // An out also written by a per-call kernel would clobber init results.
      for (ValueId v : op.outs) ok = ok && rootAllWritersOk[aliases.rootOf(v)] == 1;
      if (!ok) {
        kernelOk[i] = false;
        changed = true;
        // Propagate within this sweep so forward chains settle in O(1) sweeps.
        for (ValueId v : op.outs) rootAllWritersOk[aliases.rootOf(v)] = 0;
      }
    }
    if (!changed) {
      for (size_t v = 0; v < nv; ++v)
        if (rootAllWritersOk[v] == 1) h.initWrittenRoots[v] = 1;
      break;
    }
  }
  for (size_t i = 0; i < r.ops.size(); ++i) {
    if (!kernelOk[i]) continue;
    h.op[i] = true;
    for (ValueId v : r.ops[i].results) h.values[v] = 1;
  }
  return h;
}

void rewriteOps(std::vector<Op>& ops, const std::vector<uint8_t>& drop, std::vector<Op>& prologue,
                std::vector<std::pair<size_t, Op>>& insertAfter) {
  const size_t n = ops.size();
  size_t w = 0;     // next output position
  size_t next = 0;  // next unread input index: positions >= next still hold unread inputs (unless displaced)
  std::deque<std::pair<size_t, Op>> displaced;  // unread inputs moved out of the way, in index order
  auto emit = [&](Op&& op) {
    if (w < ops.size()) {
      if (w >= next && w < n) displaced.emplace_back(w, std::move(ops[w]));
      ops[w] = std::move(op);
    } else {
      ops.push_back(std::move(op));
    }
    ++w;
  };
  for (Op& p : prologue) emit(std::move(p));
  size_t ins = 0;
  for (size_t i = 0; i < n; ++i) {
    const bool wasDisplaced = !displaced.empty() && displaced.front().first == i;
    next = i + 1;
    if (wasDisplaced) {
      Op op = std::move(displaced.front().second);
      displaced.pop_front();
      if (!drop[i]) emit(std::move(op));
    } else if (!drop[i]) {
      if (w != i) ops[w] = std::move(ops[i]);  // w < i here: position i was never written
      ++w;
    }
    for (; ins < insertAfter.size() && insertAfter[ins].first == i; ++ins) emit(std::move(insertAfter[ins].second));
  }
  ops.erase(ops.begin() + static_cast<std::ptrdiff_t>(w), ops.end());
}

}  // namespace lk
