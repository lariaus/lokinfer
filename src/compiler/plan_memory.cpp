// plan-memory: replace exec-region allocs/frees by views into arenas.
//
// Candidates: non-output `buffer.alloc`s that are freed in the region and
// aren't computed from constants only (those keep their own alloc so
// hoist-init can move them to init without the arena reusing their memory).
// Candidates are grouped by the last symbol-defining op (dim.read) their size
// depends on:
// - group 0 (input symbols only), all sizes static: one static arena
//   `buffer.alloc {arena = true} : buffer<u8[total]>` with constant offsets;
// - otherwise one `mem.plan` per group, placed at the region start (group 0)
//   or right after the dim.read; offsets are symbols evaluated at runtime.
// Each group currently gets its own arena (no reuse across groups yet). Within a group, `gpu_only` allocs (bufferize:
// never touched by a CPU op, freed at their last GPU use) get a separate arena from the others, so a CPU kernel can
// never be handed bytes the GPU may still be reading.
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "analysis.h"
#include "lokinfer/compiler/memplan.h"
#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/verify.h"

namespace lk {

namespace {

constexpr int64_t kAlign = 256;

struct Candidate {
  size_t allocIndex;
  size_t freeIndex;
  ValueId value;
  Dim bytes;
  int64_t group;  // op index of the defining dim.read, or -1
  bool gpuOnly;
};

}  // namespace

Status planMemory(Model& m) {
  Region& r = m.exec;
  Hoistability hoist = computeHoistability(m, r);
  DimContext& dims = m.dims();

  constexpr size_t kNone = ~size_t(0);
  // Symbol -> op index of the dim.read defining it (dense; -1 = not defined by a dim.read).
  std::vector<int64_t> symDef(dims.numSymbols(), -1);
  bool anyDimRead = false;
  for (size_t i = 0; i < r.ops.size(); ++i)
    if (r.ops[i].kind == OpKind::DimRead) {
      symDef[r.ops[i].symbols[0]] = static_cast<int64_t>(i);
      anyDimRead = true;
    }

  std::vector<size_t> freeAt(m.numValues(), kNone);
  for (size_t i = 0; i < r.ops.size(); ++i)
    if (r.ops[i].kind == OpKind::BufferFree) freeAt[r.ops[i].operands[0]] = i;
  std::vector<Candidate> cands;  // in alloc order
  std::vector<int64_t> syms;
  // Byte sizes per type, interned once per distinct type (dim arithmetic hash-conses).
  std::vector<Dim> bytesOfType(m.types().numTypes(), nullptr);
  for (size_t i = 0; i < r.ops.size(); ++i) {
    const Op& op = r.ops[i];
    if (op.kind != OpKind::BufferAlloc || hoist.initWrittenRoots[op.results[0]]) continue;
    ValueId v = op.results[0];
    if (freeAt[v] == kNone || op.hasFlag("output") || op.hasFlag("arena")) continue;
    Type t = m.typeOf(v);
    if (t->id >= bytesOfType.size()) bytesOfType.resize(t->id + 1, nullptr);
    Dim& bytes = bytesOfType[t->id];
    if (!bytes) bytes = m.types().numBytes(t);
    Candidate c{i, freeAt[v], v, bytes, -1, op.hasFlag("gpu_only")};
    if (anyDimRead && !bytes->isConst()) {
      syms.clear();
      collectSymbols(c.bytes, syms);
      for (int64_t s : syms)
        if (symDef[s] >= 0) c.group = std::max(c.group, symDef[s]);
    }
    cands.push_back(c);
  }
  if (cands.empty()) return Status::ok();

  std::map<std::pair<int64_t, bool>, std::vector<size_t>> groups;  // (group, gpu_only) -> candidate indices
  for (size_t i = 0; i < cands.size(); ++i) groups[{cands[i].group, cands[i].gpuOnly}].push_back(i);

  // Replacement ops: alloc index -> view op; free indices removed; new ops
  // inserted before (group -1) or after (dim.read) given indices.
  // Each planned alloc becomes a view in place (same position, same result); frees are dropped.
  std::vector<uint8_t> dropAt(r.ops.size(), 0);
  std::vector<Op> prologue;
  std::vector<std::pair<size_t, Op>> insertAfter;  // (dim.read op index, plan), in group order

  dims.reserve(cands.size() + groups.size(), cands.size() + groups.size());  // offset symbols
  auto elemBytes = [&](ValueId v) { return static_cast<int64_t>(dtypeSize(m.typeOf(v)->dtype)); };
  int planCounter = 0;
  for (auto& [key, idxs] : groups) {
    const int64_t group = key.first;
    bool allStatic = true;
    for (size_t i : idxs) allStatic = allStatic && cands[i].bytes->isConst();
    ValueId arena;
    std::vector<Dim> offsets;  // in elements of each buffer's dtype
    Op planOp;
    if (allStatic && group < 0) {
      std::vector<PlanBuffer> bufs;
      for (size_t i : idxs) bufs.push_back({cands[i].bytes->value, (int64_t)cands[i].allocIndex, (int64_t)cands[i].freeIndex});
      PlanResult pr = planOffsets(bufs, kAlign);
      if (passNotesEnabled()) {
        int64_t sum = 0;
        for (const PlanBuffer& b : bufs) sum += (std::max<int64_t>(b.size, 1) + kAlign - 1) / kAlign * kAlign;
        passNote(std::string("static arena") + (key.second ? " (gpu_only)" : "") + ": " + std::to_string(pr.total) +
                 " bytes for " + std::to_string(bufs.size()) + " buffers; max-live lower bound " +
                 std::to_string(planLowerBound(bufs, kAlign)) + ", without reuse " + std::to_string(sum));
      }
      arena = m.newValue(m.types().buffer(DType::U8, {dims.constant(pr.total)}), "arena");
      planOp.kind = OpKind::BufferAlloc;
      planOp.setAttr("arena", Attr{true});
      planOp.results.push_back(arena);
      for (size_t k = 0; k < idxs.size(); ++k)  // element offsets (planned offsets are 256-byte aligned)
        offsets.push_back(dims.constant(pr.offsets[k] / elemBytes(cands[idxs[k]].value)));
    } else {
      // Fresh symbol prefix (the pass may run on IR that already has plans).
      while (dims.findSymbol("M" + std::to_string(planCounter) + "_size") >= 0) ++planCounter;
      std::string base = "M" + std::to_string(planCounter++), name;
      Dim total = dims.sym(base + "_size");
      planOp.kind = OpKind::MemPlan;
      planOp.symbols.push_back(total->value);
      Attr::Ints elems;
      elems.reserve(idxs.size());
      for (size_t k = 0; k < idxs.size(); ++k) {
        const Candidate& c = cands[idxs[k]];
        name.assign(base).append("_o").append(std::to_string(k));
        Dim off = dims.sym(name);  // in elements of buffer k (attr `elem`): the view uses it as is
        planOp.dims.push_back(c.bytes);
        planOp.ints.push_back(static_cast<int64_t>(c.allocIndex));
        planOp.ints.push_back(static_cast<int64_t>(c.freeIndex));
        planOp.symbols.push_back(off->value);
        elems.push_back(elemBytes(c.value));
        offsets.push_back(off);
      }
      planOp.ints.push_back(kAlign);
      planOp.setAttr("elem", Attr{std::move(elems)});
      if (key.second) planOp.setAttr("gpu_only", Attr{true});
      if (passNotesEnabled())
        passNote("mem.plan " + base + (key.second ? " (gpu_only)" : "") + ": " + std::to_string(idxs.size()) +
                 " buffers, planned at run time " +
                 (group < 0 ? std::string("on entry") : "after op " + std::to_string(group) + " (dim.read)"));
      arena = m.newValue(m.types().buffer(DType::U8, {total}), "arena");
      planOp.results.push_back(arena);
    }
    for (size_t k = 0; k < idxs.size(); ++k) {
      const Candidate& c = cands[idxs[k]];
      Op& view = r.ops[c.allocIndex];  // buffer.alloc -> view, result kept
      view.kind = OpKind::View;
      view.attrs.clear();
      view.operands.assign(1, arena);
      view.dims.assign(1, offsets[k]);
      dropAt[c.freeIndex] = 1;
    }
    if (group < 0) prologue.push_back(std::move(planOp));
    else insertAfter.emplace_back(static_cast<size_t>(group), std::move(planOp));
  }
  std::stable_sort(insertAfter.begin(), insertAfter.end(), [](auto& a, auto& b) { return a.first < b.first; });

  rewriteOps(r.ops, dropAt, prologue, insertAfter);
  return Status::ok();
}

}  // namespace lk
