// bufferize: tensor values -> buffers in destination-passing style.
//
// 1. Every tensor-typed value is retyped to the matching buffer type (value
//    ids are kept, so no renaming is needed for the common case).
// 2. Each tensor result of a kernel becomes a `buffer.alloc` right before the
//    kernel and moves to the kernel's `outs`. A kernel attr
//    `inplace = [out_k, in_j, ...]` lets result k reuse operand j's buffer
//    when that operand's alias set dies at this kernel.
// 3. `buffer.free` is inserted after the last use of every alloc alias set
//    that isn't returned. Returned exec allocs get `{output = true}`.
// `buffer.alloc {persistent}` (model-lifetime state such as a KV cache, written in place by `inout` kernels) gets
// no free and is never reused in place, so plan-memory leaves it alone and hoist-init turns it into init state.
#include <algorithm>
#include <vector>

#include "analysis.h"
#include "lokinfer/compiler/passes.h"

namespace lk {

namespace {

constexpr size_t kNone = ~size_t(0);

// Alias-set roots: a view's root is its source's root. Dense tables indexed by value id (no hashing: the
// pipeline must stay well under 1 ms on Kokoro-sized graphs).
struct AliasInfo {
  std::vector<ValueId> root;        // kNoValue = the value is its own root
  std::vector<uint8_t> allocRoots;  // roots defined by buffer.alloc in this region

  explicit AliasInfo(size_t n) : root(n, kNoValue), allocRoots(n, 0) {}
  ValueId rootOf(ValueId v) const { return root[v] == kNoValue ? v : root[v]; }
};

AliasInfo computeAliases(const std::vector<Op>& ops, size_t numValues) {
  AliasInfo a(numValues);
  for (const Op& op : ops) {
    if ((op.kind == OpKind::View || op.kind == OpKind::Sync) && !op.results.empty())
      a.root[op.results[0]] = a.rootOf(op.operands[0]);
    if (op.kind == OpKind::BufferAlloc) a.allocRoots[op.results[0]] = 1;
  }
  return a;
}

// Last op index using any alias of each root (operands, outs, return); kNone if unused.
std::vector<size_t> lastUses(const std::vector<Op>& ops, const AliasInfo& a) {
  std::vector<size_t> last(a.root.size(), kNone);
  for (size_t i = 0; i < ops.size(); ++i) {
    for (ValueId v : ops[i].operands) last[a.rootOf(v)] = i;
    for (ValueId v : ops[i].outs) last[a.rootOf(v)] = i;
  }
  return last;
}

Status bufferizeRegion(Model& m, Region& r, bool isExec, const std::vector<uint8_t>& wasTensor) {
  const size_t nv = m.numValues();
  // ---- Phase A: allocs + outs + in-place reuse.
  // Liveness on the original op list: a kernel result's alias root is itself
  // (it becomes an alloc), so treat tensor kernel results as alloc roots.
  AliasInfo pre = computeAliases(r.ops, nv);
  for (const Op& op : r.ops)
    if (op.kind == OpKind::Kernel)
      for (ValueId v : op.results)
        if (wasTensor[v]) pre.allocRoots[v] = 1;
  std::vector<size_t> preLast = lastUses(r.ops, pre);

  std::vector<ValueId> rename(nv, kNoValue);
  auto ren = [&](ValueId v) { return rename[v] == kNoValue ? v : rename[v]; };
  std::vector<ValueId> reusedRoots;  // an input buffer can be reused at most once per op
  std::vector<uint8_t> persistentRoots(nv, 0);
  for (const Op& op : r.ops)
    if (op.kind == OpKind::BufferAlloc && op.hasFlag("persistent")) persistentRoots[op.results[0]] = 1;

  // Phase A rewrites the kernels in place (renamed operands, results moved to outs) and only records where the new
  // allocs go; the final op list is built once, at the end (one move per op: Ops are big).
  std::vector<std::pair<size_t, ValueId>> allocsBefore;  // (op index, value), in op order
  for (size_t i = 0; i < r.ops.size(); ++i) {
    Op& op = r.ops[i];
    for (ValueId& v : op.operands) v = ren(v);
    for (ValueId& v : op.outs) v = ren(v);
    if (op.kind != OpKind::Kernel) continue;
    std::vector<std::pair<size_t, size_t>> inplace;
    if (const Attr* a = op.attr("inplace")) {
      if (!std::holds_alternative<Attr::Ints>(a->value)) return Status::error("bufferize: inplace must be an int list");
      auto& v = std::get<Attr::Ints>(a->value);
      for (size_t k = 0; k + 1 < v.size(); k += 2) inplace.emplace_back(v[k], v[k + 1]);
    }
    reusedRoots.clear();
    size_t kept = 0;  // non-tensor (scalar) results stay results (compacted in place)
    for (size_t k = 0; k < op.results.size(); ++k) {
      ValueId res = op.results[k];
      if (!wasTensor[res]) {
        op.results[kept++] = res;
        continue;
      }
      bool reused = false;
      for (auto [outIdx, inIdx] : inplace) {
        if (outIdx != k || inIdx >= op.operands.size()) continue;
        ValueId in = op.operands[inIdx];
        ValueId inRoot = pre.rootOf(in);
        // The operand (after renaming) must be an in-region alloc root that dies here.
        if (!pre.allocRoots[inRoot] || pre.rootOf(in) != in || persistentRoots[inRoot]) continue;
        if (preLast[inRoot] != i ||
            std::find(reusedRoots.begin(), reusedRoots.end(), inRoot) != reusedRoots.end())
          continue;
        if (m.typeOf(in) != m.typeOf(res)) continue;
        // No other operand may alias the reused buffer (same value or a view of it).
        size_t aliases = 0;
        for (ValueId o : op.operands) aliases += pre.rootOf(o) == inRoot;
        if (aliases != 1) continue;
        rename[res] = in;
        // `in` now carries `res`'s lifetime: later in-place reuse of the chain
        // must see the extended last use.
        if (preLast[res] != kNone) preLast[inRoot] = preLast[res];
        reusedRoots.push_back(inRoot);
        op.outs.push_back(in);
        reused = true;
        break;
      }
      if (reused) continue;
      allocsBefore.emplace_back(i, res);
      op.outs.push_back(res);
    }
    op.results.resize(kept);
  }
  std::vector<Op>& ops = r.ops;

  // ---- Phase B: outputs + frees.
  AliasInfo post = computeAliases(ops, nv);
  for (const auto& [at, v] : allocsBefore) post.allocRoots[v] = 1;
  std::vector<size_t> last = lastUses(ops, post);
  std::vector<uint8_t> returnedRoots(nv, 0);
  for (ValueId v : ops.back().operands) returnedRoots[post.rootOf(v)] = 1;

  // A buffer a GPU kernel touched may still be in use by that (uncommitted)
  // dispatch until the next explicit `sync gpu_to_cpu`: its free (and so its
  // planned lifetime) extends to that sync, else to the end of the region.
  // Without this the planner could hand its bytes to a later CPU kernel while
  // the GPU still reads them (write-after-read).
  // That hazard only exists against CPU users: GPU-to-GPU reuse is ordered by the queue and the runtime's byte-range
  // barriers. So a root no CPU op ever touches ("gpu_only") is freed at its real last use, and plan-memory gives
  // gpu_only buffers their own pool, never sharing bytes with CPU-touched ones (Kokoro's GPU decoder + generator:
  // the rand510 GPU arenas dropped from 7.7 GB to 0.7 GB, real117 from 2.6 GB to 0.23 GB).
  std::vector<size_t> lastGpu(nv, kNone);
  std::vector<uint8_t> cpuTouched(nv, 0);
  for (size_t i = 0; i < ops.size(); ++i) {
    const Op& o = ops[i];
    const bool gpuKernel = o.kind == OpKind::Kernel && isGpuKernelName(o.name);
    if (gpuKernel) {
      for (ValueId v : o.operands) lastGpu[post.rootOf(v)] = i;
      for (ValueId v : o.outs) lastGpu[post.rootOf(v)] = i;
    } else if (o.kind == OpKind::Kernel || o.kind == OpKind::DimRead || o.kind == OpKind::Return) {
      for (ValueId v : o.operands) cpuTouched[post.rootOf(v)] = 1;
      for (ValueId v : o.outs) cpuTouched[post.rootOf(v)] = 1;
    }
  }
  std::vector<size_t> nextGpuSync(ops.size() + 1, ops.size() - 1);  // default: before return
  for (size_t i = ops.size(); i-- > 0;) {
    nextGpuSync[i] = nextGpuSync[i + 1];
    if (ops[i].kind == OpKind::Sync && ops[i].ints[0] == static_cast<int64_t>(SyncKind::GpuToCpu)) nextGpuSync[i] = i;
  }
  std::vector<std::pair<size_t, ValueId>> frees;  // (after op index, root), sorted below
  for (ValueId root = 0; root < nv; ++root) {
    if (!post.allocRoots[root] || post.rootOf(root) != root) continue;
    if (returnedRoots[root] || persistentRoots[root]) continue;
    if (last[root] == kNone) continue;  // never used: freed at the end of the region
    size_t at = last[root];
    if (lastGpu[root] != kNone && cpuTouched[root]) {
      size_t sync = nextGpuSync[lastGpu[root]];
      at = std::max(at, sync >= ops.size() - 1 ? ops.size() - 2 : sync);
    }
    frees.emplace_back(at, root);
  }
  std::sort(frees.begin(), frees.end());
  auto flagAlloc = [&](Op& alloc) {
    ValueId v = alloc.results[0];
    if (isExec && returnedRoots[v]) alloc.setAttr("output", Attr{true});
    if (lastGpu[v] != kNone && !cpuTouched[v] && !persistentRoots[v]) alloc.setAttr("gpu_only", Attr{true});
  };
  std::vector<Op> out;
  out.reserve(ops.size() + allocsBefore.size() + frees.size());
  size_t f = 0, al = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    for (; al < allocsBefore.size() && allocsBefore[al].first == i; ++al) {
      Op& alloc = out.emplace_back();
      alloc.kind = OpKind::BufferAlloc;
      alloc.results.push_back(allocsBefore[al].second);
      flagAlloc(alloc);
    }
    if (ops[i].kind == OpKind::BufferAlloc) flagAlloc(ops[i]);
    out.push_back(std::move(ops[i]));
    for (; f < frees.size() && frees[f].first == i; ++f) {
      Op& fr = out.emplace_back();
      fr.kind = OpKind::BufferFree;
      fr.operands.push_back(frees[f].second);
    }
  }
  r.ops = std::move(out);
  return Status::ok();
}

}  // namespace

Status bufferize(Model& m) {
  std::vector<uint8_t> wasTensor(m.numValues(), 0);
  std::vector<Type> bufferType(m.types().numTypes(), nullptr);  // tensor type id -> buffer type (interned once)
  for (ValueId v = 0; v < m.numValues(); ++v) {
    Type t = m.typeOf(v);
    if (t->isTensor()) {
      wasTensor[v] = 1;
      Type& bt = bufferType[t->id];
      if (!bt) bt = m.types().withKind(t, TypeKind::Buffer);
      m.setType(v, bt);
    }
  }
  LK_RETURN_IF_ERROR(bufferizeRegion(m, m.init, false, wasTensor));
  return bufferizeRegion(m, m.exec, true, wasTensor);
}

}  // namespace lk
