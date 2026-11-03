// bake-init (include/lokinfer/runtime/bake_init.h): run init once, store its computed results as constants.
// Lives in the runtime library because it executes the model's init kernels.
#include "lokinfer/runtime/bake_init.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "lokinfer/ir/text.h"
#include "lokinfer/ir/verify.h"

namespace lk::rt {

namespace {

constexpr size_t kPage = 16384;

// Exact copy (text round trip) with the internal constants' data attached.
Expected<std::unique_ptr<Model>> cloneModel(const Model& m) {
  auto parsed = parseModel(printModel(m));
  if (!parsed) return parsed.status();
  std::unique_ptr<Model> c = parsed.take();
  for (ConstantData& k : c->constants)
    if (const ConstantData* o = m.findConstant(k.name)) {
      k.owner = o->owner;
      k.bytes = o->bytes;
      k.size = o->size;
    }
  return c;
}

// Removes init ops nothing needs any more (buffer form: a kernel is needed if it writes a needed buffer).
void pruneInit(Model& m) {
  Region& r = m.init;
  const size_t nv = m.numValues();
  std::vector<ValueId> root(nv, kNoValue);
  auto rootOf = [&](ValueId v) { return root[v] == kNoValue ? v : root[v]; };
  for (const Op& op : r.ops)
    if ((op.kind == OpKind::View || op.kind == OpKind::Sync) && !op.results.empty())
      root[op.results[0]] = rootOf(op.operands[0]);
  std::vector<uint8_t> liveVal(nv, 0), liveRoot(nv, 0);
  auto use = [&](ValueId v) {
    liveVal[v] = 1;
    liveRoot[rootOf(v)] = 1;
  };
  std::vector<uint8_t> keep(r.ops.size(), 0);
  for (size_t i = r.ops.size(); i-- > 0;) {
    const Op& op = r.ops[i];
    bool k = false;
    switch (op.kind) {
      case OpKind::Return:
      case OpKind::DimRead:
        k = true;
        break;
      case OpKind::Kernel:
        k = op.attr("inout") || (op.outs.empty() && op.results.empty());
        for (ValueId v : op.outs) k = k || liveRoot[rootOf(v)];
        for (ValueId v : op.results) k = k || liveVal[v];
        if (k)
          for (ValueId v : op.outs) use(v);
        break;
      case OpKind::BufferFree:
        break;  // decided below, from its buffer
      case OpKind::BufferAlloc:
        k = liveRoot[op.results[0]] || liveVal[op.results[0]];
        break;
      default:
        for (ValueId v : op.results) k = k || liveVal[v] || liveRoot[v];
        break;
    }
    if (!k) continue;
    keep[i] = 1;
    for (ValueId v : op.operands) use(v);
  }
  for (size_t i = 0; i < r.ops.size(); ++i)
    if (r.ops[i].kind == OpKind::BufferFree) keep[i] = liveRoot[rootOf(r.ops[i].operands[0])];
  size_t w = 0;
  for (size_t i = 0; i < r.ops.size(); ++i)
    if (keep[i]) {
      if (w != i) r.ops[w] = std::move(r.ops[i]);
      ++w;
    }
  r.ops.erase(r.ops.begin() + static_cast<std::ptrdiff_t>(w), r.ops.end());
}

}  // namespace

Expected<std::unique_ptr<Model>> bakeInit(std::shared_ptr<const Model> model, std::vector<ExternalBinding> externals,
                                          BakeStats* stats) {
  BakeStats st;
  auto cl = cloneModel(*model);
  if (!cl) return cl.status();
  std::unique_ptr<Model> c = cl.take();
  st.initOpsBefore = st.initOpsAfter = c->init.ops.size();
  for (const Op& op : c->init.ops)
    if (op.kind == OpKind::DimRead || op.kind == OpKind::MemPlan) {  // init defines symbols: leave it as is
      if (stats) *stats = st;
      return c;
    }
  auto exe = Executable::create(model, std::move(externals));
  if (!exe) return exe.status();

  Region& init = c->init;
  const size_t nv = c->numValues();
  std::vector<ValueId> root(nv, kNoValue);
  auto rootOf = [&](ValueId v) { return root[v] == kNoValue ? v : root[v]; };
  std::vector<int> allocAt(nv, -1);
  std::vector<int8_t> writerGpu(nv, -1);  // root -> device of its last init writer (-1: none)
  for (size_t i = 0; i < init.ops.size(); ++i) {
    const Op& op = init.ops[i];
    if ((op.kind == OpKind::View || op.kind == OpKind::Sync) && !op.results.empty())
      root[op.results[0]] = rootOf(op.operands[0]);
    if (op.kind == OpKind::BufferAlloc) allocAt[op.results[0]] = static_cast<int>(i);
    if (op.kind == OpKind::Kernel)
      for (ValueId v : op.outs) writerGpu[rootOf(v)] = isGpuKernel(op.name) ? 1 : 0;
  }
  std::vector<Op> consts;
  Op& ret = init.ops.back();
  for (size_t i = 0; i < ret.operands.size(); ++i) {
    ValueId v = ret.operands[i];
    ValueId rv = rootOf(v);
    if (allocAt[rv] < 0 || writerGpu[rv] < 0) continue;  // not a computed buffer (constant, arena view, ...)
    const Op& alloc = init.ops[static_cast<size_t>(allocAt[rv])];
    if (alloc.hasFlag("arena") || alloc.hasFlag("persistent") || alloc.hasFlag("output")) continue;
    Type t = c->typeOf(v);
    size_t bytes = dtypeSize(t->dtype);
    bool isStatic = true;
    for (Dim d : t->shape) {
      isStatic = isStatic && d->isConst() && d->value >= 0;
      if (isStatic) bytes *= static_cast<size_t>(d->value);
    }
    const void* src = (*exe)->stateData(i);
    if (!isStatic || !src) continue;
    void* p = std::aligned_alloc(kPage, (std::max<size_t>(bytes, 1) + kPage - 1) / kPage * kPage);
    if (!p) return Status::error("bake-init: out of memory");
    std::memcpy(p, src, bytes);
    std::string name = "baked" + std::to_string(i);
    while (c->findConstant(name)) name += "_";
    c->addConstant(ConstantData{name, t, std::shared_ptr<const void>(p, [](const void* q) { std::free(const_cast<void*>(q)); }),
                                static_cast<const uint8_t*>(p), bytes});
    Op k;
    k.kind = OpKind::ConstInternal;
    k.name = name;
    k.setAttr("device", Attr{std::string(writerGpu[rv] ? "gpu" : "cpu")});
    ValueId nvId = c->newValue(t, c->nameOf(v).empty() ? name : c->nameOf(v) + "_baked");
    k.results.push_back(nvId);
    ret.operands[i] = nvId;
    consts.push_back(std::move(k));
    ++st.baked;
    st.bakedBytes += bytes;
  }
  init.ops.insert(init.ops.end() - 1, std::make_move_iterator(consts.begin()), std::make_move_iterator(consts.end()));
  pruneInit(*c);
  st.initOpsAfter = c->init.ops.size();
  LK_RETURN_IF_ERROR(verifyModel(*c));
  if (stats) *stats = st;
  return c;
}

}  // namespace lk::rt
