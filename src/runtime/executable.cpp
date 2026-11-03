// Executable: a flat instruction list built once from the final IR, plus a
// slot table (one slot per IR value) holding pointers for the current call.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <TargetConditionals.h>

#include "gpu_internal.h"
#include "gpu_replay.h"
#include "lokinfer/compiler/memplan.h"
#include "lokinfer/ir/verify.h"
#include "lokinfer/runtime/runtime.h"

namespace lk::rt {

namespace {

constexpr size_t kPage = 16384;
constexpr size_t kPlanCacheSize = 8;  // mem.plan results kept per plan, keyed by symbol values
constexpr int64_t kUnbound = INT64_MIN;

size_t roundUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

Storage::~Storage() {
  if (mtlBuffer) gpuReleaseBuffer(mtlBuffer);
  if (ownsData) std::free(data);
}

std::shared_ptr<Storage> wrapStorage(void* data, size_t bytes, std::shared_ptr<const void> keep, GpuContext* gpu,
                                     uint64_t& offset) {
  if (!gpu || !keep || !data) return nullptr;
  uintptr_t p = reinterpret_cast<uintptr_t>(data);
  uintptr_t base = p / kPage * kPage;  // the pages holding the data are mapped while it lives
  auto s = std::make_shared<Storage>();
  s->data = reinterpret_cast<void*>(base);
  s->size = roundUp(p - base + std::max<size_t>(bytes, 1), kPage);
  s->ownsData = false;
  s->keep = std::move(keep);
  s->mtlBuffer = gpuWrapNoCopy(gpu, s->data, s->size);
  if (!s->mtlBuffer) return nullptr;
  offset = p - base;
  return s;
}

namespace {

// GPU-visible storage is recycled by size class: wrapping memory as an
// MTLBuffer and changing the residency set are far too expensive per call.
// Free storages stay wrapped and resident; the pool is bounded.
struct StoragePool {
  std::mutex mu;
  std::unordered_map<size_t, std::vector<Storage*>> free;
  size_t cachedBytes = 0;
  static constexpr size_t kMaxCached = size_t(512) << 20;
};

StoragePool& pool() {
  static auto* p = new StoragePool();  // leaked: outlives static destruction
  return *p;
}

size_t sizeClass(size_t bytes) {
  // Pages up to 1 MiB, then 8 classes per power of two (≤ 12.5 % waste;
  // plain powers of two turned a 9 MB weight into 16 MB).
  size_t s = roundUp(std::max<size_t>(bytes, 1), kPage);
  if (s <= (size_t(1) << 20)) return s;
  size_t p = size_t(1) << 20;
  while (p * 2 < s) p <<= 1;
  return roundUp(s, p / 8);
}

void returnToPool(Storage* s) {
  StoragePool& p = pool();
  {
    std::lock_guard<std::mutex> lock(p.mu);
    if (p.cachedBytes + s->size <= StoragePool::kMaxCached) {
      p.free[s->size].push_back(s);
      p.cachedBytes += s->size;
      return;
    }
  }
  delete s;
}

}  // namespace

std::shared_ptr<Storage> allocateStorage(size_t bytes, GpuContext* gpu) {
  if (gpu) {
    size_t cls = sizeClass(bytes);
    StoragePool& p = pool();
    {
      std::lock_guard<std::mutex> lock(p.mu);
      auto it = p.free.find(cls);
      if (it != p.free.end() && !it->second.empty()) {
        Storage* s = it->second.back();
        it->second.pop_back();
        p.cachedBytes -= s->size;
        return std::shared_ptr<Storage>(s, returnToPool);
      }
    }
    auto* s = new Storage();
    s->size = cls;
    void* mem = nullptr;
    if (posix_memalign(&mem, kPage, s->size) != 0) {
      delete s;
      return nullptr;
    }
    s->data = mem;
    s->mtlBuffer = gpuWrapNoCopy(gpu, mem, s->size);
    return std::shared_ptr<Storage>(s, returnToPool);
  }
  auto s = std::make_shared<Storage>();
  s->size = roundUp(std::max<size_t>(bytes, 1), kPage);
  void* mem = nullptr;
  if (posix_memalign(&mem, kPage, s->size) != 0) return nullptr;
  s->data = mem;
  return s;
}

namespace {

bool kernelNameIsGpu(std::string_view name) {
  size_t pos = 0;
  while (pos <= name.size()) {
    size_t next = name.find('_', pos);
    if (next == std::string_view::npos) next = name.size();
    if (name.substr(pos, next - pos) == "gpu") return true;
    pos = next + 1;
  }
  return false;
}

// If d == k*S + b with S the only unbound symbol (S appearing once, linearly),
// returns true with S's id, k and b. Bare symbols are k=1, b=0.
bool linearInOneSymbol(Dim d, const std::vector<int64_t>& syms, int64_t unbound, int64_t& sym, int64_t& k,
                       int64_t& b) {
  auto unboundSym = [&](Dim t) { return t->kind == DimKind::Sym && syms[t->value] == unbound; };
  auto term = [&](Dim t, int64_t& coef) -> bool {  // coef * S
    if (unboundSym(t)) {
      coef = 1;
      return true;
    }
    if (t->kind == DimKind::Mul && t->operands.size() == 2 && t->operands[0]->isConst() && unboundSym(t->operands[1])) {
      coef = t->operands[0]->value;
      return true;
    }
    return false;
  };
  int64_t coef = 0;
  if (term(d, coef)) {
    sym = d->kind == DimKind::Sym ? d->value : d->operands[1]->value;
    k = coef;
    b = 0;
    return true;
  }
  if (d->kind != DimKind::Add) return false;
  Dim symTerm = nullptr;
  int64_t constPart = 0;
  for (Dim t : d->operands) {
    int64_t c = 0;
    if (t->isConst()) constPart += t->value;
    else if (!symTerm && term(t, c)) symTerm = t;
    else return false;
  }
  if (!symTerm) return false;
  term(symTerm, coef);
  sym = symTerm->kind == DimKind::Sym ? symTerm->value : symTerm->operands[1]->value;
  k = coef;
  b = constPart;
  return true;
}

struct Slot {
  char* ptr = nullptr;
  void* mtl = nullptr;
  uint64_t off = 0;
  std::shared_ptr<Storage> owned;
  int64_t scalar = 0;  // host scalar storage (8 bytes)
};

struct DynInt {
  uint32_t index;
  Dim dim;
};

struct Instr {
  OpKind kind;
  const Op* op;
  const KernelEntry* kernel = nullptr;
  void* state = nullptr;
  std::vector<ValueId> args;  // kernel: operands then outs; view: {src}; free/dimread: {buf}
  ValueId result = kNoValue;
  std::vector<int64_t> ints;
  std::vector<DynInt> dynInts;
  std::vector<double> floats;
  // view
  Dim offset = nullptr;
  int64_t elemSize = 0;
  bool checkBounds = false;  // dynamic views: verified per call (static ones by the verifier)
  // mem.plan (compiler/runtime design: docs/impl-notes/design-compiler-passes.md "plan-memory")
  std::vector<PlanBuffer> planBufs;
  std::vector<int64_t> planElem;     // bytes per element of each buffer (offset symbols are element offsets); 1 = bytes
  int planPool = 0;                  // arena pool shared by the plans of a region: 0 CPU-touched, 1 gpu_only
  std::vector<uint32_t> planPrior;   // earlier plans of the same pool in this region (their buffers are obstacles)
  std::vector<int64_t> planKeySyms;  // symbols the plan depends on (its sizes' + those of planPrior)
  struct PlanEntry {
    std::vector<int64_t> key;      // values of planKeySyms
    std::vector<int64_t> offsets;  // bytes
    std::vector<int64_t> sizes;    // bytes (as evaluated)
    int64_t total = 0;             // bytes, incl. the prior plans' buffers
  };
  std::vector<PlanEntry> planCache;  // most recently used first, at most kPlanCacheSize entries
  int planLast = -1;                 // index in planCache of this call's plan (obstacles for later plans of the pool)
  std::shared_ptr<Storage> planOverflow;  // when the pool's arena was too small mid-call (grown next call)
  // GPU hazard tracking: per arg rank (dims prefix in ints) and element size.
  uint32_t numOperands = 0;
  std::vector<uint32_t> argRank;
  std::vector<uint32_t> argElem;
  // scratch (per call)
  std::vector<void*> ptrs;
  std::vector<BufRef> bufs;
};

// Byte ranges touched by GPU dispatches since the last barrier.
struct HazardTracker {
  struct Range {
    void* buf;
    uint64_t begin, end;
  };
  std::vector<Range> writes, reads;
  uint64_t segment = ~0ull;

  static bool overlaps(const std::vector<Range>& set, const Range& r) {
    for (const Range& q : set)
      if (q.buf == r.buf && q.begin < r.end && r.begin < q.end) return true;
    return false;
  }
};

}  // namespace

struct Executable::Impl {
  std::shared_ptr<const Model> model;
  std::vector<ExternalBinding> externals;
  GpuContext* gpu = nullptr;  // process-wide shared context
  std::vector<Instr> init, exec;
  std::vector<Slot> slots;
  std::vector<int64_t> syms;
  std::vector<int64_t> initSyms;  // symbol values after init (restored per call)
  std::unordered_set<ValueId> freshOutputs;  // exec buffer.alloc {output}
  // Return operand k -> the exec {output} alloc it is (a chain of zero-offset views of), else kNoValue; and the
  // reverse map for the first k of each such alloc (caller-provided outputs, zero-copy outputs).
  std::vector<ValueId> outRoot;
  std::unordered_map<ValueId, size_t> rootToOut;
  // Caller-provided outputs of the call in progress (run(..., callerOutputs)); outBound[k]: the model writes them.
  std::span<const TensorView> callerOuts;
  std::vector<uint8_t> outBound;
  // Per-input cache of the no-copy GPU wrap of caller memory.
  struct InputWrap {
    void* data = nullptr;
    size_t bytes = 0;
    std::weak_ptr<const void> owner;
    std::shared_ptr<Storage> storage;
    uint64_t offset = 0;
  };
  std::vector<InputWrap> inputWraps;
  std::vector<InputWrap> outWraps;  // per output: cached no-copy GPU wrap of caller memory (caller outputs)
  std::vector<ValueId> outputs;
  HazardTracker hazards;
  // Arena pools shared by the mem.plans of a call (0: CPU-touched buffers, 1: gpu_only). A pool's storage is only
  // (re)allocated at its first plan of a call, when none of its buffers is live; `need` is the high-water mark seen.
  struct ArenaPool {
    std::shared_ptr<Storage> storage;
    int64_t need = 0;
    uint64_t stamp = 0;  // call stamp of the last plan that used the pool
  };
  ArenaPool pools[2];
  uint64_t callStamp = 0;
  int64_t planHits = 0, planMisses = 0, planOverflows = 0;  // statistics (memPlanStats)
  // GPU record/replay (gpu_replay.h; opt-in): enabled by setupReplay at create for eligible models; one recorded
  // command list per input-shape signature (sizes are a function of the input shapes when there is no dim.read).
  bool replayEnabled = false;
  std::vector<std::pair<std::vector<int64_t>, std::unique_ptr<GpuReplay>>> replays;  // most recent last, <= 4
  GpuReplay* replay = nullptr;  // the current call's entry
  size_t replayLastGpu = 0;     // exec index of the last GPU kernel: where the replay is submitted
  bool replayThisCall = false;  // the exec call in progress records or replays
  int gpuKernels = -1;  // GPU kernel instructions in exec (early-commit policy), counted on the first call
  std::string replayNote;       // why replay isn't used (when requested)
  void setupReplay(int option);
  void selectReplay(std::span<const TensorView> inputs);
  // Per-kernel profiler (LOKINFER_PROFILE=1): wall time per kernel name, printed after each exec call;
  // LOKINFER_PROFILE=2 splits rows by the kernel's int args (shapes + int attrs). GPU kernels are serialized
  // while profiling (commit + wait each), so their rows are real GPU time; the overlap is lost.
  int profiling = [] {
    const char* e = std::getenv("LOKINFER_PROFILE");
    return e && *e ? std::atoi(e) : 0;
  }();
  struct ProfRow {
    double ms = 0;
    int64_t calls = 0;
  };
  std::unordered_map<std::string, ProfRow> prof;
  void printProfile(double totalMs);
  // Structured profile (Executable::setProfile, docs/profiling.md): records of the current exec call, then
  // serialized into `recCalls` (JSON objects) until takeProfileJson().
  ProfileOptions profOpts;
  bool recActive = false;  // recording the exec call in progress
  int64_t recStart = 0;    // steady_clock ns at the start of the call
  struct KernelRec {
    uint32_t op;
    int64_t t0, t1;  // ns from the call start
    std::vector<int64_t> ints;
    int64_t waitNs;  // of t1 - t0: CPU blocked on the previous in-flight GPU segment
  };
  struct EventRec {
    uint32_t op;
    int64_t t0, t1;
    const char* what;
  };
  std::vector<KernelRec> recKernels;
  std::vector<EventRec> recEvents;
  std::string recMem;  // JSON objects, comma-separated
  std::vector<std::string> recCalls;
  uint32_t recCallCount = 0;
  bool recTimestamps = false;
  // CLOCK_UPTIME_RAW: the clock Metal's sampleTimestamps() pairs GPU timestamps with (libc++'s steady_clock is
  // CLOCK_MONOTONIC_RAW, which also counts sleep).
  static int64_t nowNs() { return static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)); }
  void recordCall(int64_t totalNs);

  // Dims are evaluated through a per-call memo (DimEvaluator; a new epoch per execute): shared dims such as `N` or
  // `4*N` are computed once per call instead of once per use (symbolic shapes cost ~2x the static per-call overhead
  // before; design-ir.md §3 "dim programs").
  mutable std::unique_ptr<DimEvaluator> dimEval;
  int64_t eval(Dim d) const { return (*dimEval)(d, syms); }
  int64_t typeBytes(Type t) const {
    int64_t n = static_cast<int64_t>(dtypeSize(t->dtype));
    for (Dim d : t->shape) n *= eval(d);
    return n;
  }

  Status lower(const Region& r, std::vector<Instr>& out);
  Status execute(std::vector<Instr>& code);
  void gpuHazards(const Instr& in);
  Status allocSlot(ValueId v, size_t bytes);
  Status bindCallerOutput(ValueId alloc, int64_t bytes);
};

Status Executable::Impl::allocSlot(ValueId v, size_t bytes) {
  auto s = allocateStorage(bytes, gpu);
  if (!s) return Status::error("out of memory allocating " + std::to_string(bytes) + " bytes");
  slots[v] = Slot{static_cast<char*>(s->data), s->mtlBuffer, 0, s, 0};
  return Status::ok();
}

// An exec {output} alloc that is the root of a caller-provided output: point its slot at the caller's memory (left
// empty when it can't be used, e.g. GPU model + ownerless memory: allocated as usual, then copied at the end).
Status Executable::Impl::bindCallerOutput(ValueId alloc, int64_t bytes) {
  slots[alloc] = Slot{};
  auto it = rootToOut.find(alloc);
  if (it == rootToOut.end()) return Status::ok();
  const size_t k = it->second;
  if (k >= callerOuts.size() || !callerOuts[k].data) return Status::ok();
  const TensorView& v = callerOuts[k];
  Type rt = model->typeOf(model->exec.ops.back().operands[k]);
  int64_t vb = static_cast<int64_t>(dtypeSize(v.dtype));
  for (int64_t d : v.shape) vb *= d;
  if (v.dtype != rt->dtype || vb != bytes)
    return Status::error("output " + std::to_string(k) + ": the provided tensor (" + std::string(dtypeName(v.dtype)) +
                         ", " + std::to_string(vb) + " bytes) doesn't match the output (" + dtypeName(rt->dtype) +
                         ", " + std::to_string(bytes) + " bytes)");
  if (!gpu || v.mtlBuffer) {
    slots[alloc] = Slot{static_cast<char*>(v.data), v.mtlBuffer, v.mtlOffset, nullptr, 0};
  } else {
    InputWrap& c = outWraps[k];
    const size_t nb = static_cast<size_t>(bytes);
    if (v.owner && c.storage && c.data == v.data && c.bytes == nb && c.owner.lock() == v.owner) {
      slots[alloc] = Slot{static_cast<char*>(v.data), c.storage->mtlBuffer, c.offset, nullptr, 0};
    } else if (uint64_t off = 0; auto w = wrapStorage(v.data, nb, v.owner, gpu, off)) {
      c = InputWrap{v.data, nb, v.owner, w, off};
      slots[alloc] = Slot{static_cast<char*>(v.data), w->mtlBuffer, off, nullptr, 0};
    } else {
      return Status::ok();  // can't wrap: computed in model memory, copied at the end
    }
  }
  outBound[k] = 1;
  return Status::ok();
}

Status Executable::Impl::lower(const Region& r, std::vector<Instr>& out) {
  const Model& m = *model;
  for (const Op& op : r.ops) {
    Instr in;
    in.kind = op.kind;
    in.op = &op;
    switch (op.kind) {
      case OpKind::Kernel: {
        in.kernel = findKernel(op.name);
        if (!in.kernel) return Status::error("unknown kernel '" + op.name + "'");
        // The compiler derives devices from names (isGpuKernel, the IR verifier); the registry must agree.
        if ((in.kernel->device == Device::Gpu) != kernelNameIsGpu(op.name))
          return Status::error("kernel '" + op.name + "' is registered on a device its name doesn't say");
        if (!op.results.empty()) return Status::error("kernel '" + op.name + "' still has results (run bufferize)");
        in.args = op.operands;
        in.args.insert(in.args.end(), op.outs.begin(), op.outs.end());
        auto addDim = [&](Dim d) {
          if (d->isConst()) {
            in.ints.push_back(d->value);
          } else {
            in.dynInts.push_back({static_cast<uint32_t>(in.ints.size()), d});
            in.ints.push_back(0);
          }
        };
        for (ValueId v : in.args)
          for (Dim d : m.typeOf(v)->shape) addDim(d);
        for (const NamedAttr& a : op.attrs) {
          if (isStructuralAttr(a.name)) continue;  // compiler-only, never part of the kernel ABI
          const auto& val = a.attr.value;
          if (auto* b = std::get_if<bool>(&val)) in.ints.push_back(*b);
          else if (auto* i = std::get_if<int64_t>(&val)) in.ints.push_back(*i);
          else if (auto* f = std::get_if<double>(&val)) in.floats.push_back(*f);
          else if (auto* is = std::get_if<Attr::Ints>(&val)) in.ints.insert(in.ints.end(), is->begin(), is->end());
          else if (auto* fs = std::get_if<Attr::Floats>(&val)) in.floats.insert(in.floats.end(), fs->begin(), fs->end());
          else if (auto* d = std::get_if<Dim>(&val)) addDim(*d);
        }
        if (in.kernel->device == Device::Gpu && !gpu) return Status::error("GPU kernel '" + op.name + "' but no GPU context");
        if (in.kernel->prepare) {
          PrepareCtx pc{model.get(), gpu};
          LK_RETURN_IF_ERROR(in.kernel->prepare(pc, op, &in.state));
        }
        in.ptrs.resize(in.args.size());
        in.bufs.resize(in.args.size());
        in.numOperands = static_cast<uint32_t>(op.operands.size());
        for (ValueId v : in.args) {
          in.argRank.push_back(static_cast<uint32_t>(m.typeOf(v)->shape.size()));
          in.argElem.push_back(static_cast<uint32_t>(dtypeSize(m.typeOf(v)->dtype)));
        }
        break;
      }
      case OpKind::View:
        in.args = op.operands;
        in.result = op.results[0];
        in.offset = op.dims[0];
        in.elemSize = static_cast<int64_t>(dtypeSize(m.typeOf(op.results[0])->dtype));
        {
          bool allConst = op.dims[0]->isConst();
          for (Type t : {m.typeOf(op.operands[0]), m.typeOf(op.results[0])})
            for (Dim d : t->shape) allConst = allConst && d->isConst();
          in.checkBounds = !allConst;
        }
        break;
      case OpKind::MemPlan: {
        in.result = op.results[0];
        in.planBufs.resize(op.dims.size());
        for (size_t k = 0; k < op.dims.size(); ++k)
          in.planBufs[k] = {0, op.ints[2 * k], op.ints[2 * k + 1]};
        in.planElem.assign(op.dims.size(), 1);
        if (const Attr* e = op.attr("elem"))
          if (auto* v = std::get_if<Attr::Ints>(&e->value); v && v->size() == op.dims.size()) in.planElem = *v;
        in.planPool = op.hasFlag("gpu_only") ? 1 : 0;
        std::vector<int64_t> keySyms;
        for (size_t j = 0; j < out.size(); ++j)
          if (out[j].kind == OpKind::MemPlan && out[j].planPool == in.planPool) {
            in.planPrior.push_back(static_cast<uint32_t>(j));
            keySyms.insert(keySyms.end(), out[j].planKeySyms.begin(), out[j].planKeySyms.end());
          }
        for (Dim d : op.dims) collectSymbols(d, keySyms);
        std::sort(keySyms.begin(), keySyms.end());
        keySyms.erase(std::unique(keySyms.begin(), keySyms.end()), keySyms.end());
        in.planKeySyms = std::move(keySyms);
        break;
      }
      case OpKind::ConstInline:
      case OpKind::ConstInternal:
      case OpKind::ConstExternal:
      case OpKind::BufferAlloc:
        in.result = op.results[0];
        break;
      case OpKind::BufferFree:
      case OpKind::DimRead:
      case OpKind::Return:
        in.args = op.operands;
        break;
      case OpKind::Sync:
        in.args = op.operands;
        if (!op.results.empty()) in.result = op.results[0];
        break;
    }
    out.push_back(std::move(in));
  }
  return Status::ok();
}

// Decides whether the next GPU kernel needs a barrier: only if its byte
// ranges conflict (RAW, WAW, WAR) with dispatches since the last barrier.
void Executable::Impl::gpuHazards(const Instr& in) {
  HazardTracker& h = hazards;
  if (h.segment != gpu->segmentId()) {
    h.writes.clear();
    h.reads.clear();
    h.segment = gpu->segmentId();
  }
  std::vector<HazardTracker::Range> rs, ws;
  std::vector<bool> written(in.args.size(), false);  // outs, plus `inout` operands (written in place)
  for (size_t i = in.numOperands; i < in.args.size(); ++i) written[i] = true;
  if (const Attr* io = in.op->attr("inout"))
    if (auto* idx = std::get_if<Attr::Ints>(&io->value))
      for (int64_t j : *idx)
        if (j >= 0 && static_cast<size_t>(j) < in.numOperands) written[j] = true;
  size_t dimAt = 0;
  for (size_t i = 0; i < in.args.size(); ++i) {
    uint64_t bytes = in.argElem[i];
    for (uint32_t d = 0; d < in.argRank[i]; ++d) bytes *= static_cast<uint64_t>(in.ints[dimAt + d]);
    dimAt += in.argRank[i];
    if (!in.bufs[i].mtlBuffer) continue;  // host scalars
    HazardTracker::Range r{in.bufs[i].mtlBuffer, in.bufs[i].offset, in.bufs[i].offset + bytes};
    (written[i] ? ws : rs).push_back(r);
  }
  bool conflict = false;
  for (const auto& r : rs) conflict = conflict || HazardTracker::overlaps(h.writes, r);
  for (const auto& w : ws) conflict = conflict || HazardTracker::overlaps(h.writes, w) || HazardTracker::overlaps(h.reads, w);
  if (conflict) {
    h.writes.clear();
    h.reads.clear();
  }
  gpu->setNextBarrier(conflict);
  h.reads.insert(h.reads.end(), rs.begin(), rs.end());
  h.writes.insert(h.writes.end(), ws.begin(), ws.end());
}

void Executable::Impl::printProfile(double totalMs) {
  std::vector<std::pair<std::string, ProfRow>> rows(prof.begin(), prof.end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
  double kernMs = 0;
  for (const auto& r : rows) kernMs += r.second.ms;
  std::fprintf(stderr, "lokinfer profile: call %.2f ms, kernels %.2f ms\n", totalMs, kernMs);
  for (const auto& [name, r] : rows)
    std::fprintf(stderr, "  %9.3f ms %5.1f%%  x%-5lld %s\n", r.ms, 100.0 * r.ms / std::max(totalMs, 1e-9),
                 static_cast<long long>(r.calls), name.c_str());
  prof.clear();
}

constexpr int kEarlyCommit = 2;  // see the early commit in execute()
// iPhone: splitting a tiny graph into two submissions costs more than the overlap gains (MNIST GPU B=1, 3 kernels:
// 0.15 -> 0.17 ms with early commit; GPT-2 prefill T=128 11.44 -> 11.30, MNIST B=1024 0.68 -> 0.66) — only graphs
// with enough GPU kernels. The Mac gains even on MNIST (0.262 -> 0.216 ms).
#if TARGET_OS_IPHONE
constexpr int kEarlyMinKernels = 8;
#else
constexpr int kEarlyMinKernels = 1;
#endif

Status Executable::Impl::execute(std::vector<Instr>& code) {
  const Model& m = *model;
  ExecCtx ctx{gpu};
  ++callStamp;  // mem.plan arena pools: a new call (nothing planned is live)
  dimEval->newEpoch();  // symbols were (re)bound for this call
  int earlyCount = 0;  // GPU kernels encoded so far in this call (early commit)
  if (gpuKernels < 0) {  // GPU kernels in exec (counted once)
    gpuKernels = 0;
    for (const Instr& i : exec)
      if (i.op->kind == OpKind::Kernel && i.kernel && i.kernel->device == Device::Gpu) ++gpuKernels;
  }
  for (size_t idx = 0; idx < code.size(); ++idx) {
    Instr& in = code[idx];
    const Op& op = *in.op;
    switch (in.kind) {
      case OpKind::Kernel: {
        for (const DynInt& d : in.dynInts) in.ints[d.index] = eval(d.dim);
        for (size_t i = 0; i < in.args.size(); ++i) {
          Slot& s = slots[in.args[i]];
          in.ptrs[i] = m.typeOf(in.args[i])->isScalar() ? static_cast<void*>(&s.scalar) : s.ptr;
          in.bufs[i] = BufRef{s.mtl, s.off};
        }
        KernelArgs ka{in.ptrs.data(), in.bufs.data(), in.ints.data(), in.floats.data(), in.state,
                      static_cast<uint32_t>(in.ptrs.size()), static_cast<uint32_t>(in.ints.size())};
        if (in.kernel->device == Device::Gpu) {
          gpuHazards(in);
          if (replayThisCall && &code == &exec) {
            const auto n = static_cast<uint32_t>(in.bufs.size());
            if (replay->ready()) {  // replay: re-bind moved arguments instead of encoding; the last kernel submits
              replay->replayKernel(static_cast<uint32_t>(idx), in.bufs.data(), n, gpu->nextBarrierPending());
              gpu->setNextBarrier(true);
              if (idx == replayLastGpu) replay->submit();
              break;
            }
            replay->kernelBegin(static_cast<uint32_t>(idx), in.bufs.data(), n);  // recording (first call)
          }
        } else if (gpu && gpu->hasOpenSegment()) {
          // A CPU kernel never reads GPU results without an explicit `sync gpu_to_cpu` first, so the pending
          // GPU work is independent of it: start the GPU on it now and overlap.
          if (recActive) recEvents.push_back({static_cast<uint32_t>(idx), nowNs() - recStart, -1, "commit"});
          gpu->commitAsync();
        }
        if (recActive) {
          const bool onGpu = in.kernel->device == Device::Gpu;
          if (onGpu) gpu->profileKernelBegin(static_cast<int32_t>(idx));
          if (gpu) gpu->profileTakeWaitNs();  // (a commitAsync above never waits; start clean)
          const int64_t t0 = nowNs();
          in.kernel->run(ctx, ka);
          if (onGpu) {
            gpu->profileKernelEnd();
            if (profiling || profOpts.gpuSerialize) gpu->commitAndWait();
          }
          const int64_t t1 = nowNs();
          recKernels.push_back(
              {static_cast<uint32_t>(idx), t0 - recStart, t1 - recStart, in.ints, gpu ? gpu->profileTakeWaitNs() : 0});
        } else if (profiling) {
          auto t0 = std::chrono::steady_clock::now();
          in.kernel->run(ctx, ka);
          // Profile mode serializes GPU kernels (commit + wait after each) so each row includes its GPU execution
          // time. This removes CPU/GPU overlap, so the profiled call total is higher than a normal call.
          if (in.kernel->device == Device::Gpu) gpu->commitAndWait();
          std::string key = op.name;
          if (profiling >= 2) {
            key += " [";
            for (size_t i = 0; i < in.ints.size(); ++i) key += (i ? "," : "") + std::to_string(in.ints[i]);
            key += "]";
          }
          ProfRow& r = prof[key];
          r.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          ++r.calls;
        } else {
          in.kernel->run(ctx, ka);
          // Early commit: once the call's first kEarlyCommit GPU kernels are encoded, commit them so the GPU starts
          // (~75 us submit latency) while the CPU encodes the rest into the next segment (2 may be in flight).
          // Mac GPU 2026-11-02: MNIST B=1 0.262 -> 0.216 ms, GPT-2 decode 2.10 -> 2.04, prefill T=128 5.21 -> 4.90,
          // MobileNetV2 N=1 1.106 -> 1.088, Kokoro app59f unchanged. LOKINFER_GPU_EARLY_COMMIT=N overrides (0 = off).
          static const int early = [] {
            const char* e = std::getenv("LOKINFER_GPU_EARLY_COMMIT");
            return e ? std::atoi(e) : kEarlyCommit;
          }();
          // Needs a second command buffer to encode into while the first runs (maxSegments >= 2; iOS defaults to 1).
          if (early > 0 && in.kernel->device == Device::Gpu && &code == &exec && gpuKernels >= kEarlyMinKernels &&
              gpu->maxSegments() >= 2 && ++earlyCount == early)
            gpu->commitAsync();
        }
        if (ctx.error) return Status::error("kernel '" + op.name + "': " + ctx.error);
        break;
      }
      case OpKind::View: {
        const Slot& src = slots[in.args[0]];
        int64_t bytes = eval(in.offset) * in.elemSize;
        if (in.checkBounds) {
          int64_t srcBytes = typeBytes(m.typeOf(in.args[0])), dstBytes = typeBytes(m.typeOf(in.result));
          if (bytes < 0 || bytes + dstBytes > srcBytes)
            return Status::error("view out of bounds: " + std::to_string(bytes + dstBytes) + " > " +
                                 std::to_string(srcBytes) + " bytes");
        }
        slots[in.result] = Slot{src.ptr + bytes, src.mtl, src.off + static_cast<uint64_t>(bytes), nullptr, 0};
        break;
      }
      case OpKind::ConstInline:
        slots[in.result] = Slot{const_cast<char*>(reinterpret_cast<const char*>(op.data.data())), nullptr, 0, nullptr, 0};
        if (m.typeOf(in.result)->isScalar()) std::memcpy(&slots[in.result].scalar, op.data.data(), op.data.size());
        if (gpu && !m.typeOf(in.result)->isScalar()) {
          // Inline constants used by GPU kernels need GPU-visible memory.
          LK_RETURN_IF_ERROR(allocSlot(in.result, op.data.size()));
          std::memcpy(slots[in.result].ptr, op.data.data(), op.data.size());
        }
        break;
      case OpKind::ConstInternal: {
        const ConstantData* c = m.findConstant(op.name);
        if (!c || !c->bytes) return Status::error("internal constant '" + op.name + "' has no data");
        if (gpu) {
          // .lkm data is 16 KiB-aligned in the mmapped file: wrap it, no copy.
          uint64_t off = 0;
          if (auto w = wrapStorage(const_cast<uint8_t*>(c->bytes), c->size, c->owner, gpu, off)) {
            slots[in.result] = Slot{static_cast<char*>(w->data) + off, w->mtlBuffer, off, w, 0};
            break;
          }
          LK_RETURN_IF_ERROR(allocSlot(in.result, c->size));
          std::memcpy(slots[in.result].ptr, c->bytes, c->size);
        } else {
          slots[in.result] = Slot{const_cast<char*>(reinterpret_cast<const char*>(c->bytes)), nullptr, 0, nullptr, 0};
        }
        break;
      }
      case OpKind::ConstExternal: {
        auto it = std::find_if(externals.begin(), externals.end(), [&](auto& e) { return e.name == op.name; });
        if (it == externals.end()) return Status::error("external constant '" + op.name + "' is not bound");
        const TensorView& v = it->view;
        if (gpu && !v.mtlBuffer) {
          size_t bytes = dtypeSize(v.dtype);
          for (int64_t d : v.shape) bytes *= static_cast<size_t>(d);
          uint64_t off = 0;
          std::shared_ptr<const void> keep = it->keepAlive;
          if (!keep) keep = v.owner;
          if (auto w = wrapStorage(v.data, bytes, keep, gpu, off)) {
            slots[in.result] = Slot{static_cast<char*>(w->data) + off, w->mtlBuffer, off, w, 0};
            break;
          }
          LK_RETURN_IF_ERROR(allocSlot(in.result, bytes));
          std::memcpy(slots[in.result].ptr, v.data, bytes);
        } else {
          slots[in.result] = Slot{static_cast<char*>(v.data), v.mtlBuffer, v.mtlOffset, nullptr, 0};
        }
        break;
      }
      case OpKind::BufferAlloc: {
        Type t = m.typeOf(in.result);
        int64_t bytes = static_cast<int64_t>(dtypeSize(t->dtype));
        for (Dim d : t->shape) bytes *= eval(d);
        if (!callerOuts.empty() && &code == &exec) {
          LK_RETURN_IF_ERROR(bindCallerOutput(in.result, bytes));
          if (slots[in.result].ptr) break;
        }
        LK_RETURN_IF_ERROR(allocSlot(in.result, static_cast<size_t>(bytes)));
        if (recActive)
          recMem += (recMem.empty() ? "" : ",") + std::string("{\"op\":") + std::to_string(idx) +
                    ",\"kind\":\"alloc\",\"bytes\":" + std::to_string(bytes) +
                    ",\"output\":" + (op.hasFlag("output") ? "true" : "false") + "}";
        break;
      }
      case OpKind::BufferFree:
        slots[in.args[0]] = Slot{};
        break;
      case OpKind::MemPlan: {
        // Plans repeat for repeated symbol values: look the result up first (no size evaluation, no planning).
        const int64_t align = op.ints.back();
        int hit = -1;
        for (size_t c = 0; c < in.planCache.size() && hit < 0; ++c) {
          bool same = true;
          for (size_t q = 0; q < in.planKeySyms.size() && same; ++q)
            same = in.planCache[c].key[q] == syms[in.planKeySyms[q]];
          if (same) hit = static_cast<int>(c);
        }
        if (hit < 0) {
          ++planMisses;
          for (size_t k = 0; k < op.dims.size(); ++k) in.planBufs[k].size = eval(op.dims[k]);
          std::vector<PlacedBuffer> fixed;  // this call's buffers of the earlier plans sharing the pool's arena
          for (uint32_t j : in.planPrior) {
            const Instr& p = code[j];
            if (p.planLast < 0) continue;
            const Instr::PlanEntry& e = p.planCache[p.planLast];
            for (size_t k = 0; k < e.offsets.size(); ++k)
              fixed.push_back({e.offsets[k], (std::max<int64_t>(e.sizes[k], 1) + align - 1) / align * align,
                               p.planBufs[k].start, p.planBufs[k].end});
          }
          PlanResult pr = planOffsets(in.planBufs, align, fixed);
          Instr::PlanEntry e;
          e.key.reserve(in.planKeySyms.size());
          for (int64_t sym : in.planKeySyms) e.key.push_back(syms[sym]);
          e.offsets = std::move(pr.offsets);
          e.sizes.reserve(in.planBufs.size());
          for (const PlanBuffer& b : in.planBufs) e.sizes.push_back(b.size);
          e.total = pr.total;
          if (in.planCache.size() >= kPlanCacheSize) in.planCache.pop_back();
          in.planCache.insert(in.planCache.begin(), std::move(e));
        } else {
          ++planHits;
          if (hit > 0) std::rotate(in.planCache.begin(), in.planCache.begin() + hit, in.planCache.begin() + hit + 1);
        }
        in.planLast = 0;
        const Instr::PlanEntry& plan = in.planCache[0];
        // The pool's arena: grown only at its first plan of a call (nothing of it is live then); a later plan that
        // doesn't fit gets its own overflow storage for this call and raises the pool's need for the next call.
        ArenaPool& pool = pools[in.planPool];
        Storage* arena = nullptr;
        if (pool.stamp != callStamp) {
          pool.stamp = callStamp;
          pool.need = std::max(pool.need, plan.total);
          if (!pool.storage || pool.storage->size < static_cast<size_t>(pool.need)) {
            pool.storage.reset();  // release the old one first (its bytes count against the device limit)
            pool.storage = allocateStorage(static_cast<size_t>(pool.need + pool.need / 2), gpu);  // slack for growth
            if (!pool.storage) pool.storage = allocateStorage(static_cast<size_t>(pool.need), gpu);  // tight retry
            if (!pool.storage)
              return Status::error("out of memory for arena (" + std::to_string(pool.need >> 20) + " MiB)");
            if (profiling)
              std::fprintf(stderr, "lokinfer: arena pool %d: %lld MiB\n", in.planPool,
                           static_cast<long long>(pool.storage->size >> 20));
          }
          in.planOverflow.reset();
          arena = pool.storage.get();
        } else if (pool.storage && pool.storage->size >= static_cast<size_t>(plan.total)) {
          in.planOverflow.reset();
          arena = pool.storage.get();
        } else {
          pool.need = std::max(pool.need, plan.total);
          ++planOverflows;
          if (!in.planOverflow || in.planOverflow->size < static_cast<size_t>(plan.total))
            in.planOverflow = allocateStorage(static_cast<size_t>(plan.total), gpu);
          if (!in.planOverflow)
            return Status::error("out of memory for arena (" + std::to_string(plan.total >> 20) + " MiB)");
          arena = in.planOverflow.get();
        }
        if (recActive) {
          std::string j = "{\"op\":" + std::to_string(idx) + ",\"kind\":\"plan\",\"total\":" +
                          std::to_string(plan.total) + ",\"arena\":" + std::to_string(arena->size) +
                          ",\"align\":" + std::to_string(align) + ",\"cached\":" + (hit >= 0 ? "true" : "false") +
                          ",\"pool\":" + std::to_string(in.planPool) + ",\"buffers\":[";
          for (size_t k = 0; k < in.planBufs.size(); ++k)
            j += (k ? ",[" : "[") + std::to_string(plan.sizes[k]) + "," + std::to_string(in.planBufs[k].start) +
                 "," + std::to_string(in.planBufs[k].end) + "," + std::to_string(plan.offsets[k]) + "]";
          recMem += (recMem.empty() ? "" : ",") + j + "]}";
        }
        syms[op.symbols[0]] = plan.total;
        for (size_t k = 0; k < plan.offsets.size(); ++k) syms[op.symbols[k + 1]] = plan.offsets[k] / in.planElem[k];
        slots[in.result] = Slot{static_cast<char*>(arena->data), arena->mtlBuffer, 0, nullptr, 0};
        break;
      }
      case OpKind::DimRead: {
        const Slot& s = slots[in.args[0]];
        DType dt = m.typeOf(in.args[0])->dtype;
        int64_t idx = op.ints[0], v;
        if (dt == DType::I64) std::memcpy(&v, s.ptr + idx * 8, 8);
        else {
          int32_t v32;
          std::memcpy(&v32, s.ptr + idx * 4, 4);
          v = v32;
        }
        int64_t& sym = syms[op.symbols[0]];
        // The symbol may already be bound by an input shape (e.g. a noise
        // input sized by the frame count): the computed value must agree.
        if (sym != kUnbound && sym != v)
          return Status::error("dim.read " + m.dims().symbolName(op.symbols[0]) + " = " + std::to_string(v) +
                               " disagrees with the value bound from inputs (" + std::to_string(sym) + ")");
        sym = v;
        break;
      }
      case OpKind::Sync:
        if (gpu && recActive) {
          const int64_t t0 = nowNs();
          gpuSync(*gpu, static_cast<SyncKind>(op.ints[0]));
          recEvents.push_back({static_cast<uint32_t>(idx), t0 - recStart, nowNs() - recStart,
                               static_cast<SyncKind>(op.ints[0]) == SyncKind::GpuToCpu ? "sync_to_cpu" : "sync_to_gpu"});
        } else if (gpu && profiling) {  // the GPU's work shows up here (kernels only encode)
          auto t0 = std::chrono::steady_clock::now();
          gpuSync(*gpu, static_cast<SyncKind>(op.ints[0]));
          ProfRow& r = prof["(sync: GPU work + wait)"];
          r.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          ++r.calls;
        } else if (gpu) {
          gpuSync(*gpu, static_cast<SyncKind>(op.ints[0]));
        }
        if (gpu) {  // CPU work must not read the results of a failed GPU segment
          if (std::string e = gpu->takeGpuError(); !e.empty()) return Status::error(e);
        }
        if (in.result != kNoValue) {  // value form: alias (unified memory, nothing to copy)
          const Slot& src = slots[in.args[0]];
          slots[in.result] = Slot{src.ptr, src.mtl, src.off, nullptr, src.scalar};
        }
        break;
      case OpKind::Return:
        outputs = in.args;
        break;
    }
  }
  if (gpu && recActive) {
    const int64_t t0 = nowNs();
    gpuFinish(*gpu);
    recEvents.push_back({static_cast<uint32_t>(code.size()), t0 - recStart, nowNs() - recStart, "finish"});
  } else if (gpu) {
    gpuFinish(*gpu);
  }
  return Status::ok();
}

Executable::Executable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Executable::~Executable() {
  if (!impl_) return;
  for (auto* code : {&impl_->init, &impl_->exec})
    for (Instr& in : *code)
      if (in.state && in.kernel && in.kernel->destroy) in.kernel->destroy(in.state);
}

bool Executable::usesGpu() const { return impl_->gpu != nullptr; }

const void* Executable::stateData(size_t i) const {
  const auto& st = impl_->model->exec.state;
  return i < st.size() ? impl_->slots[st[i]].ptr : nullptr;
}

Executable::MemPlanStats Executable::memPlanStats() const {
  MemPlanStats st;
  st.hits = impl_->planHits;
  st.misses = impl_->planMisses;
  st.overflows = impl_->planOverflows;
  for (int p = 0; p < 2; ++p) st.arenaBytes[p] = impl_->pools[p].storage ? impl_->pools[p].storage->size : 0;
  return st;
}

Expected<std::unique_ptr<Executable>> Executable::create(std::shared_ptr<const Model> model,
                                                         std::vector<ExternalBinding> externals,
                                                         const RuntimeOptions& opts) {
  auto impl = std::make_unique<Impl>();
  impl->model = model;
  impl->externals = std::move(externals);
  impl->slots.resize(model->numValues());
  impl->inputWraps.resize(model->exec.args.size());
  impl->syms.assign(model->dims().numSymbols(), kUnbound);
  impl->dimEval = std::make_unique<DimEvaluator>(model->dims());

  bool needsGpu = false;
  for (const Region* r : {&model->init, &model->exec})
    for (const Op& op : r->ops)
      if (op.kind == OpKind::Kernel)
        if (const KernelEntry* k = findKernel(op.name); k && k->device == Device::Gpu) needsGpu = true;
  if (needsGpu) {
    if (!opts.enableGpu) return Status::error("model has GPU kernels but GPU is disabled");
    auto g = GpuContext::shared();
    if (!g) return g.status();
    impl->gpu = *g;
  }

  LK_RETURN_IF_ERROR(impl->lower(model->init, impl->init));
  LK_RETURN_IF_ERROR(impl->lower(model->exec, impl->exec));
  for (const Op& op : model->exec.ops)
    if (op.kind == OpKind::BufferAlloc && op.hasFlag("output")) impl->freshOutputs.insert(op.results[0]);
  {
    // Return operands that are (zero-offset views of) the call's own output allocs.
    std::unordered_map<ValueId, ValueId> zeroView;  // view result -> source (constant offset 0 only)
    for (const Op& op : model->exec.ops)
      if (op.kind == OpKind::View && op.dims[0]->isConst(0)) zeroView[op.results[0]] = op.operands[0];
    const auto& rets = model->exec.ops.back().operands;
    impl->outRoot.assign(rets.size(), kNoValue);
    for (size_t k = 0; k < rets.size(); ++k) {
      ValueId v = rets[k];
      for (auto z = zeroView.find(v); z != zeroView.end(); z = zeroView.find(v)) v = z->second;
      if (!impl->freshOutputs.count(v) || impl->rootToOut.count(v)) continue;
      impl->outRoot[k] = v;
      impl->rootToOut[v] = k;
    }
    impl->outWraps.resize(rets.size());
  }
  LK_RETURN_IF_ERROR(impl->execute(impl->init));
  if (impl->gpu)
    if (std::string e = impl->gpu->takeGpuError(); !e.empty()) return Status::error("init: " + e);
  // Init results -> exec state slots.
  const auto& initOuts = impl->outputs;
  if (initOuts.size() != model->exec.state.size()) return Status::error("init/exec state mismatch");
  for (size_t i = 0; i < initOuts.size(); ++i) impl->slots[model->exec.state[i]] = impl->slots[initOuts[i]];
  impl->outputs.clear();
  impl->initSyms = impl->syms;
  impl->setupReplay(opts.gpuReplay);

  std::unique_ptr<Executable> e(new Executable(std::move(impl)));
  e->model_ = std::move(model);
  return e;
}

Status Executable::run(std::span<const TensorView> inputs, std::vector<OwnedTensor>& outputs) {
  return run(inputs, outputs, {});
}

Status Executable::run(std::span<const TensorView> inputs, std::vector<OwnedTensor>& outputs,
                       std::span<const TensorView> callerOutputs) {
  Impl& im = *impl_;
  std::unique_lock<std::recursive_mutex> gpuLock;  // calls on other threads share the GPU context
  if (im.gpu) gpuLock = std::unique_lock<std::recursive_mutex>(im.gpu->mutex());
  const Model& m = *im.model;
  const size_t numOut = m.exec.ops.back().operands.size();
  if (!callerOutputs.empty() && callerOutputs.size() != numOut)
    return Status::error("expected " + std::to_string(numOut) + " output tensors, got " +
                         std::to_string(callerOutputs.size()));
  bool anyCaller = false;
  for (const TensorView& v : callerOutputs) anyCaller = anyCaller || v.data;
  im.callerOuts = anyCaller ? callerOutputs : std::span<const TensorView>{};
  im.outBound.assign(numOut, 0);
  struct ClearCallerOuts {  // the span refers to the caller's memory: never keep it past this call
    Impl& im;
    ~ClearCallerOuts() { im.callerOuts = {}; }
  } clearCallerOuts{im};
  const auto& args = m.exec.args;
  if (inputs.size() != args.size())
    return Status::error("expected " + std::to_string(args.size()) + " inputs, got " + std::to_string(inputs.size()));

  // Bind symbols. Symbols defined while running init survive every call.
  // A dim linear in one unbound symbol (k*S + b, e.g. 600*F) binds it; repeat
  // until nothing changes, then every dim must evaluate to the input's size.
  im.syms = im.initSyms;
  for (size_t i = 0; i < args.size(); ++i) {
    Type t = m.typeOf(args[i]);
    const TensorView& v = inputs[i];
    if (v.dtype != t->dtype) return Status::error("input " + std::to_string(i) + ": dtype mismatch");
    if (v.shape.size() != t->shape.size())
      return Status::error("input " + std::to_string(i) + ": expected rank " + std::to_string(t->shape.size()));
    for (int64_t d : v.shape)
      if (d < 0) return Status::error("input " + std::to_string(i) + ": negative dimension");
  }
  for (bool progress = true; progress;) {
    progress = false;
    for (size_t i = 0; i < args.size(); ++i) {
      Type t = m.typeOf(args[i]);
      for (size_t d = 0; d < t->shape.size(); ++d) {
        int64_t sym = -1, k = 0, b = 0;
        if (!linearInOneSymbol(t->shape[d], im.syms, kUnbound, sym, k, b)) continue;
        int64_t num = inputs[i].shape[d] - b;
        if (k == 0 || num % k != 0 || num / k < 0)
          return Status::error("input " + std::to_string(i) + " dim " + std::to_string(d) + ": " +
                               std::to_string(inputs[i].shape[d]) + " is not of the form " +
                               m.dims().toString(t->shape[d]));
        im.syms[sym] = num / k;
        progress = true;
      }
    }
  }
  for (size_t i = 0; i < args.size(); ++i) {
    Type t = m.typeOf(args[i]);
    const TensorView& v = inputs[i];
    for (size_t d = 0; d < t->shape.size(); ++d) {
      auto expected = m.dims().tryEvaluate(t->shape[d], im.syms, kUnbound);
      if (!expected)
        return Status::error("input " + std::to_string(i) + " dim " + std::to_string(d) + ": cannot bind " +
                             m.dims().toString(t->shape[d]) + " from the inputs");
      if (*expected != v.shape[d])
        return Status::error("input " + std::to_string(i) + " dim " + std::to_string(d) + ": expected " +
                             m.dims().toString(t->shape[d]) + " = " + std::to_string(*expected) + ", got " +
                             std::to_string(v.shape[d]));
    }
    Slot& s = im.slots[args[i]];
    if (t->isScalar()) {
      s = Slot{};
      std::memcpy(&s.scalar, v.data, dtypeSize(t->dtype));
    } else if (im.gpu && !v.mtlBuffer) {
      // GPU models need GPU-visible inputs: wrap the caller's memory with no
      // copy when it comes with an owner (cached per input, so calling again
      // with the same tensor costs nothing); copy only ownerless raw memory.
      size_t bytes = dtypeSize(t->dtype);
      for (int64_t d : v.shape) bytes *= static_cast<size_t>(d);
      Impl::InputWrap& c = im.inputWraps[i];
      if (v.owner && c.storage && c.data == v.data && c.bytes == bytes && c.owner.lock() == v.owner) {
        s = Slot{static_cast<char*>(v.data), c.storage->mtlBuffer, c.offset, nullptr, 0};
      } else if (uint64_t off = 0; auto w = wrapStorage(v.data, bytes, v.owner, im.gpu, off)) {
        c = Impl::InputWrap{v.data, bytes, v.owner, w, off};
        s = Slot{static_cast<char*>(v.data), w->mtlBuffer, off, nullptr, 0};
      } else {
        c = Impl::InputWrap{};
        LK_RETURN_IF_ERROR(im.allocSlot(args[i], bytes));
        std::memcpy(s.ptr, v.data, bytes);
      }
    } else {
      s = Slot{static_cast<char*>(v.data), v.mtlBuffer, v.mtlOffset, nullptr, 0};
    }
  }

  const auto callStart = std::chrono::steady_clock::now();
  if (im.profOpts.level > 0) {
    im.recActive = true;
    im.recStart = Impl::nowNs();
    im.recKernels.clear();
    im.recEvents.clear();
    im.recMem.clear();
    // Two stamps per GPU kernel; Kokoro has ~1-2k kernels per call.
    im.recTimestamps = im.gpu && im.gpu->profileStart(static_cast<uint32_t>(2 * im.exec.size() + 64));
  }
  if (im.replayEnabled) im.selectReplay(inputs);
  im.replayThisCall = im.replay && !im.replay->off() && !im.recActive && !im.profiling;
  const bool recording = im.replayThisCall && !im.replay->ready();
  if (recording) im.replay->beginRecord();
  Status st = im.execute(im.exec);
  if (im.gpu && st.isOk())
    if (std::string e = im.gpu->takeGpuError(); !e.empty()) st = Status::error(e);
  if (recording) im.replay->endRecord(st.isOk());
  im.replayThisCall = false;
  if (im.recActive) {
    im.recActive = false;
    im.recordCall(Impl::nowNs() - im.recStart);
  }
  LK_RETURN_IF_ERROR(st);
  if (im.profiling)
    im.printProfile(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - callStart).count());

  outputs.clear();
  for (size_t k = 0; k < im.outputs.size(); ++k) {
    ValueId v = im.outputs[k];
    Type t = m.typeOf(v);
    OwnedTensor o;
    o.dtype = t->dtype;
    for (Dim d : t->shape) o.shape.push_back(im.eval(d));
    Slot& s = im.slots[v];
    const ValueId root = k < im.outRoot.size() ? im.outRoot[k] : kNoValue;
    if (k < im.callerOuts.size() && im.callerOuts[k].data) {
      // Caller-provided output: written in place (outBound), else copied into it now.
      const TensorView& cv = im.callerOuts[k];
      if (cv.dtype != o.dtype || cv.shape != o.shape)
        return Status::error("output " + std::to_string(k) + ": the provided tensor's dtype/shape differs from the output");
      if (!im.outBound[k]) {
        size_t bytes = dtypeSize(t->dtype);
        for (int64_t d : o.shape) bytes *= static_cast<size_t>(d);
        std::memcpy(cv.data, t->isScalar() ? static_cast<void*>(&s.scalar) : s.ptr, bytes);
      }
      outputs.push_back(std::move(o));  // storage stays null: the data is in the caller's memory
      continue;
    }
    // Zero-copy only for this call's own output allocations (or zero-offset views of them); inputs, state and
    // constants are copied so callers never alias model memory.
    if (root != kNoValue && im.slots[root].owned && s.ptr == im.slots[root].owned->data) {
      o.storage = im.slots[root].owned;
    } else {
      size_t bytes = dtypeSize(t->dtype);
      for (int64_t d : o.shape) bytes *= static_cast<size_t>(d);
      o.storage = allocateStorage(bytes, nullptr);
      if (!o.storage) return Status::error("out of memory for output");
      std::memcpy(o.storage->data, t->isScalar() ? static_cast<void*>(&s.scalar) : s.ptr, bytes);
    }
    outputs.push_back(std::move(o));
  }
  // Drop per-call output ownership from slots so the next call allocates fresh.
  for (ValueId v : im.outRoot)
    if (v != kNoValue) im.slots[v].owned.reset();
  for (ValueId v : im.outputs)
    if (im.freshOutputs.count(v)) im.slots[v].owned.reset();
  return Status::ok();
}

void Executable::Impl::recordCall(int64_t totalNs) {
  std::vector<GpuContext::ProfStamp> stamps;
  std::vector<GpuContext::ProfDispatch> disp;
  if (gpu) gpu->profileStop(stamps, disp);
  std::unordered_map<int32_t, const GpuContext::ProfStamp*> byTag;
  for (const auto& s : stamps) byTag[s.tag] = &s;
  std::string j = "{\"call\":" + std::to_string(recCallCount++) + ",\"total_ns\":" + std::to_string(totalNs) +
                  ",\"gpu_mode\":\"" +
                  (gpu ? (profiling || profOpts.gpuSerialize ? "serialize" : "stream") : "none") +
                  "\",\"gpu_timestamps\":" + (recTimestamps ? "true" : "false") + ",\"kernels\":[";
  // [op, cpu t0, cpu t1, gpu t0, gpu t1, [ints], cpu wait] (ns from the call start; gpu -1 when not measured; cpu
  // wait = the part of t1 - t0 spent blocked on the previous in-flight GPU segment)
  for (size_t i = 0; i < recKernels.size(); ++i) {
    const KernelRec& k = recKernels[i];
    int64_t g0 = -1, g1 = -1;
    if (auto it = byTag.find(static_cast<int32_t>(k.op)); it != byTag.end()) {
      if (it->second->beginNs >= 0) g0 = it->second->beginNs - recStart;
      if (it->second->endNs >= 0) g1 = it->second->endNs - recStart;
    }
    j += (i ? ",[" : "[") + std::to_string(k.op) + "," + std::to_string(k.t0) + "," + std::to_string(k.t1) + "," +
         std::to_string(g0) + "," + std::to_string(g1) + ",[";
    for (size_t q = 0; q < k.ints.size(); ++q) j += (q ? "," : "") + std::to_string(k.ints[q]);
    j += "]," + std::to_string(k.waitNs) + "]";
  }
  j += "],\"events\":[";  // [op, kind, t0, t1]
  for (size_t i = 0; i < recEvents.size(); ++i) {
    const EventRec& e = recEvents[i];
    j += (i ? ",[" : "[") + std::to_string(e.op) + ",\"" + e.what + "\"," + std::to_string(e.t0) + "," +
         std::to_string(e.t1) + "]";
  }
  j += "],\"dispatches\":[";  // [op, function, grid xyz, group xyz, threadgroups?]
  for (size_t i = 0; i < disp.size(); ++i) {
    const auto& d = disp[i];
    j += (i ? ",[" : "[") + std::to_string(d.tag) + ",\"" + d.function + "\"," + std::to_string(d.grid[0]) + "," +
         std::to_string(d.grid[1]) + "," + std::to_string(d.grid[2]) + "," + std::to_string(d.group[0]) + "," +
         std::to_string(d.group[1]) + "," + std::to_string(d.group[2]) + "," + (d.threadgroups ? "1" : "0") + "]";
  }
  j += "],\"memory\":[" + recMem + "]}";
  recCalls.push_back(std::move(j));
  recKernels.clear();
  recEvents.clear();
  recMem.clear();
}

void Executable::setProfile(const ProfileOptions& opts) { impl_->profOpts = opts; }

std::string Executable::takeProfileJson() {
  Impl& im = *impl_;
  const Model& m = *im.model;
  std::string j = "{\"version\":1,\"uses_gpu\":" + std::string(im.gpu ? "true" : "false") + ",\"ops\":[";
  // Static table of the exec region's kernels: [op, name, device, num operands, [arg ranks], [arg dtypes]].
  bool first = true;
  for (size_t idx = 0; idx < im.exec.size(); ++idx) {
    const Instr& in = im.exec[idx];
    if (in.kind != OpKind::Kernel) continue;
    j += (first ? "[" : ",[") + std::to_string(idx) + ",\"" + in.op->name + "\",\"" +
         (in.kernel->device == Device::Gpu ? "gpu" : "cpu") + "\"," + std::to_string(in.numOperands) + ",[";
    first = false;
    for (size_t a = 0; a < in.argRank.size(); ++a) j += (a ? "," : "") + std::to_string(in.argRank[a]);
    j += "],[";
    for (size_t a = 0; a < in.args.size(); ++a)
      j += std::string(a ? ",\"" : "\"") + dtypeName(m.typeOf(in.args[a])->dtype) + "\"";
    j += "]]";
  }
  j += "],\"num_exec_ops\":" + std::to_string(im.exec.size()) + ",\"calls\":[";
  for (size_t i = 0; i < im.recCalls.size(); ++i) j += (i ? "," : "") + im.recCalls[i];
  j += "]}";
  im.recCalls.clear();
  return j;
}

// ---- GPU record/replay setup (gpu_replay.h) ------------------------------------------------------------------------

// Eligible: a GPU model without dim.read (so every size of a call is a function of its input shapes) whose GPU
// kernels form one segment (no CPU kernel or gpu_to_cpu sync between the first and the last one), so one recorded
// command list is the whole GPU work of every call with the same input shapes.
void Executable::Impl::setupReplay(int option) {
  int on = option;
  if (on < 0) {
    const char* e = std::getenv("LOKINFER_GPU_REPLAY");
    on = e && *e && *e != '0';
  }
  if (!on || !gpu) return;
  auto no = [&](const char* why) {
    replayNote = why;
    if (const char* v = std::getenv("LOKINFER_GPU_REPLAY"); v && std::strcmp(v, "verbose") == 0)
      std::fprintf(stderr, "lokinfer: GPU replay not used: %s\n", why);
  };
  size_t first = SIZE_MAX, last = 0;
  for (size_t i = 0; i < exec.size(); ++i) {
    const Instr& in = exec[i];
    if (in.kind == OpKind::DimRead) return no("dim.read");
    if (in.kind == OpKind::Kernel && in.kernel->device == Device::Gpu) first = std::min(first, i), last = i;
  }
  if (first == SIZE_MAX) return no("no GPU kernels");
  for (size_t i = first; i <= last; ++i) {
    const Instr& in = exec[i];
    if (in.kind == OpKind::Kernel && in.kernel->device != Device::Gpu) return no("CPU kernels between GPU kernels");
    if (in.kind == OpKind::Sync && static_cast<SyncKind>(in.op->ints[0]) == SyncKind::GpuToCpu)
      return no("gpu_to_cpu sync between GPU kernels");
  }
  replayEnabled = true;
  replayLastGpu = last;
}

void Executable::Impl::selectReplay(std::span<const TensorView> inputs) {
  std::vector<int64_t> key;
  for (const TensorView& v : inputs) {
    key.push_back(static_cast<int64_t>(v.shape.size()));
    key.insert(key.end(), v.shape.begin(), v.shape.end());
  }
  if (replay && !replays.empty() && replays.back().first == key) return;  // same shapes as the last call
  for (size_t i = 0; i < replays.size(); ++i) {
    if (replays[i].first != key) continue;
    std::rotate(replays.begin() + i, replays.begin() + i + 1, replays.end());
    replay = replays.back().second.get();
    return;
  }
  if (replays.size() >= 4) replays.erase(replays.begin());
  replays.emplace_back(std::move(key), std::make_unique<GpuReplay>(gpu));
  replay = replays.back().second.get();
}

std::string Executable::gpuStrategy() const {
  const Impl& im = *impl_;
  if (!im.gpu) return "none";
  if (!im.replayEnabled) return im.replayNote.empty() ? "direct" : "direct (replay: " + im.replayNote + ")";
  if (!im.replay) return "replay (records on the next call)";
  if (im.replay->ready()) return "replay (" + std::to_string(im.replay->numCommands()) + " commands)";
  if (im.replay->off()) return "direct (replay: " + im.replay->offReason() + ")";
  return "replay (records on the next call)";
}

std::string gpuInspectJson(const std::vector<std::string>& functions, const std::string& archivePath) {
  auto g = GpuContext::shared();
  if (!g) return "{\"error\":\"" + g.status().message() + "\"}";
  return (*g)->inspectJson(functions, archivePath);
}

}  // namespace lk::rt
