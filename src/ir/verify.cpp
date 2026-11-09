#include "lokinfer/ir/verify.h"

#include <string>
#include <algorithm>

namespace lk {

void collectSymbols(Dim d, std::vector<int64_t>& out) {
  if (d->kind == DimKind::Sym) {
    out.push_back(d->value);
    return;
  }
  for (Dim o : d->operands) collectSymbols(o, out);
}

namespace {

class Verifier {
 public:
  // Dense per-value / per-symbol / per-dim / per-type tables (no hashing: the verifier runs after every pass).
  Verifier(const Model& m, const VerifyOptions& o)
      : m_(m),
        opts_(o),
        defined_(m.numValues(), 0),
        bound_(m.dims().numSymbols(), 0),
        dimOk_(m.dims().numNodes(), 0),
        typeOk_(m.types().numTypes(), 0) {
    if (!opts_.requireBuffers) opts_.requireBuffers = inBufferForm(m.init) || inBufferForm(m.exec);
  }

  Status run() {
    if (!m_.init.args.empty() || !m_.init.state.empty()) return fail("init must not have arguments");
    // Exec input types bind their symbols.
    // init runs at load, before any input exists: it may only use symbols it
    // defines itself (dim.read / mem.plan). Exec inputs bind theirs afterwards.
    LK_RETURN_IF_ERROR(region(m_.init, "init"));
    for (ValueId v : m_.exec.args) bindTypeSymbols(m_.typeOf(v));
    const Op& initRet = m_.init.ops.back();
    if (initRet.operands.size() != m_.exec.state.size())
      return fail("exec state count (" + std::to_string(m_.exec.state.size()) + ") != init results (" +
                  std::to_string(initRet.operands.size()) + ")");
    for (size_t i = 0; i < m_.exec.state.size(); ++i)
      if (m_.typeOf(initRet.operands[i]) != m_.typeOf(m_.exec.state[i]))
        return fail("exec state #" + std::to_string(i) + " type differs from init result");
    std::fill(defined_.begin(), defined_.end(), 0);
    for (ValueId v : m_.exec.args) LK_RETURN_IF_ERROR(define(v, "exec argument"));
    for (ValueId v : m_.exec.state) LK_RETURN_IF_ERROR(define(v, "exec state"));
    LK_RETURN_IF_ERROR(region(m_.exec, "exec"));
    return mutability();
  }

 private:
  // Bufferize has run: destination-passing kernels, frees or planned memory exist.
  static bool inBufferForm(const Region& r) {
    for (const Op& op : r.ops)
      if ((op.kind == OpKind::Kernel && !op.outs.empty()) || op.kind == OpKind::BufferFree || op.kind == OpKind::MemPlan)
        return true;
    return false;
  }

  // In-place writes (`inout`) may only target memory that is meant to change: a mutable exec input (the caller's
  // buffer, e.g. a KV cache) or a persistent buffer (model state). Neither may be returned: a mutable input stays the
  // caller's, persistent state stays the model's. Roots are followed through views and syncs.
  Status mutability() {
    const Region& r = m_.exec;
    if (!r.mutableArgs.empty() && r.mutableArgs.size() != r.args.size())
      return fail("exec: mutable flags (" + std::to_string(r.mutableArgs.size()) + ") != args (" +
                  std::to_string(r.args.size()) + ")");
    std::vector<ValueId> root(m_.numValues(), kNoValue);
    auto rootOf = [&](ValueId v) { return root[v] == kNoValue ? v : root[v]; };
    std::vector<uint8_t> writable(m_.numValues(), 0);  // 1 = mutable arg, 2 = persistent alloc / state
    // hoist-init moves persistent allocs into init: exec state is writable iff init returns a persistent alloc (or
    // a view of one) there; the other state (hoisted weights) stays immutable.
    {
      std::vector<ValueId> iroot(m_.numValues(), kNoValue);
      auto irootOf = [&](ValueId v) { return iroot[v] == kNoValue ? v : iroot[v]; };
      std::vector<uint8_t> ipersistent(m_.numValues(), 0);
      for (const Op& op : m_.init.ops) {
        if ((op.kind == OpKind::View || op.kind == OpKind::Sync) && !op.results.empty())
          iroot[op.results[0]] = irootOf(op.operands[0]);
        if (op.kind == OpKind::BufferAlloc && op.hasFlag("persistent")) ipersistent[op.results[0]] = 1;
      }
      const Op& initRet = m_.init.ops.back();
      for (size_t i = 0; i < r.state.size() && i < initRet.operands.size(); ++i)
        if (ipersistent[irootOf(initRet.operands[i])]) writable[r.state[i]] = 2;
    }
    for (size_t i = 0; i < r.args.size(); ++i) {
      if (!r.isMutableArg(i)) continue;
      if (m_.typeOf(r.args[i])->kind != TypeKind::Buffer)
        return fail("exec argument #" + std::to_string(i) + " is mutable but not a buffer");
      writable[r.args[i]] = 1;
    }
    for (const Op& op : r.ops) {
      if ((op.kind == OpKind::View || op.kind == OpKind::Sync) && !op.results.empty())
        root[op.results[0]] = rootOf(op.operands[0]);
      if (op.kind == OpKind::BufferAlloc && op.hasFlag("persistent")) writable[op.results[0]] = 2;
      if (op.kind == OpKind::Kernel)
        if (const Attr* io = op.attr("inout"))
          for (int64_t j : std::get<Attr::Ints>(io->value)) {
            if (j < 0 || static_cast<size_t>(j) >= op.operands.size()) return failOp(op, "inout index out of range");
            if (!writable[rootOf(op.operands[j])])
              return failOp(op, "writes operand #" + std::to_string(j) +
                                " in place, but it is neither a mutable input nor persistent state (an immutable "
                                "input or a temporary): declare the input mutable");
          }
      if (op.kind == OpKind::Return)
        for (ValueId v : op.operands)
          if (writable[rootOf(v)])
            return failOp(op, std::string("returns ") + (writable[rootOf(v)] == 1 ? "a mutable input" : "persistent state") +
                                  " (or a view of it): it can't also be a caller-owned output; return a copy");
    }
    return Status::ok();
  }

  Status fail(std::string msg) { return Status::error("verify: " + msg); }
  Status failOp(const Op& op, std::string msg) { return fail(std::string(opKindName(op.kind)) + " " + op.name + ": " + msg); }

  void bindTypeSymbols(Type t) {
    std::vector<int64_t> syms;
    for (Dim d : t->shape) collectSymbols(d, syms);
    for (int64_t s : syms) bind(s);
  }
  void bind(int64_t s) {
    if (s >= 0 && static_cast<size_t>(s) < bound_.size()) bound_[s] = 1;
  }
  bool isBound(int64_t s) const { return s >= 0 && static_cast<size_t>(s) < bound_.size() && bound_[s]; }

  // All symbols of d bound? Cached per node: the bound set only grows during a walk, so "ok" stays ok.
  bool dimBound(Dim d) {
    if (d->id < dimOk_.size() && dimOk_[d->id]) return true;
    bool ok = d->kind == DimKind::Sym ? isBound(d->value) : true;
    for (Dim o : d->operands) ok = ok && dimBound(o);
    if (ok && d->id < dimOk_.size()) dimOk_[d->id] = 1;
    return ok;
  }
  // The first unbound symbol of d (for error messages), or -1.
  int64_t firstUnbound(Dim d) const {
    std::vector<int64_t> syms;
    collectSymbols(d, syms);
    for (int64_t s : syms)
      if (!isBound(s)) return s;
    return -1;
  }

  Status checkType(ValueId v, const char* what) {
    Type t = m_.typeOf(v);
    if (opts_.requireBuffers && t->isTensor()) return fail(std::string(what) + " has tensor type after bufferization");
    if (t->id < typeOk_.size() && typeOk_[t->id]) return Status::ok();
    for (Dim d : t->shape)
      if (!dimBound(d)) return fail(std::string(what) + " uses unbound symbol " + m_.dims().symbolName(firstUnbound(d)));
    if (t->id < typeOk_.size()) typeOk_[t->id] = 1;
    return Status::ok();
  }

  Status define(ValueId v, const char* what) {
    if (v >= m_.numValues()) return fail(std::string(what) + ": invalid value id");
    if (defined_[v]) return fail(std::string(what) + ": value defined twice");
    defined_[v] = 1;
    return checkType(v, what);
  }

  Status useAll(const Op& op, const std::vector<ValueId>& vs) {
    for (ValueId v : vs)
      if (v >= m_.numValues() || !defined_[v]) return failOp(op, "use of a value not defined before it in this region");
    return Status::ok();
  }

  Status region(const Region& r, const char* name) {
    if (r.ops.empty() || r.ops.back().kind != OpKind::Return)
      return fail(std::string(name) + " must end with return");
    for (size_t i = 0; i < r.ops.size(); ++i) {
      const Op& op = r.ops[i];
      if (op.kind == OpKind::Return && i + 1 != r.ops.size()) return fail(std::string(name) + ": return not last");
      LK_RETURN_IF_ERROR(useAll(op, op.operands));
      LK_RETURN_IF_ERROR(useAll(op, op.outs));
      LK_RETURN_IF_ERROR(opShape(op));
      // Symbols defined by this op become usable by its results and later ops.
      for (int64_t s : op.symbols) {
        if (s < 0 || static_cast<size_t>(s) >= bound_.size()) return failOp(op, "invalid symbol id");
        bind(s);
      }
      for (Dim d : op.dims)
        if (!dimBound(d)) return failOp(op, "uses unbound symbol " + m_.dims().symbolName(firstUnbound(d)));
      for (ValueId v : op.results) LK_RETURN_IF_ERROR(define(v, opKindName(op.kind)));
    }
    return Status::ok();
  }

  // Byte size of a type when every dim is constant, else -1.
  static int64_t staticBytes(Type t) {
    int64_t n = static_cast<int64_t>(dtypeSize(t->dtype));
    for (Dim d : t->shape) {
      if (!d->isConst() || d->value < 0) return -1;
      n *= d->value;
    }
    return n;
  }

  Status counts(const Op& op, size_t operands, size_t results) {
    if (op.operands.size() != operands || op.results.size() != results)
      return failOp(op, "expected " + std::to_string(operands) + " operands and " + std::to_string(results) + " results");
    return Status::ok();
  }

  Status opShape(const Op& op) {
    auto type = [&](ValueId v) { return m_.typeOf(v); };
    switch (op.kind) {
      case OpKind::Kernel:
        if (op.name.empty()) return failOp(op, "missing kernel name");
        for (ValueId v : op.outs)
          if (!type(v)->isBuffer()) return failOp(op, "outs must be buffers");
        return Status::ok();
      case OpKind::ConstInline: {
        LK_RETURN_IF_ERROR(counts(op, 0, 1));
        Type t = type(op.results[0]);
        int64_t n = 1;
        for (Dim d : t->shape) {
          if (!d->isConst()) return failOp(op, "inline constant must have a static shape");
          n *= d->value;
        }
        if (op.data.size() != static_cast<size_t>(n) * dtypeSize(t->dtype))
          return failOp(op, "data has " + std::to_string(op.data.size()) + " bytes, type needs " +
                                std::to_string(n * dtypeSize(t->dtype)));
        return Status::ok();
      }
      case OpKind::ConstInternal:
        LK_RETURN_IF_ERROR(counts(op, 0, 1));
        if (!m_.findConstant(op.name)) return failOp(op, "unknown internal constant");
        return Status::ok();
      case OpKind::ConstExternal:
        return counts(op, 0, 1);
      case OpKind::View: {
        LK_RETURN_IF_ERROR(counts(op, 1, 1));
        if (op.dims.size() != 1) return failOp(op, "missing offset");
        Type src = type(op.operands[0]), dst = type(op.results[0]);
        if (src->kind != dst->kind || src->isScalar()) return failOp(op, "view must map tensor->tensor or buffer->buffer");
        // Static bounds: offset + result bytes must fit in the source.
        int64_t srcBytes = staticBytes(src), dstBytes = staticBytes(dst);
        if (srcBytes >= 0 && dstBytes >= 0 && op.dims[0]->isConst()) {
          int64_t off = op.dims[0]->value * static_cast<int64_t>(dtypeSize(dst->dtype));
          if (op.dims[0]->value < 0 || off + dstBytes > srcBytes) return failOp(op, "view out of bounds of its source");
        }
        return Status::ok();
      }
      case OpKind::BufferAlloc:
        LK_RETURN_IF_ERROR(counts(op, 0, 1));
        if (!type(op.results[0])->isBuffer()) return failOp(op, "result must be a buffer");
        return Status::ok();
      case OpKind::BufferFree:
        LK_RETURN_IF_ERROR(counts(op, 1, 0));
        if (!type(op.operands[0])->isBuffer()) return failOp(op, "operand must be a buffer");
        return Status::ok();
      case OpKind::MemPlan:
        LK_RETURN_IF_ERROR(counts(op, 0, 1));
        if (op.ints.size() != 2 * op.dims.size() + 1) return failOp(op, "live list must have 2 entries per size");
        if (op.symbols.size() != op.dims.size() + 1) return failOp(op, "must define arena size + one offset per size");
        if (!type(op.results[0])->isBuffer()) return failOp(op, "result must be a buffer");
        if (const Attr* e = op.attr("elem")) {
          auto* v = std::get_if<Attr::Ints>(&e->value);
          if (!v || v->size() != op.dims.size()) return failOp(op, "elem must list one element size per buffer");
          for (int64_t b : *v)
            if (b <= 0 || op.ints.back() % b != 0) return failOp(op, "elem sizes must be positive divisors of align");
        }
        return Status::ok();
      case OpKind::DimRead: {
        LK_RETURN_IF_ERROR(counts(op, 1, 0));
        Type t = type(op.operands[0]);
        if (t->isScalar() || (t->dtype != DType::I32 && t->dtype != DType::I64))
          return failOp(op, "operand must be an i32/i64 tensor or buffer");
        if (op.symbols.size() != 1 || op.ints.size() != 1) return failOp(op, "malformed");
        if (op.ints[0] < 0) return failOp(op, "negative index");
        if (int64_t n = staticBytes(t); n >= 0 && op.ints[0] * static_cast<int64_t>(dtypeSize(t->dtype)) >= n)
          return failOp(op, "index out of bounds");
        return Status::ok();
      }
      case OpKind::Sync:
        if (op.ints.size() != 1 || (op.ints[0] != 0 && op.ints[0] != 1)) return failOp(op, "malformed");
        if (op.operands.empty()) return counts(op, 0, 0);
        LK_RETURN_IF_ERROR(counts(op, 1, 1));
        if (type(op.operands[0]) != type(op.results[0])) return failOp(op, "result type must equal operand type");
        return Status::ok();
      case OpKind::Return:
        if (!op.results.empty()) return failOp(op, "return has no results");
        return Status::ok();
    }
    return failOp(op, "unknown op kind");
  }

  const Model& m_;
  VerifyOptions opts_;
  std::vector<uint8_t> defined_, bound_, dimOk_, typeOk_;
};

}  // namespace

Status verifyModel(const Model& m, const VerifyOptions& opts) {
  // Device residency is a second walk (verifyDevices): merged into this walk it measured slower.
  LK_RETURN_IF_ERROR(Verifier(m, opts).run());
  return verifyDevices(m);
}

}  // namespace lk
