// Device-boundary verification, part of verifyModel (lokinfer/ir/verify.h): device boundaries are explicit in the
// IR (user decision 2026-11-01). The Python graph writes `%y = sync gpu_to_cpu %x` / `sync cpu_to_gpu`
// (ops.to_cpu / ops.to_gpu); this only checks, never inserts syncs.
//
// Residency of a value: constants and persistent buffer.allocs are resident on their `device` attr (set
// from the captured tensor's device / the graph); a kernel's `inout` operands become resident on its device; graph inputs are usable
// by both devices (unified memory; nothing in the call writes them); exec state has its init result's residency. A kernel's
// results/outs are resident on the kernel's device; views inherit their
// source's residency; a value-form sync's result is resident on its target.
// Errors: a kernel reading a value resident on the other device, and a
// dim.read (a CPU read) of a GPU-resident value.
#include <string>
#include <utility>
#include <vector>

#include "lokinfer/ir/verify.h"

namespace lk {

namespace {

enum class Res : uint8_t { Any, Cpu, Gpu };

const char* resName(Res r) { return r == Res::Gpu ? "gpu" : r == Res::Cpu ? "cpu" : "any"; }

// `seed` (may be empty): residency known before the region runs — exec state carries the residency of the init
// results it comes from (matched by position). `retRes` (optional) receives the return operands' residency.
// (Tried: running this in the structural verifier's walk; measured slower than this separate tight loop.)
Status checkRegion(const Model& m, const Region& r, const char* regionName,
                   const std::vector<std::pair<ValueId, Res>>& seed = {}, std::vector<Res>* retRes = nullptr) {
  std::vector<Res> res(m.numValues(), Res::Any);  // dense: runs after every pass (ids checked by verifyModel first)
  for (auto [v, rv] : seed)
    if (v < res.size()) res[v] = rv;
  auto get = [&](ValueId v) { return v < res.size() ? res[v] : Res::Any; };
  auto valueName = [&](ValueId v) {
    const std::string& n = m.nameOf(v);
    return "%" + (n.empty() ? std::to_string(v) : n);
  };
  for (const Op& op : r.ops) {
    switch (op.kind) {
      case OpKind::Kernel: {
        Res dev = isGpuKernel(op.name) ? Res::Gpu : Res::Cpu;
        for (ValueId v : op.operands) {
          Res rv = get(v);
          if (rv != Res::Any && rv != dev)
            return Status::error(std::string("verify: devices (") + regionName + "): kernel '" + op.name + "' runs on " +
                                 resName(dev) + " but operand " + valueName(v) + " is resident on " + resName(rv) +
                                 "; add ops.to_" + resName(dev) + "(...) in the graph");
        }
        for (ValueId v : op.outs) res[v] = dev;
        for (ValueId v : op.results) res[v] = dev;
        if (const Attr* io = op.attr("inout"))  // operands written in place are now resident on the writer's device
          if (auto* idx = std::get_if<Attr::Ints>(&io->value))
            for (int64_t j : *idx)
              if (j >= 0 && static_cast<size_t>(j) < op.operands.size()) res[op.operands[j]] = dev;
        break;
      }
      case OpKind::BufferAlloc:  // persistent buffers (KV caches) declare their device like constants
        if (const Attr* a = op.attr("device")) {
          auto* s = std::get_if<std::string>(&a->value);
          if (!s || (*s != "cpu" && *s != "gpu"))
            return Status::error(std::string("verify: buffer.alloc device must be \"cpu\" or \"gpu\""));
          res[op.results[0]] = *s == "gpu" ? Res::Gpu : Res::Cpu;
        }
        break;
      case OpKind::View:
        res[op.results[0]] = get(op.operands[0]);
        break;
      case OpKind::ConstInline:
      case OpKind::ConstInternal:
      case OpKind::ConstExternal:
        // Constants carry their device explicitly (from the captured tensor); a
        // constant without one (hand-written IR) is usable by both devices.
        if (const Attr* a = op.attr("device")) {
          auto* s = std::get_if<std::string>(&a->value);
          if (!s || (*s != "cpu" && *s != "gpu"))
            return Status::error(std::string("verify: constant device must be \"cpu\" or \"gpu\""));
          res[op.results[0]] = *s == "gpu" ? Res::Gpu : Res::Cpu;
        }
        break;
      case OpKind::Sync:
        if (!op.results.empty())
          res[op.results[0]] = op.ints[0] == static_cast<int64_t>(SyncKind::GpuToCpu) ? Res::Cpu : Res::Gpu;
        break;
      case OpKind::DimRead:
        if (get(op.operands[0]) == Res::Gpu)
          return Status::error(std::string("verify: devices (") + regionName + "): dim.read of " +
                               valueName(op.operands[0]) + " which is resident on gpu; add ops.to_cpu(...)");
        break;
      case OpKind::Return:
        if (retRes)
          for (ValueId v : op.operands) retRes->push_back(get(v));
        break;
      default:
        break;
    }
  }
  return Status::ok();
}

}  // namespace

Status verifyDevices(const Model& m) {
  // hoist-init moves GPU constants (and their transforms) into init; without carrying their residency into exec
  // state, a CPU kernel reading such state without a sync would pass the check after passes / .lkm loads.
  std::vector<Res> initRes;
  LK_RETURN_IF_ERROR(checkRegion(m, m.init, "init", {}, &initRes));
  std::vector<std::pair<ValueId, Res>> seed;
  for (size_t i = 0; i < m.exec.state.size() && i < initRes.size(); ++i)
    if (initRes[i] != Res::Any) seed.emplace_back(m.exec.state[i], initRes[i]);
  return checkRegion(m, m.exec, "exec", seed);
}

}  // namespace lk
