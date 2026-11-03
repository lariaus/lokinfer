#include "lokinfer/ir/ir.h"

#include <algorithm>

namespace lk {

const char* opKindName(OpKind k) {
  switch (k) {
    case OpKind::Kernel: return "kernel";
    case OpKind::ConstInline: return "const.inline";
    case OpKind::ConstInternal: return "const.internal";
    case OpKind::ConstExternal: return "const.external";
    case OpKind::View: return "view";
    case OpKind::BufferAlloc: return "buffer.alloc";
    case OpKind::BufferFree: return "buffer.free";
    case OpKind::MemPlan: return "mem.plan";
    case OpKind::DimRead: return "dim.read";
    case OpKind::Sync: return "sync";
    case OpKind::Return: return "return";
  }
  return "?";
}

bool isStructuralAttr(std::string_view n) {
  return n == "nohoist" || n == "inplace" || n == "output" || n == "arena" || n == "device" || n == "persistent" ||
         n == "inout" || n == "gpu_only" || n == "elem";
}

const Attr* Op::attr(std::string_view n) const {
  for (auto& a : attrs)
    if (a.name == n) return &a.attr;
  return nullptr;
}

void Op::setAttr(std::string n, Attr a) {
  for (auto& x : attrs) {
    if (x.name == n) {
      x.attr = std::move(a);
      return;
    }
  }
  auto pos = std::lower_bound(attrs.begin(), attrs.end(), n, [](const NamedAttr& x, const std::string& k) {
    return x.name < k;
  });
  attrs.insert(pos, NamedAttr{std::move(n), std::move(a)});
}

bool Op::hasFlag(std::string_view n) const {
  const Attr* a = attr(n);
  return a && std::holds_alternative<bool>(a->value) && std::get<bool>(a->value);
}

void Model::addConstant(ConstantData c) {
  constantIndex_[c.name] = constants.size();
  constants.push_back(std::move(c));
}

const ConstantData* Model::findConstant(std::string_view name) const {
  if (constantIndex_.size() != constants.size()) {
    constantIndex_.clear();
    for (size_t i = 0; i < constants.size(); ++i) constantIndex_[constants[i].name] = i;
  }
  auto it = constantIndex_.find(name);
  return it == constantIndex_.end() ? nullptr : &constants[it->second];
}

bool isGpuKernel(std::string_view name) {
  // A "gpu" token: "gpu" bounded by the string ends or '_' (searching for the token is much cheaper than splitting;
  // the verifier and bufferize call this for every kernel after every pass).
  for (size_t pos = name.find("gpu"); pos != std::string_view::npos; pos = name.find("gpu", pos + 1)) {
    bool startOk = pos == 0 || name[pos - 1] == '_';
    bool endOk = pos + 3 == name.size() || name[pos + 3] == '_';
    if (startOk && endOk) return true;
  }
  return false;
}

}  // namespace lk
