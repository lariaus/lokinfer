#include "lokinfer/ir/types.h"

namespace lk {

size_t dtypeSize(DType t) {
  switch (t) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::I8: return 1;
    case DType::U8: return 1;
    case DType::I16: return 2;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::Bool: return 1;
  }
  return 0;
}

const char* dtypeName(DType t) {
  switch (t) {
    case DType::F32: return "f32";
    case DType::F16: return "f16";
    case DType::BF16: return "bf16";
    case DType::I8: return "i8";
    case DType::U8: return "u8";
    case DType::I16: return "i16";
    case DType::I32: return "i32";
    case DType::I64: return "i64";
    case DType::Bool: return "bool";
  }
  return "?";
}

std::optional<DType> parseDType(std::string_view n) {
  static const std::pair<const char*, DType> table[] = {
      {"f32", DType::F32}, {"f16", DType::F16}, {"bf16", DType::BF16}, {"i8", DType::I8},   {"u8", DType::U8},
      {"i16", DType::I16}, {"i32", DType::I32}, {"i64", DType::I64},   {"bool", DType::Bool},
  };
  for (auto& [name, t] : table)
    if (n == name) return t;
  return std::nullopt;
}

bool isFloat(DType t) { return t == DType::F32 || t == DType::F16 || t == DType::BF16; }

size_t TypeContext::NodeHash::hash(TypeKind kind, DType dtype, std::span<const Dim> shape) {
  size_t h = static_cast<size_t>(kind) * 131 + static_cast<size_t>(dtype);
  for (Dim d : shape) h = h * 1000003u ^ std::hash<const void*>()(d);
  return h;
}

Type TypeContext::get(TypeKind kind, DType t, std::vector<Dim> shape) {
  auto it = table_.find(KeyView{kind, t, shape});
  if (it != table_.end()) return *it;
  auto node = std::make_unique<TypeNode>(TypeNode{kind, t, std::move(shape), static_cast<uint32_t>(nodes_.size())});
  Type result = node.get();
  nodes_.push_back(std::move(node));
  table_.insert(result);
  return result;
}

Dim TypeContext::numElements(Type t) {
  Dim n = dims_.constant(1);
  for (Dim d : t->shape) n = dims_.mul(n, d);
  return n;
}

Dim TypeContext::numBytes(Type t) {
  return dims_.mul(numElements(t), dims_.constant(static_cast<int64_t>(dtypeSize(t->dtype))));
}

std::string TypeContext::toString(Type t) const {
  if (t->isScalar()) return dtypeName(t->dtype);
  std::string s = t->isTensor() ? "tensor<" : "buffer<";
  s += dtypeName(t->dtype);
  s += "[";
  for (size_t i = 0; i < t->shape.size(); ++i) {
    if (i) s += ", ";
    s += dims_.toString(t->shape[i]);
  }
  s += "]>";
  return s;
}

}  // namespace lk
