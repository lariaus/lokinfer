// Dtypes and value types (see docs/impl-notes/design-ir.md §2, §4).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <algorithm>
#include <span>
#include <unordered_set>
#include <vector>

#include "lokinfer/ir/dim.h"

namespace lk {

enum class DType : uint8_t { F32, F16, BF16, I8, U8, I16, I32, I64, Bool };

size_t dtypeSize(DType t);
const char* dtypeName(DType t);
std::optional<DType> parseDType(std::string_view name);
bool isFloat(DType t);

enum class TypeKind : uint8_t { Scalar, Tensor, Buffer };

struct TypeNode {
  TypeKind kind;
  DType dtype;
  std::vector<Dim> shape;  // empty for Scalar (and rank-0 tensors/buffers)
  uint32_t id = 0;         // creation index in its context (dense: per-type caches in passes/verifier)

  bool isScalar() const { return kind == TypeKind::Scalar; }
  bool isTensor() const { return kind == TypeKind::Tensor; }
  bool isBuffer() const { return kind == TypeKind::Buffer; }
  size_t rank() const { return shape.size(); }
};

using Type = const TypeNode*;

// Interns types; owns nothing but types (dims live in the DimContext).
class TypeContext {
 public:
  explicit TypeContext(DimContext& dims) : dims_(dims) {}
  TypeContext(const TypeContext&) = delete;
  TypeContext& operator=(const TypeContext&) = delete;

  Type scalar(DType t) { return get(TypeKind::Scalar, t, {}); }
  Type tensor(DType t, std::vector<Dim> shape) { return get(TypeKind::Tensor, t, std::move(shape)); }
  Type buffer(DType t, std::vector<Dim> shape) { return get(TypeKind::Buffer, t, std::move(shape)); }
  Type get(TypeKind kind, DType t, std::vector<Dim> shape);

  // Same dtype/shape, different kind (bufferization: tensor -> buffer).
  Type withKind(Type t, TypeKind kind) { return get(kind, t->dtype, t->shape); }

  // Number of elements / bytes as dims.
  Dim numElements(Type t);
  Dim numBytes(Type t);

  // "f32", "tensor<f32[N, 4]>", "buffer<i64[]>".
  std::string toString(Type t) const;

  DimContext& dims() { return dims_; }
  size_t numTypes() const { return nodes_.size(); }

 private:
  // Lookup without building a key object (a hit allocates nothing).
  struct KeyView {
    TypeKind kind;
    DType dtype;
    std::span<const Dim> shape;
  };
  struct NodeHash {
    using is_transparent = void;
    static size_t hash(TypeKind kind, DType dtype, std::span<const Dim> shape);
    size_t operator()(const TypeNode* n) const { return hash(n->kind, n->dtype, n->shape); }
    size_t operator()(const KeyView& k) const { return hash(k.kind, k.dtype, k.shape); }
  };
  struct NodeEq {
    using is_transparent = void;
    static bool eq(const TypeNode* n, const KeyView& k) {
      return n->kind == k.kind && n->dtype == k.dtype && n->shape.size() == k.shape.size() &&
             std::equal(n->shape.begin(), n->shape.end(), k.shape.begin());
    }
    bool operator()(const TypeNode* a, const TypeNode* b) const { return a == b; }
    bool operator()(const TypeNode* a, const KeyView& b) const { return eq(a, b); }
    bool operator()(const KeyView& a, const TypeNode* b) const { return eq(b, a); }
  };

  DimContext& dims_;
  std::vector<std::unique_ptr<TypeNode>> nodes_;
  std::unordered_set<const TypeNode*, NodeHash, NodeEq> table_;
};

}  // namespace lk
