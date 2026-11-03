// Symbolic dimension expressions (see docs/impl-notes/design-ir.md §3).
//
// Dims are hash-consed in a DimContext: structurally equal expressions (after
// canonicalization) are the same pointer, so shape equality is pointer
// equality. Canonicalization is local and cheap - constant folding, flattened
// and sorted n-ary Add/Mul, like-term merging, a few div/mod identities - not
// an algebraic solver.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>

#include "lokinfer/ir/status.h"

namespace lk {

enum class DimKind : uint8_t { Const, Sym, Add, Mul, FloorDiv, CeilDiv, Mod, Min, Max };

struct DimNode {
  DimKind kind;
  int64_t value;                         // Const: the value; Sym: symbol id; otherwise 0
  std::vector<const DimNode*> operands;  // Add/Mul: n-ary sorted; others: binary (Min/Max sorted)
  uint32_t id;                           // creation index in its context (stable within a context)

  bool isConst() const { return kind == DimKind::Const; }
  bool isConst(int64_t v) const { return kind == DimKind::Const && value == v; }
};

using Dim = const DimNode*;

class DimContext {
 public:
  DimContext();
  DimContext(const DimContext&) = delete;
  DimContext& operator=(const DimContext&) = delete;

  Dim constant(int64_t v);
  // Returns the symbol with this name, creating it if needed.
  Dim sym(std::string_view name);
  Dim add(Dim a, Dim b);
  Dim sub(Dim a, Dim b);
  Dim mul(Dim a, Dim b);
  Dim neg(Dim a) { return mul(constant(-1), a); }
  Dim floorDiv(Dim a, Dim b);
  Dim ceilDiv(Dim a, Dim b);
  Dim mod(Dim a, Dim b);
  Dim min(Dim a, Dim b);
  Dim max(Dim a, Dim b);

  size_t numSymbols() const { return symbolNames_.size(); }
  size_t numNodes() const { return nodes_.size(); }
  // Capacity hint before creating many nodes/symbols at once (avoids rehashing; a pass creating ~1k offsets).
  void reserve(size_t extraNodes, size_t extraSymbols);
  const std::string& symbolName(int64_t symId) const { return symbolNames_[symId]; }
  // -1 if unknown.
  int64_t findSymbol(std::string_view name) const;

  // Like evaluate(), but returns nullopt instead of asserting when a referenced
  // symbol equals `unbound` or a divisor evaluates to zero.
  std::optional<int64_t> tryEvaluate(Dim d, std::span<const int64_t> symbolValues, int64_t unbound) const;

  // Evaluates with symbol values indexed by symbol id. All referenced symbols
  // must be present (asserted). Division by zero is asserted.
  int64_t evaluate(Dim d, std::span<const int64_t> symbolValues) const;

  std::string toString(Dim d) const;
  // Parses the textual form produced by toString (also accepts redundant parens).
  Expected<Dim> parse(std::string_view text);

  // Total order used for canonical operand sorting; structural, so it doesn't
  // depend on creation order (keeps printing deterministic across contexts).
  int compare(Dim a, Dim b) const;

 private:
  Dim intern(DimKind kind, int64_t value, std::span<const Dim> operands);
  Dim makeAdd(std::vector<Dim> terms);
  Dim makeMul(std::vector<Dim> factors);
  // Splits d into (coefficient, rest) with d == coefficient * rest; rest is
  // nullptr for a pure constant.
  std::pair<int64_t, Dim> splitCoefficient(Dim d);
  // Returns d / c if every term of d is a multiple of c (c > 0), else nullptr.
  Dim exactDivide(Dim d, int64_t c);
  std::pair<Dim, int64_t> splitByDivisor(Dim a, int64_t c);

  // Hash-consing table: a set of nodes looked up by (kind, value, operands) without building a key object, so a
  // lookup that hits allocates nothing.
  struct KeyView {
    DimKind kind;
    int64_t value;
    std::span<const Dim> operands;
  };
  struct NodeHash {
    using is_transparent = void;
    static size_t hash(DimKind kind, int64_t value, std::span<const Dim> ops);
    size_t operator()(const DimNode* n) const { return hash(n->kind, n->value, n->operands); }
    size_t operator()(const KeyView& k) const { return hash(k.kind, k.value, k.operands); }
  };
  struct NodeEq {
    using is_transparent = void;
    static bool eq(const DimNode* n, const KeyView& k) {
      return n->kind == k.kind && n->value == k.value && n->operands.size() == k.operands.size() &&
             std::equal(n->operands.begin(), n->operands.end(), k.operands.begin());
    }
    bool operator()(const DimNode* a, const DimNode* b) const { return a == b; }
    bool operator()(const DimNode* a, const KeyView& b) const { return eq(a, b); }
    bool operator()(const KeyView& a, const DimNode* b) const { return eq(b, a); }
  };

  std::deque<DimNode> nodes_;  // stable addresses, one allocation per chunk (not per node)
  std::unordered_set<const DimNode*, NodeHash, NodeEq> table_;
  std::vector<std::string> symbolNames_;
  std::vector<Dim> symNodes_;  // by symbol id
  struct StrHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>()(s); }
  };
  std::unordered_map<std::string, int64_t, StrHash, std::equal_to<>> symbolIds_;
};

// Per-call memoized evaluation (the runtime's dim evaluator): every distinct node is computed at most once per epoch,
// so dims shared by many kernels (hash-consed: `N`, `4*N`, ...) cost one load after the first use. Valid because a
// symbol's value never changes within an epoch once something used it (symbols are bound before use: inputs at call
// start, dim.read / mem.plan before the ops that use their symbols). Call newEpoch() when symbols are rebound.
class DimEvaluator {
 public:
  explicit DimEvaluator(const DimContext& ctx) : val_(ctx.numNodes(), 0), stamp_(ctx.numNodes(), 0) {}
  void newEpoch() { ++epoch_; }
  int64_t operator()(Dim d, std::span<const int64_t> symbolValues) {
    if (d->kind == DimKind::Const) return d->value;
    if (d->id < stamp_.size() && stamp_[d->id] == epoch_) return val_[d->id];
    int64_t v = compute(d, symbolValues);
    if (d->id < stamp_.size()) {
      stamp_[d->id] = epoch_;
      val_[d->id] = v;
    }
    return v;
  }

 private:
  int64_t compute(Dim d, std::span<const int64_t> sv);
  std::vector<int64_t> val_;
  std::vector<uint32_t> stamp_;
  uint32_t epoch_ = 1;
};

}  // namespace lk
