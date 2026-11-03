#include "lokinfer/ir/dim.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <map>

namespace lk {

namespace {

int64_t floorDivInt(int64_t a, int64_t b) {
  assert(b != 0 && "dim division by zero");
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

int64_t ceilDivInt(int64_t a, int64_t b) { return -floorDivInt(-a, b); }

int64_t floorModInt(int64_t a, int64_t b) { return a - floorDivInt(a, b) * b; }

bool isAtom(Dim d) {
  switch (d->kind) {
    case DimKind::Const: return d->value >= 0;
    case DimKind::Sym:
    case DimKind::CeilDiv:
    case DimKind::Min:
    case DimKind::Max: return true;
    default: return false;
  }
}

}  // namespace

size_t DimContext::NodeHash::hash(DimKind kind, int64_t value, std::span<const Dim> ops) {
  size_t h = std::hash<int64_t>()(value) * 31 + static_cast<size_t>(kind);
  for (Dim d : ops) h = h * 1000003u ^ std::hash<const void*>()(d);
  return h;
}

DimContext::DimContext() = default;

Dim DimContext::intern(DimKind kind, int64_t value, std::span<const Dim> operands) {
  auto it = table_.find(KeyView{kind, value, operands});
  if (it != table_.end()) return *it;
  DimNode& node = nodes_.emplace_back();
  node.kind = kind;
  node.value = value;
  node.operands.assign(operands.begin(), operands.end());
  node.id = static_cast<uint32_t>(nodes_.size() - 1);
  Dim result = &node;
  table_.insert(result);
  return result;
}

void DimContext::reserve(size_t extraNodes, size_t extraSymbols) {
  table_.reserve(table_.size() + extraNodes);
  symbolNames_.reserve(symbolNames_.size() + extraSymbols);
  symNodes_.reserve(symNodes_.size() + extraSymbols);
  symbolIds_.reserve(symbolIds_.size() + extraSymbols);
}

Dim DimContext::constant(int64_t v) { return intern(DimKind::Const, v, {}); }

Dim DimContext::sym(std::string_view name) {
  // One hash of the name; the node of a known symbol comes from symNodes_ (no table lookup).
  auto [it, inserted] = symbolIds_.try_emplace(std::string(name), static_cast<int64_t>(symbolNames_.size()));
  if (!inserted) return symNodes_[it->second];
  symbolNames_.emplace_back(name);
  Dim d = intern(DimKind::Sym, it->second, {});
  symNodes_.push_back(d);
  return d;
}

int64_t DimContext::findSymbol(std::string_view name) const {
  auto it = symbolIds_.find(name);
  return it == symbolIds_.end() ? -1 : it->second;
}

int DimContext::compare(Dim a, Dim b) const {
  if (a == b) return 0;
  if (a->kind != b->kind) return a->kind < b->kind ? -1 : 1;
  if (a->kind == DimKind::Sym) {
    int c = symbolNames_[a->value].compare(symbolNames_[b->value]);
    return c < 0 ? -1 : (c > 0 ? 1 : 0);
  }
  if (a->value != b->value) return a->value < b->value ? -1 : 1;
  size_t n = std::min(a->operands.size(), b->operands.size());
  for (size_t i = 0; i < n; ++i) {
    int c = compare(a->operands[i], b->operands[i]);
    if (c != 0) return c;
  }
  if (a->operands.size() != b->operands.size()) return a->operands.size() < b->operands.size() ? -1 : 1;
  return 0;
}

std::pair<int64_t, Dim> DimContext::splitCoefficient(Dim d) {
  if (d->kind == DimKind::Const) return {d->value, nullptr};
  if (d->kind == DimKind::Mul && d->operands[0]->kind == DimKind::Const) {
    std::vector<Dim> rest(d->operands.begin() + 1, d->operands.end());
    Dim r = rest.size() == 1 ? rest[0] : intern(DimKind::Mul, 0, rest);
    return {d->operands[0]->value, r};
  }
  return {1, d};
}

Dim DimContext::makeAdd(std::vector<Dim> terms) {
  int64_t constSum = 0;
  // Ordered by first appearance; merged by identical "rest".
  std::vector<std::pair<Dim, int64_t>> merged;
  std::vector<Dim> stack(terms.rbegin(), terms.rend());
  while (!stack.empty()) {
    Dim t = stack.back();
    stack.pop_back();
    if (t->kind == DimKind::Add) {
      for (auto it = t->operands.rbegin(); it != t->operands.rend(); ++it) stack.push_back(*it);
      continue;
    }
    auto [coef, rest] = splitCoefficient(t);
    if (!rest) {
      constSum += coef;
      continue;
    }
    auto found = std::find_if(merged.begin(), merged.end(), [&](auto& p) { return p.first == rest; });
    if (found != merged.end()) found->second += coef;
    else merged.emplace_back(rest, coef);
  }
  std::vector<Dim> out;
  for (auto& [rest, coef] : merged) {
    if (coef == 0) continue;
    out.push_back(coef == 1 ? rest : makeMul({constant(coef), rest}));
  }
  if (constSum != 0 || out.empty()) out.push_back(constant(constSum));
  if (out.size() == 1) return out[0];
  std::sort(out.begin(), out.end(), [&](Dim a, Dim b) { return compare(a, b) < 0; });
  return intern(DimKind::Add, 0, out);
}

Dim DimContext::makeMul(std::vector<Dim> factors) {
  int64_t coef = 1;
  std::vector<Dim> rest;
  std::vector<Dim> stack(factors.rbegin(), factors.rend());
  while (!stack.empty()) {
    Dim f = stack.back();
    stack.pop_back();
    if (f->kind == DimKind::Mul) {
      for (auto it = f->operands.rbegin(); it != f->operands.rend(); ++it) stack.push_back(*it);
    } else if (f->kind == DimKind::Const) {
      coef *= f->value;
    } else {
      rest.push_back(f);
    }
  }
  if (coef == 0 || rest.empty()) return constant(coef);
  // Distribute a constant coefficient over a single Add factor: 2*(N+1) -> 2*N + 2.
  if (coef != 1 && rest.size() == 1 && rest[0]->kind == DimKind::Add) {
    std::vector<Dim> terms;
    for (Dim t : rest[0]->operands) terms.push_back(makeMul({constant(coef), t}));
    return makeAdd(std::move(terms));
  }
  std::sort(rest.begin(), rest.end(), [&](Dim a, Dim b) { return compare(a, b) < 0; });
  if (coef == 1 && rest.size() == 1) return rest[0];
  if (coef != 1) rest.insert(rest.begin(), constant(coef));
  return intern(DimKind::Mul, 0, rest);
}

Dim DimContext::exactDivide(Dim d, int64_t c) {
  assert(c > 0);
  if (c == 1) return d;
  switch (d->kind) {
    case DimKind::Const:
      return d->value % c == 0 ? constant(d->value / c) : nullptr;
    case DimKind::Add: {
      std::vector<Dim> terms;
      for (Dim t : d->operands) {
        Dim q = exactDivide(t, c);
        if (!q) return nullptr;
        terms.push_back(q);
      }
      return makeAdd(std::move(terms));
    }
    case DimKind::Mul: {
      auto [coef, rest] = splitCoefficient(d);
      if (coef % c != 0) return nullptr;
      return makeMul({constant(coef / c), rest});
    }
    default:
      return nullptr;
  }
}

// a = q*c + b with q a Dim (exact division of every non-constant term) and b
// a constant: returns {q, b}, or {nullptr, 0} when a non-constant term isn't
// a multiple of c. Lets (2*F - 1) // 2 + 1 simplify to F.
std::pair<Dim, int64_t> DimContext::splitByDivisor(Dim a, int64_t c) {
  std::vector<Dim> terms;
  int64_t b = 0;
  auto take = [&](Dim t) {
    if (t->isConst()) {
      b += t->value;
      return true;
    }
    Dim q = exactDivide(t, c);
    if (!q) return false;
    terms.push_back(q);
    return true;
  };
  if (a->kind == DimKind::Add) {
    for (Dim t : a->operands)
      if (!take(t)) return {nullptr, 0};
  } else if (!take(a)) {
    return {nullptr, 0};
  }
  return {terms.empty() ? constant(0) : makeAdd(std::move(terms)), b};
}

Dim DimContext::add(Dim a, Dim b) { return makeAdd({a, b}); }
Dim DimContext::sub(Dim a, Dim b) { return makeAdd({a, neg(b)}); }
Dim DimContext::mul(Dim a, Dim b) { return makeMul({a, b}); }

Dim DimContext::floorDiv(Dim a, Dim b) {
  if (a->isConst() && b->isConst()) return constant(floorDivInt(a->value, b->value));
  if (b->isConst(1)) return a;
  if (a->isConst(0)) return a;
  if (a == b) return constant(1);
  if (b->isConst() && b->value > 0) {
    auto [q, r] = splitByDivisor(a, b->value);
    if (q) return add(q, constant(floorDivInt(r, b->value)));
  }
  const Dim ops[2] = {a, b};
  return intern(DimKind::FloorDiv, 0, ops);
}

Dim DimContext::ceilDiv(Dim a, Dim b) {
  if (a->isConst() && b->isConst()) return constant(ceilDivInt(a->value, b->value));
  if (b->isConst(1)) return a;
  if (a->isConst(0)) return a;
  if (a == b) return constant(1);
  if (b->isConst() && b->value > 0) {
    auto [q, r] = splitByDivisor(a, b->value);
    if (q) return add(q, constant(ceilDivInt(r, b->value)));
  }
  const Dim ops[2] = {a, b};
  return intern(DimKind::CeilDiv, 0, ops);
}

Dim DimContext::mod(Dim a, Dim b) {
  if (a->isConst() && b->isConst()) return constant(floorModInt(a->value, b->value));
  if (b->isConst(1) || a == b || a->isConst(0)) return constant(0);
  if (b->isConst() && b->value > 0) {
    auto [q, r] = splitByDivisor(a, b->value);
    if (q) return constant(floorModInt(r, b->value));
  }
  const Dim ops[2] = {a, b};
  return intern(DimKind::Mod, 0, ops);
}

Dim DimContext::min(Dim a, Dim b) {
  if (a->isConst() && b->isConst()) return constant(std::min(a->value, b->value));
  if (a == b) return a;
  if (compare(b, a) < 0) std::swap(a, b);
  const Dim ops[2] = {a, b};
  return intern(DimKind::Min, 0, ops);
}

Dim DimContext::max(Dim a, Dim b) {
  if (a->isConst() && b->isConst()) return constant(std::max(a->value, b->value));
  if (a == b) return a;
  if (compare(b, a) < 0) std::swap(a, b);
  const Dim ops[2] = {a, b};
  return intern(DimKind::Max, 0, ops);
}

std::optional<int64_t> DimContext::tryEvaluate(Dim d, std::span<const int64_t> sv, int64_t unbound) const {
  // Cheap pre-check: every referenced symbol must be bound.
  std::vector<const DimNode*> stack{d};
  while (!stack.empty()) {
    const DimNode* n = stack.back();
    stack.pop_back();
    if (n->kind == DimKind::Sym) {
      if (static_cast<size_t>(n->value) >= sv.size() || sv[n->value] == unbound) return std::nullopt;
    }
    for (Dim o : n->operands) stack.push_back(o);
  }
  // Division by zero is an error, not a crash.
  std::vector<const DimNode*> divs{d};
  while (!divs.empty()) {
    const DimNode* n = divs.back();
    divs.pop_back();
    if ((n->kind == DimKind::FloorDiv || n->kind == DimKind::CeilDiv || n->kind == DimKind::Mod) &&
        evaluate(n->operands[1], sv) == 0)
      return std::nullopt;
    for (Dim o : n->operands) divs.push_back(o);
  }
  return evaluate(d, sv);
}

int64_t DimContext::evaluate(Dim d, std::span<const int64_t> sv) const {
  switch (d->kind) {
    case DimKind::Const: return d->value;
    case DimKind::Sym:
      assert(static_cast<size_t>(d->value) < sv.size() && "unbound dim symbol");
      return sv[d->value];
    case DimKind::Add: {
      int64_t r = 0;
      for (Dim o : d->operands) r += evaluate(o, sv);
      return r;
    }
    case DimKind::Mul: {
      int64_t r = 1;
      for (Dim o : d->operands) r *= evaluate(o, sv);
      return r;
    }
    case DimKind::FloorDiv: return floorDivInt(evaluate(d->operands[0], sv), evaluate(d->operands[1], sv));
    case DimKind::CeilDiv: return ceilDivInt(evaluate(d->operands[0], sv), evaluate(d->operands[1], sv));
    case DimKind::Mod: return floorModInt(evaluate(d->operands[0], sv), evaluate(d->operands[1], sv));
    case DimKind::Min: return std::min(evaluate(d->operands[0], sv), evaluate(d->operands[1], sv));
    case DimKind::Max: return std::max(evaluate(d->operands[0], sv), evaluate(d->operands[1], sv));
  }
  return 0;
}

int64_t DimEvaluator::compute(Dim d, std::span<const int64_t> sv) {
  // Same semantics as DimContext::evaluate.
  auto& self = *this;
  switch (d->kind) {
    case DimKind::Const: return d->value;
    case DimKind::Sym:
      assert(static_cast<size_t>(d->value) < sv.size() && "unbound dim symbol");
      return sv[d->value];
    case DimKind::Add: {
      int64_t r = 0;
      for (Dim o : d->operands) r += self(o, sv);
      return r;
    }
    case DimKind::Mul: {
      int64_t r = 1;
      for (Dim o : d->operands) r *= self(o, sv);
      return r;
    }
    case DimKind::FloorDiv: return floorDivInt(self(d->operands[0], sv), self(d->operands[1], sv));
    case DimKind::CeilDiv: return ceilDivInt(self(d->operands[0], sv), self(d->operands[1], sv));
    case DimKind::Mod: return floorModInt(self(d->operands[0], sv), self(d->operands[1], sv));
    case DimKind::Min: return std::min(self(d->operands[0], sv), self(d->operands[1], sv));
    case DimKind::Max: return std::max(self(d->operands[0], sv), self(d->operands[1], sv));
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Printing
//
// Grammar (also what parse() accepts):
//   expr    := term (('+' | '-') term)*
//   term    := unary (('*' | '//' | '%') unary)*
//   unary   := '-' unary | primary
//   primary := INT | IDENT | ('ceildiv'|'min'|'max') '(' expr ',' expr ')' | '(' expr ')'

std::string DimContext::toString(Dim d) const {
  auto wrap = [&](Dim x) {
    std::string s = toString(x);
    return isAtom(x) ? s : "(" + s + ")";
  };
  switch (d->kind) {
    case DimKind::Const: return std::to_string(d->value);
    case DimKind::Sym: return symbolNames_[d->value];
    case DimKind::Add: {
      std::string out;
      const DimNode* constTerm = nullptr;
      for (Dim t : d->operands) {
        if (t->isConst()) {
          constTerm = t;
          continue;
        }
        // Split the coefficient without interning (const method).
        int64_t coef = 1;
        std::string body;
        if (t->kind == DimKind::Mul && t->operands[0]->isConst()) {
          coef = t->operands[0]->value;
          std::string rest;
          for (size_t i = 1; i < t->operands.size(); ++i) {
            if (i > 1) rest += "*";
            rest += wrap(t->operands[i]);
          }
          int64_t mag = coef < 0 ? -coef : coef;
          body = mag == 1 ? rest : std::to_string(mag) + "*" + rest;
        } else {
          body = toString(t);
        }
        if (out.empty()) out = (coef < 0 ? "-" : "") + body;
        else out += (coef < 0 ? " - " : " + ") + body;
      }
      if (constTerm) {
        int64_t v = constTerm->value;
        out += (v < 0 ? " - " : " + ") + std::to_string(v < 0 ? -v : v);
      }
      return out;
    }
    case DimKind::Mul: {
      std::string out;
      size_t first = 0;
      if (d->operands[0]->isConst() && d->operands[0]->value < 0) {
        int64_t mag = -d->operands[0]->value;
        out = mag == 1 ? "-" : "-" + std::to_string(mag) + "*";
        first = 1;
      }
      for (size_t i = first; i < d->operands.size(); ++i) {
        if (i > first) out += "*";
        out += wrap(d->operands[i]);
      }
      return out;
    }
    case DimKind::FloorDiv: return wrap(d->operands[0]) + " // " + wrap(d->operands[1]);
    case DimKind::Mod: return wrap(d->operands[0]) + " % " + wrap(d->operands[1]);
    case DimKind::CeilDiv:
      return "ceildiv(" + toString(d->operands[0]) + ", " + toString(d->operands[1]) + ")";
    case DimKind::Min: return "min(" + toString(d->operands[0]) + ", " + toString(d->operands[1]) + ")";
    case DimKind::Max: return "max(" + toString(d->operands[0]) + ", " + toString(d->operands[1]) + ")";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Parsing

namespace {

class DimParser {
 public:
  DimParser(DimContext& ctx, std::string_view text) : ctx_(ctx), s_(text) {}

  Expected<Dim> parseAll() {
    Dim d = expr();
    if (!error_.empty()) return Status::error(error_);
    skipWs();
    if (pos_ != s_.size()) return Status::error("unexpected '" + std::string(s_.substr(pos_)) + "' in dim");
    return d;
  }

 private:
  void skipWs() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
  }
  bool eat(std::string_view tok) {
    skipWs();
    if (s_.substr(pos_, tok.size()) == tok) {
      pos_ += tok.size();
      return true;
    }
    return false;
  }
  Dim fail(std::string msg) {
    if (error_.empty()) error_ = std::move(msg);
    return ctx_.constant(0);
  }

  static constexpr int kMaxDepth = 64;

  Dim expr() {
    if (++depth_ > kMaxDepth) return fail("dim expression nested too deeply");
    Dim lhs = term();
    while (error_.empty()) {
      if (eat("+")) lhs = ctx_.add(lhs, term());
      else if (eat("-")) lhs = ctx_.sub(lhs, term());
      else break;
    }
    --depth_;
    return lhs;
  }
  Dim term() {
    Dim lhs = unary();
    while (error_.empty()) {
      if (eat("//")) lhs = divLike(lhs, unary(), 0);
      else if (eat("*")) lhs = ctx_.mul(lhs, unary());
      else if (eat("%")) lhs = divLike(lhs, unary(), 2);
      else break;
    }
    return lhs;
  }
  Dim divLike(Dim a, Dim b, int which) {
    if (!error_.empty()) return a;
    if (b->isConst(0)) return fail("division by zero in dim expression");
    if (which == 0) return ctx_.floorDiv(a, b);
    if (which == 1) return ctx_.ceilDiv(a, b);
    return ctx_.mod(a, b);
  }
  Dim unary() {
    if (++depth_ > kMaxDepth) return fail("dim expression nested too deeply");
    Dim r = eat("-") ? ctx_.neg(unary()) : primary();
    --depth_;
    return r;
  }
  Dim primary() {
    skipWs();
    if (pos_ >= s_.size()) return fail("unexpected end of dim expression");
    char c = s_[pos_];
    if (c == '(') {
      ++pos_;
      Dim d = expr();
      if (!eat(")")) return fail("expected ')' in dim expression");
      return d;
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
      int64_t v = 0;
      while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
        int d = s_[pos_++] - '0';
        if (v > (INT64_MAX - d) / 10) return fail("integer too large in dim expression");
        v = v * 10 + d;
      }
      return ctx_.constant(v);
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t start = pos_;
      while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) ++pos_;
      std::string_view name = s_.substr(start, pos_ - start);
      size_t save = pos_;
      if ((name == "ceildiv" || name == "min" || name == "max") && eat("(")) {
        Dim a = expr();
        if (!eat(",")) return fail("expected ',' in " + std::string(name) + "(...)");
        Dim b = expr();
        if (!eat(")")) return fail("expected ')' in " + std::string(name) + "(...)");
        if (name == "ceildiv") return divLike(a, b, 1);
        if (name == "min") return ctx_.min(a, b);
        return ctx_.max(a, b);
      }
      pos_ = save;
      return ctx_.sym(name);
    }
    return fail(std::string("unexpected character '") + c + "' in dim expression");
  }

  DimContext& ctx_;
  std::string_view s_;
  size_t pos_ = 0;
  int depth_ = 0;
  std::string error_;
};

}  // namespace

Expected<Dim> DimContext::parse(std::string_view text) { return DimParser(*this, text).parseAll(); }

}  // namespace lk
