// Textual IR printer and parser.
//
//   model {
//     init -> buffer<f32[4]> {
//       %w = const.internal @"w" : buffer<f32[4]>
//       return %w
//     }
//     exec(%x: buffer<f32[N]>, %s: f32) state(%w2: buffer<f32[4]>) -> buffer<f32[N]> {
//       %o = buffer.alloc {output} : buffer<f32[N]>
//       kernel "add_f32_cpu"(%x, %x) outs(%o) {alpha = 1.0}
//       %t = kernel "relu_f32_cpu"(%y) : tensor<f32[N]>
//       %c = const.inline dense<[1, 2]> : tensor<i64[2]>
//       %v = view %o[2*N] : buffer<f32[N]>
//       %a = mem.plan sizes[4*N, 64] live[0, 3, 1, 5] align 256 -> P, O0, O1 : buffer<u8[P]>
//       dim.read %len[0] -> F
//       sync gpu_to_cpu
//       buffer.free %o
//       return %o
//     }
//   }
//
// A region's result types (its return operands' types) follow `->`, MLIR style: one type bare, several or zero
// in parentheses; the header omits them when the region returns nothing. The parser checks them against the
// return.
#include "lokinfer/ir/text.h"

#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "lokinfer/ir/float16.h"

namespace lk {

namespace {

std::string formatDouble(double v, int precision) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*g", precision, v);
  std::string s = buf;
  if (s.find_first_of(".e") == std::string::npos) s += ".0";
  return s;
}

std::string quote(std::string_view s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

}  // namespace

std::string formatScalar(DType t, const uint8_t* p) {
  switch (t) {
    case DType::F32: {
      float f;
      std::memcpy(&f, p, 4);
      return formatDouble(f, 9);
    }
    case DType::F16: {
      uint16_t h;
      std::memcpy(&h, p, 2);
      return formatDouble(halfToFloat(h), 9);
    }
    case DType::BF16: {
      uint16_t h;
      std::memcpy(&h, p, 2);
      return formatDouble(bf16ToFloat(h), 9);
    }
    case DType::I8: return std::to_string(static_cast<int8_t>(*p));
    case DType::U8: return std::to_string(*p);
    case DType::Bool: return *p ? "1" : "0";
    case DType::I16: {
      int16_t v;
      std::memcpy(&v, p, 2);
      return std::to_string(v);
    }
    case DType::I32: {
      int32_t v;
      std::memcpy(&v, p, 4);
      return std::to_string(v);
    }
    case DType::I64: {
      int64_t v;
      std::memcpy(&v, p, 8);
      return std::to_string(v);
    }
  }
  return "?";
}

// ===========================================================================
// Printer

namespace {

class Printer {
 public:
  explicit Printer(const Model& m) : m_(m) {}

  std::string run() {
    out_ = "model {\n";
    out_ += "  init" + resultTypes(m_.init) + " {\n";
    printOps(m_.init);
    out_ += "  }\n";
    out_ += "  exec(";
    printArgList(m_.exec.args);
    out_ += ")";
    if (!m_.exec.state.empty()) {
      out_ += " state(";
      printArgList(m_.exec.state);
      out_ += ")";
    }
    out_ += resultTypes(m_.exec) + " {\n";
    printOps(m_.exec);
    out_ += "  }\n}\n";
    return out_;
  }

 private:
  const std::string& name(ValueId v) {
    auto it = names_.find(v);
    if (it != names_.end()) return it->second;
    std::string base = m_.nameOf(v);
    std::string n;
    if (base.empty()) {
      do n = std::to_string(next_++);
      while (used_.count(n));
    } else {
      n = base;
      for (int i = 1; used_.count(n); ++i) n = base + "_" + std::to_string(i);
    }
    used_.insert(n);
    return names_.emplace(v, "%" + n).first->second;
  }

  std::string type(ValueId v) { return m_.types().toString(m_.typeOf(v)); }

  // " -> t" / " -> (t1, t2)" from the region's return; "" when it returns nothing.
  std::string resultTypes(const Region& r) {
    if (r.ops.empty() || r.ops.back().kind != OpKind::Return || r.ops.back().operands.empty()) return "";
    const std::vector<ValueId>& vs = r.ops.back().operands;
    std::string s = vs.size() == 1 ? " -> " : " -> (";
    for (size_t i = 0; i < vs.size(); ++i) s += (i ? ", " : "") + type(vs[i]);
    return vs.size() == 1 ? s : s + ")";
  }

  void printArgList(const std::vector<ValueId>& vs) {
    for (size_t i = 0; i < vs.size(); ++i) {
      if (i) out_ += ", ";
      out_ += name(vs[i]) + ": " + type(vs[i]);
    }
  }

  std::string valueList(const std::vector<ValueId>& vs) {
    std::string s;
    for (size_t i = 0; i < vs.size(); ++i) {
      if (i) s += ", ";
      s += name(vs[i]);
    }
    return s;
  }

  std::string attrValue(const Attr& a) {
    const DimContext& dims = m_.dims();
    return std::visit(
        [&](auto&& v) -> std::string {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, bool>) return v ? "true" : "false";
          else if constexpr (std::is_same_v<T, int64_t>) return std::to_string(v);
          else if constexpr (std::is_same_v<T, double>) return formatDouble(v, 17);
          else if constexpr (std::is_same_v<T, std::string>) return quote(v);
          else if constexpr (std::is_same_v<T, Attr::Ints>) {
            std::string s = "[";
            for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
            return s + "]";
          } else if constexpr (std::is_same_v<T, Attr::Floats>) {
            std::string s = "f[";
            for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + formatDouble(v[i], 17);
            return s + "]";
          } else {
            return "dim<" + dims.toString(v) + ">";
          }
        },
        a.value);
  }

  std::string attrDict(const Op& op) {
    if (op.attrs.empty()) return "";
    std::string s = " {";
    for (size_t i = 0; i < op.attrs.size(); ++i) {
      if (i) s += ", ";
      s += op.attrs[i].name + " = " + attrValue(op.attrs[i].attr);
    }
    return s + "}";
  }

  std::string resultTypes(const Op& op) {
    if (op.results.empty()) return "";
    std::string s = " : ";
    for (size_t i = 0; i < op.results.size(); ++i) s += (i ? ", " : "") + type(op.results[i]);
    return s;
  }

  void printOps(const Region& r) {
    for (const Op& op : r.ops) {
      out_ += "    ";
      if (!op.results.empty()) out_ += valueList(op.results) + " = ";
      printOp(op);
      out_ += "\n";
    }
  }

  void printOp(const Op& op) {
    const DimContext& dims = m_.dims();
    switch (op.kind) {
      case OpKind::Kernel:
        out_ += "kernel " + quote(op.name) + "(" + valueList(op.operands) + ")";
        if (!op.outs.empty()) out_ += " outs(" + valueList(op.outs) + ")";
        out_ += attrDict(op) + resultTypes(op);
        break;
      case OpKind::ConstInline: {
        Type t = m_.typeOf(op.results[0]);
        size_t es = dtypeSize(t->dtype);
        out_ += "const.inline dense<[";
        for (size_t i = 0; i * es < op.data.size(); ++i) {
          if (i) out_ += ", ";
          out_ += formatScalar(t->dtype, op.data.data() + i * es);
        }
        out_ += "]>" + attrDict(op) + resultTypes(op);
        break;
      }
      case OpKind::ConstInternal:
      case OpKind::ConstExternal:
        out_ += std::string(opKindName(op.kind)) + " @" + quote(op.name) + attrDict(op) + resultTypes(op);
        break;
      case OpKind::View:
        out_ += "view " + name(op.operands[0]) + "[" + dims.toString(op.dims[0]) + "]" + attrDict(op) +
                resultTypes(op);
        break;
      case OpKind::BufferAlloc:
        out_ += "buffer.alloc" + attrDict(op) + resultTypes(op);
        break;
      case OpKind::BufferFree:
        out_ += "buffer.free " + valueList(op.operands);
        break;
      case OpKind::MemPlan: {
        out_ += "mem.plan sizes[";
        for (size_t i = 0; i < op.dims.size(); ++i) out_ += (i ? ", " : "") + dims.toString(op.dims[i]);
        out_ += "] live[";
        for (size_t i = 0; i + 1 < op.ints.size(); ++i) out_ += (i ? ", " : "") + std::to_string(op.ints[i]);
        out_ += "] align " + std::to_string(op.ints.back()) + " ->";
        for (size_t i = 0; i < op.symbols.size(); ++i)
          out_ += std::string(i ? ", " : " ") + dims.symbolName(op.symbols[i]);
        out_ += attrDict(op) + resultTypes(op);
        break;
      }
      case OpKind::DimRead:
        out_ += "dim.read " + name(op.operands[0]) + "[" + std::to_string(op.ints[0]) + "] -> " +
                dims.symbolName(op.symbols[0]);
        break;
      case OpKind::Sync:
        out_ += op.ints[0] == static_cast<int64_t>(SyncKind::GpuToCpu) ? "sync gpu_to_cpu" : "sync cpu_to_gpu";
        if (!op.operands.empty()) out_ += " " + name(op.operands[0]) + resultTypes(op);
        break;
      case OpKind::Return:
        out_ += "return";
        if (!op.operands.empty()) out_ += " " + valueList(op.operands);
        break;
    }
  }

  const Model& m_;
  std::string out_;
  std::unordered_map<ValueId, std::string> names_;
  std::unordered_set<std::string> used_;
  int next_ = 0;
};

}  // namespace

std::string printModel(const Model& model) { return Printer(model).run(); }

// ===========================================================================
// Parser

namespace {

class Parser {
 public:
  explicit Parser(std::string_view s) : s_(s), m_(std::make_unique<Model>()) {}

  Expected<std::unique_ptr<Model>> run() {
    if (!expectWord("model") || !expect("{")) return err();
    std::optional<std::vector<Type>> initTypes, execTypes;
    if (!expectWord("init") || !parseResultTypes(initTypes) || !expect("{")) return err();
    if (!parseOps(m_->init) || !checkResultTypes(m_->init, initTypes, "init")) return err();
    if (!expectWord("exec") || !expect("(")) return err();
    if (!parseArgList(m_->exec.args)) return err();
    if (peekWord("state")) {
      expectWord("state");
      if (!expect("(") || !parseArgList(m_->exec.state)) return err();
    }
    if (!parseResultTypes(execTypes) || !expect("{") || !parseOps(m_->exec) ||
        !checkResultTypes(m_->exec, execTypes, "exec"))
      return err();
    if (!expect("}")) return err();
    skipWs();
    if (pos_ != s_.size()) return fail("trailing text after model"), err();
    return std::move(m_);
  }

 private:
  // ---- lexing helpers
  void skipWs() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == '\n') {
        ++line_;
        ++pos_;
      } else if (std::isspace(static_cast<unsigned char>(c))) {
        ++pos_;
      } else if (c == '/' && pos_ + 1 < s_.size() && s_[pos_ + 1] == '/') {
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
      } else {
        break;
      }
    }
  }
  bool peek(std::string_view tok) {
    skipWs();
    return s_.substr(pos_, tok.size()) == tok;
  }
  bool accept(std::string_view tok) {
    if (!peek(tok)) return false;
    pos_ += tok.size();
    return true;
  }
  bool expect(std::string_view tok) {
    if (accept(tok)) return true;
    return fail("expected '" + std::string(tok) + "'");
  }
  static bool isIdentChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.'; }
  std::string_view word() {
    skipWs();
    size_t start = pos_;
    while (pos_ < s_.size() && isIdentChar(s_[pos_])) ++pos_;
    return s_.substr(start, pos_ - start);
  }
  bool peekWord(std::string_view w) {
    size_t save = pos_;
    bool ok = word() == w;
    pos_ = save;
    return ok;
  }
  bool expectWord(std::string_view w) {
    size_t save = pos_;
    if (word() == w) return true;
    pos_ = save;
    return fail("expected '" + std::string(w) + "'");
  }
  bool fail(std::string msg) {
    if (error_.empty()) {
      size_t end = std::min(s_.size(), pos_ + 30);
      error_ = "line " + std::to_string(line_) + ": " + msg + " near '" + std::string(s_.substr(pos_, end - pos_)) + "'";
    }
    return false;
  }
  Status err() { return Status::error(error_.empty() ? "parse error" : error_); }

  bool parseString(std::string& out) {
    if (!expect("\"")) return false;
    out.clear();
    while (pos_ < s_.size() && s_[pos_] != '"') {
      if (s_[pos_] == '\\' && pos_ + 1 < s_.size()) ++pos_;
      out += s_[pos_++];
    }
    if (pos_ >= s_.size()) return fail("unterminated string");
    ++pos_;
    return true;
  }

  // Reads raw text up to the matching closing bracket (for dims inside [] / <>).
  bool parseDimUntil(char close, Dim& out) {
    skipWs();
    size_t start = pos_;
    int depth = 0;
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == '(') ++depth;
      else if (c == ')') --depth;
      if (depth == 0 && (c == close || c == ',')) break;
      if (depth < 0) break;
      ++pos_;
    }
    auto r = m_->dims().parse(s_.substr(start, pos_ - start));
    if (!r) return fail(r.status().message());
    out = *r;
    return true;
  }

  bool parseNumber(double& d, bool& isInt, int64_t& i) {
    skipWs();
    const char* begin = s_.data() + pos_;
    if (s_.substr(pos_, 3) == "nan") { pos_ += 3; d = NAN; isInt = false; return true; }
    if (s_.substr(pos_, 3) == "inf") { pos_ += 3; d = INFINITY; isInt = false; return true; }
    if (s_.substr(pos_, 4) == "-inf") { pos_ += 4; d = -INFINITY; isInt = false; return true; }
    size_t p = pos_;
    if (p < s_.size() && (s_[p] == '-' || s_[p] == '+')) ++p;
    bool sawDigit = false, sawFloat = false;
    while (p < s_.size()) {
      char c = s_[p];
      if (std::isdigit(static_cast<unsigned char>(c))) sawDigit = true;
      else if (c == '.' || c == 'e' || c == 'E') sawFloat = true;
      else if ((c == '-' || c == '+') && (s_[p - 1] == 'e' || s_[p - 1] == 'E')) {}
      else break;
      ++p;
    }
    if (!sawDigit) return fail("expected number");
    std::string tok(begin, p - pos_);
    pos_ = p;
    isInt = !sawFloat;
    if (isInt) i = std::strtoll(tok.c_str(), nullptr, 10);
    d = std::strtod(tok.c_str(), nullptr);
    return true;
  }

  // ---- values
  bool parseValueName(std::string& n) {
    if (!expect("%")) return false;
    size_t start = pos_;
    while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) ++pos_;
    if (pos_ == start) return fail("expected value name");
    n = std::string(s_.substr(start, pos_ - start));
    return true;
  }
  static bool isNumeric(const std::string& n) {
    for (char c : n)
      if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
  }
  bool define(const std::string& n, Type t, ValueId& id) {
    if (values_.count(n)) return fail("redefinition of %" + n);
    id = m_->newValue(t, isNumeric(n) ? "" : n);
    values_[n] = id;
    return true;
  }
  bool use(ValueId& id) {
    std::string n;
    if (!parseValueName(n)) return false;
    auto it = values_.find(n);
    if (it == values_.end()) return fail("use of undefined value %" + n);
    id = it->second;
    return true;
  }
  bool parseUseList(std::vector<ValueId>& out, std::string_view close) {
    if (accept(close)) return true;
    do {
      ValueId v;
      if (!use(v)) return false;
      out.push_back(v);
    } while (accept(","));
    return expect(close);
  }

  bool parseType(Type& t) {
    std::string_view w = word();
    if (auto dt = parseDType(w)) {
      t = m_->types().scalar(*dt);
      return true;
    }
    TypeKind kind;
    if (w == "tensor") kind = TypeKind::Tensor;
    else if (w == "buffer") kind = TypeKind::Buffer;
    else return fail("expected type");
    if (!expect("<")) return false;
    auto dt = parseDType(word());
    if (!dt) return fail("expected dtype");
    if (!expect("[")) return false;
    std::vector<Dim> shape;
    if (!accept("]")) {
      do {
        Dim d;
        if (!parseDimUntil(']', d)) return false;
        shape.push_back(d);
      } while (accept(","));
      if (!expect("]")) return false;
    }
    if (!expect(">")) return false;
    t = m_->types().get(kind, *dt, std::move(shape));
    return true;
  }

  // Optional `-> t` / `-> (t1, ...)` region result types; nullopt if absent (then nothing to check: hand-written IR
  // may omit them).
  bool parseResultTypes(std::optional<std::vector<Type>>& types) {
    if (!accept("->")) return true;
    std::vector<Type>& out = types.emplace();
    if (!accept("(")) {
      Type t;
      if (!parseType(t)) return false;
      out.push_back(t);
      return true;
    }
    if (accept(")")) return true;
    do {
      Type t;
      if (!parseType(t)) return false;
      out.push_back(t);
    } while (accept(","));
    return expect(")");
  }

  bool checkResultTypes(const Region& r, const std::optional<std::vector<Type>>& types, const char* name) {
    if (!types) return true;
    const std::vector<Type>& declared = *types;
    const std::vector<ValueId>& ret = r.ops.back().operands;  // parseOps guarantees a trailing return
    if (ret.size() != declared.size())
      return fail(std::string(name) + " header declares " + std::to_string(declared.size()) + " result types, return has " +
                  std::to_string(ret.size()));
    for (size_t i = 0; i < ret.size(); ++i)
      if (m_->typeOf(ret[i]) != declared[i])
        return fail(std::string(name) + " result #" + std::to_string(i) + " is " + m_->types().toString(m_->typeOf(ret[i])) +
                    ", header declares " + m_->types().toString(declared[i]));
    return true;
  }

  bool parseArgList(std::vector<ValueId>& out) {
    if (accept(")")) return true;
    do {
      std::string n;
      Type t;
      if (!parseValueName(n) || !expect(":") || !parseType(t)) return false;
      ValueId id;
      if (!define(n, t, id)) return false;
      out.push_back(id);
    } while (accept(","));
    return expect(")");
  }

  bool parseAttrValue(Attr& a) {
    skipWs();
    if (peek("\"")) {
      std::string s;
      if (!parseString(s)) return false;
      a.value = std::move(s);
      return true;
    }
    if (peekWord("true") || peekWord("false")) {
      a.value = word() == "true";
      return true;
    }
    if (accept("dim<")) {
      Dim d;
      if (!parseDimUntil('>', d) || !expect(">")) return false;
      a.value = d;
      return true;
    }
    bool floatList = accept("f[");
    if (floatList || accept("[")) {
      Attr::Ints ints;
      Attr::Floats floats;
      if (!accept("]")) {
        do {
          double d;
          bool isInt;
          int64_t i = 0;
          if (!parseNumber(d, isInt, i)) return false;
          ints.push_back(i);
          floats.push_back(d);
          if (!isInt) floatList = true;
        } while (accept(","));
        if (!expect("]")) return false;
      }
      if (floatList) a.value = std::move(floats);
      else a.value = std::move(ints);
      return true;
    }
    double d;
    bool isInt;
    int64_t i = 0;
    if (!parseNumber(d, isInt, i)) return false;
    if (isInt) a.value = i;
    else a.value = d;
    return true;
  }

  bool parseAttrDict(Op& op) {
    if (!accept("{")) return true;
    if (accept("}")) return true;
    do {
      std::string name(word());
      if (name.empty()) return fail("expected attribute name");
      Attr a;
      if (!expect("=") || !parseAttrValue(a)) return false;
      op.setAttr(std::move(name), std::move(a));
    } while (accept(","));
    return expect("}");
  }

  bool parseResultTypes(Op& op, const std::vector<std::string>& resultNames) {
    if (resultNames.empty()) return true;
    if (!expect(":")) return false;
    for (size_t i = 0; i < resultNames.size(); ++i) {
      if (i && !expect(",")) return false;
      Type t;
      if (!parseType(t)) return false;
      ValueId id;
      if (!define(resultNames[i], t, id)) return false;
      op.results.push_back(id);
    }
    return true;
  }

  bool encodeScalar(DType t, double d, int64_t i, bool isInt, std::vector<uint8_t>& out) {
    size_t at = out.size();
    out.resize(at + dtypeSize(t));
    uint8_t* p = out.data() + at;
    switch (t) {
      case DType::F32: { float f = static_cast<float>(d); std::memcpy(p, &f, 4); break; }
      case DType::F16: { uint16_t h = floatToHalf(static_cast<float>(d)); std::memcpy(p, &h, 2); break; }
      case DType::BF16: { uint16_t h = floatToBf16(static_cast<float>(d)); std::memcpy(p, &h, 2); break; }
      default: {
        if (!isInt) return fail("expected integer constant");
        if (t == DType::I8 || t == DType::U8 || t == DType::Bool) { *p = static_cast<uint8_t>(i); }
        else if (t == DType::I16) { int16_t v = static_cast<int16_t>(i); std::memcpy(p, &v, 2); }
        else if (t == DType::I32) { int32_t v = static_cast<int32_t>(i); std::memcpy(p, &v, 4); }
        else { std::memcpy(p, &i, 8); }
      }
    }
    return true;
  }

  bool parseOps(Region& r) {
    while (true) {
      if (accept("}")) {
        if (r.ops.empty() || r.ops.back().kind != OpKind::Return) return fail("region must end with return");
        return true;
      }
      std::vector<std::string> resultNames;
      if (peek("%")) {
        do {
          std::string n;
          if (!parseValueName(n)) return false;
          resultNames.push_back(n);
        } while (accept(","));
        if (!expect("=")) return false;
      }
      Op op;
      if (!parseOp(op, resultNames)) return false;
      r.ops.push_back(std::move(op));
    }
  }

  bool parseOp(Op& op, const std::vector<std::string>& resultNames) {
    std::string kw(word());
    if (kw == "kernel") {
      op.kind = OpKind::Kernel;
      if (!parseString(op.name) || !expect("(") || !parseUseList(op.operands, ")")) return false;
      if (peekWord("outs")) {
        word();
        if (!expect("(") || !parseUseList(op.outs, ")")) return false;
      }
      return parseAttrDict(op) && parseResultTypes(op, resultNames);
    }
    if (kw == "const.inline") {
      op.kind = OpKind::ConstInline;
      if (!expectWord("dense") || !expect("<") || !expect("[")) return false;
      std::vector<std::tuple<double, int64_t, bool>> nums;
      if (!accept("]")) {
        do {
          double d;
          bool isInt;
          int64_t i = 0;
          if (!parseNumber(d, isInt, i)) return false;
          nums.emplace_back(d, i, isInt);
        } while (accept(","));
        if (!expect("]")) return false;
      }
      if (!expect(">") || !parseAttrDict(op) || !parseResultTypes(op, resultNames)) return false;
      if (op.results.size() != 1) return fail("const.inline must have one result");
      DType dt = m_->typeOf(op.results[0])->dtype;
      for (auto& [d, i, isInt] : nums)
        if (!encodeScalar(dt, d, i, isInt, op.data)) return false;
      return true;
    }
    if (kw == "const.internal" || kw == "const.external") {
      op.kind = kw == "const.internal" ? OpKind::ConstInternal : OpKind::ConstExternal;
      if (!expect("@") || !parseString(op.name) || !parseAttrDict(op) || !parseResultTypes(op, resultNames))
        return false;
      if (op.kind == OpKind::ConstInternal && !m_->findConstant(op.name))
        m_->addConstant({op.name, m_->typeOf(op.results[0]), nullptr, nullptr, 0});
      return true;
    }
    if (kw == "view") {
      op.kind = OpKind::View;
      ValueId src;
      Dim off;
      if (!use(src) || !expect("[") || !parseDimUntil(']', off) || !expect("]")) return false;
      op.operands.push_back(src);
      op.dims.push_back(off);
      return parseAttrDict(op) && parseResultTypes(op, resultNames);
    }
    if (kw == "buffer.alloc") {
      op.kind = OpKind::BufferAlloc;
      return parseAttrDict(op) && parseResultTypes(op, resultNames);
    }
    if (kw == "buffer.free") {
      op.kind = OpKind::BufferFree;
      ValueId v;
      if (!use(v)) return false;
      op.operands.push_back(v);
      return true;
    }
    if (kw == "mem.plan") {
      op.kind = OpKind::MemPlan;
      if (!expectWord("sizes") || !expect("[")) return false;
      if (!accept("]")) {
        do {
          Dim d;
          if (!parseDimUntil(']', d)) return false;
          op.dims.push_back(d);
        } while (accept(","));
        if (!expect("]")) return false;
      }
      if (!expectWord("live") || !expect("[")) return false;
      if (!accept("]")) {
        do {
          double d;
          bool isInt;
          int64_t i = 0;
          if (!parseNumber(d, isInt, i) || !isInt) return fail("expected integer");
          op.ints.push_back(i);
        } while (accept(","));
        if (!expect("]")) return false;
      }
      double d;
      bool isInt;
      int64_t align = 0;
      if (!expectWord("align") || !parseNumber(d, isInt, align) || !expect("->")) return false;
      op.ints.push_back(align);
      do {
        std::string_view sym = word();
        if (sym.empty()) return fail("expected symbol name");
        op.symbols.push_back(m_->dims().sym(sym)->value);
      } while (accept(","));
      return parseAttrDict(op) && parseResultTypes(op, resultNames);
    }
    if (kw == "dim.read") {
      op.kind = OpKind::DimRead;
      ValueId src;
      double d;
      bool isInt;
      int64_t idx = 0;
      if (!use(src) || !expect("[") || !parseNumber(d, isInt, idx) || !expect("]") || !expect("->")) return false;
      std::string_view sym = word();
      if (sym.empty()) return fail("expected symbol name");
      op.operands.push_back(src);
      op.ints.push_back(idx);
      op.symbols.push_back(m_->dims().sym(sym)->value);
      return true;
    }
    if (kw == "sync") {
      op.kind = OpKind::Sync;
      std::string_view k = word();
      if (k == "gpu_to_cpu") op.ints.push_back(static_cast<int64_t>(SyncKind::GpuToCpu));
      else if (k == "cpu_to_gpu") op.ints.push_back(static_cast<int64_t>(SyncKind::CpuToGpu));
      else return fail("expected gpu_to_cpu or cpu_to_gpu");
      if (!resultNames.empty()) {  // value form: %y = sync <kind> %x : type
        ValueId v;
        if (!use(v)) return false;
        op.operands.push_back(v);
        return parseResultTypes(op, resultNames);
      }
      return true;
    }
    if (kw == "return") {
      op.kind = OpKind::Return;
      if (peek("%")) {
        do {
          ValueId v;
          if (!use(v)) return false;
          op.operands.push_back(v);
        } while (accept(","));
      }
      return true;
    }
    return fail("unknown op '" + kw + "'");
  }

  std::string_view s_;
  size_t pos_ = 0;
  int line_ = 1;
  std::string error_;
  std::unique_ptr<Model> m_;
  std::unordered_map<std::string, ValueId> values_;
};

}  // namespace

Expected<std::unique_ptr<Model>> parseModel(std::string_view text) { return Parser(text).run(); }

}  // namespace lk
