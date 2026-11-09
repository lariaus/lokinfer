// In-memory IR (see docs/impl-notes/design-ir.md §5-6).
//
// A Model owns its dim/type contexts, a value table and two regions: `init`
// (no inputs, runs once at load) and `exec` (runs per call). The op order of
// a region is its schedule. Values are dense ids into Model::values.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "lokinfer/ir/dim.h"
#include "lokinfer/ir/types.h"

namespace lk {

using ValueId = uint32_t;
constexpr ValueId kNoValue = ~0u;

struct Value {
  Type type;
  std::string name;  // optional printing hint (e.g. "x" prints as %x)
};

// Attribute values. Kernel ints/floats are passed to kernels; the rest are
// structural (e.g. `output` on buffer.alloc).
struct Attr {
  using Ints = std::vector<int64_t>;
  using Floats = std::vector<double>;
  std::variant<bool, int64_t, double, std::string, Ints, Floats, Dim> value;
};

struct NamedAttr {
  std::string name;
  Attr attr;
};

enum class OpKind : uint8_t {
  Kernel,         // name = kernel name; operands; outs (buffer form) or results (tensor form)
  ConstInline,    // data = raw little-endian bytes; one result
  ConstInternal,  // name = constant name (Model::constants); one result
  ConstExternal,  // name = binding name; one result
  View,           // operands[0] = source; dims[0] = element offset; one result
  BufferAlloc,    // one result
  BufferFree,     // operands[0]
  MemPlan,        // one result (arena); dims = sizes (bytes); ints = live intervals (start,end pairs), then the
                  // alignment; symbols = (arena size in bytes, offset_0..offset_n-1). Offsets are in elements of
                  // each buffer's dtype when attr `elem = [bytes per element...]` is present (plan-memory always
                  // writes it), else in bytes. Attr `gpu_only`: the plan's arena pool (see plan-memory).
  DimRead,        // operands[0] = i32/i64 buffer; ints[0] = element index; symbols[0] = defined symbol
  Sync,           // ints[0] = SyncKind. Value form: operands[0] -> results[0] (a zero-cost alias
                  // of the operand, resident on the target device). Written by the Python
                  // graph (ops.to_cpu / ops.to_gpu); devices are never inferred.
  Return,         // operands
};

enum class SyncKind : int64_t { GpuToCpu = 0, CpuToGpu = 1 };

const char* opKindName(OpKind k);

// Attributes read by the compiler/runtime, never passed to kernels:
// nohoist, inplace, inout (kernels), output, arena, persistent, gpu_only (buffer.alloc; gpu_only and elem on mem.plan),
// device (constants and persistent allocs: "cpu" or "gpu", the device the value is resident on).
bool isStructuralAttr(std::string_view name);

// Kernel device from the naming convention <op>..._<dtype>_<device>[_<variant>]:
// a kernel runs on the GPU iff one of its '_'-separated tokens is "gpu".
bool isGpuKernel(std::string_view name);

struct Op {
  OpKind kind;
  std::string name;
  std::vector<ValueId> operands;
  std::vector<ValueId> outs;
  std::vector<ValueId> results;
  std::vector<NamedAttr> attrs;  // sorted by name
  std::vector<uint8_t> data;
  std::vector<Dim> dims;
  std::vector<int64_t> ints;
  std::vector<int64_t> symbols;  // symbol ids defined by this op

  const Attr* attr(std::string_view n) const;
  void setAttr(std::string n, Attr a);
  bool hasFlag(std::string_view n) const;  // bool attr that is true
};

struct Region {
  std::vector<ValueId> args;   // exec: inputs; init: none
  // exec: 1 = a mutable input (`%kv: buffer<...> mutable`): the caller's buffer, written in place by `inout` kernels
  // (e.g. a KV cache); never an output, bound without a copy. Empty = no mutable inputs; else one entry per arg.
  std::vector<uint8_t> mutableArgs;
  std::vector<ValueId> state;  // exec only: values produced by init's return, positionally
  std::vector<Op> ops;         // last op is a Return

  bool isMutableArg(size_t i) const { return i < mutableArgs.size() && mutableArgs[i]; }
};

struct ConstantData {
  std::string name;
  Type type;
  // Internal constants own their bytes (or point into an mmapped file).
  std::shared_ptr<const void> owner;
  const uint8_t* bytes = nullptr;
  size_t size = 0;
};

class Model {
 public:
  Model() : types_(dims_) {}
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  DimContext& dims() { return dims_; }
  const DimContext& dims() const { return dims_; }
  TypeContext& types() { return types_; }
  const TypeContext& types() const { return types_; }

  ValueId newValue(Type t, std::string name = {}) {
    values_.push_back({t, std::move(name)});
    return static_cast<ValueId>(values_.size() - 1);
  }
  Type typeOf(ValueId v) const { return values_[v].type; }
  void setType(ValueId v, Type t) { values_[v].type = t; }
  const std::string& nameOf(ValueId v) const { return values_[v].name; }
  size_t numValues() const { return values_.size(); }

  Region init;
  Region exec;
  // Internal constants by name. Use addConstant() so the index stays valid.
  std::vector<ConstantData> constants;

  void addConstant(ConstantData c);
  const ConstantData* findConstant(std::string_view name) const;

 private:
  DimContext dims_;
  TypeContext types_;
  std::vector<Value> values_;
  struct StrHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const { return std::hash<std::string_view>()(s); }
  };
  // Rebuilt lazily if stale; heterogeneous lookup (no std::string per findConstant).
  mutable std::unordered_map<std::string, size_t, StrHash, std::equal_to<>> constantIndex_;
};

}  // namespace lk
