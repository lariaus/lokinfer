#include "lokinfer/ir/binary.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace lk {

namespace {

constexpr char kMagic[4] = {'L', 'K', 'M', '\0'};
constexpr uint32_t kVersion = 2;  // 2: exec mutable-arg flags after the args (version 1 still read: none)
constexpr size_t kHeaderSize = 64;
constexpr size_t kDataAlign = 16384;

struct Header {
  char magic[4];
  uint32_t version;
  uint64_t metaOffset, metaSize;
  uint64_t dataOffset, dataSize;
  uint8_t reserved[24];
};
static_assert(sizeof(Header) == kHeaderSize);

size_t alignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// ---------------------------------------------------------------- writer

class Writer {
 public:
  std::vector<uint8_t> out;
  void u8(uint8_t v) { out.push_back(v); }
  void uv(uint64_t v) {
    while (v >= 0x80) {
      out.push_back(static_cast<uint8_t>(v | 0x80));
      v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
  }
  void sv(int64_t v) { uv((static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63)); }
  void f64(double d) {
    uint64_t b;
    std::memcpy(&b, &d, 8);
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(b >> (8 * i)));
  }
  void str(const std::string& s) {
    uv(s.size());
    out.insert(out.end(), s.begin(), s.end());
  }
  void bytes(const std::vector<uint8_t>& b) {
    uv(b.size());
    out.insert(out.end(), b.begin(), b.end());
  }
};

// Dims/types are referenced by index in first-use order.
class ModelEncoder {
 public:
  explicit ModelEncoder(const Model& m) : m_(m) {}

  Status encode(Writer& w, std::vector<std::pair<const ConstantData*, uint64_t>>& data) {
    // Collect dims and types reachable from values and ops.
    for (ValueId v = 0; v < m_.numValues(); ++v) typeIndex(m_.typeOf(v));
    for (const ConstantData& c : m_.constants) typeIndex(c.type);
    for (const Region* r : {&m_.init, &m_.exec})
      for (const Op& op : r->ops) {
        for (Dim d : op.dims) dimIndex(d);
        for (auto& a : op.attrs)
          if (auto* d = std::get_if<Dim>(&a.attr.value)) dimIndex(*d);
      }
    const DimContext& dims = m_.dims();
    w.uv(dims.numSymbols());
    for (size_t i = 0; i < dims.numSymbols(); ++i) w.str(dims.symbolName(static_cast<int64_t>(i)));
    w.uv(dimList_.size());
    for (Dim d : dimList_) {
      w.u8(static_cast<uint8_t>(d->kind));
      w.sv(d->value);
      w.uv(d->operands.size());
      for (Dim o : d->operands) w.uv(dimIdx_.find(o)->second);
    }
    w.uv(typeList_.size());
    for (Type t : typeList_) {
      w.u8(static_cast<uint8_t>(t->kind));
      w.u8(static_cast<uint8_t>(t->dtype));
      w.uv(t->shape.size());
      for (Dim d : t->shape) w.uv(dimIdx_.find(d)->second);
    }
    w.uv(m_.numValues());
    for (ValueId v = 0; v < m_.numValues(); ++v) {
      w.uv(typeIdx_.find(m_.typeOf(v))->second);
      w.str(m_.nameOf(v));
    }
    // Constants: data offsets relative to the data section.
    uint64_t off = 0;
    w.uv(m_.constants.size());
    for (const ConstantData& c : m_.constants) {
      if (!c.bytes && c.size) return Status::error("constant '" + c.name + "' has no data");
      w.str(c.name);
      w.uv(typeIdx_.find(c.type)->second);
      w.uv(off);
      w.uv(c.size);
      data.emplace_back(&c, off);
      off = alignUp(off + c.size, kDataAlign);
    }
    for (const Region* r : {&m_.init, &m_.exec}) LK_RETURN_IF_ERROR(region(w, *r));
    return Status::ok();
  }

 private:
  size_t dimIndex(Dim d) {
    auto it = dimIdx_.find(d);
    if (it != dimIdx_.end()) return it->second;
    for (Dim o : d->operands) dimIndex(o);  // operands first (topological)
    size_t i = dimList_.size();
    dimList_.push_back(d);
    dimIdx_[d] = i;
    return i;
  }
  size_t typeIndex(Type t) {
    auto it = typeIdx_.find(t);
    if (it != typeIdx_.end()) return it->second;
    for (Dim d : t->shape) dimIndex(d);
    size_t i = typeList_.size();
    typeList_.push_back(t);
    typeIdx_[t] = i;
    return i;
  }
  static void ids(Writer& w, const std::vector<ValueId>& v) {
    w.uv(v.size());
    for (ValueId x : v) w.uv(x);
  }
  Status region(Writer& w, const Region& r) {
    ids(w, r.args);
    w.uv(r.mutableArgs.size());
    for (uint8_t f : r.mutableArgs) w.u8(f);
    ids(w, r.state);
    w.uv(r.ops.size());
    for (const Op& op : r.ops) {
      w.u8(static_cast<uint8_t>(op.kind));
      w.str(op.name);
      ids(w, op.operands);
      ids(w, op.outs);
      ids(w, op.results);
      w.uv(op.attrs.size());
      for (const NamedAttr& a : op.attrs) {
        w.str(a.name);
        const auto& v = a.attr.value;
        w.u8(static_cast<uint8_t>(v.index()));
        if (auto* b = std::get_if<bool>(&v)) w.u8(*b);
        else if (auto* i = std::get_if<int64_t>(&v)) w.sv(*i);
        else if (auto* f = std::get_if<double>(&v)) w.f64(*f);
        else if (auto* s = std::get_if<std::string>(&v)) w.str(*s);
        else if (auto* is = std::get_if<Attr::Ints>(&v)) {
          w.uv(is->size());
          for (int64_t x : *is) w.sv(x);
        } else if (auto* fs = std::get_if<Attr::Floats>(&v)) {
          w.uv(fs->size());
          for (double x : *fs) w.f64(x);
        } else if (auto* d = std::get_if<Dim>(&v)) {
          w.uv(dimIdx_.find(*d)->second);
        }
      }
      w.bytes(op.data);
      w.uv(op.dims.size());
      for (Dim d : op.dims) w.uv(dimIdx_.find(d)->second);
      w.uv(op.ints.size());
      for (int64_t x : op.ints) w.sv(x);
      w.uv(op.symbols.size());
      for (int64_t x : op.symbols) w.uv(static_cast<uint64_t>(x));
    }
    return Status::ok();
  }

  const Model& m_;
  std::vector<Dim> dimList_;
  std::unordered_map<Dim, size_t> dimIdx_;
  std::vector<Type> typeList_;
  std::unordered_map<Type, size_t> typeIdx_;
};

// ---------------------------------------------------------------- reader

class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
  bool ok() const { return ok_; }
  uint8_t u8() {
    if (p_ >= end_) return fail();
    return *p_++;
  }
  uint64_t uv() {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (p_ >= end_) return fail();
      uint8_t b = *p_++;
      v |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) return v;
    }
    return fail();
  }
  int64_t sv() {
    uint64_t u = uv();
    return static_cast<int64_t>((u >> 1) ^ (~(u & 1) + 1));
  }
  double f64() {
    if (end_ - p_ < 8) return fail();
    uint64_t b = 0;
    for (int i = 0; i < 8; ++i) b |= static_cast<uint64_t>(p_[i]) << (8 * i);
    p_ += 8;
    double d;
    std::memcpy(&d, &b, 8);
    return d;
  }
  std::string str() {
    uint64_t n = uv();
    if (static_cast<uint64_t>(end_ - p_) < n) return fail(), std::string();
    std::string s(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return s;
  }
  std::vector<uint8_t> bytes() {
    uint64_t n = uv();
    if (static_cast<uint64_t>(end_ - p_) < n) return fail(), std::vector<uint8_t>();
    std::vector<uint8_t> b(p_, p_ + n);
    p_ += n;
    return b;
  }
  // Bounded count (guards against corrupt files allocating huge vectors).
  uint64_t count(uint64_t max = 1u << 26) {
    uint64_t n = uv();
    if (n > max) return fail();
    return n;
  }

 private:
  uint8_t fail() {
    ok_ = false;
    p_ = end_;
    return 0;
  }
  const uint8_t* p_;
  const uint8_t* end_;
  bool ok_ = true;
};

// A constant's bytes taken from outside the file (readModelSplit): pointer, size, owner (keeps the mapping alive).
struct SplitConstant {
  const uint8_t* bytes = nullptr;
  size_t size = 0;
  std::shared_ptr<const void> owner;
};

// split: the "skeleton" form (header + metadata only; constant i's bytes from split[i]).
Expected<std::unique_ptr<Model>> decode(const uint8_t* base, size_t size, std::shared_ptr<const void> owner,
                                        const std::vector<SplitConstant>* split = nullptr) {
  if (size < kHeaderSize) return Status::error("lkm: file too small");
  Header h;
  std::memcpy(&h, base, sizeof h);
  if (std::memcmp(h.magic, kMagic, 4) != 0) return Status::error("lkm: bad magic");
  if (h.version != kVersion && h.version != 1) return Status::error("lkm: unsupported version " + std::to_string(h.version));
  // Overflow-safe: compare against the remaining space, never add untrusted sizes.
  if (h.metaOffset > size || h.metaSize > size - h.metaOffset ||
      (!split && (h.dataOffset > size || h.dataSize > size - h.dataOffset)))
    return Status::error("lkm: truncated file");
  Reader r(base + h.metaOffset, h.metaSize);
  auto m = std::make_unique<Model>();
  DimContext& dims = m->dims();
  std::vector<int64_t> symIds;
  for (uint64_t i = 0, n = r.count(); i < n && r.ok(); ++i) symIds.push_back(dims.sym(r.str())->value);

  std::vector<Dim> dimList;
  for (uint64_t i = 0, n = r.count(); i < n && r.ok(); ++i) {
    auto kind = static_cast<DimKind>(r.u8());
    int64_t value = r.sv();
    std::vector<Dim> ops;
    for (uint64_t k = 0, no = r.count(); k < no && r.ok(); ++k) {
      uint64_t idx = r.uv();
      if (idx >= dimList.size()) return Status::error("lkm: bad dim reference");
      ops.push_back(dimList[idx]);
    }
    Dim d = nullptr;
    // Rebuild through the canonicalizing API (stored nodes are canonical).
    switch (kind) {
      case DimKind::Const: d = dims.constant(value); break;
      case DimKind::Sym:
        if (value < 0 || static_cast<size_t>(value) >= symIds.size()) return Status::error("lkm: bad symbol");
        d = dims.sym(dims.symbolName(symIds[value]));
        break;
      case DimKind::Add:
      case DimKind::Mul:
        if (ops.empty()) return Status::error("lkm: empty n-ary dim");
        d = ops[0];
        for (size_t k = 1; k < ops.size(); ++k) d = kind == DimKind::Add ? dims.add(d, ops[k]) : dims.mul(d, ops[k]);
        break;
      default:
        if (ops.size() != 2) return Status::error("lkm: binary dim needs 2 operands");
        if ((kind == DimKind::FloorDiv || kind == DimKind::CeilDiv || kind == DimKind::Mod) && ops[1]->isConst(0))
          return Status::error("lkm: division by zero in a dim");
        if (kind == DimKind::FloorDiv) d = dims.floorDiv(ops[0], ops[1]);
        else if (kind == DimKind::CeilDiv) d = dims.ceilDiv(ops[0], ops[1]);
        else if (kind == DimKind::Mod) d = dims.mod(ops[0], ops[1]);
        else if (kind == DimKind::Min) d = dims.min(ops[0], ops[1]);
        else if (kind == DimKind::Max) d = dims.max(ops[0], ops[1]);
        else return Status::error("lkm: bad dim kind");
    }
    dimList.push_back(d);
  }
  auto dimAt = [&](uint64_t i) -> Dim { return i < dimList.size() ? dimList[i] : nullptr; };

  std::vector<Type> typeList;
  for (uint64_t i = 0, n = r.count(); i < n && r.ok(); ++i) {
    uint8_t kindByte = r.u8(), dtByte = r.u8();
    if (kindByte > static_cast<uint8_t>(TypeKind::Buffer) || dtByte > static_cast<uint8_t>(DType::Bool))
      return Status::error("lkm: bad type");
    auto kind = static_cast<TypeKind>(kindByte);
    auto dt = static_cast<DType>(dtByte);
    std::vector<Dim> shape;
    for (uint64_t k = 0, rank = r.count(64); k < rank && r.ok(); ++k) {
      Dim d = dimAt(r.uv());
      if (!d) return Status::error("lkm: bad dim reference in type");
      shape.push_back(d);
    }
    typeList.push_back(m->types().get(kind, dt, std::move(shape)));
  }
  auto typeAt = [&](uint64_t i) -> Type { return i < typeList.size() ? typeList[i] : nullptr; };

  for (uint64_t i = 0, n = r.count(); i < n && r.ok(); ++i) {
    Type t = typeAt(r.uv());
    std::string name = r.str();
    if (!t) return Status::error("lkm: bad type reference");
    m->newValue(t, std::move(name));
  }
  const uint8_t* data = base + h.dataOffset;
  for (uint64_t i = 0, n = r.count(); i < n && r.ok(); ++i) {
    ConstantData c;
    c.name = r.str();
    c.type = typeAt(r.uv());
    uint64_t off = r.uv();
    c.size = r.uv();
    if (!c.type || off > h.dataSize || c.size > h.dataSize - off)
      return Status::error("lkm: bad constant '" + c.name + "'");
    {  // the stored size must match the type exactly (kernels trust the type)
      uint64_t bytes = dtypeSize(c.type->dtype);
      bool known = true;
      for (Dim d : c.type->shape) {
        if (!d->isConst() || d->value < 0 || (d->value && bytes > UINT64_MAX / static_cast<uint64_t>(d->value))) {
          known = false;
          break;
        }
        bytes *= static_cast<uint64_t>(d->value);
      }
      if (known && bytes != c.size) return Status::error("lkm: constant '" + c.name + "' size doesn't match its type");
    }
    if (split) {
      if (i >= split->size() || (*split)[i].size != c.size)
        return Status::error("lkm: split constant '" + c.name + "' missing or of the wrong size");
      c.bytes = (*split)[i].bytes;
      c.owner = (*split)[i].owner;
    } else {
      c.bytes = data + off;
      c.owner = owner;
    }
    m->addConstant(std::move(c));
  }
  auto ids = [&](std::vector<ValueId>& v) {
    for (uint64_t k = 0, n = r.count(); k < n && r.ok(); ++k) v.push_back(static_cast<ValueId>(r.uv()));
  };
  for (Region* reg : {&m->init, &m->exec}) {
    ids(reg->args);
    if (h.version >= 2) {
      uint64_t n = r.count();
      if (n != 0 && n != reg->args.size()) return Status::error("lkm: bad mutable-arg flags");
      for (uint64_t k = 0; k < n && r.ok(); ++k) reg->mutableArgs.push_back(r.u8() != 0);
    }
    ids(reg->state);
    for (uint64_t i = 0, n = r.count(); i < n && r.ok(); ++i) {
      Op op;
      uint8_t kindByte = r.u8();
      if (kindByte > static_cast<uint8_t>(OpKind::Return)) return Status::error("lkm: bad op kind");
      op.kind = static_cast<OpKind>(kindByte);
      op.name = r.str();
      ids(op.operands);
      ids(op.outs);
      ids(op.results);
      for (uint64_t k = 0, na = r.count(); k < na && r.ok(); ++k) {
        NamedAttr a;
        a.name = r.str();
        switch (r.u8()) {
          case 0: a.attr.value = r.u8() != 0; break;
          case 1: a.attr.value = r.sv(); break;
          case 2: a.attr.value = r.f64(); break;
          case 3: a.attr.value = r.str(); break;
          case 4: {
            Attr::Ints v;
            for (uint64_t j = 0, nv = r.count(); j < nv && r.ok(); ++j) v.push_back(r.sv());
            a.attr.value = std::move(v);
            break;
          }
          case 5: {
            Attr::Floats v;
            for (uint64_t j = 0, nv = r.count(); j < nv && r.ok(); ++j) v.push_back(r.f64());
            a.attr.value = std::move(v);
            break;
          }
          case 6: {
            Dim d = dimAt(r.uv());
            if (!d) return Status::error("lkm: bad dim attribute");
            a.attr.value = d;
            break;
          }
          default: return Status::error("lkm: bad attribute tag");
        }
        op.attrs.push_back(std::move(a));
      }
      op.data = r.bytes();
      for (uint64_t k = 0, nd = r.count(); k < nd && r.ok(); ++k) {
        Dim d = dimAt(r.uv());
        if (!d) return Status::error("lkm: bad dim reference in op");
        op.dims.push_back(d);
      }
      for (uint64_t k = 0, ni = r.count(); k < ni && r.ok(); ++k) op.ints.push_back(r.sv());
      for (uint64_t k = 0, ns = r.count(); k < ns && r.ok(); ++k) {
        uint64_t s = r.uv();
        if (s >= symIds.size()) return Status::error("lkm: bad symbol reference");
        op.symbols.push_back(symIds[s]);
      }
      reg->ops.push_back(std::move(op));
    }
  }
  if (!r.ok()) return Status::error("lkm: corrupt metadata");
  auto bad = [&](const std::vector<ValueId>& vs) {
    for (ValueId v : vs)
      if (v >= m->numValues()) return true;
    return false;
  };
  for (const Region* reg : {&m->init, &m->exec}) {
    if (bad(reg->args) || bad(reg->state)) return Status::error("lkm: bad value reference");
    for (const Op& op : reg->ops)
      if (bad(op.operands) || bad(op.outs) || bad(op.results)) return Status::error("lkm: bad value reference");
  }
  return m;
}

}  // namespace

Expected<std::vector<uint8_t>> writeModelBytes(const Model& m) {
  for (const Region* r : {&m.init, &m.exec})
    for (const Op& op : r->ops)
      if (op.kind == OpKind::ConstExternal)
        return Status::error("lkm: external constant '" + op.name + "' must be internalized before export");
  Writer w;
  std::vector<std::pair<const ConstantData*, uint64_t>> data;
  LK_RETURN_IF_ERROR(ModelEncoder(m).encode(w, data));
  Header h{};
  std::memcpy(h.magic, kMagic, 4);
  h.version = kVersion;
  h.metaOffset = kHeaderSize;
  h.metaSize = w.out.size();
  h.dataOffset = alignUp(kHeaderSize + w.out.size(), kDataAlign);
  uint64_t dataEnd = 0;
  for (auto& [c, off] : data) dataEnd = std::max<uint64_t>(dataEnd, off + c->size);
  h.dataSize = dataEnd;
  std::vector<uint8_t> out(h.dataOffset + h.dataSize, 0);
  std::memcpy(out.data(), &h, sizeof h);
  std::memcpy(out.data() + kHeaderSize, w.out.data(), w.out.size());
  for (auto& [c, off] : data)
    if (c->size) std::memcpy(out.data() + h.dataOffset + off, c->bytes, c->size);
  return out;
}

Expected<LkmSplit> lkmSplit(const Model& m) {
  for (const Region* r : {&m.init, &m.exec})
    for (const Op& op : r->ops)
      if (op.kind == OpKind::ConstExternal)
        return Status::error("lkm: external constant '" + op.name + "' must be internalized before export");
  Writer w;
  std::vector<std::pair<const ConstantData*, uint64_t>> data;
  LK_RETURN_IF_ERROR(ModelEncoder(m).encode(w, data));
  Header h{};
  std::memcpy(h.magic, kMagic, 4);
  h.version = kVersion;
  h.metaOffset = kHeaderSize;
  h.metaSize = w.out.size();
  h.dataOffset = alignUp(kHeaderSize + w.out.size(), kDataAlign);
  uint64_t dataEnd = 0;
  for (auto& [c, off] : data) dataEnd = std::max<uint64_t>(dataEnd, off + c->size);
  h.dataSize = dataEnd;
  LkmSplit s;
  s.skeleton.assign(h.dataOffset, 0);  // as writeModelBytes: header, metadata, zero padding up to the data
  std::memcpy(s.skeleton.data(), &h, sizeof h);
  std::memcpy(s.skeleton.data() + kHeaderSize, w.out.data(), w.out.size());
  for (auto& [c, off] : data) s.constants.emplace_back(c->size ? c->bytes : nullptr, c->size);
  return s;
}

Expected<std::unique_ptr<Model>> readModelBytes(std::shared_ptr<const std::vector<uint8_t>> bytes) {
  const uint8_t* base = bytes->data();
  size_t n = bytes->size();
  return decode(base, n, std::move(bytes));
}

Status writeModelFile(const Model& m, const std::string& path) {
  auto bytes = writeModelBytes(m);
  if (!bytes) return bytes.status();
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return Status::error("cannot open " + path + " for writing");
  size_t written = std::fwrite(bytes->data(), 1, bytes->size(), f);
  bool closed = std::fclose(f) == 0;
  if (written != bytes->size() || !closed) return Status::error("failed writing " + path);
  return Status::ok();
}

namespace {

// Read-only private mapping of a whole file (page-aligned); an empty file maps to nothing.
Expected<SplitConstant> mapFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return Status::error("cannot open " + path);
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return Status::error("cannot stat " + path);
  }
  size_t size = static_cast<size_t>(st.st_size);
  if (size == 0) {
    ::close(fd);
    return SplitConstant{};
  }
  void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) return Status::error("cannot mmap " + path);
  std::shared_ptr<const void> owner(p, [size](const void* q) { ::munmap(const_cast<void*>(q), size); });
  return SplitConstant{static_cast<const uint8_t*>(p), size, std::move(owner)};
}

}  // namespace

Expected<std::unique_ptr<Model>> readModelFile(const std::string& path) {
  auto f = mapFile(path);
  if (!f) return f.status();
  if (!f->bytes) return Status::error("lkm: file too small");
  return decode(f->bytes, f->size, f->owner);
}

Expected<std::vector<std::pair<uint64_t, uint64_t>>> lkmConstantRanges(const std::string& path) {
  auto f = mapFile(path);
  if (!f) return f.status();
  if (!f->bytes) return Status::error("lkm: file too small");
  auto m = decode(f->bytes, f->size, f->owner);
  if (!m) return m.status();
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (const ConstantData& c : (*m)->constants)
    out.emplace_back(c.size ? static_cast<uint64_t>(c.bytes - f->bytes) : 0, c.size);
  return out;
}

Expected<std::unique_ptr<Model>> readModelSplit(const std::string& skeletonPath,
                                                const std::vector<std::string>& constantPaths) {
  auto sk = mapFile(skeletonPath);
  if (!sk) return sk.status();
  if (!sk->bytes) return Status::error("lkm: file too small");
  std::vector<SplitConstant> split;
  split.reserve(constantPaths.size());
  for (const std::string& p : constantPaths) {
    auto c = mapFile(p);
    if (!c) return c.status();
    split.push_back(std::move(*c));
  }
  return decode(sk->bytes, sk->size, sk->owner, &split);
}

}  // namespace lk
