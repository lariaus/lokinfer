// internalize: const.external -> const.internal (design-ir.md §5; design-python-api.md "internalize").
//
// External constants are bound by pointer when an executable is created (tensors captured while tracing). A model
// file can't hold pointers, and a model that keeps aliasing caller tensors changes when the caller writes them, so
// `internalize` turns every const.external whose data is given into a const.internal that owns (copy = true) or
// references (copy = false, the caller keeps `owner` alive) its bytes. Not a pipeline pass: it needs the data.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "lokinfer/compiler/passes.h"

namespace lk {

Status internalize(Model& m, std::span<const ExternalData> data, bool copy, size_t* count) {
  size_t n = 0;
  for (Region* r : {&m.init, &m.exec}) {
    for (Op& op : r->ops) {
      if (op.kind != OpKind::ConstExternal) continue;
      const ExternalData* d = nullptr;
      for (const ExternalData& e : data)
        if (e.name == op.name) d = &e;
      if (!d) return Status::error("internalize: no data for external constant '" + op.name + "'");
      if (!m.findConstant(op.name)) {
        Type t = m.typeOf(op.results[0]);
        size_t bytes = dtypeSize(t->dtype);
        for (Dim dim : t->shape) {
          if (!dim->isConst() || dim->value < 0)
            return Status::error("internalize: external constant '" + op.name + "' has a symbolic shape");
          bytes *= static_cast<size_t>(dim->value);
        }
        if (d->size != bytes)
          return Status::error("internalize: '" + op.name + "' has " + std::to_string(d->size) + " bytes, its type " +
                               std::to_string(bytes));
        ConstantData c{op.name, t, d->owner, d->bytes, bytes};
        if (copy) {
          // 16 KiB-aligned like .lkm data, so GPU models can wrap it without another copy.
          constexpr size_t kPage = 16384;
          void* p = std::aligned_alloc(kPage, (std::max<size_t>(bytes, 1) + kPage - 1) / kPage * kPage);
          if (!p) return Status::error("internalize: out of memory");
          if (bytes) std::memcpy(p, d->bytes, bytes);
          c.owner = std::shared_ptr<const void>(p, [](const void* q) { std::free(const_cast<void*>(q)); });
          c.bytes = static_cast<const uint8_t*>(p);
        }
        m.addConstant(std::move(c));
      }
      op.kind = OpKind::ConstInternal;
      ++n;
    }
  }
  if (count) *count = n;
  return Status::ok();
}

}  // namespace lk
