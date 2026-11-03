// bake-init: evaluate init at compile time into constants (design-compiler-passes.md "bake-init").
#pragma once

#include <memory>
#include <vector>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"
#include "lokinfer/runtime/runtime.h"

namespace lk::rt {

struct BakeStats {
  size_t baked = 0;           // init results replaced by constants
  size_t bakedBytes = 0;      // their bytes (stored in the model now)
  size_t initOpsBefore = 0, initOpsAfter = 0;
};

// Runs `model`'s init (an Executable with `externals` bound) and returns a copy of the model where every init
// result that is computed data (the contents of a plain buffer.alloc written by init kernels, or a view of one) is a
// `const.internal` holding that data, resident on the device of the kernel that wrote it; init ops nothing needs any
// more are removed (e.g. the original weights only a hoisted transpose read). Left alone ("safe" = same results):
// memory without meaningful contents (static arenas), persistent buffers (state that calls write), constants, and
// everything when init defines symbols (dim.read / mem.plan in init). Use for export: loading the baked model does
// no init work, and the file stores the transformed weights instead of the originals.
Expected<std::unique_ptr<Model>> bakeInit(std::shared_ptr<const Model> model, std::vector<ExternalBinding> externals,
                                          BakeStats* stats = nullptr);

}  // namespace lk::rt
