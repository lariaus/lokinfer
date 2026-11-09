// Binary model files (.lkm), see docs/impl-notes/design-ir.md §8.
//
//   [header 64 B][meta: varint-encoded symbols/dims/types/values/constants/
//   regions][padding][data: each constant 16 KiB-aligned]
//
// Loading mmaps the file; internal constants point straight into the mapping
// (zero-copy, and page-aligned so the runtime can wrap them as no-copy
// MTLBuffers).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"

namespace lk {

// External constants can't be stored: internalize them first (their data
// must be attached to Model::constants as internal constants).
Status writeModelFile(const Model& m, const std::string& path);
Expected<std::unique_ptr<Model>> readModelFile(const std::string& path);

// In-memory variants (tests, remote executor transport). The returned Model
// from readModelBytes references `bytes` (kept alive via the shared_ptr).
Expected<std::vector<uint8_t>> writeModelBytes(const Model& m);
Expected<std::unique_ptr<Model>> readModelBytes(std::shared_ptr<const std::vector<uint8_t>> bytes);

// Split transport (remote executor: weights cached on the device independently of the graph). A .lkm splits into
// its "skeleton" (the bytes before the data section: header + metadata) and one file per constant (the bytes of
// lkmConstantRanges' (file offset, size), in constant order). readModelSplit maps each file (page-aligned, zero-copy,
// as readModelFile) and gives the same Model as reading the whole .lkm.
Expected<std::vector<std::pair<uint64_t, uint64_t>>> lkmConstantRanges(const std::string& path);
// The same split straight from a Model, without writing the .lkm (no copy of the data): the skeleton bytes (identical
// to the .lkm's first dataOffset bytes) and each constant's (data, size), in constant order. The pointers alias `m`.
struct LkmSplit {
  std::vector<uint8_t> skeleton;
  std::vector<std::pair<const uint8_t*, uint64_t>> constants;
};
Expected<LkmSplit> lkmSplit(const Model& m);
Expected<std::unique_ptr<Model>> readModelSplit(const std::string& skeletonPath,
                                                const std::vector<std::string>& constantPaths);

}  // namespace lk
