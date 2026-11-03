// Offset assignment shared by the compiler (static plans) and the runtime
// (mem.plan at call time): greedy-by-size with best-fit gaps.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace lk {

struct PlanBuffer {
  int64_t size;   // bytes
  int64_t start;  // first op index using it (inclusive)
  int64_t end;    // last op index using it (inclusive)
};

struct PlanResult {
  std::vector<int64_t> offsets;  // bytes, aligned
  int64_t total = 0;             // arena bytes
};

// A buffer already placed in the arena (by an earlier plan sharing it): an obstacle for the buffers being planned.
struct PlacedBuffer {
  int64_t offset;  // bytes, aligned
  int64_t size;    // bytes (aligned up)
  int64_t start, end;
};

// Buffers whose [start, end] intervals intersect never overlap in memory, neither with each other nor with the
// `fixed` buffers. `total` covers the planned and the fixed buffers.
PlanResult planOffsets(std::span<const PlanBuffer> buffers, int64_t alignment,
                       std::span<const PlacedBuffer> fixed = {});

// Lower bound: max over time of the sum of live (aligned) sizes.
int64_t planLowerBound(std::span<const PlanBuffer> buffers, int64_t alignment);

}  // namespace lk
