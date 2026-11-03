#include "lokinfer/compiler/memplan.h"

#include <algorithm>
#include <numeric>

namespace lk {

namespace {

int64_t alignUp(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

}  // namespace

PlanResult planOffsets(std::span<const PlanBuffer> bufs, int64_t alignment, std::span<const PlacedBuffer> fixed) {
  const size_t n = bufs.size();
  PlanResult r;
  r.offsets.assign(n, 0);
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (bufs[a].size != bufs[b].size) return bufs[a].size > bufs[b].size;
    return a < b;
  });

  struct Placed {
    int64_t offset, size;
    int64_t start, end;
  };
  std::vector<Placed> placed;  // sorted by offset
  placed.reserve(n + fixed.size());
  for (const PlacedBuffer& f : fixed) {
    placed.push_back({f.offset, f.size, f.start, f.end});
    r.total = std::max(r.total, f.offset + f.size);
  }
  std::sort(placed.begin(), placed.end(), [](const Placed& a, const Placed& c) { return a.offset < c.offset; });
  std::vector<const Placed*> conflicts;
  for (size_t idx : order) {
    const PlanBuffer& b = bufs[idx];
    int64_t size = alignUp(std::max<int64_t>(b.size, 1), alignment);
    // Best-fit: smallest gap between time-overlapping placed buffers that fits.
    int64_t best = -1, bestGap = INT64_MAX, cursor = 0;
    for (const Placed& p : placed) {
      if (p.end < b.start || b.end < p.start) continue;
      if (p.offset >= cursor) {
        int64_t gap = p.offset - cursor;
        if (gap >= size && gap < bestGap) {
          best = cursor;
          bestGap = gap;
        }
      }
      cursor = std::max(cursor, p.offset + p.size);
    }
    if (best < 0) best = cursor;
    r.offsets[idx] = best;
    r.total = std::max(r.total, best + size);
    Placed np{best, size, b.start, b.end};
    placed.insert(std::upper_bound(placed.begin(), placed.end(), np,
                                   [](const Placed& a, const Placed& c) { return a.offset < c.offset; }),
                  np);
  }
  return r;
}

int64_t planLowerBound(std::span<const PlanBuffer> bufs, int64_t alignment) {
  std::vector<std::pair<int64_t, int64_t>> events;  // (time, delta); frees after uses at same time
  for (const PlanBuffer& b : bufs) {
    int64_t s = alignUp(std::max<int64_t>(b.size, 1), alignment);
    events.emplace_back(2 * b.start, s);
    events.emplace_back(2 * b.end + 1, -s);
  }
  std::sort(events.begin(), events.end());
  int64_t live = 0, peak = 0;
  for (auto& [t, d] : events) {
    live += d;
    peak = std::max(peak, live);
  }
  return peak;
}

}  // namespace lk
