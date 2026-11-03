// Kernel thread pool (user decision 2026-11-01: small kernels run inline on the
// calling thread; big kernels run their work in parallel here).
//
// Persistent workers at QOS_CLASS_USER_INTERACTIVE (macOS/iOS have no hard core
// affinity; this QoS prefers P-cores), sized to the P-core count — plus the E-cores on chips with < 4 P-cores
// (the iPhone: 2 P + 4 E, where all 6 help).
// parallelFor() hands out task indices with an atomic counter; the caller
// participates, then waits. Workers spin 50 us before sleeping so back-to-back
// kernels don't pay a wake-up each time. The hand-off is lock-free (workers join
// a job through atomics; the mutex / condition variable are only for sleeping):
// 8 trivial tasks on the M2 Max cost 12 -> 4-6 us per parallelFor.
#pragma once

#include <cstdint>
#include <functional>

namespace lk::rt {

class ThreadPool {
 public:
  // Process-wide pool (created on first use).
  static ThreadPool& global();

  // Runs fn(i) for i in [0, n) across the pool + the calling thread; returns
  // when all are done. n <= 1 runs inline. Not reentrant (no nested calls).
  void parallelFor(int64_t n, const std::function<void(int64_t)>& fn);

  int threads() const;  // workers + the caller

  // Workers still spinning after the last job go to sleep now (they would otherwise compete for the cores with a
  // multi-threaded library call, e.g. Accelerate's own BLAS threads). One relaxed atomic add; the next parallelFor
  // pays a wake-up.
  void park();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ~ThreadPool();

 private:
  explicit ThreadPool(int workers);
  struct Impl;
  Impl* impl_;
};

// P-core count of this device (hw.perflevel0.physicalcpu), >= 1.
int performanceCores();

}  // namespace lk::rt
