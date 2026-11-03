#include "lokinfer/runtime/thread_pool.h"

#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace lk::rt {

int performanceCores() {
  int n = 0;
  size_t len = sizeof n;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) != 0 || n < 1)
    n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  return n;
}

// Workers spin this long for the next job before sleeping. 50 us; measured 2026-11-01 with this pool on the iPhone 18
// Pro Max (ResNet-18 / MobileNetV2 N=1): 15 us ~ 50 us (within noise), 200 us and 1000 us slower (4.7 / 6.5 ms vs
// 4.2-4.6 ms ResNet-18): spinning E-cores take power from the P-cores.
constexpr int kSpinUs = 50;

// Runtime variants for benchmarking (docs/benchmarking.md "CPU runtime variants"), read once when the pool is created:
//   LOKINFER_THREADS=N           pool size incl. the caller (1 = mono-thread: every kernel runs inline on the caller)
//   LOKINFER_POOL_QOS=default|utility|user-interactive   the workers' QoS (default: user-interactive, P-core preferring)
static qos_class_t workerQos() {
  const char* e = std::getenv("LOKINFER_POOL_QOS");
  if (e && std::strcmp(e, "default") == 0) return QOS_CLASS_DEFAULT;
  if (e && std::strcmp(e, "utility") == 0) return QOS_CLASS_UTILITY;
  if (e && std::strcmp(e, "user-initiated") == 0) return QOS_CLASS_USER_INITIATED;
  return QOS_CLASS_USER_INTERACTIVE;
}

// Lock-free job hand-off. `generation` is even when a job is published, odd while the caller sets one up. A worker
// that sees a new even generation registers in `active` and re-checks the generation (Dekker with the caller's
// "odd, then wait active == 0"), so it never reads a half-written job. Workers only take the mutex to sleep after
// spinning; the caller only notifies when someone sleeps.
struct ThreadPool::Impl {
  std::vector<std::thread> workers;
  std::mutex mu;
  std::condition_variable cv;
  const std::function<void(int64_t)>* fn = nullptr;
  int64_t n = 0;
  alignas(128) std::atomic<int64_t> next{0};
  alignas(128) std::atomic<int64_t> done{0};
  alignas(128) std::atomic<uint64_t> generation{0};
  alignas(128) std::atomic<int> active{0};
  alignas(128) std::atomic<int> sleepers{0};
  alignas(128) std::atomic<uint32_t> parkEpoch{0};  // bumped by park(): spinning workers go to sleep now
  std::atomic<bool> stop{false};

  void runTasks() {
    for (int64_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) {
      (*fn)(i);
      done.fetch_add(1, std::memory_order_release);
    }
  }

  void workerLoop() {
    static const qos_class_t qos = workerQos();
    pthread_set_qos_class_self_np(qos, 0);
    uint64_t seen = 0;
    auto ready = [&](uint64_t g) { return g != seen && (g & 1) == 0; };
    while (true) {
      uint64_t g = generation.load(std::memory_order_acquire);
      if (!ready(g)) {
        auto spinUntil = std::chrono::steady_clock::now() + std::chrono::microseconds(kSpinUs);
        const uint32_t parked = parkEpoch.load(std::memory_order_relaxed);
        int k = 0;
        while (!ready(g = generation.load(std::memory_order_acquire))) {
          if (parkEpoch.load(std::memory_order_relaxed) != parked) break;
          if ((++k & 63) == 0 && std::chrono::steady_clock::now() >= spinUntil) break;
        }
        if (!ready(g)) {
          std::unique_lock<std::mutex> lock(mu);
          sleepers.fetch_add(1, std::memory_order_seq_cst);
          cv.wait(lock, [&] { return stop.load() || ready(g = generation.load(std::memory_order_seq_cst)); });
          sleepers.fetch_sub(1, std::memory_order_relaxed);
          if (stop.load()) return;
        }
      }
      active.fetch_add(1, std::memory_order_seq_cst);
      if (generation.load(std::memory_order_seq_cst) == g) {
        seen = g;
        runTasks();
      }
      active.fetch_sub(1, std::memory_order_release);
    }
  }
};

ThreadPool::ThreadPool(int workers) : impl_(new Impl()) {
  for (int i = 0; i < workers; ++i) impl_->workers.emplace_back([this] { impl_->workerLoop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->stop = true;
  }
  impl_->cv.notify_all();
  for (auto& t : impl_->workers) t.join();
  delete impl_;
}

// Pool size: the P-cores, plus the E-cores on chips with few P-cores. Measured 2026-11-01: on the M2 Max (8 P + 4 E)
// adding the E-cores slows the tiled conv (12 streams < 8), on the iPhone 18 Pro Max (2 P + 4 E) 6 streams are fastest
// (M2 conv proof; ExecuTorch's best thread count is also 6 there).
static int poolSize() {
  if (const char* e = std::getenv("LOKINFER_THREADS")) {
    const int n = std::atoi(e);
    if (n >= 1) return n;
  }
  const int p = performanceCores();
  if (p >= 4) return p;
  int e = 0;
  size_t len = sizeof e;
  if (sysctlbyname("hw.perflevel1.physicalcpu", &e, &len, nullptr, 0) != 0 || e < 0) e = 0;
  return p + e;
}

ThreadPool& ThreadPool::global() {
  static auto* pool = new ThreadPool(std::max(0, poolSize() - 1));
  return *pool;
}

void ThreadPool::park() { impl_->parkEpoch.fetch_add(1, std::memory_order_relaxed); }

int ThreadPool::threads() const { return static_cast<int>(impl_->workers.size()) + 1; }

void ThreadPool::parallelFor(int64_t n, const std::function<void(int64_t)>& fn) {
  if (n <= 0) return;
  if (n == 1 || impl_->workers.empty()) {
    for (int64_t i = 0; i < n; ++i) fn(i);
    return;
  }
  Impl& im = *impl_;
  im.generation.fetch_add(1, std::memory_order_seq_cst);  // odd: setting up
  while (im.active.load(std::memory_order_seq_cst) != 0) {
  }
  im.fn = &fn;
  im.n = n;
  im.next.store(0, std::memory_order_relaxed);
  im.done.store(0, std::memory_order_relaxed);
  im.generation.fetch_add(1, std::memory_order_seq_cst);  // even: published
  if (im.sleepers.load(std::memory_order_seq_cst) != 0) {
    std::lock_guard<std::mutex> lock(im.mu);
    im.cv.notify_all();
  }
  im.runTasks();
  while (im.done.load(std::memory_order_acquire) < n || im.active.load(std::memory_order_acquire) != 0) {
  }
}

}  // namespace lk::rt
