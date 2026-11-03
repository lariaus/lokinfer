#include "lokinfer/compiler/passes.h"

#include <chrono>

#include "lokinfer/ir/verify.h"

namespace lk {

namespace {

const PassInfo kPasses[] = {
    {"dce", "remove ops whose results are unused (keeps side-effecting / inout kernels, dim.read)", dce},
    {"bufferize", "tensors -> buffers, allocs, outs, frees", bufferize},
    {"plan-memory", "allocs/frees -> arena views (static) or mem.plan (dynamic)", planMemory},
    {"hoist-init", "move input-independent work into init", hoistInit},
};

thread_local std::vector<std::string>* tlNotes = nullptr;

}  // namespace

bool passNotesEnabled() { return tlNotes != nullptr; }
void passNote(std::string line) {
  if (tlNotes) tlNotes->push_back(std::move(line));
}

std::span<const PassInfo> allPasses() { return kPasses; }

const PassInfo* findPass(std::string_view name) {
  for (const PassInfo& p : kPasses)
    if (name == p.name) return &p;
  return nullptr;
}

const std::vector<std::string>& defaultPipeline() {
  static const std::vector<std::string> p = {"dce", "bufferize", "plan-memory", "hoist-init"};
  return p;
}

Status runPasses(Model& m, std::span<const std::string> names, bool verifyEach, std::vector<PassStats>* stats) {
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  for (const std::string& n : names) {
    const PassInfo* p = findPass(n);
    if (!p) return Status::error("unknown pass '" + n + "'");
    std::vector<std::string> notes;
    tlNotes = stats ? &notes : nullptr;
    auto t0 = Clock::now();
    Status s = p->fn(m);
    auto t1 = Clock::now();
    tlNotes = nullptr;
    if (!s) return Status::error(n + ": " + s.message());
    if (verifyEach)
      if (Status v = verifyModel(m); !v) return Status::error("IR invalid after pass '" + n + "': " + v.message());
    if (stats)
      stats->push_back(
          {n, ms(t0, t1), verifyEach ? ms(t1, Clock::now()) : 0.0, m.init.ops.size(), m.exec.ops.size(), std::move(notes)});
  }
  return Status::ok();
}

}  // namespace lk
