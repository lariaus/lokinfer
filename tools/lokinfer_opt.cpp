// lokinfer-opt: parse textual IR, run passes, print the result.
//   lokinfer-opt [--pass=a,b,c | --pipeline] [--no-verify] [--stats] [file|-]
// The IR is verified after parsing and after every pass (lokinfer/ir/verify.h); --no-verify disables both.
// --stats prints to stderr: parse time, per-pass time (pass + verification) and op counts after each pass, then the
// final op counts per kind. Passes add their own lines (plan-memory: each static arena's bytes vs. the max-live lower
// bound and the no-reuse sum; each mem.plan's buffer count).
#include <chrono>
#include <cstdio>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/text.h"
#include "lokinfer/ir/verify.h"

namespace {

void printStats(const lk::Model& m, double parseMs, double verifyMs, size_t initOps, size_t execOps,
                const std::vector<lk::PassStats>& ps) {
  std::fprintf(stderr, "lokinfer-opt stats\n  %-14s %9.3f ms\n  %-14s %9.3f ms   init %zu ops, exec %zu ops\n", "parse",
               parseMs, "verify(input)", verifyMs, initOps, execOps);
  double total = 0;
  for (const lk::PassStats& p : ps) {
    std::fprintf(stderr, "  %-14s %9.3f ms + verify %7.3f ms   init %zu ops, exec %zu ops\n", p.name.c_str(), p.passMs,
                 p.verifyMs, p.initOps, p.execOps);
    for (const std::string& n : p.notes) std::fprintf(stderr, "    %s\n", n.c_str());
    total += p.passMs + p.verifyMs;
  }
  std::fprintf(stderr, "  %-14s %9.3f ms (passes + verification)\n", "pipeline", total);
  for (const auto* r : {&m.init, &m.exec}) {
    std::map<std::string, size_t> kinds;
    for (const lk::Op& op : r->ops) ++kinds[lk::opKindName(op.kind)];
    std::fprintf(stderr, "  %s:", r == &m.init ? "init" : "exec");
    for (auto& [k, n] : kinds) std::fprintf(stderr, " %s=%zu", k.c_str(), n);
    std::fprintf(stderr, "\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> passes;
  std::string path = "-";
  bool verify = true, stats = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--pass=", 0) == 0) {
      std::stringstream ss(a.substr(7));
      std::string p;
      while (std::getline(ss, p, ',')) passes.push_back(p);
    } else if (a == "--pipeline") {
      for (auto& p : lk::defaultPipeline()) passes.push_back(p);
    } else if (a == "--stats") {
      stats = true;
    } else if (a == "--no-verify") {
      verify = false;
    } else if (a == "--list-passes") {
      for (auto& p : lk::allPasses()) std::printf("%-16s %s\n", p.name, p.description);
      return 0;
    } else {
      path = a;
    }
  }
  std::stringstream buf;
  if (path == "-") {
    buf << std::cin.rdbuf();
  } else {
    std::ifstream f(path);
    if (!f) {
      std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
      return 1;
    }
    buf << f.rdbuf();
  }
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  auto t0 = Clock::now();
  auto m = lk::parseModel(buf.str());
  auto t1 = Clock::now();
  if (!m) {
    std::fprintf(stderr, "error: %s\n", m.status().message().c_str());
    return 1;
  }
  if (verify) {
    if (auto s = lk::verifyModel(**m); !s) {
      std::fprintf(stderr, "error: %s\n", s.message().c_str());
      return 1;
    }
  }
  auto t2 = Clock::now();
  const size_t initOps = (*m)->init.ops.size(), execOps = (*m)->exec.ops.size();
  std::vector<lk::PassStats> ps;
  if (auto s = lk::runPasses(**m, passes, verify, stats ? &ps : nullptr); !s) {
    std::fprintf(stderr, "error: %s\n", s.message().c_str());
    return 1;
  }
  if (stats) printStats(**m, ms(t0, t1), ms(t1, t2), initOps, execOps, ps);
  std::fputs(lk::printModel(**m).c_str(), stdout);
  return 0;
}
