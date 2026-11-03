#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <random>

#include "lokinfer/compiler/memplan.h"
#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/text.h"
#include "lokinfer/ir/verify.h"
#include "lokinfer/ir/binary.h"
#include "lokinfer/runtime/bake_init.h"
#include "lokinfer/runtime/runtime.h"
#include "../../src/compiler/analysis.h"

using namespace lk;

TEST_CASE("memplan: no two time-overlapping buffers overlap in memory", "[memplan]") {
  std::mt19937 rng(42);
  for (int trial = 0; trial < 200; ++trial) {
    int n = 1 + rng() % 60;
    std::vector<PlanBuffer> bufs;
    for (int i = 0; i < n; ++i) {
      int64_t s = rng() % 100, len = rng() % 20;
      bufs.push_back({static_cast<int64_t>(1 + rng() % 5000), s, s + len});
    }
    PlanResult r = planOffsets(bufs, 256);
    int64_t lb = planLowerBound(bufs, 256);
    REQUIRE(r.total >= lb);
    for (int i = 0; i < n; ++i) {
      CHECK(r.offsets[i] % 256 == 0);
      CHECK(r.offsets[i] + bufs[i].size <= r.total);
      for (int j = i + 1; j < n; ++j) {
        bool timeOverlap = !(bufs[i].end < bufs[j].start || bufs[j].end < bufs[i].start);
        bool memOverlap = r.offsets[i] < r.offsets[j] + bufs[j].size && r.offsets[j] < r.offsets[i] + bufs[i].size;
        if (timeOverlap) REQUIRE_FALSE(memOverlap);
      }
    }
  }
}

TEST_CASE("memplan: chain of equal buffers ping-pongs between two slots", "[memplan]") {
  std::vector<PlanBuffer> bufs;
  for (int i = 0; i < 10; ++i) bufs.push_back({1024, i, i + 1});
  PlanResult r = planOffsets(bufs, 256);
  CHECK(r.total == 2048);
}

namespace {

// A Kokoro-sized synthetic graph: n chained kernels with some views and
// constants, half on GPU, symbolic shape.
std::string syntheticGraph(int n) {
  std::string s = "model {\n  init {\n    return\n  }\n  exec(%x: tensor<f32[T, 512]>) {\n";
  std::string prev = "%x";
  for (int i = 0; i < n; ++i) {
    std::string w = "%w" + std::to_string(i), v = "%v" + std::to_string(i);
    s += "    " + w + " = const.internal @\"w" + std::to_string(i) + "\" : tensor<f32[512, 512]>\n";
    const char* dev = i % 2 ? "gpu" : "cpu";
    if (i > 0) {  // explicit device boundary between alternating kernels
      std::string sv = "%s" + std::to_string(i);
      s += "    " + sv + " = sync " + (i % 2 ? "cpu_to_gpu " : "gpu_to_cpu ") + prev + " : tensor<f32[T, 512]>\n";
      prev = sv;
    }
    s += "    " + v + " = kernel \"gemm_f32_" + dev + "\"(" + prev + ", " + w + ") : tensor<f32[T, 512]>\n";
    if (i % 5 == 4) {
      s += "    %r" + std::to_string(i) + " = view " + v + "[0] : tensor<f32[512*T]>\n";
      s += "    %q" + std::to_string(i) + " = kernel \"relu_f32_" + dev + "\"(%r" + std::to_string(i) + ") : tensor<f32[512*T]>\n";
      s += "    %u" + std::to_string(i) + " = view %q" + std::to_string(i) + "[0] : tensor<f32[T, 512]>\n";
      v = "%u" + std::to_string(i);
    }
    prev = v;
  }
  s += "    return " + prev + "\n  }\n}\n";
  return s;
}

}  // namespace

TEST_CASE("pipeline: Kokoro-sized graph compiles fast", "[pipeline]") {
  std::string src = syntheticGraph(1000);
  // Best of 3 runs: a single run on a just-launched iOS test host can land on an efficiency core or be preempted, and
  // this guards against algorithmic blowups, not scheduling noise.
  double ms = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    auto m = parseModel(src);
    REQUIRE(m.isOk());
    double total = 0;
    for (const std::string& pass : defaultPipeline()) {
      auto t0 = std::chrono::steady_clock::now();
      Status s = runPasses(**m, std::span<const std::string>(&pass, 1));
      auto t1 = std::chrono::steady_clock::now();
      INFO(s.message());
      REQUIRE(s.isOk());
      double passMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (rep == 0) WARN("  " << pass << ": " << passMs << " ms");
      total += passMs;
    }
    ms = std::min(ms, total);
  }
  WARN("pipeline on ~1400 ops: " << ms << " ms (best of 3)");
  // Debug builds are slower; the design target (< 1 ms, met: 0.92 ms on the M2 Max, docs/compiler.md) is measured in
  // Release. This guards against accidental quadratic blowups (and, in Release, against a 3x regression).
  CHECK(ms < 200.0);
#ifdef NDEBUG
  CHECK(ms < 3.0);
#endif
}

// ---- mem.plan at run time: cache keyed by symbol values, one arena shared by the plans of a call -----------------

namespace {

// Plan 0 (on entry, sized by T) and plan 1 (after dim.read F) share the CPU pool's arena; %b (plan 0) is live across
// plan 1, so plan 1 must place %c, %d around it.
const char* kTwoPlans = R"(model {
  init {
    return
  }
  exec(%n: tensor<i64[1]>, %x: tensor<f32[T]>, %z: tensor<f32[F]>) {
    %a = kernel "add_f32_cpu"(%x, %x) : tensor<f32[T]>
    %b = kernel "add_f32_cpu"(%a, %x) : tensor<f32[T]>
    dim.read %n[0] -> F
    %c = kernel "add_f32_cpu"(%z, %z) : tensor<f32[F]>
    %d = kernel "add_f32_cpu"(%c, %z) : tensor<f32[F]>
    %e = kernel "add_f32_cpu"(%d, %z) : tensor<f32[F]>
    %o = kernel "add_f32_cpu"(%b, %x) : tensor<f32[T]>
    return %e, %o
  }
}
)";

std::unique_ptr<rt::Executable> compileExe(const std::string& src) {
  auto m = parseModel(src);
  INFO(m.status().message());
  REQUIRE(m.isOk());
  std::shared_ptr<Model> model(m.take());
  Status s = runPasses(*model, defaultPipeline());
  INFO(s.message());
  REQUIRE(s.isOk());
  auto e = rt::Executable::create(model, {});
  INFO(e.status().message());
  REQUIRE(e.isOk());
  return e.take();
}

void runTwoPlans(rt::Executable& e, int64_t t, int64_t f) {
  std::vector<int64_t> n = {f};
  std::vector<float> x(t), z(f);
  for (int64_t i = 0; i < t; ++i) x[i] = float(i % 7) - 3.f;
  for (int64_t i = 0; i < f; ++i) z[i] = float(i % 5) + 0.5f;
  rt::TensorView in[] = {{n.data(), DType::I64, {1}}, {x.data(), DType::F32, {t}}, {z.data(), DType::F32, {f}}};
  std::vector<rt::OwnedTensor> out;
  Status s = e.run(in, out);
  INFO(s.message());
  REQUIRE(s.isOk());
  const float* eo = static_cast<const float*>(out[0].data());
  const float* oo = static_cast<const float*>(out[1].data());
  for (int64_t i = 0; i < f; ++i) REQUIRE(eo[i] == 4 * z[i]);
  for (int64_t i = 0; i < t; ++i) REQUIRE(oo[i] == 4 * x[i]);
}

}  // namespace

TEST_CASE("mem.plan: repeated symbol values hit the plan cache", "[memplan][runtime]") {
  auto e = compileExe(kTwoPlans);
  runTwoPlans(*e, 1000, 300);
  auto s0 = e->memPlanStats();
  CHECK(s0.misses == 2);
  CHECK(s0.hits == 0);
  for (int i = 0; i < 3; ++i) runTwoPlans(*e, 1000, 300);  // same shapes: no re-planning
  auto s1 = e->memPlanStats();
  CHECK(s1.misses == 2);
  CHECK(s1.hits == 6);
  runTwoPlans(*e, 1000, 500);  // plan 0 still hits (depends on T only); plan 1 depends on T and F
  auto s2 = e->memPlanStats();
  CHECK(s2.misses == 3);
  CHECK(s2.hits == 7);
  runTwoPlans(*e, 1000, 300);  // back to an earlier shape: still cached
  CHECK(e->memPlanStats().misses == 3);
  runTwoPlans(*e, 77, 9);  // smaller: fits, correct
  CHECK(e->memPlanStats().overflows == 0);
}

TEST_CASE("mem.plan: the plans of a call share one arena", "[memplan][runtime]") {
  auto e = compileExe(kTwoPlans);
  // Call 1 sizes the arena from plan 0 alone; plan 1 (bigger, F >> T) overflows into its own storage this call...
  runTwoPlans(*e, 64, 100000);
  auto s1 = e->memPlanStats();
  CHECK(s1.overflows == 1);
  // ... and the next call grows the shared arena at plan 0, so both plans fit in it.
  runTwoPlans(*e, 64, 100000);
  auto s2 = e->memPlanStats();
  CHECK(s2.overflows == 1);
  // %b (256 B) stays live across plan 1 (%c, %d: 400000 B each, live together) — one arena holds all three.
  CHECK(s2.arenaBytes[0] >= size_t(2 * 400000 + 256));
  CHECK(s2.arenaBytes[1] == 0);  // no gpu_only buffers
  runTwoPlans(*e, 64, 1000);
  CHECK(e->memPlanStats().overflows == 1);
}

TEST_CASE("rewriteOps: in-place drops and inserts match a rebuilt list", "[compiler]") {
  std::mt19937 rng(7);
  auto mk = [](int64_t tag) {
    Op op;
    op.kind = OpKind::Kernel;
    op.ints.push_back(tag);
    return op;
  };
  for (int trial = 0; trial < 500; ++trial) {
    size_t n = 1 + rng() % 40;
    std::vector<Op> ops;
    std::vector<uint8_t> drop(n);
    for (size_t i = 0; i < n; ++i) {
      ops.push_back(mk(static_cast<int64_t>(i)));
      drop[i] = rng() % 3 == 0;
    }
    std::vector<int64_t> expect;
    std::vector<Op> prologue;
    for (size_t k = rng() % 4; k-- > 0;) {
      prologue.push_back(mk(1000 + static_cast<int64_t>(prologue.size())));
      expect.push_back(prologue.back().ints[0]);
    }
    std::vector<std::pair<size_t, Op>> ins;
    for (size_t k = rng() % 6; k-- > 0;) ins.emplace_back(rng() % n, Op{});
    std::stable_sort(ins.begin(), ins.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (size_t k = 0; k < ins.size(); ++k) ins[k].second = mk(2000 + static_cast<int64_t>(k));
    size_t q = 0;
    for (size_t i = 0; i < n; ++i) {
      if (!drop[i]) expect.push_back(static_cast<int64_t>(i));
      for (; q < ins.size() && ins[q].first == i; ++q) expect.push_back(ins[q].second.ints[0]);
    }
    rewriteOps(ops, drop, prologue, ins);
    std::vector<int64_t> got;
    for (const Op& op : ops) got.push_back(op.ints.empty() ? -1 : op.ints[0]);
    REQUIRE(got == expect);
  }
}

// ---- bake-init --------------------------------------------------------------------------------------------------

TEST_CASE("bake-init: init results become constants, unused init work disappears", "[bake]") {
  // 2*w is hoisted into init (a kernel on a constant); the static arena (scratch for %a) is not baked.
  const char* src = R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[4]>) {
    %w = const.inline dense<[1.0, 2.0, 3.0, 4.0]> : tensor<f32[4]>
    %w2 = kernel "add_f32_cpu"(%w, %w) : tensor<f32[4]>
    %a = kernel "add_f32_cpu"(%x, %w2) : tensor<f32[4]>
    %o = kernel "mul_f32_cpu"(%a, %x) : tensor<f32[4]>
    return %o
  }
}
)";
  auto m = parseModel(src);
  REQUIRE(m.isOk());
  std::shared_ptr<Model> model(m.take());
  REQUIRE(runPasses(*model, defaultPipeline()).isOk());
  rt::BakeStats st;
  auto baked = rt::bakeInit(model, {}, &st);
  INFO(baked.status().message());
  REQUIRE(baked.isOk());
  CHECK(st.baked == 1);
  CHECK(st.bakedBytes == 16);
  CHECK(st.initOpsAfter < st.initOpsBefore);
  std::string text = printModel(**baked);
  INFO(text);
  CHECK(text.find("const.internal @\"baked") != std::string::npos);
  CHECK(text.find("kernel \"add_f32_cpu\"(%w") == std::string::npos);  // the init kernel is gone
  CHECK(text.find("arena = true") != std::string::npos);               // scratch memory stays an alloc

  std::shared_ptr<Model> bm(baked.take());
  auto e0 = rt::Executable::create(model, {});
  auto e1 = rt::Executable::create(bm, {});
  REQUIRE(e0.isOk());
  REQUIRE(e1.isOk());
  std::vector<float> x = {0.5f, -1.f, 2.f, 3.f};
  rt::TensorView in[] = {{x.data(), DType::F32, {4}}};
  std::vector<rt::OwnedTensor> o0, o1;
  REQUIRE((*e0)->run(in, o0).isOk());
  REQUIRE((*e1)->run(in, o1).isOk());
  for (int i = 0; i < 4; ++i) {
    float ref = (x[i] + 2.f * float(i + 1)) * x[i];
    CHECK(static_cast<const float*>(o0[0].data())[i] == ref);
    CHECK(static_cast<const float*>(o1[0].data())[i] == ref);
  }
  // The baked model survives the binary format (that's what it is for).
  auto bytes = writeModelBytes(*bm);
  REQUIRE(bytes.isOk());
  auto back = readModelBytes(std::make_shared<const std::vector<uint8_t>>(std::move(*bytes)));
  INFO(back.status().message());
  REQUIRE(back.isOk());
  CHECK(verifyModel(**back).isOk());
}
