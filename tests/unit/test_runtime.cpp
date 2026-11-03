// End-to-end C++ tests: text IR -> pipeline -> Executable -> run, checked
// against straightforward references.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstring>
#include <random>

#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/text.h"
#include "lokinfer/ir/verify.h"
#include "lokinfer/runtime/runtime.h"
#include "../../src/runtime/gpu_internal.h"

using namespace lk;
using namespace lk::rt;

namespace {

std::unique_ptr<Executable> compile(const std::string& src) {
  auto m = parseModel(src);
  INFO(m.status().message());
  REQUIRE(m.isOk());
  std::shared_ptr<Model> model(m.take());
  Status s = runPasses(*model, defaultPipeline());
  INFO(s.message());
  REQUIRE(s.isOk());
  auto e = Executable::create(model, {});
  INFO(e.status().message());
  REQUIRE(e.isOk());
  return e.take();
}

std::vector<float> randomVec(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

TensorView view(std::vector<float>& v, std::vector<int64_t> shape) { return TensorView{v.data(), DType::F32, std::move(shape)}; }

const float* data(const OwnedTensor& t) { return static_cast<const float*>(t.data()); }

std::string addGraph(const char* dev, const char* shape) {
  return std::string("model {\n init {\n return\n }\n exec(%x: tensor<f32[") + shape + "]>, %y: tensor<f32[" + shape +
         "]>) {\n %o = kernel \"add_f32_" + dev + "\"(%x, %y) : tensor<f32[" + shape + "]>\n return %o\n }\n}\n";
}

std::string addMulGraph(const char* dev, const char* shape) {
  std::string t = std::string("tensor<f32[") + shape + "]>";
  return "model {\n init {\n return\n }\n exec(%x: " + t + ", %y: " + t + ", %z: " + t + ") {\n" +
         " %a = kernel \"add_f32_" + dev + "\"(%x, %y) : " + t + "\n" + " %o = kernel \"mul_f32_" + dev +
         "\"(%a, %z) : " + t + "\n return %o\n }\n}\n";
}

void checkAdd(const char* dev, const char* shape, int64_t n) {
  auto e = compile(addGraph(dev, shape));
  auto x = randomVec(n, 1), y = randomVec(n, 2);
  TensorView in[] = {view(x, {n}), view(y, {n})};
  std::vector<OwnedTensor> out;
  Status s = e->run(in, out);
  INFO(s.message());
  REQUIRE(s.isOk());
  REQUIRE(out.size() == 1);
  REQUIRE(out[0].shape == std::vector<int64_t>{n});
  for (int64_t i = 0; i < n; ++i) REQUIRE(data(out[0])[i] == x[i] + y[i]);
}

void checkAddMul(const char* dev, const char* shape, int64_t n) {
  auto e = compile(addMulGraph(dev, shape));
  // Run twice to check per-call state (arena reuse, fresh outputs).
  for (uint32_t call = 0; call < 2; ++call) {
    auto x = randomVec(n, 10 + call), y = randomVec(n, 20 + call), z = randomVec(n, 30 + call);
    TensorView in[] = {view(x, {n}), view(y, {n}), view(z, {n})};
    std::vector<OwnedTensor> out;
    Status s = e->run(in, out);
    INFO(s.message());
    REQUIRE(s.isOk());
    for (int64_t i = 0; i < n; ++i) REQUIRE(data(out[0])[i] == (x[i] + y[i]) * z[i]);
  }
}

void checkGemm(const char* kernel, int64_t m, int64_t n, int64_t k) {
  std::string src = "model {\n init {\n return\n }\n exec(%a: tensor<f32[" + std::to_string(m) + ", " +
                    std::to_string(k) + "]>, %b: tensor<f32[" + std::to_string(k) + ", " + std::to_string(n) +
                    "]>) {\n %o = kernel \"" + kernel + "\"(%a, %b) : tensor<f32[" + std::to_string(m) + ", " +
                    std::to_string(n) + "]>\n return %o\n }\n}\n";
  auto e = compile(src);
  auto a = randomVec(m * k, 3), b = randomVec(k * n, 4);
  TensorView in[] = {view(a, {m, k}), view(b, {k, n})};
  std::vector<OwnedTensor> out;
  Status s = e->run(in, out);
  INFO(s.message());
  REQUIRE(s.isOk());
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      double ref = 0;
      for (int64_t p = 0; p < k; ++p) ref += double(a[i * k + p]) * b[p * n + j];
      INFO(kernel << " (" << i << ", " << j << ")");
      REQUIRE_THAT(data(out[0])[i * n + j], Catch::Matchers::WithinAbs(ref, 1e-4 * std::sqrt(double(k)) + 1e-5));
    }
}

}  // namespace

TEST_CASE("e2e: CPU static vector add", "[e2e][cpu]") { checkAdd("cpu", "1000", 1000); }
TEST_CASE("e2e: GPU static vector add", "[e2e][gpu]") { checkAdd("gpu", "1003", 1003); }
TEST_CASE("e2e: CPU dynamic vector add", "[e2e][cpu]") {
  for (int64_t n : {1, 7, 1000, 4097}) {
    auto e = compile(addGraph("cpu", "N"));
    auto x = randomVec(n, 5), y = randomVec(n, 6);
    TensorView in[] = {view(x, {n}), view(y, {n})};
    std::vector<OwnedTensor> out;
    REQUIRE(e->run(in, out).isOk());
    for (int64_t i = 0; i < n; ++i) REQUIRE(data(out[0])[i] == x[i] + y[i]);
  }
}
TEST_CASE("e2e: GPU dynamic vector add (one executable, many sizes)", "[e2e][gpu]") {
  auto e = compile(addGraph("gpu", "N"));
  for (int64_t n : {1, 7, 1000, 4097, 100000}) {
    auto x = randomVec(n, 5), y = randomVec(n, 6);
    TensorView in[] = {view(x, {n}), view(y, {n})};
    std::vector<OwnedTensor> out;
    Status s = e->run(in, out);
    INFO(s.message());
    REQUIRE(s.isOk());
    for (int64_t i = 0; i < n; ++i) REQUIRE(data(out[0])[i] == x[i] + y[i]);
  }
}
TEST_CASE("e2e: CPU static add+mul", "[e2e][cpu]") { checkAddMul("cpu", "513", 513); }
TEST_CASE("e2e: GPU static add+mul", "[e2e][gpu]") { checkAddMul("gpu", "513", 513); }
TEST_CASE("e2e: CPU dynamic add+mul", "[e2e][cpu]") { checkAddMul("cpu", "N", 2049); }
TEST_CASE("e2e: GPU dynamic add+mul", "[e2e][gpu]") { checkAddMul("gpu", "N", 2049); }
TEST_CASE("e2e: CPU static matmul (hand-written)", "[e2e][cpu]") {
  checkGemm("gemm_f32_cpu", 64, 48, 32);
  checkGemm("gemm_f32_cpu", 37, 53, 29);
}
TEST_CASE("e2e: CPU static matmul (Accelerate)", "[e2e][cpu]") { checkGemm("gemm_f32_cpu_accelerate", 37, 53, 29); }
TEST_CASE("e2e: GPU static matmul (hand-written)", "[e2e][gpu]") {
  checkGemm("gemm_f32_gpu", 64, 64, 64);
  checkGemm("gemm_f32_gpu", 37, 53, 29);
}

TEST_CASE("e2e: GPU static matmul (MPS)", "[e2e][gpu]") {
  checkGemm("gemm_f32_gpu_mps", 64, 64, 64);
  checkGemm("gemm_f32_gpu_mps", 37, 53, 29);
}
TEST_CASE("e2e: GPU MPS matmul chained with custom kernels", "[e2e][gpu]") {
  // Exercises argument-table rebinding after an MPS dispatch.
  auto e = compile(R"(model {
  init {
    return
  }
  exec(%a: tensor<f32[33, 20]>, %b: tensor<f32[20, 17]>, %c: tensor<f32[33, 17]>) {
    %p = kernel "gemm_f32_gpu_mps"(%a, %b) : tensor<f32[33, 17]>
    %o = kernel "add_f32_gpu"(%p, %c) : tensor<f32[33, 17]>
    return %o
  }
}
)");
  auto a = randomVec(33 * 20, 1), b = randomVec(20 * 17, 2), c = randomVec(33 * 17, 3);
  TensorView in[] = {view(a, {33, 20}), view(b, {20, 17}), view(c, {33, 17})};
  std::vector<OwnedTensor> out;
  REQUIRE(e->run(in, out).isOk());
  for (int i = 0; i < 33; ++i)
    for (int j = 0; j < 17; ++j) {
      double ref = c[i * 17 + j];
      for (int p = 0; p < 20; ++p) ref += double(a[i * 20 + p]) * b[p * 17 + j];
      REQUIRE_THAT(data(out[0])[i * 17 + j], Catch::Matchers::WithinAbs(ref, 1e-4));
    }
}

TEST_CASE("e2e: GPU barriers only where buffers conflict", "[e2e][gpu]") {
  // Two independent chains interleaved (no barrier needed between chains),
  // each step depending on the previous one (barrier needed), plus arena
  // reuse (WAR hazards on recycled offsets). Any missed barrier corrupts results.
  std::string t = "tensor<f32[65536]>";
  std::string src = "model {\n init {\n return\n }\n exec(%x: " + t + ", %y: " + t + ") {\n";
  std::string a = "%x", b = "%y";
  for (int i = 0; i < 24; ++i) {
    std::string na = "%a" + std::to_string(i), nb = "%b" + std::to_string(i);
    src += " " + na + " = kernel \"add_f32_gpu\"(" + a + ", %x) : " + t + "\n";
    src += " " + nb + " = kernel \"mul_f32_gpu\"(" + b + ", %y) : " + t + "\n";
    a = na;
    b = nb;
  }
  src += " %o = kernel \"add_f32_gpu\"(" + a + ", " + b + ") : " + t + "\n return %o\n }\n}\n";
  auto e = compile(src);
  int64_t n = 65536;
  std::vector<float> x(n), y(n);
  for (int64_t i = 0; i < n; ++i) {
    x[i] = float(i % 7) * 0.25f;
    y[i] = (i % 2) ? 1.0f : -1.0f;
  }
  TensorView in[] = {view(x, {n}), view(y, {n})};
  for (int rep = 0; rep < 3; ++rep) {
    std::vector<OwnedTensor> out;
    REQUIRE(e->run(in, out).isOk());
    for (int64_t i = 0; i < n; ++i) {
      float ra = x[i] * 25.0f, rb = y[i];
      for (int k = 0; k < 24; ++k) rb *= y[i];
      REQUIRE(data(out[0])[i] == ra + rb);
    }
  }
}

TEST_CASE("runtime: input shape checks", "[runtime]") {
  auto e = compile(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[N, N]>) {
    %o = kernel "add_f32_cpu"(%x, %x) : tensor<f32[N, N]>
    return %o
  }
}
)");
  std::vector<float> v(6);
  TensorView bad[] = {view(v, {2, 3})};
  std::vector<OwnedTensor> out;
  Status s = e->run(bad, out);
  REQUIRE_FALSE(s.isOk());
  CHECK(s.message().find("expected N = 2") != std::string::npos);
}

TEST_CASE("runtime: linear input dims bind their symbol (600*F)", "[runtime]") {
  auto e = compile(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[600*F + 1, 2]>, %y: tensor<f32[F]>) {
    %o = kernel "add_f32_cpu"(%y, %y) : tensor<f32[F]>
    return %o
  }
}
)");
  std::vector<float> x(1801 * 2), y(3, 1.5f);
  TensorView ok[] = {view(x, {1801, 2}), view(y, {3})};
  std::vector<OwnedTensor> out;
  Status s = e->run(ok, out);
  INFO(s.message());
  REQUIRE(s.isOk());
  CHECK(data(out[0])[2] == 3.0f);
  std::vector<float> y4(4);
  TensorView mismatch[] = {view(x, {1801, 2}), view(y4, {4})};
  CHECK_FALSE(e->run(mismatch, out).isOk());
  std::vector<float> x2(1800 * 2);
  TensorView notForm[] = {view(x2, {1800, 2}), view(y, {3})};
  Status s2 = e->run(notForm, out);
  REQUIRE_FALSE(s2.isOk());
  CHECK(s2.message().find("not of the form") != std::string::npos);
}

TEST_CASE("runtime: dynamic view out of bounds is an error", "[runtime]") {
  auto e = compile(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[N]>, %y: tensor<f32[M]>) {
    %v = view %x[0] : tensor<f32[M]>
    %o = kernel "add_f32_cpu"(%v, %y) : tensor<f32[M]>
    return %o
  }
}
)");
  std::vector<float> x(4), y(8);
  TensorView in[] = {view(x, {4}), view(y, {8})};
  std::vector<OwnedTensor> out;
  Status s = e->run(in, out);
  REQUIRE_FALSE(s.isOk());
  CHECK(s.message().find("view out of bounds") != std::string::npos);
}

TEST_CASE("runtime: returning an input or constant copies, never aliases", "[runtime]") {
  auto e = compile(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[4]>) {
    %c = const.inline dense<[1.0, 2.0, 3.0, 4.0]> : tensor<f32[4]>
    return %x, %c
  }
}
)");
  std::vector<float> x = {9, 8, 7, 6};
  TensorView in[] = {view(x, {4})};
  std::vector<OwnedTensor> out;
  REQUIRE(e->run(in, out).isOk());
  REQUIRE(out.size() == 2);
  CHECK(out[0].data() != static_cast<void*>(x.data()));
  CHECK(data(out[0])[0] == 9.0f);
  static_cast<float*>(out[1].data())[0] = 100.0f;  // mutate the returned constant
  std::vector<OwnedTensor> out2;
  REQUIRE(e->run(in, out2).isOk());
  CHECK(data(out2[1])[0] == 1.0f);  // model memory untouched
}

TEST_CASE("verify: init may not use exec input symbols", "[runtime]") {
  auto m = parseModel(R"(model {
  init {
    %a = buffer.alloc : buffer<f32[N]>
    return %a
  }
  exec(%x: buffer<f32[N]>) state(%s: buffer<f32[N]>) {
    return %x
  }
}
)");
  REQUIRE(m.isOk());
  Status s = verifyModel(**m);
  REQUIRE_FALSE(s.isOk());
  CHECK(s.message().find("unbound symbol N") != std::string::npos);
}

TEST_CASE("e2e: mixed CPU/GPU with explicit syncs (write-after-read safe)", "[e2e][gpu]") {
  // The reviewer's case: a CPU result read by a GPU kernel that isn't committed
  // until the sync, then another CPU kernel of the same size. With the
  // lifetime extension the planner can't give %b the bytes the GPU still reads.
  auto e = compile(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[1024]>, %y: tensor<f32[1024]>) {
    %a = kernel "add_f32_cpu"(%x, %y) : tensor<f32[1024]>
    %ag = sync cpu_to_gpu %a : tensor<f32[1024]>
    %o1 = kernel "mul_f32_gpu"(%ag, %ag) : tensor<f32[1024]>
    %b = kernel "mul_f32_cpu"(%x, %x) : tensor<f32[1024]>
    %o1c = sync gpu_to_cpu %o1 : tensor<f32[1024]>
    %o2 = kernel "add_f32_cpu"(%b, %o1c) : tensor<f32[1024]>
    return %o2
  }
}
)");
  auto x = randomVec(1024, 1), y = randomVec(1024, 2);
  TensorView in[] = {view(x, {1024}), view(y, {1024})};
  for (int rep = 0; rep < 3; ++rep) {
    std::vector<OwnedTensor> out;
    REQUIRE(e->run(in, out).isOk());
    for (int i = 0; i < 1024; ++i) {
      float a = x[i] + y[i];
      // 1e-6: the reference may be FMA-contracted; a write-after-read bug is off by O(1).
      REQUIRE_THAT(data(out[0])[i], Catch::Matchers::WithinAbs(x[i] * x[i] + a * a, 1e-6));
    }
  }
}

TEST_CASE("verifier: implicit crossing is rejected", "[runtime]") {
  auto m = parseModel(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[8]>) {
    %a = kernel "add_f32_gpu"(%x, %x) : tensor<f32[8]>
    %b = kernel "add_f32_cpu"(%a, %a) : tensor<f32[8]>
    return %b
  }
}
)");
  REQUIRE(m.isOk());
  Status s = runPasses(**m, defaultPipeline());
  REQUIRE_FALSE(s.isOk());
  CHECK(s.message().find("add ops.to_cpu") != std::string::npos);
}

// ---- thread pool
#include <atomic>
#include <vector>

#include "lokinfer/runtime/thread_pool.h"

TEST_CASE("thread pool runs every index exactly once") {
  auto& pool = lk::rt::ThreadPool::global();
  REQUIRE(pool.threads() >= 1);
  for (int64_t n : {0, 1, 2, 7, 64, 1000}) {
    std::vector<std::atomic<int>> hits(static_cast<size_t>(n));
    for (int rep = 0; rep < 50; ++rep) {
      pool.parallelFor(n, [&](int64_t i) { hits[static_cast<size_t>(i)].fetch_add(1); });
    }
    for (auto& h : hits) REQUIRE(h.load() == 50);
  }
  REQUIRE(lk::rt::performanceCores() >= 1);
}

// ---- GPU record/replay (indirect command buffer) ----------------------------------------------------------------

namespace {

std::unique_ptr<Executable> compileWith(const std::string& src, const RuntimeOptions& opts) {
  auto m = parseModel(src);
  REQUIRE(m.isOk());
  std::shared_ptr<Model> model(m.take());
  REQUIRE(runPasses(*model, defaultPipeline()).isOk());
  auto e = Executable::create(model, {}, opts);
  INFO(e.status().message());
  REQUIRE(e.isOk());
  return e.take();
}

// The barrier stress graph: two interleaved dependent chains + arena reuse (any missing barrier corrupts results).
std::string chainsGraph(int steps, int64_t n) {
  std::string t = "tensor<f32[" + std::to_string(n) + "]>";
  std::string src = "model {\n init {\n return\n }\n exec(%x: " + t + ", %y: " + t + ") {\n";
  std::string a = "%x", b = "%y";
  for (int i = 0; i < steps; ++i) {
    std::string na = "%a" + std::to_string(i), nb = "%b" + std::to_string(i);
    src += " " + na + " = kernel \"add_f32_gpu\"(" + a + ", %x) : " + t + "\n";
    src += " " + nb + " = kernel \"mul_f32_gpu\"(" + b + ", %y) : " + t + "\n";
    a = na;
    b = nb;
  }
  return src + " %o = kernel \"add_f32_gpu\"(" + a + ", " + b + ") : " + t + "\n return %o\n }\n}\n";
}

}  // namespace

TEST_CASE("gpu replay: same results as direct encode, inputs and outputs moving between calls", "[e2e][gpu][replay]") {
  const int64_t n = 65536;
  auto direct = compileWith(chainsGraph(24, n), RuntimeOptions{true, 0});
  auto replay = compileWith(chainsGraph(24, n), RuntimeOptions{true, 1});
  REQUIRE(direct->gpuStrategy() == "direct");
  REQUIRE(replay->gpuStrategy() == "replay (records on the next call)");
  std::vector<OwnedTensor> keep;  // keep some outputs alive so the pool hands out different storage
  for (int call = 0; call < 8; ++call) {
    // Owned inputs (wrapped in place, a new wrap whenever the array changes) on even calls, ownerless raw memory
    // (copied into pool storage) on odd calls.
    auto x = std::make_shared<std::vector<float>>(randomVec(n, 100 + call));
    auto y = std::make_shared<std::vector<float>>(n);
    for (int64_t i = 0; i < n; ++i) (*y)[i] = (i % 3) ? 0.5f : -1.0f + 0.01f * call;
    TensorView in[] = {view(*x, {n}), view(*y, {n})};
    if (call % 2 == 0) in[0].owner = x, in[1].owner = y;
    std::vector<OwnedTensor> od, orp;
    REQUIRE(direct->run(in, od).isOk());
    Status s = replay->run(in, orp);
    INFO("call " << call << ": " << s.message());
    REQUIRE(s.isOk());
    if (call == 0) REQUIRE(replay->gpuStrategy().rfind("replay (", 0) == 0);
    if (call >= 1) REQUIRE(replay->gpuStrategy() == "replay (49 commands)");
    REQUIRE(std::memcmp(od[0].data(), orp[0].data(), n * sizeof(float)) == 0);
    if (call % 3 == 0) keep.push_back(orp[0]);
  }
}

TEST_CASE("gpu replay: static matmul, inline constants, repeated inputs", "[e2e][gpu][replay]") {
  std::string src = R"(model {
  init {
    return
  }
  exec(%a: tensor<f32[37, 29]>, %b: tensor<f32[29, 53]>) {
    %c = const.inline dense<[1.0, 2.0, 3.0, 4.0]> : tensor<f32[4]>
    %m = kernel "gemm_f32_gpu"(%a, %b) : tensor<f32[37, 53]>
    %t = kernel "transpose_f32_gpu"(%m) : tensor<f32[53, 37]>
    %u = kernel "add_f32_gpu"(%c, %c) : tensor<f32[4]>
    return %t, %u
  }
}
)";
  auto direct = compileWith(src, RuntimeOptions{true, 0});
  auto replay = compileWith(src, RuntimeOptions{true, 1});
  auto a = randomVec(37 * 29, 5), b = randomVec(29 * 53, 6);
  TensorView in[] = {view(a, {37, 29}), view(b, {29, 53})};
  for (int call = 0; call < 4; ++call) {
    std::vector<OwnedTensor> od, orp;
    REQUIRE(direct->run(in, od).isOk());
    REQUIRE(replay->run(in, orp).isOk());
    REQUIRE(std::memcmp(od[0].data(), orp[0].data(), 37 * 53 * 4) == 0);
    for (int i = 0; i < 4; ++i) REQUIRE(data(orp[1])[i] == 2.f * float(i + 1));
  }
  REQUIRE(replay->gpuStrategy() == "replay (3 commands)");
}

TEST_CASE("gpu replay: falls back to direct encode when it can't reproduce the call", "[e2e][gpu][replay]") {
  // MPS encodes its own work: the recording call aborts the strategy (and still computes the right result).
  std::string mps = R"(model {
  init {
    return
  }
  exec(%a: tensor<f32[8, 8]>, %b: tensor<f32[8, 8]>) {
    %m = kernel "gemm_f32_gpu_mps"(%a, %b) : tensor<f32[8, 8]>
    %o = kernel "add_f32_gpu"(%m, %m) : tensor<f32[8, 8]>
    return %o
  }
}
)";
  auto e = compileWith(mps, RuntimeOptions{true, 1});
  auto a = randomVec(64, 1), b = randomVec(64, 2);
  TensorView in[] = {view(a, {8, 8}), view(b, {8, 8})};
  for (int call = 0; call < 2; ++call) {
    std::vector<OwnedTensor> out;
    REQUIRE(e->run(in, out).isOk());
    double ref = 0;
    for (int p = 0; p < 8; ++p) ref += double(a[p]) * b[p * 8];
    REQUIRE_THAT(data(out[0])[0], Catch::Matchers::WithinAbs(2 * ref, 1e-4));
  }
  REQUIRE(e->gpuStrategy() == "direct (replay: a kernel encoded its own GPU work (MPS))");
  // CPU work between GPU kernels (two segments) is not eligible.
  std::string mixed = R"(model {
  init {
    return
  }
  exec(%a: tensor<f32[16]>) {
    %g = kernel "add_f32_gpu"(%a, %a) : tensor<f32[16]>
    %c = sync gpu_to_cpu %g : tensor<f32[16]>
    %h = kernel "add_f32_cpu"(%c, %c) : tensor<f32[16]>
    %i = sync cpu_to_gpu %h : tensor<f32[16]>
    %o = kernel "mul_f32_gpu"(%i, %i) : tensor<f32[16]>
    return %o
  }
}
)";
  REQUIRE(compileWith(mixed, RuntimeOptions{true, 1})->gpuStrategy() == "direct (replay: gpu_to_cpu sync between GPU kernels)");
}

TEST_CASE("gpu replay: symbolic shapes record one command list per input-shape signature", "[e2e][gpu][replay]") {
  auto e = compileWith(addMulGraph("gpu", "N"), RuntimeOptions{true, 1});
  // Sizes cycle through 5 signatures (cache holds 4): records, replays and re-records must all be exact.
  const int64_t sizes[] = {1000, 4097, 1000, 33, 4097, 70000, 9, 1000, 33, 1000};
  for (int call = 0; call < 10; ++call) {
    const int64_t n = sizes[call];
    auto x = randomVec(n, 10 + call), y = randomVec(n, 20 + call), z = randomVec(n, 30 + call);
    TensorView in[] = {view(x, {n}), view(y, {n}), view(z, {n})};
    std::vector<OwnedTensor> out;
    Status s = e->run(in, out);
    INFO("call " << call << " n " << n << ": " << s.message());
    REQUIRE(s.isOk());
    for (int64_t i = 0; i < n; ++i) REQUIRE(data(out[0])[i] == (x[i] + y[i]) * z[i]);
  }
  REQUIRE(e->gpuStrategy() == "replay (2 commands)");
}

TEST_CASE("gpu: several segments in flight keep cross-segment dependencies", "[e2e][gpu][inflight]") {
  // GPU work, then an independent CPU kernel (the executor commits the GPU segment without waiting), then GPU work
  // that reads the first segment's results: with 2-3 segments in flight the later segment is encoded while the
  // earlier runs, and only the queue barrier orders them. Large tensors so a missing barrier would show.
  const int64_t n = 1 << 22;
  std::string t = "tensor<f32[" + std::to_string(n) + "]>";
  std::string src = "model {\n init {\n return\n }\n exec(%x: " + t + ", %y: " + t + ") {\n";
  std::string a = "%x", c = "%y";
  for (int i = 0; i < 6; ++i) {
    std::string s = std::to_string(i);
    src += " %g" + s + " = kernel \"add_f32_gpu\"(" + a + ", %x) : " + t + "\n";      // GPU chain
    src += " %h" + s + " = kernel \"mul_f32_gpu\"(%g" + s + ", %g" + s + ") : " + t + "\n";
    src += " %c" + s + " = kernel \"add_f32_cpu\"(" + c + ", %y) : " + t + "\n";      // independent CPU chain
    src += " %d" + s + " = sync cpu_to_gpu %c" + s + " : " + t + "\n";
    src += " %a" + s + " = kernel \"add_f32_gpu\"(%h" + s + ", %d" + s + ") : " + t + "\n";
    a = "%a" + s;
    c = "%c" + s;
  }
  src += " return " + a + "\n }\n}\n";
  auto x = randomVec(n, 1), y = randomVec(n, 2);
  // Reference in the same f32 order.
  std::vector<float> ref(n);
  for (int64_t i = 0; i < n; ++i) {
    float av = x[i], cv = y[i];
    for (int k = 0; k < 6; ++k) {
      float g = av + x[i];
      cv = cv + y[i];
      const float h = g * g;  // separate statements: no FMA contraction (the GPU runs two kernels)
      av = h + cv;
    }
    ref[i] = av;
  }
  auto gpu = GpuContext::shared();
  REQUIRE(gpu.isOk());
  const size_t before = (*gpu)->maxSegments();
  for (size_t segs : {1, 2, 3}) {
    (*gpu)->setMaxSegments(segs);
    auto e = compile(src);
    TensorView in[] = {view(x, {n}), view(y, {n})};
    for (int rep = 0; rep < 3; ++rep) {
      std::vector<OwnedTensor> out;
      REQUIRE(e->run(in, out).isOk());
      INFO("segments " << segs << " rep " << rep);
      REQUIRE(std::memcmp(out[0].data(), ref.data(), n * sizeof(float)) == 0);
    }
  }
  (*gpu)->setMaxSegments(before);
}
