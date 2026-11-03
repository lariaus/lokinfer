// Runtime overhead: a one-kernel graph through Executable::run vs calling the
// typed kernel directly (design goal: graph overhead ~ 0, so correctness tests
// can go through graphs). CPU and GPU.
#include <benchmark/benchmark.h>

#include <vector>

#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/text.h"
#include "lokinfer/kernels/cpu.h"
#include "lokinfer/runtime/runtime.h"

using namespace lk;
using namespace lk::rt;

namespace {

std::unique_ptr<Executable> compileAdd(const char* dev, int64_t n) {
  std::string t = "tensor<f32[" + std::to_string(n) + "]>";
  std::string src = "model {\n init {\n return\n }\n exec(%x: " + t + ", %y: " + t + ") {\n %o = kernel \"add_f32_" +
                    dev + "\"(%x, %y) : " + t + "\n return %o\n }\n}\n";
  std::shared_ptr<Model> m(parseModel(src).take());
  if (!runPasses(*m, defaultPipeline())) std::abort();
  auto e = Executable::create(m, {});
  if (!e) std::abort();
  return e.take();
}

void BM_DirectAddCpu(benchmark::State& st) {
  int64_t n = st.range(0);
  std::vector<float> x(n, 1.f), y(n, 2.f), o(n);
  for (auto _ : st) {
    kernels::cpu::add_f32(x.data(), y.data(), o.data(), n);
    benchmark::DoNotOptimize(o.data());
  }
}

void BM_GraphAddCpu(benchmark::State& st) {
  int64_t n = st.range(0);
  auto e = compileAdd("cpu", n);
  std::vector<float> x(n, 1.f), y(n, 2.f);
  TensorView in[] = {{x.data(), DType::F32, {n}}, {y.data(), DType::F32, {n}}};
  std::vector<OwnedTensor> out;
  for (auto _ : st) {
    if (!e->run(in, out)) std::abort();
    benchmark::DoNotOptimize(out[0].data());
  }
}

void BM_GraphAddGpu(benchmark::State& st) {
  int64_t n = st.range(0);
  auto e = compileAdd("gpu", n);
  std::vector<float> x(n, 1.f), y(n, 2.f);
  TensorView in[] = {{x.data(), DType::F32, {n}}, {y.data(), DType::F32, {n}}};
  std::vector<OwnedTensor> out;
  for (auto _ : st) {
    if (!e->run(in, out)) std::abort();
    benchmark::DoNotOptimize(out[0].data());
  }
}

}  // namespace

BENCHMARK(BM_DirectAddCpu)->Arg(1024)->Arg(1 << 20);
BENCHMARK(BM_GraphAddCpu)->Arg(1024)->Arg(1 << 20);
BENCHMARK(BM_GraphAddGpu)->Arg(1024)->Arg(1 << 20);
BENCHMARK_MAIN();
