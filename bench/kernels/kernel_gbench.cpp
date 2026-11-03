// Kernel harness, CPU (Google Benchmark): every CPU kernel at its real model shapes, called directly through the
// kernel registry (KernelEntry::prepare / run) — no Executable, no interpreter (docs/benchmarking.md "Kernel harness").
//
// Cases come from `tools/kbench gbench-export DIR` (one `caseNNNN.mlir` per real-shape case: a one-kernel buffer-form
// model, plus `caseNNNN.argI.bin` raw operand bytes for operands whose values matter — ints, transform tables, small
// vectors; the other float operands are uniform [-1, 1]). Usage:
//
//   build-release/lokinfer_bench_kernels DIR [--filter REGEX] [--part I/N] [google benchmark flags]
//   e.g. ... --benchmark_min_warmup_time=1 --benchmark_min_time=2s --benchmark_repetitions=1 \
//            --benchmark_out=gb.json --benchmark_out_format=json
//   tools/kbench gbench-import gb.json --dir DIR      # -> perf/kernels.sqlite (harness "gbench")
//
// Benchmark names are `<kernel>/<caseNNNN.mlir>`.
#include <benchmark/benchmark.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "lokinfer/ir/text.h"
#include "lokinfer/runtime/kernel.h"

using namespace lk;
namespace fs = std::filesystem;

namespace {

struct AlignedFree {
  void operator()(void* p) const { std::free(p); }
};

struct Case {
  std::string file, kernel;
  std::shared_ptr<Model> model;
  const Op* op = nullptr;
  const rt::KernelEntry* entry = nullptr;
  std::vector<std::unique_ptr<void, AlignedFree>> mem;
  std::vector<int64_t> scalars;  // 8-byte host slots for scalar operands
  std::vector<void*> ptrs;
  std::vector<rt::BufRef> bufs;
  std::vector<int64_t> ints;
  std::vector<double> floats;
  void* state = nullptr;
};

std::string readFile(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// Builds the kernel ABI arguments exactly like the runtime (src/runtime/executable.cpp, lower()).
bool setup(Case& c, const fs::path& dir) {
  auto parsed = parseModel(readFile(dir / c.file));
  if (!parsed) {
    std::fprintf(stderr, "%s: %s\n", c.file.c_str(), parsed.status().message().c_str());
    return false;
  }
  c.model = std::shared_ptr<Model>(parsed.take());
  const Model& m = *c.model;
  for (const Op& op : m.exec.ops)
    if (op.kind == OpKind::Kernel) {
      c.op = &op;
      break;
    }
  if (!c.op) return false;
  c.kernel = c.op->name;
  c.entry = rt::findKernel(c.kernel);
  if (!c.entry || c.entry->device != rt::Device::Cpu) return false;
  std::vector<ValueId> args = c.op->operands;
  args.insert(args.end(), c.op->outs.begin(), c.op->outs.end());
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  c.scalars.resize(args.size());
  const std::string stem = fs::path(c.file).stem().string();
  for (size_t i = 0; i < args.size(); ++i) {
    Type t = m.typeOf(args[i]);
    int64_t n = 1;
    for (Dim d : t->shape) {
      if (!d->isConst()) return false;
      n *= d->value;
      c.ints.push_back(d->value);
    }
    const size_t bytes = static_cast<size_t>(n) * dtypeSize(t->dtype);
    const fs::path bin = dir / (stem + ".arg" + std::to_string(i) + ".bin");
    if (t->isScalar()) {
      c.scalars[i] = 0;
      if (fs::exists(bin)) {
        std::string b = readFile(bin);
        std::memcpy(&c.scalars[i], b.data(), std::min<size_t>(8, b.size()));
      }
      c.ptrs.push_back(&c.scalars[i]);
      c.bufs.push_back({nullptr, 0});
      continue;
    }
    void* p = nullptr;
    if (posix_memalign(&p, 16384, std::max<size_t>(bytes, 64)) != 0) return false;
    c.mem.emplace_back(p);
    if (fs::exists(bin)) {
      std::string b = readFile(bin);
      std::memcpy(p, b.data(), std::min(bytes, b.size()));
    } else if (t->dtype == DType::F32) {
      float* f = static_cast<float*>(p);
      for (int64_t k = 0; k < n; ++k) f[k] = u(rng);
    } else {
      std::memset(p, 0, bytes);
    }
    c.ptrs.push_back(p);
    c.bufs.push_back({nullptr, 0});
  }
  for (const NamedAttr& a : c.op->attrs) {
    if (isStructuralAttr(a.name)) continue;
    const auto& val = a.attr.value;
    if (auto* b = std::get_if<bool>(&val)) c.ints.push_back(*b);
    else if (auto* i = std::get_if<int64_t>(&val)) c.ints.push_back(*i);
    else if (auto* f = std::get_if<double>(&val)) c.floats.push_back(*f);
    else if (auto* is = std::get_if<Attr::Ints>(&val)) c.ints.insert(c.ints.end(), is->begin(), is->end());
    else if (auto* fs_ = std::get_if<Attr::Floats>(&val)) c.floats.insert(c.floats.end(), fs_->begin(), fs_->end());
    else if (auto* d = std::get_if<Dim>(&val)) {
      if (!(*d)->isConst()) return false;
      c.ints.push_back((*d)->value);
    }
  }
  if (c.entry->prepare) {
    rt::PrepareCtx pc{c.model.get(), nullptr};
    if (!c.entry->prepare(pc, *c.op, &c.state)) {
      std::fprintf(stderr, "%s: prepare failed\n", c.file.c_str());
      return false;
    }
  }
  return true;
}

void runCase(benchmark::State& st, Case* c) {
  rt::KernelArgs ka{c->ptrs.data(), c->bufs.data(), c->ints.data(), c->floats.data(), c->state,
                    static_cast<uint32_t>(c->ptrs.size()), static_cast<uint32_t>(c->ints.size())};
  rt::ExecCtx ctx;
  for (auto _ : st) {
    c->entry->run(ctx, ka);
    if (ctx.error) {
      st.SkipWithError(ctx.error);
      break;
    }
    benchmark::ClobberMemory();
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '-') {
    std::fprintf(stderr, "usage: %s CASES_DIR [--filter REGEX] [benchmark flags]\n", argv[0]);
    return 2;
  }
  const fs::path dir = argv[1];
  std::string filter;
  int partI = 1, partN = 1;  // --part I/N: the I-th of N slices of the case files (short locked sessions)
  std::vector<char*> rest = {argv[0]};
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) filter = argv[++i];
    else if (std::strcmp(argv[i], "--part") == 0 && i + 1 < argc) std::sscanf(argv[++i], "%d/%d", &partI, &partN);
    else rest.push_back(argv[i]);
  }
  rt::registerBuiltinKernels();
  std::vector<std::string> files;
  for (auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".mlir") files.push_back(e.path().filename().string());
  std::sort(files.begin(), files.end());
  if (partN > 1) {
    const size_t n = files.size(), b = n * (partI - 1) / partN, e = n * partI / partN;
    files = std::vector<std::string>(files.begin() + b, files.begin() + e);
  }
  static std::vector<std::unique_ptr<Case>> cases;
  const std::regex rx(filter.empty() ? ".*" : filter);
  for (auto& f : files) {
    auto c = std::make_unique<Case>();
    c->file = f;
    if (!setup(*c, dir)) {
      std::fprintf(stderr, "skipping %s\n", f.c_str());
      continue;
    }
    if (!std::regex_search(c->kernel, rx)) continue;
    benchmark::RegisterBenchmark((c->kernel + "/" + f).c_str(), runCase, c.get())->UseRealTime()->Unit(
        benchmark::kMicrosecond);
    cases.push_back(std::move(c));
  }
  int n = static_cast<int>(rest.size());
  benchmark::Initialize(&n, rest.data());
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  for (auto& c : cases)
    if (c->state && c->entry->destroy) c->entry->destroy(c->state);
  return 0;
}
