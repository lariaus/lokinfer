#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <cstring>
#include <unistd.h>

#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/binary.h"
#include "lokinfer/ir/text.h"
#include "lokinfer/runtime/runtime.h"

using namespace lk;

namespace {

const char* kAllOps = R"(model {
  init {
    %w = const.internal @"w" : buffer<f32[2, 3]>
    %wh = buffer.alloc : buffer<f32[2, 3]>
    kernel "copy_f32_cpu"(%w) outs(%wh)
    return %wh
  }
  exec(%ids: tensor<i64[1, T]>, %s: f32) state(%w2: buffer<f32[2, 3]>) {
    %e = const.external @"ext0" : tensor<f32[4]>
    %c = const.inline dense<[1, -2, 3]> : tensor<i64[3]>
    %t, %u = kernel "split_f32_cpu"(%e) {axis = 0, eps = 1.0e-05, flags = [1, -2], mode = "x", scales = f[0.5, 2.0], sz = dim<2*T + 1>, on = true} : tensor<f32[2]>, tensor<f32[2]>
    %a = mem.plan sizes[4*T, 64, max(T, 3)] live[0, 3, 1, 5, 2, 4] align 256 -> P, O0, O1, O2 : buffer<u8[P]>
    %v = view %a[O0 // 4] : buffer<f32[T]>
    dim.read %v[0] -> F
    sync gpu_to_cpu
    %g = buffer.alloc : buffer<f32[F*600 + ceildiv(T, 2) - F % 3]>
    buffer.free %v
    return %g, %t
  }
}
)";

std::unique_ptr<Model> parseOk(const char* s) {
  auto m = parseModel(s);
  INFO(m.status().message());
  REQUIRE(m.isOk());
  return m.take();
}

std::vector<float> kW = {1, 2, 3, 4, 5, 6};

void attachW(Model& m) {
  for (auto& c : m.constants)
    if (c.name == "w") {
      c.bytes = reinterpret_cast<const uint8_t*>(kW.data());
      c.size = kW.size() * 4;
    }
}

// Drops the external constant so the model is exportable.
std::string withoutExternal(std::string s) {
  auto a = s.find("    %e = const.external");
  s.erase(a, s.find('\n', a) - a + 1);
  auto b = s.find("kernel \"split_f32_cpu\"(%e)");
  s.replace(b, std::strlen("kernel \"split_f32_cpu\"(%e)"), "kernel \"split_f32_cpu\"(%c)");
  return s;
}

}  // namespace

TEST_CASE("binary: text -> binary -> text is identical", "[binary]") {
  std::string src = withoutExternal(kAllOps);
  auto m = parseOk(src.c_str());
  attachW(*m);
  auto bytes = writeModelBytes(*m);
  INFO(bytes.status().message());
  REQUIRE(bytes.isOk());
  auto shared = std::make_shared<const std::vector<uint8_t>>(bytes.take());
  auto m2 = readModelBytes(shared);
  INFO(m2.status().message());
  REQUIRE(m2.isOk());
  CHECK(printModel(**m2) == printModel(*m));
  // Constant data survives, 16 KiB-aligned inside the blob.
  const ConstantData* c = (*m2)->findConstant("w");
  REQUIRE(c);
  REQUIRE(c->size == 24);
  CHECK(std::memcmp(c->bytes, kW.data(), 24) == 0);
  CHECK((c->bytes - shared->data()) % 16384 == 0);
}

TEST_CASE("binary: compiled pipeline output round trips", "[binary]") {
  auto m = parseOk(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[N, 64]>) {
    %w = const.internal @"w" : tensor<f32[64, 64]>
    %a = kernel "gemm_f32_cpu"(%x, %w) : tensor<f32[N, 64]>
    %b = kernel "add_f32_cpu"(%a, %a) : tensor<f32[N, 64]>
    return %b
  }
}
)");
  std::vector<float> w(64 * 64, 0.5f);
  m->constants[0].bytes = reinterpret_cast<const uint8_t*>(w.data());
  m->constants[0].size = w.size() * 4;
  REQUIRE(runPasses(*m, defaultPipeline()).isOk());
  auto bytes = writeModelBytes(*m);
  REQUIRE(bytes.isOk());
  auto m2 = readModelBytes(std::make_shared<const std::vector<uint8_t>>(bytes.take()));
  REQUIRE(m2.isOk());
  CHECK(printModel(**m2) == printModel(*m));
}

TEST_CASE("binary: file round trip + run from mmapped weights", "[binary]") {
  auto m = parseOk(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[2, 3]>) {
    %w = const.internal @"w" : tensor<f32[2, 3]>
    %o = kernel "add_f32_cpu"(%x, %w) : tensor<f32[2, 3]>
    return %o
  }
}
)");
  attachW(*m);
  REQUIRE(runPasses(*m, defaultPipeline()).isOk());
  // temp_directory_path() is the app's sandboxed tmp dir on iOS.
  std::string tmpl = (std::filesystem::temp_directory_path() / "lokinfer_test_XXXXXX").string();
  std::vector<char> path(tmpl.begin(), tmpl.end());
  path.push_back('\0');
  int fd = mkstemp(path.data());
  REQUIRE(fd >= 0);
  close(fd);
  REQUIRE(writeModelFile(*m, path.data()).isOk());
  auto loaded = readModelFile(path.data());
  std::remove(path.data());  // the mapping stays valid after unlink
  INFO(loaded.status().message());
  REQUIRE(loaded.isOk());
  std::shared_ptr<Model> model(loaded.take());
  auto e = rt::Executable::create(model, {});
  REQUIRE(e.isOk());
  std::vector<float> x = {10, 20, 30, 40, 50, 60};
  rt::TensorView in[] = {{x.data(), DType::F32, {2, 3}}};
  std::vector<rt::OwnedTensor> out;
  REQUIRE((*e)->run(in, out).isOk());
  for (int i = 0; i < 6; ++i) CHECK(static_cast<const float*>(out[0].data())[i] == x[i] + kW[i]);
}

TEST_CASE("binary: external constants must be internalized first", "[binary]") {
  auto m = parseOk(kAllOps);
  attachW(*m);
  auto r = writeModelBytes(*m);
  REQUIRE_FALSE(r.isOk());
  CHECK(r.status().message().find("internalized") != std::string::npos);
}

TEST_CASE("binary: corrupt input is rejected, never crashes", "[binary]") {
  std::string src = withoutExternal(kAllOps);
  auto m = parseOk(src.c_str());
  attachW(*m);
  auto bytes = writeModelBytes(*m).take();
  // Truncations and bit flips across the header + metadata.
  for (size_t cut : {size_t(0), size_t(10), size_t(64), size_t(100), bytes.size() / 2}) {
    auto b = std::make_shared<const std::vector<uint8_t>>(bytes.begin(), bytes.begin() + std::min(cut, bytes.size()));
    CHECK_FALSE(readModelBytes(b).isOk());
  }
  for (size_t i = 0; i < 64 + 400 && i < bytes.size(); i += 7) {
    auto copy = bytes;
    copy[i] ^= 0x5a;
    (void)readModelBytes(std::make_shared<const std::vector<uint8_t>>(std::move(copy)));  // must not crash
  }
}

TEST_CASE("binary: hostile headers and enums are rejected", "[binary]") {
  std::string src = withoutExternal(kAllOps);
  auto m = parseOk(src.c_str());
  attachW(*m);
  auto bytes = writeModelBytes(*m).take();
  auto load = [](std::vector<uint8_t> b) { return readModelBytes(std::make_shared<const std::vector<uint8_t>>(std::move(b))); };
  {  // metaOffset + metaSize wraps around 2^64
    auto b = bytes;
    uint64_t off = ~uint64_t(0) - 100, sz = 200;
    std::memcpy(b.data() + 8, &off, 8);
    std::memcpy(b.data() + 16, &sz, 8);
    CHECK_FALSE(load(b).isOk());
  }
  {  // dataOffset + dataSize wraps
    auto b = bytes;
    uint64_t off = ~uint64_t(0) - 10, sz = 20;
    std::memcpy(b.data() + 24, &off, 8);
    std::memcpy(b.data() + 32, &sz, 8);
    CHECK_FALSE(load(b).isOk());
  }
  // Every single-byte value at every metadata position: never crashes, and
  // whatever loads must pass the verifier or have been rejected.
  for (size_t i = 64; i < std::min<size_t>(bytes.size(), 64 + 600); ++i) {
    for (int v : {0x00, 0x7f, 0xff}) {
      auto b = bytes;
      b[i] = static_cast<uint8_t>(v);
      (void)load(std::move(b));
    }
  }
}

TEST_CASE("binary: GPU model runs straight from mmapped weights", "[binary][gpu]") {
  auto m = parseOk(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[2, 3]>) {
    %w = const.internal @"w" {device = "gpu"} : tensor<f32[2, 3]>
    %o = kernel "add_f32_gpu"(%x, %w) : tensor<f32[2, 3]>
    return %o
  }
}
)");
  attachW(*m);
  REQUIRE(runPasses(*m, defaultPipeline()).isOk());
  std::string tmpl = (std::filesystem::temp_directory_path() / "lokinfer_gpu_XXXXXX").string();
  std::vector<char> path(tmpl.begin(), tmpl.end());
  path.push_back('\0');
  int fd = mkstemp(path.data());
  REQUIRE(fd >= 0);
  close(fd);
  REQUIRE(writeModelFile(*m, path.data()).isOk());
  auto loaded = readModelFile(path.data());
  std::remove(path.data());
  REQUIRE(loaded.isOk());
  std::shared_ptr<Model> model(loaded.take());
  auto e = rt::Executable::create(model, {});
  REQUIRE(e.isOk());
  std::vector<float> x = {10, 20, 30, 40, 50, 60};
  rt::TensorView in[] = {{x.data(), DType::F32, {2, 3}}};
  std::vector<rt::OwnedTensor> out;
  for (int rep = 0; rep < 2; ++rep) {
    REQUIRE((*e)->run(in, out).isOk());
    for (int i = 0; i < 6; ++i) CHECK(static_cast<const float*>(out[0].data())[i] == x[i] + kW[i]);
  }
}
