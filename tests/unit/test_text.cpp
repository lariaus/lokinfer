#include <catch2/catch_test_macros.hpp>

#include "lokinfer/ir/text.h"

using namespace lk;

namespace {

std::unique_ptr<Model> parseOk(std::string_view s) {
  auto r = parseModel(s);
  INFO(r.status().message());
  REQUIRE(r.isOk());
  return r.take();
}

void checkRoundTrip(std::string_view s) {
  auto m = parseOk(s);
  std::string p1 = printModel(*m);
  auto m2 = parseOk(p1);
  std::string p2 = printModel(*m2);
  INFO(p1);
  CHECK(p1 == p2);
}

}  // namespace

TEST_CASE("text: minimal model round trip", "[text]") {
  const char* src = R"(model {
  init {
    return
  }
  exec(%x: buffer<f32[N]>, %y: buffer<f32[N]>) -> buffer<f32[N]> {
    %o = buffer.alloc {output = true} : buffer<f32[N]>
    kernel "add_f32_cpu"(%x, %y) outs(%o)
    return %o
  }
}
)";
  auto m = parseOk(src);
  CHECK(printModel(*m) == src);
}

TEST_CASE("text: region result types are checked against the return", "[text]") {
  auto ok = parseModel("model { init -> (f32, i64) { %a = const.inline dense<[1.0]> : f32\n %b = const.inline "
                       "dense<[2]> : i64\n return %a, %b } exec() state(%s: f32, %t: i64) -> f32 { return %s } }");
  INFO(ok.status().message());
  CHECK(ok.isOk());
  auto bad = parseModel("model { init { return } exec(%x: f32) -> i64 { return %x } }");
  REQUIRE_FALSE(bad.isOk());
  CHECK(bad.status().message().find("exec result #0 is f32, header declares i64") != std::string::npos);
  auto count = parseModel("model { init { return } exec(%x: f32) -> (f32, f32) { return %x } }");
  REQUIRE_FALSE(count.isOk());
  CHECK(count.status().message().find("declares 2 result types, return has 1") != std::string::npos);
}

TEST_CASE("text: every op kind round trips", "[text]") {
  checkRoundTrip(R"(
// comment
model {
  init {
    %w = const.internal @"bert.w" : buffer<f16[178, 128]>
    %wh = buffer.alloc : buffer<f32[178, 128]>
    kernel "cast_f16_f32_cpu"(%w) outs(%wh)
    return %wh
  }
  exec(%ids: tensor<i64[1, T]>, %s: f32) state(%w2: buffer<f32[178, 128]>) {
    %e = const.external @"ext0" : tensor<f32[4]>
    %c = const.inline dense<[1, -2, 3]> : tensor<i64[3]>
    %f = const.inline dense<[0.5, 1.0e-07, -3.25, inf, nan]> : tensor<f32[5]>
    %h = const.inline dense<[0.333251953]> : tensor<f16[1]>
    %t, %u = kernel "split_f32_cpu"(%e) {axis = 0, eps = 1.0e-05, flags = [1, 2], mode = "x", scales = f[0.5, 2.0], sz = dim<2*T + 1>, on = true} : tensor<f32[2]>, tensor<f32[2]>
    %a = mem.plan sizes[4*T, 64, max(T, 3)] live[0, 3, 1, 5, 2, 4] align 256 -> P, O0, O1, O2 : buffer<u8[P]>
    %v = view %a[O0 // 4] : buffer<f32[T]>
    dim.read %v[0] -> F
    sync gpu_to_cpu
    sync cpu_to_gpu
    %g = buffer.alloc : buffer<f32[F*600]>
    buffer.free %v
    return %g, %t
  }
}
)");
}

TEST_CASE("text: inline constant values are exact", "[text]") {
  auto m = parseOk(R"(model {
  init {
    return
  }
  exec() {
    %c = const.inline dense<[0.1, 3.4028235e+38, -0.0]> : tensor<f32[3]>
    return %c
  }
}
)");
  const Op& op = m->exec.ops[0];
  REQUIRE(op.data.size() == 12);
  float f[3];
  std::memcpy(f, op.data.data(), 12);
  CHECK(f[0] == 0.1f);
  CHECK(f[1] == 3.4028235e+38f);
  CHECK(std::signbit(f[2]));
}

TEST_CASE("text: parse errors are reported, not thrown", "[text]") {
  CHECK_FALSE(parseModel("model {").isOk());
  CHECK_FALSE(parseModel("model { init { return } exec() { %x = bogus.op : f32 return } }").isOk());
  CHECK_FALSE(parseModel("model { init { return } exec() { return %undefined } }").isOk());
  CHECK_FALSE(parseModel("model { init { return } exec(%x: f32, %x: f32) { return } }").isOk());
  CHECK_FALSE(parseModel("model { init { return } exec() { %b = buffer.alloc : buffer<q7[3]> return } }").isOk());
  CHECK_FALSE(parseModel("model { init { } exec() { return } }").isOk());
  auto r = parseModel("model { init { return } exec() {\n  %b = buffer.alloc : buffer<f32[N +]>\n return } }");
  REQUIRE_FALSE(r.isOk());
  CHECK(r.status().message().find("line 2") != std::string::npos);
}

TEST_CASE("text: symbols and shapes share the model's contexts", "[text]") {
  auto m = parseOk(R"(model {
  init {
    return
  }
  exec(%x: buffer<f32[N, N]>, %y: buffer<f32[N, N]>) {
    return %x
  }
}
)");
  CHECK(m->typeOf(m->exec.args[0]) == m->typeOf(m->exec.args[1]));
  CHECK(m->dims().numSymbols() == 1);
}
