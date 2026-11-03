#include <catch2/catch_test_macros.hpp>

#include "lokinfer/ir/dim.h"

using namespace lk;

namespace {

std::string str(DimContext& c, Dim d) { return c.toString(d); }

Dim parse(DimContext& c, std::string_view s) {
  auto r = c.parse(s);
  REQUIRE(r.isOk());
  return *r;
}

}  // namespace

TEST_CASE("dim: constants and symbols are interned", "[dim]") {
  DimContext c;
  CHECK(c.constant(3) == c.constant(3));
  CHECK(c.sym("N") == c.sym("N"));
  CHECK(c.sym("N") != c.sym("M"));
  CHECK(c.numSymbols() == 2);
}

TEST_CASE("dim: constant folding", "[dim]") {
  DimContext c;
  auto k = [&](int64_t v) { return c.constant(v); };
  CHECK(c.add(k(2), k(3)) == k(5));
  CHECK(c.mul(k(4), k(3)) == k(12));
  CHECK(c.floorDiv(k(7), k(2)) == k(3));
  CHECK(c.floorDiv(k(-7), k(2)) == k(-4));
  CHECK(c.ceilDiv(k(7), k(2)) == k(4));
  CHECK(c.mod(k(-7), k(3)) == k(2));
  CHECK(c.min(k(4), k(9)) == k(4));
  CHECK(c.max(k(4), k(9)) == k(9));
}

TEST_CASE("dim: canonicalization makes equal expressions identical", "[dim]") {
  DimContext c;
  Dim N = c.sym("N"), M = c.sym("M");
  CHECK(c.add(N, M) == c.add(M, N));
  CHECK(c.mul(N, M) == c.mul(M, N));
  CHECK(c.add(N, N) == c.mul(c.constant(2), N));
  CHECK(c.sub(N, N) == c.constant(0));
  CHECK(c.add(c.add(N, c.constant(1)), c.constant(-1)) == N);
  CHECK(c.mul(N, c.constant(1)) == N);
  CHECK(c.mul(N, c.constant(0)) == c.constant(0));
  CHECK(c.mul(c.constant(2), c.add(N, c.constant(1))) == c.add(c.mul(c.constant(2), N), c.constant(2)));
  CHECK(c.add(c.mul(N, M), c.mul(M, N)) == c.mul(c.constant(2), c.mul(N, M)));
  CHECK(c.min(N, M) == c.min(M, N));
  CHECK(c.max(N, N) == N);
}

TEST_CASE("dim: exact division identities", "[dim]") {
  DimContext c;
  Dim N = c.sym("N");
  CHECK(c.floorDiv(c.mul(c.constant(4), N), c.constant(2)) == c.mul(c.constant(2), N));
  CHECK(c.ceilDiv(c.add(c.mul(c.constant(4), N), c.constant(8)), c.constant(4)) == c.add(N, c.constant(2)));
  CHECK(c.mod(c.mul(c.constant(6), N), c.constant(3)) == c.constant(0));
  CHECK(c.floorDiv(N, c.constant(1)) == N);
  CHECK(c.floorDiv(N, N) == c.constant(1));
  // Not exactly divisible: stays symbolic.
  CHECK(c.floorDiv(N, c.constant(2))->kind == DimKind::FloorDiv);
}

TEST_CASE("dim: evaluation", "[dim]") {
  DimContext c;
  Dim N = c.sym("N"), M = c.sym("M");
  // Conv output size: (N + 2*1 - 3) // 2 + 1
  Dim conv = c.add(c.floorDiv(c.add(N, c.constant(-1)), c.constant(2)), c.constant(1));
  std::vector<int64_t> vals = {10, 3};  // N=10, M=3
  CHECK(c.evaluate(conv, vals) == 5);
  CHECK(c.evaluate(c.ceilDiv(N, M), vals) == 4);
  CHECK(c.evaluate(c.mod(N, M), vals) == 1);
  CHECK(c.evaluate(c.min(N, M), vals) == 3);
  CHECK(c.evaluate(c.max(N, M), vals) == 10);
  CHECK(c.evaluate(c.mul(N, M), vals) == 30);
}

TEST_CASE("dim: printing", "[dim]") {
  DimContext c;
  Dim N = c.sym("N"), M = c.sym("M");
  CHECK(str(c, N) == "N");
  CHECK(str(c, c.constant(-3)) == "-3");
  CHECK(str(c, c.add(N, c.constant(1))) == "N + 1");
  CHECK(str(c, c.sub(N, c.constant(1))) == "N - 1");
  CHECK(str(c, c.mul(c.constant(2), N)) == "2*N");
  CHECK(str(c, c.sub(M, N)) == "M - N");
  CHECK(str(c, c.neg(N)) == "-N");
  CHECK(str(c, c.mul(N, c.add(M, c.constant(1)))) == "N*(M + 1)");
  CHECK(str(c, c.floorDiv(c.add(N, c.constant(1)), c.constant(2))) == "(N + 1) // 2");
  CHECK(str(c, c.ceilDiv(N, c.constant(4))) == "ceildiv(N, 4)");
  CHECK(str(c, c.min(N, M)) == "min(M, N)");
}

TEST_CASE("dim: parse/print round trip", "[dim]") {
  const char* cases[] = {
      "N", "42", "N + 1", "2*N - 3", "N*M", "N*(M + 1)", "(N + 1) // 2", "N % 4",
      "ceildiv(N, 4)", "min(M, N)", "max(N, 512)", "(2*N - 3) // 2 + 1", "-N + M",
      "3*(N // 2)", "N*M*K + K*2",
  };
  for (const char* s : cases) {
    DimContext a;
    Dim d = parse(a, s);
    std::string printed = a.toString(d);
    DimContext b;  // fresh context: printing must not depend on creation order
    Dim d2 = parse(b, printed);
    INFO(s << " -> " << printed);
    CHECK(b.toString(d2) == printed);
  }
}

TEST_CASE("dim: parse precedence and errors", "[dim]") {
  DimContext c;
  Dim N = c.sym("N");
  CHECK(parse(c, "2*N + 1") == c.add(c.mul(c.constant(2), N), c.constant(1)));
  CHECK(parse(c, "N*2 // 3") == c.floorDiv(c.mul(c.constant(2), N), c.constant(3)));
  CHECK(parse(c, "-(N - 1)") == c.sub(c.constant(1), N));
  CHECK_FALSE(c.parse("N +").isOk());
  CHECK_FALSE(c.parse("(N").isOk());
  CHECK_FALSE(c.parse("N $ 2").isOk());
  CHECK_FALSE(c.parse("min(N)").isOk());
}

TEST_CASE("dim: affine floor/ceil division and mod simplify", "[dim]") {
  DimContext c;
  Dim F = c.sym("F");
  auto k = [&](int64_t v) { return c.constant(v); };
  // Kokoro: conv k3 s2 p1 over 2F -> F; conv k12 s6 p3 over 120F+1 -> 20F.
  CHECK(c.add(c.floorDiv(c.sub(c.mul(k(2), F), k(1)), k(2)), k(1)) == F);
  CHECK(c.add(c.floorDiv(c.sub(c.mul(k(120), F), k(5)), k(6)), k(1)) == c.mul(k(20), F));
  CHECK(c.ceilDiv(c.add(c.mul(k(4), F), k(3)), k(4)) == c.add(F, k(1)));
  CHECK(c.mod(c.add(c.mul(k(6), F), k(7)), k(3)) == k(1));
  CHECK(c.floorDiv(c.add(c.mul(k(4), F), k(-1)), k(4)) == c.sub(F, k(1)));
  // Not simplifiable: stays symbolic.
  CHECK(c.floorDiv(c.add(F, k(1)), k(2))->kind == DimKind::FloorDiv);
}

TEST_CASE("dim: tryEvaluate reports unbound symbols and division by zero", "[dim]") {
  DimContext c;
  Dim N = c.sym("N"), M = c.sym("M");
  const int64_t U = INT64_MIN;
  std::vector<int64_t> vals = {5, U};
  CHECK(c.tryEvaluate(c.mul(c.constant(2), N), vals, U) == 10);
  CHECK_FALSE(c.tryEvaluate(c.add(N, M), vals, U).has_value());
  std::vector<int64_t> zero = {5, 0};
  CHECK_FALSE(c.tryEvaluate(c.floorDiv(N, M), zero, U).has_value());
}

TEST_CASE("dim: hostile parser input returns errors", "[dim]") {
  DimContext c;
  CHECK_FALSE(c.parse("N // 0").isOk());
  CHECK_FALSE(c.parse("N % 0").isOk());
  CHECK_FALSE(c.parse("ceildiv(N, 0)").isOk());
  CHECK_FALSE(c.parse("99999999999999999999999").isOk());
  CHECK(c.parse("9223372036854775807").isOk());
  CHECK_FALSE(c.parse(std::string(100000, '(') + "N" + std::string(100000, ')')).isOk());
  CHECK_FALSE(c.parse(std::string(100000, '-') + "N").isOk());
}
