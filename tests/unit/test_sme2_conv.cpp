// SME2 conv tile + the Winograd fused conv (kernels/cpu/sme2_conv.cpp, kokoro_conv_fused.cpp) against direct float64
// references. The SME2 cases only run on CPUs that have it (the iPhone 18 Pro Max test host; not the M2 Max); the Winograd cases run
// everywhere (Accelerate GEMMs without SME2). The Winograd weights are built here in double from the same transforms
// as python/lokinfer/ops/winograd.py (points {0, ±1, ±2, ∞} for K = 3, {0, ±1, ±2, -1/2, ∞} for K = 4..12).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../../kernels/cpu/neon_math.h"
#include "../../kernels/cpu/sme2_gemm.h"
#include "lokinfer/kernels/cpu_kokoro.h"

using namespace lk::kernels::cpu;

namespace {

std::vector<float> randv(size_t n, std::mt19937& g, float scale = 1.f) {
  std::normal_distribution<float> d(0.f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// Exact Lagrange-style transforms over doubles: F(m = 4, r) with the given finite points (+ infinity).
struct Transforms {
  int a, r;
  std::vector<double> at, g, bt;  // [4][a], [a][r], [a][a]
};

std::vector<double> inverse(std::vector<double> m, int n) {
  std::vector<double> inv(n * n, 0.0);
  for (int i = 0; i < n; ++i) inv[i * n + i] = 1.0;
  for (int c = 0; c < n; ++c) {
    int piv = c;
    while (m[piv * n + c] == 0.0) ++piv;
    for (int j = 0; j < n; ++j) std::swap(m[c * n + j], m[piv * n + j]), std::swap(inv[c * n + j], inv[piv * n + j]);
    const double s = 1.0 / m[c * n + c];
    for (int j = 0; j < n; ++j) m[c * n + j] *= s, inv[c * n + j] *= s;
    for (int rr = 0; rr < n; ++rr)
      if (rr != c && m[rr * n + c] != 0.0) {
        const double f = m[rr * n + c];
        for (int j = 0; j < n; ++j) m[rr * n + j] -= f * m[c * n + j], inv[rr * n + j] -= f * inv[c * n + j];
      }
  }
  return inv;
}

Transforms transforms(int r) {
  const std::vector<double> pts = r == 3 ? std::vector<double>{0, 1, -1, 2, -2} : std::vector<double>{0, 1, -1, 2, -2, -0.5};
  const int a = 4 + r - 1;
  auto vander = [&](int n) {
    std::vector<double> v(a * n, 0.0);
    for (int i = 0; i + 1 < a; ++i)
      for (int j = 0; j < n; ++j) v[i * n + j] = std::pow(pts[i], j);
    v[(a - 1) * n + n - 1] = 1.0;
    return v;
  };
  const auto va = vander(4), vb = vander(r), c = inverse(vander(a), a);
  Transforms t{a, r, std::vector<double>(4 * a), vb, std::vector<double>(a * a)};
  for (int i = 0; i < a; ++i)
    for (int j = 0; j < a; ++j) t.bt[i * a + j] = c[j * a + i];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < a; ++j) t.at[i * a + j] = va[j * 4 + i];
  for (int i = 0; i < a; ++i) {
    double s = 0;
    for (int j = 0; j < a; ++j) s = std::max(s, std::abs(t.bt[i * a + j]));
    for (int j = 0; j < a; ++j) t.bt[i * a + j] /= s;
    for (int j = 0; j < r; ++j) t.g[i * r + j] *= s;
  }
  return t;
}

// Direct conv of act(affine(x)) in double (snake / leaky / none), + bias [+ res] [* oscale].
std::vector<double> reference(const ConvFusedArgs& p) {
  const int64_t lo = p.l + 2 * p.padding - p.dilation * (p.k - 1);
  std::vector<double> xa(p.ci * p.l);
  for (int64_t c = 0; c < p.ci; ++c)
    for (int64_t t = 0; t < p.l; ++t) {
      double v = (double(p.x[c * p.l + t]) - p.affine[c]) * p.affine[p.ci + c] + p.affine[2 * p.ci + c];
      if (p.act == ConvAct::Snake) v += std::pow(std::sin(p.alpha[c] * v), 2) / p.alpha[c];
      if (p.act == ConvAct::Leaky && v < 0) v *= p.slope;
      xa[c * p.l + t] = v;
    }
  std::vector<double> out(p.co * lo);
  for (int64_t o = 0; o < p.co; ++o)
    for (int64_t t = 0; t < lo; ++t) {
      double s = p.bias[o];
      for (int64_t c = 0; c < p.ci; ++c)
        for (int64_t kk = 0; kk < p.k; ++kk) {
          const int64_t pos = t - p.padding + kk * p.dilation;
          if (pos >= 0 && pos < p.l) s += double(p.w[(o * p.ci + c) * p.k + kk]) * xa[c * p.l + pos];
        }
      if (p.residual) s = (s + p.residual[o * lo + t]) * p.outScale;
      out[o * lo + t] = s;
    }
  return out;
}

double maxRelErr(const std::vector<float>& got, const std::vector<double>& ref) {
  double e = 0, m = 1e-12;
  for (size_t i = 0; i < ref.size(); ++i) e = std::max(e, std::abs(got[i] - ref[i])), m = std::max(m, std::abs(ref[i]));
  return e / m;
}

}  // namespace

TEST_CASE("sme2: conv window GEMM (shared engine) vs direct", "[sme2]") {
  if (!sme2::available()) SKIP("no SME2 on this CPU");
  std::mt19937 g(7);
  struct C { int64_t co, ci, k, d, n; };
  for (const C c : {C{32, 16, 3, 1, 32}, C{40, 8, 7, 3, 77}, C{128, 128, 11, 5, 512}, C{5, 3, 1, 0, 19}, C{1024, 64, 3, 1, 37}, C{96, 16, 2, 1, 13}}) {
    const int64_t ck = c.ci * c.k, win = c.n + (c.k - 1) * c.d;
    auto w = randv(c.co * ck, g), xw = randv(c.ci * win, g);
    std::vector<float> wp((c.co + 31) / 32 * 32 * ck), o(c.co * c.n, -1.f);
    sme2::packPanels(w.data(), ck, c.co, ck, wp.data());
    sme2::gemm(wp.data(), c.co, ck, sme2::BOperand{xw.data(), win, 32, c.k, c.d}, c.n, o.data(), c.n);
    std::vector<double> ref(c.co * c.n, 0.0);
    for (int64_t oo = 0; oo < c.co; ++oo)
      for (int64_t t = 0; t < c.n; ++t)
        for (int64_t ci = 0; ci < c.ci; ++ci)
          for (int64_t kk = 0; kk < c.k; ++kk)
            ref[oo * c.n + t] += double(w[oo * ck + ci * c.k + kk]) * xw[ci * win + kk * c.d + t];
    CHECK(maxRelErr(o, ref) < 1e-5);
  }
}

TEST_CASE("kokoro: Winograd fused conv vs direct", "[sme2][wino]") {
  std::mt19937 g(11);
  struct C { int64_t ci, co, l, k, d, tile; ConvAct act; bool res; };
  const C cases[] = {
      {8, 8, 50, 3, 1, 128, ConvAct::Snake, false},     {16, 12, 700, 7, 3, 128, ConvAct::Snake, true},
      {12, 16, 97, 11, 5, 256, ConvAct::Snake, false},  {128, 128, 5000, 11, 5, 1024, ConvAct::Snake, true},
      {64, 40, 3000, 3, 1, 640, ConvAct::Leaky, true},  {256, 256, 1100, 7, 3, 1024, ConvAct::Snake, false},
      {300, 256, 147, 3, 1, 256, ConvAct::Leaky, false}, {8, 8, 1, 3, 1, 128, ConvAct::None, false},
  };
  for (const C c : cases) {
    const Transforms t = transforms(c.k == 3 ? 3 : 4);
    const int64_t grp = c.k == 3 ? 1 : std::max<int64_t>(2, (c.k + 3) / 4), cb = (c.co + 31) / 32, a = t.a;
    auto x = randv(c.ci * c.l, g, 2.f), w = randv(c.co * c.ci * c.k, g, 1.f / std::sqrt(float(c.ci * c.k)));
    auto b = randv(c.co, g), res = randv(c.co * c.l, g);
    std::vector<float> affine(3 * c.ci), alpha(c.ci);
    for (int64_t i = 0; i < c.ci; ++i)
      affine[i] = 0.3f * std::sin(float(i)), affine[c.ci + i] = 1.f + 0.2f * std::cos(float(i)),
      affine[2 * c.ci + i] = 0.1f, alpha[i] = 0.5f + 0.01f * float(i % 50);
    // U[ξ][cb][c·G + gg][32] = Σ_j G[ξ][j] · w[o, c, 4·gg + j] (taps past K are zero).
    std::vector<float> up(a * cb * c.ci * grp * 32, 0.f);
    for (int xi = 0; xi < a; ++xi)
      for (int64_t o = 0; o < c.co; ++o)
        for (int64_t ci = 0; ci < c.ci; ++ci)
          for (int64_t gg = 0; gg < grp; ++gg) {
            double s = 0;
            for (int j = 0; j < t.r; ++j) {
              const int64_t tap = gg * 4 + j;
              if (tap < c.k) s += t.g[xi * t.r + j] * w[(o * c.ci + ci) * c.k + tap];
            }
            up[((xi * cb + o / 32) * c.ci * grp + ci * grp + gg) * 32 + o % 32] = static_cast<float>(s);
          }
    std::vector<float> tr(a * a + 4 * a);
    for (int i = 0; i < a * a; ++i) tr[i] = static_cast<float>(t.bt[i]);
    for (int i = 0; i < 4 * a; ++i) tr[a * a + i] = static_cast<float>(t.at[i]);
    const int64_t pad = c.d * (c.k - 1) / 2, lo = c.l;
    std::vector<float> out(c.co * lo), stats(c.co * statChunks(lo) * 2);
    ConvFusedArgs p;
    p.x = x.data(), p.w = w.data(), p.bias = b.data(), p.affine = affine.data(), p.alpha = alpha.data();
    p.residual = c.res ? res.data() : nullptr, p.out = out.data(), p.stats = stats.data();
    p.ci = c.ci, p.l = c.l, p.co = c.co, p.k = c.k, p.dilation = c.d, p.padding = pad, p.tile = c.tile, p.act = c.act;
    p.slope = 0.2f, p.outScale = c.res && c.act == ConvAct::Leaky ? 0.7f : 1.f;
    conv1d_wino_f32(p, up.data(), tr.data(), grp);
    const auto ref = reference(p);
    INFO("ci " << c.ci << " co " << c.co << " l " << c.l << " k " << c.k << " d " << c.d);
    CHECK(maxRelErr(out, ref) < 2e-5);
    // Chunk means of the output (stats[c][chunk][0]) against the output itself.
    double e = 0;
    for (int64_t o = 0; o < c.co; ++o)
      for (int64_t ch = 0; ch < statChunks(lo); ++ch) {
        const int64_t t0 = ch * kStatChunk, n = std::min<int64_t>(kStatChunk, lo - t0);
        double m = 0;
        for (int64_t j = 0; j < n; ++j) m += out[o * lo + t0 + j];
        e = std::max(e, std::abs(m / n - stats[(o * statChunks(lo) + ch) * 2]));
      }
    CHECK(e < 1e-4);
  }
}


// kernels/cpu/neon_math.h polynomials vs double libm (the bounds documented in the header).
TEST_CASE("neon_math: polynomial transcendentals vs double") {
  namespace d = lk::kernels::cpu::detail;
  std::mt19937 g(7);
  auto maxErr = [&](auto vecFn, auto refFn, float lo, float hi, bool rel, int n = 200000) {
    std::uniform_real_distribution<float> u(lo, hi);
    double worst = 0;
    for (int i = 0; i < n; i += 4) {
      float in[4], out[4];
      for (float& v : in) v = u(g);
      vst1q_f32(out, vecFn(vld1q_f32(in)));
      for (int q = 0; q < 4; ++q) {
        const double r = refFn(static_cast<double>(in[q]));
        const double e = std::fabs(out[q] - r) / (rel ? std::max(std::fabs(r), 1e-30) : 1.0);
        worst = std::max(worst, e);
      }
    }
    return worst;
  };
  auto sinsq = [](double x) { return std::sin(x) * std::sin(x); };
  const double eSq = maxErr(d::sinSq4, sinsq, -1e4f, 1e4f, false);
  const double eSin = maxErr(d::sin4, [](double x) { return std::sin(x); }, -1e4f, 1e4f, false);
  const double eCos = maxErr(d::cos4, [](double x) { return std::cos(x); }, -1e4f, 1e4f, false);
  const double eWide = maxErr(d::sin4Wide, [](double x) { return std::sin(x); }, -1.6e7f, 1.6e7f, false);
  const double eExp = maxErr(d::exp4, [](double x) { return std::exp(x); }, -80.f, 80.f, true);
  const double eLog = std::max(maxErr(d::log4, [](double x) { return std::log(x); }, 0.5f, 1.f, false),
                              maxErr(d::log4, [](double x) { return std::log(x); }, 1e-9f, 0.5f, true));
  const double eLogR = maxErr(d::log4, [](double x) { return std::log(x); }, 2.f, 1e6f, true);
  const double eTanh = maxErr(d::tanh4, [](double x) { return std::tanh(x); }, -12.f, 12.f, false);
  const double eErfc = maxErr(d::erfc4, [](double x) { return std::erfc(x); }, -6.f, 9.f, true);
  // vs the basic kernels' f32 semantics: the argument -x / sqrt(2) is rounded to f32 first (f16_basic.cpp geluErf);
  // that rounding alone moves erfc by ~2 z^2 6e-8 relative in the far tail, for the scalar kernel too
  const double eGelu = maxErr(d::geluErf4, [](double x) {
    return 0.5 * x * std::erfc(static_cast<double>(static_cast<float>(x) * -0.70710678118654752f)); }, -12.f, 12.f, true);
  double eAt2 = 0;
  {
    std::uniform_real_distribution<float> u(-3.f, 3.f);
    for (int i = 0; i < 100000; ++i) {
      float ys[4], xs[4], out[4];
      for (int q = 0; q < 4; ++q) ys[q] = u(g), xs[q] = u(g);
      if (i == 0) ys[0] = 0.f, xs[0] = -1.f, ys[1] = 0.f, xs[1] = 0.f, ys[2] = -0.f, xs[2] = -2.f;
      vst1q_f32(out, d::atan2_4(vld1q_f32(ys), vld1q_f32(xs)));
      for (int q = 0; q < 4; ++q) eAt2 = std::max(eAt2, std::fabs(out[q] - std::atan2(double(ys[q]), double(xs[q]))));
    }
  }
  INFO("sinSq4 " << eSq << " sin4 " << eSin << " cos4 " << eCos << " sin4Wide " << eWide << " exp4(rel) " << eExp
                 << " log4(abs [.5,1], rel below) " << eLog << " log4(rel) " << eLogR << " tanh4 " << eTanh << " atan2 " << eAt2);
  std::printf("neon_math max errors: sinSq4 %.2e sin4 %.2e cos4 %.2e sin4Wide %.2e exp4(rel) %.2e log4(abs [.5,1], rel below) %.2e "
              "log4(rel) %.2e tanh4 %.2e atan2 %.2e\n", eSq, eSin, eCos, eWide, eExp, eLog, eLogR, eTanh, eAt2);
  CHECK(eSq < 3e-7);
  CHECK(eSin < 3e-7);
  CHECK(eCos < 3e-7);
  CHECK(eWide < 3e-7);
  CHECK(eExp < 3e-7);
  CHECK(eLog < 3e-7);
  CHECK(eLogR < 3e-7);
  CHECK(eTanh < 3e-7);
  CHECK(eAt2 < 4e-7);
  std::printf("neon_math max errors: erfc4(rel, [-6, 9]) %.2e geluErf4(rel vs f32-argument erfc, [-12, 12]) %.2e\n", eErfc, eGelu);
  CHECK(eErfc < 1e-6);
  CHECK(eGelu < 1e-6);
}


// The LSTM step GEMV: sme2::gemvT must equal, bit for bit, the 4-partial order of kokoro_lstm.cpp's NEON gemvT4 (the
// order the Kokoro SNR tests validate on the Mac): partial j mod 4 with fused multiply-adds, (p0 + p1) + (p2 + p3), + y.
TEST_CASE("sme2: LSTM gemvT is bit-identical to the 4-partial order", "[sme2]") {
  if (!sme2::available()) SKIP("no SME2 on this CPU");
  std::mt19937 g(5);
  for (const int64_t k : {256, 7, 1}) {
    for (const int64_t n : {1024, 64, 320}) {
      const int64_t ldw = 2 * n + 16;
      auto w = randv(k * ldw, g), x = randv(k, g), y0 = randv(n, g);
      std::vector<float> y = y0, ref(n);
      sme2::gemvT(w.data(), ldw, k, n, x.data(), y.data());
      for (int64_t i = 0; i < n; ++i) {
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        for (int64_t j = 0; j < k; ++j) p[j & 3] = std::fma(w[j * ldw + i], x[j], p[j & 3]);
        ref[i] = y0[i] + ((p[0] + p[1]) + (p[2] + p[3]));
      }
      CHECK(y == ref);
    }
  }
}
