"""f16 optimized CPU kernels (Q5, docs/impl-notes/qwen3-asr-f16-opt-kernels.md; kernels/cpu/f16_fused.cpp) vs the
composition of the basic f16 CPU kernels (the oracle) AND the float64 torch / numpy composition.

Every op runs through a one-op graph (eager call; a compiled graph with persistent caches for the KV write), inputs
drawn as f16 and upcast for the reference. Real Qwen3-ASR shapes (M = 1 decode, 44 / 401 / 2048 prefill chunks, 13
conv_out rows, encoder S up to 15,600 in 104-row windows, caches up to P = 25,600 at pos 20,000) + odd sizes + length 1
+ the path boundaries (M = 4 / 5: K-split GEMV vs tiles; M = 16 / 32 / 33: the SME2 panels on the iPhone; column
tails; key splits). Also runs on the iPhone (`pytest --device=iphone`). Long cases are `slow` (`--run-slow`).

Tolerances (u = 2⁻¹¹, the f16 unit roundoff; 2⁻²⁵ = half the f16 subnormal spacing), as in test_f16_basic_cpu.py:
a kernel computes c in f32 and rounds once, so |out - ref| <= u·|ref| + (1 + u)·|c - ref| + 2⁻²⁵, and each test bounds
the f32 error |c - ref| ("extra"):
- GEMM sums over K: e_y = 2·2⁻²⁴·√K·(|x|@|w| + |b|) (random-walk size of an f32 sum; the K-split / tile orders are
  sums of the same products in another order, same bound);
- epilogues: + residual / clamp (1-Lipschitz): e_y; GELU: 1.13·e_y (max |gelu'|) + 2⁻¹⁸·|ref| (f32 erfc);
  SwiGLU: 1.1·|u|·e_g + |silu(g)|·e_u + 2⁻¹⁸·|ref| (max |silu'| = 1.1; vector exp, f32 divide);
- attention: score error ds = 2⁻²⁴·√hd·scale·max‖q‖·max‖k‖, f32 accumulation over n keys:
  (2·ds + 2⁻²⁴·(4√n + 32))·max|v| (the vector exp's 2.2e-7 relative error is inside the 32·2⁻²⁴ term);
- qknorm_rope_kv: the composition rounds rmsnorm's y to f16 before RoPE (kept by the fused kernel), so the f64
  reference rounds it too; an h one ulp off (f32 h at a rounding midpoint, see the rmsnorm test) moves g·h by
  <= 2⁻¹⁰·|y| and y may then round to the other neighbour (another 2⁻¹⁰·|y|): 2⁻⁹·(|y_j| + |y_j+64|) (seen: 1.12x
  the one-ulp bound, once in 4M outputs, by the oracle itself); RoPE's f32 cos / sin: 2⁻²⁰·(|y_j| + |y_j+64|). Same code as the oracle: >= 99.9 % of
  the outputs must be bit-exact to it;
- lm_head_last (f32 output, no rounding): 2·2⁻²⁴·√D·(|table|@|x|);
- audio_cnn: the last GEMM's e_y + u·rms(ref) for the f16 rounding flips of the intermediate layers (both the kernel
  and the reference round between layers).
The oracle (basic composition) rounds intermediates the fused kernel keeps in f32 (y before GELU / SwiGLU / the
residual add), so its own bound adds those roundings: (u·|y| + e_y) through the epilogue's derivative. The two outputs
must then agree within the sum of both bounds (each is also checked against float64 on its own).

Speedups vs basic: docs/models/qwen3-asr.md, "Q5 CPU kernels".
"""
import numpy as np
import pytest
import torch

import lokinfer as lk
from lokinfer.ops import f16 as F

U = 2.0 ** -11
TINY = 2.0 ** -25


def H(a):
    return lk.from_numpy(np.ascontiguousarray(a, np.float16))


def I(a):
    return lk.from_numpy(np.ascontiguousarray(a, np.int64))


def rnd(r, *shape, scale=1.0, loc=0.0):
    return (r.standard_normal(shape) * scale + loc).astype(np.float16)


def d(a):
    return np.asarray(a, np.float64)


def tol(ref, extra):
    return U * np.abs(ref) + (1 + U) * extra + TINY


def check(got, ref, extra=0.0, what=""):
    """|got - ref| <= u·|ref| + (1 + u)·extra + 2⁻²⁵ elementwise (got f16, ref float64), all finite."""
    _within(d(got), ref, tol(ref, extra), what)


def check_oracle(got, oracle, ref, extra, extra_oracle, what=""):
    """The fused output vs the basic composition: within the sum of both outputs' float64 bounds."""
    _within(d(got), d(oracle), tol(ref, extra) + tol(ref, extra_oracle), what + " vs oracle")


def _within(got, ref, t, what):
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.isfinite(got).all(), f"{what}: non-finite output"
    err = np.abs(got - ref)
    if (err > t).any():
        i = np.unravel_index(np.argmax(err - t), err.shape)
        raise AssertionError(f"{what}: {(err > t).sum()} / {err.size} outside tolerance; worst at {i}: got "
                             f"{got[i]!r} ref {ref[i]!r} err {err[i]:.3g} tol {np.broadcast_to(t, err.shape)[i]:.3g}")


def exact(got, ref):
    got = np.asarray(got)
    assert got.dtype == np.float16
    np.testing.assert_array_equal(got.view(np.uint16), np.asarray(ref, np.float16).view(np.uint16))


def gemm_extra(x, w, b=None):
    s = np.abs(d(x)) @ np.abs(d(w))
    if b is not None:
        s = s + np.abs(d(b))
    return 2 * 2.0 ** -24 * np.sqrt(x.shape[1]) * s


# --- GEMMs ---------------------------------------------------------------------------------------------------------

GEMM_SHAPES = [
    # real: decode GEMVs (LM q / kv / o, gate-up halves, down; projector), prefill chunks, conv_out, encoder
    (1, 1024, 1024), (1, 2048, 2048), (1, 2048, 1024), (1, 6144, 2048), (1, 1024, 2048), (13, 7680, 1024),
    (44, 1024, 2048), (44, 2048, 4096), (104, 1024, 3072), (401, 1024, 4096), (401, 4096, 1024), (2048, 2048, 1024),
    # path boundaries (GEMV M <= 4, tiles, SME2 panels of 32 rows) and odd sizes / column tails
    (2, 3, 17), (3, 5, 7), (4, 16, 16), (5, 37, 19), (6, 8, 31), (7, 100, 33), (16, 64, 48), (17, 129, 47),
    (31, 66, 130), (32, 128, 32), (33, 130, 65), (1, 1, 1), (1, 7, 9), (4, 300, 1000), (64, 9, 480),
]


@pytest.mark.parametrize("m,k,n", GEMM_SHAPES)
def test_gemm_opt(m, k, n):
    r = np.random.default_rng(m * 7 + k * 3 + n)
    x, w, b = rnd(r, m, k), rnd(r, k, n, scale=1 / np.sqrt(k)), rnd(r, n, scale=0.5)
    ref, e = d(x) @ d(w), gemm_extra(x, w)
    got = F.gemm_opt(H(x), H(w)).numpy()
    check(got, ref, e, "gemm_opt")
    check_oracle(got, F.gemm(H(x), H(w)).numpy(), ref, e, e, "gemm_opt")
    eb = gemm_extra(x, w, b)
    got = F.gemm_bias_opt(H(x), H(w), H(b)).numpy()
    check(got, ref + d(b), eb, "gemm_bias_opt")
    check_oracle(got, F.gemm_bias(H(x), H(w), H(b)).numpy(), ref + d(b), eb, eb, "gemm_bias_opt")


def _gelu(y):
    return torch.nn.functional.gelu(torch.from_numpy(y)).numpy()


@pytest.mark.parametrize("m,k,n", [(1, 1024, 4096), (104, 1024, 4096), (401, 1024, 4096), (13, 1024, 1024),
                                   (44, 1024, 2048), (5, 37, 19), (3, 9, 480), (33, 40, 70)])
def test_gemm_bias_gelu_opt(m, k, n):
    """Encoder fc1 / projector 1. Oracle: gelu_erf(gemm_bias(...)), y rounded in between."""
    r = np.random.default_rng(m + k + n)
    x, w, b = rnd(r, m, k), rnd(r, k, n, scale=2 / np.sqrt(k)), rnd(r, n, scale=0.5)
    y = d(x) @ d(w) + d(b)
    ref, ey = _gelu(y), gemm_extra(x, w, b)
    extra = 1.13 * ey + 2.0 ** -18 * np.abs(ref)
    got = F.gemm_bias_gelu_opt(H(x), H(w), H(b)).numpy()
    check(got, ref, extra, "gemm_bias_gelu_opt")
    oracle = F.gelu_erf(F.gemm_bias(H(x), H(w), H(b))).numpy()
    check_oracle(got, oracle, ref, extra, extra + 1.13 * (U * np.abs(y) + ey + TINY), "gemm_bias_gelu_opt")


def _res(r, m, n):
    return rnd(r, m, n, scale=2)


@pytest.mark.parametrize("m,k,n", [(1, 2048, 2048), (1, 6144, 2048), (44, 2048, 2048), (401, 6144, 2048),
                                   (7, 100, 33), (2, 64, 24), (40, 50, 60)])
def test_gemm_add_opt(m, k, n):
    """LM o_proj / down_proj + residual. Oracle: add(res, gemm(...))."""
    r = np.random.default_rng(m * 5 + n)
    x, w, res = rnd(r, m, k), rnd(r, k, n, scale=1 / np.sqrt(k)), _res(r, m, n)
    y = d(x) @ d(w)
    ref, ey = d(res) + y, gemm_extra(x, w)
    got = F.gemm_add_opt(H(x), H(w), H(res)).numpy()
    check(got, ref, ey, "gemm_add_opt")
    check_oracle(got, F.add(H(res), F.gemm(H(x), H(w))).numpy(), ref, ey, ey + U * np.abs(y) + TINY, "gemm_add_opt")


@pytest.mark.parametrize("m,k,n", [(1, 1024, 1024), (104, 1024, 1024), (401, 1024, 1024), (3, 33, 17)])
def test_gemm_bias_add_opt(m, k, n):
    """Encoder out_proj + residual. Oracle: add(res, gemm_bias(...))."""
    r = np.random.default_rng(m * 3 + k)
    x, w, b, res = rnd(r, m, k), rnd(r, k, n, scale=1 / np.sqrt(k)), rnd(r, n, scale=0.5), _res(r, m, n)
    y = d(x) @ d(w) + d(b)
    ref, ey = d(res) + y, gemm_extra(x, w, b)
    got = F.gemm_bias_add_opt(H(x), H(w), H(b), H(res)).numpy()
    check(got, ref, ey, "gemm_bias_add_opt")
    oracle = F.add(H(res), F.gemm_bias(H(x), H(w), H(b))).numpy()
    check_oracle(got, oracle, ref, ey, ey + U * np.abs(y) + TINY, "gemm_bias_add_opt")


@pytest.mark.parametrize("m,k,n", [(1, 4096, 1024), (104, 4096, 1024), (401, 4096, 1024), (6, 20, 40)])
def test_gemm_bias_add_clamp_opt(m, k, n):
    """Encoder fc2 + residual + the f16 clamp (lim 64504; residuals at ±64512 exceed it), and a small lim that clamps
    a third of the outputs. Oracle: add_clamp(res, gemm_bias(...), lim)."""
    r = np.random.default_rng(m + 2 * k + n)
    x, w, b, res = rnd(r, m, k), rnd(r, k, n, scale=1 / np.sqrt(k)), rnd(r, n, scale=0.5), _res(r, m, n)
    big = r.random((m, n)) < 0.2
    res[big] = np.float16(64512) * np.sign(r.standard_normal(big.sum())).astype(np.float16)
    y = d(x) @ d(w) + d(b)
    ey = gemm_extra(x, w, b)
    for lim in (64504.0, 2.0):
        ref = np.clip(d(res) + y, -lim, lim)
        got = F.gemm_bias_add_clamp_opt(H(x), H(w), H(b), H(res), lim).numpy()
        check(got, ref, ey, f"gemm_bias_add_clamp_opt lim {lim}")
        oracle = F.add_clamp(H(res), F.gemm_bias(H(x), H(w), H(b)), lim).numpy()
        check_oracle(got, oracle, ref, ey, ey + U * np.abs(y) + TINY, f"gemm_bias_add_clamp_opt lim {lim}")


def _silu(g):
    return g / (1 + np.exp(-g))


def _swiglu_case(m, k, n, seed):
    r = np.random.default_rng(seed)
    x, w = rnd(r, m, k), rnd(r, k, 2 * n, scale=2 / np.sqrt(k))
    y = d(x) @ d(w)
    e = gemm_extra(x, w)
    g, u, eg, eu = y[:, :n], y[:, n:], e[:, :n], e[:, n:]
    ref = _silu(g) * u
    extra = 1.1 * np.abs(u) * eg + np.abs(_silu(g)) * eu + 2.0 ** -18 * np.abs(ref)
    got = F.gemm_swiglu_opt(H(x), H(w)).numpy()
    check(got, ref, extra, "gemm_swiglu_opt")
    yo = F.gemm(H(x), H(w)).numpy()
    oracle = F.silu_mul(H(yo[:, :n]), H(yo[:, n:])).numpy()
    eo = extra + 1.1 * np.abs(u) * (U * np.abs(g) + TINY) + np.abs(_silu(g)) * (U * np.abs(u) + TINY)
    check_oracle(got, oracle, ref, extra, eo, "gemm_swiglu_opt")


@pytest.mark.parametrize("m,k,n", [(1, 2048, 6144), (44, 2048, 6144), (3, 5, 7), (6, 40, 19), (17, 129, 47),
                                   (1, 64, 20), (33, 64, 40)])
def test_gemm_swiglu_opt(m, k, n):
    """LM gate / up (w = [gate | up], 12,288 columns) + SwiGLU. Oracle: silu_mul of gemm's two halves."""
    _swiglu_case(m, k, n, m + k + n)


@pytest.mark.slow
@pytest.mark.parametrize("m,k,n,kind", [(401, 2048, 6144, "swiglu"), (2048, 2048, 6144, "swiglu"),
                                        (2048, 6144, 2048, "gemm"), (2048, 2048, 2048, "gemm")])
def test_gemm_opt_full_chunk(m, k, n, kind):
    """Full 2048-row prefill chunks on the biggest LM GEMMs."""
    if kind == "swiglu":
        return _swiglu_case(m, k, n, 99)
    r = np.random.default_rng(m + n)
    x, w = rnd(r, m, k), rnd(r, k, n, scale=1 / np.sqrt(k))
    ref, e = d(x) @ d(w), gemm_extra(x, w)
    got = F.gemm_opt(H(x), H(w)).numpy()
    check(got, ref, e, "gemm_opt")
    check_oracle(got, F.gemm(H(x), H(w)).numpy(), ref, e, e, "gemm_opt")


# --- LM head ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t,v,dd", [(1, 151936, 2048), (44, 4099, 2048), (3, 1000, 24), (1, 7, 5), (2, 513, 1),
                                    (1, 9, 13)])
def test_lm_head_last_opt(t, v, dd):
    """f32 output, no rounding: only the f32 accumulation error (real shape: V = 151,936, D = 2048)."""
    r = np.random.default_rng(v + dd)
    x, table = rnd(r, t, dd), rnd(r, v, dd, scale=1 / np.sqrt(dd))
    got = F.lm_head_last_opt(H(x), H(table)).numpy()
    assert got.dtype == np.float32 and got.shape == (1, v)
    ref = (d(table) @ d(x[-1]))[None]
    bound = 2 * 2.0 ** -24 * np.sqrt(dd) * (np.abs(d(table)) @ np.abs(d(x[-1])))[None] + 1e-30
    assert (np.abs(d(got) - ref) <= bound).all(), np.abs(d(got) - ref).max()
    oracle = F.lm_head_last(H(x), H(table)).numpy()
    assert (np.abs(d(got) - d(oracle)) <= 2 * bound).all()


# --- QK-norm + RoPE + KV write -------------------------------------------------------------------------------------


def _rope_ref(y, pos, theta=1e6):
    """rope_f16 on f16 y (HF Qwen3RotaryEmbedding angles in f32, cos / sin in float64), and |y_j| + |y_j+64|."""
    inv = 1.0 / (theta ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))
    p = torch.arange(pos, pos + y.shape[0], dtype=torch.int64).float()
    ang = d((p[:, None] * inv[None]).numpy())
    cs, sn = np.cos(ang), np.sin(ang)
    yd = d(y).reshape(y.shape[0], -1, 128)
    lo, hi = yd[..., :64], yd[..., 64:]
    out = np.concatenate([lo * cs[:, None] - hi * sn[:, None], hi * cs[:, None] + lo * sn[:, None]], -1)
    mag = np.concatenate([np.abs(lo) + np.abs(hi)] * 2, -1)
    return out.reshape(y.shape), mag.reshape(y.shape)


def _norm_rope_ref(x, g, pos, eps=1e-6):
    xd = d(x).reshape(x.shape[0], -1, 128)
    h = (xd / np.sqrt((xd ** 2).mean(-1, keepdims=True) + eps)).astype(np.float16)
    y = (d(h) * d(g)).astype(np.float16).reshape(x.shape)
    ref, mag = _rope_ref(y, pos)
    return ref, (2.0 ** -20 + 2.0 ** -9) * mag


def _qkv_model(cap, dq, dk):
    T = lk.Sym("T")

    def fn(qkv, qg, kg, pos, rows):
        kc = lk.persistent_buffer(lk.f16, [cap, dk])
        vc = lk.persistent_buffer(lk.f16, [cap, dk])
        q = F.qknorm_rope_kv_opt(qkv, qg, kg, pos, kc, vc)
        return q, F.embedding(rows, kc), F.embedding(rows, vc)

    return lk.compile(fn, input_types=[lk.TensorType(lk.f16, [T, dq + 2 * dk]), lk.TensorType(lk.f16, [128]),
                                       lk.TensorType(lk.f16, [128]), lk.TensorType(lk.i64, [1]),
                                       lk.TensorType(lk.i64, [T])])


def _bitexact_frac(a, b):
    return (np.asarray(a, np.float16).view(np.uint16) == np.asarray(b, np.float16).view(np.uint16)).mean()


@pytest.mark.parametrize("t,pos,cap,dq,dk", [(1, 0, 64, 2048, 1024), (1, 37, 64, 2048, 1024), (44, 0, 64, 2048, 1024),
                                             (401, 7, 512, 2048, 1024), (2048, 0, 2048, 2048, 1024),
                                             (64, 20000, 25600, 2048, 1024), (1, 25599, 25600, 2048, 1024),
                                             (5, 3, 9, 384, 128)])
def test_qknorm_rope_kv_opt(t, pos, cap, dq, dk):
    """q returned, k / v written at rows pos .. pos+T-1 of the caches (read back by a gather). Oracle: rope(
    rmsnorm_heads(.)) for q and k, the v slice."""
    r = np.random.default_rng(t * 11 + pos)
    qkv, qg, kg = rnd(r, t, dq + 2 * dk, scale=3), rnd(r, 128), rnd(r, 128)
    qkv[:, :128] *= np.float16(0.01)  # a head of very different scale
    q, kc, vc = _qkv_model(cap, dq, dk)(H(qkv), H(qg), H(kg), I([pos]), I(np.arange(pos, pos + t)))
    q, kc, vc = q.numpy(), kc.numpy(), vc.numpy()
    exact(vc, qkv[:, dq + dk:])
    for got, x, g in ((q, qkv[:, :dq], qg), (kc, qkv[:, dq:dq + dk], kg)):
        ref, extra = _norm_rope_ref(x, g, pos)
        check(got, ref, extra, "qknorm_rope_kv_opt")
        oracle = F.rope(F.rmsnorm_heads(H(x), H(g)), I([pos])).numpy()
        check_oracle(got, oracle, ref, extra, extra, "qknorm_rope_kv_opt")
        assert _bitexact_frac(got, oracle) >= 0.999


def test_qknorm_rope_kv_opt_bounds():
    m = _qkv_model(16, 384, 128)
    r = np.random.default_rng(3)
    qkv, g = rnd(r, 3, 640), rnd(r, 128)
    for pos in (14, -1):
        with pytest.raises(RuntimeError, match="exceeds the cache"):
            m(H(qkv), H(g), H(g), I([pos]), I([0, 1, 2]))


# --- attention -------------------------------------------------------------------------------------------------------


def _attn_extra(q, k, v, hd, n):
    qn = np.linalg.norm(d(q).reshape(q.shape[0], -1, hd), axis=-1).max()
    kn = np.linalg.norm(d(k).reshape(k.shape[0], -1, hd), axis=-1).max()
    ds = 2.0 ** -24 * np.sqrt(hd) * qn * kn / np.sqrt(hd)
    return (2 * ds + 2.0 ** -24 * (4 * np.sqrt(n) + 32)) * np.abs(d(v)).max()


def _softmax_av(qh, kh, vh, mask=None):
    s = qh @ kh.T / np.sqrt(qh.shape[-1])
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ vh


def _window_ref(q, k, v, heads, window):
    s, dd = q.shape
    hd = dd // heads
    out = np.zeros((s, dd))
    for w0 in range(0, s, window):
        w1 = min(s, w0 + window)
        for h in range(heads):
            sl = slice(h * hd, (h + 1) * hd)
            out[w0:w1, sl] = _softmax_av(d(q[w0:w1, sl]), d(k[w0:w1, sl]), d(v[w0:w1, sl]))
    return out


def _window_case(s, heads, dd, window):
    r = np.random.default_rng(s + dd + window)
    qkv = np.concatenate([rnd(r, s, dd, scale=1.5), rnd(r, s, dd, scale=1.5), rnd(r, s, dd)], 1)
    q, k, v = qkv[:, :dd], qkv[:, dd:2 * dd], qkv[:, 2 * dd:]
    got = F.attention_window_qkv_opt(H(qkv), heads, window).numpy()
    ref, extra = _window_ref(q, k, v, heads, window), _attn_extra(q, k, v, dd // heads, min(s, window))
    check(got, ref, extra, "attention_window_qkv_opt")
    oracle = F.attention_window(H(q), H(k), H(v), heads, window).numpy()
    check_oracle(got, oracle, ref, extra, extra, "attention_window_qkv_opt")


@pytest.mark.parametrize("s,heads,dd,window", [
    (1, 16, 1024, 104), (7, 16, 1024, 104), (104, 16, 1024, 104), (105, 16, 1024, 104), (208, 16, 1024, 104),
    (1000, 16, 1024, 104), (3121, 16, 1024, 104), (10, 2, 16, 3), (9, 1, 8, 9), (5, 2, 512, 2), (300, 2, 64, 130),
    (40, 4, 32, 40)])
def test_attention_window_qkv_opt(s, heads, dd, window):
    """Encoder self-attention on the fused QKV rows: windows of 104 (last one partial; 3121 = 30 windows + 1 row),
    windows longer than one 128-query block (130), odd head layouts."""
    _window_case(s, heads, dd, window)


@pytest.mark.slow
def test_attention_window_qkv_opt_full():
    """The longest encoder input: S = 15,600 (150 windows)."""
    _window_case(15600, 16, 1024, 104)


def _gqa_ref(kall, vall, q, p0, q_heads, kv_heads):
    t, dq = q.shape
    hd = dq // q_heads
    group = q_heads // kv_heads
    out = np.zeros((t, dq))
    n = p0 + t
    mask = np.arange(n)[None, :] <= p0 + np.arange(t)[:, None]
    for h in range(q_heads):
        kv = slice((h // group) * hd, (h // group + 1) * hd)
        sl = slice(h * hd, (h + 1) * hd)
        out[:, sl] = _softmax_av(d(q[:, sl]), d(kall[:n, kv]), d(vall[:n, kv]), mask)
    return out


def _gqa_case(p0, t, cap, q_heads=16, kv_heads=8, hd=128):
    r = np.random.default_rng(p0 * 3 + t + cap)
    kc, vc = np.zeros((cap, kv_heads * hd), np.float16), np.zeros((cap, kv_heads * hd), np.float16)
    kc[:p0 + t], vc[:p0 + t] = rnd(r, p0 + t, kv_heads * hd, scale=1.5), rnd(r, p0 + t, kv_heads * hd)
    kc[p0 + t:], vc[p0 + t:] = np.float16(1000), np.float16(-1000)  # rows past the causal limit must not matter
    q = rnd(r, t, q_heads * hd, scale=1.5)
    got = F.attention_gqa_opt(H(q), H(kc), H(vc), I([p0]), q_heads, kv_heads).numpy()
    ref = _gqa_ref(kc, vc, q, p0, q_heads, kv_heads)
    extra = _attn_extra(q, kc[:p0 + t], vc[:p0 + t], hd, p0 + t)
    check(got, ref, extra, "attention_gqa_opt")
    oracle = F.attention_gqa(H(q), H(kc), H(vc), I([p0]), q_heads, kv_heads).numpy()
    check_oracle(got, oracle, ref, extra, extra, "attention_gqa_opt")


GQA_CASES = [  # (pos, T, cap): decode (key split), prompts, a later chunk, long context
    (0, 1, 64), (5, 1, 64), (63, 1, 64), (0, 44, 64), (37, 27, 64), (0, 401, 512), (1850, 1, 2048), (300, 3, 512),
    (2048, 300, 4096), (19999, 1, 20000)]


@pytest.mark.parametrize("p0,t,cap", GQA_CASES)
def test_attention_gqa_opt(p0, t, cap):
    """Causal GQA over the f16 caches (16 q heads over 8 kv heads, hd 128); rows past pos + T hold garbage."""
    _gqa_case(p0, t, cap)


@pytest.mark.slow
@pytest.mark.parametrize("p0,t,cap", [(0, 2048, 2048), (20000, 1, 25600), (20000, 64, 25600), (18000, 2048, 25600)])
def test_attention_gqa_opt_long(p0, t, cap):
    """A full 2048-row chunk, and the long-context cache (P = 25,600, pos 20,000)."""
    _gqa_case(p0, t, cap)


def test_attention_gqa_opt_odd_heads():
    """Other head layouts: 6 q heads over 2 kv heads (hd 16), MHA (hd 8), 4 over 1 (hd 32), 8 q per kv head."""
    for qh, kvh, hd, p0, t in [(6, 2, 16, 9, 4), (3, 3, 8, 0, 5), (4, 1, 32, 2, 1), (8, 1, 64, 700, 1), (8, 1, 64, 0, 9)]:
        _gqa_case(p0, t, p0 + t + 3, qh, kvh, hd)


def test_attention_gqa_opt_bounds():
    r = np.random.default_rng(4)
    kc = rnd(r, 16, 128)
    with pytest.raises(RuntimeError, match="exceeds the cache"):
        F.attention_gqa_opt(H(rnd(r, 3, 256)), H(kc), H(kc), I([14]), 2, 1)


# --- audio CNN -------------------------------------------------------------------------------------------------------


def _cnn_weights(seed):
    r = np.random.default_rng(seed)
    w1 = rnd(r, 480, 3, 3, 1, scale=np.sqrt(2 / 9))
    w2, w3 = (rnd(r, 480, 3, 3, 480, scale=np.sqrt(2 / 4320)) for _ in range(2))
    b1, b2, b3 = (rnd(r, 480, scale=0.1) for _ in range(3))
    w_out = rnd(r, 7680, 1024, scale=1 / np.sqrt(7680))
    return w1, b1, w2, b2, w3, b3, w_out


def _cnn_ref(feats, w1, b1, w2, b2, w3, b3, w_out):
    """Per chunk, float64, f16 rounding of the input and between layers (as the reference model)."""
    def conv(x, w, b):  # x [1, Ci, H, W]; w [Co, kt, kf, Ci]
        y = torch.nn.functional.conv2d(x, torch.from_numpy(d(w)).permute(0, 3, 1, 2), torch.from_numpy(d(b)),
                                       stride=2, padding=1)
        return torch.from_numpy(d(torch.nn.functional.gelu(y).numpy().astype(np.float16)))

    outs, a3s = [], []
    for c in range(feats.shape[0]):
        x = torch.from_numpy(d(feats[c].T.astype(np.float16)))[None, None]
        a3 = conv(conv(conv(x, w1, b1), w2, b2), w3, b3)[0].permute(1, 2, 0).reshape(13, 7680).numpy()
        a3s.append(a3)
        outs.append(a3 @ d(w_out))
    return np.stack(outs), np.stack(a3s)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_audio_cnn_opt(n):
    """Same weights and semantics as audio_cnn; chunks independent. Oracle: audio_cnn."""
    ws = _cnn_weights(n)
    feats = np.random.default_rng(n + 20).uniform(-1, 1.5, (n, 128, 100)).astype(np.float32)
    args = [lk.from_numpy(feats), *(H(w) for w in ws)]
    got = F.audio_cnn_opt(*args).numpy()
    assert got.shape == (n, 13, 1024) and got.dtype == np.float16
    ref, a3 = _cnn_ref(feats, *ws)
    extra = 2 * 2.0 ** -24 * np.sqrt(7680) * (np.abs(a3) @ np.abs(d(ws[-1]))) + U * np.sqrt((ref ** 2).mean())
    check(got, ref, extra, "audio_cnn_opt")
    check_oracle(got, F.audio_cnn(*args).numpy(), ref, extra, extra, "audio_cnn_opt")
    if n > 1:
        exact(F.audio_cnn_opt(lk.from_numpy(feats[1:2].copy()), *(H(w) for w in ws)).numpy()[0], got[1])
