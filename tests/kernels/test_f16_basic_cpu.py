"""f16 basic CPU kernels (docs/impl-notes/qwen3-asr-f16-kernels.md) vs the float64 torch / numpy composition.

Every op runs through a one-op graph (eager call, or a compiled graph for the dim / cache ops); inputs are drawn as f16
and upcast for the reference. Real Qwen3-ASR shapes, odd sizes, length 1, chunk / window boundaries, and the
long-context GQA case (cap 25,600, pos 20,000). Also runs on the iPhone (`pytest --device=iphone`).

Tolerances (u = 2⁻¹¹, the f16 unit roundoff; 2⁻²⁵ = half the f16 subnormal spacing). A kernel computes c in f32 and
rounds once, so |out - ref| <= u·|ref| + (1 + u)·|c - ref| + 2⁻²⁵, and each test bounds the f32 error |c - ref|:
- exact ops (add, add_clamp, add_pos_prefix: the f32 sum of two f16 values is exact; gathers / copies): bit-exact;
- elementwise transcendentals (gelu_erf, silu_mul): f32 error < 2⁻¹⁸·|ref| (a few f32 ulps; erfc keeps GELU's
  negative tail accurate);
- reductions over K (gemm, lm_head_last): 2·2⁻²⁴·√K·(|x|@|w|), the random-walk size of a sequential f32 sum;
- layernorm: 2⁻¹⁶·|g|·(1 + |x̂|) (f32 statistics over C <= 4096 are good to ~√C·2⁻²⁴);
- rmsnorm: the reference rounds h = f16(x·rsqrt(...)) too, so a kernel h one ulp off (f32 value at a rounding
  midpoint) is allowed: + |g|·2⁻¹⁰·|h|; at least 99 % of the outputs must be bit-exact;
- rope: f32 cos / sin and products, 2⁻²⁰·(|x_j| + |x_j+64|);
- attention: score error ds = 2⁻²⁴·√hd·scale·max‖q‖·max‖k‖ and the f32 accumulation over n keys:
  (2·ds + 2⁻²⁴·(4√n + 32))·max|v|;
- audio_cnn: gemm's bound for the last GEMM plus the f16 rounding flips of the intermediate layers (the reference
  rounds between layers too, a kernel value near a rounding midpoint may land one ulp away and that random-walks
  through the next layers): + u·rms(ref) (measured: at most 0.11·u·rms beyond the GEMM bound, n = 1 and 3).
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


def check(got, ref, extra=0.0, what=""):
    """|got - ref| <= u·|ref| + (1 + u)·extra + 2⁻²⁵ elementwise (got f16, ref float64), all finite."""
    got = d(got)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.isfinite(got).all(), f"{what}: non-finite output"
    tol = U * np.abs(ref) + (1 + U) * extra + TINY
    err = np.abs(got - ref)
    if (err > tol).any():
        i = np.unravel_index(np.argmax(err - tol), err.shape)
        raise AssertionError(f"{what}: {(err > tol).sum()} / {err.size} outside tolerance; worst at {i}: got "
                             f"{got[i]!r} ref {ref[i]!r} err {err[i]:.3g} tol {np.broadcast_to(tol, err.shape)[i]:.3g}")


def exact(got, ref):
    got = np.asarray(got)
    assert got.dtype == np.float16
    np.testing.assert_array_equal(got.view(np.uint16), np.asarray(ref, np.float16).view(np.uint16))


def gemm_extra(x, w, b=None):
    k = x.shape[1]
    s = np.abs(d(x)) @ np.abs(d(w))
    if b is not None:
        s = s + np.abs(d(b))
    return 2 * 2.0 ** -24 * np.sqrt(k) * s


# --- GEMMs ---------------------------------------------------------------------------------------------------------

GEMM_SHAPES = [
    # real: conv_out, encoder FFN / attention, projector, LM (decode M = 1, prompt T = 44 / 401, full chunk 2048)
    (13, 7680, 1024), (104, 1024, 3072), (401, 1024, 4096), (401, 4096, 1024), (44, 1024, 2048), (1, 1024, 1024),
    (1, 2048, 6144), (1, 6144, 2048), (44, 2048, 2048), (44, 2048, 1024), (2048, 2048, 1024),
    # odd
    (1, 1, 1), (3, 5, 7), (5, 37, 19), (7, 100, 33), (17, 129, 47), (4, 16, 16), (6, 8, 31), (2, 3, 17),
]


@pytest.mark.parametrize("m,k,n", GEMM_SHAPES)
def test_gemm(m, k, n):
    r = np.random.default_rng(m * 7 + k * 3 + n)
    x, w, b = rnd(r, m, k), rnd(r, k, n, scale=1 / np.sqrt(k)), rnd(r, n, scale=0.5)
    ref = d(x) @ d(w)
    check(F.gemm(H(x), H(w)).numpy(), ref, gemm_extra(x, w), "gemm")
    check(F.gemm_bias(H(x), H(w), H(b)).numpy(), ref + d(b), gemm_extra(x, w, b), "gemm_bias")


@pytest.mark.parametrize("t,v,dd", [(1, 151936, 2048), (44, 4099, 2048), (3, 1000, 24), (1, 7, 5), (2, 513, 1)])
def test_lm_head_last(t, v, dd):
    """Output f32, no f16 rounding: only the f32 accumulation error (real shape: V = 151,936, D = 2048)."""
    r = np.random.default_rng(v + dd)
    x, table = rnd(r, t, dd), rnd(r, v, dd, scale=1 / np.sqrt(dd))
    got = F.lm_head_last(H(x), H(table)).numpy()
    assert got.dtype == np.float32 and got.shape == (1, v)
    ref = (d(table) @ d(x[-1]))[None]
    tol = 2 * 2.0 ** -24 * np.sqrt(dd) * (np.abs(d(table)) @ np.abs(d(x[-1])))[None] + 1e-30
    assert (np.abs(d(got) - ref) <= tol).all(), np.abs(d(got) - ref).max()


# --- elementwise ---------------------------------------------------------------------------------------------------

EW_SHAPES = [(1, 1), (3, 7), (13, 1024), (401, 4096), (5, 6144)]


def _specials(x):
    """Plant special values: zeros, subnormals, tiny / huge magnitudes."""
    flat = x.reshape(-1)
    sp = np.array([0, -0.0, 6e-8, -6e-8, 1e-5, -3e-5, 1, -1, 4, -4, 6, -6, 10, -10, 30, -30, 60000, -60000],
                  np.float16)
    flat[: min(len(sp), flat.size)] = sp[: flat.size]
    return x


@pytest.mark.parametrize("shape", EW_SHAPES)
def test_gelu_erf(shape):
    x = _specials(rnd(np.random.default_rng(shape[0]), *shape, scale=3))
    xd = torch.from_numpy(d(x))
    ref = torch.nn.functional.gelu(xd).numpy()
    check(F.gelu_erf(H(x)).numpy(), ref, 2.0 ** -18 * np.abs(ref), "gelu_erf")


@pytest.mark.parametrize("shape", EW_SHAPES)
def test_silu_mul(shape):
    r = np.random.default_rng(shape[1])
    g, u = _specials(rnd(r, *shape, scale=3)), rnd(r, *shape, scale=2)
    u.reshape(-1)[:18] = np.float16(0.75)  # g = ±60000 at the specials: keep g·u finite in f16
    ref = (torch.nn.functional.silu(torch.from_numpy(d(g))) * torch.from_numpy(d(u))).numpy()
    check(F.silu_mul(H(g), H(u)).numpy(), ref, 2.0 ** -18 * np.abs(ref), "silu_mul")


@pytest.mark.parametrize("shape", EW_SHAPES)
def test_add_and_clamp(shape):
    """f32 sum of two f16 values is exact: bit-exact vs the f64 sum rounded once; the clamp (lim 64504, rounds to
    64512 in f16) also on sums that overflow f16."""
    r = np.random.default_rng(shape[0] + 1)
    x, y = rnd(r, *shape, scale=100), rnd(r, *shape, scale=100)
    big = r.random(shape) < 0.2
    x[big] = np.float16(60000) * np.sign(x[big])
    y[big] = np.float16(40000) * np.sign(x[big])
    exact(F.add_clamp(H(x), H(y)).numpy(), np.clip(d(x) + d(y), -64504, 64504))
    exact(F.add_clamp(H(x), H(y), 300.5).numpy(), np.clip(d(x) + d(y), -300.5, 300.5))
    x[big], y[big] = 0, 0
    exact(F.add(H(x), H(y)).numpy(), d(x) + d(y))


# --- normalization ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows,c", [(1, 1024), (104, 1024), (3, 7), (15, 1000), (1, 1), (2, 4096)])
def test_layernorm(rows, c):
    r = np.random.default_rng(rows * c)
    x, g, b = rnd(r, rows, c, scale=3, loc=1), rnd(r, c), rnd(r, c)
    xd = d(x)
    m = xd.mean(-1, keepdims=True)
    xh = (xd - m) / np.sqrt(((xd - m) ** 2).mean(-1, keepdims=True) + 1e-5)
    ref = xh * d(g) + d(b)
    check(F.layernorm(H(x), H(g), H(b)).numpy(), ref, 2.0 ** -16 * np.abs(d(g)) * (1 + np.abs(xh)), "layernorm")


def _rms_ref(x, g, width, eps=1e-6):
    xd = d(x).reshape(x.shape[0], -1, width)
    h = (xd / np.sqrt((xd ** 2).mean(-1, keepdims=True) + eps)).astype(np.float16)
    return (d(h) * d(g)).reshape(x.shape), d(h).reshape(x.shape)


def _check_rms(got, x, g, width):
    ref, h = _rms_ref(x, g, width)
    gt = np.tile(d(g), x.shape[1] // width)
    check(got, ref, np.abs(gt) * 2.0 ** -10 * np.abs(h), "rmsnorm")
    same = (np.asarray(got, np.float16) == ref.astype(np.float16)).mean()
    assert same >= 0.99, f"only {same:.4f} bit-exact"


@pytest.mark.parametrize("rows,c", [(1, 2048), (44, 2048), (401, 2048), (3, 7), (2, 1), (1, 1024)])
def test_rmsnorm(rows, c):
    r = np.random.default_rng(rows + c)
    x, g = rnd(r, rows, c, scale=4), rnd(r, c)
    _check_rms(F.rmsnorm(H(x), H(g)).numpy(), x, g, c)


@pytest.mark.parametrize("rows,c", [(1, 2048), (44, 1024), (401, 2048), (5, 384), (1, 128)])
def test_rmsnorm_heads(rows, c):
    r = np.random.default_rng(rows * 3 + c)
    x, g = rnd(r, rows, c, scale=4), rnd(r, 128)
    x[:, :128] *= np.float16(0.01)  # heads of very different scale
    _check_rms(F.rmsnorm_heads(H(x), H(g)).numpy(), x, g, 128)


# --- RoPE ------------------------------------------------------------------------------------------------------------


def _rope_ref(x, pos, theta=1e6):
    """HF Qwen3RotaryEmbedding: inv_freq and the angle products in f32 (the golden runs them in f32 too); cos / sin
    and the rotation in float64."""
    inv = 1.0 / (theta ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))
    p = torch.arange(pos, pos + x.shape[0], dtype=torch.int64).float()
    ang = d((p[:, None] * inv[None]).numpy())
    cs, sn = np.cos(ang), np.sin(ang)
    xd = d(x).reshape(x.shape[0], -1, 128)
    lo, hi = xd[..., :64], xd[..., 64:]
    y = np.concatenate([lo * cs[:, None] - hi * sn[:, None], hi * cs[:, None] + lo * sn[:, None]], -1)
    mag = np.concatenate([np.abs(lo) + np.abs(hi)] * 2, -1)
    return y.reshape(x.shape), mag.reshape(x.shape)


@pytest.mark.parametrize("t,c,pos", [(1, 2048, 0), (1, 1024, 19999), (44, 2048, 0), (64, 1024, 20000), (401, 2048, 7),
                                     (2048, 1024, 0), (3, 128, 25597)])
def test_rope(t, c, pos):
    x = rnd(np.random.default_rng(t + pos), t, c, scale=2)
    ref, mag = _rope_ref(x, pos)
    check(F.rope(H(x), I([pos])).numpy(), ref, 2.0 ** -20 * mag, "rope")


# --- gathers ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,rows", [(1, 13), (1, 7), (1, 1), (2, 26), (2, 14), (3, 32), (3, 39)])
def test_add_pos_prefix(n, rows):
    """Positions restart every 13-row chunk; rows from dim_read (N_a); bit-exact (exact f32 sum, one rounding)."""
    r = np.random.default_rng(n * 100 + rows)
    x, pe = rnd(r, n, 13, 1024, scale=20), rnd(r, 13, 1024)
    model = lk.compile(lambda x_, p_, na: F.add_pos_prefix(x_, p_, lk.dim_read(na)),
                       input_types=[lk.TensorType(lk.f16, [lk.Sym("n"), 13, 1024]), lk.TensorType(lk.f16, [13, 1024]),
                                    lk.TensorType(lk.i64, [1])])
    got = model(H(x), H(pe), I([rows])).numpy()
    exact(got, (d(x) + d(pe)[None]).reshape(13 * n, 1024)[:rows])
    with pytest.raises(RuntimeError, match="rows exceeds"):
        model(H(x), H(pe), I([13 * n + 1]))


def test_embedding():
    r = np.random.default_rng(5)
    table = rnd(r, 151936, 64)
    ids = np.array([0, 151935, 151676, 7, 7, 1000], np.int64)
    model = lk.compile(lambda i, t: F.embedding(i, t), input_types=[lk.TensorType(lk.i64, [lk.Sym("T")]),
                                                                    lk.TensorType(lk.f16, [151936, 64])])
    exact(model(I(ids), H(table)).numpy(), table[ids])
    exact(model(I(ids[:1]), H(table)).numpy(), table[ids[:1]])
    for bad in ([151936], [-1], [3, 10 ** 9]):
        with pytest.raises(RuntimeError, match="out of range"):
            model(I(bad), H(table))


def test_scatter_audio():
    r = np.random.default_rng(6)
    pad = 151676
    model = lk.compile(lambda x, i, a: F.scatter_audio(x, i, a),
                       input_types=[lk.TensorType(lk.f16, [lk.Sym("T"), 2048]), lk.TensorType(lk.i64, [lk.Sym("T")]),
                                    lk.TensorType(lk.f16, [lk.Sym("A"), 2048])])
    for t, a, npad in [(25, 7, 7), (44, 13, 13), (10, 5, 3), (4, 2, 0), (1, 1, 1), (1, 3, 0)]:
        x, audio = rnd(r, t, 2048), rnd(r, a, 2048)
        ids = r.integers(0, 151000, t)
        slots = np.sort(r.choice(t, npad, replace=False))
        ids[slots] = pad
        ref = x.copy()
        ref[slots] = audio[:npad]
        exact(model(H(x), I(ids), H(audio)).numpy(), ref)
    x, audio, ids = rnd(r, 5, 2048), rnd(r, 2, 2048), np.full(5, pad)
    with pytest.raises(RuntimeError, match="more audio pad ids"):
        model(H(x), I(ids), H(audio))


# --- attention -------------------------------------------------------------------------------------------------------


def _attn_extra(q, k, v, hd, n):
    """f32 error bound of a streamed attention row (see the module docstring), per output column block."""
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


@pytest.mark.parametrize("s,heads,dd,window", [
    (1, 16, 1024, 104), (7, 16, 1024, 104), (104, 16, 1024, 104), (105, 16, 1024, 104), (208, 16, 1024, 104),
    (1000, 16, 1024, 104), (15600, 16, 1024, 104), (10, 2, 16, 3), (9, 1, 8, 9), (5, 2, 512, 2)])
def test_attention_window(s, heads, dd, window):
    """Encoder self-attention: block-diagonal windows of 104 (last window partial), real S up to 15,600."""
    r = np.random.default_rng(s + dd)
    q, k, v = rnd(r, s, dd, scale=1.5), rnd(r, s, dd, scale=1.5), rnd(r, s, dd)
    got = F.attention_window(H(q), H(k), H(v), heads, window).numpy()
    hd = dd // heads
    check(got, _window_ref(q, k, v, heads, window), _attn_extra(q, k, v, hd, min(s, window)), "attention_window")


def _cache_model(cap, dk=1024, dq=2048, q_heads=16, kv_heads=8):
    """caches (persistent f16[cap, dk]) <- hist k / v at pos 0, then the chunk's k / v at pos (kv_write), then
    attention_gqa of the chunk's queries. Returns the attention output."""
    P, T = lk.Sym("P"), lk.Sym("T")

    def fn(hk, hv, zero, k, v, q, pos):
        kc = lk.persistent_buffer(lk.f16, [cap, dk])
        vc = lk.persistent_buffer(lk.f16, [cap, dk])
        F.kv_write(kc, vc, hk, hv, zero)
        F.kv_write(kc, vc, k, v, pos)
        return F.attention_gqa(q, kc, vc, pos, q_heads, kv_heads)

    f = lambda shape: lk.TensorType(lk.f16, shape)  # noqa: E731
    return lk.compile(fn, input_types=[f([P, dk]), f([P, dk]), lk.TensorType(lk.i64, [1]), f([T, dk]), f([T, dk]),
                                       f([T, dq]), lk.TensorType(lk.i64, [1])])


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


GQA_CASES = [  # (pos, T, cap): decode, prompt, full prefill chunk, a later chunk, long context (cap 25,600)
    (0, 1, 64), (5, 1, 64), (0, 44, 64), (37, 27, 64), (63, 1, 64), (0, 401, 512), (1850, 1, 2048), (0, 2048, 2048),
    (2048, 300, 4096), (20000, 1, 25600), (20000, 64, 25600)]


@pytest.mark.parametrize("p0,t,cap", GQA_CASES)
def test_kv_write_attention_gqa(p0, t, cap):
    """Causal GQA over the f16 KV cache (16 q heads over 8 kv heads, hd 128), the caches written by kv_write."""
    r = np.random.default_rng(p0 * 3 + t)
    kall, vall = rnd(r, p0 + t, 1024, scale=1.5), rnd(r, p0 + t, 1024)
    q = rnd(r, t, 2048, scale=1.5)
    hk, hv = (kall[:p0], vall[:p0]) if p0 else (kall[:1], vall[:1])  # P >= 1: a dummy row 0, overwritten at pos 0
    got = _cache_model(cap)(H(hk), H(hv), I([0]), H(kall[p0:]), H(vall[p0:]), H(q), I([p0])).numpy()
    check(got, _gqa_ref(kall, vall, q, p0, 16, 8), _attn_extra(q, kall, vall, 128, p0 + t), "attention_gqa")


def test_attention_gqa_odd_heads():
    """Other head layouts: 6 q heads over 2 kv heads, hd 16; MHA (q_heads == kv_heads)."""
    r = np.random.default_rng(11)
    for qh, kvh, hd, p0, t in [(6, 2, 16, 9, 4), (3, 3, 8, 0, 5), (4, 1, 32, 2, 1)]:
        kall, vall, q = rnd(r, p0 + t, kvh * hd), rnd(r, p0 + t, kvh * hd), rnd(r, t, qh * hd)
        m = _cache_model(16, dk=kvh * hd, dq=qh * hd, q_heads=qh, kv_heads=kvh)
        hk, hv = (kall[:p0], vall[:p0]) if p0 else (kall[:1], vall[:1])
        got = m(H(hk), H(hv), I([0]), H(kall[p0:]), H(vall[p0:]), H(q), I([p0])).numpy()
        check(got, _gqa_ref(kall, vall, q, p0, qh, kvh), _attn_extra(q, kall, vall, hd, p0 + t), "gqa odd")


def test_kv_write_rows_and_bounds():
    """kv_write copies rows exactly at pos .. pos+T-1; pos + T > cap (or pos < 0) is a run-time error, also for
    attention_gqa."""
    cap = 16

    def fn(k, v, pos, rows):
        kc = lk.persistent_buffer(lk.f16, [cap, 64])
        vc = lk.persistent_buffer(lk.f16, [cap, 64])
        F.kv_write(kc, vc, k, v, pos)
        # Read the caches through a gather: a persistent buffer returned directly as a graph output does not keep
        # its contents across calls (the runtime's output handling; seen with the f32 kv_write_rows too).
        return F.embedding(rows, kc), F.embedding(rows, vc)

    T = lk.Sym("T")
    m = lk.compile(fn, input_types=[lk.TensorType(lk.f16, [T, 64]), lk.TensorType(lk.f16, [T, 64]),
                                    lk.TensorType(lk.i64, [1]), lk.TensorType(lk.i64, [cap])])
    rows = I(np.arange(cap))
    r = np.random.default_rng(12)
    k1, v1 = rnd(r, 5, 64), rnd(r, 5, 64)
    m(H(k1), H(v1), I([0]), rows)
    k2, v2 = rnd(r, 3, 64), rnd(r, 3, 64)
    kc, vc = m(H(k2), H(v2), I([13]), rows)  # the last rows of the cache
    exact(kc.numpy()[:5], k1)
    exact(vc.numpy()[:5], v1)
    exact(kc.numpy()[13:], k2)
    exact(vc.numpy()[13:], v2)
    for pos in (14, -1):
        with pytest.raises(RuntimeError, match="exceeds the cache"):
            m(H(k2), H(v2), I([pos]), rows)
    a = _cache_model(16, dk=64, dq=128, q_heads=2, kv_heads=1)
    with pytest.raises(RuntimeError, match="exceeds the cache"):
        a(H(k1[:1]), H(v1[:1]), I([0]), H(k2), H(v2), H(rnd(r, 3, 128)), I([14]))


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
        x = torch.from_numpy(d(feats[c].T.astype(np.float16)))[None, None]  # [1, 1, 100 time, 128 freq]
        a3 = conv(conv(conv(x, w1, b1), w2, b2), w3, b3)[0].permute(1, 2, 0).reshape(13, 7680).numpy()
        a3s.append(a3)
        outs.append(a3 @ d(w_out))
    return np.stack(outs), np.stack(a3s)


@pytest.mark.parametrize("n", [1, 3])
def test_audio_cnn(n):
    """Chunks are independent (each chunk's output equals a 1-chunk run of it); real weight shapes."""
    ws = _cnn_weights(n)
    feats = np.random.default_rng(n + 20).uniform(-1, 1.5, (n, 128, 100)).astype(np.float32)
    got = F.audio_cnn(lk.from_numpy(feats), *(H(w) for w in ws)).numpy()
    assert got.shape == (n, 13, 1024) and got.dtype == np.float16
    ref, a3 = _cnn_ref(feats, *ws)
    k = 7680
    lin = 2 * 2.0 ** -24 * np.sqrt(k) * (np.abs(a3) @ np.abs(d(ws[-1])))
    flips = U * np.sqrt((ref ** 2).mean())
    check(got, ref, lin + flips, "audio_cnn")
    if n > 1:
        exact(F.audio_cnn(lk.from_numpy(feats[1:2].copy()), *(H(w) for w in ws)).numpy()[0], got[1])
