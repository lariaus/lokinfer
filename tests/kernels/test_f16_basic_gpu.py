"""f16 basic GPU kernels (Qwen3-ASR phase 1, docs/impl-notes/qwen3-asr-f16-kernels.md) vs the float64 torch
composition of the contract's semantics: each op through a one-op graph (eager call, or a small compiled graph for
the persistent KV caches and the dim_read-driven row count), inputs drawn as f16 and upcast for the reference. Real
model shapes + odd sizes + length 1 + the chunk / window / cache boundaries. Runs on the iPhone with
`pytest --device=iphone`.

Tolerances (per element, |got - ref| <= tol):
- U = 2^-11 is f16's unit roundoff: rounding the exact result once to f16 costs <= U * |ref| (normal range), plus half
  the subnormal spacing (2^-25) near zero. Every check allows that (times 1 + 2^-10 for the float error that can push
  a value across a rounding midpoint), plus an op-specific float32 term:
  - elementwise (add, add_clamp, silu_mul, gelu_erf): float math on exact f16 inputs, <= a few 2^-24 relative ->
    2^-20 * |ref|;
  - GEMM / LM head: float accumulation over K, error ~ sqrt(K) * 2^-24 * sqrt(sum_k (x_k w_k)^2) -> 2^-22 * sqrt(K) *
    that root-sum-square (4x the typical size; measured: <= 0.3 of this term);
  - layernorm: float statistics -> 2^-18 * (|g * xhat| + |b|);
  - rmsnorm: the contract rounds h to f16 before the weight multiply; the float h may round to the other neighbour
    than the float64 h -> one ulp of h, 2^-10 * |g * h|;
  - rope: angle reduction + float cos / sin -> 2^-20 * (|x_j| + |x_j+64|) (measured: up to 0.85 of it; the angle
    itself is the contract's float product, reproduced exactly by the reference);
  - attention: softmax weights and the weighted sum in float over P keys, ~ (8 + sqrt(P)) * 2^-24 * max|v| ->
    2^-24 * 4 * (8 + sqrt(P)) * max|v| (measured: <= 0.2 of this term);
  - audio_cnn: each conv layer's f16 output may round to the other neighbour (float vs float64 sum near a midpoint,
    rare); one such ulp propagates through the next layers -> 2^-9 * rms(ref) on top of the GEMM terms (measured:
    up to 0.26 of it).
"""
import numpy as np
import pytest
import torch

import lokinfer as lk
from lokinfer.ops import f16_gpu as F

U = 2.0 ** -11
SUB = 2.0 ** -25


def G(a):
    return lk.from_numpy(np.ascontiguousarray(a), device="gpu", copy=True)


def C(a):
    return lk.from_numpy(np.ascontiguousarray(a))


def P(p):
    return C(np.array([p], np.int64))


def rng(seed):
    return np.random.default_rng(seed)


def h16(r, *shape, scale=1.0, mean=0.0):
    return (mean + scale * r.standard_normal(shape)).astype(np.float16)


def t64(a):
    return torch.from_numpy(np.asarray(a, np.float64))


def r16(t):
    """Rounds a float64 tensor to f16 and back (the contract's intermediate roundings)."""
    return t.to(torch.float16).to(torch.float64)


def check_f16(got, ref, extra=0.0, what=""):
    got = np.asarray(got.numpy() if hasattr(got, "numpy") and not isinstance(got, np.ndarray) else got, np.float64)
    ref = np.asarray(ref.numpy() if isinstance(ref, torch.Tensor) else ref, np.float64)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.isfinite(got).all(), f"{what}: non-finite output"
    tol = U * (1 + 2.0 ** -10) * np.abs(ref) + extra + SUB
    err = np.abs(got - ref)
    bad = err > tol
    if bad.any():
        i = np.unravel_index(np.argmax(err - tol), err.shape)
        raise AssertionError(f"{what}: {bad.sum()} / {bad.size} elements beyond tolerance; worst at {i}: got "
                             f"{got[i]!r} ref {ref[i]!r} err {err[i]:.3g} tol {np.broadcast_to(tol, err.shape)[i]:.3g}")


# --- elementwise ----------------------------------------------------------------------------------------------------

ELT_SHAPES = [(1, 1), (3, 7), (13, 1024), (44, 6144), (1, 2048), (257, 4096)]


@pytest.mark.parametrize("shape", ELT_SHAPES)
def test_add(shape):
    r = rng(sum(shape))
    x, y = h16(r, *shape, scale=4), h16(r, *shape, scale=4)
    check_f16(F.add_f16_gpu(G(x), G(y)), t64(x) + t64(y), what="add")


@pytest.mark.parametrize("shape", ELT_SHAPES)
def test_add_clamp(shape):
    """Sums beyond +-64504 are clamped (the encoder's clamp; f16(64504) = 64512, as torch's f16 clamp gives)."""
    r = rng(sum(shape) + 1)
    x = np.clip(40000 * r.standard_normal(shape), -65000, 65000).astype(np.float16)
    y = np.clip(40000 * r.standard_normal(shape), -65000, 65000).astype(np.float16)
    ref = torch.clamp(t64(x) + t64(y), -64504, 64504)
    got = F.add_clamp_f16_gpu(G(x), G(y))
    check_f16(got, ref, what="add_clamp")
    assert np.abs(got.numpy().astype(np.float64)).max() <= 64512


@pytest.mark.parametrize("shape", ELT_SHAPES)
def test_silu_mul(shape):
    r = rng(sum(shape) + 2)
    g, u = h16(r, *shape, scale=3), h16(r, *shape, scale=2)
    ref = torch.nn.functional.silu(t64(g)) * t64(u)
    check_f16(F.silu_mul_f16_gpu(G(g), G(u)), ref, extra=2.0 ** -20 * ref.abs().numpy(), what="silu_mul")


@pytest.mark.parametrize("shape", ELT_SHAPES)
def test_gelu_erf(shape):
    r = rng(sum(shape) + 3)
    x = h16(r, *shape, scale=3)
    ref = torch.nn.functional.gelu(t64(x))  # exact erf
    check_f16(F.gelu_erf_f16_gpu(G(x)), ref, extra=2.0 ** -20 * ref.abs().numpy(), what="gelu_erf")


def test_gelu_erf_range():
    """Every f16 value in [-16, 16]: the negative tail (1 + erf cancels, tiny / subnormal results), around 0, large."""
    neg = np.arange(0x8001, int(np.float16(-16).view(np.uint16)) + 1, dtype=np.uint16)  # -2^-24 .. -16
    pos = np.arange(0, int(np.float16(16).view(np.uint16)) + 1, dtype=np.uint16)  # 0 .. 16
    x = np.concatenate([neg, pos]).view(np.float16)
    x = np.ascontiguousarray(np.pad(x, (0, (-len(x)) % 64))).reshape(-1, 64)
    ref = torch.nn.functional.gelu(t64(x))
    check_f16(F.gelu_erf_f16_gpu(G(x)), ref, extra=2.0 ** -20 * ref.abs().numpy(), what="gelu_erf range")


# --- norms ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows,c", [(1, 1024), (15, 1024), (1, 1), (5, 7), (3, 2048), (700, 1024), (2, 4097)])
def test_layernorm(rows, c):
    r = rng(rows * 3 + c)
    x = h16(r, rows, c, scale=2, mean=3)  # an offset mean: the variance must come from the centered values
    g, b = h16(r, c, scale=0.5, mean=1), h16(r, c, scale=0.3)
    xd = t64(x)
    xhat = (xd - xd.mean(-1, keepdim=True)) / torch.sqrt(xd.var(-1, unbiased=False, keepdim=True) + 1e-5)
    ref = xhat * t64(g) + t64(b)
    extra = 2.0 ** -18 * ((xhat * t64(g)).abs() + t64(b).abs()).numpy()
    check_f16(F.layernorm_f16_gpu(G(x), G(g), G(b), eps=1e-5), ref, extra=extra, what="layernorm")


def _rms_ref(x, g, w, eps=1e-6):
    xd = t64(x).reshape(x.shape[0], -1, w)
    hh = r16(xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps))
    y = (t64(g) * hh).reshape(x.shape)
    return y, (2.0 ** -10 * (t64(g) * hh).abs()).reshape(x.shape).numpy()


@pytest.mark.parametrize("rows,c", [(1, 2048), (44, 2048), (1, 1), (3, 7), (401, 2048), (2, 1024)])
def test_rmsnorm(rows, c):
    r = rng(rows + c)
    x, g = h16(r, rows, c, scale=3), h16(r, c, scale=0.5, mean=1)
    ref, extra = _rms_ref(x, g, c)
    check_f16(F.rmsnorm_f16_gpu(G(x), G(g), eps=1e-6), ref, extra=extra, what="rmsnorm")


@pytest.mark.parametrize("rows,heads", [(1, 16), (44, 16), (1, 1), (3, 8), (401, 8), (2048, 16)])
def test_rmsnorm_heads(rows, heads):
    r = rng(rows * 5 + heads)
    x, g = h16(r, rows, heads * 128, scale=3), h16(r, 128, scale=0.5, mean=1)
    x[0, :128] *= np.float16(0.01)  # heads of very different scales: each is normalized on its own
    ref, extra = _rms_ref(x, g, 128)
    check_f16(F.rmsnorm_heads_f16_gpu(G(x), G(g), eps=1e-6), ref, extra=extra, what="rmsnorm_heads")


# --- RoPE -----------------------------------------------------------------------------------------------------------


def _rope_ref(x, pos, theta=1e6):
    t, c = x.shape
    inv = (theta ** (-2.0 * np.arange(64) / 128)).astype(np.float32)  # correctly rounded float inv_freq
    positions = (pos + np.arange(t)).astype(np.float32)
    angle = (positions[:, None] * inv[None, :]).astype(np.float64)  # the contract's float product (one rounding)
    cos, sin = torch.from_numpy(np.cos(angle))[:, None, :], torch.from_numpy(np.sin(angle))[:, None, :]
    xd = t64(x).reshape(t, c // 128, 128)
    x0, x1 = xd[..., :64], xd[..., 64:]
    y = torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).reshape(t, c)
    extra = 2.0 ** -20 * torch.cat([x0.abs() + x1.abs()] * 2, -1).reshape(t, c)
    return y, extra.numpy()


@pytest.mark.parametrize("t,heads,pos", [(1, 16, 0), (44, 16, 0), (1, 8, 51), (401, 8, 1000), (3, 1, 7),
                                          (64, 16, 19_999), (1, 8, 25_599), (2048, 16, 13_570)])
def test_rope(t, heads, pos):
    r = rng(t + heads + pos)
    x = h16(r, t, heads * 128, scale=2)
    ref, extra = _rope_ref(x, pos)
    check_f16(F.rope_f16_gpu(G(x), P(pos), theta=1e6), ref, extra=extra, what="rope")


def test_rope_theta():
    x = h16(rng(9), 5, 256)
    ref, extra = _rope_ref(x, 3, theta=10000.0)
    check_f16(F.rope_f16_gpu(G(x), P(3), theta=10000.0), ref, extra=extra, what="rope theta")


# --- GEMM -----------------------------------------------------------------------------------------------------------

GEMM_SHAPES = [
    (13, 7680, 1024),  # conv_out
    (44, 1024, 3072), (37, 1024, 4096), (1, 4096, 1024), (300, 1024, 1024),  # encoder
    (2, 1024, 2048), (1, 1024, 1024),  # projector
    (44, 2048, 2048), (1, 2048, 1024), (17, 2048, 6144), (5, 6144, 2048), (256, 2048, 6144),  # LM
    (1, 1, 1), (3, 17, 5), (33, 31, 65), (16, 16, 16), (100, 7, 300), (1, 15, 1),  # odd sizes / tile edges
]


def _gemm_data(m, k, n, seed):
    r = rng(seed)
    x, w = h16(r, m, k), h16(r, k, n, scale=1 / np.sqrt(k))
    xd, wd = t64(x), t64(w)
    extra = 2.0 ** -22 * np.sqrt(k) * torch.sqrt((xd * xd) @ (wd * wd)).numpy()
    return x, w, xd @ wd, extra


@pytest.mark.parametrize("m,k,n", GEMM_SHAPES)
def test_gemm(m, k, n):
    x, w, ref, extra = _gemm_data(m, k, n, m * 7 + k + n)
    check_f16(F.gemm_f16_gpu(G(x), G(w)), ref, extra=extra, what="gemm")


@pytest.mark.parametrize("m,k,n", GEMM_SHAPES[::2] + [(1, 1, 1)])
def test_gemm_bias(m, k, n):
    x, w, ref, extra = _gemm_data(m, k, n, m + k * 3 + n)
    b = h16(rng(n), n)
    ref = ref + t64(b)
    check_f16(F.gemm_bias_f16_gpu(G(x), G(w), G(b)), ref, extra=extra + 2.0 ** -23 * ref.abs().numpy(),
              what="gemm_bias")


@pytest.mark.parametrize("t,d,v", [(1, 2048, 151_936), (44, 2048, 4099), (3, 64, 1000), (1, 33, 7), (2, 1, 1)])
def test_lm_head_last(t, d, v):
    r = rng(t + d + v)
    x, table = h16(r, t, d), h16(r, v, d, scale=0.05)
    got = F.lm_head_last_f16_gpu(G(x), G(table)).numpy()
    assert got.dtype == np.float32 and got.shape == (1, v)
    xl = t64(x[-1])
    ref = np.empty(v)
    extra = np.empty(v)
    for i in range(0, v, 16384):  # float64 in blocks (the full table in float64 would be 2.5 GB)
        tb = t64(table[i:i + 16384])
        ref[i:i + 16384] = (tb @ xl).numpy()
        extra[i:i + 16384] = (2.0 ** -22 * np.sqrt(d) * torch.sqrt((tb * tb) @ (xl * xl))).numpy()
    err = np.abs(got[0].astype(np.float64) - ref)
    assert (err <= extra + 2.0 ** -23 * np.abs(ref) + 1e-30).all(), f"max err {err.max():.3g}"


# --- gathers --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t,v,d", [(1, 1000, 2048), (44, 1000, 2048), (401, 151_936, 64), (2048, 151_936, 16)])
def test_embedding(t, v, d):
    r = rng(t + v)
    table = h16(r, v, d)
    ids = r.integers(0, v, t).astype(np.int64)
    ids[0], ids[-1] = v - 1, 0
    got = F.embedding_f16_gpu(C(ids), G(table)).numpy()
    np.testing.assert_array_equal(got.view(np.uint16), table[ids].view(np.uint16))


def test_embedding_rejects_out_of_range_ids():
    table = h16(rng(1), 10, 8)
    model = lk.compile(lambda i, tb: F.embedding_f16_gpu(i, lk.ops.to_gpu(tb)),
                       input_types=[lk.TensorType(lk.i64, [lk.Sym("T")]), lk.TensorType(lk.f16, [10, 8])])
    np.testing.assert_array_equal(model(C(np.array([0, 9], np.int64)), C(table)).numpy(), table[[0, 9]])
    for bad in ([10], [-1], [3, 151_936]):
        with pytest.raises(RuntimeError, match="out of range"):
            model(C(np.array(bad, np.int64)), C(table))


PAD = 151_676


def _scatter_ref(x, ids, audio):
    out = x.copy()
    rows = np.flatnonzero(ids == PAD)
    out[rows] = audio[:len(rows)]
    return out


@pytest.mark.parametrize("t,pads_at,a,d", [
    (44, (9, 39), 30, 2048),  # a prompt: text, the audio pads in one run, text
    (44, (9, 39), 35, 2048),  # more audio rows than pads (fine: only the first 30 used)
    (2048, (5, 2005), 2000, 64),  # a full prefill chunk
    (7, (0, 7), 7, 3),  # all pads
    (7, (3, 3), 2, 3),  # no pads
    (1, (0, 1), 1, 5), (1, (0, 0), 1, 5),  # length 1: pad / text
])
def test_scatter_audio(t, pads_at, a, d):
    r = rng(t + a + d)
    x, audio = h16(r, t, d), h16(r, a, d)
    ids = r.integers(0, 151_000, t).astype(np.int64)
    ids[pads_at[0]:pads_at[1]] = PAD
    got = F.scatter_audio_f16_gpu(G(x), C(ids), G(audio), pad_id=PAD).numpy()
    np.testing.assert_array_equal(got.view(np.uint16), _scatter_ref(x, ids, audio).view(np.uint16))


def test_scatter_audio_scattered_pads_and_error():
    r = rng(5)
    x, audio = h16(r, 300, 32), h16(r, 120, 32)
    ids = r.integers(0, 151_000, 300).astype(np.int64)
    ids[r.choice(300, 120, replace=False)] = PAD  # pads anywhere: the j-th pad (in order) takes audio[j]
    model = lk.compile(lambda x_, i, a_: F.scatter_audio_f16_gpu(lk.ops.to_gpu(x_), i, lk.ops.to_gpu(a_)),
                       input_types=[lk.TensorType(lk.f16, [lk.Sym("T"), 32]), lk.TensorType(lk.i64, [lk.Sym("T")]),
                                    lk.TensorType(lk.f16, [lk.Sym("A"), 32])])
    np.testing.assert_array_equal(model(C(x), C(ids), C(audio)).numpy().view(np.uint16),
                                  _scatter_ref(x, ids, audio).view(np.uint16))
    with pytest.raises(RuntimeError, match="more pad ids than audio rows"):
        model(C(x), C(ids), C(audio[:119]))


def _add_pos_ref(x, pos, rows):
    return (t64(x) + t64(pos)[None]).reshape(-1, x.shape[2])[:rows]


@pytest.mark.parametrize("n,rows", [(1, 13), (1, 1), (2, 26), (2, 20), (7, 85), (300, 3900), (300, 3891)])
def test_add_pos_prefix(n, rows):
    r = rng(n + rows)
    x, pos = h16(r, n, 13, 1024), h16(r, 13, 1024)
    check_f16(F.add_pos_prefix_f16_gpu(G(x), G(pos), rows), _add_pos_ref(x, pos, rows), what="add_pos_prefix")


def test_add_pos_prefix_dim_read():
    """rows = N_a from dim_read of an i64 input (the encoder graph's mechanism); rows > 13 n is a run-time error."""
    n_sym = lk.Sym("n")

    def fn(x, pos, n_audio):
        rows = lk.ops.dim_read(n_audio, 0, name="Na")
        return F.add_pos_prefix_f16_gpu(lk.ops.to_gpu(x), lk.ops.to_gpu(pos), rows)

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f16, [n_sym, 13, 1024]), lk.TensorType(lk.f16, [13, 1024]),
                                        lk.TensorType(lk.i64, [1])])
    r = rng(11)
    pos = h16(r, 13, 1024)
    for n, rows in [(3, 39), (3, 30), (5, 53), (1, 13)]:
        x = h16(r, n, 13, 1024)
        check_f16(model(C(x), C(pos), P(rows)), _add_pos_ref(x, pos, rows), what="add_pos_prefix dim_read")
    with pytest.raises(RuntimeError, match="rows exceeds"):
        model(C(h16(r, 2, 13, 1024)), C(pos), P(27))


# --- attention ------------------------------------------------------------------------------------------------------


def _attn_extra(v, keys):
    return 2.0 ** -24 * 4 * (8 + np.sqrt(keys)) * float(np.abs(v.astype(np.float64)).max())


def _window_ref(q, k, v, heads, window):
    s, c = q.shape
    hd = c // heads
    qd, kd, vd = (t64(a).reshape(s, heads, hd).transpose(0, 1) for a in (q, k, v))
    out = torch.empty_like(qd)
    for w0 in range(0, s, window):
        sl = slice(w0, min(s, w0 + window))
        p = torch.softmax(qd[:, sl] @ kd[:, sl].transpose(1, 2) / np.sqrt(hd), -1)
        out[:, sl] = p @ vd[:, sl]
    return out.transpose(0, 1).reshape(s, c)


@pytest.mark.parametrize("s,heads,c,window", [
    (1, 16, 1024, 104), (5, 16, 1024, 104), (104, 16, 1024, 104), (105, 16, 1024, 104), (207, 16, 1024, 104),
    (208, 16, 1024, 104), (209, 16, 1024, 104), (1000, 16, 1024, 104), (15_600, 16, 1024, 104),  # real N_a 1200 s
    (9, 2, 64, 1), (33, 2, 64, 7), (20, 4, 512, 8), (50, 3, 288, 50),  # head_dim 32 / 128 / 96, window 1
])
def test_attention_window(s, heads, c, window):
    r = rng(s + heads + window)
    q, k, v = h16(r, s, c, scale=2), h16(r, s, c, scale=2), h16(r, s, c)
    got = F.attention_window_f16_gpu(G(q), G(k), G(v), heads=heads, window=window)
    check_f16(got, _window_ref(q, k, v, heads, window), extra=_attn_extra(v, window), what="attention_window")


def _gqa_ref(q, kc, vc, pos, q_heads, kv_heads):
    t, cq = q.shape
    hd = cq // q_heads
    keys = pos + t
    kd = t64(kc[:keys]).reshape(keys, kv_heads, hd)
    vd = t64(vc[:keys]).reshape(keys, kv_heads, hd)
    qd = t64(q).reshape(t, q_heads, hd)
    mask = torch.arange(keys)[None, :] > (pos + torch.arange(t))[:, None]
    out = torch.empty(t, q_heads, hd, dtype=torch.float64)
    for h in range(q_heads):
        g = h // (q_heads // kv_heads)
        sc = (qd[:, h] @ kd[:, g].T) / np.sqrt(hd)
        sc[mask] = -torch.inf
        out[:, h] = torch.softmax(sc, -1) @ vd[:, g]
    return out.reshape(t, cq)


def _gqa_data(cap, pos, t, q_heads, kv_heads, hd, seed):
    r = rng(seed)
    q = h16(r, t, q_heads * hd, scale=2)
    kc = np.full((cap, kv_heads * hd), np.nan, np.float16)  # rows past pos + T: never read (NaN would show)
    vc = np.full((cap, kv_heads * hd), np.nan, np.float16)
    kc[:pos + t], vc[:pos + t] = h16(r, pos + t, kv_heads * hd, scale=2), h16(r, pos + t, kv_heads * hd)
    return q, kc, vc


@pytest.mark.parametrize("cap,pos,t,q_heads,kv_heads,hd", [
    (64, 0, 1, 16, 8, 128), (64, 0, 44, 16, 8, 128), (300, 7, 33, 16, 8, 128), (64, 63, 1, 16, 8, 128),
    (4196, 100, 2048, 16, 8, 128),  # a later prefill chunk (T = 2048) at pos 100
    (25_600, 20_000, 1, 16, 8, 128), (25_600, 20_000, 64, 16, 8, 128),  # long context: decode / a 64-row chunk
    (25_600, 25_536, 64, 16, 8, 128),  # the chunk ends at the cache's last row
    (40, 5, 3, 4, 1, 32), (40, 0, 9, 2, 2, 128), (17, 16, 1, 6, 3, 96),  # odd: MQA hd 32, MHA, hd 96
])
def test_attention_gqa(cap, pos, t, q_heads, kv_heads, hd):
    q, kc, vc = _gqa_data(cap, pos, t, q_heads, kv_heads, hd, cap + pos + t)
    got = F.attention_gqa_f16_gpu(G(q), G(kc), G(vc), P(pos), q_heads=q_heads, kv_heads=kv_heads)
    check_f16(got, _gqa_ref(q, kc, vc, pos, q_heads, kv_heads), extra=_attn_extra(vc[:pos + t], pos + t),
              what="attention_gqa")


def test_attention_gqa_rejects_overflow():
    q, kc, vc = _gqa_data(16, 0, 4, 16, 8, 128, 1)
    model = lk.compile(lambda q_, k_, v_, p_: F.attention_gqa_f16_gpu(lk.ops.to_gpu(q_), lk.ops.to_gpu(k_),
                                                                      lk.ops.to_gpu(v_), p_),
                       input_types=[lk.TensorType(lk.f16, [lk.Sym("T"), 2048]), lk.TensorType(lk.f16, [16, 1024]),
                                    lk.TensorType(lk.f16, [16, 1024]), lk.TensorType(lk.i64, [1])])
    check_f16(model(C(q), C(kc), C(vc), P(0)), _gqa_ref(q, kc, vc, 0, 16, 8), extra=_attn_extra(vc[:4], 4))
    for pos in (13, -1):
        with pytest.raises(RuntimeError, match="exceeds the cache"):
            model(C(q), C(kc), C(vc), P(pos))


def _kv_model(cap, d):
    """Two kv_writes into persistent caches f16[cap, d] in one call; returns copies of both caches."""
    T1, T2 = lk.Sym("T1"), lk.Sym("T2")

    def fn(k1, v1, p1, k2, v2, p2):
        kc = lk.persistent_buffer(lk.f16, [cap, d], device="gpu")
        vc = lk.persistent_buffer(lk.f16, [cap, d], device="gpu")
        F.kv_write_f16_gpu(kc, vc, lk.ops.to_gpu(k1), lk.ops.to_gpu(v1), p1)
        F.kv_write_f16_gpu(kc, vc, lk.ops.to_gpu(k2), lk.ops.to_gpu(v2), p2)
        # Persistent state can't be a graph output (bufferize rejects it): read the caches back through a copy, a
        # gather of every row.
        rows = lk.ops.to_gpu(lk.ops.arange_offset_i64_cpu(lk.from_numpy(np.zeros(1, np.int64)), cap))
        return (lk.ops.to_cpu(F.embedding_f16_gpu(rows, kc)), lk.ops.to_cpu(F.embedding_f16_gpu(rows, vc)))

    return lk.compile(fn, input_types=[lk.TensorType(lk.f16, [T1, d]), lk.TensorType(lk.f16, [T1, d]),
                                       lk.TensorType(lk.i64, [1]), lk.TensorType(lk.f16, [T2, d]),
                                       lk.TensorType(lk.f16, [T2, d]), lk.TensorType(lk.i64, [1])])


@pytest.mark.parametrize("cap,d,writes", [
    (25_600, 1024, [(0, 2048, 20_000, 64)]),  # real cap: a prefill chunk at 0, a chunk near 20k
    (25_600, 1024, [(25_599, 1, 0, 1)]),  # decode at the cache's last row, and row 0
    (6656, 1024, [(0, 44, 44, 1), (45, 1, 6000, 656)]),  # iPhone cap; the last write ends at the cache end
    (10, 3, [(0, 10, 3, 2), (9, 1, 0, 1)]),  # odd sizes; the second write overwrites rows of the first
])
def test_kv_write(cap, d, writes):
    model = _kv_model(cap, d)
    r = rng(cap + d)
    for p1, t1, p2, t2 in writes:
        k1, v1, k2, v2 = h16(r, t1, d), h16(r, t1, d), h16(r, t2, d), h16(r, t2, d)
        kc, vc = model(C(k1), C(v1), P(p1), C(k2), C(v2), P(p2))
        kc, vc = kc.numpy().view(np.uint16), vc.numpy().view(np.uint16)
        ek, ev = {}, {}
        for p, k, v in ((p1, k1, v1), (p2, k2, v2)):  # later writes win
            for i in range(len(k)):
                ek[p + i], ev[p + i] = k[i], v[i]
        rows = np.array(sorted(ek))
        np.testing.assert_array_equal(kc[rows], np.stack([ek[i] for i in rows]).view(np.uint16))
        np.testing.assert_array_equal(vc[rows], np.stack([ev[i] for i in rows]).view(np.uint16))


def test_kv_write_rejects_overflow():
    model = _kv_model(16, 8)
    k = h16(rng(2), 4, 8)
    model(C(k), C(k), P(0), C(k), C(k), P(12))  # ends exactly at the cache end: fine
    for p in (13, -1):
        with pytest.raises(RuntimeError, match="exceeds the cache"):
            model(C(k), C(k), P(0), C(k), C(k), P(p))


def test_prefill_then_decode_with_caches():
    """The generate graph's pattern: kv_write the chunk's k / v at pos, then attention_gqa over the persistent caches;
    a 2-chunk prefill then decode steps, one compiled graph (T symbolic)."""
    cap, qh, kvh, hd = 96, 16, 8, 128
    T = lk.Sym("T")

    def fn(q, k, v, pos):
        kc = lk.persistent_buffer(lk.f16, [cap, kvh * hd], device="gpu")
        vc = lk.persistent_buffer(lk.f16, [cap, kvh * hd], device="gpu")
        F.kv_write_f16_gpu(kc, vc, lk.ops.to_gpu(k), lk.ops.to_gpu(v), pos)
        return lk.ops.to_cpu(F.attention_gqa_f16_gpu(lk.ops.to_gpu(q), kc, vc, pos, q_heads=qh, kv_heads=kvh))

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f16, [T, qh * hd]), lk.TensorType(lk.f16, [T, kvh * hd]),
                                        lk.TensorType(lk.f16, [T, kvh * hd]), lk.TensorType(lk.i64, [1])])
    r = rng(3)
    kall, vall = h16(r, cap, kvh * hd, scale=2), h16(r, cap, kvh * hd)
    pos = 0
    for t in (40, 33, 1, 1, 1):
        q = h16(r, t, qh * hd, scale=2)
        got = model(C(q), C(kall[pos:pos + t]), C(vall[pos:pos + t]), P(pos))
        check_f16(got, _gqa_ref(q, kall, vall, pos, qh, kvh), extra=_attn_extra(vall[:pos + t], pos + t),
                  what=f"prefill/decode at {pos}")
        pos += t


# --- audio_cnn ------------------------------------------------------------------------------------------------------


def _cnn_weights(co, d, f, tm, seed):
    r = rng(seed)
    w3_ = (((f - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1
    ws = [h16(r, co, 3, 3, 1, scale=1.5 / 3)] + [h16(r, co, 3, 3, co, scale=1.5 / np.sqrt(9 * co)) for _ in range(2)]
    bs = [h16(r, co, scale=0.1) for _ in range(3)]
    w_out = h16(r, w3_ * co, d, scale=1 / np.sqrt(w3_ * co))
    return ws, bs, w_out


def _cnn_ref(features, ws, bs, w_out, chunks):
    """Per chunk: x = f16(features[c]^T) as NCHW [1, 1, Tm, F]; 3 x f16(gelu(conv(x) + b)); [H3, W3 * Co] @ w_out.
    Returns the float64 result and the GEMM's float-accumulation term."""
    outs, extras = [], []
    for c in chunks:
        x = r16(t64(features[c]).T[None, None])
        for w, b in zip(ws, bs):
            x = r16(torch.nn.functional.gelu(torch.nn.functional.conv2d(x, t64(w).permute(0, 3, 1, 2), t64(b),
                                                                         stride=2, padding=1)))
        _, co, h3, w3 = x.shape
        flat = x[0].permute(1, 2, 0).reshape(h3, w3 * co)  # (freq, channel) per time step
        wd = t64(w_out)
        outs.append(flat @ wd)
        extras.append(2.0 ** -22 * np.sqrt(w3 * co) * torch.sqrt((flat * flat) @ (wd * wd)))
    return torch.stack(outs), torch.stack(extras).numpy()


@pytest.mark.parametrize("n,co,d,f,tm,chunks", [
    (1, 480, 1024, 128, 100, None), (3, 480, 1024, 128, 100, None),  # real shapes: [n, 128, 100] -> [n, 13, 1024]
    (300, 480, 1024, 128, 100, [0, 1, 150, 299]),  # iPhone capacity; checked chunks (chunk offsets at large n)
    (2, 8, 7, 9, 5, None), (1, 4, 3, 1, 1, None), (4, 32, 40, 17, 12, None),  # odd sizes
])
def test_audio_cnn(n, co, d, f, tm, chunks):
    ws, bs, w_out = _cnn_weights(co, d, f, tm, n + co + f)
    features = (0.8 * rng(n + tm).standard_normal((n, f, tm))).astype(np.float32)
    got = F.audio_cnn_f16_gpu(G(features), G(ws[0]), G(bs[0]), G(ws[1]), G(bs[1]), G(ws[2]), G(bs[2]),
                              G(w_out)).numpy()
    chunks = list(range(n)) if chunks is None else chunks
    ref, extra = _cnn_ref(features, ws, bs, w_out, chunks)
    rms = float(torch.sqrt((ref * ref).mean()))
    check_f16(got[chunks], ref, extra=extra + 2.0 ** -9 * rms, what="audio_cnn")


def test_audio_cnn_repeated_calls():
    """The per-op scratch is reused across calls and chunks: two calls with different n give independent results."""
    co, d, f, tm = 16, 24, 20, 14
    ws, bs, w_out = _cnn_weights(co, d, f, tm, 7)
    model = lk.compile(lambda x: F.audio_cnn_f16_gpu(lk.ops.to_gpu(x), *(G(a) for a in (ws[0], bs[0], ws[1], bs[1],
                                                                                         ws[2], bs[2], w_out))),
                       input_types=[lk.TensorType(lk.f32, [lk.Sym("n"), f, tm])])
    r = rng(8)
    for n in (5, 2, 7):
        feats = (0.8 * r.standard_normal((n, f, tm))).astype(np.float32)
        ref, extra = _cnn_ref(feats, ws, bs, w_out, range(n))
        rms = float(torch.sqrt((ref * ref).mean()))
        check_f16(model(C(feats)), ref, extra=extra + 2.0 ** -9 * rms, what=f"audio_cnn n={n}")
