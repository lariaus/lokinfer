"""Fused frame-rate conv kernels (M3) vs the composition of torch ops in float64.

conv1d_{snake,leaky,affine}[_res] = conv1d(act((x - mean) * scale + beta)) + b [+ res]; channel_stats +
adain_affine = AdaIN as a per-channel affine [3, C] = (mean, scale, beta). Tolerances: GEMM-based, atol/rtol 1e-4 (as the
basic conv); stats/affine 2e-5. Lengths cover one tile, many tiles, a partial last tile, and the
parallel (thread pool) path.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import lokinfer as lk
from lokinfer import ops

torch.set_num_threads(1)


def T(a):
    return lk.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.float32)), copy=True)


def t64(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def randn(r, *shape, scale=1.0):
    return (r.standard_normal(shape) * scale).astype(np.float32)


def inorm64(v):
    """Instance norm over the last dim (biased variance, eps 1e-5); torch's F.instance_norm rejects L = 1."""
    m = v.mean(dim=-1, keepdim=True)
    return (v - m) / torch.sqrt(((v - m) ** 2).mean(dim=-1, keepdim=True) + 1e-5)


def snake64(x, a):
    return x + (1 / a) * torch.sin(a * x) ** 2


# (Ci, Co, L, K, dilation, tile); tile None = ops.conv_tile default
CASES = [
    (8, 8, 50, 3, 1, 128),  # one tile, partial stats chunk
    (16, 12, 700, 7, 3, 128),  # many tiles, partial last
    (12, 16, 97, 11, 5, 256),  # window wider than L at the edges
    (128, 128, 5000, 11, 5, None),  # parallel path (> 4M MACs)
    (64, 64, 3000, 3, 1, 640),  # tile spanning several stats chunks
    (8, 8, 1, 3, 1, 128),  # L = 1
]


def _setup(ci, co, l, k, seed):
    r = np.random.default_rng(seed)
    x = randn(r, ci, l, scale=2)
    w = randn(r, co, ci, k, scale=1 / math.sqrt(ci * k))
    b = randn(r, co)
    affine = np.stack([randn(r, ci, scale=0.5), r.uniform(0.5, 1.5, ci), randn(r, ci, scale=0.3)]).astype(np.float32)
    alpha = r.uniform(0.2, 3.0, ci).astype(np.float32)
    res = randn(r, co, l)
    return x, w, b, affine, alpha, res


def _stats_ref(y, tile=128):
    c, l = y.shape
    out = []
    for t0 in range(0, l, tile):
        seg = y[:, t0:t0 + tile]
        m = seg.mean(axis=1)
        out.append(np.stack([m, ((seg - m[:, None]) ** 2).sum(axis=1)], axis=1))
    return np.stack(out, axis=1)


def _check_stats(stats, y):
    ref = _stats_ref(np.asarray(y, dtype=np.float64))
    assert stats.shape == ref.shape
    np.testing.assert_allclose(stats[..., 0], ref[..., 0], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(stats[..., 1], ref[..., 1], atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("ci,co,l,k,d,tile", CASES)
@pytest.mark.parametrize("variant", ["snake", "snake_res", "leaky", "leaky_res", "affine"])
def test_conv1d_fused(ci, co, l, k, d, tile, variant):
    x, w, b, affine, alpha, res = _setup(ci, co, l, k, ci * 31 + l + k)
    pad = d * (k - 1) // 2
    af = t64(affine)
    xa = (t64(x) - af[0][:, None]) * af[1][:, None] + af[2][:, None]
    if variant.startswith("snake"):
        xa = snake64(xa, t64(alpha)[:, None])
        out, stats = ops.conv1d_snake_f32_cpu(T(x), T(w), T(b), T(affine), T(alpha),
                                              residual=T(res) if variant == "snake_res" else None,
                                              padding=pad, dilation=d, tile=tile)
    elif variant.startswith("leaky"):
        xa = F.leaky_relu(xa, 0.2)
        res_kw = dict(residual=T(res), out_scale=0.7071067811865476) if variant == "leaky_res" else {}
        out, stats = ops.conv1d_leaky_f32_cpu(T(x), T(w), T(b), T(affine), slope=0.2, padding=pad,
                                              dilation=d, tile=tile, **res_kw)
    else:
        out, stats = ops.conv1d_affine_f32_cpu(T(x), T(w), T(b), T(affine), padding=pad, dilation=d,
                                               tile=tile)
    ref = F.conv1d(xa[None], t64(w), t64(b), padding=pad, dilation=d)[0]
    if variant == "snake_res":
        ref = ref + t64(res)
    if variant == "leaky_res":
        ref = (ref + t64(res)) * 0.7071067811865476
    ref = ref.numpy()
    out, stats = out.numpy(), stats.numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
    _check_stats(stats, out)


# Winograd variant (conv1d_wino_f32_cpu): the CASES plus a short sequence on a wide input (one tile, two parallel
# phases) and a K = 3 decoder-like shape. tile None = ops.wino_tile.
WINO_CASES = CASES + [(256, 256, 1100, 7, 3, None), (300, 256, 147, 3, 1, None), (64, 40, 3000, 3, 1, None)]


@pytest.mark.parametrize("ci,co,l,k,d,tile", WINO_CASES)
@pytest.mark.parametrize("variant", ["snake", "snake_res", "leaky", "leaky_res", "affine"])
def test_conv1d_wino(ci, co, l, k, d, tile, variant):
    x, w, b, affine, alpha, res = _setup(ci, co, l, k, ci * 31 + l + k)
    pad = d * (k - 1) // 2
    af = t64(affine)
    xa = (t64(x) - af[0][:, None]) * af[1][:, None] + af[2][:, None]
    u, tr = ops.wino_weights_cpu(w)
    kw = dict(padding=pad, dilation=d, tile=tile)
    if variant.startswith("snake"):
        xa = snake64(xa, t64(alpha)[:, None])
        out, stats = ops.conv1d_wino_f32_cpu(T(x), T(u), T(tr), T(b), T(affine), k, alpha=T(alpha),
                                             residual=T(res) if variant == "snake_res" else None, **kw)
    elif variant.startswith("leaky"):
        xa = F.leaky_relu(xa, 0.2)
        res_kw = dict(residual=T(res), out_scale=0.7071067811865476) if variant == "leaky_res" else {}
        out, stats = ops.conv1d_wino_f32_cpu(T(x), T(u), T(tr), T(b), T(affine), k, act="leaky", slope=0.2, **kw,
                                             **res_kw)
    else:
        out, stats = ops.conv1d_wino_f32_cpu(T(x), T(u), T(tr), T(b), T(affine), k, act="none", **kw)
    ref = F.conv1d(xa[None], t64(w), t64(b), padding=pad, dilation=d)[0]
    if variant == "snake_res":
        ref = ref + t64(res)
    if variant == "leaky_res":
        ref = (ref + t64(res)) * 0.7071067811865476
    ref = ref.numpy()
    out, stats = out.numpy(), stats.numpy()
    assert out.shape == ref.shape
    # Winograd's f32 error is ~1-2x the direct conv's (docs/impl-notes/winograd-conv.md "Accuracy").
    np.testing.assert_allclose(out, ref, atol=2e-4, rtol=2e-4)
    _check_stats(stats, out)


@pytest.mark.parametrize("c,l", [(4, 1), (16, 700), (128, 5000)])
def test_channel_stats_and_adain_affine(c, l):
    r = np.random.default_rng(c + l)
    x = randn(r, c, l, scale=3) + 5  # offset mean: catches naive sum-of-squares variance
    gb = randn(r, 2 * c, scale=0.5)
    stats = ops.channel_stats_f32_cpu(T(x)).numpy()
    _check_stats(stats, x)
    af = ops.adain_affine_f32_cpu(T(stats), T(gb), length=l).numpy()
    got = (x - af[0][:, None]) * af[1][:, None] + af[2][:, None]
    xt = t64(x)[None]
    ref = (1 + t64(gb[:c])[:, None]) * inorm64(xt)[0] + t64(gb[c:])[:, None]
    np.testing.assert_allclose(got, ref.numpy(), atol=2e-5 * max(1, np.abs(ref.numpy()).max()), rtol=1e-4)


def test_resblock1_iteration_symbolic():
    """One AdaINResBlock1 iteration through the fused path (symbolic L) vs torch float64:
    xt = conv2(snake(adain2(conv1(snake(adain1(x)))))) + x."""
    c, k, d, tile = 16, 7, 3, 128
    r = np.random.default_rng(3)
    w1, w2 = (randn(r, c, c, k, scale=1 / math.sqrt(c * k)) for _ in range(2))
    b1, b2 = randn(r, c), randn(r, c)
    a1, a2 = (r.uniform(0.2, 3.0, c).astype(np.float32) for _ in range(2))
    gb1, gb2 = randn(r, 2 * c, scale=0.5), randn(r, 2 * c, scale=0.5)
    L = lk.Sym("L")

    def fn(x):
        n = x.shape[1]
        st = ops.channel_stats_f32_cpu(x)
        af1 = ops.adain_affine_f32_cpu(st, T(gb1), length=n)
        y, st = ops.conv1d_snake_f32_cpu(x, T(w1), T(b1), af1, T(a1), padding=d * (k - 1) // 2, dilation=d,
                                         tile=tile)
        af2 = ops.adain_affine_f32_cpu(st, T(gb2), length=n)
        z, _ = ops.conv1d_snake_f32_cpu(y, T(w2), T(b2), af2, T(a2), residual=x, padding=(k - 1) // 2, tile=tile)
        return z

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f32, [c, L])])

    def adain64(v, gb):
        return (1 + t64(gb[:c])[:, None]) * inorm64(v) + t64(gb[c:])[:, None]

    for l in [5, 64, 1000]:
        x = randn(r, c, l)
        xt = t64(x)
        h = snake64(adain64(xt, gb1), t64(a1)[:, None])
        h = F.conv1d(h[None], t64(w1), t64(b1), padding=d * (k - 1) // 2, dilation=d)[0]
        h = snake64(adain64(h, gb2), t64(a2)[:, None])
        ref = (F.conv1d(h[None], t64(w2), t64(b2), padding=(k - 1) // 2)[0] + xt).numpy()
        np.testing.assert_allclose(model(T(x)).numpy(), ref, atol=2e-4, rtol=2e-4)


# --- polyphase ConvTranspose1d (generator ups) vs leaky -> F.conv_transpose1d -> reflect pad -> + res, float64 ------
# (Ci, Co, L, K, s, pad_left): Kokoro ups[0] / ups[1] shapes (word / app59f lengths), plus small, odd and tail cases.
CONVT_CASES = [
    (8, 32, 5, 4, 2, 0),
    (16, 40, 37, 8, 4, 1),  # rows not a multiple of 32
    (512, 256, 26, 20, 10, 0),  # ups[0], word
    (512, 256, 294, 20, 10, 0),  # ups[0], app59f
    (256, 128, 2940, 12, 6, 1),  # ups[1], app59f
    (256, 128, 1, 12, 6, 1),  # L = 1
    (64, 64, 700, 12, 6, 1),  # several column chunks with a partial one
]


@pytest.mark.parametrize("ci,co,l,k,s,pl", CONVT_CASES)
def test_conv_transpose1d_poly(ci, co, l, k, s, pl):
    r = np.random.default_rng(ci + l + k)
    x = randn(r, ci, l, scale=2)
    w = randn(r, ci, co, k, scale=1 / math.sqrt(ci * k / s))
    b = randn(r, co)
    p = (k - s) // 2
    lo = l * s + pl
    res = randn(r, co, lo)
    y = F.conv_transpose1d(F.leaky_relu(t64(x), 0.1), t64(w), t64(b), stride=s, padding=p)
    if pl:
        y = F.pad(y, (1, 0), mode="reflect") if y.shape[-1] > 1 else torch.cat([y, y], -1)
    ref = (y + t64(res)).numpy()
    wp = ops.convt_poly_weights_cpu(w, s, p)
    out = ops.conv_transpose1d_poly_f32_cpu(T(x), T(wp), T(b), T(res), stride=s, padding=p, taps=k // s, slope=0.1,
                                            pad_left=pl).numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


# --- 3-input-mean prologues (the generator's resblock mean fused into its consumer) vs add3_scale -> kernel ----------
@pytest.mark.parametrize("ci,co,l", [(16, 8, 300), (128, 22, 2000), (8, 4, 5)])
def test_conv1d_leaky_wino_mean3(ci, co, l):
    r = np.random.default_rng(ci + l)
    xs = [randn(r, ci, l) for _ in range(3)]
    w = randn(r, co, ci, 7, scale=1 / math.sqrt(ci * 7))
    b = randn(r, co)
    u, tr = ops.wino_weights_cpu(w)
    ident = np.stack([np.zeros(ci), np.ones(ci), np.zeros(ci)]).astype(np.float32)
    got, st = ops.conv1d_leaky_wino_mean3_f32_cpu(T(xs[0]), T(xs[1]), T(xs[2]), T(u), T(tr), T(b), T(ident), 7,
                                                  in_scale=1 / 3, slope=0.01, padding=3)
    m = ops.add3_scale_f32_cpu(T(xs[0]), T(xs[1]), T(xs[2]), scale=1 / 3)
    exp, est = ops.conv1d_wino_f32_cpu(m, T(u), T(tr), T(b), T(ident), 7, act="leaky", slope=0.01, padding=3)
    np.testing.assert_array_equal(got.numpy(), exp.numpy())  # same window values -> bit-identical
    np.testing.assert_array_equal(st.numpy(), est.numpy())


@pytest.mark.parametrize("ci,co,l,k,s,pl", [(16, 40, 37, 8, 4, 1), (256, 128, 294, 12, 6, 1)])
def test_conv_transpose1d_poly_mean3(ci, co, l, k, s, pl):
    r = np.random.default_rng(ci + l + 1)
    xs = [randn(r, ci, l, scale=2) for _ in range(3)]
    w = randn(r, ci, co, k, scale=1 / math.sqrt(ci * k / s))
    b, res = randn(r, co), randn(r, co, l * s + pl)
    p = (k - s) // 2
    wp = T(ops.convt_poly_weights_cpu(w, s, p))
    m = ops.add3_scale_f32_cpu(T(xs[0]), T(xs[1]), T(xs[2]), scale=1 / 3)
    kw = dict(stride=s, padding=p, taps=k // s, slope=0.1, pad_left=pl)
    exp = ops.conv_transpose1d_poly_f32_cpu(m, wp, T(b), T(res), **kw).numpy()
    got = ops.conv_transpose1d_poly_f32_cpu(T(xs[0]), wp, T(b), T(res), mean3=(T(xs[1]), T(xs[2]), 1 / 3), **kw).numpy()
    np.testing.assert_array_equal(got, exp)
