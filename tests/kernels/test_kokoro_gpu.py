"""Kokoro channel-last GPU kernels (M4, kernels/gpu/kokoro_lc.metal) vs float64 torch.

Tolerances: conv (steel f32 implicit GEMM) atol/rtol 1e-4 like the CPU convs; elementwise 1e-5; stats/affine 2e-5.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import lokinfer as lk
from lokinfer import ops

torch.set_num_threads(1)


def G(a):
    return lk.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.float32)), device="gpu", copy=True)


def t64(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def randn(r, *shape, scale=1.0):
    return (r.standard_normal(shape) * scale).astype(np.float32)


# (Ci, Co, L, K, dilation)
CONVS = [(16, 16, 50, 3, 1), (128, 128, 1000, 11, 5), (128, 128, 777, 7, 3), (256, 256, 300, 3, 1),
         (32, 64, 5, 3, 1), (128, 64, 1001, 7, 1)]


@pytest.mark.parametrize("ci,co,l,k,d", CONVS)
@pytest.mark.parametrize("res", [False, True])
def test_conv1d_lc(ci, co, l, k, d, res):
    r = np.random.default_rng(ci + l + k)
    x, w, b = randn(r, l, ci), randn(r, co, ci, k, scale=1 / math.sqrt(ci * k)), randn(r, co)
    pad = d * (k - 1) // 2
    ref = F.conv1d(t64(x).T[None], t64(w), t64(b), padding=pad, dilation=d)[0].T  # [Lo, Co]
    wl = np.ascontiguousarray(w.transpose(0, 2, 1))  # [Co, K, Ci]
    if res:
        rr = randn(r, *ref.shape)
        out = ops.conv1d_lc_f32_gpu(G(x), G(wl), G(b), residual=G(rr), out_scale=0.5, padding=pad, dilation=d)
        ref = (ref + t64(rr)) * 0.5
    else:
        out = ops.conv1d_lc_f32_gpu(G(x), G(wl), G(b), padding=pad, dilation=d)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("l,c", [(1, 16), (300, 128), (1000, 256)])
def test_act_lc(l, c):
    r = np.random.default_rng(l + c)
    x = randn(r, l, c, scale=2)
    af = np.stack([randn(r, c, scale=0.5), r.uniform(0.5, 1.5, c), randn(r, c, scale=0.3)]).astype(np.float32)
    alpha = r.uniform(0.2, 3.0, c).astype(np.float32)
    v = (t64(x) - t64(af[0])) * t64(af[1]) + t64(af[2])
    a = t64(alpha)
    snake = v + (1 / a) * torch.sin(a * v) ** 2
    np.testing.assert_allclose(ops.snake_affine_lc_f32_gpu(G(x), G(af), G(alpha)).numpy(), snake.numpy(),
                               atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ops.leaky_lc_f32_gpu(G(x), 0.2, affine=G(af)).numpy(), F.leaky_relu(v, 0.2).numpy(),
                               atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ops.leaky_lc_f32_gpu(G(x), 0.1).numpy(), F.leaky_relu(t64(x), 0.1).numpy(),
                               atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("l,c", [(1, 16), (700, 128), (5000, 256)])
def test_stats_and_adain_affine_lc(l, c):
    r = np.random.default_rng(l * c)
    x = randn(r, l, c, scale=3) + 5
    gb = randn(r, 2 * c, scale=0.5)
    st = ops.channel_stats_lc_f32_gpu(G(x))
    s = st.numpy()
    xd = x.astype(np.float64)
    for i in range(s.shape[0]):
        seg = xd[i * 128:(i + 1) * 128]
        np.testing.assert_allclose(s[i, :, 0], seg.mean(0), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(s[i, :, 1], ((seg - seg.mean(0)) ** 2).sum(0), atol=2e-4, rtol=1e-4)
    af = ops.adain_affine_lc_f32_gpu(st, G(gb), length=l).numpy()
    got = (x - af[0]) * af[1] + af[2]
    m = xd.mean(0)
    ref = (1 + gb[:c].astype(np.float64)) * (xd - m) / np.sqrt(((xd - m) ** 2).mean(0) + 1e-5) + gb[c:]
    np.testing.assert_allclose(got, ref, atol=2e-5 * max(1, np.abs(ref).max()), rtol=1e-4)


# (Ci, Co, Lin, K, stride, padding, reflect_left): Kokoro's ups[0], ups[1] (+ reflect) and edge cases
CONVTS = [(64, 32, 37, 20, 10, 5, False), (32, 16, 50, 12, 6, 3, True), (16, 8, 1, 20, 10, 5, False),
          (16, 16, 13, 4, 2, 1, True)]


@pytest.mark.parametrize("ci,co,lin,k,s,p,reflect", CONVTS)
@pytest.mark.parametrize("add", [False, True])
def test_convt_lc(ci, co, lin, k, s, p, reflect, add):
    r = np.random.default_rng(ci + lin + k)
    x, w, b = randn(r, lin, ci), randn(r, ci, co, k, scale=1 / math.sqrt(ci * k)), randn(r, co)
    ref = F.conv_transpose1d(t64(x).T[None], t64(w), t64(b), stride=s, padding=p)
    if reflect:
        ref = F.pad(ref, (1, 0), mode="reflect")
    ref = ref[0].T
    wk = np.ascontiguousarray(w.transpose(0, 2, 1).reshape(ci, k * co))  # [Ci, K*Co]
    y = ops.gemm_f32_gpu_mps(G(x), G(wk))
    kw = {}
    if add:
        a = randn(r, *ref.shape)
        kw["add"] = G(a)
        ref = ref + t64(a)
    out = ops.convt_gather_lc_f32_gpu(y, G(b), kernel=k, stride=s, padding=p, reflect_left=reflect, **kw)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-4, rtol=1e-4)


def test_add3_and_lc_to_cf():
    r = np.random.default_rng(0)
    a, b, c = (randn(r, 100, 64) for _ in range(3))
    np.testing.assert_allclose(ops.add3_scale_f32_gpu(G(a), G(b), G(c), 1 / 3).numpy(), (a + b + c) / 3, rtol=1e-6,
                               atol=1e-6)
    np.testing.assert_array_equal(ops.lc_to_cf_f32_gpu(G(a), 22).numpy(), a[:, :22].T)


def test_concat_pad_dwconvt_upsample_lc():
    r = np.random.default_rng(1)
    a, b = randn(r, 50, 514), randn(r, 50, 66)
    out = ops.concat_pad_lc_f32_gpu(G(a), G(b)).numpy()
    assert out.shape == (50, 592)
    np.testing.assert_array_equal(out[:, :514], a)
    np.testing.assert_array_equal(out[:, 514:580], b)
    assert not out[:, 580:].any()
    np.testing.assert_array_equal(ops.concat_pad_lc_f32_gpu(G(a)).numpy()[:, :514], a)
    x, w, bb = randn(r, 37, 32), randn(r, 32, 3), randn(r, 32)
    ref = F.conv_transpose1d(t64(x).T[None], t64(w)[:, None, :], t64(bb), stride=2, padding=1, output_padding=1,
                             groups=32)[0].T
    np.testing.assert_allclose(ops.dwconvt_lc_f32_gpu(G(x), G(w), G(bb)).numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(ops.upsample2_lc_f32_gpu(G(x)).numpy(), np.repeat(x, 2, axis=0))


def test_adain_affine_lc_padded_channels():
    r = np.random.default_rng(2)
    x = np.concatenate([randn(r, 300, 20), np.zeros((300, 12), np.float32)], axis=1)
    gb = randn(r, 40, scale=0.5)
    af = ops.adain_affine_lc_f32_gpu(ops.channel_stats_lc_f32_gpu(G(x)), G(gb), length=300).numpy()
    assert not af[:, 20:].any()  # padding stays exactly 0 through (x - 0) * 0 + 0
    ref = ops.adain_affine_lc_f32_gpu(ops.channel_stats_lc_f32_gpu(G(x[:, :20])), G(gb), length=300).numpy()
    np.testing.assert_array_equal(af[:, :20], ref)


@pytest.mark.parametrize("t", [1, 31, 126, 512])
def test_albert_gpu_kernels(t):
    r = np.random.default_rng(t)
    d, heads = 768, 12
    qkv, b = randn(r, t, 3 * d), randn(r, 3 * d, scale=0.1)
    q, k, v = (t64(qkv[:, i * d:(i + 1) * d] + b[i * d:(i + 1) * d]).reshape(t, heads, 64).transpose(0, 1)
               for i in range(3))
    ref = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(t, d)
    np.testing.assert_allclose(ops.attention_qkv_f32_gpu(G(qkv), G(b), heads=12).numpy(), ref.numpy(), atol=2e-5,
                               rtol=1e-4)
    x, bb = randn(r, t, 2048), randn(r, 2048)
    np.testing.assert_allclose(ops.bias_gelu_f32_gpu(G(x), G(bb)).numpy(),
                               F.gelu(t64(x) + t64(bb), approximate="tanh").numpy(), atol=1e-5, rtol=1e-5)
    a, res = randn(r, t, d), randn(r, t, d)
    bias, gam, bet = randn(r, d), randn(r, d), randn(r, d)
    ref = F.layer_norm(t64(a) + t64(bias) + t64(res), (d,), t64(gam), t64(bet), eps=1e-12)
    np.testing.assert_allclose(ops.add_bias_layernorm_f32_gpu(G(a), G(bias), G(res), G(gam), G(bet)).numpy(),
                               ref.numpy(), atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("l", [2, 5, 1471, 17641, 132121])
def test_istft_logits_lc_matches_cpu(l):
    """GPU iSTFT from channel-last logits (64 padded columns, like conv_post's output) vs the CPU kernel."""
    r = np.random.default_rng(l)
    x = np.zeros((l, 64), np.float32)
    x[:, :22] = r.standard_normal((l, 22)).astype(np.float32) * 0.8
    got = ops.istft_logits_lc_f32_gpu(G(x), n_fft=20, hop=5).numpy()
    cf = lk.from_numpy(np.ascontiguousarray(x[:, :22].T))
    exp = ops.istft_logits_f32_cpu(cf, n_fft=20, hop=5).numpy()
    assert got.shape == exp.shape == ((l - 1) * 5,)
    np.testing.assert_allclose(got, exp, rtol=2e-5, atol=2e-5 * max(1.0, float(np.abs(exp).max())))
