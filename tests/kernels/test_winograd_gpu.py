"""Fused Winograd conv (kernels/gpu/winograd.metal) vs float64 torch, Kokoro shapes + edge cases.

Same tolerance as the steel conv (tests/kernels/test_kokoro_gpu.py): atol/rtol 1e-4 on unit-scale data. The measured
error on real Kokoro activations is 1-3x the direct f32 conv's (docs/impl-notes/winograd-conv.md).
"""
import math
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import lokinfer as lk
from lokinfer import ops
from lokinfer.ops import winograd as W

torch.set_num_threads(1)


def G(a):
    return lk.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.float32)), device="gpu", copy=True)


def t64(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


@pytest.mark.parametrize("r", [3, 4])
def test_transforms_exact(r):
    at, g, bt = W.transforms(4, r)
    rng = np.random.default_rng(r)
    d, w = rng.standard_normal(3 + r), rng.standard_normal(r)
    y = at @ ((g @ w) * (bt @ d))
    ref = [sum(w[k] * d[i + k] for k in range(r)) for i in range(4)]
    np.testing.assert_allclose(y, ref, rtol=1e-12, atol=1e-12)
    assert W._POINTS[r][-1] in (-2, Fraction(-1, 2))


# (Ci, Co, L, K, dilation)
CONVS = [(128, 128, 1000, 11, 5), (128, 128, 777, 7, 3), (256, 256, 300, 3, 1), (128, 128, 1001, 3, 5),
         (256, 256, 513, 7, 1), (128, 128, 37, 11, 3), (16, 64, 5, 3, 1), (128, 64, 1001, 7, 1),
         (32, 128, 70, 5, 2), (64, 64, 9, 12, 1), (16, 64, 40, 4, 1), (128, 128, 2, 3, 1)]


@pytest.mark.parametrize("ci,co,l,k,d", CONVS)
@pytest.mark.parametrize("res", [False, True])
def test_wino_conv1d_lc(ci, co, l, k, d, res):
    r = np.random.default_rng(ci + l + k + d)
    x = r.standard_normal((l, ci)).astype(np.float32)
    w = (r.standard_normal((co, ci, k)) / math.sqrt(ci * k)).astype(np.float32)
    b = r.standard_normal(co).astype(np.float32)
    pad = d * (k - 1) // 2
    if k % 2 == 0:  # even K: pad must stay a multiple of d
        pad = d * (k // 2)
    ref = F.conv1d(t64(x).T[None], t64(w), t64(b), padding=pad, dilation=d)[0].T  # [Lo, Co]
    u, tr = W.wino_weights(w)
    if res:
        rr = r.standard_normal(tuple(ref.shape)).astype(np.float32)
        out = ops.conv1d_wino_lc_f32_gpu(G(x), G(u), G(b), G(tr), k, residual=G(rr), out_scale=0.5, padding=pad,
                                         dilation=d)
        ref = (ref + t64(rr)) * 0.5
    else:
        out = ops.conv1d_wino_lc_f32_gpu(G(x), G(u), G(b), G(tr), k, padding=pad, dilation=d)
    assert out.shape == tuple(ref.shape)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-4, rtol=1e-4)


def test_wino_rejects_bad_shapes():
    w = np.zeros((64, 16, 13), np.float32)
    with pytest.raises(lk.ops.OpError):
        W.wino_weights(w)


@pytest.mark.parametrize("ci,co,l,k,d", [(128, 128, 1000, 11, 5), (256, 256, 300, 3, 3), (128, 128, 501, 7, 1)])
def test_wino_conv1d_lc_snake(ci, co, l, k, d):
    """Snake(AdaIN affine) fused into the input transform == snake_affine_lc_f32_gpu followed by the conv."""
    r = np.random.default_rng(l + k)
    x = (r.standard_normal((l, ci)) * 2).astype(np.float32)
    af = np.stack([r.standard_normal(ci) * 0.5, r.uniform(0.5, 1.5, ci), r.standard_normal(ci) * 0.3]).astype(np.float32)
    alpha = r.uniform(0.2, 3.0, ci).astype(np.float32)
    w = (r.standard_normal((co, ci, k)) / math.sqrt(ci * k)).astype(np.float32)
    b = r.standard_normal(co).astype(np.float32)
    pad = d * (k - 1) // 2
    h = (t64(x) - t64(af[0])) * t64(af[1]) + t64(af[2])
    h = h + (1 / t64(alpha)) * torch.sin(t64(alpha) * h) ** 2
    ref = F.conv1d(h.T[None], t64(w), t64(b), padding=pad, dilation=d)[0].T
    u, tr = W.wino_weights(w)
    out = ops.conv1d_wino_lc_f32_gpu(G(x), G(u), G(b), G(tr), k, padding=pad, dilation=d, affine=G(af),
                                     alpha=G(alpha))
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=2e-4, rtol=2e-4)


# 2-D F(2x2, 3x3): (N, H, W, Ci, Co); odd sizes exercise partial tiles
CONVS2D = [(2, 56, 56, 64, 64), (1, 7, 7, 512, 512), (3, 13, 9, 128, 64), (1, 1, 1, 16, 64), (8, 14, 14, 256, 256)]


@pytest.mark.parametrize("n,h,w,ci,co", CONVS2D)
@pytest.mark.parametrize("res,relu", [(False, False), (False, True), (True, True)])
def test_wino_conv2d_nhwc(n, h, w, ci, co, res, relu):
    r = np.random.default_rng(n + h + ci)
    x = r.standard_normal((n, h, w, ci)).astype(np.float32)
    wt = (r.standard_normal((3, 3, ci, co)) / math.sqrt(9 * ci)).astype(np.float32)  # [KH, KW, Ci, Co]
    b = r.standard_normal(co).astype(np.float32)
    ref = F.conv2d(t64(x).permute(0, 3, 1, 2), t64(wt).permute(3, 2, 0, 1), t64(b), padding=1).permute(0, 2, 3, 1)
    u, bt, tr = W.wino2d_weights(wt)
    rr = r.standard_normal(tuple(ref.shape)).astype(np.float32) if res else None
    if res:
        ref = ref + t64(rr)
    if relu:
        ref = torch.relu(ref)
    out = ops.conv2d_wino_nhwc_f32_gpu(G(x), G(u), G(b), G(bt), G(tr), residual=G(rr) if res else None, relu=relu)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("ci,co,l", [(1104, 1024, 147), (528, 1024, 60), (256, 256, 294)])
@pytest.mark.parametrize("res", [False, True])
def test_wino_conv1d_lc_leaky(ci, co, l, res):
    """leaky_relu(AdaIN affine) fused into the input transform (the decoder's AdainResBlk1d convs, K = 3)."""
    r = np.random.default_rng(l + ci)
    x = (r.standard_normal((l, ci)) * 2).astype(np.float32)
    af = np.stack([r.standard_normal(ci) * 0.5, r.uniform(0.5, 1.5, ci), r.standard_normal(ci) * 0.3]).astype(np.float32)
    w = (r.standard_normal((co, ci, 3)) / math.sqrt(ci * 3)).astype(np.float32)
    b = r.standard_normal(co).astype(np.float32)
    h = F.leaky_relu((t64(x) - t64(af[0])) * t64(af[1]) + t64(af[2]), 0.2)
    ref = F.conv1d(h.T[None], t64(w), t64(b), padding=1)[0].T
    u, tr = W.wino_weights(w)
    kw = {}
    if res:
        rr = r.standard_normal(tuple(ref.shape)).astype(np.float32)
        kw = dict(residual=G(rr), out_scale=0.7071)
        ref = (ref + t64(rr)) * 0.7071
    out = ops.conv1d_wino_lc_f32_gpu(G(x), G(u), G(b), G(tr), 3, padding=1, affine=G(af), slope=0.2, **kw)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("ci,co,l,k,d", [(128, 128, 1000, 11, 5), (128, 128, 777, 7, 3), (256, 256, 300, 3, 1),
                                         (128, 128, 129, 3, 1), (128, 64, 5, 7, 3)])
@pytest.mark.parametrize("res", [False, True])
def test_wino_stats_epilogue_affine(ci, co, l, k, d, res):
    """Channel stats from the GEMM epilogue -> AdaIN affine == the float64 mean / std of the conv output."""
    r = np.random.default_rng(l + k + d)
    x = r.standard_normal((l, ci)).astype(np.float32)
    w = (r.standard_normal((co, ci, k)) / math.sqrt(ci * k)).astype(np.float32)
    b = (r.standard_normal(co) + 3).astype(np.float32)  # a non-zero mean
    gb = r.standard_normal(2 * co).astype(np.float32) * 0.3
    pad = d * (k - 1) // 2
    u, tr = W.wino_weights(w)
    kw = {}
    if res:
        rr = r.standard_normal((l, co)).astype(np.float32)
        kw = dict(residual=G(rr), out_scale=0.5)
    y, st = ops.conv1d_wino_lc_f32_gpu(G(x), G(u), G(b), G(tr), k, padding=pad, dilation=d, stats=True, **kw)
    af = ops.wino_adain_affine_lc_f32_gpu(st, G(gb), l, dilation=d).numpy()
    y64 = y.numpy().astype(np.float64)
    mean, var = y64.mean(0), y64.var(0)
    np.testing.assert_allclose(af[0], mean, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(af[1], (1 + gb[:co]) / np.sqrt(var + 1e-5), rtol=1e-4)
    np.testing.assert_allclose(af[2], gb[co:], rtol=0, atol=0)


@pytest.mark.parametrize("ci,co,l,d", [(256, 256, 300, 1), (128, 128, 1001, 5), (16, 64, 5, 1), (128, 128, 4097, 3)])
@pytest.mark.parametrize("stats", [False, True])
def test_wino_f23_conv1d_lc(ci, co, l, d, stats):
    """F(2,3) (wino_weights(w, tile=2)) for K = 3: same contract as F(4,3), incl. the epilogue stats."""
    r = np.random.default_rng(ci + l + d)
    x = r.standard_normal((l, ci)).astype(np.float32)
    w = (r.standard_normal((co, ci, 3)) / math.sqrt(ci * 3)).astype(np.float32)
    b = r.standard_normal(co).astype(np.float32)
    rr = r.standard_normal((l, co)).astype(np.float32)
    ref = (F.conv1d(t64(x).T[None], t64(w), t64(b), padding=d, dilation=d)[0].T + t64(rr)) * 0.5
    u, tr = W.wino_weights(w, tile=2)
    assert u.shape[0] == 4
    out = ops.conv1d_wino_lc_f32_gpu(G(x), G(u), G(b), G(tr), 3, residual=G(rr), out_scale=0.5, padding=d,
                                     dilation=d, stats=stats)
    if stats:
        out, st = out
        af = ops.wino_adain_affine_lc_f32_gpu(st, G(np.zeros(2 * co, np.float32)), l, dilation=d).numpy()
        np.testing.assert_allclose(af[0], ref.numpy().mean(0), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-4, rtol=1e-4)
