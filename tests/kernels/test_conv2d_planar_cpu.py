"""CPU vision kernels added for the iPhone convnet round (kernels/cpu/conv2d.cpp) vs float64 numpy:
- conv2d_wino_cnhw[_res]_relu_f32_cpu: Winograd F(2x2, 3x3) in the channel-planar padded layout (odd / even sizes,
  batches, residual; both run-time branches: Winograd and the direct planar fallback below the tile threshold), and the
  zero border of its output;
- conv2d_cnhw on input planes whose stride is a multiple of 2 KB (copied to padded planes first);
- conv2d_smallci_nchw_{relu6,leaky}_f32_cpu: direct NEON stem conv, NCHW in -> NHWC out.
With --device=iphone the graphs run on the phone (SME2 GEMMs)."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops


def ref_conv_nchw(x, w, b, stride, pad):
    """x [N, C, H, W], w [KH, KW, Ci, Co] -> [N, Co, Ho, Wo] float64."""
    x = x.astype(np.float64)
    n, c, h, wd = x.shape
    kh, kw, _, co = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((n, co, ho, wo)) + b[None, :, None, None]
    for y in range(kh):
        for z in range(kw):
            patch = xp[:, :, y:y + stride * (ho - 1) + 1:stride, z:z + stride * (wo - 1) + 1:stride]
            out += np.einsum("nchw,co->nohw", patch, w[y, z].astype(np.float64))
    return out


def to_cnhw(x):
    """[N, C, H, W] -> cnhw [C, N, H + 2, W + 2] (zero border)."""
    n, c, h, w = x.shape
    out = np.zeros((c, n, h + 2, w + 2), np.float32)
    out[:, :, 1:-1, 1:-1] = x.transpose(1, 0, 2, 3)
    return out


def run(fn, *xs):
    m = lk.compile(fn, input_types=[lk.TensorType(lk.f32, list(x.shape)) for x in xs])
    return m(*[lk.from_numpy(x) for x in xs]).numpy()


# (C_in, C_out, N, H, W): Winograd above the threshold (>= 512 tiles, or >= 64 tiles with N >= 4), direct below.
WINO = [(64, 64, 1, 56, 56), (32, 48, 1, 9, 7), (16, 32, 4, 15, 17), (24, 40, 4, 8, 8), (48, 32, 2, 33, 31),
        (128, 96, 1, 7, 7), (32, 160, 4, 8, 8), (16, 32, 1, 61, 30), (16, 16, 4, 14, 14)]


@pytest.mark.parametrize("ci,co,n,h,w", WINO)
@pytest.mark.parametrize("res", [False, True])
def test_conv2d_wino_cnhw(ci, co, n, h, w, res):
    rng = np.random.default_rng(ci * 7 + co + n + h * w)
    x = rng.standard_normal((n, ci, h, w)).astype(np.float32)
    wt = (rng.standard_normal((3, 3, ci, co)) / np.sqrt(9 * ci)).astype(np.float32)
    b = rng.standard_normal(co).astype(np.float32) * 0.1
    r = rng.standard_normal((n, co, h, w)).astype(np.float32)
    exp = ref_conv_nchw(x, wt, b, 1, 1) + (r if res else 0)
    exp = np.maximum(exp, 0)
    wp = lk.from_numpy(ops.pack_conv_weights_cnhw(wt))
    u = lk.from_numpy(ops.pack_wino_weights_cnhw(wt))
    bt = lk.from_numpy(b)
    if res:
        got = run(lambda x, r: ops.conv2d_wino_cnhw_f32_cpu(x, wp, u, bt, residual=r), to_cnhw(x), to_cnhw(r))
    else:
        got = run(lambda x: ops.conv2d_wino_cnhw_f32_cpu(x, wp, u, bt), to_cnhw(x))
    border = got.copy()
    border[:, :, 1:-1, 1:-1] = 0
    assert not border.any(), "the output border must be zero"
    got = got[:, :, 1:-1, 1:-1].transpose(1, 0, 2, 3)
    err = np.abs(got - exp).max() / np.abs(exp).max()
    assert err < 2e-5, err


def test_conv2d_cnhw_power_of_two_planes():
    """N = 8, 14 x 14: plane stride 8 * 16 * 16 floats (the padded-copy path), stride 1 and the stride-2 phases."""
    rng = np.random.default_rng(3)
    n, ci, co, h = 8, 32, 64, 14
    x = np.maximum(rng.standard_normal((n, ci, h, h)), 0).astype(np.float32)
    wt = (rng.standard_normal((3, 3, ci, co)) / np.sqrt(9 * ci)).astype(np.float32)
    b = np.zeros(co, np.float32)
    wp, bt = lk.from_numpy(ops.pack_conv_weights_cnhw(wt)), lk.from_numpy(b)
    got = run(lambda x: ops.conv2d_cnhw_f32_cpu(x, wp, bt, 3, padding=1, act="relu"), to_cnhw(x))
    exp = np.maximum(ref_conv_nchw(x, wt, b, 1, 1), 0)
    assert np.abs(got[:, :, 1:-1, 1:-1].transpose(1, 0, 2, 3) - exp).max() / np.abs(exp).max() < 2e-5
    x2 = np.maximum(rng.standard_normal((n, ci, 2 * h, 2 * h)), 0).astype(np.float32)
    got = run(lambda x: ops.conv2d_cnhw_f32_cpu(ops.cnhw_to_phases_f32_cpu(x), wp, bt, 3, stride=2, padding=1,
                                                 act="relu"), to_cnhw(x2))
    exp = np.maximum(ref_conv_nchw(x2, wt, b, 2, 1), 0)
    assert np.abs(got[:, :, 1:-1, 1:-1].transpose(1, 0, 2, 3) - exp).max() / np.abs(exp).max() < 2e-5


@pytest.mark.parametrize("n,c,h,w,k,stride,pad,co,act", [
    (1, 3, 224, 224, 3, 2, 1, 32, "relu6"), (2, 3, 225, 223, 3, 2, 1, 32, "relu6"), (3, 3, 64, 64, 4, 2, 1, 64, "leaky"),
    (1, 3, 9, 11, 4, 2, 1, 64, "leaky"), (1, 2, 13, 10, 3, 1, 1, 16, "relu6"), (2, 4, 7, 6, 5, 1, 2, 48, "leaky")])
def test_conv2d_smallci_nchw(n, c, h, w, k, stride, pad, co, act):
    rng = np.random.default_rng(n + c * h + w)
    x = rng.standard_normal((n, c, h, w)).astype(np.float32) * 2
    wt = rng.standard_normal((k, k, c, co)).astype(np.float32)
    b = rng.standard_normal(co).astype(np.float32)
    exp = ref_conv_nchw(x, wt, b, stride, pad).transpose(0, 2, 3, 1)
    exp = np.clip(exp, 0, 6) if act == "relu6" else np.where(exp >= 0, exp, 0.2 * exp)
    wl, bl = lk.from_numpy(wt), lk.from_numpy(b)
    got = run(lambda x: ops.conv2d_smallci_nchw_f32_cpu(x, wl, bl, stride=stride, padding=pad, act=act), x)
    assert got.shape == exp.shape
    assert np.abs(got - exp).max() / np.abs(exp).max() < 1e-5

