"""dwconv2d_nhwc[_relu6]_f32_gpu vs float64 numpy: the register-blocked 3x3 float4 kernel (stride 1 / 2, ragged W,
padding 0 / 1) and the generic kernel (5x5, C % 4 != 0)."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops


def G(a):
    return lk.from_numpy(np.ascontiguousarray(a, np.float32), device="gpu", copy=True)


def ref_dw(x, w, b, stride, pad, act):
    n, h, wd, c = x.shape
    kh, kw = w.shape[:2]
    xp = np.pad(x.astype(np.float64), ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    ho, wo = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((n, ho, wo, c)) + b
    for y in range(kh):
        for z in range(kw):
            out += xp[:, y:y + stride * (ho - 1) + 1:stride, z:z + stride * (wo - 1) + 1:stride, :] * w[y, z]
    return np.clip(out, 0, 6) if act == "relu6" else out


CASES = [(1, 112, 112, 32, 3, 1, 1), (2, 56, 56, 96, 3, 2, 1), (1, 7, 7, 16, 3, 1, 1), (3, 9, 11, 8, 3, 2, 1),
         (1, 10, 13, 4, 3, 1, 0), (1, 6, 6, 4, 3, 2, 0), (1, 9, 9, 6, 3, 1, 1), (2, 12, 12, 16, 5, 2, 2)]


@pytest.mark.parametrize("n,h,w,c,k,stride,pad", CASES)
@pytest.mark.parametrize("act", ["none", "relu6"])
def test_dwconv_gpu(n, h, w, c, k, stride, pad, act):
    r = np.random.default_rng(n * 1000 + h * 10 + c + k)
    x = r.standard_normal((n, h, w, c)).astype(np.float32)
    wt = r.standard_normal((k, k, c)).astype(np.float32)
    b = r.standard_normal(c).astype(np.float32)
    got = ops.dwconv2d_nhwc_f32_gpu(G(x), G(wt), G(b), stride=stride, padding=pad, act=act).numpy()
    np.testing.assert_allclose(got, ref_dw(x, wt, b, stride, pad, act), atol=1e-4, rtol=1e-4)
