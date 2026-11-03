"""On-device reference sweeps (docs/correctness-testing.md): one graph computes an optimized kernel and its oracle
(the composition of basic kernels, or the basic CPU kernel for a GPU kernel) side by side; the inputs are generated
on the device from a seed and the two outputs are compared on the device (RUN2), so no tensor crosses the cable —
only the small model and a few numbers. Runs on a local lokinfer-serve by default, on the phone with --device=iphone.
"""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops
from lokinfer.ops.vision import mbv2_block_nhwc_f32_cpu
from lokinfer.remote import Generated

f32 = lk.f32


def T(*shape):
    return lk.TensorType(f32, list(shape))


def sweep(dev, fn, shapes, rtol, atol, seed):
    model = lk.compile(fn, input_types=[T(*s) for s in shapes])
    mid = dev.load(model)
    inputs = [Generated(s, seed=seed * 101 + i, a=0.0, b=0.5) for i, s in enumerate(shapes)]
    _, stats, _ = dev.run2(mid, inputs, compare=[(0, 1, rtol, atol)], return_outputs=False)
    s = stats[0]
    assert s.ok, s
    assert s.max_abs_ref > 0, "degenerate case: all-zero output"
    return s


# (N, H, W, Ci, Ce, Co, stride, residual): MobileNetV2's block shapes, ragged sizes, both strides.
MBV2 = [(1, 112, 112, 16, 96, 24, 2, False), (1, 56, 56, 24, 144, 24, 1, True), (1, 28, 28, 32, 192, 32, 1, True),
        (2, 14, 14, 64, 384, 96, 1, False), (1, 7, 7, 160, 960, 160, 1, True), (1, 9, 13, 8, 48, 8, 2, False),
        (3, 5, 5, 4, 24, 4, 1, True)]


@pytest.mark.parametrize("n,h,w,ci,ce,co,stride,res", MBV2)
def test_sweep_mbv2_block_vs_basic(remote_device, n, h, w, ci, ce, co, stride, res):
    """Fused expand -> dw -> project block vs conv2d 1x1 / dwconv 3x3 / conv2d 1x1 (CPU)."""
    def fn(x, we, be, wd, bd, wp, bp):
        fused = mbv2_block_nhwc_f32_cpu(x, we, be, wd, bd, wp, bp, stride=stride, residual=res)
        e = ops.conv2d_nhwc_f32_cpu(x, ops.reshape(we, [1, 1, ci, ce]), be, act="relu6")
        d = ops.dwconv2d_nhwc_f32_cpu(e, wd, bd, stride=stride, padding=1, act="relu6")
        basic = ops.conv2d_nhwc_f32_cpu(d, ops.reshape(wp, [1, 1, ce, co]), bp, residual=x if res else None)
        return fused, basic

    sweep(remote_device, fn, [(n, h, w, ci), (ci, ce), (ce,), (3, 3, ce), (ce,), (ce, co), (co,)],
          rtol=1e-4, atol=1e-4, seed=n * h + ci)


# (N, H, W, C, K, stride, padding)
DW = [(1, 112, 112, 32, 3, 1, 1), (2, 56, 56, 96, 3, 2, 1), (1, 7, 7, 16, 3, 1, 1), (3, 9, 11, 8, 3, 2, 1),
      (1, 10, 13, 4, 3, 1, 0), (2, 12, 12, 16, 5, 2, 2)]


@pytest.mark.parametrize("n,h,w,c,k,stride,pad", DW)
@pytest.mark.parametrize("act", ["none", "relu6"])
def test_sweep_dwconv_gpu_vs_cpu(remote_device, n, h, w, c, k, stride, pad, act):
    def fn(x, wt, b):
        g = ops.to_cpu(ops.dwconv2d_nhwc_f32_gpu(ops.to_gpu(x), ops.to_gpu(wt), ops.to_gpu(b), stride=stride,
                                                 padding=pad, act=act))
        return g, ops.dwconv2d_nhwc_f32_cpu(x, wt, b, stride=stride, padding=pad, act=act)

    sweep(remote_device, fn, [(n, h, w, c), (k, k, c), (c,)], rtol=1e-5, atol=1e-5, seed=h * c + k)


# (M, K, N): the small-M split-K kernel (M <= 8), steel (8 < M <= 256 on iOS), the tiled kernel above.
GEMM = [(1, 784, 128), (5, 1280, 1000), (8, 64, 10), (33, 512, 96), (64, 784, 128), (200, 300, 70), (1024, 784, 128)]


@pytest.mark.parametrize("m,k,n", GEMM)
def test_sweep_gemm_bias_gpu_vs_basic_cpu(remote_device, m, k, n):
    def fn(a, b, bias):
        g = ops.to_cpu(ops.gemm_bias_f32_gpu(ops.to_gpu(a), ops.to_gpu(b), ops.to_gpu(bias)))
        return g, ops.gemm_bias_f32_cpu(a, b, bias)

    sweep(remote_device, fn, [(m, k), (k, n), (n,)], rtol=1e-4, atol=1e-5 * np.sqrt(k), seed=m + k + n)
