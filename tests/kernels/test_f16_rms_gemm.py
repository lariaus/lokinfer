"""rmsnorm_gemm_opt / rmsnorm_gemm_swiglu_opt (CPU and GPU) vs the composition they replace: rmsnorm then
gemm_opt / gemm_swiglu_opt. The fused GPU decode path (M = 1, K <= 4096: lk_f16o_rms_gemv) normalizes the row in
each threadgroup with lk_f16_rmsnorm's exact summation order and runs the same GEMV, so the outputs must be
bit-identical; the other paths (M > 1, K > 4096, the CPU) are rmsnorm into the workspace + the same GEMM: identical
too. The composition itself is tested against float64 in test_f16_fused_{cpu,gpu}.py."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer.ops import f16 as FC
from lokinfer.ops import f16_gpu as FG

SHAPES = [(1, 2048, 4096), (1, 2048, 2048), (2, 2048, 4096), (44, 2048, 4096), (401, 2048, 2048), (1, 64, 96),
          (1, 33, 7), (3, 17, 5), (1, 5000, 64), (7, 5000, 64), (1, 1, 1)]
SWIGLU = [(1, 2048, 6144), (44, 2048, 6144), (1, 64, 48), (1, 33, 7), (5, 31, 9), (1, 5000, 32)]


def _data(m, k, n2, seed):
    r = np.random.default_rng(seed)
    x = (r.standard_normal((m, k)) * 3).astype(np.float16)
    g = (1 + 0.3 * r.standard_normal(k)).astype(np.float16)
    w = (r.standard_normal((k, n2)) / np.sqrt(k)).astype(np.float16)
    return x, g, w


def _types(*arrays):
    return [lk.TensorType(lk.f16, list(a.shape)) for a in arrays]


def _run(device, swiglu, x, g, w):
    if device == "cpu":
        fused = FC.rmsnorm_gemm_swiglu_opt if swiglu else FC.rmsnorm_gemm_opt
        plain = FC.gemm_swiglu_opt if swiglu else FC.gemm_opt

        def fn(x_, g_, w_):
            y, _ = fused(x_, g_, w_, 1e-6)
            return y, plain(FC.rmsnorm(x_, g_, 1e-6), w_)
    else:
        fused = FG.rmsnorm_gemm_swiglu_opt_f16_gpu if swiglu else FG.rmsnorm_gemm_opt_f16_gpu
        plain = FG.gemm_swiglu_opt_f16_gpu if swiglu else FG.gemm_opt_f16_gpu

        def fn(x_, g_, w_):
            xs, gs, ws = (lk.ops.to_gpu(t) for t in (x_, g_, w_))
            y, _ = fused(xs, gs, ws, 1e-6)
            return lk.ops.to_cpu(y), lk.ops.to_cpu(plain(FG.rmsnorm_f16_gpu(xs, gs, 1e-6), ws))
    model = lk.compile(fn, input_types=_types(x, g, w))
    a, b = model(x, g, w)
    return a.numpy(), b.numpy()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("m,k,n", SHAPES)
def test_rmsnorm_gemm_opt(device, m, k, n):
    x, g, w = _data(m, k, n, m + 7 * k + n)
    fused, ref = _run(device, False, x, g, w)
    assert fused.shape == (m, n) and np.isfinite(fused).all()
    np.testing.assert_array_equal(fused, ref)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("m,k,n", SWIGLU)
def test_rmsnorm_gemm_swiglu_opt(device, m, k, n):
    x, g, w = _data(m, k, 2 * n, m + 11 * k + n)
    fused, ref = _run(device, True, x, g, w)
    assert fused.shape == (m, n) and np.isfinite(fused).all()
    np.testing.assert_array_equal(fused, ref)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_rmsnorm_gemm_repeated_decode_calls(device):
    """The decode pattern: the same compiled graph called many times with M = 1 (no stale state)."""
    x, g, w = _data(1, 2048, 4096, 3)
    for i in range(5):
        xi = (x * (1 + i)).astype(np.float16)
        fused, ref = _run(device, False, xi, g, w)
        np.testing.assert_array_equal(fused, ref)
