"""lokinfer's steel GEMM (kernels/gpu/gemm_steel.metal) vs float64 numpy: all epilogue variants, transposed B,
aligned and ragged shapes (M / N / K not multiples of the tile), GEMV-like M = 1, both tile configs."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops


def G(a):
    return lk.from_numpy(np.ascontiguousarray(a, np.float32), device="gpu", copy=True)


SHAPES = [(1, 768, 2304), (1, 3072, 768), (7, 33, 65), (128, 768, 768), (512, 768, 3072), (300, 1000, 517),
          (64, 16, 64), (2, 768, 50257)]


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_gemm_steel(m, k, n):
    r = np.random.default_rng(m + k + n)
    a, b = r.standard_normal((m, k)).astype(np.float32), (r.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    bias, res = r.standard_normal(n).astype(np.float32), r.standard_normal((m, n)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    tol = dict(atol=2e-4 * max(1, np.abs(ref).max()), rtol=1e-4)
    np.testing.assert_allclose(ops.gemm_steel_f32_gpu(G(a), G(b)).numpy(), ref, **tol)
    np.testing.assert_allclose(ops.gemm_steel_f32_gpu(G(a), G(b), bias=G(bias)).numpy(), ref + bias, **tol)
    v = ref + bias
    gelu = 0.5 * v * (1 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))
    np.testing.assert_allclose(ops.gemm_steel_f32_gpu(G(a), G(b), bias=G(bias), act="gelu").numpy(), gelu, **tol)
    np.testing.assert_allclose(ops.gemm_steel_f32_gpu(G(a), G(b), bias=G(bias), residual=G(res)).numpy(),
                               ref + bias + res, **tol)
    np.testing.assert_allclose(ops.gemm_nt_steel_f32_gpu(G(a), G(np.ascontiguousarray(b.T))).numpy(), ref, **tol)
