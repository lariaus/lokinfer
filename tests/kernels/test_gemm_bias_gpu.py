"""gemm_bias[_relu]_f32_gpu vs float64 numpy: the split-K small-M kernel (M <= 8, gemm.metal), the steel split-K
kernel with the in-kernel "last split applies the epilogue" combine (8 < M <= 256) and the big-M paths, aligned and
ragged shapes (MNIST, classifier heads, N / K not multiples of 32 or 16), every steel tile config (env knobs, read
at prepare) and the legacy dispatch."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops
from lokinfer.ops import _def


@pytest.fixture
def fresh_env(monkeypatch):
    """The knobs are read at prepare: drop cached eager executables before and after."""
    _def._eager_cache.clear()
    yield monkeypatch
    monkeypatch.undo()
    _def._eager_cache.clear()


def G(a):
    return lk.from_numpy(np.ascontiguousarray(a, np.float32), device="gpu", copy=True)


SHAPES = [(1, 784, 128), (1, 128, 64), (1, 64, 10), (3, 33, 65), (8, 1280, 1000), (8, 7, 1), (9, 100, 50),
          (64, 784, 128), (64, 128, 64), (64, 64, 10), (1024, 128, 64), (1, 1, 1), (5, 2000, 40), (1, 4096, 10),
          (16, 784, 128), (100, 300, 70), (256, 1000, 33), (300, 784, 128), (1024, 784, 128)]


def _data(m, k, n):
    r = np.random.default_rng(m * 7 + k + n)
    a, b = r.standard_normal((m, k)).astype(np.float32), (r.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    bias = r.standard_normal(n).astype(np.float32)
    return a, b, bias, a.astype(np.float64) @ b.astype(np.float64) + bias


def _check(m, k, n):
    a, b, bias, ref = _data(m, k, n)
    tol = dict(atol=2e-4 * max(1, np.abs(ref).max()), rtol=1e-4)
    np.testing.assert_allclose(ops.gemm_bias_f32_gpu(G(a), G(b), G(bias)).numpy(), ref, **tol)
    np.testing.assert_allclose(ops.gemm_bias_relu_f32_gpu(G(a), G(b), G(bias)).numpy(), np.maximum(ref, 0), **tol)


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_gemm_bias_gpu(m, k, n):
    _check(m, k, n)


@pytest.mark.parametrize("cfg", [*range(8), -3, -5])
@pytest.mark.parametrize("m,k,n", [(9, 100, 50), (64, 784, 128), (100, 300, 70), (300, 784, 128), (1024, 128, 64)])
def test_gemm_bias_gpu_tile_configs(fresh_env, cfg, m, k, n):
    monkeypatch = fresh_env
    monkeypatch.setenv("LOKINFER_GEMMB_MID", str(cfg))
    monkeypatch.setenv("LOKINFER_GEMMB_BIG", str(cfg))
    monkeypatch.setenv("LOKINFER_GEMMB_TGS", "256")  # force many splits
    monkeypatch.setenv("LOKINFER_GEMMB_MINKC", "16")
    _check(m, k, n)


def test_gemm_bias_gpu_legacy(fresh_env):
    monkeypatch = fresh_env
    monkeypatch.setenv("LOKINFER_GEMMB_LEGACY", "1")
    for m in (1, 64, 1024):
        _check(m, 784, 128)


def test_gemm_bias_gpu_split_repeated():
    """One compiled model, many calls with varying M: the split counters must reset after every call."""
    k, n = 784, 128
    r = np.random.default_rng(1)
    b = (r.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    bias = r.standard_normal(n).astype(np.float32)
    bt, biast = lk.from_numpy(b, device="gpu"), lk.from_numpy(bias, device="gpu")
    M = lk.Sym("M")
    model = lk.compile(lambda x: ops.gemm_bias_relu_f32_gpu(x, bt, biast), input_types=[lk.TensorType(lk.f32, [M, k])])
    for i in range(60):
        m = [1, 3, 8, 9, 64, 33, 200, 1][i % 8]
        a = r.standard_normal((m, k)).astype(np.float32)
        ref = np.maximum(a.astype(np.float64) @ b + bias, 0)
        out = model(lk.from_numpy(a, device="gpu"))
        out = out[0] if isinstance(out, (list, tuple)) else out
        np.testing.assert_allclose(out.numpy(), ref, atol=2e-4 * max(1, np.abs(ref).max()), rtol=1e-4)



@pytest.mark.parametrize("smallm", ["64", "1024"])
@pytest.mark.parametrize("m,k,n", [(9, 100, 50), (64, 784, 128), (33, 300, 70), (1024, 128, 64)])
def test_gemm_bias_gpu_smallm_threshold(fresh_env, smallm, m, k, n):
    """The split-K small-M kernel over many row blocks (LOKINFER_GEMMB_SMALLM raises its threshold)."""
    fresh_env.setenv("LOKINFER_GEMMB_SMALLM", smallm)
    fresh_env.setenv("LOKINFER_GEMMB_TGS", "256")
    _check(m, k, n)
