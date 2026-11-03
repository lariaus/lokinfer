"""Tensor API + interop (zero-copy where possible)."""
import numpy as np
import pytest

import lokinfer as lk


def test_numpy_round_trip_zero_copy():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = lk.from_numpy(a)
    assert t.shape == (3, 4) and t.dtype is lk.f32 and t.device == "cpu"
    assert t.data_ptr() == a.ctypes.data  # zero-copy import
    v = t.numpy()
    v[0, 0] = 42
    assert a[0, 0] == 42  # zero-copy export aliases the same memory


def test_copy_and_gpu_tensors():
    a = np.arange(5, dtype=np.float32)
    g = lk.from_numpy(a, device="gpu")
    assert g.device == "gpu" and g.data_ptr() != a.ctypes.data
    np.testing.assert_array_equal(g.numpy(), a)
    assert g.cpu().device == "cpu"


def test_non_contiguous_input_is_copied_not_aliased():
    a = np.arange(10, dtype=np.float32)[::2]
    t = lk.from_numpy(a)
    np.testing.assert_array_equal(t.numpy(), a)
    assert t.data_ptr() != a.ctypes.data


@pytest.mark.parametrize("dtype", [lk.f32, lk.f16, lk.i32, lk.i64, lk.u8])
def test_dtypes(dtype):
    t = lk.tensor([[1, 2], [3, 4]], dtype=dtype)
    assert t.dtype is dtype
    np.testing.assert_array_equal(t.numpy(), np.array([[1, 2], [3, 4]], dtype=dtype.np))


def test_constructors_and_repr():
    z = lk.zeros(2, 3)
    assert z.shape == (2, 3) and float(z.numpy().sum()) == 0.0
    r = lk.randn(4, seed=1)
    np.testing.assert_array_equal(r.numpy(), lk.randn(4, seed=1).numpy())
    assert "Tensor(" in repr(r)
    assert np.asarray(r).shape == (4,)


def test_dims():
    N, M = lk.Sym("N"), lk.Sym("M")
    assert N + M == M + N
    assert N * 2 == 2 * N
    assert (N + 1) - 1 == N
    assert str((N + 1) // 2) == "(N + 1) // 2"
    assert lk.ceildiv(4 * N, 4) == N
    assert lk.Dim.const(3).value == 3


def test_torch_interop_if_available():
    torch = pytest.importorskip("torch")
    x = torch.arange(6, dtype=torch.float32)
    t = lk.from_torch(x)
    assert t.data_ptr() == x.data_ptr()
    back = t.torch()
    back[0] = 7
    assert x[0] == 7
