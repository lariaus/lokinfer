"""Phase-6 goal graphs, through the Python API (trace -> compile -> run),
checked against numpy references."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops

DEVICES = ["cpu", "gpu"]


def rand(*shape, seed=0):
    return lk.randn(*shape, seed=seed)


def add_fn(dev):
    return getattr(ops, f"add_f32_{dev}")


def mul_fn(dev):
    return getattr(ops, f"mul_f32_{dev}")


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("n", [1, 7, 1000, 4099])
def test_static_vector_add(dev, n):
    x, y = rand(n, seed=1), rand(n, seed=2)
    model = lk.compile(lambda a, b: add_fn(dev)(a, b), inputs=[x, y])
    assert model.uses_gpu == (dev == "gpu")
    out = model(x, y)
    np.testing.assert_array_equal(out.numpy(), x.numpy() + y.numpy())


@pytest.mark.parametrize("dev", DEVICES)
def test_dynamic_vector_add(dev):
    # Two inputs sharing a dynamic dim: explicit input_types (dynamic=True gives
    # every input dim its own symbol, by design).
    N = lk.Sym("N")
    t = lk.TensorType(lk.f32, [N])
    model = lk.compile(lambda a, b: add_fn(dev)(a, b), input_types=[t, t])
    for n in [1, 5, 1000, 100_003]:
        x, y = rand(n, seed=n), rand(n, seed=n + 1)
        np.testing.assert_array_equal(model(x, y).numpy(), x.numpy() + y.numpy())


@pytest.mark.parametrize("dev", DEVICES)
def test_dynamic_true_single_input(dev):
    model = lk.compile(lambda a: add_fn(dev)(a, a), inputs=[rand(8)], dynamic=True)
    for n in [3, 4096]:
        x = rand(n, seed=n)
        np.testing.assert_array_equal(model(x).numpy(), 2 * x.numpy())


def test_dynamic_true_gives_independent_dims():
    with pytest.raises(ops.OpError, match="shape mismatch"):
        lk.compile(lambda a, b: ops.add_f32_cpu(a, b), inputs=[rand(8), rand(8)], dynamic=True)


@pytest.mark.parametrize("dev", DEVICES)
def test_parametric_input_types_are_checked(dev):
    N = lk.Sym("N")
    t = lk.TensorType(lk.f32, [N, N])
    model = lk.compile(lambda a, b: add_fn(dev)(a, b), input_types=[t, t])
    x = rand(3, 3)
    np.testing.assert_array_equal(model(x, x).numpy(), 2 * x.numpy())
    with pytest.raises(RuntimeError, match="expected N = 3"):
        model(rand(3, 4), rand(3, 4))


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dynamic", [False, True])
def test_add_mul(dev, dynamic):
    def fn(a, b, c):
        return mul_fn(dev)(add_fn(dev)(a, b), c)

    n = 2049
    x, y, z = rand(n, seed=1), rand(n, seed=2), rand(n, seed=3)
    if dynamic:
        t = lk.TensorType(lk.f32, [lk.Sym("N")])
        model = lk.compile(fn, input_types=[t, t, t])
    else:
        model = lk.compile(fn, inputs=[x, y, z])
    sizes = [n, 17, 50_000] if dynamic else [n, n]
    for i, m in enumerate(sizes):
        a, b, c = rand(m, seed=10 + i), rand(m, seed=20 + i), rand(m, seed=30 + i)
        np.testing.assert_array_equal(model(a, b, c).numpy(), (a.numpy() + b.numpy()) * c.numpy())


GEMMS = [
    ("cpu", ops.gemm_f32_cpu),
    ("gpu", ops.gemm_f32_gpu),
    ("cpu-accelerate", ops.gemm_f32_cpu_accelerate),
    ("gpu-mps", ops.gemm_f32_gpu_mps),
]


@pytest.mark.parametrize("name,op", GEMMS)
@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (37, 53, 29), (64, 64, 64), (128, 96, 200)])
def test_static_matmul(name, op, m, n, k):
    a, b = rand(m, k, seed=1), rand(k, n, seed=2)
    model = lk.compile(lambda x, y: op(x, y), inputs=[a, b])
    ref = a.numpy().astype(np.float64) @ b.numpy().astype(np.float64)
    np.testing.assert_allclose(model(a, b).numpy(), ref, rtol=1e-4, atol=1e-4 * np.sqrt(k))


def test_weights_are_captured_as_external_constants():
    w = rand(16, 8, seed=5)

    def fn(x):
        return ops.gemm_f32_cpu(x, w)

    x = rand(4, 16, seed=6)
    model = lk.compile(fn, inputs=[x])
    assert 'const.external @"ext0"' in model.source_ir
    np.testing.assert_allclose(model(x).numpy(), x.numpy() @ w.numpy(), rtol=1e-5, atol=1e-5)


def test_eager_kernel_call():
    x, y = rand(10, seed=1), rand(10, seed=2)
    np.testing.assert_array_equal(ops.add_f32_cpu(x, y).numpy(), x.numpy() + y.numpy())


def test_type_rule_errors_at_build_time():
    with pytest.raises(ops.OpError, match="shape mismatch"):
        lk.compile(lambda a, b: ops.add_f32_cpu(a, b), inputs=[rand(3), rand(4)])


def test_mixed_devices_need_explicit_boundaries():
    def implicit(a):
        return ops.add_f32_cpu(ops.add_f32_gpu(a, a), a)

    with pytest.raises(RuntimeError, match="add ops.to_cpu"):
        lk.compile(implicit, inputs=[rand(16)])

    def explicit(a):
        g = ops.add_f32_gpu(a, a)
        return ops.mul_f32_cpu(ops.to_cpu(g), a)

    x = rand(4096, seed=3)
    model = lk.compile(explicit, inputs=[x])
    assert "sync gpu_to_cpu" in model.source_ir
    np.testing.assert_allclose(model(x).numpy(), (x.numpy() * 2) * x.numpy(), rtol=1e-6)


def test_gpu_constant_device_and_init_hoisting():
    # GPU model end to end: weights resident on the GPU, the GPU transpose is a
    # kernel on a constant, so it is hoisted into init (runs once at load).
    w = lk.randn(48, 32, seed=7, device="gpu")
    x = rand(5, 32, seed=8)
    model = lk.compile(lambda a: ops.gemm_f32_gpu(a, ops.transpose_f32_gpu(w)), inputs=[x])
    final = model.ir_text()
    init, exec_ = final[: final.index("exec(")], final[final.index("exec("):]
    assert 'kernel "transpose_f32_gpu"' in init and "transpose" not in exec_
    assert 'device = "gpu"' in model.source_ir
    np.testing.assert_allclose(model(x).numpy(), x.numpy() @ w.numpy().T, rtol=1e-4, atol=1e-4)


def test_cpu_constant_on_gpu_kernel_is_rejected():
    w = lk.randn(8, 8, seed=1)  # cpu
    with pytest.raises(RuntimeError, match="resident on cpu"):
        lk.compile(lambda a: ops.gemm_f32_gpu(a, w), inputs=[rand(4, 8)])


def test_gpu_model_inputs_are_not_copied():
    # A CPU tensor (numpy-backed, unaligned) fed to a GPU model is wrapped, not
    # copied: writes to the numpy array after compile are seen by the next call.
    a = np.arange(1000, dtype=np.float32)
    x = lk.from_numpy(a)
    model = lk.compile(lambda t: ops.add_f32_gpu(t, t), inputs=[x])
    np.testing.assert_array_equal(model(x).numpy(), 2 * a)
    a[:] = 1.0  # same memory, new contents
    np.testing.assert_array_equal(model(x).numpy(), np.full(1000, 2.0, np.float32))
    b = np.arange(1, 1001, dtype=np.float32)[1:]  # different, offset array
    y = lk.from_numpy(np.ascontiguousarray(b))
    m2 = lk.compile(lambda t: ops.add_f32_gpu(t, t), inputs=[y])
    np.testing.assert_array_equal(m2(y).numpy(), 2 * b)


def test_temporary_constants_not_aliased():
    """Regression: graph.external cached by id(); a freed temporary Tensor's id could be reused by the
    next temporary, binding the wrong constant."""
    import numpy as np

    from lokinfer import ops

    def fn(x):
        for i in range(50):
            x = ops.add_f32_cpu(x, lk.from_numpy(np.full(4, float(i), np.float32), copy=True))
        return x

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f32, [4])])
    out = model(lk.from_numpy(np.zeros(4, np.float32), copy=True)).numpy()
    assert (out == sum(range(50))).all()
