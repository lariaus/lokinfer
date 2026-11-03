"""Call API (todo M6): model(x, out=y), submit()/wait(), internalize(), torch MPS zero-copy."""
import threading

import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops

DEVICES = ["cpu", "gpu"]

# The call API itself (out=, threads, model rewriting) is Mac-side; --device=iphone only intercepts plain calls.
pytestmark = pytest.mark.local_only


def add(dev):
    return getattr(ops, f"add_f32_{dev}")


def mul(dev):
    return getattr(ops, f"mul_f32_{dev}")


# ---- model(x, out=y) -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("kind", ["lokinfer", "numpy", "torch"])
def test_out_is_written_in_place(dev, kind):
    n = 1000
    x, y = lk.randn(n, seed=1), lk.randn(n, seed=2)
    model = lk.compile(lambda a, b: add(dev)(a, b), inputs=[x, y])
    if kind == "lokinfer":
        out = lk.empty(n, device=dev)
        view = out.numpy
    elif kind == "numpy":
        out = np.empty(n, np.float32)
        view = lambda: out  # noqa: E731
    else:
        import torch

        out = torch.empty(n)
        view = out.numpy
    for call in range(3):  # the same caller buffer, again and again (cached GPU wrap)
        x, y = lk.randn(n, seed=10 + call), lk.randn(n, seed=20 + call)
        r = model(x, y, out=out)
        assert r is out
        np.testing.assert_array_equal(view(), x.numpy() + y.numpy())


@pytest.mark.parametrize("dev", DEVICES)
def test_out_through_a_view_and_dynamic_shapes(dev):
    # The output is a reshape (zero-offset view) of the model's output buffer: still computed in place.
    N = lk.Sym("N")
    model = lk.compile(lambda a: ops.reshape(add(dev)(a, a), [2, N]), input_types=[lk.TensorType(lk.f32, [2 * N])])
    for n in [3, 500]:
        x = lk.randn(2 * n, seed=n)
        out = lk.empty(2, n, device=dev)
        assert model(x, out=out) is out
        np.testing.assert_array_equal(out.numpy(), (2 * x.numpy()).reshape(2, n))


def test_out_copy_when_the_output_is_an_input():
    # Not computed by the model (the output is the input): copied into out, never aliased.
    model = lk.compile(lambda a, b: (add("cpu")(a, b), a), inputs=[lk.randn(4), lk.randn(4)])
    x, y = lk.randn(4, seed=1), lk.randn(4, seed=2)
    o1, o2 = np.empty(4, np.float32), np.empty(4, np.float32)
    r1, r2 = model(x, y, out=(o1, o2))
    assert r1 is o1 and r2 is o2
    np.testing.assert_array_equal(o1, x.numpy() + y.numpy())
    np.testing.assert_array_equal(o2, x.numpy())


def test_out_none_entries_allocate():
    model = lk.compile(lambda a: (add("cpu")(a, a), mul("cpu")(a, a)), inputs=[lk.randn(5)])
    x = lk.randn(5, seed=3)
    o2 = np.empty(5, np.float32)
    r1, r2 = model(x, out=[None, o2])
    assert isinstance(r1, lk.Tensor) and r2 is o2
    np.testing.assert_array_equal(r1.numpy(), 2 * x.numpy())
    np.testing.assert_array_equal(o2, x.numpy() * x.numpy())


def test_out_shape_and_dtype_are_checked():
    model = lk.compile(lambda a: add("cpu")(a, a), inputs=[lk.randn(6)])
    with pytest.raises(RuntimeError, match="output 0"):
        model(lk.randn(6), out=np.empty(7, np.float32))
    with pytest.raises(RuntimeError, match="output 0"):
        model(lk.randn(6), out=np.empty(6, np.int32))
    with pytest.raises(RuntimeError, match="output tensors"):
        model(lk.randn(6), out=[np.empty(6, np.float32), None])


# ---- submit() / wait() -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("dev", DEVICES)
def test_submit_wait(dev):
    model = lk.compile(lambda a, b: mul(dev)(add(dev)(a, b), b), inputs=[lk.randn(4096), lk.randn(4096)])
    args = [(lk.randn(4096, seed=i), lk.randn(4096, seed=100 + i)) for i in range(8)]
    pending = [model.submit(x, y) for x, y in args]  # returns at once; runs in submit order
    for p, (x, y) in zip(pending, args):
        r = p.wait()
        assert p.done()
        np.testing.assert_allclose(r.numpy(), (x.numpy() + y.numpy()) * y.numpy(), rtol=1e-6)
    # Synchronous calls after submits keep working (and stay ordered with them).
    x, y = args[0]
    np.testing.assert_allclose(model(x, y).numpy(), (x.numpy() + y.numpy()) * y.numpy(), rtol=1e-6)


def test_submit_runs_off_the_calling_thread_and_with_out():
    model = lk.compile(lambda a: add("cpu")(a, a), inputs=[lk.randn(10)])
    x = lk.randn(10, seed=7)
    out = np.empty(10, np.float32)
    p = model.submit(x, out=out)
    assert p.wait() is out
    np.testing.assert_array_equal(out, 2 * x.numpy())
    assert model._worker is not None and threading.current_thread() is threading.main_thread()


def test_submit_propagates_errors():
    model = lk.compile(lambda a: add("cpu")(a, a), inputs=[lk.randn(10)])
    p = model.submit(lk.randn(11))  # wrong shape: the error comes out of wait()
    with pytest.raises(RuntimeError, match="expected"):
        p.wait()


# ---- internalize() ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dev", DEVICES)
def test_internalize_copies_captured_tensors(dev, tmp_path):
    w_np = np.arange(16, dtype=np.float32)
    w = lk.from_numpy(w_np.copy(), device=dev)
    model = lk.compile(lambda a: add(dev)(a, w), inputs=[lk.randn(16)])
    x = lk.randn(16, seed=1)
    w.numpy()[:] += 1000  # before internalize: the model aliases the captured tensor
    w_new = w_np + 1000
    np.testing.assert_array_equal(model(x).numpy(), x.numpy() + w_new)
    assert model.num_external_constants == 1
    assert model.internalize() is model
    assert model.num_external_constants == 0
    assert "const.external" not in model.ir_text()
    w.numpy()[:] = 0  # after: the model owns its copy
    del w
    np.testing.assert_array_equal(model(x).numpy(), x.numpy() + w_new)
    model.save(tmp_path / "m.lkm")  # still saves (nothing external left)
    np.testing.assert_array_equal(lk.load(tmp_path / "m.lkm")(x).numpy(), x.numpy() + w_new)


# ---- torch MPS zero-copy ---------------------------------------------------------------------------------------


def _mps():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    return torch


def test_mps_tensor_to_lokinfer_is_zero_copy():
    torch = _mps()
    x = torch.arange(64, dtype=torch.float32, device="mps")
    torch.mps.synchronize()
    t = lk.from_torch(x)  # wraps the same MTLBuffer
    assert t.device == "gpu"
    x.mul_(3)  # written by torch after the import
    torch.mps.synchronize()
    np.testing.assert_array_equal(t.numpy(), 3 * np.arange(64, dtype=np.float32))


def test_lokinfer_gpu_tensor_to_mps_is_zero_copy():
    torch = _mps()
    t = lk.from_numpy(np.arange(32, dtype=np.float32), device="gpu")
    m = t.torch(device="mps")
    assert m.device.type == "mps"
    m.add_(1)  # written by torch: visible through lokinfer's tensor (same buffer)
    torch.mps.synchronize()
    np.testing.assert_array_equal(t.numpy(), np.arange(32, dtype=np.float32) + 1)


def test_gpu_model_reads_and_writes_mps_tensors_without_copies():
    torch = _mps()
    n = 4096
    model = lk.compile(lambda a, b: add("gpu")(a, b), inputs=[lk.randn(n), lk.randn(n)])
    x = torch.randn(n, device="mps")
    y = torch.randn(n, device="mps")
    out = torch.empty(n, device="mps")
    torch.mps.synchronize()  # torch's writes must land before lokinfer reads (separate queues)
    for _ in range(2):
        assert model(x, y, out=out) is out
        torch.testing.assert_close(out, x + y)


# ---- bake_init() -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dev", DEVICES)
def test_bake_init_stores_transformed_weights(dev, tmp_path):
    w_np = np.random.default_rng(0).standard_normal((16, 24)).astype(np.float32)
    w = lk.from_numpy(w_np, device=dev)
    transpose = getattr(ops, f"transpose_f32_{dev}")
    gemm = getattr(ops, f"gemm_f32_{dev}")
    model = lk.compile(lambda x: gemm(x, transpose(w)), inputs=[lk.randn(5, 24)])  # transpose hoisted into init
    x = lk.randn(5, 24, seed=1)
    ref = model(x).numpy().copy()
    stats = model.bake_init()
    assert stats["baked"] == 1 and stats["baked_bytes"] == 24 * 16 * 4
    assert stats["init_ops_after"] < stats["init_ops_before"]
    ir = model.ir_text()
    assert f'transpose_f32_{dev}' not in ir and "baked" in ir
    assert f'device = "{dev}"' in ir  # resident where the init kernel wrote it
    np.testing.assert_allclose(model(x).numpy(), ref, rtol=1e-5, atol=1e-5)
    # Export: the file holds w^T only (the original w is gone with the init work that read it).
    path = tmp_path / "baked.lkm"
    lk.compile(lambda x: gemm(x, transpose(w)), inputs=[lk.randn(5, 24)]).save(path, bake_init=True)
    loaded = lk.load(path)
    assert "transpose" not in loaded.ir_text()
    np.testing.assert_allclose(loaded(x).numpy(), ref, rtol=1e-5, atol=1e-5)
