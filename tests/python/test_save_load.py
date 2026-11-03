"""Export (.lkm, internalized weights) and reload."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_save_load_with_captured_weights(tmp_path, dev):
    w = lk.randn(16, 8, seed=3, device=dev)  # constants live on the device that uses them
    b = lk.randn(4, 8, seed=4, device=dev)
    gemm = ops.gemm_f32_cpu if dev == "cpu" else ops.gemm_f32_gpu
    add = ops.add_f32_cpu if dev == "cpu" else ops.add_f32_gpu

    def fn(x):
        return add(gemm(x, w), b)

    x = lk.randn(4, 16, seed=5)
    model = lk.compile(fn, inputs=[x])
    ref = model(x).numpy().copy()
    path = tmp_path / "m.lkm"
    model.save(path)
    del model, w, b  # the file must be self-contained
    loaded = lk.load(path)
    np.testing.assert_array_equal(loaded(x).numpy(), ref)
    assert "const.internal" in loaded.ir_text()
    assert "const.external" not in loaded.ir_text()


def test_load_rejects_garbage(tmp_path):
    p = tmp_path / "bad.lkm"
    p.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="lkm"):
        lk.load(p)
