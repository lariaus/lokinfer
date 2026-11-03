"""Remote executor protocol, against a local `lokinfer-serve` (same server code
as the iPhone test host). The iPhone variant runs with --device=iphone (tests/conftest.py)."""
import os
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest

import lokinfer as lk
from lokinfer import ops
from lokinfer.remote import RemoteDevice, RemoteError

ROOT = Path(__file__).resolve().parents[2]
SERVE = Path(os.environ.get("LOKINFER_BUILD_DIR", ROOT / "build")) / "lokinfer-serve"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def device(remote_device):
    """The executor for --device (tests/conftest.py): a local lokinfer-serve, or the phone."""
    return remote_device


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_remote_matches_local(device, dev):
    w = lk.randn(32, 16, seed=1, device=dev)  # constants live on the device that uses them
    gemm = ops.gemm_f32_cpu if dev == "cpu" else ops.gemm_f32_gpu
    N = lk.Sym("N")
    model = lk.compile(lambda x: gemm(x, w), input_types=[lk.TensorType(lk.f32, [N, 32])])
    mid = device.load(model)
    for n in [1, 9, 100]:
        x = np.random.default_rng(n).standard_normal((n, 32)).astype(np.float32)
        (out,), times = device.run(mid, [x], warmup=1, iters=3)
        np.testing.assert_allclose(out, model(x).numpy(), rtol=1e-5, atol=1e-5)
        assert len(times) == 3 and all(t > 0 for t in times)


def test_blob_cache_and_errors(device):
    model = lk.compile(lambda a: ops.add_f32_cpu(a, a), inputs=[lk.randn(4)])
    a, b = device.load(model), device.load(model)  # second upload skipped (HAS_BLOB)
    assert a != b
    with pytest.raises(RemoteError, match="unknown model"):
        device.run(12345, [np.zeros(4, np.float32)])
    with pytest.raises(RemoteError, match="dim 0"):
        device.run(a, [np.zeros(5, np.float32)])


def test_remote_scalar_input(device):
    """Regression: 0-d inputs were sent as shape (1,) (np.ascontiguousarray promotes 0-d to 1-d)."""
    model = lk.compile(lambda x, s: ops.mul_f32_cpu(x, x),
                       input_types=[lk.TensorType(lk.f32, [4]), lk.ScalarType(lk.f32)])
    mid = device.load(model)
    x = np.arange(4, dtype=np.float32)
    (out,), _ = device.run(mid, [x, np.array(2.0, np.float32)])
    np.testing.assert_array_equal(out, x * x)


def test_put_tensor_run2_and_device_compare(device):
    """Content-addressed inputs / references (sent once), inputs generated on the device, comparisons on the device."""
    from lokinfer.remote import Blob, Generated, generate

    model = lk.compile(lambda x, y: ops.mul_f32_cpu(ops.add_f32_cpu(x, y), x),
                       input_types=[lk.TensorType(lk.f32, [lk.Sym("N")])] * 2)
    mid = device.load(model)
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    y = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
    ref = (x + y) * x
    xs, rs = device.put_tensor(x), device.put_tensor(ref)
    assert xs == device.put_tensor(x.copy())  # same content, same handle (no second upload)
    outs, stats, times = device.run2(mid, [xs, y], compare=[(0, rs, 1e-6, 1e-6)], iters=2)
    np.testing.assert_array_equal(outs[0], ref)
    assert stats[0].ok and stats[0].count == 1000 and stats[0].max_abs_err == 0 and len(times) == 2
    wrong = ref.copy()
    wrong[17] += 1
    _, stats, _ = device.run2(mid, [xs, y], compare=[(0, device.put_tensor(wrong), 0.0, 1e-3)], return_outputs=False)
    assert stats[0].mismatches == 1 and stats[0].worst == 17 and abs(stats[0].max_abs_err - 1) < 1e-5

    # Generated inputs: the Mac reproduces the device's values (normals to an ulp: libm log/cos may round differently).
    g = Generated([1000], seed=42)
    gu = Generated([1000], seed=43, dist="uniform", a=-2, b=3)
    outs, _, _ = device.run2(mid, [g, gu])
    gx, gy = generate([1000], 42), generate([1000], 43, dist="uniform", a=-2, b=3)
    assert -2 <= gy.min() and gy.max() < 3 and abs(gx.mean()) < 0.1 and abs(gx.std() - 1) < 0.1
    np.testing.assert_allclose(outs[0], (gx + gy) * gx, rtol=1e-6, atol=1e-6)
    ints = generate([10000], 5, np.int64, "int", 3, 9)
    assert ints.min() == 3 and ints.max() == 8


def test_memory_report(device):
    m = device.memory()
    assert m["phys_footprint"] > 0 and m["phys_footprint_peak"] >= m["phys_footprint"]
    assert m["resident_peak"] >= m["resident"] > 0 and m["gpu_allocated"] >= 0


@pytest.mark.local_only
def test_hostile_frames_against_lokinfer_serve(tmp_path):
    """A few hostile frames against a real lokinfer-serve process (the full fuzz suite is the C++ [remote][fuzz]
    tests): it must answer errors or drop the client, then keep serving."""
    import struct
    import subprocess
    import time

    from lokinfer.remote import MAGIC, VERSION, RemoteDevice

    if not SERVE.exists():
        pytest.skip("lokinfer-serve not built")
    port = free_port()
    proc = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", str(tmp_path)], stdout=subprocess.DEVNULL)
    try:
        def frame(mtype, payload, magic=MAGIC, length=None):
            return struct.pack("<IHHIIQ", magic, VERSION, mtype, 1, 0, len(payload) if length is None else length) + payload

        def exchange(data, expect_reply):
            for _ in range(100):
                try:
                    s = socket.create_connection(("127.0.0.1", port), timeout=10)
                    break
                except OSError:
                    time.sleep(0.05)
            with s:
                s.sendall(data)
                s.shutdown(socket.SHUT_WR)
                got = b""
                while chunk := s.recv(65536):
                    got += chunk
            if expect_reply:
                assert len(got) >= 24 and struct.unpack_from("<I", got, 12)[0] == 1, got[:64]  # error flag
            return got

        exchange(frame(1, b"", magic=0x12345678), False)            # bad magic: dropped
        exchange(frame(3, b"x" * 100, length=(1 << 31) - 1), False)  # 2 GiB claimed, 100 bytes sent: dropped
        exchange(frame(5, struct.pack("<IIII", 1, 0, 1, 1) + struct.pack("<IIqQ", 0, 1, -4, 16) + bytes(16)), True)
        exchange(frame(5, struct.pack("<IIII", 1, 0, 1, 1) + struct.pack("<IIqQ", 256, 1, 4, 16) + bytes(16)), True)
        exchange(frame(9, b"\x00\xff\xfe\n\x80"), True)            # non-UTF-8 function names (used to abort)
        exchange(frame(4242, b"junk"), True)
        with RemoteDevice.connect("127.0.0.1", port) as dev:  # still serving
            assert "features" in dev.info
            dev.shutdown()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
