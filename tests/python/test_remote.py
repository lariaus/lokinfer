"""Remote executor protocol, against a local `lokinfer-serve` (same server code
as the iPhone test host). The iPhone variant runs with --device=iphone (tests/conftest.py)."""
import os
import socket
import struct
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


def _put_counter(device, monkeypatch):
    """Sizes of the blobs actually uploaded (PUT_BLOB) through `device`."""
    sent = []
    orig = device._call

    def spy(mtype, payload):
        if mtype == 3:  # PUT_BLOB: sha + data
            sent.append(len(payload) - 64)
        return orig(mtype, payload)

    monkeypatch.setattr(device, "_call", spy)
    return sent


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_split_model_weights_cached(device, dev, monkeypatch):
    """Split transport ("split_models"): the graph and each weight are separate blobs, so a weight already on the
    device is never sent again: the same weights in another graph upload only the new graph's skeleton."""
    if not device.supports("split_models"):
        pytest.skip("server without split_models")
    gemm = ops.gemm_f32_cpu if dev == "cpu" else ops.gemm_f32_gpu
    add = ops.add_f32_cpu if dev == "cpu" else ops.add_f32_gpu
    seed = 1000 + np.random.default_rng().integers(1 << 30)  # fresh weights: not on the device from an earlier run
    w1, w2 = lk.randn(256, 64, seed=int(seed), device=dev), lk.randn(64, 64, seed=int(seed) + 1, device=dev)
    N = lk.Sym("N")
    a = lk.compile(lambda x: gemm(gemm(x, w1), w2), input_types=[lk.TensorType(lk.f32, [N, 256])])
    b = lk.compile(lambda x: add(gemm(gemm(x, w1), w2), gemm(gemm(x, w1), w2)),
                   input_types=[lk.TensorType(lk.f32, [N, 256])])
    sent = _put_counter(device, monkeypatch)
    ma = device.load(a)
    weights = 4 * (256 * 64 + 64 * 64)
    assert sum(sent) >= weights  # first upload: the weights + a skeleton
    sent.clear()
    mb = device.load(b)  # another graph over the same weights: only its skeleton goes up
    assert sum(sent) < weights / 4, sent
    x = np.random.default_rng(3).standard_normal((5, 256)).astype(np.float32)
    (oa,), _ = device.run(ma, [x])
    (ob,), _ = device.run(mb, [x])
    np.testing.assert_allclose(oa, a(x).numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ob, b(x).numpy(), rtol=1e-5, atol=1e-5)
    sent.clear()
    device.load(a)  # everything cached
    assert sent == []


def test_blobs_report_and_remove(device):
    """BLOBS ("blobs"): the store's size, and removing listed blobs (never clear_blobs here: on the phone the store
    holds the cached model weights)."""
    if not device.supports("blobs"):
        pytest.skip("server without blobs")
    data = np.random.default_rng().bytes(40_000)  # fresh: not on the device yet
    before = device.blobs()
    sha = device.put_blob(data)
    assert device.has_blob(sha)
    mid = device.blobs()
    assert mid["count"] == before["count"] + 1 and mid["bytes"] == before["bytes"] + len(data)
    assert device.stored_blobs([sha, "0" * 64, sha]) == [True, False, True]
    r = device.remove_blobs([sha, "0" * 64])  # an unknown sha is ignored
    assert r["removed"] == 1 and r["removed_bytes"] == len(data)
    assert not device.has_blob(sha)
    assert device.blobs()["count"] == before["count"]
    with pytest.raises(Exception):
        device.remove_blobs(["xyz"])


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


# ---- device tensors and mutable inputs (feature "device_tensors")

P, D = 8, 4


def _rows_model():
    """f32 CPU: kc[pos + i] = k_i, vc[pos + i] = v_i for qkv f32[T, 3D] (q | k | v), caches mutable inputs."""
    def fn(qkv, pos, kc, vc):
        ops.kv_write_rows_f32_cpu(kc, vc, qkv, pos)
        return ops.add_f32_cpu(qkv, qkv)

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [lk.Sym("T"), 3 * D]), lk.TensorType(lk.i64, [1]),
                                       lk.MutableType(lk.f32, [P, D]), lk.MutableType(lk.f32, [P, D])])


def _f16_model(dev: str):
    """f16 cache f16[2, P, D] (one mutable input; k and v caches are views of it): kc[pos:pos+T] = k, vc[...] = v."""
    from lokinfer.graph import current_graph
    from lokinfer.ops._def import _as_value

    T = lk.Sym("T")

    def fn(k, v, pos, kv):
        g = current_graph()
        kc = g.view(_as_value(g, kv), lk.MutableType(lk.f16, [P, D]), 0)
        vc = g.view(_as_value(g, kv), lk.MutableType(lk.f16, [P, D]), P * D)
        if dev == "cpu":
            ops.f16.kv_write(kc, vc, k, v, pos)
            return ops.f16.add(k, v)
        kg, vg = ops.to_gpu(k), ops.to_gpu(v)
        ops.f16_gpu.kv_write_f16_gpu(kc, vc, kg, vg, pos)
        return ops.to_cpu(ops.f16_gpu.add_f16_gpu(kg, vg))

    return lk.compile(fn, input_types=[lk.TensorType(lk.f16, [T, D]), lk.TensorType(lk.f16, [T, D]),
                                       lk.TensorType(lk.i64, [1]), lk.MutableType(lk.f16, [2, P, D])])


def test_device_tensor_round_trip(device):
    from lokinfer.remote import FREE_TENSOR, READ_TENSOR, WRITE_TENSOR

    assert device.supports("device_tensors")
    for dtype, shape in [(np.float32, (3, 5)), (np.float16, (2, 4, 6)), (np.int64, (7,)), (np.float32, ())]:
        t = device.alloc_tensor(dtype, shape)
        z = device.read_tensor(t)
        assert z.dtype == dtype and z.shape == shape and not z.any()  # zeroed
        a = (np.arange(int(np.prod(shape, dtype=np.int64))) - 3).astype(dtype).reshape(shape)
        device.write_tensor(t, a)
        np.testing.assert_array_equal(device.read_tensor(t), a)
        device.free_tensor(t)
        with pytest.raises(RemoteError, match="unknown device tensor"):
            device.read_tensor(t)
        with pytest.raises(RemoteError, match="unknown device tensor"):
            device.free_tensor(t)
    t = device.alloc_tensor(np.float32, (4,))
    with pytest.raises(ValueError):
        device.write_tensor(t, np.zeros(5, np.float32))  # checked by the client ...
    import struct

    from lokinfer.remote import _encode_tensor
    with pytest.raises(RemoteError, match="dtype or shape"):  # ... and by the server
        device._call(WRITE_TENSOR, struct.pack("<I", t.handle) + _encode_tensor(np.zeros(5, np.float32)))
    with pytest.raises(RemoteError, match="dtype or shape"):
        device._call(WRITE_TENSOR, struct.pack("<I", t.handle) + _encode_tensor(np.zeros(4, np.int32)))
    with pytest.raises(RemoteError, match="trailing"):
        device._call(WRITE_TENSOR, struct.pack("<I", t.handle) + _encode_tensor(np.zeros(4, np.float32)) + b"x")
    with pytest.raises(RemoteError, match="bad READ_TENSOR"):
        device._call(READ_TENSOR, struct.pack("<II", t.handle, 0))
    with pytest.raises(RemoteError, match="bad FREE_TENSOR"):
        device._call(FREE_TENSOR, b"\x01")
    with pytest.raises(RemoteError, match="bad tensor shape"):
        device.alloc_tensor(np.float32, (1 << 31,))  # 8 GiB: above the 2 GiB frame cap (the 4 GiB budget: C++ fuzz)
    with pytest.raises(RemoteError, match="bad tensor shape"):
        device.alloc_tensor(np.float32, (-1, 4))
    device.free_tensor(t)


def test_mutable_input_device_tensor_rows_f32(device):
    """Several RUN2 calls write the device tensors in place: the rows accumulate on the device."""
    model = _rows_model()
    mid = device.load(model)
    kc, vc = device.alloc_tensor(np.float32, (P, D)), device.alloc_tensor(np.float32, (P, D))
    ek, ev = np.zeros((P, D), np.float32), np.zeros((P, D), np.float32)
    pos = 0
    for step, t in enumerate([3, 1, 2]):
        qkv = np.random.default_rng(step).standard_normal((t, 3 * D)).astype(np.float32)
        outs, _, _ = device.run2(mid, [qkv, np.array([pos], np.int64), kc, vc])
        np.testing.assert_array_equal(outs[0], qkv + qkv)
        ek[pos:pos + t], ev[pos:pos + t] = qkv[:, D:2 * D], qkv[:, 2 * D:]
        pos += t
        np.testing.assert_array_equal(device.read_tensor(kc), ek)
        np.testing.assert_array_equal(device.read_tensor(vc), ev)
    # Inline / blob / generated mutable inputs are refused (the writes would land in a copy), in RUN and RUN2.
    from lokinfer.remote import Generated

    qkv, p0 = np.zeros((1, 3 * D), np.float32), np.array([0], np.int64)
    z = np.zeros((P, D), np.float32)
    with pytest.raises(RemoteError, match="input 2 is mutable: pass a device tensor"):
        device.run2(mid, [qkv, p0, z, vc])
    with pytest.raises(RemoteError, match="input 3 is mutable: pass a device tensor"):
        device.run2(mid, [qkv, p0, kc, device.put_tensor(z)])
    with pytest.raises(RemoteError, match="input 2 is mutable: pass a device tensor"):
        device.run2(mid, [qkv, p0, Generated((P, D), seed=1), vc])
    with pytest.raises(RemoteError, match="input 2 is mutable: pass a device tensor"):
        device.run(mid, [qkv, p0, z, z])
    with pytest.raises(RemoteError, match="overlaps"):  # one device tensor as both caches
        device.run2(mid, [qkv, p0, kc, kc])
    bad = device.alloc_tensor(np.float32, (P + 1, D))
    with pytest.raises(RemoteError, match="dim 0"):
        device.run2(mid, [qkv, p0, bad, vc])
    np.testing.assert_array_equal(device.read_tensor(kc), ek)  # failed calls wrote nothing
    for t in (kc, vc, bad):
        device.free_tensor(t)
    from lokinfer.remote import DeviceTensor

    with pytest.raises(RemoteError, match="unknown device tensor"):
        device.run2(mid, [qkv, p0, DeviceTensor(kc.handle, np.float32, (P, D)), vc])


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_mutable_input_device_tensor_f16(device, dev):
    """f16 KV cache as one mutable device tensor (CPU kv_write / GPU kv_write_f16_gpu). On the GPU the device tensor
    must be bound with no copy: the writes of every call persist."""
    model = _f16_model(dev)
    mid = device.load(model)
    kv = device.alloc_tensor(np.float16, (2, P, D))
    expect = np.zeros((2, P, D), np.float16)
    pos = 0
    for step, t in enumerate([2, 1, 1, 3]):
        rng = np.random.default_rng(10 + step)
        k = rng.standard_normal((t, D)).astype(np.float16)
        v = rng.standard_normal((t, D)).astype(np.float16)
        outs, _, _ = device.run2(mid, [k, v, np.array([pos], np.int64), kv])
        np.testing.assert_array_equal(outs[0], k + v)
        expect[0, pos:pos + t], expect[1, pos:pos + t] = k, v
        pos += t
        np.testing.assert_array_equal(device.read_tensor(kv), expect)
    # Overwrite from the Mac, then one more call: it sees (and keeps) the written contents.
    seeded = np.full((2, P, D), 0.5, np.float16)
    device.write_tensor(kv, seeded)
    k = np.ones((1, D), np.float16)
    device.run2(mid, [k, 2 * k, np.array([P - 1], np.int64), kv])
    seeded[0, P - 1], seeded[1, P - 1] = 1, 2
    np.testing.assert_array_equal(device.read_tensor(kv), seeded)
    with pytest.raises(RemoteError, match="is mutable: pass a device tensor"):
        device.run2(mid, [k, k, np.array([0], np.int64), seeded])
    device.free_tensor(kv)


def _root_conftest(config):
    """tests/conftest.py as pytest loaded it (`import conftest` can resolve to tests/models/conftest.py instead when
    both directories are collected in one run)."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    for plugin in config.pluginmanager.get_plugins():
        if os.path.abspath(getattr(plugin, "__file__", "") or "") == root:
            return plugin
    raise RuntimeError("tests/conftest.py is not loaded")


@pytest.fixture
def remote_calls(device, device_name, monkeypatch, request):
    """CompiledModel.__call__ routed through `device` (tests/conftest.py), as with --device=serve / iphone."""
    conftest = _root_conftest(request.config)
    from lokinfer import compiler

    if device_name not in conftest.REMOTE:
        monkeypatch.setattr(compiler.CompiledModel, "__call__", compiler.CompiledModel.__call__)  # restored after
        monkeypatch.setitem(conftest._state, "device", device)
        conftest._install_remote_call()
    return device


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_mutable_inputs_through_remote_call(remote_calls, dev, request):
    """The transparent --device=serve / iphone path: a mutable input is written into the caller's own array / tensor
    exactly as a local run does (device tensor in, run, read back). The device tensors are mirrors kept across calls
    (conftest._mirror: re-uploaded only when the host contents changed), bounded, one per argument slot / dtype /
    shape."""
    state = _root_conftest(request.config)._state
    live = remote_calls.memory()["device_tensors"] - len(state.get("mirrors", {}))
    rows = _rows_model()
    kc, vc = np.zeros((P, D), np.float32), np.zeros((P, D), np.float32)
    lkc = lk.from_numpy(np.zeros((P, D), np.float32))  # a lokinfer Tensor as the caller's cache
    ek = np.zeros((P, D), np.float32)
    pos = 0
    for step, t in enumerate([2, 3]):
        qkv = np.random.default_rng(step).standard_normal((t, 3 * D)).astype(np.float32)
        out = rows(qkv, np.array([pos], np.int64), kc, vc)
        np.testing.assert_array_equal(out.numpy(), qkv + qkv)
        rows(qkv, np.array([pos], np.int64), lkc, vc)
        ek[pos:pos + t] = qkv[:, D:2 * D]
        pos += t
        np.testing.assert_array_equal(kc, ek)
        np.testing.assert_array_equal(lkc.numpy(), ek)
    model = _f16_model(dev)
    kv = np.zeros((2, P, D), np.float16) if dev == "cpu" else lk.zeros(2, P, D, dtype=lk.f16, device="gpu")
    k, v = np.full((2, D), 1.5, np.float16), np.full((2, D), -2, np.float16)
    out = model(k, v, np.array([3], np.int64), kv)
    np.testing.assert_array_equal(out.numpy(), k + v)
    got = kv if dev == "cpu" else kv.numpy()
    expect = np.zeros((2, P, D), np.float16)
    expect[0, 3:5], expect[1, 3:5] = k, v
    np.testing.assert_array_equal(got, expect)
    assert remote_calls.memory()["device_tensors"] == live + len(state["mirrors"])  # only the mirrors stay
    assert _root_conftest(request.config)._state["remote_runs"] >= 5  # the calls went through the device


def test_lkm_split_matches_saved_file(tmp_path):
    """lkm_split (RemoteDevice.load's in-memory split) gives exactly the bytes a saved .lkm splits into: the skeleton
    = the file's first data_offset bytes, each constant = its lkm_constant_ranges slice."""
    from lokinfer import _lokinfer as _C

    w1, w2 = lk.randn(64, 32, seed=5), lk.randn(32, 16, seed=6)  # captured (external) constants
    model = lk.compile(lambda x: ops.gemm_f32_cpu(ops.gemm_f32_cpu(x, w1), w2), inputs=[lk.randn(3, 64)])
    path = tmp_path / "m.lkm"
    model.save(path)
    data = path.read_bytes()
    data_offset = struct.unpack_from("<Q", data, 24)[0]
    skeleton, views = _C.lkm_split(model._model, model._externals)
    assert skeleton == data[:data_offset]
    ranges = _C.lkm_constant_ranges(str(path))
    assert len(views) == len(ranges)
    for v, (off, size) in zip(views, ranges):
        assert bytes(memoryview(v)) == data[off:off + size]


def test_reconnect_in_place(tmp_path):
    """RemoteDevice.reconnect(): the same object gets a new connection after the old one broke (the test harness uses
    it so one server crash fails one test); the old session's models are gone, `session` counts up."""
    if not SERVE.exists():
        pytest.skip("lokinfer-serve not built")
    port = free_port()
    proc = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", str(tmp_path)], stdout=subprocess.DEVNULL)
    try:
        dev = RemoteDevice.connect("127.0.0.1", port)
        model = lk.compile(lambda a: ops.add_f32_cpu(a, a), inputs=[lk.randn(4)])
        mid = dev.load(model)
        dev._sock.close()  # break the connection under it
        with pytest.raises(Exception):
            dev.run(mid, [np.ones(4, np.float32)])
        assert dev.broken
        dev.reconnect()
        assert not dev.broken and dev.session == 1
        with pytest.raises(RemoteError):
            dev.run(mid, [np.ones(4, np.float32)])  # the old session's model is gone
        assert not dev.broken  # a server-side error keeps the stream usable
        mid2 = dev.load(model)
        (out,), _ = dev.run(mid2, [np.ones(4, np.float32)])
        np.testing.assert_array_equal(out, np.full(4, 2, np.float32))
        dev.shutdown()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_free_model(device):
    """FREE_MODEL ("free_model"): a model goes at once; its id is unknown afterwards."""
    if not device.supports("free_model"):
        pytest.skip("server without free_model")
    model = lk.compile(lambda a: ops.add_f32_cpu(a, a), inputs=[lk.randn(4)])
    mid = device.load(model)
    device.run(mid, [np.ones(4, np.float32)])
    device.free_model(mid)
    with pytest.raises(RemoteError):
        device.run(mid, [np.ones(4, np.float32)])
    with pytest.raises(RemoteError):
        device.free_model(mid)


def test_memory_pressure_and_pin_model(device):
    """MEMORY's "memory_pressure" fields and PIN_MODEL ("pin_model"): a model's mapped weights are counted, resident
    after a run, and can be wired (mlock) and unwired again."""
    if not (device.supports("memory_pressure") and device.supports("pin_model")):
        pytest.skip("server without memory_pressure / pin_model")
    w = lk.randn(256, 512, seed=3)  # 512 KiB: its own mapped blob in a split load
    model = lk.compile(lambda x: ops.gemm_f32_cpu(x, w), input_types=[lk.TensorType(lk.f32, [4, 256])])
    m0 = device.memory()
    assert m0["pressure"] in ("normal", "warn", "critical") and m0["pressure_events"] >= 0
    assert m0["pageins"] >= 0 and m0["page_size"] in (4096, 16384) and m0["system_free"] > 0
    ios = device.info.get("device", "").startswith(("iPhone", "iPad"))
    assert m0["available"] > 0 if ios else m0["available"] == -1  # os_proc_available_memory: iOS only
    mid = device.load(model)
    device.run(mid, [np.ones((4, 256), np.float32)])
    m1 = device.memory()
    assert m1["weights"] - m0["weights"] >= w.numpy().nbytes
    assert m1["weights_resident"] - m0["weights_resident"] >= w.numpy().nbytes
    r = device.pin_model(mid, "willneed")
    assert r["bytes"] >= w.numpy().nbytes and r["error"] == "" and r["locked"] == 0
    r = device.pin_model(mid, "lock")
    assert r["error"] == "" and r["locked"] == r["bytes"] and r["resident"] == r["bytes"]
    assert device.memory()["weights_locked"] - m0["weights_locked"] == r["bytes"]
    (out,), _ = device.run(mid, [np.ones((4, 256), np.float32)])
    np.testing.assert_allclose(out, model(np.ones((4, 256), np.float32)).numpy(), rtol=1e-5, atol=1e-5)
    r = device.pin_model(mid, "unlock")
    assert r["locked"] == 0 and r["error"] == ""
    r = device.pin_model(mid, "gpu")  # no-copy Metal buffers in the residency set (a CPU-only model)
    assert r["error"] == "" and r["locked"] == r["bytes"] and device.memory()["weights_locked"] - m0["weights_locked"] == r["bytes"]
    (out,), _ = device.run(mid, [np.ones((4, 256), np.float32)])
    np.testing.assert_allclose(out, model(np.ones((4, 256), np.float32)).numpy(), rtol=1e-5, atol=1e-5)
    assert device.pin_model(mid, "unlock")["locked"] == 0
    r = device.pin_model(mid, "gpu")
    device.free_model(mid)  # with its GPU pins
    ok, why = device.memory_check()
    assert isinstance(ok, bool) and "pressure" in why
    with pytest.raises(RemoteError):
        device.pin_model(987654, "lock")
