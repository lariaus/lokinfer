"""Device parameter for the Python graph suite (docs/correctness-testing.md "On the iPhone").

    pytest --device=iphone            # or LOKINFER_DEVICE=iphone; the phone's UDID from LOKINFER_IPHONE or devicectl
    pytest --device=serve             # the same remote path against a local lokinfer-serve (validates it on the Mac)

With --device=iphone every compiled graph (`CompiledModel.__call__`, which also serves eager kernel calls) runs on
the phone through the remote executor: the model is uploaded once by content hash, the inputs go over the cable,
and the phone's outputs are what the test checks (they are written into the local result tensors, so the test sees
the usual `lokinfer.Tensor`s with the right device and dtype). The graph also runs on the Mac to produce those
result objects; nothing of its output survives. A model's mutable inputs (lk.MutableType) are written in place on
the device: each call allocates a device tensor per mutable input, writes the caller's array into it, runs (RUN2,
bound in place), reads it back into the caller's own memory (so the caller sees the writes exactly as on the Mac;
the Mac run's writes are overwritten) and frees it. Slow but exact, fine for tests with small caches; real iPhone
use (benchmarks, Qwen3-ASR transcription) keeps the cache on the device (`RemoteDevice.alloc_tensor`) and never
transfers it. A test that never ran a graph on the phone (tensor interop,
save/load, registry, profiling tools) is reported as skipped, so a pass always means "passed on the phone".

`remote_device` (session fixture) is the executor for the selected device: the phone, or a local `lokinfer-serve`
on the Mac (same server code), for tests that talk to the remote protocol directly (RUN2 sweeps, memory).
"""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVE = Path(os.environ.get("LOKINFER_BUILD_DIR", ROOT / "build")) / "lokinfer-serve"

_state = {"device": None, "remote_runs": 0, "proc": None}
REMOTE = ("serve", "iphone")  # devices where compiled graphs run through the remote executor


def pytest_addoption(parser):
    parser.addoption("--device", choices=["mac", "serve", "iphone"], default=os.environ.get("LOKINFER_DEVICE", "mac"),
                     help="where compiled graphs run (iphone: through the remote executor; env LOKINFER_DEVICE)")
    # Test tiers: plain `pytest` runs the default tier (fast, every code path); `slow` tests run at the end of a
    # milestone, `long` ones (e.g. Qwen3-ASR's 20 min pass) before a release.
    parser.addoption("--run-slow", action="store_true", default=bool(os.environ.get("LOKINFER_RUN_SLOW")),
                     help="also run tests marked slow (milestone tier; env LOKINFER_RUN_SLOW)")
    parser.addoption("--run-long", action="store_true", default=bool(os.environ.get("LOKINFER_RUN_LONG")),
                     help="also run tests marked long (release tier; implies --run-slow; env LOKINFER_RUN_LONG)")


def iphone_udid() -> str:
    udid = os.environ.get("LOKINFER_IPHONE")
    if udid:
        return udid
    out = subprocess.run(["xcrun", "devicectl", "list", "devices"], capture_output=True, text=True).stdout
    import re

    for line in out.splitlines():
        if "physical" in line and "unavailable" not in line:
            m = re.search(r"[0-9A-F]{8}-[0-9A-F]{16}", line)
            if m:
                return m.group(0)
    raise pytest.UsageError("--device=iphone: no LOKINFER_IPHONE and no connected iPhone")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def device_name(pytestconfig) -> str:
    return pytestconfig.getoption("--device")


@pytest.fixture(autouse=True)
def _remote_reconnect():
    """Before every test: a remote session broken by an earlier test (server crash / kill) is reconnected, so one
    crash fails one test instead of every test after it."""
    dev = _state["device"]
    if dev is not None and dev.broken:
        dev.reconnect()
        _state["mirrors"] = {}
    _state["test_models"] = []
    yield
    # Free the models this test loaded through the harness (FREE_MODEL; their cached ids go too): the server keeps a
    # connection's models until it ends, and a long --device=iphone session piled them up until iOS killed LokiHost
    # (2026-11-06). Fixtures that hold their own models / device tensors on the session are not touched (a reconnect
    # here would free those too).
    dev = _state["device"]
    held, _state["test_models"] = _state.get("test_models", []), []
    if dev is not None and not dev.broken and dev.supports("free_model"):
        session = dev.__dict__.get("session", 0)
        for ref, mid in held:
            m = ref()
            if m is not None:
                m.__dict__.pop("_lk_remote_id", None)
            if mid[0] == session:
                dev.free_model(mid[1])
    if dev is not None and not dev.broken and os.environ.get("LOKINFER_TEST_MEMLOG") and dev.supports("memory"):
        m = dev.memory()  # diagnostic: the executor's memory after every test (MiB)
        print(f"\n[memlog] " + " ".join(f"{k}={v / 2**20:.0f}" if isinstance(v, int) and v > 2**20 else f"{k}={v}"
                                       for k, v in m.items()), flush=True)


@pytest.fixture(scope="session")
def remote_device(pytestconfig, tmp_path_factory):
    """RemoteDevice for the selected device (the phone, or a local lokinfer-serve)."""
    from lokinfer.remote import RemoteDevice

    if pytestconfig.getoption("--device") in REMOTE:
        if _state["device"] is None:
            _state["device"] = RemoteDevice.connect(f"{iphone_udid()}.coredevice.local", timeout=900.0)
            ok, why = _state["device"].memory_check()
            if not ok:  # correctness still holds, but big models may page their weights from flash (slow)
                print(f"\nWARNING: the device reports memory pressure ({why}); close other apps", flush=True)
        yield _state["device"]
        return
    if not SERVE.exists():
        pytest.skip("lokinfer-serve not built")
    port = _free_port()
    proc = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", str(tmp_path_factory.mktemp("blobs"))],
                            stdout=subprocess.DEVNULL)
    try:
        dev = RemoteDevice.connect("127.0.0.1", port)
        yield dev
        dev.shutdown()
        dev.close()
        proc.wait(timeout=10)
    finally:  # a failed call (or shutdown) must not leak the server (leaked ones piled up, 2026-11-06)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _to_array(a):
    from lokinfer.tensor import Tensor

    if isinstance(a, Tensor):
        return np.ascontiguousarray(a.numpy())
    if isinstance(a, np.ndarray):
        return a
    if isinstance(a, bool):
        raise TypeError("bool scalar inputs are not supported")
    if isinstance(a, float):
        return np.array(a, np.float32)
    if isinstance(a, int):
        return np.array(a, np.int64)
    if hasattr(a, "__dlpack__"):  # torch
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _mutable_view(a) -> np.ndarray:
    """A writable numpy view of a mutable input's own memory (the device's writes are copied back into it)."""
    from lokinfer.tensor import Tensor

    if isinstance(a, Tensor):
        return a.numpy()  # zero-copy (unified memory, GPU tensors too)
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "__dlpack__") and getattr(getattr(a, "device", None), "type", "cpu") == "cpu":  # torch (CPU)
        return a.detach().numpy()
    raise TypeError(f"mutable input of type {type(a).__name__}: no host view to write the device's results into")


# Device-resident copies of mutable inputs kept across calls. 256 MiB: with 1 GiB, LokiHost was killed (signal 9) late in
# a long --device=iphone session (every model of the session stays loaded until disconnect, the mirrors came on top);
# big caches belong in explicit device tensors anyway.
_MIRROR_BYTES = 256 << 20


def _digest(v: np.ndarray) -> bytes:
    import hashlib

    return hashlib.blake2b(np.ascontiguousarray(v).view(np.uint8).ravel(), digest_size=16).digest()


def _mirror(dev, v: np.ndarray, slot: int):
    """The device tensor mirroring a mutable host array: one per (argument slot, dtype, shape) (two mutable arguments
    of one call never share one), reused across calls, written only
    when the host contents differ from what the device holds (a digest of the last synced contents). Correct for any
    host array (same contents = same device data); least recently used mirrors are freed past _MIRROR_BYTES."""
    m = _state.setdefault("mirrors", {})  # (slot, dtype, shape) -> [DeviceTensor, digest]
    key = (slot, v.dtype.str, tuple(v.shape))
    if key not in m:
        while m and sum(e[0].nbytes for e in m.values()) + v.nbytes > _MIRROR_BYTES:
            old = next(iter(m))
            dev.free_tensor(m.pop(old)[0])
        m[key] = [dev.alloc_tensor(v.dtype, v.shape), None]
    m[key] = m.pop(key)  # most recently used last
    t, d = m[key]
    h = _digest(v)
    if h != d:
        dev.write_tensor(t, v)
        m[key][1] = h
    return t


def _mirror_synced(t, v: np.ndarray):
    """After a call: the device tensor and the host array (just copied back) hold the same contents."""
    for e in _state.get("mirrors", {}).values():
        if e[0] is t:
            e[1] = _digest(v)


def _install_remote_call():
    from lokinfer import compiler
    from lokinfer.tensor import Tensor, from_numpy

    local_call = compiler.CompiledModel.__call__

    def remote_call(self, *args):
        dev = _state["device"]
        if dev.broken:
            # The server died in an earlier test (2026-11-06: one LokiHost kill failed the 108 tests after it on the
            # dead connection): reconnect (the watchdog relaunches it) and drop what lived in the old session.
            dev.reconnect()
            _state["mirrors"] = {}
        session = dev.__dict__.get("session", 0)
        mid = self.__dict__.get("_lk_remote_id")
        if mid is None or mid[0] != session:
            mid = (session, dev.load(self))  # content-addressed: uploaded once per distinct model, cached on the phone
            import weakref

            # by id, not only by object: models built as temporaries are gone by the test's end
            _state.setdefault("test_models", []).append((weakref.ref(self), mid))
            self._lk_remote_id = mid
        mid = mid[1]
        from lokinfer.types import BufferType

        mut = [i for i, t in enumerate(self.input_types) if isinstance(t, BufferType) and i < len(args)]
        if not mut:
            outs, _ = dev.run(mid, [_to_array(a) for a in args])
        else:  # mutable inputs: written in place on the device (a device tensor each), then copied back
            views = {i: _mutable_view(args[i]) for i in mut}
            dts = {i: _mirror(dev, v, i) for i, v in views.items()}
            outs, _, _ = dev.run2(mid, [dts[i] if i in dts else _to_array(a) for i, a in enumerate(args)])
            for i, v in views.items():
                np.copyto(v, dev.read_tensor(dts[i]))
                _mirror_synced(dts[i], v)
        devices = self.__dict__.get("_lk_out_devices")
        local = None
        if devices is not None:
            if len(devices) != len(outs):
                raise AssertionError(f"device returned {len(outs)} outputs, expected {len(devices)}")
            try:
                local = [from_numpy(np.ascontiguousarray(o), device=d, copy=d != "cpu") for o, d in zip(outs, devices)]
                res = local[0] if self._single else tuple(local)
            except (TypeError, ValueError):  # a dtype from_numpy can't wrap: build them through the local run
                local = None
        if local is None:
            # The first call per model also runs locally, only to learn the result objects' devices (and to check
            # the output count / types); later calls build them from the device's outputs without a Mac run (it
            # used to re-run every call: a whole extra Qwen3-ASR forward per call). Mutable inputs: the local run's
            # writes are undone (the device's values, copied back above, are restored).
            saved = {i: v.copy() for i, v in views.items()} if mut else {}
            res = local_call(self, *args)
            for i, v in saved.items():
                np.copyto(views[i], v)
            local = [res] if isinstance(res, Tensor) else list(res)
            if len(local) != len(outs):
                raise AssertionError(f"device returned {len(outs)} outputs, expected {len(local)}")
            for t, o in zip(local, outs):
                dst = t.numpy()
                if dst.shape != o.shape or dst.dtype != o.dtype:
                    raise AssertionError(f"device output {o.dtype}{o.shape} vs expected {dst.dtype}{dst.shape}")
                np.copyto(dst, o)
            self._lk_out_devices = [t.device for t in local]
        _state["remote_runs"] += 1
        return res

    compiler.CompiledModel.__call__ = remote_call


def pytest_configure(config):
    config.addinivalue_line("markers", "local_only: never runs on --device=iphone / serve (Mac-only tooling)")
    config.addinivalue_line("markers", "long: very long (minutes to an hour); runs only with --run-long")
    dev = config.getoption("--device")
    if dev not in REMOTE:
        return
    from lokinfer.remote import RemoteDevice

    if dev == "iphone":
        _state["device"] = RemoteDevice.connect(f"{iphone_udid()}.coredevice.local", timeout=900.0)
    else:
        import tempfile

        if not SERVE.exists():
            raise pytest.UsageError(f"--device=serve: {SERVE} not built")
        port = _free_port()
        _state["blobs"] = tempfile.TemporaryDirectory()
        _state["proc"] = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", _state["blobs"].name],
                                          stdout=subprocess.DEVNULL)
        _state["device"] = RemoteDevice.connect("127.0.0.1", port, timeout=900.0)  # big models load for a while
    _install_remote_call()


def pytest_unconfigure(config):
    if _state["proc"] is not None:
        try:
            _state["device"].shutdown()
            _state["device"].close()
            _state["proc"].wait(timeout=30)
        finally:  # never leak the server, even when the connection already failed
            if _state["proc"].poll() is None:
                _state["proc"].kill()
                _state["proc"].wait()
            _state["proc"] = None


def pytest_collection_modifyitems(config, items):
    run_long = config.getoption("--run-long")
    run_slow = config.getoption("--run-slow") or run_long
    skip_slow = pytest.mark.skip(reason="slow tier: pass --run-slow (milestone runs)")
    skip_long = pytest.mark.skip(reason="long tier: pass --run-long (release runs)")
    for it in items:
        if it.get_closest_marker("long") and not run_long:
            it.add_marker(skip_long)
        elif it.get_closest_marker("slow") and not run_slow:
            it.add_marker(skip_slow)
    if config.getoption("--device") not in REMOTE:
        return
    skip = pytest.mark.skip(reason="Mac-only (profiling tools / local perf)")
    for it in items:
        p = str(it.fspath)
        if "/tests/tools/" in p or "/tests/perf/" in p or it.get_closest_marker("local_only"):
            it.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _state["remote_runs"] = 0
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if (call.when == "call" and rep.passed and item.config.getoption("--device") in REMOTE
            and _state["remote_runs"] == 0 and "remote_device" not in item.fixturenames):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), item.location[1] or 0, "Skipped: ran no graph on the remote device")
