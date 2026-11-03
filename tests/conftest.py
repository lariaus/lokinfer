"""Device parameter for the Python graph suite (docs/correctness-testing.md "On the iPhone").

    pytest --device=iphone            # or LOKINFER_DEVICE=iphone; the phone's UDID from LOKINFER_IPHONE or devicectl
    pytest --device=serve             # the same remote path against a local lokinfer-serve (validates it on the Mac)

With --device=iphone every compiled graph (`CompiledModel.__call__`, which also serves eager kernel calls) runs on
the phone through the remote executor: the model is uploaded once by content hash, the inputs go over the cable,
and the phone's outputs are what the test checks (they are written into the local result tensors, so the test sees
the usual `lokinfer.Tensor`s with the right device and dtype). The graph also runs on the Mac to produce those
result objects; nothing of its output survives. A test that never ran a graph on the phone (tensor interop,
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


@pytest.fixture(scope="session")
def remote_device(pytestconfig, tmp_path_factory):
    """RemoteDevice for the selected device (the phone, or a local lokinfer-serve)."""
    from lokinfer.remote import RemoteDevice

    if pytestconfig.getoption("--device") in REMOTE:
        if _state["device"] is None:
            _state["device"] = RemoteDevice.connect(f"{iphone_udid()}.coredevice.local", timeout=900.0)
        yield _state["device"]
        return
    if not SERVE.exists():
        pytest.skip("lokinfer-serve not built")
    port = _free_port()
    proc = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", str(tmp_path_factory.mktemp("blobs"))],
                            stdout=subprocess.DEVNULL)
    dev = RemoteDevice.connect("127.0.0.1", port)
    yield dev
    dev.shutdown()
    dev.close()
    proc.wait(timeout=10)


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


def _install_remote_call():
    from lokinfer import compiler
    from lokinfer.tensor import Tensor

    local_call = compiler.CompiledModel.__call__

    def remote_call(self, *args):
        dev = _state["device"]
        mid = self.__dict__.get("_lk_remote_id")
        if mid is None:
            mid = dev.load(self)  # content-addressed: uploaded once per distinct model, cached on the phone
            self._lk_remote_id = mid
        outs, _ = dev.run(mid, [_to_array(a) for a in args])
        res = local_call(self, *args)
        local = [res] if isinstance(res, Tensor) else list(res)
        if len(local) != len(outs):
            raise AssertionError(f"device returned {len(outs)} outputs, expected {len(local)}")
        for t, o in zip(local, outs):
            dst = t.numpy()
            if dst.shape != o.shape or dst.dtype != o.dtype:
                raise AssertionError(f"device output {o.dtype}{o.shape} vs expected {dst.dtype}{dst.shape}")
            np.copyto(dst, o)
        _state["remote_runs"] += 1
        return res

    compiler.CompiledModel.__call__ = remote_call


def pytest_configure(config):
    config.addinivalue_line("markers", "local_only: never runs on --device=iphone / serve (Mac-only tooling)")
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
        _state["device"].shutdown()
        _state["device"].close()
        _state["proc"].wait(timeout=30)
        _state["proc"] = None


def pytest_collection_modifyitems(config, items):
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
