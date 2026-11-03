"""Executable::setProfile / takeProfileJson through the Python binding, the remote PROFILE / TAKE_PROFILE /
GPU_INSPECT messages against a local lokinfer-serve, and gpu_inspect."""
import json
import os
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest

import lokinfer as lk
from lokinfer import _lokinfer as _C
from lokinfer import ops
from lokinfer.prof import analysis
from lokinfer.remote import RemoteDevice

ROOT = Path(__file__).resolve().parents[2]
SERVE = Path(os.environ.get("LOKINFER_BUILD_DIR", ROOT / "build")) / "lokinfer-serve"


def model_for(dev):
    w = lk.randn(32, 16, seed=1, device=dev)
    gemm = ops.gemm_f32_cpu if dev == "cpu" else ops.gemm_f32_gpu
    add = ops.add_f32_cpu if dev == "cpu" else ops.add_f32_gpu
    N = lk.Sym("N")
    return lk.compile(lambda x: add(gemm(x, w), gemm(x, w)), input_types=[lk.TensorType(lk.f32, [N, 32])])


def x(n=64):
    return np.random.default_rng(n).standard_normal((n, 32)).astype(np.float32)


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_profile_records_calls(dev):
    m = model_for(dev)
    m(x())
    exe = m._exe
    assert json.loads(exe.take_profile())["calls"] == []  # off by default: nothing recorded
    exe.set_profile(1)
    for n in (64, 8):
        m(x(n))
    exe.set_profile(0)
    m(x())
    p = json.loads(exe.take_profile())
    assert p["version"] == 1 and p["uses_gpu"] == (dev == "gpu")
    assert len(p["calls"]) == 2 and json.loads(exe.take_profile())["calls"] == []  # take clears
    names = sorted(o[1] for o in p["ops"])
    assert len(names) == 3 and all(n.endswith(f"_{dev}") for n in names)
    for call in p["calls"]:
        assert call["total_ns"] > 0 and len(call["kernels"]) == 3
        for k in call["kernels"]:
            op, t0, t1, g0, g1, ints = k[:6]
            assert 0 <= t0 <= t1 <= call["total_ns"]
            if dev == "gpu" and call["gpu_timestamps"]:
                assert 0 <= g0 <= g1, k  # GPU work happens inside the call
                assert g1 <= call["total_ns"] + 1_000_000
            else:
                assert g0 == -1 and g1 == -1
        assert call["gpu_mode"] == ("stream" if dev == "gpu" else "none")
    # dynamic dims are recorded per call: N = 64 then 8
    gemm_op = next(o for o in p["ops"] if o[1].startswith("gemm"))
    dims = [next(k for k in c["kernels"] if k[0] == gemm_op[0])[5][0] for c in p["calls"]]
    assert dims == [64, 8]
    if dev == "gpu":
        disp = p["calls"][0]["dispatches"]
        assert len(disp) == 3 and {d[1] for d in disp} == {"gemm_f32", "add_f32"}
    run = analysis.build_run(p, meta={"id": "t", "model": "m", "case": "c", "device": dev, "target": "mac",
                                      "backend": dev}, e2e_ms=[1.0])
    assert {k["name"] for k in run["kernels"]} == set(names)


def test_profile_gpu_serialize():
    m = model_for("gpu")
    m(x())
    m._exe.set_profile(1, True)
    m(x())
    m._exe.set_profile(0)
    call = json.loads(m._exe.take_profile())["calls"][0]
    assert call["gpu_mode"] == "serialize"
    gk = sorted((k[3], k[4]) for k in call["kernels"])
    assert all(a[1] <= b[0] for a, b in zip(gk, gk[1:]))  # serialized: GPU intervals never overlap


def test_gpu_inspect_stats():
    info = json.loads(_C.gpu_inspect(["add_f32", "no_such_function"], ""))
    assert info["device"]["max_threadgroup_memory"] > 0 and info["device"]["timestamp_frequency"] > 0
    f = {r["name"]: r for r in info["functions"]}
    assert f["add_f32"]["max_total_threads_per_threadgroup"] >= 32 and f["add_f32"]["thread_execution_width"] == 32
    assert "error" in f["no_such_function"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not SERVE.exists():
        pytest.skip("lokinfer-serve not built")
    port = free_port()
    proc = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", str(tmp_path_factory.mktemp("blobs"))],
                            stdout=subprocess.DEVNULL)
    dev = RemoteDevice.connect("127.0.0.1", port)
    yield dev
    dev.shutdown()
    dev.close()
    proc.wait(timeout=10)


@pytest.mark.parametrize("dev", ["cpu", "gpu"])
def test_remote_profile(server, dev):
    assert server.supports("profile") and server.supports("gpu_inspect")
    assert server.info["device"] == "mac"  # the original HELLO fields are kept
    mid = server.load(model_for(dev))
    server.run(mid, [x()], warmup=1, iters=1)
    server.set_profile(mid, 1)
    server.run(mid, [x()], warmup=0, iters=3)
    server.set_profile(mid, 0)
    server.run(mid, [x()], warmup=0, iters=2)
    p = server.take_profile(mid)
    assert len(p["calls"]) == 3 and all(len(c["kernels"]) == 3 for c in p["calls"])
    assert server.take_profile(mid)["calls"] == []


def test_remote_gpu_inspect(server, tmp_path):
    info, blob = server.gpu_inspect(["add_f32"], archive=True)
    assert info["functions"][0]["name"] == "add_f32" and "max_total_threads_per_threadgroup" in info["functions"][0]
    assert len(blob) > 1000  # a binary archive with native code
    info, blob = server.gpu_inspect([], archive=False)
    assert len(info["functions"]) > 10 and blob == b""
