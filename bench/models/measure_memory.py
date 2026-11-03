"""Peak memory per model (docs/runtime.md "Memory footprint"): each (model, graph, case) runs in a FRESH executor
process (a local lokinfer-serve, or LokiHost on the iPhone restarted through SHUTDOWN + the deploy watchdog), so the
process peaks belong to that model alone. The executor reports task_vm_info (phys_footprint = what iOS's memory
limit counts; resident) and MTLDevice.currentAllocatedSize through the remote MEMORY message.

Usage: .venv/bin/python bench/models/measure_memory.py [mac|iphone] [model ...]   (models: kokoro mnist resnet18
       mobilenet_v2 dcgan_g dcgan_d gpt2; default all). Kokoro: the workload cases app59f + s28f; others: one case.
"""
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "models"))
from bench_remote_models import UDID, suites  # noqa: E402

from lokinfer.remote import RemoteDevice  # noqa: E402

# A Release executor: the Debug build's metallib is -O0 (cmake/LokiMetal.cmake), which makes long GPU cases take minutes.
SERVE = Path(os.environ.get("LOKINFER_SERVE", ROOT / "build-release" / "lokinfer-serve"))
CASES = {"kokoro": ["app59f", "s28f"]}  # others: their first case
RUNS = 3


def mib(b):
    return b / (1 << 20)


def fresh_device(where, blobs):
    """A new executor process with nothing loaded."""
    if where == "mac":
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        proc = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", blobs], stdout=subprocess.DEVNULL)
        return RemoteDevice.connect("127.0.0.1", port, timeout=900.0), proc
    # iPhone: the previous measurement shut LokiHost down; the deploy watchdog relaunches it in server mode.
    return RemoteDevice.connect(f"{os.environ.get('LOKINFER_IPHONE', UDID)}.coredevice.local", timeout=900.0,
                                retries=120), None


def shutdown(dev, proc):
    try:
        dev.shutdown()
    finally:
        dev.close()
        if proc:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def model_blob(model):
    import hashlib

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.lkm"
        model.save(p)
        data = p.read_bytes()
    return hashlib.sha256(data).hexdigest(), data


def measure(where, model, inputs, blobs):
    sha, data = model_blob(model)
    dev, proc = fresh_device(where, blobs)
    if not dev.has_blob(sha):
        # Upload in a throwaway process: the measured one finds the blob cached (no upload buffer in its peak).
        dev.put_blob(data)
        shutdown(dev, proc)
        dev, proc = fresh_device(where, blobs)
    try:
        base = dev.memory()
        mid = dev.load_blob(sha)
        loaded = dev.memory()
        dev.run(mid, inputs, warmup=0, iters=RUNS)
        after = dev.memory()
    finally:
        shutdown(dev, proc)
    return base, loaded, after


def main():
    args = sys.argv[1:]
    where = args.pop(0) if args and args[0] in ("mac", "iphone") else "mac"
    keys = args or ["kokoro", "mnist", "resnet18", "mobilenet_v2", "dcgan_g", "dcgan_d", "gpt2"]
    all_suites = suites()
    rows = []
    with tempfile.TemporaryDirectory() as blobs:
        for key in keys:
            title, cases, impls, _ = all_suites[key]()
            names = CASES.get(key, [next(iter(cases))])
            for impl, (model, _) in impls.items():
                if impl == "lokinfer accelerate":
                    continue
                kind = "GPU" if "GPU" in impl else "CPU"
                for case in names:
                    base, loaded, after = measure(where, model, cases[case], blobs)
                    row = (title, kind, case,
                           mib(after["phys_footprint_peak"] - base["phys_footprint"]),
                           mib(after["phys_footprint"] - base["phys_footprint"]),
                           mib(after["resident_peak"] - base["resident"]),
                           mib(after["gpu_allocated"] - base["gpu_allocated"]),
                           mib(loaded["phys_footprint"] - base["phys_footprint"]))
                    rows.append(row)
                    print("%-12s %-3s %-14s peak footprint +%8.1f MiB, steady +%8.1f, resident peak +%8.1f, "
                          "GPU allocated +%8.1f, after load +%8.1f" % row, flush=True)
                    time.sleep(0.5)
    print("\n| model | graph | case | peak footprint | steady footprint | peak resident | GPU allocated | after load |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print("| %s | %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f |" % r)


if __name__ == "__main__":
    main()
