"""MNIST MLP: lokinfer variants vs PyTorch (CPU eager, MPS), f32.

Median of rounds (each round = mean over N iterations, N sized for ~0.2 s),
after warmup. Records Low Power Mode / thermal state with the results.
Usage: .venv/bin/python bench/models/bench_mnist.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models import mnist_mlp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perfdb  # noqa: E402  (dashboard store: perf/results.json, docs/perf-dashboard.md)

# results key -> (dashboard device, dashboard impl); values are seconds here, milliseconds in the store
PERF_IMPLS = {"torch-cpu": ("Mac CPU", "torch eager"), "torch-mps": ("Mac GPU", "torch MPS"),
              "lokinfer-cpu": ("Mac CPU", "lokinfer"), "lokinfer-cpu-accelerate": ("Mac CPU", "lokinfer accelerate"),
              "lokinfer-gpu": ("Mac GPU", "lokinfer GPU")}


def machine_state() -> dict:
    pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    low = any(l.split()[:1] == ["lowpowermode"] and l.split()[1] == "1" for l in pm.splitlines())
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()
    return {"low_power_mode": low, "thermal": therm.splitlines()[-1] if therm else ""}


def timeit(fn, rounds=7, target_s=0.2) -> float:
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    fn()
    one = max(time.perf_counter() - t0, 1e-6)
    n = max(1, int(target_s / one))
    means = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        means.append((time.perf_counter() - t0) / n)
    return float(np.median(means))


def main():
    ref = mnist_mlp.torch_reference()
    weights = mnist_mlp.weights_from_torch(ref)
    models = {v: mnist_mlp.build(weights, v) for v in mnist_mlp.VARIANTS}
    mps_ref = mnist_mlp.torch_reference().to("mps")
    results = {"machine": machine_state(), "rows": []}
    perf_state = perfdb.machine_state()  # Low Power Mode / AC / thermal, stored with each dashboard entry
    print(f"machine: {results['machine']}")
    print(f"{'B':>6} {'impl':>16} {'ms':>9}")
    for b in [1, 64, 1024]:
        x = np.random.default_rng(0).standard_normal((b, 1, 28, 28)).astype(np.float32)
        xt = torch.from_numpy(x)
        xm = xt.to("mps")
        row = {"B": b}

        def torch_cpu():
            with torch.inference_mode():
                ref(xt)

        def torch_mps():
            with torch.inference_mode():
                mps_ref(xm)
            torch.mps.synchronize()

        row["torch-cpu"] = timeit(torch_cpu)
        row["torch-mps"] = timeit(torch_mps)
        for v, m in models.items():
            xl = __import__("lokinfer").from_numpy(x, device="gpu" if v == "gpu" else "cpu")
            row[f"lokinfer-{v}"] = timeit(lambda: m(xl))
        for k, val in row.items():
            if k != "B":
                print(f"{b:>6} {k:>16} {val * 1e3:9.4f}")
        results["rows"].append(row)
        for key, (device, impl) in PERF_IMPLS.items():
            if key in row:
                perfdb.record("MNIST MLP", device, f"B={b}", impl, row[key] * 1e3, state=perf_state)
    out = Path(__file__).with_name("results_mnist.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
