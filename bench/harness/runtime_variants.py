"""CPU runtime variants (original plan: mono-thread vs thread pool vs P-core-pinned QoS) on MNIST / MobileNetV2 /
Kokoro, Mac and iPhone (docs/benchmarking.md "CPU runtime variants").

  .venv/bin/python bench/harness/runtime_variants.py --mac      # each variant in a child process with its env
  .venv/bin/python bench/harness/runtime_variants.py --iphone   # relaunches LokiHost per variant (phone lock!)

Variants are environment knobs read once by the runtime's thread pool (src/runtime/thread_pool.cpp):
  mono-thread          LOKINFER_THREADS=1 VECLIB_MAXIMUM_THREADS=1: every kernel inline on the calling thread
  pool, default QoS    LOKINFER_POOL_QOS=default: the pool's workers at QOS_CLASS_DEFAULT (the scheduler may use E-cores)
  pool, P-core QoS     the shipped default: workers at QOS_CLASS_USER_INTERACTIVE (macOS/iOS have no hard affinity;
                       this QoS is how a thread asks for P-cores)
Timing protocol: docs/perf-dashboard.md (warm-up >= 1 s, >= 30 runs and >= 2 s, median). Results ->
bench/harness/results_runtime_variants.json.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
OUT = HERE / "results_runtime_variants.json"

VARIANTS = {
    "mono-thread": {"LOKINFER_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"},
    "pool, default QoS": {"LOKINFER_POOL_QOS": "default"},
    "pool, P-core QoS (shipped)": {},
}
MODELS = [("mnist", "B=64"), ("mobilenet_v2", "N=1 224x224"), ("kokoro", "app59f")]
from lokinfer import localconfig  # noqa: E402

UDID = localconfig.require("iphone_udid")


def _state_mac() -> dict:
    sys.path.insert(0, str(ROOT / "bench"))
    import perfdb

    return perfdb.machine_state()


def child() -> None:
    """Runs in a process launched with one variant's env: times the three CPU models, prints one JSON line."""
    from lokinfer.prof.collect import timed
    from lokinfer.prof.targets import target_for

    out = {}
    for key, case in MODELS:
        tgt = target_for(key)
        m = tgt.build("cpu")
        args = tgt.inputs(case)
        ts = timed(lambda: m(*args), 1.0, 3, 30, 2.0)  # noqa: B023
        out[key] = {"case": case, "median_ms": statistics.median(ts), "n": len(ts)}
    out["state"] = _state_mac()
    print("RESULT " + json.dumps(out), flush=True)


def run_mac() -> dict:
    res = {}
    for name, env in VARIANTS.items():
        e = dict(os.environ)
        e.update(env)
        p = subprocess.run([sys.executable, __file__, "--child"], env=e, capture_output=True, text=True)
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            raise SystemExit(f"{name}: child failed\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        res[name] = json.loads(line[7:])
        print(name, {k: round(v["median_ms"], 3) for k, v in res[name].items() if k != "state"}, flush=True)
    return res


def run_iphone() -> dict:
    from lokinfer.prof.targets import target_for
    from lokinfer.remote import RemoteDevice

    sys.path.insert(0, str(ROOT / "bench" / "models"))
    built = {}
    for key, case in MODELS:
        tgt = target_for(key)
        built[key] = (tgt.build("cpu"), tgt.remote_inputs(case), case)
    res = {}
    try:
        for name, env in VARIANTS.items():
            subprocess.run([str(HERE / "lokihost.sh"), "launch", *[f"{k}={v}" for k, v in env.items()]], check=True)
            time.sleep(4)
            dev = RemoteDevice.connect(f"{UDID}.coredevice.local", timeout=600.0, retries=40)
            st = dev.state()
            want = " ".join(f"{k}={v}" for k, v in env.items())
            if st.get("runtime_env", "") != want:
                raise SystemExit(f"{name}: LokiHost runs with env '{st.get('runtime_env')}', wanted '{want}' "
                                 "(redeploy LokiHost: scripts/ios_deploy.sh)")
            r = {}
            for key, (model, inputs, case) in built.items():
                mid = dev.load(model)
                _, t = dev.run(mid, inputs, warmup=1, iters=3)
                est = max(statistics.median(t) / 1e6, 0.01)
                warm = int(min(5000, max(3, 1000 / est)))
                n = int(min(5000, max(30, 2000 / est)))
                _, times = dev.run(mid, inputs, warmup=warm, iters=n)
                r[key] = {"case": case, "median_ms": statistics.median(times) / 1e6, "n": len(times),
                          "thermal_end": dev.state().get("thermal")}
            r["state"] = {k: st.get(k) for k in ("thermal", "low_power_mode", "build", "runtime_env", "device")}
            dev.close()
            res[name] = r
            print(name, {k: round(v["median_ms"], 3) for k, v in r.items() if k != "state"}, r["state"], flush=True)
    finally:
        subprocess.run([str(HERE / "lokihost.sh"), "restore"])
    return res


def main():
    if "--child" in sys.argv:
        child()
        return
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if "--mac" in sys.argv:
        allres["Mac CPU"] = {"measured_at": stamp, "variants": run_mac()}
    if "--iphone" in sys.argv:
        allres["iPhone CPU"] = {"measured_at": stamp, "variants": run_iphone()}
    OUT.write_text(json.dumps(allres, indent=1) + "\n")


if __name__ == "__main__":
    main()
