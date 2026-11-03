"""prof-lokinfer collection: run a model on Mac CPU / Mac GPU (in process) or iPhone CPU / GPU (remote executor in
LokiHost), time it with the dashboard protocol, record a per-kernel profile, dump the IR after every pass, and write
a run directory (run.json + raw profile + IR files + report.html).
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import platform
import socket
import statistics
import time
from pathlib import Path

from .. import localconfig
from . import analysis
from .targets import ROOT, Target, target_for

DEVICES = {"mac-cpu": ("mac", "cpu", "Mac CPU"), "mac-gpu": ("mac", "gpu", "Mac GPU"),
           "iphone-cpu": ("iphone", "cpu", "iPhone CPU"), "iphone-gpu": ("iphone", "gpu", "iPhone GPU")}


def runs_dir() -> Path:
    return Path(os.environ.get("LOKINFER_PROF_DIR", ROOT / "prof-runs"))


def new_run_dir(kind: str, model: str, device: str, case: str, base: Path | None = None) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "-".join(x for x in (stamp, kind, Path(model).stem, device, case) if x)
    slug = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in slug)
    d = (base or runs_dir()) / slug
    d.mkdir(parents=True, exist_ok=False)
    return d


def mac_state() -> dict:
    """Low Power Mode / AC / thermal (bench/perfdb.machine_state, the dashboard's definition)."""
    try:
        spec = importlib.util.spec_from_file_location("_lk_perfdb", ROOT / "bench" / "perfdb.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.machine_state()
    except Exception as e:  # noqa: BLE001 - state is informative, never fatal
        return {"error": str(e)}


def host_info() -> dict:
    return {"host": socket.gethostname(), "machine": platform.machine(), "macos": platform.mac_ver()[0],
            "python": platform.python_version()}


def timed(fn, min_warm_s: float, warmup: int, runs: int, min_run_s: float) -> list[float]:
    """Dashboard timing protocol (docs/perf-dashboard.md): >= `warmup` calls and >= `min_warm_s` of warm-up (GPU DVFS
    ramp-up), then >= `runs` calls and >= `min_run_s` measured. Returns the measured times in ms."""
    t_end = time.perf_counter() + min_warm_s
    n = 0
    while n < warmup or (time.perf_counter() < t_end and n < 5000):
        fn()
        n += 1
    ts = []
    t_end = time.perf_counter() + min_run_s
    while len(ts) < runs or (time.perf_counter() < t_end and len(ts) < 5000):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def ir_passes(model) -> list[dict]:
    """The IR after each pass of the default pipeline: [{"name": "0-source", "text"}, {"name": "dce", ...}]."""
    from lokinfer import _lokinfer as _C

    src = getattr(model, "source_ir", None)
    final = model.ir_text()
    if not src or src == final:  # a loaded .lkm: only its final IR exists
        return [{"name": "final (.lkm)", "text": final}]
    passes = list(_C.default_pipeline())
    out = [{"name": "source (Python trace)", "text": src}]
    for k in range(1, len(passes) + 1):
        try:
            text = _C.compile_text(src, passes[:k], []).ir_text()
        except Exception as e:  # noqa: BLE001 - a pass prefix may not be valid on its own
            text = f"// pipeline prefix {passes[:k]} failed: {e}\n"
        out.append({"name": passes[k - 1], "text": text})
    return out


def write_ir(run_dir: Path, passes: list[dict]) -> list[dict]:
    d = run_dir / "ir"
    d.mkdir(exist_ok=True)
    for i, p in enumerate(passes):
        fname = f"{i:02d}-{''.join(c if c.isalnum() or c in '-_' else '_' for c in p['name'])}.mlir"
        (d / fname).write_text(p["text"])
        p["file"] = f"ir/{fname}"
    return passes


def _iphone_host(udid: str) -> str:
    return os.environ.get("LOKINFER_IPHONE_HOST", f"{udid}.coredevice.local")


def profile_lokinfer(model_name: str, device: str, case: str | None = None, *, iters: int = 5, runs: int = 10,
                     min_time: float = 2.0, warm_time: float = 1.0, serialize: bool = False, udid: str | None = None,
                     inputs_npz: str | None = None, out_base: Path | None = None, with_ir: bool = True,
                     report: bool = True, log=print) -> Path:
    """Profiles one model on one device; returns the run directory."""
    target_kind, backend, dev_label = DEVICES[device]
    tgt: Target = target_for(model_name, inputs_npz)
    case = case or tgt.default_case
    log(f"[prof-lokinfer] {tgt.title} case={case} on {dev_label}: building the {backend.upper()} graph ...")
    t0 = time.time()
    model = tgt.build(backend)
    build_s = time.time() - t0
    if backend == "gpu" and not model.uses_gpu and not tgt.notes.get("lkm"):
        log("  note: this model has no GPU kernels for this build")
    settings = {"iters": iters, "runs": runs, "min_time_s": min_time, "warm_time_s": warm_time,
                "gpu_serialize": serialize, "build_s": round(build_s, 2)}
    if target_kind == "mac":
        args = tgt.inputs(case)
        call = lambda: model(*args)  # noqa: E731
        log("  timing (dashboard protocol: warm-up >= 1 s, >= 2 s measured) ...")
        e2e = timed(call, warm_time, 3, runs, min_time)
        exe = model._exe
        exe.take_profile()  # drop anything stale
        exe.set_profile(1, serialize)
        for _ in range(iters):
            call()
        exe.set_profile(0)
        raw = json.loads(exe.take_profile())
        state = mac_state()
        dev_info = {"device": dev_label, **host_info()}
    else:
        from lokinfer.remote import RemoteDevice

        udid = udid or localconfig.require("iphone_udid")
        log(f"  connecting to LokiHost on {_iphone_host(udid)} ...")
        dev = RemoteDevice.connect(_iphone_host(udid), retries=8)
        try:
            if not dev.supports("profile"):
                raise SystemExit("the LokiHost on the phone predates profiling: redeploy it (scripts/ios_deploy.sh)")
            state0 = dev.state()
            mid = dev.load(model)
            inputs = tgt.remote_inputs(case)
            _, t = dev.run(mid, inputs, warmup=1, iters=3)
            est = max(statistics.median(t) / 1e6, 0.01)
            warm = int(min(5000, max(3, warm_time * 1e3 / est)))
            n = int(min(5000, max(runs, min_time * 1e3 / est)))
            log(f"  timing on the phone: {warm} warm-up + {n} measured calls ...")
            _, times = dev.run(mid, inputs, warmup=warm, iters=n)
            e2e = [x / 1e6 for x in times]
            dev.set_profile(mid, 1, serialize)
            dev.run(mid, inputs, warmup=0, iters=iters)
            dev.set_profile(mid, 0)
            raw = dev.take_profile(mid)
            state1 = dev.state()
            state = {"start": {k: state0.get(k) for k in ("thermal", "low_power_mode")},
                     "end": {k: state1.get(k) for k in ("thermal", "low_power_mode")},
                     "low_power_mode": state1.get("low_power_mode"), "thermal": state1.get("thermal")}
            dev_info = {"device": dev_label, **{k: v for k, v in state1.items() if k != "features"}}
        finally:
            dev.close()
    run_dir = new_run_dir("lokinfer", tgt.name, device, case, out_base)
    (run_dir / "raw_profile.json").write_text(json.dumps(raw))
    passes = write_ir(run_dir, ir_passes(model)) if with_ir else None
    meta = {"id": run_dir.name, "model": tgt.title, "model_key": tgt.name, "case": case, "device": dev_label,
            "target": target_kind, "backend": backend,
            "created": _dt.datetime.now().isoformat(timespec="seconds"), "machine_state": state, "host": dev_info,
            "settings": settings, "files": {"raw": "raw_profile.json"}}
    run = analysis.build_run(raw, meta=meta, e2e_ms=e2e, ir_passes=passes)
    warn = notes_for(run)
    if warn:
        run["notes"] = warn
    (run_dir / "run.json").write_text(json.dumps(run, indent=1))
    if report:
        from .report import write_report

        write_report(run_dir)
    s = run["summary"]
    log(f"  e2e median {s['e2e_ms']['median']:.3f} ms (p10 {s['e2e_ms']['p10']:.3f}, p90 {s['e2e_ms']['p90']:.3f}, "
        f"n={s['e2e_ms']['n']}); profiled call {s['profiled_ms']['median']:.3f} ms")
    for k in run["kernels"][:12]:
        tf = f"{k['tflops']:6.2f} TF/s" if k.get("tflops") else " " * 11
        log(f"  {k['total_ms']:9.3f} ms {k['pct']:5.1f}%  x{k['calls']:<6g} {k['device']}  {tf}  {k['name']}")
    log(f"  -> {run_dir}")
    return run_dir


def notes_for(run: dict) -> list[str]:
    """Caveats shown on the report (power state, profiler overhead, missing GPU timestamps)."""
    n = []
    st = run.get("machine_state") or {}
    if st.get("low_power_mode"):
        n.append("Low Power Mode was ON: timings are not comparable with the baselines.")
    th = str(st.get("thermal", "nominal"))
    if th not in ("nominal", "unknown", "None") and th:
        n.append(f"Thermal state '{th}': the device may be throttled.")
    if st.get("power") == "battery":
        n.append("On battery power.")
    s = run["summary"]
    if s.get("gpu_mode") == "serialize":
        n.append("GPU kernels were serialized (commit + wait each): no CPU/GPU overlap in the profiled calls; "
                 "per-kernel GPU times are exact but the call total is inflated.")
    if run["backend"] == "gpu" and s.get("gpu_kernels") and not s.get("gpu_timestamps"):
        n.append("No GPU timestamps (counter heap unavailable): GPU rows are CPU wall times.")
    if s.get("profile_overhead_pct", 0) > 10:
        n.append(f"Profiled calls are {s['profile_overhead_pct']:.0f}% slower than unprofiled ones.")
    return n
