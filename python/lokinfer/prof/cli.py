"""Command-line entry points (thin wrappers in tools/: prof-lokinfer, prof-minfer, prof-torch, mlogs-viewer).

    .venv/bin/python -m lokinfer.prof.cli prof-lokinfer <command> ...
    .venv/bin/python -m lokinfer.prof.cli prof-torch MODEL ...
    .venv/bin/python -m lokinfer.prof.cli mlogs-viewer ...

See docs/profiling.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import localconfig

DEVICE_CHOICES = ["mac-cpu", "mac-gpu", "iphone-cpu", "iphone-gpu"]


def _phone_busy() -> bool:
    """The bench sessions drive the phone with bench_remote_models.py: never share the phone with one that is running.
    Only a Python interpreter actually running the script counts (argv[1] is the script), not the shells / lk-lock
    wrappers whose command lines merely mention it while they wait for /tmp/lokinfer-phone.lock."""
    ps = subprocess.run(["ps", "-axo", "args="], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        args = line.split()
        if len(args) >= 2 and "python" in Path(args[0]).name.lower() and args[1].endswith("bench_remote_models.py"):
            return True
    return False


def _check_phone(device: str, force: bool) -> None:
    if device.startswith("iphone") and _phone_busy() and not force:
        raise SystemExit("bench_remote_models.py is using the iPhone right now; retry later (or --force)")


def _open(url: str) -> None:
    subprocess.run(["open", url], check=False)


# ----------------------------------------------------------------------------------------------- prof-lokinfer

def cmd_run(a) -> None:
    from .collect import profile_lokinfer
    from .viewer import DEFAULT_PORT, ensure_server, url_for

    devices = DEVICE_CHOICES if a.device == "all" else a.device.split(",")
    dirs = []
    for dev in devices:
        _check_phone(dev, a.force)
        try:
            dirs.append(profile_lokinfer(a.model, dev, a.case, iters=a.iters, runs=a.runs, min_time=a.min_time,
                                         serialize=a.serialize, udid=a.udid, inputs_npz=a.inputs, with_ir=not a.no_ir))
        except SystemExit as e:
            if len(devices) == 1:
                raise
            print(f"  {dev}: skipped ({e})")
    if a.open and dirs:
        ensure_server(dirs[0].parent, DEFAULT_PORT)
        url = url_for(DEFAULT_PORT, dirs[-1].name)
        print(url)
        _open(url)


def cmd_ls(a) -> None:
    from .collect import runs_dir
    from .viewer import list_runs

    for r in list_runs(runs_dir()):
        e2e = f"{r['e2e_ms']:.3f}" if r["e2e_ms"] is not None else "-"
        print(f"{r['created'] or '':19}  {r['kind']:8} {r['model'] or '':12} {r['case'] or '':14} {r['device'] or '':11} "
              f"{e2e:>10} ms  {r['id']}")


def _load_run(ref: str) -> dict:
    from .collect import runs_dir

    p = Path(ref)
    if not p.exists():
        p = runs_dir() / ref
    if p.is_dir():
        p = p / "run.json"
    return json.loads(p.read_text())


def cmd_compare(a) -> None:
    from .analysis import compare_runs

    ra, rb = _load_run(a.a), _load_run(a.b)
    c = compare_runs(ra, rb)
    print(f"A: {c['a']}\nB: {c['b']}")
    e = c["e2e"]
    print(f"e2e: A {e['a_ms']:.3f} ms, B {e['b_ms']:.3f} ms, B/A {e['ratio']:.3f}x\n")
    print(f"{'category':16} {'A ms':>10} {'B ms':>10} {'B/A':>8}")
    for r in c["categories"]:
        ratio = f"{r['ratio']:.2f}x" if r["ratio"] else "-"
        print(f"{r['category']:16} {r['a_ms']:10.3f} {r['b_ms']:10.3f} {ratio:>8}")
    print(f"\nkernels in both: {c['kernels_matched']}")
    for r in c["kernels"][:a.top]:
        f = lambda x: f"{x:10.3f}" if x is not None else f"{'-':>10}"  # noqa: E731
        ratio = f"{r['ratio']:.2f}x" if r["ratio"] else "-"
        print(f"{f(r['a_ms'])} {f(r['b_ms'])} {ratio:>8}  {r['name']}")
    if a.open:
        from .viewer import DEFAULT_PORT, ensure_server, url_for
        from .collect import runs_dir

        ensure_server(runs_dir(), DEFAULT_PORT)
        url = url_for(DEFAULT_PORT, compare=(ra["id"], rb["id"]))
        print(url)
        _open(url)


def cmd_report(a) -> None:
    from .collect import runs_dir
    from .report import write_report

    d = Path(a.run) if Path(a.run).is_dir() else runs_dir() / a.run
    print(write_report(d))


def cmd_workload(a) -> None:
    """Internal: runs the model in an open-ended loop (until killed) for xctrace / sample to observe."""
    from .targets import target_for

    tgt = target_for(a.model, a.inputs)
    m = tgt.build("gpu" if a.device == "mac-gpu" else "cpu")
    args = tgt.inputs(a.case or tgt.default_case)
    t_end = time.time() + 1.0
    while time.time() < t_end:
        m(*args)
    print("ready", flush=True)
    while True:
        m(*args)


def _trace_dir(model: str, device: str, what: str) -> Path:
    from .collect import new_run_dir, runs_dir

    return new_run_dir(what, model, device, "", runs_dir() / "traces")


def _start_workload(a) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "lokinfer.prof.cli", "prof-lokinfer", "_workload", a.model, "--device", a.device]
    if a.case:
        cmd += ["--case", a.case]
    if getattr(a, "inputs", None):
        cmd += ["--inputs", a.inputs]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.strip() == "ready":
            return p
    raise SystemExit("workload exited before warming up")


def cmd_trace(a) -> None:
    from . import metal

    _check_phone(a.device, a.force)
    out = _trace_dir(a.model, a.device, "xctrace-" + a.template)
    trace = out / f"{a.template}.trace"
    if a.device.startswith("mac"):
        print(f"[trace] warming up {a.model} on {a.device} ...")
        p = _start_workload(a)
        try:
            print(f"[trace] xctrace {metal.template_label(a.template)} for {a.time_limit} (pid {p.pid}) ...")
            r = metal.xctrace_record(a.template, trace, attach=str(p.pid), time_limit=a.time_limit)
        finally:
            p.kill()
            p.wait()
        process = "python"
    else:
        from lokinfer.remote import RemoteDevice

        from .collect import _iphone_host
        from .targets import target_for

        udid = a.udid or localconfig.require("iphone_udid")
        tgt = target_for(a.model, a.inputs)
        model = tgt.build("gpu" if a.device == "iphone-gpu" else "cpu")
        dev = RemoteDevice.connect(_iphone_host(udid), retries=8)
        mid = dev.load(model)
        inputs = tgt.remote_inputs(a.case or tgt.default_case)
        dev.run(mid, inputs, warmup=3, iters=1)
        stop = threading.Event()

        def drive():  # keeps the phone busy while xctrace records (ends with the recording)
            while not stop.is_set():
                dev.run(mid, inputs, warmup=0, iters=1)

        th = threading.Thread(target=drive, daemon=True)
        th.start()
        try:
            print(f"[trace] xctrace {metal.template_label(a.template)} on the iPhone for {a.time_limit} ...")
            # xctrace can't find device processes by name: attach by the pid devicectl reports
            pid = metal.device_pid(udid, "LokiHost.app/LokiHost") or "LokiHost"
            r = metal.xctrace_record(a.template, trace, attach=str(pid), device=udid, time_limit=a.time_limit)
            if r.returncode != 0 and not trace.exists() and "Cannot find process" in (r.stdout + r.stderr):
                # Some templates (Time Profiler on iOS 27) reject the devicectl pid (xctrace exit 21): record every
                # process on the device instead; the summary keeps LokiHost's rows by name.
                print("[trace] xctrace can't attach to that pid on the device; recording all processes instead")
                r = metal.xctrace_record(a.template, trace, device=udid, time_limit=a.time_limit)
        finally:
            stop.set()
            th.join(timeout=60)
            dev.close()
        process = "LokiHost"
    (out / "xctrace.log").write_text(r.stdout + "\n" + r.stderr)
    if not trace.exists():
        raise SystemExit(f"xctrace produced no trace (exit {r.returncode}): {r.stderr.strip()[-800:]}")
    s = metal.summarize_trace(trace, out, process)
    print(metal.format_trace_summary(s))
    print(f"-> {out}\n   (open {trace.name} in Instruments yourself if you want the GUI; this tool never launches it)")


def cmd_sample(a) -> None:
    from . import metal

    if not a.device.startswith("mac"):
        raise SystemExit("sample(1) is Mac-only; on the iPhone use `prof-lokinfer trace MODEL --device iphone-cpu "
                         "--template time`")
    out = _trace_dir(a.model, a.device, "sample")
    p = _start_workload(a)
    try:
        text = metal.sample_pid(p.pid, a.seconds, out / "sample.txt")
    finally:
        p.kill()
        p.wait()
    print("top of stack (samples):")
    for sym, n in metal.sample_top_of_stack(text):
        print(f"  {n:7d}  {sym}")
    print(f"-> {out / 'sample.txt'}")


def cmd_metal(a) -> None:
    from . import metal

    lib = Path(a.metallib) if a.metallib else None
    if a.what == "list":
        for fn in metal.list_functions(lib or metal.default_metallib()):
            print(fn)
    elif a.what in ("air", "air-stats"):
        mods = metal.air_modules(lib or metal.default_metallib())
        names = a.functions or sorted(mods)
        if a.what == "air":
            for fn in names:
                if fn not in mods:
                    raise SystemExit(f"{fn}: not in the metallib (prof-lokinfer metal list)")
                print(f"// ---- {fn}\n{mods[fn]}")
        else:
            rows = [{"function": fn, **metal.air_stats(mods[fn])} for fn in names if fn in mods]
            if a.json:
                print(json.dumps(rows, indent=1))
                return
            keys = ["instructions", "blocks", "loops", "load_device", "store_device", "load_threadgroup",
                    "store_threadgroup", "fma", "fmul", "fadd", "simdgroup_matrix", "simd_shuffle", "barrier", "math_call"]
            print(f"{'function':34} " + " ".join(f"{k[:10]:>10}" for k in keys))
            for r in rows:
                print(f"{r['function'][:34]:34} " + " ".join(f"{r.get(k, 0):>10}" for k in keys))
    elif a.what == "source":
        for fn in a.functions:
            loc = metal.find_source(fn)
            print(f"{fn}: {loc[0]}:{loc[1]}" if loc else f"{fn}: not found in kernels/gpu/*.metal")
            if loc and a.emit_air:
                out = Path(a.out or metal.tempdir()) / (loc[0].stem + ".ll")
                metal.compile_air_text(loc[0], out, "iphoneos" if a.iphone else "macosx")
                print(f"  textual AIR of {loc[0].name}: {out}")
    elif a.what in ("stats", "info", "native"):
        dev = None
        if a.iphone:
            from lokinfer.remote import RemoteDevice

            from .collect import _iphone_host

            if _phone_busy() and not a.force:
                raise SystemExit("bench_remote_models.py is using the iPhone right now; retry later (or --force)")
            dev = RemoteDevice.connect(_iphone_host(a.udid or localconfig.require("iphone_udid")), retries=8)
            if not dev.supports("gpu_inspect"):
                raise SystemExit("the LokiHost on the phone predates GPU_INSPECT: redeploy it (scripts/ios_deploy.sh)")
        inspect = (lambda fns, arc=None: metal.inspect_remote(dev, fns, arc)) if dev else \
            (lambda fns, arc=None: metal.inspect_local(fns, arc))
        try:
            if a.what == "info":
                print(json.dumps(inspect([])["device"], indent=1))
            elif a.what == "stats":
                info = inspect(a.functions or [])
                if a.json:
                    print(json.dumps(info, indent=1))
                    return
                d = info["device"]
                print(f"{d['name']} ({d.get('architecture', '?')}): max threadgroup memory {d['max_threadgroup_memory']} B, "
                      f"working set {d['recommended_max_working_set'] >> 20} MiB")
                print(f"{'function':40} {'maxThr/TG':>9} {'SIMD':>5} {'TG mem':>8} {'PSO ms':>8}")
                for f in info["functions"]:
                    if "error" in f:
                        print(f"{f['name'][:40]:40} error: {f['error']}")
                        continue
                    print(f"{f['name'][:40]:40} {f['max_total_threads_per_threadgroup']:>9} {f['thread_execution_width']:>5} "
                          f"{f['static_threadgroup_memory']:>8} {f['pipeline_create_ms']:>8.2f}")
            else:
                work = Path(a.out) if a.out else metal.tempdir()
                work.mkdir(parents=True, exist_ok=True)
                fns = a.functions or metal.list_functions(lib or metal.default_metallib())
                rows = metal.native_sizes(fns, inspect, work)
                if a.json:
                    print(json.dumps(rows, indent=1))
                    return
                print(f"{'function':40} {'arch':16} {'__compute B':>11} {'maxThr/TG':>9} {'TG mem':>8}")
                for r in rows:
                    print(f"{r['function'][:40]:40} {r.get('arch', '-'):16} {r.get('native_compute_bytes') or 0:>11} "
                          f"{r.get('max_total_threads_per_threadgroup', '-'):>9} {r.get('static_threadgroup_memory', '-'):>8}")
                print(f"native binaries (no public disassembler): {work}")
        finally:
            if dev:
                dev.close()


def cmd_devices(a) -> None:
    from .collect import _iphone_host

    try:
        from lokinfer import _lokinfer as _C

        d = json.loads(_C.gpu_inspect(["add_f32"], ""))["device"]
        print(f"mac-gpu: {d['name']} ({d.get('architecture')}) ok")
    except Exception as e:  # noqa: BLE001
        print(f"mac-gpu: unavailable ({e})")
    print("mac-cpu: ok")
    if _phone_busy():
        print("iphone: busy (bench_remote_models.py is running)")
        return
    from lokinfer.remote import RemoteDevice, RemoteError

    host = _iphone_host(a.udid or localconfig.require("iphone_udid"))
    try:
        with RemoteDevice.connect(host, retries=2, timeout=5) as dev:
            print(f"iphone: {host} {json.dumps(dev.info)}")
            print("  profiling:", "yes" if dev.supports("profile") else "NO (redeploy LokiHost: scripts/ios_deploy.sh)")
    except (RemoteError, OSError) as e:
        print(f"iphone: unreachable at {host} ({e}); start it with scripts/ios_deploy.sh")


def prof_lokinfer(argv: list[str], prog: str = "prof-lokinfer") -> None:
    if argv[:1] == ["torch"]:  # `prof-lokinfer torch ...` == `prof-torch ...`
        return prof_torch(argv[1:])
    ap = argparse.ArgumentParser(prog=prog, description="Profile lokinfer models on Mac CPU/GPU and iPhone CPU/GPU "
                                 "(docs/profiling.md).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p, devices=DEVICE_CHOICES):
        p.add_argument("model", help="kokoro, mnist, resnet18, mobilenet_v2, dcgan_g, dcgan_d, gpt2, or a .lkm path")
        p.add_argument("--device", "-d", default="mac-cpu", help=f"{', '.join(devices)} (run: also 'all' or a,b)")
        p.add_argument("--case", "-c", help="input case (default: the model's dashboard case)")
        p.add_argument("--inputs", help=".npz with the arguments of a .lkm model")
        p.add_argument("--udid", help="iPhone UDID (default $LOKINFER_IPHONE or the test phone)")
        p.add_argument("--force", action="store_true", help="use the phone even if a bench is running")

    p = sub.add_parser("run", help="profile a model: per-kernel table, GPU timestamps, IR per pass, memory, report")
    model_args(p)
    p.add_argument("--iters", type=int, default=5, help="profiled calls (after the timed ones)")
    p.add_argument("--runs", type=int, default=10, help="min timed calls for the e2e number")
    p.add_argument("--min-time", type=float, default=2.0, help="min seconds of timed calls")
    p.add_argument("--serialize", action="store_true", help="commit + wait after each GPU kernel (no overlap)")
    p.add_argument("--no-ir", action="store_true", help="skip the IR-per-pass dump")
    p.add_argument("--open", action="store_true", help="open the report in mlogs-viewer")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ls", help="list profile runs")
    p.set_defaults(fn=cmd_ls)
    p = sub.add_parser("compare", help="compare two runs (ids or paths): e2e, categories, kernels")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--open", action="store_true", help="open the side-by-side view in mlogs-viewer")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("report", help="(re)generate a run's report.html")
    p.add_argument("run")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("trace", help="record an Instruments trace with xctrace (Mac process or LokiHost on the iPhone)")
    model_args(p)
    p.add_argument("--template", "-t", default="metal",
                   help="metal (Metal System Trace), gpu-counters, time (Time Profiler), cpu-counters, system, or a "
                        "template name")
    p.add_argument("--time-limit", default="10s", help="recording length (xctrace syntax: 10s, 1m)")
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser("sample", help="CPU call-stack sampling of a Mac run with sample(1)")
    model_args(p, ["mac-cpu", "mac-gpu"])
    p.add_argument("--seconds", type=int, default=5)
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("metal", help="shader / pipeline tools: list, air, air-stats, source, stats, native, info")
    p.add_argument("what", choices=["list", "air", "air-stats", "source", "stats", "native", "info"])
    p.add_argument("functions", nargs="*")
    p.add_argument("--metallib", help="metallib to read (default: the Python package's)")
    p.add_argument("--iphone", action="store_true", help="stats / native / info from the iPhone (remote executor)")
    p.add_argument("--udid")
    p.add_argument("--json", action="store_true")
    p.add_argument("--emit-air", action="store_true", help="source: also recompile the file to textual AIR (metal -S)")
    p.add_argument("--out", help="output directory (native binaries, AIR)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_metal)

    sub.add_parser("torch", help="same as prof-torch (PyTorch reference profile): prof-lokinfer torch MODEL ...")
    p = sub.add_parser("devices", help="check which devices can be profiled")
    p.add_argument("--udid")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("_workload", help=argparse.SUPPRESS)
    p.add_argument("model")
    p.add_argument("--device", default="mac-cpu")
    p.add_argument("--case")
    p.add_argument("--inputs")
    p.set_defaults(fn=cmd_workload)

    a = ap.parse_args(argv)
    a.fn(a)


# -------------------------------------------------------------------------------------------------- prof-torch

def prof_torch(argv: list[str]) -> None:
    from .targets import REGISTRY

    ap = argparse.ArgumentParser(prog="prof-torch", description="Profile the PyTorch reference of a lokinfer model "
                                 "(same inputs) into the prof-lokinfer run format.")
    ap.add_argument("model", help=", ".join(REGISTRY))
    ap.add_argument("--device", "-d", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--case", "-c")
    ap.add_argument("--iters", type=int, default=3, help="per-op profiled calls")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--min-time", type=float, default=2.0)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--compare", help="after profiling, print a comparison with this lokinfer run id")
    a = ap.parse_args(argv)
    from .torch_prof import profile_torch

    d = profile_torch(a.model, a.device, a.case, iters=a.iters, runs=a.runs, min_time=a.min_time)
    if a.compare:
        cmd_compare(argparse.Namespace(a=a.compare, b=str(d), top=30, open=a.open))
    elif a.open:
        from .viewer import DEFAULT_PORT, ensure_server, url_for

        ensure_server(d.parent, DEFAULT_PORT)
        url = url_for(DEFAULT_PORT, d.name)
        print(url)
        _open(url)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python -m lokinfer.prof.cli {prof-lokinfer|prof-minfer|prof-torch|mlogs-viewer} ...")
    prog, rest = argv[0], argv[1:]
    if prog in ("prof-lokinfer", "prof-minfer"):
        prof_lokinfer(rest, prog)
    elif prog == "prof-torch":
        prof_torch(rest)
    elif prog == "mlogs-viewer":
        from .viewer import main as viewer_main

        viewer_main(rest)
    else:
        raise SystemExit(f"unknown tool {prog}")


if __name__ == "__main__":
    main()
