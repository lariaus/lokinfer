"""kbench: the kernel benchmark harness (docs/benchmarking.md, "Kernel harness").

  kbench extract                         # real-shape cases from every model build -> bench/harness/cases.json
  kbench run -d mac-cpu,mac-gpu          # every case on the Mac; -d iphone-cpu,iphone-gpu via LokiHost's remote executor
  kbench run -d iphone-gpu --part 1/4    # a slice of the cases (short phone sessions); --append RUN adds to a run
  kbench check [-d ...]                  # compile + run every case once (no timing): finds broken cases
  kbench ls                              # runs in perf/kernels.sqlite
  kbench show RUN [--top N]              # one run's results (slowest first)
  kbench compare A B [--threshold 0.1]   # B vs A per case, regressions flagged (exit 1 with --fail)
  kbench coverage                        # registered kernels without a case / without a result per device
  kbench gbench-export DIR               # CPU cases as IR files for build-release/lokinfer_bench_kernels
  kbench gbench-import F.json... --dir D # Google Benchmark JSON -> a "gbench" run

Every case is one registered kernel at one shape a real model build executes, run as a stand-alone graph: inputs =
the operands (seeded random data), the kernel repeated R times on the same output buffers (R so one call lasts
~2 ms; on the GPU the repeats are dependent dispatches in one command buffer). Timing protocol (docs/perf-dashboard.md):
warm-up >= 3 calls and >= 1 s, then >= 30 calls and >= 2 s, median. CPU kernels: wall time per call / R. GPU kernels:
then 10+ profiled calls whose per-kernel Metal counter-heap timestamps give the GPU time of the R dispatches / R.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "python"))

import cases as C  # noqa: E402
from lokinfer import localconfig  # noqa: E402
import kdb  # noqa: E402

DEVICES = {"mac-cpu": ("mac", "cpu", "Mac CPU"), "mac-gpu": ("mac", "gpu", "Mac GPU"),
           "iphone-cpu": ("iphone", "cpu", "iPhone CPU"), "iphone-gpu": ("iphone", "gpu", "iPhone GPU")}

WARM_S, MIN_RUNS, MIN_S, MAX_RUNS = 1.0, 30, 2.0, 5000


# ----------------------------------------------------------------------------------------------- environment

def source_rev() -> str | None:
    """The commit the working tree is on, read from .git's files (no git command)."""
    try:
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            p = ROOT / ".git" / ref
            if p.exists():
                return p.read_text().strip()[:12]
            packed = (ROOT / ".git" / "packed-refs").read_text()
            m = re.search(rf"^([0-9a-f]{{40}}) {re.escape(ref)}$", packed, re.M)
            return m.group(1)[:12] if m else None
        return head[:12]
    except OSError:
        return None


def mac_build_id() -> str:
    import lokinfer._lokinfer as _C

    h = hashlib.sha256()
    so = Path(_C.__file__)
    h.update(so.read_bytes())
    lib = so.with_name("lokinfer.metallib")
    if lib.exists():
        h.update(lib.read_bytes())
    return h.hexdigest()[:12]


def mac_state() -> dict:
    sys.path.insert(0, str(ROOT / "bench"))
    import perfdb

    return perfdb.machine_state()


def mac_hardware() -> str:
    import subprocess

    return subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip()


def pct(xs: list[float]) -> dict:
    s = sorted(xs)
    return {"median": statistics.median(s), "p10": s[len(s) // 10], "p90": s[min(len(s) - 1, 9 * len(s) // 10)]}


# ------------------------------------------------------------------------------------------------- one case

def _gpu_span_us(raw: dict) -> list[float]:
    """Per profiled call: GPU time from the first kernel's begin to the last kernel's end timestamp (us)."""
    out = []
    for call in raw["calls"]:
        g = [(k[3], k[4]) for k in call["kernels"] if k[3] >= 0 and k[4] >= k[3]]
        if g:
            out.append((max(b for _, b in g) - min(a for a, _ in g)) / 1e3)
    return out


def _counts(est_ms: float) -> tuple[int, int]:
    est = max(est_ms, 0.005)
    warm = int(min(MAX_RUNS, max(3, math.ceil(WARM_S * 1e3 / est))))
    n = int(min(MAX_RUNS, max(MIN_RUNS, math.ceil(MIN_S * 1e3 / est))))
    return warm, n


def bench_local(case: dict, repeats: int) -> dict:
    import lokinfer as lk

    model = lk.compile_ir(C.case_ir(case, repeats), pipeline=[], single=False)
    args = [lk.from_numpy(a) if a.ndim else a.item() for a in C.case_inputs(case)]
    call = lambda: model(*args)  # noqa: E731
    t0 = time.perf_counter()
    call()
    est = (time.perf_counter() - t0) * 1e3
    warm, n = _counts(est)
    # warm-up: >= 3 calls and >= WARM_S
    t_end = time.perf_counter() + WARM_S
    k = 0
    while k < 3 or (time.perf_counter() < t_end and k < MAX_RUNS):
        call()
        k += 1
    ts = []
    t_end = time.perf_counter() + MIN_S
    while len(ts) < MIN_RUNS or (time.perf_counter() < t_end and len(ts) < MAX_RUNS):
        t0 = time.perf_counter()
        call()
        ts.append((time.perf_counter() - t0) * 1e6)
    res = {"wall": ts}
    if case["device"] == "gpu":
        exe = model._exe
        exe.take_profile()
        exe.set_profile(1, False)
        for _ in range(max(10, min(200, int(500 / max(est, 0.01))))):
            call()
        exe.set_profile(0)
        res["gpu"] = _gpu_span_us(json.loads(exe.take_profile()))
    return res


def bench_remote(dev, case: dict, repeats: int) -> dict:
    import lokinfer as lk

    model = lk.compile_ir(C.case_ir(case, repeats), pipeline=[], single=False)
    mid = dev.load(model)
    inputs = C.case_inputs(case)
    _, t = dev.run(mid, inputs, warmup=1, iters=2)
    est = statistics.median(t) / 1e6
    warm, n = _counts(est)
    _, times = dev.run(mid, inputs, warmup=warm, iters=n)
    res = {"wall": [x / 1e3 for x in times]}
    if case["device"] == "gpu":
        dev.set_profile(mid, 1, False)
        dev.run(mid, inputs, warmup=0, iters=max(10, min(200, int(500 / max(est, 0.01)))))
        dev.set_profile(mid, 0)
        res["gpu"] = _gpu_span_us(dev.take_profile(mid))
    return res


def summarize(case: dict, repeats: int, res: dict) -> dict:
    from lokinfer.prof import analysis

    wall = pct([x / repeats for x in res["wall"]])
    main = pct([x / repeats for x in res["gpu"]]) if res.get("gpu") else wall
    sig = case["sig"].split(" {")[0]
    fl = analysis.flops_of(case["kernel"], sig)
    out = {"repeats": repeats, "n": len(res["wall"]), "median_us": main["median"], "p10_us": main["p10"],
           "p90_us": main["p90"], "wall_us": wall["median"]}
    if fl:
        out["gflop"] = fl / 1e9
        out["tflops"] = fl / (main["median"] * 1e-6) / 1e12
    return out


# ------------------------------------------------------------------------------------------------- commands

def select_cases(args, dev_kind: str) -> list[dict]:
    cs = [c for c in C.load() if c["device"] == dev_kind]
    if args.kernel:
        rx = re.compile(args.kernel)
        cs = [c for c in cs if rx.search(c["kernel"])]
    if args.model:
        cs = [c for c in cs if args.model in c["instances"]]
    if args.part:
        i, n = (int(x) for x in args.part.split("/"))
        cs = cs[(i - 1) * len(cs) // n: i * len(cs) // n]
    if args.limit:
        cs = cs[: args.limit]
    return cs


def cmd_extract(args):
    cs = C.extract()
    C.save(cs)
    print(f"{len(cs)} cases -> {C.CASES_JSON}")


def cmd_check(args):
    """Compiles and runs every selected case once (no timing, nothing recorded): finds broken cases fast."""
    import lokinfer as lk

    bad = 0
    for d in args.device.split(","):
        _, kind, label = DEVICES[d]
        for c in select_cases(args, kind):
            try:
                model = lk.compile_ir(C.case_ir(c, 2), pipeline=[], single=False)
                model(*[lk.from_numpy(a) if a.ndim else a.item() for a in C.case_inputs(c)])
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"{label}: {c['kernel']} {c['sig'][:100]}: {str(e).splitlines()[0][:200]}")
    print(f"{bad} failing cases")


def cmd_run(args):
    con = kdb.connect()
    for d in args.device.split(","):
        target, kind, label = DEVICES[d]
        cs = select_cases(args, kind)
        if not cs:
            print(f"{label}: no cases")
            continue
        dev = None
        if target == "iphone":
            from lokinfer.remote import RemoteDevice

            udid = args.udid or localconfig.require("iphone_udid")
            dev = RemoteDevice.connect(os.environ.get("LOKINFER_IPHONE_HOST", f"{udid}.coredevice.local"),
                                       timeout=600.0, retries=8)
            st = dev.state()
            state = {"low_power_mode": st.get("low_power_mode"), "power": None, "thermal": st.get("thermal")}
            hw, build = st.get("device"), st.get("build")
        else:
            state = mac_state()
            hw, build = mac_hardware(), mac_build_id()
        settings = {"warm_s": WARM_S, "min_runs": MIN_RUNS, "min_s": MIN_S, "target_call_us": 2000,
                    "cases_file": "bench/harness/cases.json", "filter": {"kernel": args.kernel, "model": args.model,
                                                                        "part": args.part, "limit": args.limit}}
        if args.append:
            run_id = kdb.resolve_run(con, args.append)
        else:
            run_id = kdb.new_run(con, device=label, hardware=hw, build_id=build, source_rev=source_rev(),
                                 harness="runtime", low_power_mode=state.get("low_power_mode"),
                                 power=state.get("power"), thermal_start=state.get("thermal"), label=args.label,
                                 settings=settings)
        print(f"run {run_id}: {label} ({hw}, build {build}), {len(cs)} cases, state {state}", flush=True)
        t_start = time.time()
        thermal = state.get("thermal")
        try:
            for i, c in enumerate(cs):
                r = C.repeats_for(c)
                try:
                    res = bench_remote(dev, c, r) if dev else bench_local(c, r)
                    vals = summarize(c, r, res)
                    thermal = dev.state().get("thermal") if dev else mac_state().get("thermal")
                    kdb.add_result(con, run_id, c, thermal=thermal, **vals)
                    tf = f" {vals['tflops']:.2f} TF/s" if vals.get("tflops") else ""
                    print(f"  [{i + 1}/{len(cs)}] {vals['median_us']:10.2f} us (x{r}, n={vals['n']}){tf}  "
                          f"{c['kernel']} {c['sig'][:110]}", flush=True)
                except Exception as e:  # noqa: BLE001 - record the failure, keep going
                    msg = f"{type(e).__name__}: {e}".splitlines()[0][:300]
                    kdb.add_result(con, run_id, c, repeats=r, error=msg)
                    print(f"  [{i + 1}/{len(cs)}] ERROR {c['kernel']}: {msg}", flush=True)
                    if dev is not None and isinstance(e, OSError):
                        raise
        finally:
            kdb.finish_run(con, run_id, thermal)
            if dev:
                dev.close()
        print(f"run {run_id} done in {time.time() - t_start:.0f} s (thermal at end: {thermal})")


def _fmt_run(r) -> str:
    lpm = {None: "?", 0: "off", 1: "ON"}[r["low_power_mode"]]
    return (f"{r['id']:4d}  {r['created']}  {r['device']:<10}  {r['harness']:<7}  build {r['build_id'] or '?':<12}  "
            f"rev {r['source_rev'] or '?':<12}  LPM {lpm:<3}  {r['power'] or '-':<7}  "
            f"thermal {r['thermal_start']}->{r['thermal_end']}  {r['ok']} ok / {r['failed']} failed  {r['label'] or ''}")


def cmd_ls(args):
    con = kdb.connect()
    for r in kdb.runs(con, args.device):
        print(_fmt_run(r))


def cmd_show(args):
    con = kdb.connect()
    rid = kdb.resolve_run(con, args.run)
    rows = sorted(kdb.results(con, rid).values(), key=lambda r: -(r["median_us"] or 0))
    total = sum((r["median_us"] or 0) * sum(json.loads(r["models"]).values()) for r in rows)
    print(f"run {rid}: {len(rows)} cases; sum over cases of time x instances = {total / 1e3:.2f} ms")
    for r in rows[: args.top]:
        if r["error"]:
            print(f"  ERROR {r['kernel']} {r['shape'][:80]}: {r['error']}")
            continue
        tf = f"{r['tflops']:5.2f} TF/s" if r["tflops"] else " " * 10
        print(f"  {r['median_us']:10.2f} us  [{r['p10_us']:.2f}-{r['p90_us']:.2f}]  {tf}  x{r['repeats']:<3} "
              f"{r['kernel']}  {r['shape'][:100]}")


def compare(con, a: int, b: int, threshold: float):
    ra, rb = kdb.results(con, a), kdb.results(con, b)
    rows, regress, improve = [], [], []
    for key in sorted(set(ra) & set(rb)):
        x, y = ra[key], rb[key]
        if x["error"] or y["error"] or not x["median_us"] or not y["median_us"]:
            continue
        ratio = y["median_us"] / x["median_us"]
        # A change counts only beyond the threshold AND outside both runs' p10-p90 spread (noise).
        slower = ratio > 1 + threshold and y["p10_us"] > x["p90_us"]
        faster = ratio < 1 / (1 + threshold) and y["p90_us"] < x["p10_us"]
        row = (ratio, key, x, y)
        rows.append(row)
        if slower:
            regress.append(row)
        elif faster:
            improve.append(row)
    return rows, regress, improve


def cmd_compare(args):
    con = kdb.connect()
    a, b = kdb.resolve_run(con, args.a), kdb.resolve_run(con, args.b)
    run_rows = {r["id"]: r for r in kdb.runs(con)}
    print("A " + _fmt_run(run_rows[a]))
    print("B " + _fmt_run(run_rows[b]))
    rows, regress, improve = compare(con, a, b, args.threshold)
    if not rows:
        print("no common cases")
        return 0
    geo = math.exp(sum(math.log(r[0]) for r in rows) / len(rows))
    print(f"{len(rows)} common cases; geometric mean B/A = {geo:.3f} ({'B faster' if geo < 1 else 'B slower'}); "
          f"{len(regress)} regressions, {len(improve)} improvements (> {args.threshold:.0%} and outside p10-p90)")

    def show(title, rs):
        if rs:
            print(title)
        for ratio, key, x, y in rs[: args.top]:
            print(f"  {ratio:6.3f}x  {x['median_us']:10.2f} -> {y['median_us']:10.2f} us  {key[:140]}")

    show("REGRESSIONS (B slower):", sorted(regress, key=lambda r: -r[0]))
    show("improvements (B faster):", sorted(improve, key=lambda r: r[0]))
    if args.all:
        show("all:", sorted(rows, key=lambda r: -r[0]))
    return 1 if (args.fail and regress) else 0


def cmd_coverage(args):
    import lokinfer._lokinfer as _C

    con = kdb.connect()
    reg = set(_C.registered_kernels())
    cs = C.load()
    have = {c["kernel"] for c in cs}
    print(f"{len(reg)} registered kernels, {len(reg & have)} with a real-shape case, {len(cs)} cases")
    print("no model build runs these (no real shape):", " ".join(sorted(reg - have)) or "-")
    for label in ("Mac CPU", "Mac GPU", "iPhone CPU", "iPhone GPU"):
        kind = "gpu" if label.endswith("GPU") else "cpu"
        want = {c["key"] for c in cs if c["device"] == kind}
        got = {r[0] for r in con.execute(
            "SELECT DISTINCT x.case_key FROM results x JOIN runs r ON r.id = x.run_id "
            "WHERE r.device = ? AND x.error IS NULL AND r.harness = 'runtime'", (label,))}
        missing = sorted(want - got)
        print(f"{label}: {len(want & got)}/{len(want)} cases measured" + (f"; missing e.g. {missing[:3]}" if missing else ""))


def cmd_gbench_export(args):
    """CPU cases for build-release/lokinfer_bench_kernels: caseNNNN.mlir (one kernel, R = 1) + caseNNNN.argI.bin for
    operands whose values matter (integers, scalars, small vectors such as Winograd tables); the C++ side fills the
    other float operands with uniform [-1, 1]."""
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    cpu = [x for x in C.load() if x["device"] == "cpu"]
    if args.one_per_kernel:  # targeted: each kernel's most frequent real shape only
        best: dict[str, dict] = {}
        for x in cpu:
            if x["kernel"] not in best or sum(x["instances"].values()) > sum(best[x["kernel"]]["instances"].values()):
                best[x["kernel"]] = x
        cpu = [x for x in cpu if best[x["kernel"]] is x]
    for i, c in enumerate(cpu):
        name = f"case{i:04d}.mlir"
        (out / name).write_text(C.case_ir(c, 1))
        for j, a in enumerate(C.case_inputs(c)):
            if a.ndim == 0 or a.dtype.kind in "iub" or a.size <= 4096:
                (out / f"case{i:04d}.arg{j}.bin").write_bytes(a.tobytes())
        index.append({"file": name, "key": c["key"]})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} CPU cases -> {out}")


def cmd_gbench_import(args):
    con = kdb.connect()
    datas = [json.loads(Path(f).read_text()) for f in args.file]
    context = dict(datas[0].get("context", {}))
    context.pop("host_name", None)  # the machine's name is not recorded (the device model is, in the run row)
    data = {"context": context, "benchmarks": [b for d in datas for b in d["benchmarks"]]}
    index = {e["file"]: e["key"] for e in json.loads((Path(args.dir) / "index.json").read_text())}
    by_key = {c["key"]: c for c in C.load()}
    state = mac_state()
    build_id = args.build_id
    exe = ROOT / "build-release" / "lokinfer_bench_kernels"
    if not build_id and exe.exists():
        build_id = "gb-" + hashlib.sha256(exe.read_bytes()).hexdigest()[:9]
    rid = kdb.new_run(con, device="Mac CPU", hardware=mac_hardware(), build_id=build_id, source_rev=source_rev(),
                      harness="gbench", low_power_mode=state.get("low_power_mode"), power=state.get("power"),
                      thermal_start=state.get("thermal"), label=args.label,
                      settings={"context": data.get("context", {})})
    from lokinfer.prof import analysis

    per: dict[str, dict] = {}
    for b in data["benchmarks"]:
        f = b["name"].split("/")[1] if "/" in b["name"] else b["name"]
        key = index.get(f)
        if key is None:
            continue
        agg = b.get("aggregate_name")
        d = per.setdefault(key, {})
        if b.get("error_occurred"):
            d["error"] = b.get("error_message", "error")
        elif agg in ("median", "mean", "stddev") or agg is None:
            unit = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[b.get("time_unit", "ns")]
            d[agg or "single"] = b["real_time"] * unit
            d["n"] = b.get("iterations")
    for key, d in per.items():
        c = by_key[key]
        if "error" in d:
            kdb.add_result(con, rid, c, error=d["error"])
            continue
        med = d.get("median", d.get("single"))
        fl = analysis.flops_of(c["kernel"], c["sig"].split(" {")[0])
        sd = d.get("stddev", 0.0)
        kdb.add_result(con, rid, c, repeats=1, n=d.get("n"), median_us=med, p10_us=med - sd, p90_us=med + sd,
                       wall_us=med, gflop=fl / 1e9 if fl else None,
                       tflops=fl / (med * 1e-6) / 1e12 if fl else None, thermal=mac_state().get("thermal"))
    kdb.finish_run(con, rid, mac_state().get("thermal"))
    print(f"gbench run {rid}: {len(per)} cases")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    r = sub.add_parser("run")
    r.add_argument("-d", "--device", required=True, help="mac-cpu,mac-gpu,iphone-cpu,iphone-gpu")
    r.add_argument("-k", "--kernel", help="regex on the kernel name")
    r.add_argument("-m", "--model", help="only cases of this model build (e.g. kokoro, mnist)")
    r.add_argument("--part", help="i/n: the i-th of n slices of the selected cases")
    r.add_argument("--limit", type=int)
    r.add_argument("--append", help="add to an existing run (id or latest:<device>) instead of a new one")
    r.add_argument("--label")
    r.add_argument("--udid")
    ck = sub.add_parser("check")
    ck.add_argument("-d", "--device", default="mac-cpu,mac-gpu")
    ck.add_argument("-k", "--kernel")
    ck.add_argument("-m", "--model")
    ck.add_argument("--part")
    ck.add_argument("--limit", type=int)
    sub.add_parser("ls").add_argument("--device")
    s = sub.add_parser("show")
    s.add_argument("run")
    s.add_argument("--top", type=int, default=40)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--threshold", type=float, default=0.10)
    c.add_argument("--top", type=int, default=30)
    c.add_argument("--all", action="store_true")
    c.add_argument("--fail", action="store_true", help="exit 1 when there is a regression")
    sub.add_parser("coverage")
    e = sub.add_parser("gbench-export")
    e.add_argument("dir")
    e.add_argument("--one-per-kernel", action="store_true", help="only each kernel's most frequent shape (targeted)")
    gi = sub.add_parser("gbench-import")
    gi.add_argument("file", nargs="+", help="Google Benchmark JSON output(s) (e.g. one per --part)")
    gi.add_argument("--dir", required=True, help="the gbench-export directory (index.json)")
    gi.add_argument("--build-id")
    gi.add_argument("--label")
    a = p.parse_args(argv)
    fn = {"extract": cmd_extract, "run": cmd_run, "ls": cmd_ls, "show": cmd_show, "compare": cmd_compare,
          "coverage": cmd_coverage, "check": cmd_check, "gbench-export": cmd_gbench_export, "gbench-import": cmd_gbench_import}[a.cmd]
    rc = fn(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
