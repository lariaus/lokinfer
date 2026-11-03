"""Apple-side equivalents of the usual GPU/CPU profiling tools, wrapped as prof-lokinfer subcommands
(docs/profiling.md "Tool map"):

  shader code   metal-objdump on the metallib -> AIR (LLVM IR) per function       (the "PTX" level)
                MTLBinaryArchive -> native GPU binary per GPU architecture        (the "SASS" level: size only;
                                    Apple ships no disassembler for it)
  pipelines     MTLComputePipelineState: max threads / threadgroup, SIMD width, static threadgroup memory
  traces        xcrun xctrace: Metal System Trace, Metal GPU Counters, Time Profiler, CPU Counters (Mac + iPhone)
  CPU sampling  /usr/bin/sample (Mac)

Everything shells out to Xcode's CLI tools (xcrun); nothing opens a GUI.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .targets import ROOT


def sh(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}): {p.stderr.strip()[-2000:]}")
    return p


# ------------------------------------------------------------------------------------------------- metallib / AIR

def default_metallib() -> Path:
    """The metallib the Python package (and so prof-lokinfer's Mac runs) actually loads."""
    from lokinfer import _lokinfer as _C

    return Path(_C.__file__).with_name("lokinfer.metallib")


def list_functions(metallib: Path) -> list[str]:
    out = sh(["xcrun", "metal-nm", str(metallib)]).stdout
    return sorted({ln.split()[-1] for ln in out.splitlines() if " T " in ln})


_FN_HDR = re.compile(r"^0x[0-9a-f]+ -- (\S+):\s*$")


def split_air_dump(dump: str) -> dict[str, str]:
    """`metal-objdump -d lib.metallib` output -> {function: its AIR module text}."""
    out: dict[str, list[str]] = {}
    cur = None
    for line in dump.splitlines():
        m = _FN_HDR.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v).strip() + "\n" for k, v in out.items()}


def air_modules(metallib: Path) -> dict[str, str]:
    return split_air_dump(sh(["xcrun", "metal-objdump", "-d", str(metallib)]).stdout)


_ADDRSPACE = {"1": "device", "2": "constant", "3": "threadgroup", "0": "thread"}


def air_stats(text: str) -> dict:
    """Static instruction mix of one function's AIR: instruction count, loads / stores per address space, FP ops,
    simdgroup-matrix ops, barriers, loops (back edges), source file."""
    c: Counter = Counter()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith((";", "!", "declare", "attributes", "target", "source_filename", "define", "}", "%struct")):
            continue
        if re.match(r"^\d+:", s):
            c["blocks"] += 1
            continue
        c["instructions"] += 1
        m = re.search(r"=\s*(load|store)\b|^(store)\b", s)
        op = (m.group(1) or m.group(2)) if m else None
        if op:
            sp = re.search(r"addrspace\((\d)\)\*", s)
            c[f"{op}_{_ADDRSPACE.get(sp.group(1), 'other') if sp else 'thread'}"] += 1
        for k, pat in (("fma", r"@llvm\.fma|@air\.fma"), ("fmul", r"\bfmul\b"), ("fadd", r"\bfadd\b|\bfsub\b"),
                       ("simdgroup_matrix", r"simdgroup_matrix"), ("simd_shuffle", r"simd_shuffle|simd_sum|simd_max"),
                       ("barrier", r"air\.wg\.barrier|threadgroup_barrier|air\.mem\.barrier"),
                       ("math_call", r"@air\.(fast_)?(exp|log|sin|cos|tanh|sqrt|rsqrt|pow)"), ("call", r"\bcall\b")):
            if re.search(pat, s):
                c[k] += 1
        if re.search(r"\bbr\b.*label", s) and "!llvm.loop" in s:
            c["loops"] += 1
    m = re.search(r'!DIFile\(filename: "([^"]+\.metal)"', text)
    if m:
        c_src = m.group(1)
    else:
        c_src = None
    out = dict(c)
    out["source"] = c_src
    return out


def find_source(function: str) -> tuple[Path, int] | None:
    """The .metal file + line declaring `function` (plain `kernel void name(` or `[[host_name("name")]]`)."""
    exact = [re.compile(rf"\bkernel\s+void\s+{re.escape(function)}\s*\("),
             re.compile(rf'host_name\("{re.escape(function)}"\)')]
    loose = re.compile(rf"\b{re.escape(function)}\b")  # macro-generated kernels: LK_BINARY_F32(add_f32, ...)
    files = sorted((ROOT / "kernels" / "gpu").glob("*.metal")) + sorted((ROOT / "bench" / "kernels").glob("*.metal"))
    for pats in (exact, [loose]):
        for f in files:
            for i, line in enumerate(f.read_text().splitlines()):
                if any(p.search(line) for p in pats):
                    return f, i + 1
    return None


def compile_air_text(source: Path, out: Path, sdk: str = "macosx", opt: str = "-O3") -> Path:
    """Recompiles one .metal to textual AIR (`metal -S`), with the build's flags (cmake/LokiMetal.cmake)."""
    minver = "-mios-version-min=27.0" if sdk == "iphoneos" else "-mmacosx-version-min=27.0"
    sh(["xcrun", "-sdk", sdk, "metal", "-std=metal4.1", minver, opt, "-S", "-I", str(ROOT / "third_party" / "mlx_steel"),
        str(source), "-o", str(out)])
    return out


# --------------------------------------------------------------------------------------------- pipelines / native

def inspect_local(functions=(), archive: Path | None = None) -> dict:
    from lokinfer import _lokinfer as _C

    return json.loads(_C.gpu_inspect(list(functions), str(archive) if archive else ""))


def inspect_remote(dev, functions=(), archive: Path | None = None) -> dict:
    info, blob = dev.gpu_inspect(list(functions), archive=archive is not None)
    if archive is not None and blob:
        archive.write_bytes(blob)
        info["archive"] = str(archive)
    return info


def archive_slices(archive: Path) -> dict[str, dict]:
    """Native GPU slices of a binary archive: {arch: {"sections": {name: bytes}, "file": path}}. The air64 slice is
    the AIR fallback; applegpu_* slices hold machine code (__compute)."""
    info = sh(["xcrun", "metal-lipo", "-info", str(archive)], check=False).stdout
    archs = re.findall(r"(applegpu_\w+|air64\w*)", info)
    out = {}
    for arch in dict.fromkeys(archs):
        if not arch.startswith("applegpu"):
            continue
        thin = archive.with_name(f"{archive.stem}.{arch}")
        sh(["xcrun", "metal-lipo", "-thin", arch, str(archive), "-output", str(thin)])
        hdr = sh(["xcrun", "metal-objdump", "-h", str(thin)], check=False).stdout
        secs = {}
        for ln in hdr.splitlines():
            m = re.match(r"\s*\d+\s+(\S+)\s+([0-9a-f]+)\s", ln)
            if m:
                secs[m.group(1)] = int(m.group(2), 16)
        out[arch] = {"sections": secs, "file": str(thin)}
    return out


def native_sizes(functions: list[str], inspect, workdir: Path) -> list[dict]:
    """Per function: native code size (__compute bytes) on the inspected device's GPU, plus its pipeline stats."""
    rows = []
    for fn in functions:
        arc = workdir / f"{fn}.binarchive"
        info = inspect([fn], arc)
        st = (info.get("functions") or [{}])[0]
        row = {"function": fn, **{k: v for k, v in st.items() if k != "name"}}
        if arc.exists():
            for arch, s in archive_slices(arc).items():
                row["arch"] = arch
                row["native_compute_bytes"] = s["sections"].get("__compute")
                row["native_file"] = s["file"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------- xctrace

# Short name -> (xctrace template, extra instruments). "Metal GPU Counters" is an instrument, not a template.
TEMPLATES = {"metal": ("Metal System Trace", []), "gpu-counters": ("Metal System Trace", ["Metal GPU Counters"]),
             "time": ("Time Profiler", []), "cpu-counters": ("CPU Counters", []), "system": ("System Trace", [])}


def template_label(name: str) -> str:
    t, extra = TEMPLATES.get(name, (name, []))
    return t + "".join(f" + {i}" for i in extra)


def xctrace_record(template: str, output: Path, *, launch: list[str] | None = None, attach: str | None = None,
                   device: str | None = None, time_limit: str = "10s", env: dict | None = None) -> subprocess.CompletedProcess:
    tmpl, instruments = TEMPLATES.get(template, (template, []))
    cmd = ["xcrun", "xctrace", "record", "--template", tmpl, "--output", str(output),
           "--time-limit", time_limit, "--no-prompt"]
    for i in instruments:
        cmd += ["--instrument", i]
    if device:
        cmd += ["--device", device]
    for k, v in (env or {}).items():
        cmd += ["--env", f"{k}={v}"]
    if attach:
        cmd += ["--attach", attach]
    elif launch:
        cmd += ["--launch", "--"] + launch
    else:
        cmd += ["--all-processes"]
    return subprocess.run(cmd, capture_output=True, text=True)


def device_pid(udid: str, path_suffix: str) -> int | None:
    """PID of a process on an attached device (`devicectl device info processes`), by executable path suffix."""
    out = sh(["xcrun", "devicectl", "device", "info", "processes", "--device", udid], check=False).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].endswith(path_suffix):
            return int(parts[0])
    return None


def xctrace_toc(trace: Path) -> str:
    return sh(["xcrun", "xctrace", "export", "--input", str(trace), "--toc"]).stdout


def toc_schemas(toc_xml: str) -> list[str]:
    root = ET.fromstring(toc_xml)
    return [t.get("schema") for t in root.iter("table") if t.get("schema")]


def xctrace_table(trace: Path, schema: str, run: int = 1) -> str:
    xp = f'/trace-toc/run[@number="{run}"]/data/table[@schema="{schema}"]'
    return sh(["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xp]).stdout


def parse_rows(xml_text: str) -> tuple[list[str], list[list[ET.Element]]]:
    """xctrace export rows with id/ref de-duplication resolved: (column mnemonics, rows of resolved elements)."""
    root = ET.fromstring(xml_text)
    cols = [c.findtext("mnemonic") for c in root.iter("col")]
    ids: dict[str, ET.Element] = {}

    def index(el):
        if el.get("id") is not None:
            ids[el.get("id")] = el
        for ch in el:
            index(ch)

    rows = []
    for row in root.iter("row"):
        index(row)
        rows.append([ids.get(el.get("ref"), el) if el.get("ref") is not None else el for el in row])
    # second pass: refs inside nested elements (frames) resolve against the full index
    return cols, [[_resolve(el, ids) for el in r] for r in rows]


def _resolve(el: ET.Element, ids: dict) -> ET.Element:
    if el.get("ref") is not None:
        el = ids.get(el.get("ref"), el)
    return el


def time_profile_summary(xml_text: str, process: str | None = None, top: int = 40) -> dict:
    """Self (leaf frame) and total (any frame) weight per symbol from a `time-profile` table."""
    root = ET.fromstring(xml_text)
    ids: dict[str, ET.Element] = {}
    for el in root.iter():
        if el.get("id") is not None:
            ids[el.get("id")] = el
    r = lambda el: ids.get(el.get("ref"), el) if el is not None and el.get("ref") is not None else el  # noqa: E731
    self_w: Counter = Counter()
    total_w: Counter = Counter()
    threads: Counter = Counter()
    n = 0
    for row in root.iter("row"):
        proc = r(row.find("process"))
        if process and proc is not None and process.lower() not in (proc.get("fmt") or "").lower():
            continue
        w_el = r(row.find("weight"))
        w = int(w_el.text) if w_el is not None and w_el.text else 1
        bt = r(row.find("tagged-backtrace"))
        if bt is None:
            bt = r(row.find("backtrace"))
        frames = [] if bt is None else [r(f) for f in bt.iter("frame")]
        names = [f.get("name") or f.get("addr") or "?" for f in frames if f is not None]
        th = r(row.find("thread"))
        threads[(th.get("fmt") if th is not None else "?")] += w
        n += 1
        if names:
            self_w[names[0]] += w
            for nm in set(names):
                total_w[nm] += w
    tot = sum(self_w.values()) or 1
    return {"samples": n, "weight_ns": tot,
            "self": [{"symbol": k, "ms": v / 1e6, "pct": 100 * v / tot} for k, v in self_w.most_common(top)],
            "total": [{"symbol": k, "ms": v / 1e6, "pct": 100 * v / tot} for k, v in total_w.most_common(top)],
            "threads": [{"thread": k, "ms": v / 1e6} for k, v in threads.most_common(12)]}


def _cell(el) -> str:
    return (el.get("fmt") or el.text or "").strip() if el is not None else ""


def interval_summary(xml_text: str, label_col: str | None = None, process: str | None = None, top: int = 40) -> dict:
    """Interval table (start / duration columns): total duration and count per `label_col` (default: the first of
    label / name / channel-name ...), optionally only rows whose `process` column contains `process`."""
    cols, rows = parse_rows(xml_text)
    out = {"rows": len(rows), "columns": cols}
    if not rows or "duration" not in cols:
        return out
    di = cols.index("duration")
    cands = [label_col] if label_col else ["name", "label", "channel-name", "event-label", "event-type"]
    li = next((cols.index(c) for c in cands if c in cols), None)
    if li is None:
        li = next((i for i, c in enumerate(cols) if c not in ("start", "duration", "timestamp")), None)
    pi = cols.index("process") if (process and "process" in cols) else None
    acc: dict = defaultdict(lambda: [0, 0])
    busy = []
    for r in rows:
        if di >= len(r) or (pi is not None and pi < len(r) and process.lower() not in _cell(r[pi]).lower()):
            continue
        try:
            d = int(r[di].text or 0)
        except ValueError:
            continue
        lab = _cell(r[li]) if li is not None and li < len(r) else "(all)"
        acc[lab][0] += d
        acc[lab][1] += 1
        if "start" in cols:
            try:
                t0 = int(r[cols.index("start")].text or 0)
                busy.append((t0, t0 + d))
            except ValueError:
                pass
    items = sorted(acc.items(), key=lambda kv: -kv[1][0])[:top]
    out.update({"label_column": cols[li] if li is not None else None, "filtered_process": process,
                "total_ms": sum(v[0] for v in acc.values()) / 1e6, "count": sum(v[1] for v in acc.values()),
                "by_label": [{"label": k, "ms": v[0] / 1e6, "count": v[1]} for k, v in items]})
    if busy:
        from .analysis import union_ns

        out["busy_ms"] = union_ns(busy) / 1e6
        out["span_ms"] = (max(b[1] for b in busy) - min(b[0] for b in busy)) / 1e6
    return out


def counter_summary(xml_text: str, top: int = 60) -> dict:
    """GPU counter intervals (Metal GPU Counters template): duration-weighted mean value per counter name."""
    cols, rows = parse_rows(xml_text)
    out = {"rows": len(rows), "columns": cols}
    if not rows or "name" not in cols:
        return out
    ni, di = cols.index("name"), cols.index("duration") if "duration" in cols else None
    vi = cols.index("percent-value") if "percent-value" in cols else cols.index("value") if "value" in cols else None
    if vi is None:
        return out
    acc: dict = defaultdict(lambda: [0.0, 0.0, 0.0])  # weighted sum, weight, max
    for r in rows:
        try:
            v = float(r[vi].text or 0)
            w = float(r[di].text or 1) if di is not None else 1.0
        except (ValueError, IndexError):
            continue
        a = acc[_cell(r[ni])]
        a[0] += v * w
        a[1] += w
        a[2] = max(a[2], v)
    items = sorted(acc.items(), key=lambda kv: kv[0])[:top]
    out["counters"] = [{"counter": k, "mean": a[0] / a[1] if a[1] else 0.0, "max": a[2]} for k, a in items]
    return out


_ELEM = re.compile(r'<([\w-]+)((?:\s+[\w-]+="[^"]*")*)\s*(?:/>|>([^<]*)</\1>)')
_ATTR_ID = re.compile(r'\bid="(\d+)"')
_ATTR_REF = re.compile(r'\bref="(\d+)"')


def counter_values_summary(value_xml: Path, info_xml: str) -> dict:
    """GPU counter samples (`gpu-counter-value`, millions of rows for a few seconds) streamed line by line: mean, mean
    over non-zero samples (~ while the GPU works) and max per counter, named through `gpu-counter-info`."""
    from array import array

    cols, rows = parse_rows(info_xml)
    names = {}
    if "counter-id" in cols and "name" in cols:
        ci, ni = cols.index("counter-id"), cols.index("name")
        names = {int(r[ci].text): _cell(r[ni]) for r in rows if ci < len(r) and ni < len(r)}
    vals = array("d")  # element id -> numeric value (ids are dense and increasing)
    acc: dict = defaultdict(lambda: [0.0, 0, 0.0, 0, 0.0])  # sum, n, sum nonzero, n nonzero, max
    n_rows = 0
    with open(value_xml, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("<row>"):
                continue
            els = _ELEM.findall(line[5:])
            if len(els) < 3:
                continue
            got = []
            for tag, attrs, text in els[1:3]:  # counter-id, value
                m = _ATTR_REF.search(attrs)
                if m:
                    i = int(m.group(1))
                    got.append(vals[i] if i < len(vals) else 0.0)
                    continue
                try:
                    v = float(text)
                except ValueError:
                    v = 0.0
                m = _ATTR_ID.search(attrs)
                if m:
                    i = int(m.group(1))
                    if i >= len(vals):
                        vals.extend([0.0] * (i + 1 - len(vals) + 4096))
                    vals[i] = v
                got.append(v)
            # ids of the other columns (timestamps, accelerator) are never referenced as counter ids / values
            cid, v = int(got[0]), got[1]
            a = acc[cid]
            a[0] += v
            a[1] += 1
            if v:
                a[2] += v
                a[3] += 1
            a[4] = max(a[4], v)
            n_rows += 1
    out = [{"counter": names.get(c, f"counter {c}"), "mean": a[0] / a[1] if a[1] else 0.0,
            "mean_nonzero": a[2] / a[3] if a[3] else 0.0, "nonzero_pct": 100.0 * a[3] / a[1] if a[1] else 0.0,
            "max": a[4]} for c, a in sorted(acc.items())]
    return {"rows": n_rows, "counters": out}


def metric_summary(xml_text: str, process: str | None = None) -> dict:
    """CPU Counters (`MetricAggregationForProcess`): per metric name (useful / processing / delivery / discarded
    bottleneck breakdown, cycles, ...) the mean of its double value, the sum of its int value, and each double
    metric's share of their total (the top-down breakdown)."""
    cols, rows = parse_rows(xml_text)
    out = {"rows": len(rows), "columns": cols}
    need = ("metric-name", "metric-value-double", "metric-value-int")
    if not rows or not all(c in cols for c in need):
        return out
    ni, di, ii = (cols.index(c) for c in need)
    pi = cols.index("process") if (process and "process" in cols) else None
    acc: dict = defaultdict(lambda: [0.0, 0, 0])  # sum double, n, sum int
    for r in rows:
        if pi is not None and process.lower() not in _cell(r[pi]).lower():
            continue
        try:
            a = acc[_cell(r[ni])]
            a[0] += float(r[di].text or 0)
            a[1] += 1
            a[2] += int(r[ii].text or 0)
        except (ValueError, IndexError):
            continue
    tot = sum(a[0] for a in acc.values()) or 1.0
    out["metrics"] = [{"metric": k, "mean": a[0] / a[1] if a[1] else 0.0, "share_pct": 100 * a[0] / tot, "int_sum": a[2]}
                      for k, a in sorted(acc.items(), key=lambda kv: -kv[1][0])]
    return out


# Tables summarized from a trace (schema -> how). Everything else stays in the .trace (and the TOC lists it).
CURATED = {
    "time-profile": "time_profile",
    "metal-gpu-intervals": ("intervals", "channel-name"),
    "gpu-performance-state-intervals": ("intervals", "gpu-performance-state"),
    "device-thermal-state-intervals": ("intervals", "thermal-state"),
    "metal-shader-profiler-intervals": ("intervals", "name"),
    "metal-gpu-counter-intervals": "counters",
    "gpu-counter-value": "counter_values",
    "MetricAggregationForProcess": "metrics",  # -t cpu-counters  # only recorded with -t gpu-counters; large (~1 GB of XML for 3 s)
}


def summarize_trace(trace: Path, out_dir: Path, process: str | None = None) -> dict:
    """Exports the TOC and the curated tables (CURATED); writes trace_summary.json + trace_summary.txt."""
    toc = xctrace_toc(trace)
    (out_dir / "toc.xml").write_text(toc)
    schemas = toc_schemas(toc)
    summary: dict = {"trace": str(trace), "schemas": sorted(set(schemas)), "tables": {}}
    for schema, how in CURATED.items():
        if schema not in schemas:
            continue
        if how == "counter_values":  # streamed to a file, summarized line by line, then deleted
            dest = out_dir / f"{schema}.xml"
            xp = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
            try:
                sh(["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xp, "--output", str(dest)])
                info = xctrace_table(trace, "gpu-counter-info") if "gpu-counter-info" in schemas else "<x/>"
                summary["tables"][schema] = counter_values_summary(dest, info)
            except RuntimeError as e:
                summary["tables"][schema] = {"error": str(e)[-300:]}
            finally:
                dest.unlink(missing_ok=True)
            continue
        try:
            x = xctrace_table(trace, schema)
        except RuntimeError as e:
            summary["tables"][schema] = {"error": str(e)[-300:]}
            continue
        (out_dir / f"{schema}.xml").write_text(x)
        if how == "time_profile":
            summary["tables"][schema] = time_profile_summary(x, process)
        elif how == "counters":
            summary["tables"][schema] = counter_summary(x)
        elif how == "metrics":
            summary["tables"][schema] = metric_summary(x, process)
        else:
            summary["tables"][schema] = interval_summary(
                x, how[1], process if schema in ("metal-gpu-intervals", "metal-shader-profiler-intervals") else None)
    (out_dir / "trace_summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "trace_summary.txt").write_text(format_trace_summary(summary))
    return summary


def format_trace_summary(s: dict) -> str:
    lines = [f"trace: {s['trace']}", f"tables in the trace: {len(s['schemas'])} (toc.xml)", ""]
    for schema, t in s["tables"].items():
        lines.append(f"== {schema}")
        if "self" in t:
            lines.append(f"  {t['samples']} samples, {t['weight_ns'] / 1e6:.1f} ms weight; top self:")
            lines += [f"  {r['ms']:9.2f} ms {r['pct']:5.1f}%  {r['symbol']}" for r in t["self"][:20]]
            lines.append("  top total (inclusive):")
            lines += [f"  {r['ms']:9.2f} ms {r['pct']:5.1f}%  {r['symbol']}" for r in t["total"][:15]]
        elif "by_label" in t:
            extra = f", busy {t['busy_ms']:.1f} ms of a {t['span_ms']:.1f} ms span" if "busy_ms" in t else ""
            flt = f" (process ~ {t['filtered_process']})" if t.get("filtered_process") else ""
            lines.append(f"  {t['count']} intervals{flt}, {t['total_ms']:.1f} ms total{extra}; by {t['label_column']}:")
            lines += [f"  {r['ms']:10.3f} ms  x{r['count']:<6} {r['label']}" for r in t["by_label"][:20]]
        elif "counters" in t and t["counters"] and "mean_nonzero" in t["counters"][0]:
            lines.append(f"  {t['rows']} samples; mean | mean while non-zero (share of samples) | max:")
            lines += [f"  {r['mean']:9.2f} | {r['mean_nonzero']:9.2f} ({r['nonzero_pct']:5.1f}%) | {r['max']:9.2f}  "
                      f"{r['counter']}" for r in t["counters"]]
        elif "counters" in t:
            lines += [f"  {r['mean']:12.3f} (max {r['max']:.3f})  {r['counter']}" for r in t["counters"]]
        elif "metrics" in t:
            lines.append("  metric: mean per sample | share of the double-valued metrics | int sum")
            lines += [f"  {r['mean']:12.4f} | {r['share_pct']:5.1f}% | {r['int_sum']:>16}  {r['metric']}"
                      for r in t["metrics"]]
        elif "error" in t:
            lines.append(f"  export failed: {t['error']}")
        else:
            lines.append(f"  {t.get('rows', 0)} rows")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- CPU: sample(1)

def sample_pid(pid: int, seconds: int, out: Path) -> str:
    sh(["/usr/bin/sample", str(pid), str(seconds), "-file", str(out)], check=False)
    return out.read_text() if out.exists() else ""


def sample_top_of_stack(text: str, top: int = 30) -> list[tuple[str, int]]:
    """The 'Sort by top of stack' section of sample(1)'s report: [(symbol, samples)]."""
    i = text.find("Sort by top of stack")
    if i < 0:
        return []
    rows = []
    for ln in text[i:].splitlines()[1:]:
        m = re.match(r"\s+(.+?)\s+(\d+)\s*$", ln)
        if not m:
            if rows:
                break
            continue
        rows.append((m.group(1).strip(), int(m.group(2))))
        if len(rows) >= top:
            break
    return rows


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None or sh(["xcrun", "-f", name], check=False).returncode == 0


def tempdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="lokinfer-metal-"))
