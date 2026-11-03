"""Turns raw profiles into the run format shared by prof-lokinfer, prof-torch and mlogs-viewer.

Raw lokinfer profile (Executable::takeProfileJson, docs/profiling.md "Raw profile"):
    {"version": 1, "uses_gpu": bool, "num_exec_ops": n,
     "ops": [[op, kernel, "cpu"|"gpu", num_operands, [arg ranks], [arg dtypes]], ...],
     "calls": [{"call", "total_ns", "gpu_mode": "stream"|"serialize"|"none", "gpu_timestamps": bool,
                "kernels": [[op, cpu_t0, cpu_t1, gpu_t0, gpu_t1, [ints], cpu_wait], ...],  # ns from the call start
                # cpu_wait (optional 7th field): part of cpu_t1 - cpu_t0 blocked on the previous in-flight segment
                "events": [[op, "commit"|"sync_to_cpu"|"sync_to_gpu"|"finish", t0, t1], ...],
                "dispatches": [[op, function, gx, gy, gz, tx, ty, tz, threadgroups], ...],
                "memory": [{"op", "kind": "plan", "total", "arena", "align", "buffers": [[size, start, end, offset]]},
                           {"op", "kind": "alloc", "bytes", "output"}]}]}

Run format ("lokinfer-prof/1", one run.json per run directory): see build_run() and docs/profiling.md.
Everything here is pure (no lokinfer / torch imports) so it is unit-tested on fixtures (tests/tools).
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict

SCHEMA = "lokinfer-prof/1"

# --------------------------------------------------------------------------------------------- op categories
# Coarse buckets that exist on both sides, so a lokinfer kernel table and a torch op table line up
# ("conv" vs aten::conv1d / mkldnn_convolution / _mps_convolution). First match wins; order matters.
CATEGORIES: list[tuple[str, str, str]] = [
    # (category, lokinfer kernel regex, torch op regex)
    ("lstm", r"lstm", r"lstm|rnn|gru"),
    ("attention", r"attention|softmax|flash", r"softmax|scaled_dot_product|attention|baddbmm"),
    ("conv_transpose", r"convt|conv_transpose|upsample_conv|deconv", r"conv_transpose|convolution_backward"),
    ("conv", r"conv|resblock|resblk", r"conv|convolution|mkldnn_conv"),
    ("matmul", r"gemm|gemv|matmul|linear|style_bank|lm_head|qkv", r"addmm|\bmm\b|aten::mm$|bmm|matmul|linear|addmv|einsum|\bdot\b"),
    ("norm", r"norm|adain|layernorm|ln\d?\b|_ln|instance|stats", r"norm|instance_norm|layer_norm|group_norm|batch_norm|native_batch"),
    ("signal", r"stft|istft|sine|harmonic|source|noise|rng|randn|f0|phase|window|interp|upsample(?!_conv)",
     r"stft|istft|fft|cumsum|randn|rand|interpolate|upsample|_fft|atan2|angle|polar"),
    ("embedding", r"embed|gather_rows|lookup", r"embedding|index_select"),
    ("layout", r"transpose|concat|broadcast|copy|pad|gather|expand|repeat|align|duration|cast|to_|slice|view",
     r"copy_|\bcat\b|aten::cat|transpose|permute|contiguous|clone|pad|index|repeat|zeros|ones|empty|_to_copy|aten::to$|"
     r"expand|stack|slice|select|narrow|reshape|view|flip|fill_|scatter|arange|where|masked_fill|unsqueeze|squeeze"),
    ("activation", r"relu|gelu|snake|act|sigmoid|tanh|leaky|silu|exp|sin_|clamp",
     r"relu|gelu|sigmoid|tanh|leaky|silu|\bexp\b|aten::exp|clamp|hardtanh|elu|softplus|snake|aten::sin\b|aten::cos\b"),
    ("elementwise", r"add|mul|sub|div|scale|bias|axpy|residual|sum|mean|reduce|round",
     r"add|mul|sub|div|pow|sqrt|rsqrt|neg|abs|sum|mean|round|reciprocal|max|min|argmax|lerp|addcmul|rsub"),
]
_CAT_RE = [(c, re.compile(lk, re.I), re.compile(tt, re.I)) for c, lk, tt in CATEGORIES]


def category(name: str, side: str = "lokinfer") -> str:
    """Coarse op category of a lokinfer kernel name (side="lokinfer") or a torch op name (side="torch")."""
    for c, lk_re, torch_re in _CAT_RE:
        if (lk_re if side == "lokinfer" else torch_re).search(name):
            return c
    return "other"


# ------------------------------------------------------------------------------------------------- helpers

def fmt_shape(dtype: str, dims) -> str:
    return f"{dtype}[{','.join(str(d) for d in dims)}]"


def split_ints(ints: list[int], ranks: list[int]) -> tuple[list[list[int]], list[int]]:
    """A kernel's int args = every argument's dims (in argument order) then its int attributes."""
    shapes, at = [], 0
    for r in ranks:
        shapes.append(list(ints[at:at + r]))
        at += r
    return shapes, list(ints[at:])


def signature(op_info: list, ints: list[int]) -> str:
    """"f32[1,128] f32[128,64] -> f32[1,64] {attrs}" for one kernel instance."""
    _, _, _, nin, ranks, dtypes = op_info
    shapes, attrs = split_ints(ints, ranks)
    args = [fmt_shape(dtypes[i] if i < len(dtypes) else "?", s) for i, s in enumerate(shapes)]
    sig = " ".join(args[:nin]) + " -> " + " ".join(args[nin:])
    if attrs:
        sig += " {" + ",".join(str(a) for a in attrs) + "}"
    return sig


_MATMUL_LIKE = ("gemm", "gemv", "conv1d", "conv2d", "dwconv")


def flops_of(name: str, sig: str) -> float | None:
    """Arithmetic FLOPs (2 per multiply-add) of one GEMM / GEMV / conv kernel instance from its signature, else None.
    One rule covers them all: out f32[..., Co] and the weight (2nd input) holding every multiply per output row, so
    FLOPs = 2 · (out elements / last dim) · (weight elements) — gemm a[M,K] b[K,N] -> [M,N]; conv w[Co,K(,K),Ci] ->
    [.., L, Co]; depthwise w[KH,KW,C] -> [.., C]. Not for gathers (convT), im2col or layout kernels."""
    if not any(t in name for t in _MATMUL_LIKE) or "gather" in name or "im2col" in name:
        return None
    if "wino" in name:  # Winograd ops carry transformed weights (~2x the direct conv's), not the conv's real work
        return None
    try:
        ins, outs = sig.split(" -> ")
        shapes = [[int(d) for d in a[a.index("[") + 1:a.index("]")].split(",") if d] for a in ins.split()]
        out = [int(d) for d in outs.split()[0].split("[")[1].split("]")[0].split(",")]
    except (ValueError, IndexError):
        return None
    if len(shapes) < 2 or not out:
        return None
    if "splitk" in name and len(out) == 3:  # split-K partials f32[S, M, N]: the S planes split K, not M
        out = out[1:]
    # CPU 1-D convs (conv1d*_cpu, conv_transpose1d*) are channel-first: out [Co, L], one row of work per time step.
    channel_first = "conv1d" in name and "_lc" not in name and name.endswith("_cpu")
    rows = out[-1] if channel_first else 1
    for d in ([] if channel_first else out[:-1]):
        rows *= d
    w = 1
    for d in shapes[1]:
        w *= d
    return 2.0 * rows * w


def union_ns(intervals) -> int:
    """Total length of the union of [t0, t1) intervals (device busy time)."""
    total, end = 0, None
    for t0, t1 in sorted(i for i in intervals if i[1] > i[0]):
        if end is None or t0 > end:
            total += t1 - t0
            end = t1
        elif t1 > end:
            total += t1 - end
            end = t1
    return total


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"median": statistics.median(s), "p10": s[max(0, len(s) // 10)], "p90": s[min(len(s) - 1, (9 * len(s)) // 10)],
            "min": s[0], "max": s[-1], "n": len(s)}


def plan_lower_bound(buffers: list[list[int]], align: int) -> int:
    """Max over time of the sum of live (aligned) sizes: what any planner needs at least (mirrors planLowerBound)."""
    events = defaultdict(int)
    for size, start, end, _ in buffers:
        a = (size + align - 1) // align * align
        events[start] += a
        events[end + 1] -= a
    live = peak = 0
    for t in sorted(events):
        live += events[t]
        peak = max(peak, live)
    return peak


# ------------------------------------------------------------------------------------ lokinfer: raw -> run

def kernel_instances(raw: dict) -> list[dict]:
    """One row per (call, kernel instance) with derived times in ns."""
    ops = {o[0]: o for o in raw["ops"]}
    rows = []
    for call in raw["calls"]:
        for k in call["kernels"]:
            op, t0, t1, g0, g1, ints = k[:6]
            wait = k[6] if len(k) > 6 else 0
            info = ops.get(op, [op, f"op{op}", "cpu", 0, [], []])
            dev = info[2]
            gpu_ns = g1 - g0 if (g0 >= 0 and g1 >= g0) else None
            cpu_ns = t1 - t0
            # GPU kernels: the GPU timestamp interval is the kernel's own time; without it (serialize mode on a GPU
            # without counter heaps) the CPU wall time around commit + wait is the best estimate.
            time_ns = gpu_ns if (dev == "gpu" and gpu_ns is not None) else cpu_ns
            rows.append({"call": call["call"], "op": op, "name": info[1], "device": dev, "time_ns": time_ns,
                         "cpu_ns": cpu_ns, "wait_ns": wait, "gpu_ns": gpu_ns, "t0": t0, "t1": t1, "g0": g0, "g1": g1,
                         "sig": signature(info, ints)})
    return rows


def aggregate(rows: list[dict], key, ncalls: int, e2e_ns: float) -> list[dict]:
    """Per-key table: calls per iteration, total per iteration, mean / min / max per call, % of the e2e time."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = []
    for k, rs in groups.items():
        times = [r["time_ns"] for r in rs]
        total = sum(times) / max(ncalls, 1)
        first = rs[0]
        row = {"name": first["name"], "device": first["device"], "calls": len(rs) / max(ncalls, 1),
               "total_ms": total / 1e6, "mean_us": statistics.fmean(times) / 1e3, "min_us": min(times) / 1e3,
               "max_us": max(times) / 1e3, "pct": 100.0 * total / e2e_ns if e2e_ns else 0.0,
               "category": first.get("category") or category(first["name"], "torch" if first.get("torch") else "lokinfer")}
        if first["device"] == "gpu":
            row["encode_us"] = statistics.fmean(r["cpu_ns"] - r.get("wait_ns", 0) for r in rs) / 1e3
            row["wait_us"] = statistics.fmean(r.get("wait_ns", 0) for r in rs) / 1e3
        fl = [flops_of(r["name"], r["sig"]) for r in rs if r.get("sig")]
        if fl and all(x is not None for x in fl) and sum(times):
            row["gflop"] = sum(fl) / max(ncalls, 1) / 1e9  # per iteration
            row["tflops"] = sum(fl) / (sum(times) * 1e-9) / 1e12  # achieved over the kernel's own time
        sigs = sorted({r["sig"] for r in rs if r.get("sig")})
        row["shapes"] = sigs[:12]
        row["num_shapes"] = len(sigs)
        if "op" in first:
            row["ops"] = sorted({r["op"] for r in rs})[:64]
        out.append(row)
    out.sort(key=lambda r: -r["total_ms"])
    return out


def category_table(kernels: list[dict], e2e_ms: float) -> list[dict]:
    acc: dict = defaultdict(lambda: {"total_ms": 0.0, "calls": 0.0, "kernels": 0})
    for k in kernels:
        a = acc[k["category"]]
        a["total_ms"] += k["total_ms"]
        a["calls"] += k["calls"]
        a["kernels"] += 1
    rows = [{"category": c, **v, "pct": 100 * v["total_ms"] / e2e_ms if e2e_ms else 0.0} for c, v in acc.items()]
    rows.sort(key=lambda r: -r["total_ms"])
    return rows


def timeline(raw: dict, call: dict) -> dict:
    """Lanes for one call: host thread (CPU kernels, GPU encodes, syncs) and GPU (timestamped kernels)."""
    ops = {o[0]: o for o in raw["ops"]}
    ev = []
    for k in call["kernels"]:
        op, t0, t1, g0, g1, ints = k[:6]
        wait = k[6] if len(k) > 6 else 0
        info = ops.get(op, [op, f"op{op}", "cpu", 0, [], []])
        sig = signature(info, ints)
        if info[2] == "gpu":
            if wait > 0:  # blocked on the previous in-flight segment before encoding (at the start of the kernel)
                ev.append({"lane": "CPU: sync", "name": "wait in-flight segment", "t0": t0, "t1": t0 + wait, "op": op})
            ev.append({"lane": "CPU: encode", "name": info[1], "t0": t0 + wait, "t1": t1, "op": op, "sig": sig})
            if g0 >= 0 and g1 >= g0:
                ev.append({"lane": "GPU", "name": info[1], "t0": g0, "t1": g1, "op": op, "sig": sig})
        else:
            ev.append({"lane": "CPU: kernels", "name": info[1], "t0": t0, "t1": t1, "op": op, "sig": sig})
    for op, kind, t0, t1 in call.get("events", []):
        ev.append({"lane": "CPU: sync", "name": kind, "t0": t0, "t1": t1 if t1 >= 0 else t0, "op": op})
    lanes = [l for l in ("CPU: kernels", "CPU: encode", "CPU: sync", "GPU") if any(e["lane"] == l for e in ev)]
    return {"call": call["call"], "total_ns": call["total_ns"], "lanes": lanes, "events": ev}


def memory_summary(call: dict) -> dict:
    plans, allocs = [], []
    for m in call.get("memory", []):
        if m["kind"] == "plan":
            lb = plan_lower_bound(m["buffers"], m["align"]) if m["buffers"] else 0
            plans.append({"op": m["op"], "total": m["total"], "arena": m["arena"], "align": m["align"],
                          "num_buffers": len(m["buffers"]), "lower_bound": lb,
                          "efficiency": (lb / m["total"]) if m["total"] else 1.0, "buffers": m["buffers"]})
        else:
            allocs.append(m)
    return {"plans": plans, "allocs": allocs, "arena_bytes": sum(p["arena"] for p in plans),
            "planned_bytes": sum(p["total"] for p in plans), "alloc_bytes": sum(a["bytes"] for a in allocs)}


def dispatch_table(raw: dict) -> list[dict]:
    """Per Metal function: dispatches per call, the kernels using it, example grids."""
    ops = {o[0]: o for o in raw["ops"]}
    ncalls = max(len(raw["calls"]), 1)
    acc: dict = {}
    for call in raw["calls"]:
        for op, fn, gx, gy, gz, tx, ty, tz, groups in call.get("dispatches", []):
            a = acc.setdefault(fn, {"function": fn, "dispatches": 0, "kernels": set(), "grids": set()})
            a["dispatches"] += 1
            a["kernels"].add(ops[op][1] if op in ops else str(op))
            if len(a["grids"]) < 6:
                unit = "tg" if groups else "threads"
                a["grids"].add(f"{gx}x{gy}x{gz} {unit} of {tx}x{ty}x{tz}")
    out = [{"function": a["function"], "dispatches": a["dispatches"] / ncalls, "kernels": sorted(a["kernels"]),
            "grids": sorted(a["grids"])} for a in acc.values()]
    out.sort(key=lambda r: -r["dispatches"])
    return out


def ir_exec_line(final_ir: str) -> int | None:
    """0-based line of the exec region header in the final IR: exec op i is on line exec_line + 1 + i."""
    for i, line in enumerate(final_ir.splitlines()):
        if line.lstrip().startswith("exec(") or line.lstrip().startswith("exec "):
            return i
    return None


def ir_stats(text: str) -> dict:
    """Op counts of an IR text (kernels per name, views, allocs, syncs, constants)."""
    kinds: dict = defaultdict(int)
    kernels: dict = defaultdict(int)
    for line in text.splitlines():
        m = re.search(r'kernel "([^"]+)"', line)
        if m:
            kinds["kernel"] += 1
            kernels[m.group(1)] += 1
            continue
        m = re.search(r"=\s*([a-z_]+(?:\.[a-z_]+)?)\b", line)
        if m:
            kinds[m.group(1)] += 1
        elif line.strip().startswith(("sync", "buffer.free", "return")):
            kinds[line.strip().split()[0]] += 1
    return {"ops": dict(kinds), "kernels": dict(sorted(kernels.items(), key=lambda kv: -kv[1]))}


def build_run(raw: dict, *, meta: dict, e2e_ms: list[float], ir_passes: list[dict] | None = None) -> dict:
    """The run.json of a lokinfer profile. `meta`: model, case, device, target, backend, created, machine, settings."""
    calls = raw["calls"]
    if not calls:
        raise ValueError("the profile recorded no calls")
    ncalls = len(calls)
    rows = kernel_instances(raw)
    profiled = [c["total_ns"] for c in calls]
    e2e = percentiles(e2e_ms) if e2e_ms else percentiles([p / 1e6 for p in profiled])
    e2e_ns = e2e["median"] * 1e6
    kernels = aggregate(rows, lambda r: r["name"], ncalls, e2e_ns)
    by_shape = aggregate(rows, lambda r: (r["name"], r["sig"]), ncalls, e2e_ns)
    per_op = aggregate(rows, lambda r: r["op"], ncalls, e2e_ns)
    med_call = sorted(calls, key=lambda c: c["total_ns"])[ncalls // 2]
    gpu_busy = [union_ns((k[3], k[4]) for k in c["kernels"] if k[3] >= 0 and k[4] > k[3]) for c in calls]
    ops = {o[0]: o for o in raw["ops"]}
    cpu_kern = [sum(k[2] - k[1] for k in c["kernels"] if ops.get(k[0], [0, 0, "cpu"])[2] == "cpu") for c in calls]
    sync_wait = [sum(max(0, e[3] - e[2]) for e in c.get("events", []) if e[1] != "commit") for c in calls]
    inflight_wait = [sum(k[6] for k in c["kernels"] if len(k) > 6) for c in calls]
    summary = {
        "e2e_ms": e2e,
        "profiled_ms": percentiles([p / 1e6 for p in profiled]),
        "profile_overhead_pct": 100.0 * (statistics.median(profiled) / e2e_ns - 1.0) if e2e_ns else 0.0,
        "gpu_busy_ms": statistics.fmean(gpu_busy) / 1e6,
        "cpu_kernel_ms": statistics.fmean(cpu_kern) / 1e6,
        "sync_wait_ms": statistics.fmean(sync_wait) / 1e6,
        "inflight_wait_ms": statistics.fmean(inflight_wait) / 1e6,
        "kernel_calls": len(med_call["kernels"]),
        "gpu_kernels": sum(1 for k in med_call["kernels"] if ops.get(k[0], [0, 0, "cpu"])[2] == "gpu"),
        "dispatches": len(med_call.get("dispatches", [])),
        "gpu_mode": med_call.get("gpu_mode"),
        "gpu_timestamps": med_call.get("gpu_timestamps"),
        "profiled_calls": ncalls,
    }
    if summary["gpu_busy_ms"] and e2e["median"]:
        summary["gpu_utilization_pct"] = 100.0 * summary["gpu_busy_ms"] / e2e["median"]
    run = {
        "schema": SCHEMA, "kind": "lokinfer", **meta,
        "summary": summary,
        "kernels": kernels,
        "kernels_by_shape": by_shape,
        "ops": per_op,
        "categories": category_table(kernels, e2e["median"]),
        "dispatches": dispatch_table(raw),
        "memory": memory_summary(calls[0]),
        "timeline": timeline(raw, med_call),
    }
    if ir_passes:
        run["ir"] = {"passes": [{"name": p["name"], "file": p.get("file"), "stats": ir_stats(p["text"]),
                                 "lines": p["text"].count("\n")} for p in ir_passes],
                     "exec_line": ir_exec_line(ir_passes[-1]["text"])}
    return run


# ------------------------------------------------------------------------------------------ torch: raw -> run

def build_torch_run(events: list[dict], *, meta: dict, e2e_ms: list[float], ncalls: int) -> dict:
    """run.json from per-op torch events (torch_prof.per_op_events): [{"name": "aten::mul", "device": "cpu"|"gpu",
    "t0", "t1" (ns from the call start), "call", "shapes": str, "module": innermost nn.Module class, "category"
    (optional: from the module, e.g. "lstm")}]. The events are leaf aten ops, so nothing is double counted."""
    rows = []
    for e in events:
        # ops inside a known nn.Module carry its category: shown as "aten::mul @lstm" (a row of its own)
        name = e["name"] + (f" @{e['category']}" if e.get("category") else "")
        rows.append({"call": e["call"], "name": name, "device": e.get("device", "cpu"),
                     "time_ns": e["t1"] - e["t0"], "cpu_ns": e["t1"] - e["t0"], "gpu_ns": None,
                     "sig": (e.get("module", "") + ": " if e.get("module") else "") + e.get("shapes", ""),
                     "torch": True, "category": e.get("category") or category(e["name"], "torch")})
    e2e = percentiles(e2e_ms)
    e2e_ns = e2e["median"] * 1e6
    kernels = aggregate(rows, lambda r: r["name"], ncalls, e2e_ns)
    by_shape = aggregate(rows, lambda r: (r["name"], r["sig"]), ncalls, e2e_ns)
    calls = sorted({e["call"] for e in events})
    mid = calls[len(calls) // 2] if calls else 0
    tl = [{"lane": "torch: " + e.get("lane", e.get("device", "cpu")), "name": e["name"], "t0": e["t0"], "t1": e["t1"],
           "sig": e.get("shapes", "")} for e in events if e["call"] == mid]
    tl_total = max((e["t1"] for e in tl), default=0)
    return {
        "schema": SCHEMA, "kind": "torch", **meta,
        "summary": {"e2e_ms": e2e, "kernel_calls": sum(1 for e in events if e["call"] == mid), "profiled_calls": ncalls,
                    "cpu_kernel_ms": sum(r["time_ns"] for r in rows) / max(ncalls, 1) / 1e6},
        "kernels": kernels, "kernels_by_shape": by_shape,
        "categories": category_table(kernels, e2e["median"]),
        "timeline": {"call": mid, "total_ns": tl_total, "lanes": sorted({e["lane"] for e in tl}), "events": tl},
    }


# --------------------------------------------------------------------------------------------------- compare

def compare_runs(a: dict, b: dict) -> dict:
    """Side-by-side of two runs: summary deltas, per-category and per-kernel (same names; e.g. two lokinfer runs)
    tables. Ratios are b / a (> 1: b is slower)."""
    def ratio(x, y):
        return (y / x) if x else None

    sa, sb = a["summary"]["e2e_ms"], b["summary"]["e2e_ms"]
    out = {"a": _label(a), "b": _label(b),
           "e2e": {"a_ms": sa.get("median"), "b_ms": sb.get("median"), "ratio": ratio(sa.get("median"), sb.get("median"))}}
    ca = {r["category"]: r for r in a.get("categories", [])}
    cb = {r["category"]: r for r in b.get("categories", [])}
    out["categories"] = sorted(
        ({"category": c, "a_ms": ca.get(c, {}).get("total_ms", 0.0), "b_ms": cb.get(c, {}).get("total_ms", 0.0),
          "a_calls": ca.get(c, {}).get("calls", 0.0), "b_calls": cb.get(c, {}).get("calls", 0.0),
          "ratio": ratio(ca.get(c, {}).get("total_ms", 0.0), cb.get(c, {}).get("total_ms", 0.0))}
         for c in set(ca) | set(cb)), key=lambda r: -max(r["a_ms"], r["b_ms"]))
    ka = {r["name"]: r for r in a.get("kernels", [])}
    kb = {r["name"]: r for r in b.get("kernels", [])}
    common = set(ka) & set(kb)
    rows = []
    for n in set(ka) | set(kb):
        x, y = ka.get(n), kb.get(n)
        rows.append({"name": n, "category": (x or y)["category"],
                     "a_ms": x["total_ms"] if x else None, "b_ms": y["total_ms"] if y else None,
                     "a_calls": x["calls"] if x else None, "b_calls": y["calls"] if y else None,
                     "ratio": ratio(x["total_ms"], y["total_ms"]) if (x and y) else None})
    rows.sort(key=lambda r: -max(r["a_ms"] or 0, r["b_ms"] or 0))
    out["kernels"] = rows
    out["kernels_matched"] = len(common)
    return out


def _label(r: dict) -> str:
    return f"{r.get('kind')} {r.get('model')} {r.get('case', '')} {r.get('device', '')} ({r.get('id', '')})".strip()
