"""Kernel benchmark cases at real model shapes (docs/benchmarking.md, "Kernel harness").

A *case* is one registered kernel at one concrete signature (operand / output types + attributes) that a real model
build executes. `extract()` builds every model of the profiler registry (`lokinfer.prof.targets`) for the CPU and the
GPU, runs one profiled call on the Mac, and pairs each exec-region kernel op of the final IR (attribute names and
values, scalar vs buffer operands) with the profile's concrete int arguments (every dim, incl. symbolic ones, resolved
for that call). Identical cases from several models / instances are merged; `instances` counts them per model.

`case_ir()` turns a case into a stand-alone buffer-form model: the exec region takes the operands as inputs, allocates
the outputs once and runs the kernel `repeats` times in a row on the same buffers (so on the GPU the repeats are
dependent dispatches in one command buffer: write-after-write on the outputs puts a barrier between them).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES_JSON = Path(__file__).with_name("cases.json")

# (registry model, case) whose kernels define the "real shapes": the dashboard's representative case per model.
MODELS = [
    ("kokoro", "app59f"),
    ("mnist", "B=64"),
    ("mobilenet_v2", "N=1 224x224"),
    ("resnet18", "N=1 224x224"),
    ("dcgan_g", "N=1"),
    ("dcgan_d", "N=1"),
    ("gpt2", "prefill T=128"),
    ("gpt2_decode", "decode P=512"),
]

STRUCTURAL = {"nohoist", "inplace", "output", "arena", "device", "persistent", "gpu_only"}  # dropped ("inout" kept)

_KERNEL_RE = re.compile(r'^\s*kernel "(?P<name>[^"]+)"\((?P<ops>[^)]*)\)(?: outs\((?P<outs>[^)]*)\))?'
                        r'(?: \{(?P<attrs>.*)\})?\s*$')
_PARAM_RE = re.compile(r"(%[\w.]+): (buffer<[^>]*>|[a-z][a-z0-9]*)")
_DEF_RE = re.compile(r"^\s*(%[\w.]+) = .* : (buffer<[^>]*>|[a-z][a-z0-9]*)\s*$")


def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[<(":
            depth += 1
        elif ch in "]>)":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _attr_kind(v: str) -> tuple[str, int]:
    """(kind, number of kernel ints it contributes) of an attribute value as printed by the IR printer."""
    if v in ("true", "false"):
        return "bool", 1
    if v.startswith("dim<"):
        return "dim", 1
    if v.startswith("["):
        items = [x for x in _split_top(v[1:-1])]
        if not items:
            return "ints", 0
        if any(re.search(r"[.eE]|nan|inf", x) for x in items):
            return "floats", 0
        return "ints", len(items)
    if v.startswith('"'):
        return "str", 0
    if re.fullmatch(r"-?\d+", v):
        return "int", 1
    return "float", 0


def parse_kernel_lines(ir_text: str) -> tuple[list[dict], dict]:
    """Kernel ops of the exec region in order, and every value's IR type (exec / state params and definitions)."""
    types: dict[str, str] = {}
    ex = ir_text[ir_text.index("exec("):]
    lines = ex.splitlines()
    for m in _PARAM_RE.finditer(lines[0]):
        types[m.group(1)] = m.group(2)
    kernels = []
    for line in lines[1:]:
        d = _DEF_RE.match(line)
        if d:
            types[d.group(1)] = d.group(2)
        k = _KERNEL_RE.match(line)
        if k:
            attrs = []
            for a in _split_top(k.group("attrs") or ""):
                name, val = (x.strip() for x in a.split("=", 1))
                attrs.append((name, val))
            kernels.append({"name": k.group("name"),
                            "operands": [x.strip() for x in k.group("ops").split(",") if x.strip()],
                            "outs": [x.strip() for x in (k.group("outs") or "").split(",") if x.strip()],
                            "attrs": attrs})
    return kernels, types


def _concrete_attrs(attrs: list[tuple[str, str]], attr_ints: list[int]) -> list[tuple[str, str]]:
    """Attributes with dim<...> values replaced by the profile's evaluated int (ABI order = printed order)."""
    out, at = [], 0
    for name, val in attrs:
        if name in STRUCTURAL:
            continue
        kind, n = _attr_kind(val)
        if kind == "dim":
            out.append((name, str(attr_ints[at])))
        else:
            out.append((name, val))
        at += n
    return out


def cases_from_profile(model_key: str, device: str, ir_text: str, raw: dict) -> list[dict]:
    """One case per kernel instance of the first profiled call (merged later)."""
    kernels, types = parse_kernel_lines(ir_text)
    ops = sorted(raw["ops"], key=lambda o: o[0])
    if len(ops) != len(kernels):
        raise RuntimeError(f"{model_key}/{device}: {len(ops)} profiled kernel ops vs {len(kernels)} kernel lines")
    by_op = {}
    for info, k in zip(ops, kernels):
        if info[1] != k["name"]:
            raise RuntimeError(f"{model_key}/{device}: op {info[0]} is {info[1]} in the profile, {k['name']} in the IR")
        by_op[info[0]] = (info, k)
    call = raw["calls"][0]
    # cpu wall time of each instance (GPU: GPU timestamps when present) as the size estimate for repeats
    out = []
    for rec in call["kernels"]:
        op, t0, t1, g0, g1, ints = rec[:6]
        info, k = by_op[op]
        _, name, dev, nin, ranks, dtypes = info
        args = k["operands"] + k["outs"]
        at, arg_types = 0, []
        for i, r in enumerate(ranks):
            dims = ints[at:at + r]
            at += r
            ir_t = types.get(args[i], "buffer<")
            if r == 0 and not ir_t.startswith("buffer<"):
                arg_types.append(dtypes[i])  # scalar
            else:
                arg_types.append(f"buffer<{dtypes[i]}[{', '.join(str(d) for d in dims)}]>")
        attrs = _concrete_attrs(k["attrs"], ints[at:])
        est_ns = (g1 - g0) if (dev == "gpu" and g0 >= 0 and g1 >= g0) else (t1 - t0)
        out.append({"kernel": name, "device": dev, "nin": nin, "args": arg_types,
                    "inout": [int(x) for x in re.findall(r"\d+", dict(k["attrs"]).get("inout", ""))],
                    "attrs": attrs, "est_us": est_ns / 1e3, "model": model_key})
    return out


def case_key(c: dict) -> str:
    """Stable identity of a case: kernel + signature + attributes."""
    return f"{c['kernel']}|{signature(c)}"


def signature(c: dict) -> str:
    def short(t: str) -> str:
        return t.replace("buffer<", "").replace(">", "").replace(", ", ",")

    sig = " ".join(short(t) for t in c["args"][:c["nin"]]) + " -> " + " ".join(short(t) for t in c["args"][c["nin"]:])
    if c["attrs"]:
        sig += " {" + ", ".join(f"{n}={v}" for n, v in c["attrs"]) + "}"
    return sig


def merge(instances: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for c in instances:
        key = case_key(c)
        m = merged.get(key)
        if m is None:
            m = merged[key] = {k: v for k, v in c.items() if k not in ("model", "est_us")}
            m.update(key=key, sig=signature(c), instances={}, est_us=c["est_us"])
        m["instances"][c["model"]] = m["instances"].get(c["model"], 0) + 1
        m["est_us"] = min(m["est_us"], c["est_us"])
    return sorted(merged.values(), key=lambda c: (c["device"], c["kernel"], c["sig"]))


def _kokoro_basic_cpu():
    """The M1 basic Kokoro CPU graph (fused=False): the parity reference, and the only user of several basic kernels."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from models.kokoro import model as K
    from models.kokoro import reference as R

    return K.build(K.fold_weights(R.load_model()), fused=False)


# Extra builds beyond each target's default (key, target, case, device, build).
EXTRA_BUILDS = [("kokoro_basic", "kokoro", "app59f", "cpu", _kokoro_basic_cpu)]


def extract(models=MODELS, devices=("cpu", "gpu"), log=print) -> list[dict]:
    from lokinfer.prof.targets import target_for

    builds = [(key, key, case, dev, None) for key, case in models for dev in devices]
    builds += [b for b in EXTRA_BUILDS if b[3] in devices]
    instances = []
    for key, tkey, case, dev, make in builds:
        tgt = target_for(tkey)
        model = make() if make else tgt.build(dev)
        args = tgt.inputs(case)
        for _ in range(2):
            model(*args)
        exe = model._exe
        exe.take_profile()
        exe.set_profile(1, False)
        model(*args)
        exe.set_profile(0)
        raw = json.loads(exe.take_profile())
        got = cases_from_profile(key, dev, model.ir_text(), raw)
        log(f"  {key} [{case}] {dev}: {len(got)} kernel instances")
        instances += got
    return merge(instances)


def save(cases: list[dict], path: Path = CASES_JSON) -> None:
    models = [list(m) for m in MODELS] + [[b[0], b[2]] for b in EXTRA_BUILDS]
    path.write_text(json.dumps({"models": models, "cases": cases}, indent=1) + "\n")


def load(path: Path = CASES_JSON) -> list[dict]:
    return json.loads(path.read_text())["cases"]


# ------------------------------------------------------------------------------------------- stand-alone models

def _attr_text(attrs) -> str:
    return " {" + ", ".join(f"{n} = {v}" for n, v in attrs) + "}" if attrs else ""


def case_ir(c: dict, repeats: int = 1) -> str:
    nin = c["nin"]
    params = ", ".join(f"%a{i}: {t}" for i, t in enumerate(c["args"][:nin]))
    outs = c["args"][nin:]
    body = [f"    %o{j} = buffer.alloc {{output = true}} : {t}" for j, t in enumerate(outs)]
    attrs = list(c["attrs"])
    if c.get("inout"):
        attrs.append(("inout", "[" + ", ".join(str(x) for x in c["inout"]) + "]"))
    call = (f'    kernel "{c["kernel"]}"(' + ", ".join(f"%a{i}" for i in range(nin)) + ")"
            + (" outs(" + ", ".join(f"%o{j}" for j in range(len(outs))) + ")" if outs else "") + _attr_text(attrs))
    body += [call] * max(1, repeats)
    ret = "    return " + ", ".join(f"%o{j}" for j in range(len(outs))) if outs else "    return"
    return "model {\n  init {\n    return\n  }\n  exec(" + params + ") {\n" + "\n".join(body) + "\n" + ret + "\n  }\n}\n"


_NP = {"f32": np.float32, "f16": np.float16, "i64": np.int64, "i32": np.int32, "u8": np.uint8, "bool": np.bool_}


def _parse_type(t: str) -> tuple[str, list[int] | None]:
    m = re.fullmatch(r"buffer<(\w+)\[([^\]]*)\]>", t)
    if not m:
        return t, None  # scalar
    return m.group(1), [int(x) for x in m.group(2).split(",") if x.strip()]


# Kernels whose int inputs must hold valid values (ids in range, durations summing to the frame count, ...).
def case_inputs(c: dict, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for t in c["args"][:c["nin"]]:
        dt, shape = _parse_type(t)
        npdt = _NP.get(dt, np.float32)
        if shape is None:  # scalar operand: a 0-d array
            out.append(np.array(1 if dt != "f32" else 1.0, npdt))
            continue
        if np.issubdtype(npdt, np.floating):
            out.append(rng.uniform(-1, 1, shape).astype(npdt))
        else:
            out.append(np.zeros(shape, npdt))
    return _fix_inputs(c, out)


def _fix_inputs(c: dict, xs: list[np.ndarray]) -> list[np.ndarray]:
    """Valid data for data-dependent kernels (otherwise their checks reject the call or the work changes)."""
    k = c["kernel"]
    attrs = dict(c["attrs"])
    if k.startswith("repeat_interleave"):
        # durations (the i64 operand) must sum to the `frames` attribute
        f = int(attrs["frames"])
        for i, x in enumerate(xs):
            if x.dtype == np.int64 and x.ndim == 1:
                d = np.full(x.shape, f // max(1, x.size), np.int64)
                d[: f - int(d.sum())] += 1
                xs[i] = d
    if k.startswith(("attention_decode", "kv_write")):
        # decode position (i64[1]): the model's case, "decode P=512" (rows 0..P attended / row P written)
        for i, x in enumerate(xs):
            if x.dtype == np.int64 and x.shape == (1,):
                xs[i] = np.array([512], np.int64)
    if k.startswith("embedding"):
        rows = _parse_type(c["args"][0])[1][0]
        for i, x in enumerate(xs):
            if x.dtype == np.int64:
                xs[i] = np.random.default_rng(1).integers(0, rows, x.shape).astype(np.int64)
    if "wino" in k and "kernel" in attrs:
        # the Winograd transform table operand tr f32[A*A + m*A] must be the real table (CPU kernels check it)
        if str(ROOT / "python") not in sys.path:
            sys.path.insert(0, str(ROOT / "python"))
        from lokinfer.ops.winograd import wino_weights

        kk = int(attrs["kernel"])
        tables = []
        for m in (4, 2):
            try:
                tables.append(wino_weights(np.zeros((1, 1, kk), np.float32), tile=m)[1])
            except Exception:  # noqa: BLE001 - tile 2 only exists for K = 3
                pass
        for i, x in enumerate(xs):
            for tr in tables:
                if x.dtype == np.float32 and x.ndim == 1 and x.shape == tr.shape:
                    xs[i] = tr.copy()
    return xs


def repeats_for(c: dict, target_us: float = 2000.0, cap: int = 64) -> int:
    """Kernels repeated per call so one call lasts >= ~2 ms (amortizes the per-call runtime / GPU submit cost)."""
    est = max(c.get("est_us", 0.0), 1.0)
    return int(min(cap, max(1, round(target_us / est))))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "python"))
    cs = extract()
    save(cs)
    print(f"{len(cs)} cases -> {CASES_JSON}")
