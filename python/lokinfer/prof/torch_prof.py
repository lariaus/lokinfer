"""prof-torch: the PyTorch reference of a registry model (same inputs as lokinfer's run), profiled into the same run
format, so mlogs-viewer can put both side by side.

Two passes:
  1. Per-op timing with a TorchDispatchMode (every aten op after autograd, i.e. the leaf ops). On MPS each op is
     followed by torch.mps.synchronize(), so its row is real GPU time (serialized, like prof-lokinfer --serialize);
     on CPU nothing is added. This is what the tables and the timeline show.
  2. torch.profiler (CPU activity, record_shapes) over the same calls: Chrome trace (torch_trace.json, open it in
     https://ui.perfetto.dev) and key_averages() as text (torch_key_averages.txt), unchanged from torch.
The e2e number is measured separately, without any instrumentation, with the dashboard protocol.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

from . import analysis
from .collect import host_info, mac_state, new_run_dir, timed
from .targets import target_for

TORCH_DEVICES = {"cpu": "Mac CPU", "mps": "Mac GPU"}


def _sig(args) -> str:
    import torch

    parts = []

    def walk(a):
        if isinstance(a, torch.Tensor):
            dt = str(a.dtype).replace("torch.", "").replace("float", "f").replace("int", "i")
            parts.append(f"{dt}[{','.join(str(d) for d in a.shape)}]")
        elif isinstance(a, (list, tuple)):
            for x in a:
                walk(x)

    walk(args)
    return " ".join(parts[:8])


# nn.Module class -> category: an op inside an nn.LSTM is "lstm" work even when torch runs it as mul / sigmoid / linear
# (the MPS LSTM decomposes per time step), which is what the lokinfer side's fused lstm kernel does.
MODULE_CATEGORIES = [(r"LSTM|GRU|RNN", "lstm"), (r"ConvTranspose\dd", "conv_transpose"), (r"Conv\dd", "conv"),
                     (r"Linear|LinearNorm", "matmul"),
                     (r"AdaIN1d|AdaLayerNorm|(Instance|Layer|Batch|Group)Norm\w*", "norm"),
                     (r"Embedding", "embedding"), (r"TorchSTFT|SineGen|SourceModule\w*", "signal")]


def module_category(stack: list[str]) -> str | None:
    """Category of the innermost *leaf-like* module on the stack (composite blocks such as AdaINResBlock1 don't
    count: their own forward code, e.g. the Snake activation, keeps the op-name category)."""
    import re

    for cls in reversed(stack):
        for pat, cat in MODULE_CATEGORIES:
            if re.fullmatch(pat, cls):
                return cat
    return None


def per_op_events(fn, device: str, ncalls: int) -> list[dict]:
    import torch
    from torch.nn.modules import module as nn_module
    from torch.utils._python_dispatch import TorchDispatchMode

    sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: None)
    events: list[dict] = []
    state = {"call": 0, "start": 0}
    stack: list[str] = []
    def push(m, i):
        stack.append(type(m).__name__)

    def pop(m, i, o):  # must return None (a hook's return value replaces the module output)
        if stack:
            stack.pop()

    pre = nn_module.register_module_forward_pre_hook(push)
    post = nn_module.register_module_forward_hook(pop)

    class Timer(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            t0 = time.perf_counter_ns()
            out = func(*args, **(kwargs or {}))
            sync()
            t1 = time.perf_counter_ns()
            name = "aten::" + func.__name__.split(".")[0] if hasattr(func, "__name__") else str(func)
            ev = {"call": state["call"], "name": name, "device": "gpu" if device == "mps" else "cpu",
                  "lane": "MPS (synced per op)" if device == "mps" else "CPU",
                  "t0": t0 - state["start"], "t1": t1 - state["start"], "shapes": _sig(args),
                  "module": stack[-1] if stack else ""}
            cat = module_category(stack)
            if cat:
                ev["category"] = cat
            events.append(ev)
            return out

    try:
        with torch.no_grad():
            for i in range(ncalls):
                state["call"] = i
                stack.clear()
                sync()
                state["start"] = time.perf_counter_ns()
                with Timer():
                    fn()
                sync()
    finally:
        pre.remove()
        post.remove()
    return events


def torch_profiler_artifacts(fn, device: str, ncalls: int, run_dir: Path) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for i in range(ncalls):
            with record_function(f"call{i}"):
                fn()
            if device == "mps":
                torch.mps.synchronize()
    prof.export_chrome_trace(str(run_dir / "torch_trace.json"))
    (run_dir / "torch_key_averages.txt").write_text(
        prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=80))


def profile_torch(model_name: str, device: str = "cpu", case: str | None = None, *, iters: int = 3, runs: int = 10,
                  min_time: float = 2.0, warm_time: float = 1.0, out_base: Path | None = None, report: bool = True,
                  log=print) -> Path:
    import torch

    if device not in TORCH_DEVICES:
        raise SystemExit(f"prof-torch devices: {', '.join(TORCH_DEVICES)} (the iPhone has no PyTorch)")
    tgt = target_for(model_name)
    if tgt.torch is None:
        raise SystemExit(f"no PyTorch reference for {model_name}")
    case = case or tgt.default_case
    log(f"[prof-torch] {tgt.title} case={case} on torch {device} ...")
    make = tgt.torch(device)
    fn = make(case)
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: None)

    def call():
        with torch.no_grad():
            out = fn()
        sync()
        return out

    e2e = timed(call, warm_time, 3, runs, min_time)
    events = per_op_events(fn, device, iters)
    run_dir = new_run_dir("torch", tgt.name, "mac-" + ("gpu" if device == "mps" else "cpu"), case, out_base)
    try:
        torch_profiler_artifacts(fn, device, min(iters, 2), run_dir)
    except Exception as e:  # noqa: BLE001 - the extra trace is optional
        log(f"  torch.profiler pass failed: {e}")
    meta = {"id": run_dir.name, "model": tgt.title, "model_key": tgt.name, "case": case,
            "device": TORCH_DEVICES[device], "target": "mac", "backend": device,
            "created": _dt.datetime.now().isoformat(timespec="seconds"), "machine_state": mac_state(),
            "host": {"device": TORCH_DEVICES[device], "torch": torch.__version__, **host_info()},
            "settings": {"iters": iters, "runs": runs, "per_op_sync": device == "mps"},
            "files": {k: v for k, v in (("chrome_trace", "torch_trace.json"),
                                         ("key_averages", "torch_key_averages.txt")) if (run_dir / v).exists()}}
    run = analysis.build_torch_run(events, meta=meta, e2e_ms=e2e, ncalls=iters)
    run["notes"] = ["Per-op rows come from a TorchDispatchMode timer (leaf aten ops)"
                    + ("; each MPS op is followed by torch.mps.synchronize(), so rows are serialized GPU times and "
                       "their sum exceeds the e2e time (no overlap)." if device == "mps" else ".")]
    (run_dir / "run.json").write_text(json.dumps(run, indent=1))
    if report:
        from .report import write_report

        write_report(run_dir)
    log(f"  e2e median {run['summary']['e2e_ms']['median']:.3f} ms")
    for k in run["kernels"][:12]:
        log(f"  {k['total_ms']:9.3f} ms {k['pct']:5.1f}%  x{k['calls']:<6g} {k['name']}")
    log(f"  -> {run_dir}")
    return run_dir
