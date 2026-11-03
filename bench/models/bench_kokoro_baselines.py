"""Apples-to-apples Kokoro-82M baselines on this Mac: torch.compile (CPU / MPS) and MLX (GPU), on exactly the
reference cases lokinfer is measured on (same ids, voice row, speed; Kokoro v1.0 weights, f32). Each run's audio
length must equal the reference audio's (600·F samples) — i.e. the same durations, so the same work. A framework
that computes its own durations may land a frame off (MLX: rand510 F = 1100 vs 1101): within 0.2 % the entry is
recorded with the exact F difference in its notes; beyond that it is NOT recorded (not comparable).

Usage: .venv/bin/python bench/models/bench_kokoro_baselines.py [--compile] [--mlx] [case ...]
(no case = every Kokoro case of the dashboard, perfdb.DEFAULT_META). Timing protocol: ~1 s warm-up, then >= 2 s and
>= 7 runs (torch; bench_kokoro.timeit) or >= 30 runs / >= 2 s (MLX runner), median.
MLX needs a Python with mlx + mlx-audio and the MLX weights dir (fp32 safetensors converted from the hexgrad pth):
mlx_python (lokinfer.local.json / $LOKINFER_MLX_PYTHON) and $LOKINFER_MLX_MODEL (default <work_dir>/mlx_f32; see
docs/baselines.md §3.4).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perfdb  # noqa: E402
from bench_kokoro import TIMING_NOTE, length_check, timeit, torch_infer  # noqa: E402
from models.kokoro import reference as R  # noqa: E402

from lokinfer import localconfig  # noqa: E402

CASES = perfdb.DEFAULT_META["models"]["Kokoro-82M"]["cases"]


def compile_rows(cases, state):
    from kokoro.model import KModel

    snap = R.snapshot_dir()
    for dev, device, impl in (("cpu", "Mac CPU", "torch.compile"), ("mps", "Mac GPU", "torch.compile MPS")):
        m = KModel(config=str(snap / "config.json"), model=str(snap / "kokoro-v1_0.pth")).to(dev).eval()
        m.decoder = torch.compile(m.decoder, dynamic=True)  # as in research-kokoro.md: decoder + bert, dynamic
        m.bert = torch.compile(m.bert, dynamic=True)
        for name in cases:
            c = R.load_case(name)
            run = torch_infer(m, c, dev)
            ok, length = length_check(run().numel(), c["audio"].size, impl)  # the first call also compiles
            if not ok:
                print(f"{impl} {name}: NOT COMPARABLE ({length})")
                continue
            t = timeit(run, warmup=3, runs=7)["median_ms"]
            print(f"{impl:>18} {name:>8}: {t:7.1f} ms  ({length})", flush=True)
            perfdb.record("Kokoro-82M", device, name, impl, t, state=state,
                          notes=f"torch {torch.__version__}, torch.compile(decoder, bert, dynamic=True); same inputs "
                                f"as the reference case; {length}; {TIMING_NOTE}")


def mlx_rows(cases, state):
    py = localconfig.require("mlx_python")
    mdir = os.environ.get("LOKINFER_MLX_MODEL", os.path.join(localconfig.get("work_dir"), "mlx_f32"))
    if not Path(py).exists() or not (Path(mdir) / "config.json").exists():
        print(f"MLX: no Python at {py} or no MLX weights in {mdir} (set LOKINFER_MLX_PYTHON / LOKINFER_MLX_MODEL)")
        return
    refs = {name: R.load_case(name) for name in cases}
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "cases.npz"
        arrs = {}
        for name, c in refs.items():
            arrs["ids_" + name] = np.concatenate([[0], c["ids"], [0]]).astype(np.int64)
            arrs["style_" + name] = np.asarray(c["ref_s"], np.float32)
            arrs["speed_" + name] = np.asarray(c["speed"], np.float32).reshape(())
        np.savez(npz, **arrs)
        runner = Path(__file__).resolve().parent / "runners" / "kokoro_mlx.py"
        out = subprocess.run([py, "-I", str(runner), mdir, str(npz), "7", "2"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        r = json.loads(line)
        if "case" not in r:
            print("MLX:", r)
            ver = r.get("mlx")
            continue
        c = refs[r["case"]]
        ref_n = c["audio"].size
        if abs(r["samples"] - ref_n) > 0.002 * ref_n:
            print(f"MLX {r['case']}: NOT COMPARABLE ({r['samples']} samples vs reference {ref_n}: different durations)")
            continue
        length = ("output length matches the reference" if r["samples"] == ref_n else
                  f"MLX's own durations give F = {r['samples'] // 600} vs reference {ref_n // 600} "
                  f"({100 * (r['samples'] - ref_n) / ref_n:+.2f} % work)")
        print(f"{'MLX':>18} {r['case']:>8}: {r['median_ms']:7.1f} ms  ({length}; f32 weights: {r['f32']})")
        perfdb.record("Kokoro-82M", "Mac GPU", r["case"], "MLX", r["median_ms"], state=state,
                      notes=f"mlx {ver} + mlx-audio, fp32 weights from the hexgrad pth (f32 check: {r['f32']}); "
                            f"same inputs as the reference case; {length}; median of {r['runs']} runs after "
                            f"{r['warmup']} warm-up calls (>= 1 s)")
    if out.returncode:
        print(out.stderr[-2000:])


def main():
    cases = [a for a in sys.argv[1:] if not a.startswith("--")] or CASES
    state = perfdb.machine_state()
    print("machine:", state)
    if "--mlx" in sys.argv:
        mlx_rows(cases, state)
    if "--compile" in sys.argv:
        compile_rows(cases, state)


if __name__ == "__main__":
    main()
