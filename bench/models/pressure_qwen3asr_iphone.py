"""Qwen3-ASR-1.7B decode under memory pressure on the iPhone: do the mapped weights get evicted (and re-read from flash
every step: docs/todo.md "fc-thrashing"), and does PIN_MODEL (mlock) prevent it?

One session: the generate graph (--device gpu / cpu), a KV cache filled with a clip's prompt (zero audio rows: the
values don't matter here), then decode steps timed (RUN2, --steps per point) while a "balloon" of incompressible
device tensors (random bytes, --step-gb each, up to --max-gb; the server caps device tensors at 4 GiB) squeezes the
phone's free memory. Per point: step median, page-ins during the steps (a weight re-read from flash is one), the
weights resident (mincore), os_proc_available_memory, the pressure level, the system's free / file-backed /
compressor memory. --pin lock (mlock) / gpu (GPU residency) wires the weights first. The balloon goes at the end (FREE_TENSOR).

It stops growing the balloon when: the pressure level is critical, the step time is > 5x the first point, or the
app would come within --min-available-gb of its jetsam limit (so the measurement doesn't kill LokiHost).
Background apps may be jetsammed by iOS meanwhile (that is the point: their memory, not our weights).

Usage: scripts/lk-lock /tmp/lokinfer-phone.lock scripts/lk-lock /tmp/lokinfer-bigmodel.lock \\
           .venv/bin/python bench/models/pressure_qwen3asr_iphone.py --device cpu --pin none gpu lock
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "bench", ROOT / "bench" / "models", ROOT / "python"):
    sys.path.insert(0, str(p))
from bench_qwen3asr import Clip  # noqa: E402
from lokinfer import localconfig  # noqa: E402
from lokinfer.remote import RemoteDevice  # noqa: E402
from models.qwen3_asr import model as M  # noqa: E402

GB = 1 << 30


def point(dev, mid, kv, pos, steps, label):
    m0 = dev.memory()
    tok, dummy = np.array([872], np.int64), np.zeros((1, M.LM_D), np.float16)
    t0 = time.time()
    _, _, t = dev.run2(mid, [tok, dummy, np.array([pos], np.int64), kv], return_outputs=False, iters=steps)
    wall = time.time() - t0
    m = dev.memory()
    ms = sorted(x * 1e-6 for x in t)
    r = {"label": label, "step_ms": ms[len(ms) // 2], "step_max_ms": ms[-1], "wall_s": wall,
         "pageins_gb": (m["pageins"] - m0["pageins"]) * m["page_size"] / GB,
         "weights_resident_gb": m["weights_resident"] / GB, "weights_gb": m["weights"] / GB,
         "available_gb": m["available"] / GB, "pressure": m["pressure"], "events": m["pressure_events"],
         "footprint_gb": m["phys_footprint"] / GB, "free_gb": m.get("system_free", 0) / GB,
         "file_gb": m.get("system_file_backed", 0) / GB, "compressor_gb": m.get("system_compressor", 0) / GB,
         "wired_gb": m.get("system_wired", 0) / GB}
    print("   {label:>14}: step {step_ms:7.1f} ms (max {step_max_ms:7.1f}), page-ins {pageins_gb:6.2f} GB, weights "
          "resident {weights_resident_gb:.2f}/{weights_gb:.2f}, available {available_gb:.2f}, footprint "
          "{footprint_gb:.2f}, pressure {pressure} ({events}), sys free {free_gb:.2f} file {file_gb:.2f} "
          "compressor {compressor_gb:.2f} wired {wired_gb:.2f}".format(**r), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--pin", nargs="+", default=["none", "gpu", "lock"], choices=["none", "lock", "gpu", "willneed"])
    ap.add_argument("--clip", default="s30")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step-gb", type=float, default=0.5)
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--min-available-gb", type=float, default=0.6)
    a = ap.parse_args()
    udid = localconfig.require("iphone_udid")
    dev = RemoteDevice.connect(f"{udid}.coredevice.local", timeout=900.0)
    st = dev.state()
    print("device:", st, flush=True)
    assert dev.supports("pin_model"), "deploy first (feature pin_model)"
    sd = M.load_weights_opt()
    mid = dev.load(M.build_generate(sd, a.device, variant="opt"))
    c = Clip(a.clip)
    cap = -(-(len(c.ids) + c.max_new) // 1024) * 1024
    kv = dev.alloc_tensor(np.float16, (M.LM_LAYERS, 2, cap, M.LM_KVH * M.LM_HD))
    for s in range(0, len(c.ids), 2048):
        e = min(len(c.ids), s + 2048)
        dev.run2(mid, [c.ids[s:e], np.zeros((e - s, M.LM_D), np.float16), np.array([s], np.int64), kv],
                 return_outputs=False)
    pos = len(c.ids)
    block = np.random.default_rng(0).integers(0, 1 << 16, size=int(a.step_gb * GB) // 2, dtype=np.uint16)
    block = block.view(np.float16)
    try:
        for pin in a.pin:
            dev.pin_model(mid, "unlock")
            print(f"== {a.device}, pin {pin}: {dev.pin_model(mid, pin if pin != 'none' else 'unlock')}", flush=True)
            first = point(dev, mid, kv, pos, a.steps, "no balloon")
            balloon = []
            try:
                while (len(balloon) + 1) * a.step_gb <= a.max_gb:
                    m = dev.memory()
                    if m["available"] / GB - a.step_gb < a.min_available_gb:
                        print(f"   stop: available {m['available'] / GB:.2f} GB", flush=True)
                        break
                    try:
                        t = dev.alloc_tensor(np.float16, block.shape)
                    except Exception as e:  # the server's 4 GiB device-tensor cap (the KV cache counts)
                        print(f"   stop: {e}", flush=True)
                        break
                    balloon.append(t)
                    dev.write_tensor(t, block)
                    r = point(dev, mid, kv, pos, a.steps, f"balloon {len(balloon) * a.step_gb:.1f} GB")
                    if r["pressure"] == "critical" or r["step_ms"] > 5 * first["step_ms"]:
                        print("   stop: pressure critical / steps 5x slower", flush=True)
                        break
            finally:
                for t in balloon:
                    dev.free_tensor(t)
            point(dev, mid, kv, pos, a.steps, "balloon freed")
    finally:
        dev.pin_model(mid, "unlock")
        dev.free_tensor(kv)
        dev.free_model(mid)
        dev.close()


if __name__ == "__main__":
    main()
