"""Qwen3-ASR-1.7B: the remote executor's memory per case (docs/models/qwen3-asr.md "Memory").

Each (graph, kind) runs in a FRESH executor process (LokiHost restarted through SHUTDOWN + the deploy watchdog, or a
local Release lokinfer-serve with --where mac), cases in increasing clip length, so the process peaks belong to that
kind alone (the runtime's activation arenas are grow-only per executable: a case's footprint includes the arenas of
the longest clip so far, which is the clip itself). Per case, after its calls (the KV cache still allocated):
footprint = phys_footprint (what iOS's per-app limit counts; the KV cache and the activation arenas are in it, the
mapped weights are not), its peak, resident (incl. the mapped weights once touched), the GPU's allocated size (the
no-copy weight wraps included), the weights resident (mincore), os_proc_available_memory, the pressure level and the
case's page-ins. kinds: enc (encoder graph only), prefill (generate graph only: the prompt in <= 2048-id chunks),
decode (generate: the clip's whole context filled, then 3 steps), e2e (both graphs: encoder + prefill + greedy
decode). The audio rows prefill / decode need come from the enc session's outputs (golden-checked by the e2e
transcripts).

Usage (iPhone: under the phone lock and the big-model lock; keep a session short, e.g. one device at a time):
  scripts/lk-lock /tmp/lokinfer-phone.lock scripts/lk-lock /tmp/lokinfer-bigmodel.lock \\
      .venv/bin/python bench/models/measure_qwen3asr_memory.py --device gpu [--kind enc e2e] [--clip 2s 300s]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
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

CLIPS = ["2s", "5s", "10s", "30s", "60s", "111s", "300s"]
KINDS = ["enc", "prefill", "decode", "e2e"]
CHUNK = 2048
SERVE = Path(os.environ.get("LOKINFER_SERVE", ROOT / "build-release" / "lokinfer-serve"))
GB = 1 << 30


def fresh(where: str, blobs: str | None, dev=None, proc=None):
    """Shuts the current executor down (if any) and connects to a new process."""
    if dev is not None:
        try:
            dev.shutdown()
        except Exception:
            pass
        dev.close()
        if proc is not None:
            proc.wait(timeout=60)
        time.sleep(2)
    if where == "mac":
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        proc = subprocess.Popen([str(SERVE), "--port", str(port), "--blobs", blobs], stdout=subprocess.DEVNULL)
        return RemoteDevice.connect("127.0.0.1", port, timeout=900.0), proc
    udid = localconfig.require("iphone_udid")
    return RemoteDevice.connect(f"{udid}.coredevice.local", timeout=900.0, retries=180), None


def rows(audio, s, e, n_audio):
    k0, k1 = min(max(s - M.AUDIO_START, 0), n_audio), min(max(e - M.AUDIO_START, 0), n_audio)
    return np.ascontiguousarray(audio[k0:k1]) if k1 > k0 else np.zeros((1, M.LM_D), np.float16)


def cap_for(c: Clip, where: str) -> int:
    """The bench's KV capacity: prompt + max_new rounded up to 1024, at most the device's MAX_CAP."""
    return min(M.MAX_CAP["iphone" if where == "iphone" else "mac"],
               -(-(len(c.ids) + max(c.max_new, len(c.tokens))) // 1024) * 1024)


def run_case(dev, kind, c, mids, audio, where, snap):
    """Runs one case's calls once; returns (device ms, note). snap[0] = MEMORY after the calls, the KV cache still
    allocated."""
    mid_e, mid_g = mids
    enc_in = [c.features, np.array([c.n_audio], np.int64)]
    if kind == "enc":
        (a,), _, t = dev.run2(mid_e, enc_in)
        audio[c.name] = a
        snap.append(dev.memory())
        return sum(t) * 1e-6, ""
    kv = dev.alloc_tensor(np.float16, (M.LM_LAYERS, 2, cap_for(c, where), M.LM_KVH * M.LM_HD))
    try:
        dummy = np.zeros((1, M.LM_D), np.float16)
        total = 0.0
        if kind == "e2e":
            (a,), _, t = dev.run2(mid_e, enc_in)
            total += sum(t) * 1e-6
        else:
            a = audio[c.name]
        seq = c.ids if kind in ("prefill", "e2e") else np.concatenate([c.ids, np.array(c.tokens[:-1], np.int64)])
        n_fill = len(seq) if kind != "decode" else len(seq) - 1
        lg = None
        for s in range(0, n_fill, CHUNK):
            e = min(n_fill, s + CHUNK)
            (lg,), _, t = dev.run2(mid_g, [np.ascontiguousarray(seq[s:e]), rows(a, s, e, c.n_audio),
                                           np.array([s], np.int64), kv])
            total += sum(t) * 1e-6
        if kind == "decode":
            _, _, t = dev.run2(mid_g, [seq[-1:], dummy, np.array([len(seq) - 1], np.int64), kv],
                               return_outputs=False, iters=3)
            return total + sum(t) * 1e-6, "fill + 3 steps"
        if kind == "e2e":
            toks, pos = [], len(c.ids)
            while True:
                tok = int(np.argmax(lg))
                toks.append(tok)
                if tok in M.EOS or len(toks) >= c.max_new:
                    break
                (lg,), _, t = dev.run2(mid_g, [np.array([tok], np.int64), dummy, np.array([pos], np.int64), kv])
                total += sum(t) * 1e-6
                pos += 1
            return total, f"golden {toks == c.tokens}, {len(toks)} tokens"
        return total, ""
    finally:
        snap.append(dev.memory())
        dev.free_tensor(kv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--where", default="iphone", choices=["iphone", "mac"])
    ap.add_argument("--device", nargs="+", default=["gpu", "cpu"], choices=["gpu", "cpu"])
    ap.add_argument("--kind", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--clip", nargs="+", default=CLIPS, choices=CLIPS)
    ap.add_argument("--pin", default="none", choices=["none", "lock", "gpu", "willneed"],
                    help="PIN_MODEL the loaded graphs' weights first (feature pin_model)")
    ap.add_argument("--json", type=Path, help="append one JSON line per case here")
    ap.add_argument("--blobs", default=str(Path.home() / ".cache" / "lokinfer" / "serve-blobs"),
                    help="--where mac: the local server's blob store (weights go up once)")
    a = ap.parse_args()
    if "prefill" in a.kind or "decode" in a.kind:
        assert "enc" in a.kind, "prefill / decode use the enc session's audio rows: include --kind enc"
    Path(a.blobs).mkdir(parents=True, exist_ok=True)
    sd = M.load_weights_opt()
    clips = {n: Clip("s" + n.rstrip("s")) for n in a.clip}
    dev, proc = fresh(a.where, a.blobs)
    print("device:", dev.state(), flush=True)
    assert dev.supports("memory_pressure"), "the executor predates the memory_pressure feature: deploy first"
    table = []
    try:
        for d in a.device:
            audio: dict[str, np.ndarray] = {}
            for kind in [k for k in KINDS if k in a.kind]:
                dev, proc = fresh(a.where, a.blobs, dev, proc)
                base = dev.memory()
                t0 = time.time()
                mid_e = dev.load(M.build_encoder(sd, d, variant="opt")) if kind in ("enc", "e2e") else None
                mid_g = dev.load(M.build_generate(sd, d, variant="opt")) if kind != "enc" else None
                for mid in (mid_e, mid_g):
                    if mid is not None and a.pin != "none":
                        print("   pin:", dev.pin_model(mid, a.pin), flush=True)
                loaded = dev.memory()
                print(f"== {d} {kind}: fresh process, loaded in {time.time() - t0:.0f} s; base footprint "
                      f"{base['phys_footprint'] / GB:.3f} GB, after load {loaded['phys_footprint'] / GB:.3f} GB",
                      flush=True)
                for n in a.clip:
                    if kind == "e2e" and n == "2s":
                        continue  # not a dashboard case
                    c = clips[n]
                    m0 = dev.memory()
                    snap = []
                    ms, note = run_case(dev, kind, c, (mid_e, mid_g), audio, a.where, snap)
                    m = snap[0]
                    m["kv_bytes"] = (M.LM_LAYERS * 2 * cap_for(c, a.where) * M.LM_KVH * M.LM_HD * 2
                                     if kind != "enc" else 0)
                    r = {"where": a.where, "device": d, "kind": kind, "clip": n, "pin": a.pin, "device_ms": ms,
                         "note": note, "base": base, "loaded": loaded, "after": m,
                         "pageins_case": (m["pageins"] - m0["pageins"]) * m["page_size"]}
                    table.append(r)
                    print(f"   {kind} {n}: footprint {(m['phys_footprint']) / GB:.3f} GB (peak "
                          f"{m['phys_footprint_peak'] / GB:.3f}; KV {m['kv_bytes'] / GB:.3f}), resident "
                          f"{m['resident'] / GB:.3f} (peak {m['resident_peak'] / GB:.3f}), GPU "
                          f"{m['gpu_allocated'] / GB:.3f}, weights resident {m['weights_resident'] / GB:.2f} / "
                          f"{m['weights'] / GB:.2f}, available {m['available'] / GB:.2f}, pressure {m['pressure']} "
                          f"({m['pressure_events']} events), page-ins {r['pageins_case'] / GB:.3f} GB, "
                          f"{ms:.0f} ms device {note}", flush=True)
                    if a.json:
                        with a.json.open("a") as f:
                            f.write(json.dumps(r) + "\n")
    finally:
        try:
            dev.close()
        finally:
            if proc is not None:
                proc.terminate()
    print("\n| device | kind | clip | footprint (KV) | peak footprint | peak resident | GPU allocated | page-ins |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in table:
        m = r["after"]
        print(f"| {r['device']} | {r['kind']} | {r['clip']} | {m['phys_footprint'] / GB:.2f} ({m['kv_bytes'] / GB:.2f}) "
              f"| {m['phys_footprint_peak'] / GB:.2f} | {m['resident_peak'] / GB:.2f} | {m['gpu_allocated'] / GB:.2f} "
              f"| {r['pageins_case'] / GB:.2f} |")


if __name__ == "__main__":
    main()
