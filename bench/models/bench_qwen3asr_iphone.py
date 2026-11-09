"""Qwen3-ASR-1.7B on the iPhone (remote executor in LokiHost, scripts/ios_deploy.sh), CPU and GPU optimized graphs,
recorded to the dashboard as "iPhone CPU" / "iPhone GPU" (impl "lokinfer fused"). Same cases and golden inputs as
bench_qwen3asr.py, up to one 5 min pass (the iPhone's KV cap): enc / prefill / decode 2 s ... 300 s, e2e 5 s ... 300 s.

Times are measured on the device (RUN2's per-iteration times; the KV cache is a device tensor that stays on the
phone). prefill = the sum of its <= 2048-id chunks; decode = one step at the clip's final context (cache filled
untimed); e2e = encoder + prefill + every greedy step, the sum of their on-device times (model time only, as the
dashboard defines it: the Mac-side greedy loop and the transfers are not counted), median of 3 passes.
Memory pressure (docs/todo.md "fc-thrashing"): a case that starts or ends with the phone under memory pressure, or
pages in more than 1.5x the weights' size meanwhile (weights re-read from flash), is printed with a WARNING and never
recorded (like Low Power Mode: those timings are flash reads, not the model).
Timing protocol (decision 11): warm-up >= 2 calls and >= 1 s (1 call when a call takes > 10 s), then >= 30 runs or
>= 2 s, at least 7 runs (3 when a call takes > 10 s); median. Weights go up once (split transport, cached on the
phone across sessions and graphs).

Usage (under the phone lock and the big-model lock):
  scripts/lk-lock /tmp/lokinfer-phone.lock scripts/lk-lock /tmp/lokinfer-bigmodel.lock \\
      .venv/bin/python bench/models/bench_qwen3asr_iphone.py [--device gpu cpu] [--case "enc 10s" ...] [--record]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "bench" / "models"))
sys.path.insert(0, str(ROOT / "python"))
import perfdb  # noqa: E402
from bench_qwen3asr import Clip  # noqa: E402
from bench_remote_models import wait_nominal  # noqa: E402
from lokinfer import localconfig  # noqa: E402
from lokinfer.remote import RemoteDevice  # noqa: E402
from models.qwen3_asr import model as M  # noqa: E402

MODEL = "Qwen3-ASR-1.7B"
DEVICE_NAME = {"cpu": "iPhone CPU", "gpu": "iPhone GPU"}
IMPL = {"opt": "lokinfer fused", "basic": "lokinfer basic"}
CASES = [c for c in perfdb.DEFAULT_META["models"][MODEL]["cases"]
         if c.split()[1] in ("2s", "5s", "10s", "30s", "60s", "111s", "300s")]
CHUNK = 2048


def med(times_ns):
    t = sorted(x * 1e-6 for x in times_ns)
    return t[len(t) // 2]


QUICK = False


def timed(dev, mid, inputs):
    """Decision-11 protocol on the device (or --quick: 1 warm-up + 5 runs); returns (median ms, runs)."""
    _, _, t = dev.run2(mid, inputs, return_outputs=False, warmup=0, iters=1)
    est = max(med(t), 0.01)
    slow = est > 10_000
    if QUICK:
        _, _, times = dev.run2(mid, inputs, return_outputs=False, warmup=0, iters=3 if slow else 5)
        return med(times), len(times)
    warm = 1 if slow else int(min(5000, max(2, math.ceil(1000 / est))))
    iters = 3 if slow else int(min(5000, max(7, min(30, math.ceil(2000 / est)))))
    _, _, times = dev.run2(mid, inputs, return_outputs=False, warmup=warm, iters=iters)
    return med(times), len(times)


def cool_down(dev, target: str, max_wait_s: float) -> tuple[str, float]:
    """Waits (idle) until the phone's thermal state is at most `target` ("nominal" / "fair"), at most max_wait_s;
    returns (state reached, seconds waited). Per case, not per device (2026-11-06: one wait per device let the long
    cases run at "serious" after the first minutes)."""
    order = ["nominal", "fair", "serious", "critical"]
    t0 = time.time()
    th = dev.state().get("thermal")
    while th in order and order.index(th) > order.index(target) and time.time() - t0 < max_wait_s:
        time.sleep(15)
        th = dev.state().get("thermal")
    return th, time.time() - t0


def rows(audio, s, e, n_audio):
    k0, k1 = min(max(s - M.AUDIO_START, 0), n_audio), min(max(e - M.AUDIO_START, 0), n_audio)
    return np.ascontiguousarray(audio[k0:k1]) if k1 > k0 else np.zeros((1, M.LM_D), np.float16)


def chunks(n):
    return [(s, min(n, s + CHUNK)) for s in range(0, n, CHUNK)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", nargs="+", default=["gpu", "cpu"], choices=["gpu", "cpu"])
    ap.add_argument("--case", nargs="+", default=CASES)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--variant", default="opt", choices=["opt", "basic"],
                    help="opt (default) or the basic oracle kernels (timing only; long clips can exceed the iOS GPU "
                         "time limit with basic's single-dispatch attention)")
    ap.add_argument("--cool-to", default="nominal", choices=["nominal", "fair"],
                    help="before each case, wait until the phone is at most this hot (default nominal)")
    ap.add_argument("--cool-max-wait", type=float, default=600.0,
                    help="seconds to wait per case at most (then time anyway; the state is recorded)")
    ap.add_argument("--pin", default="auto", choices=["auto", "none", "gpu", "lock"],
                    help="PIN_MODEL the graphs' weights after loading so a CPU step never re-reads them from flash "
                         "under memory pressure (gpu: a residency set of its own; lock: mlock; on iOS both count the "
                         "3.8 GB in the app's footprint). auto (default) = gpu for the CPU graphs (no speed cost, "
                         "A/B 2026-11-07), none for the GPU graphs (Metal keeps them resident while they run)")
    ap.add_argument("--quick", action="store_true",
                    help="A/B during development: 1 warm-up + 5 runs (3 when a call takes > 10 s), e2e one pass; "
                         "never recorded")
    a = ap.parse_args()
    global QUICK
    QUICK = a.quick
    assert not (a.quick and a.record), "--quick numbers are never recorded"
    bad = [c for c in a.case if c not in CASES]
    assert not bad, f"not an iPhone case: {bad} (cases: {CASES})"
    udid = localconfig.require("iphone_udid")
    dev = RemoteDevice.connect(f"{udid}.coredevice.local", timeout=900.0)
    st = dev.state()
    print("device:", st, flush=True)
    if st.get("low_power_mode"):
        sys.exit("the iPhone is in Low Power Mode (CPU / GPU throttled): turn it off before timing")
    sd = M.load_weights_opt() if a.variant == "opt" else M.load_weights()
    clips: dict[str, Clip] = {}
    for d in a.device:
        t0 = time.time()
        mid_e = dev.load(M.build_encoder(sd, d, variant=a.variant))
        mid_g = dev.load(M.build_generate(sd, d, variant=a.variant))
        pin = ("gpu" if d == "cpu" else "none") if a.pin == "auto" else a.pin
        if pin != "none" and dev.supports("pin_model"):
            for mid in (mid_e, mid_g):
                r = dev.pin_model(mid, pin)
                assert not r["error"], f"PIN_MODEL {pin}: {r}"
        print(f"== {DEVICE_NAME[d]}: graphs loaded in {time.time() - t0:.0f} s (pin {pin})", flush=True)
        wait_nominal(dev, max_wait_s=300)
        dummy = np.zeros((1, M.LM_D), np.float16)
        for case in a.case:
            kind, n = case.split()
            name = "s" + n.rstrip("s")
            if name not in clips:
                clips[name] = Clip(name)
            c = clips[name]
            th, waited = cool_down(dev, a.cool_to, a.cool_max_wait)
            if waited > 1:
                print(f"   (cooled {waited:.0f} s -> {th})", flush=True)
            state = dev.state()
            mem_ok, why = dev.memory_check()
            if not mem_ok:
                print(f"   WARNING: memory pressure before {case} ({why}): not recorded", flush=True)
            pageins0 = dev.memory().get("pageins", 0) if dev.supports("memory_pressure") else 0
            enc_in = [c.features, np.array([c.n_audio], np.int64)]
            (audio,), _, _ = dev.run2(mid_e, enc_in)
            seq = np.concatenate([c.ids, np.array(c.tokens[:-1], np.int64)])
            cap = min(M.MAX_CAP["iphone"], -(-(len(c.ids) + max(c.max_new, len(c.tokens))) // 1024) * 1024)
            kv = dev.alloc_tensor(np.float16, (M.LM_LAYERS, 2, cap, M.LM_KVH * M.LM_HD))
            try:
                if kind == "enc":
                    ms, runs = timed(dev, mid_e, enc_in)
                elif kind == "prefill":
                    ms, runs = 0.0, 0
                    for s, e in chunks(len(c.ids)):
                        t, r = timed(dev, mid_g, [c.ids[s:e], rows(audio, s, e, c.n_audio), np.array([s], np.int64), kv])
                        ms, runs = ms + t, r
                elif kind == "decode":
                    for s, e in chunks(len(seq) - 1):
                        dev.run2(mid_g, [seq[s:e], rows(audio, s, e, c.n_audio), np.array([s], np.int64), kv],
                                 return_outputs=False)
                    ms, runs = timed(dev, mid_g, [seq[-1:], dummy, np.array([len(seq) - 1], np.int64), kv])
                else:  # e2e: three passes, each the sum of its calls' on-device times
                    totals, match = [], None
                    for _ in range(1 if QUICK else 3):
                        _, _, t = dev.run2(mid_e, enc_in, return_outputs=False)
                        total = sum(t) * 1e-6
                        for s, e in chunks(len(c.ids)):
                            (lg,), _, t = dev.run2(mid_g, [c.ids[s:e], rows(audio, s, e, c.n_audio),
                                                           np.array([s], np.int64), kv])
                            total += sum(t) * 1e-6
                        toks, pos = [], len(c.ids)
                        while True:
                            tok = int(np.argmax(lg))
                            toks.append(tok)
                            if tok in M.EOS or len(toks) >= c.max_new:
                                break
                            (lg,), _, t = dev.run2(mid_g, [np.array([tok], np.int64), dummy, np.array([pos], np.int64), kv])
                            total += sum(t) * 1e-6
                            pos += 1
                        totals.append(total)
                        match = toks == c.tokens
                    ms, runs = sorted(totals)[len(totals) // 2], len(totals)
            finally:
                dev.free_tensor(kv)
            end = dev.state()
            extra = f", transcript = golden: {match}" if kind == "e2e" else ""
            # Thrash check (docs/todo.md "fc-thrashing"): weights re-read from flash during the case show up as
            # page-ins. A case touches each weight page once when cold (the first case after loading: 3.86 GB for
            # 3.80 GB of weights + code, 2026-11-07), so more than 1.5x the weights' size, or pressure at the end,
            # means the timings include flash reads (a thrashing case re-reads them every few steps).
            if dev.supports("memory_pressure"):
                m = dev.memory()
                paged = (m["pageins"] - pageins0) * m["page_size"]
                end_ok, why = dev.memory_check()
                if paged > max(1.5 * m["weights"], 1 << 30) or not end_ok:
                    mem_ok = False
                    print(f"   WARNING: {paged / 2**30:.2f} GB paged in during {case} ({why}): not recorded",
                          flush=True)
            print(f"   {DEVICE_NAME[d]} {case}: {ms:.2f} ms ({runs} runs; thermal {state.get('thermal')} -> "
                  f"{end.get('thermal')}){extra}", flush=True)
            if a.record and not mem_ok:
                print("   (not recorded: memory pressure)", flush=True)
            elif a.record:
                perfdb.record(MODEL, DEVICE_NAME[d], case, IMPL[a.variant], ms,
                              state={"thermal": end.get("thermal"), "thermal_start": state.get("thermal"),
                                     "low_power_mode": bool(end.get("low_power_mode"))},
                              notes="bench/models/bench_qwen3asr_iphone.py: on-device times (RUN2), "
                                    f"{'optimized' if a.variant == 'opt' else 'basic'} f16 graphs, median of {runs} runs" + (" (sum of the calls' device times)"
                                                                       if kind in ("prefill", "e2e") else ""))


if __name__ == "__main__":
    main()
