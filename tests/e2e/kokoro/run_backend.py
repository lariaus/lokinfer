"""Step 2: synthesizes every prepared text with one backend; same per-chunk inputs, chunk audio concatenated, 24 kHz
16-bit WAV (the app's encoding) per text, per-text timing.

    .venv/bin/python tests/e2e/kokoro/run_backend.py BACKEND [--run NAME] [--part I/N] [--ids=ID,ID...] [--redo]

BACKEND:
  python    english-practice's exact path: kokoro_model.py (copied from the app) load() + infer(ids, ref_s bytes, 1,
            seed) per chunk, PyTorch CPU, as native-server/kokoro_tts/lib/kokoro_tts.cpp drives it;
  python_b  the same with other seeds: the run-to-run baseline (Kokoro's decoder draws noise on every call);
  mac_gpu   lokinfer models/kokoro build(weights, gpu=True) (the shipped variant: ids with the 0 pads, ref_s,
            speed 1.0, seed), in process (the editable package's Release build);
  ios_gpu   the same compiled model on the iPhone via the remote executor (LokiHost, scripts/ios_deploy.sh); the
            audio comes back over the cable. Time = the device's own run time (dev.run's times_ns).
Timing per chunk excludes setup (model load/compile, one warm-up call per backend); a text's time is the sum over its
chunks. Results: <run>/results/BACKEND.json, audio: <run>/audio/BACKEND/<id>.wav. Already-done texts are skipped
unless --redo; --part I/N runs only every text with index % N == I - 1 (to split iPhone sessions); --ids=t140,t141
runs (and redoes) only those texts (e.g. after changing them in texts.json and re-running prepare.py).

Locks (docs/benchmarking.md): mac_gpu under scripts/lk-lock /tmp/lokinfer-macbench.lock; ios_gpu inside ONE
scripts/lk-lock /tmp/lokinfer-phone.lock session per invocation.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402

from common import (HERE, ROOT, arg, chunk_seed, load_inputs, load_results, run_dir, save_results,  # noqa: E402
                    iphone_udid, write_wav)


def python_backend(run):
    """kokoro_tts.cpp's loadModelOnPythonThread + inferOnPythonThread."""
    sys.path.insert(0, str(HERE))
    import kokoro_model  # the app's file (copied)
    from prepare import snapshot_dir
    import importlib.util
    kokoro_pkg = os.path.dirname(importlib.util.find_spec("kokoro").origin)  # KOKORO_TTS_VENV_SITE_PACKAGES/kokoro
    snap = snapshot_dir()
    kokoro_model.load(kokoro_pkg, str(snap / "config.json"), str(snap / "kokoro-v1_0.pth"))

    def infer(ids, ref_s, seed):
        t0 = time.perf_counter()
        # KPipeline called the model with speed=1 (an int); refS as raw float32 bytes
        raw = kokoro_model.infer(list(ids), ref_s.astype(np.float32).tobytes(), 1, seed)
        dt = time.perf_counter() - t0
        return np.frombuffer(raw, np.float32).copy(), dt * 1e3, {}
    return infer, None


def lokinfer_model():
    sys.path.insert(0, str(ROOT))
    from models.kokoro import model as K, reference as R
    return K.build(K.fold_weights(R.load_model()), gpu=True)


def model_inputs(ids, ref_s, seed):
    return [np.asarray([0, *ids, 0], np.int64), np.ascontiguousarray(ref_s, np.float32), np.float32(1.0),
            np.int64(seed)]


def mac_gpu_backend(run):
    model = lokinfer_model()

    def infer(ids, ref_s, seed):
        inputs = model_inputs(ids, ref_s, seed)
        t0 = time.perf_counter()
        audio = model(*inputs).numpy()
        dt = time.perf_counter() - t0
        return np.array(audio, np.float32).reshape(-1), dt * 1e3, {}
    return infer, None


def ios_gpu_backend(run):
    """The compiled model is saved once per run (<run>/kokoro_gpu.lkm) and uploaded from that file (the phone caches
    it by sha256), so later iPhone sessions don't recompile on the Mac."""
    from lokinfer.remote import RemoteDevice
    model = run / "kokoro_gpu.lkm"
    if not model.exists():
        lokinfer_model().save(str(model) + ".tmp")
        os.replace(str(model) + ".tmp", model)
    dev = RemoteDevice.connect(f"{iphone_udid()}.coredevice.local", timeout=600.0)
    state = dev.state()
    print("device:", state, flush=True)
    t0 = time.time()
    mid = dev.load(model)
    print(f"model loaded on the phone ({time.time() - t0:.1f} s)", flush=True)

    def infer(ids, ref_s, seed):
        inputs = model_inputs(ids, ref_s, seed)
        inputs[2] = np.asarray(1.0, np.float32).reshape(())
        inputs[3] = np.asarray(seed, np.int64).reshape(())
        t0 = time.perf_counter()
        outs, times = dev.run(mid, inputs)
        wall = (time.perf_counter() - t0) * 1e3
        return np.array(outs[0], np.float32).reshape(-1), times[0] * 1e-6, {"wall_ms": wall}
    return infer, dev


def main():
    argv = sys.argv[1:]
    backend = argv[0]
    run = run_dir(arg(argv, "--run"))
    part = arg(argv, "--part")
    redo = "--redo" in argv
    ids = next((set(a.split("=", 1)[1].split(",")) for a in argv if a.startswith("--ids=")), None)
    meta, refs = load_inputs(run)
    texts = meta["texts"]
    todo = [(i, t) for i, t in enumerate(texts) if not t["no_audio"]]
    if part:
        pi, pn = (int(x) for x in part.split("/"))
        todo = [(i, t) for i, t in todo if i % pn == pi - 1]
    if ids is not None:
        unknown = ids - {t["id"] for t in texts}
        if unknown:
            raise SystemExit(f"unknown ids: {sorted(unknown)}")
        todo = [(i, t) for i, t in todo if t["id"] in ids]
        redo = True
    res = load_results(run, backend)
    if not redo:
        todo = [(i, t) for i, t in todo if t["id"] not in res["texts"]]
    print(f"{backend}: {len(todo)} texts to run ({run})", flush=True)
    if not todo:
        return
    if backend == "ios_gpu" and os.path.exists("/tmp/lokinfer-phone-disabled"):
        raise SystemExit("iPhone disabled by the user: " + open("/tmp/lokinfer-phone-disabled").read().strip())
    make = {"python": python_backend, "python_b": python_backend, "mac_gpu": mac_gpu_backend,
            "ios_gpu": ios_gpu_backend}[backend]
    t_setup = time.time()
    infer, dev = make(run)
    # warm-up (excluded from the timing): a short, a median and a long real chunk, twice each, so first-call costs
    # (pipeline creation, allocations, GPU clocks) don't land on the first timed text
    every = sorted(((len(c["ids"]), t["id"], k) for t in texts if not t["no_audio"] for k, c in enumerate(t["chunks"])))
    for _ in range(2):
        for L, tid, k in (every[0], every[len(every) // 2], every[-1]):
            ids = next(t for t in texts if t["id"] == tid)["chunks"][k]["ids"]
            infer(ids, refs[f"{tid}_{k}"], 12345)
    if dev is not None:
        res["device_state_start"] = dev.state()
    res["setup_s"] = round(time.time() - t_setup, 2)
    adir = run / "audio" / backend
    t_all = time.time()
    for n, (i, t) in enumerate(todo):
        pieces, chunk_ms, chunk_samples, extra = [], [], [], []
        try:
            for k, c in enumerate(t["chunks"]):
                audio, ms, ex = infer(c["ids"], refs[f"{t['id']}_{k}"], chunk_seed(backend, i, k))
                pieces.append(audio)
                chunk_ms.append(round(ms, 3))
                chunk_samples.append(int(audio.size))
                extra.append(ex)
        except Exception as e:  # report precisely, keep going
            res["texts"][t["id"]] = {"error": f"chunk {len(pieces)}: {type(e).__name__}: {e}"}
            print(f"  {t['id']} ERROR chunk {len(pieces)} ({t['voice']}): {e}", flush=True)
            save_results(run, backend, res)
            continue
        audio = np.concatenate(pieces)
        write_wav(adir / f"{t['id']}.wav", audio)
        rec = {"samples": int(audio.size), "chunk_samples": chunk_samples, "ms": round(sum(chunk_ms), 3),
               "chunk_ms": chunk_ms, "nonfinite": int((~np.isfinite(audio)).sum()),
               "peak": float(np.max(np.abs(np.nan_to_num(audio)))) if audio.size else 0.0}
        if extra and "wall_ms" in extra[0]:
            rec["wall_ms"] = round(sum(e["wall_ms"] for e in extra), 3)
        res["texts"][t["id"]] = rec
        if n % 10 == 9 or n == len(todo) - 1:
            save_results(run, backend, res)
            print(f"  {n + 1}/{len(todo)}  ({time.time() - t_all:.0f} s)", flush=True)
    if dev is not None:
        res["device_state_end"] = dev.state()
        dev.close()
    save_results(run, backend, res)
    print(f"{backend}: done in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
