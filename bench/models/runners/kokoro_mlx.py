"""MLX (mlx-audio) Kokoro-82M runner for bench_kokoro_baselines.py. Runs in a separate venv with mlx + mlx-audio
(not lokinfer's .venv): python -I kokoro_mlx.py MODEL_DIR CASES.npz RUNS WARMUP
Timing protocol (docs/perf-dashboard.md): >= WARMUP calls and >= 1 s of warm-up, then >= RUNS calls, stopping once
>= 30 calls or >= 2 s have been measured; median.
CASES.npz holds, per case c: ids_c (i64 [T], padded with 0 at both ends), style_c (f32 [256]), speed_c (f32).
mlx-audio's model takes a phoneme string, so the ids are mapped back through the model's own vocab (exact inverse:
it re-tokenizes to the same ids). Prints one JSON line per case with the output sample count, so the caller can
check the durations (and thus the work) match the reference exactly."""
import json
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

mdir, cases_path, runs, warmup = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model = load_model(mdir)
vocab = json.load(open(mdir + "/config.json"))["vocab"]
inv = {v: k for k, v in vocab.items()}
d = np.load(cases_path)
print(json.dumps({"mlx": mx.__version__, "device": str(mx.default_device())}), flush=True)
for c in sorted({k.split("_", 1)[1] for k in d.files}):
    ids = d["ids_" + c][1:-1]
    ph = "".join(inv[int(i)] for i in ids)
    ref = mx.array(d["style_" + c].reshape(1, 256))
    speed = float(d["speed_" + c])
    n_warm, warm_end = 0, time.perf_counter() + 1.0
    while n_warm < warmup or time.perf_counter() < warm_end:
        a = model(ph, ref, speed)
        mx.eval(a)
        n_warm += 1
    ts = []
    t_start = time.perf_counter()
    while len(ts) < runs or not (len(ts) >= 30 or time.perf_counter() - t_start >= 2.0):
        t = time.perf_counter()
        a = model(ph, ref, speed)
        mx.eval(a)
        ts.append(time.perf_counter() - t)
    print(json.dumps({"case": c, "samples": int(a.size), "median_ms": statistics.median(ts) * 1e3,
                      "min_ms": min(ts) * 1e3, "runs": len(ts), "warmup": n_warm, "f32": all(v.dtype == mx.float32 for _, v in
                                                          __import__("mlx.utils").utils.tree_flatten(model.parameters()))}),
          flush=True)
