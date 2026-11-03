"""Benchmark mlx-audio's Kokoro (MLX, GPU) with fp32 weights converted from the hexgrad pth.
usage: python -I bench_mlx.py MODEL_DIR INPUTS.npz [runs] [warmup] [cpu]"""
import json, statistics, sys, time
import numpy as np
import mlx.core as mx
if len(sys.argv) > 5 and sys.argv[5] == 'cpu':
    mx.set_default_device(mx.cpu)
from mlx_audio.tts.utils import load_model
mdir, inputs = sys.argv[1], sys.argv[2]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 2
model = load_model(mdir)
vocab = json.load(open(mdir + '/config.json'))['vocab']
inv = {v: k for k, v in vocab.items()}
print(json.dumps({'mlx': mx.__version__, 'device': str(mx.default_device())}), flush=True)
d = np.load(inputs)
cases = sorted({k.split('_', 1)[1] for k in d.files}, key=lambda c: d['ids_' + c].shape[1])
for c in cases:
    ids = d['ids_' + c][0, 1:-1]
    ph = ''.join(inv[int(i)] for i in ids)
    ref = mx.array(d['style_' + c])
    for _ in range(warmup):
        a = model(ph, ref, 1.0)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        a = model(ph, ref, 1.0)
        mx.eval(a)
        ts.append(time.perf_counter() - t)
    print(json.dumps({'case': c, 'T': len(ids) + 2, 'samples': int(a.size), 'median_ms': statistics.median(ts) * 1e3,
                      'min_ms': min(ts) * 1e3}), flush=True)
