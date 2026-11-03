"""torch.compile (inductor) on the decoder + bert, eager elsewhere, dynamic shapes.
usage: python bench_compile.py DEVICE [runs] [warmup]"""
import json, os, statistics, sys, time
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.abspath('inductor_cache'))
import common
from common import torch, kokoro_model, sync
dev = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 2
m = common.load(dev)
mode = os.environ.get('COMPILE_MODE', 'default')
m.decoder = torch.compile(m.decoder, dynamic=True, mode=mode)
m.bert = torch.compile(m.bert, dynamic=True, mode=mode)
real = json.load(open('real_ids.json'))
pack = torch.load(f'{common.SNAP}/voices/af_heart.pt', weights_only=True)
cases = [(f'rand{n}', *common.make_inputs(n)) for n in (16, 64, 128, 256, 510)]
for k, ids in real.items():
    cases.append((f'real_{k}{len(ids)}', ids, pack[len(ids) - 1].reshape(256).numpy().tobytes()))
for name, ids, ref in cases:
    t0 = time.perf_counter()
    for _ in range(warmup):
        kokoro_model.infer(ids, ref, 1.0, seed=0); sync(dev)
    tw = time.perf_counter() - t0
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); kokoro_model.infer(ids, ref, 1.0, seed=0); sync(dev); ts.append(time.perf_counter() - t)
    print(json.dumps({'dev': dev, 'mode': mode, 'case': name, 'warmup_s': round(tw, 2),
                      'median_ms': round(statistics.median(ts) * 1e3, 1), 'min_ms': round(min(ts) * 1e3, 1)}), flush=True)
