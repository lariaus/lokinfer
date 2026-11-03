"""Clean totals-only eager benchmark (exact kokoro_model.infer), all cases. usage: bench_eager.py DEVICE [runs] [warmup]"""
import json, statistics, sys, time
import common
from common import torch, kokoro_model, sync
dev = sys.argv[1]; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 9; warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 3
common.load(dev)
real = json.load(open('real_ids.json'))
pack = torch.load(f'{common.SNAP}/voices/af_heart.pt', weights_only=True)
cases = [(f'rand{n}', *common.make_inputs(n)) for n in (16, 64, 128, 256, 510)]
for k, ids in real.items():
    cases.append((f'real_{k}{len(ids)}', ids, pack[len(ids) - 1].reshape(256).numpy().tobytes()))
for name, ids, ref in cases:
    for _ in range(warmup):
        kokoro_model.infer(ids, ref, 1.0, seed=0); sync(dev)
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); a = kokoro_model.infer(ids, ref, 1.0, seed=0); sync(dev); ts.append(time.perf_counter() - t)
    q = statistics.quantiles(ts, n=4)
    print(json.dumps({'dev': dev, 'case': name, 'samples': len(a) // 4, 'median_ms': round(statistics.median(ts) * 1e3, 1),
                      'min_ms': round(min(ts) * 1e3, 1), 'iqr_ms': round((q[2] - q[0]) * 1e3, 1)}), flush=True)
