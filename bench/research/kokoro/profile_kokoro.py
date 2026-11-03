"""Profile Kokoro-82M forward (english-practice kokoro_model.infer) on cpu/mps.

usage: python profile_kokoro.py DEVICE [--runs N] [--warmup W] [--ops]
Outputs JSON to results_<device>.json and prints tables.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict

import numpy as np

import common
from common import torch, kokoro_model, sync

ap = argparse.ArgumentParser()
ap.add_argument('device')
ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--warmup', type=int, default=2)
ap.add_argument('--ops', action='store_true', help='torch.profiler op breakdown (cpu meaningful)')
ap.add_argument('--lens', default='16,64,128,256,510')
args = ap.parse_args()
dev = args.device

torch.manual_seed(0)
model = common.load(dev)
print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'device', dev)

real = json.load(open('real_ids.json'))
cases = [(f'rand{n}', *common.make_inputs(n)) for n in map(int, args.lens.split(','))]
pack = torch.load(f'{common.SNAP}/voices/af_heart.pt', weights_only=True)
for k, ids in real.items():
    cases.append((f'real_{k}{len(ids)}', ids, pack[len(ids) - 1].reshape(256).numpy().tobytes()))


def run(ids, ref_s):
    out = kokoro_model.infer(ids, ref_s, 1.0, seed=0)
    sync(dev)
    return out


# ---------------- module timing hooks ----------------
times = defaultdict(float)
stack = {}
hook_handles = []


def add_hooks():
    m = model
    named = {
        'bert': m.bert, 'bert.embeddings': m.bert.embeddings, 'bert.encoder': m.bert.encoder,
        'bert.pooler': m.bert.pooler,
        'bert_encoder': m.bert_encoder,
        'pred.text_encoder(DurEnc)': m.predictor.text_encoder,
        'pred.lstm': m.predictor.lstm, 'pred.duration_proj': m.predictor.duration_proj,
        'pred.shared_lstm': m.predictor.shared,
        'pred.F0_proj': m.predictor.F0_proj, 'pred.N_proj': m.predictor.N_proj,
        'text_encoder': m.text_encoder, 'text_encoder.lstm': m.text_encoder.lstm,
        'decoder': m.decoder, 'dec.encode': m.decoder.encode, 'dec.asr_res': m.decoder.asr_res,
        'dec.generator': m.decoder.generator, 'gen.m_source': m.decoder.generator.m_source,
        'gen.conv_post': m.decoder.generator.conv_post,
    }
    for i, b in enumerate(m.predictor.F0):
        named[f'pred.F0[{i}]'] = b
    for i, b in enumerate(m.predictor.N):
        named[f'pred.N[{i}]'] = b
    for i, c in enumerate(m.text_encoder.cnn):
        named[f'text_encoder.cnn[{i}]'] = c
    for i, b in enumerate(m.decoder.decode):
        named[f'dec.decode[{i}]'] = b
    g = m.decoder.generator
    for i in range(2):
        named[f'gen.noise_convs[{i}]'] = g.noise_convs[i]
        named[f'gen.noise_res[{i}]'] = g.noise_res[i]
        named[f'gen.ups[{i}]'] = g.ups[i]
    for i, b in enumerate(g.resblocks):
        named[f'gen.resblocks[{i}]'] = b

    for name, mod in named.items():
        def pre(_m, _in, name=name):
            sync(dev)
            stack[name] = time.perf_counter()

        def post(_m, _in, _out, name=name):
            sync(dev)
            times[name] += time.perf_counter() - stack.pop(name)
        hook_handles.append(mod.register_forward_pre_hook(pre))
        hook_handles.append(mod.register_forward_hook(post))

    # STFT transform/inverse are plain methods, wrap them
    stft = g.stft
    for meth in ('transform', 'inverse'):
        orig = getattr(stft, meth)

        def wrapped(*a, orig=orig, meth=meth, **k):
            sync(dev)
            t0 = time.perf_counter()
            r = orig(*a, **k)
            sync(dev)
            times[f'gen.stft.{meth}'] += time.perf_counter() - t0
            return r
        setattr(stft, meth, wrapped)


def frames_for(ids, ref_s):
    # reproduce the duration prediction to get F
    with torch.no_grad():
        m = model
        x = torch.LongTensor([[0, *ids, 0]]).to(dev)
        L = torch.LongTensor([x.shape[-1]]).to(dev)
        mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=dev)
        bd = m.bert(x, attention_mask=(~mask).int())
        d_en = m.bert_encoder(bd).transpose(-1, -2)
        s = torch.frombuffer(bytearray(ref_s), dtype=torch.float32).reshape(1, 256).to(dev)[:, 128:]
        d = m.predictor.text_encoder(d_en, s, L, mask)
        xx, _ = m.predictor.lstm(d)
        dur = torch.sigmoid(m.predictor.duration_proj(xx)).sum(-1)
        return int(torch.round(dur).clamp(min=1).long().sum().item())


results = {'device': dev, 'torch': torch.__version__, 'threads': torch.get_num_threads(), 'cases': []}
for name, ids, ref_s in cases:
    F = frames_for(ids, ref_s)
    for _ in range(args.warmup):
        audio = run(ids, ref_s)
    ts = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        audio = run(ids, ref_s)
        ts.append(time.perf_counter() - t0)
    n_samples = len(audio) // 4
    med = statistics.median(ts)
    rec = {'case': name, 'T': len(ids) + 2, 'F': F, 'samples': n_samples, 'audio_s': n_samples / 24000,
           'median_ms': med * 1e3, 'min_ms': min(ts) * 1e3, 'max_ms': max(ts) * 1e3,
           'rtf': med / (n_samples / 24000), 'all_ms': [t * 1e3 for t in ts]}
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k != 'all_ms'}),
          flush=True)
    results['cases'].append(rec)

# module breakdown on a subset (hooks add syncs)
add_hooks()
results['modules'] = {}
for name, ids, ref_s in cases:
    per = defaultdict(list)
    tot = []
    for r in range(args.warmup + args.runs):
        times.clear()
        t0 = time.perf_counter()
        run(ids, ref_s)
        dt = time.perf_counter() - t0
        if r >= args.warmup:
            tot.append(dt)
            for k, v in times.items():
                per[k].append(v)
    medm = {k: statistics.median(v) * 1e3 for k, v in per.items()}
    medm['TOTAL(with hooks)'] = statistics.median(tot) * 1e3
    results['modules'][name] = medm
    print(f'\n== modules {name} (ms, median)')
    for k, v in medm.items():
        print(f'  {k:32s} {v:9.2f}')
for h in hook_handles:
    h.remove()

if args.ops:
    from torch.profiler import profile, ProfilerActivity
    results['ops'] = {}
    for name, ids, ref_s in cases:
        if name not in ('rand128', 'rand510', 'real_long394'):
            continue
        run(ids, ref_s)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            run(ids, ref_s)
        ka = prof.key_averages()
        rows = sorted(((e.key, e.count, e.self_cpu_time_total / 1e3) for e in ka), key=lambda r: -r[2])
        total = sum(r[2] for r in rows)
        results['ops'][name] = {'total_self_ms': total, 'rows': rows[:40], 'n_op_calls': sum(r[1] for r in rows)}
        print(f'\n== ops {name}: total self {total:.1f} ms, {sum(r[1] for r in rows)} op calls')
        for k, c, t in rows[:30]:
            print(f'  {k:45s} {c:6d} {t:9.2f} {100 * t / total:5.1f}%')

json.dump(results, open(f'results_{dev}.json', 'w'), indent=1)
