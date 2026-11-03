"""Count MACs per module group and op kind for one forward, plus tensor shapes log."""
import json, sys
from collections import defaultdict
import common
from common import torch, kokoro_model
nn = torch.nn
m = common.load('cpu')
macs_by_group = defaultdict(float); macs_by_kind = defaultdict(float); counts = defaultdict(int)
shapes = []
def group(name):
    p = name.split('.')
    if p[0] == 'decoder' and len(p) > 2 and p[1] == 'generator':
        return 'decoder.generator.' + p[2] + ('.' + p[3] if p[2] in ('resblocks', 'noise_res', 'ups', 'noise_convs') else '')
    return '.'.join(p[:2])
def hook(name):
    def f(mod, inp, out):
        x = inp[0]
        o = out[0] if isinstance(out, tuple) else out
        if isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d)):
            k = mod.kernel_size[0]; cin = mod.in_channels // mod.groups
            if isinstance(mod, nn.Conv1d):
                macs = o.shape[1] * o.shape[2] * cin * k
            else:
                macs = x.shape[1] * x.shape[2] * (mod.out_channels // mod.groups) * k
            kind = ('convT1d' if isinstance(mod, nn.ConvTranspose1d) else 'conv1d') + (f'_dw' if mod.groups > 1 else '') + f'_k{k}'
        elif isinstance(mod, nn.Linear):
            macs = x.numel() // x.shape[-1] * mod.in_features * mod.out_features; kind = 'linear'
        elif isinstance(mod, nn.LSTM):
            L = x.data.shape[0] if isinstance(x, nn.utils.rnn.PackedSequence) else x.shape[1]
            H = mod.hidden_size; I = mod.input_size
            macs = 2 * L * 4 * H * (I + H); kind = 'lstm'
        else:
            return
        macs_by_group[group(name)] += macs; macs_by_kind[kind] += macs; counts[kind] += 1
        xs = x.data if isinstance(x, nn.utils.rnn.PackedSequence) else x
        os_ = o.data if isinstance(o, nn.utils.rnn.PackedSequence) else o
        shapes.append((name, type(mod).__name__, list(xs.shape), list(os_.shape)))
    return f
for name, mod in m.named_modules():
    if isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.LSTM)):
        mod.register_forward_hook(hook(name))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
ids, ref = common.make_inputs(n)
audio = kokoro_model.infer(ids, ref, 1.0, seed=0)
tot = sum(macs_by_kind.values())
print(f'n={n} samples={len(audio)//4} total GMAC={tot/1e9:.2f}')
for k, v in sorted(macs_by_kind.items(), key=lambda r: -r[1]): print(f'  kind {k:22s} n={counts[k]:4d} {v/1e9:8.3f} GMAC {100*v/tot:5.1f}%')
for k, v in sorted(macs_by_group.items(), key=lambda r: -r[1]): print(f'  group {k:40s} {v/1e9:8.3f} GMAC {100*v/tot:5.1f}%')
json.dump(shapes, open(f'shapes_{n}.json', 'w'), indent=0)
