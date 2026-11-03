import time, statistics, common
from common import torch, kokoro_model
m=common.load('cpu')
ids,ref=common.make_inputs(128)
def t(n=5):
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); kokoro_model.infer(ids,ref,1.0,seed=0); ts.append(time.perf_counter()-t0)
    return statistics.median(ts)*1e3
for _ in range(2): kokoro_model.infer(ids,ref,1.0,seed=0)
print('nohook',t())
hs=[]
for mod in m.modules():
    hs.append(mod.register_forward_pre_hook(lambda *a: None)); hs.append(mod.register_forward_hook(lambda *a: None))
print('allhooks',t())
for h in hs: h.remove()
print('nohook',t())
torch.set_flush_denormal(True)
print('flush_denormal',t())
