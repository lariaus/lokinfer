"""Check: (1) which RNG calls happen per forward and shapes; (2) whether rand_ini has any effect;
(3) whether the unused noise branch matters."""
import numpy as np, common
from common import torch, kokoro_model
common.load('cpu')
ids, ref = common.make_inputs(64)
calls = []
orig_rand, orig_randn_like, orig_randn = torch.rand, torch.randn_like, torch.randn
def rec(name, f):
    def g(*a, **k):
        r = f(*a, **k); calls.append((name, tuple(r.shape))); return r
    return g
torch.rand = rec('rand', orig_rand); torch.randn_like = rec('randn_like', orig_randn_like); torch.randn = rec('randn', orig_randn)
a = np.frombuffer(kokoro_model.infer(ids, ref, 1.0, seed=0), dtype=np.float32)
print('RNG calls:', calls)
# zero out rand_ini but keep RNG consumption identical
torch.rand = lambda *a, **k: orig_rand(*a, **k) * 0
b = np.frombuffer(kokoro_model.infer(ids, ref, 1.0, seed=0), dtype=np.float32)
print('rand_ini zeroed: max|diff| =', float(np.abs(a - b).max()))
torch.rand = orig_rand
# zero out the 2nd randn_like (unused noise branch) keeping consumption
cnt = [0]
def rl(x, *a, **k):
    r = orig_randn_like(x, *a, **k); cnt[0] += 1
    return r * 0 if cnt[0] == 2 else r
torch.randn_like = rl
c = np.frombuffer(kokoro_model.infer(ids, ref, 1.0, seed=0), dtype=np.float32)
print('2nd randn_like zeroed: max|diff| =', float(np.abs(a - c).max()))
torch.randn_like = orig_randn_like
d = np.frombuffer(kokoro_model.infer(ids, ref, 1.0, seed=0), dtype=np.float32)
print('determinism same seed: max|diff| =', float(np.abs(a - d).max()))
