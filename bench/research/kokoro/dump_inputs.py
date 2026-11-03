"""Write the benchmark inputs (padded ids + style row) to inputs.npz for non-torch runners."""
import json

import numpy as np

import common

out = {}
for n in (16, 64, 128, 256, 510):
    ids, ref = common.make_inputs(n)
    out[f'ids_rand{n}'] = np.array([[0, *ids, 0]], dtype=np.int64)
    out[f'style_rand{n}'] = np.frombuffer(ref, dtype=np.float32).reshape(1, 256)
pack = common.torch.load(f'{common.SNAP}/voices/af_heart.pt', weights_only=True).numpy()
for k, ids in json.load(open('real_ids.json')).items():
    out[f'ids_real{len(ids)}'] = np.array([[0, *ids, 0]], dtype=np.int64)
    out[f'style_real{len(ids)}'] = pack[len(ids) - 1].reshape(1, 256).astype(np.float32)
np.savez('inputs.npz', **out)
print(sorted(out))
