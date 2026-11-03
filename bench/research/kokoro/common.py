"""Shared loader for profiling Kokoro-82M exactly as english-practice's
kokoro_model.infer() runs it (read-only import of that module)."""
import json
import os
import sys

import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lokinfer import localconfig  # noqa: E402
from models.kokoro import reference as _R  # noqa: E402

SNAP = str(_R.snapshot_dir())  # the pinned Kokoro-82M checkpoint (Hugging Face cache)
_APP = localconfig.require('english_practice_dir')
KPKG = os.path.join(_APP, '.venv/lib/python3.14/site-packages/kokoro')
sys.path.insert(0, os.path.join(_APP, 'native-server/kokoro_tts'))

import kokoro_model  # noqa: E402
import torch  # noqa: E402


def load(device='cpu'):
    kokoro_model.load(KPKG, f'{SNAP}/config.json', f'{SNAP}/kokoro-v1_0.pth')
    m = kokoro_model._model
    if device != 'cpu':
        m = m.to(device)
        kokoro_model._model = m
    return m


def vocab_ids():
    with open(f'{SNAP}/config.json') as f:
        cfg = json.load(f)
    return sorted(set(cfg['vocab'].values()))


def make_inputs(n_tokens, seed=0, voice='af_heart'):
    """Random valid vocab ids (no 0 pad) + the voice pack row kokoro would pick."""
    rng = np.random.default_rng(seed)
    ids = [int(x) for x in rng.choice(vocab_ids(), size=n_tokens)]
    pack = torch.load(f'{SNAP}/voices/{voice}.pt', weights_only=True)
    ref_s = pack[n_tokens - 1].reshape(256).numpy().astype(np.float32).tobytes()
    return ids, ref_s


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
