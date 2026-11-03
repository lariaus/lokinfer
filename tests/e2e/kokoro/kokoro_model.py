# COPIED VERBATIM from english-practice: native-server/kokoro_tts/kokoro_model.py (2026-11-03) - the app's
# reference model path (PyTorch CPU), driven per chunk by run_backend.py's 'python' backend exactly as
# native-server/kokoro_tts/lib/kokoro_tts.cpp does (load(kokoro pkg dir, config.json, kokoro-v1_0.pth), then
# infer(chunk.inputIds, chunk.refS bytes, 1, seed) per chunk). Do not edit; re-copy if the app's file changes.
#
# The only Python left in Kokoro TTS: running the Kokoro-82M PyTorch model.
# Everything else - text normalization, G2P (misaki/spaCy/espeak-ng),
# chunking, voice packs, concatenation and WAV encoding - happens in C++
# (native-server/kokoro_frontend), which calls infer() once per chunk with
# the model's exact inputs. Imported once into the embedded interpreter
# living inside native_server_cli (see kokoro_tts.cpp) and kept warm for
# the process's lifetime.
#
# Loading never touches misaki/spaCy (kokoro/__init__.py imports its
# pipeline module, which imports both - avoided by importing kokoro.model
# through a stub package) and never hits the network (config/weights come
# from explicit paths). Two more start-up savings keep the model itself
# bit-identical:
# - weight init is skipped during construction (every parameter is
#   overwritten by the checkpoint right after anyway),
# - scipy.signal isn't imported just for the one Hann window kokoro's
#   iSTFTNet asks it for (a hardcoded, bit-exact copy is served instead).
import contextlib
import os
import sys
import types

import numpy as np

_HANN_20 = [
    '0x0.0p+0', '0x1.90f1ecbbab010p-6', '0x1.8722191a02d64p-4', '0x1.a61b9f7154b46p-3', '0x1.61c8864680b59p-2',
    '0x1.0000000000000p-1', '0x1.4f1bbcdcbfa54p-1', '0x1.96791823aad2fp-1', '0x1.cf1bbcdcbfa54p-1',
    '0x1.f378709a22a80p-1', '0x1.0000000000000p+0', '0x1.f378709a22a80p-1', '0x1.cf1bbcdcbfa54p-1',
    '0x1.96791823aad2fp-1', '0x1.4f1bbcdcbfa54p-1', '0x1.0000000000000p-1', '0x1.61c8864680b59p-2',
    '0x1.a61b9f7154b46p-3', '0x1.8722191a02d64p-4', '0x1.90f1ecbbab010p-6',
]

_model = None
_torch = None


def _scipy_get_window(window, n, fftbins=True):
    # scipy.signal.get_window('hann', 20, fftbins=True), bit for bit - the
    # only call kokoro/istftnet.py makes (n_fft=20).
    if (window, n, fftbins) != ('hann', 20, True):
        raise ValueError(f'unexpected get_window{(window, n, fftbins)}')
    return np.array([float.fromhex(h) for h in _HANN_20], dtype=np.float64)


def _import_kmodel(kokoro_package_dir):
    # A bare 'kokoro' package object, so `kokoro/__init__.py` (-> pipeline
    # -> misaki -> spaCy) never runs; loguru's 'kokoro' logger is disabled
    # exactly like that __init__ does, or forward() would print debug lines.
    package = types.ModuleType('kokoro')
    package.__path__ = [kokoro_package_dir]
    sys.modules['kokoro'] = package
    from loguru import logger
    logger.disable('kokoro')

    # Serve scipy.signal only while kokoro.istftnet imports it. The real
    # top-level scipy is imported first (cheap) - transformers checks it.
    import scipy  # noqa: F401
    saved = sys.modules.get('scipy.signal')
    stub = types.ModuleType('scipy.signal')
    stub.get_window = _scipy_get_window
    sys.modules['scipy.signal'] = stub
    try:
        from kokoro.model import KModel
    finally:
        if saved is None:
            sys.modules.pop('scipy.signal', None)
        else:
            sys.modules['scipy.signal'] = saved
    return KModel


@contextlib.contextmanager
def _skip_weight_init(torch):
    # Random weight init is ~half of KModel() construction time and is
    # entirely overwritten by load_state_dict - every parameter is in the
    # checkpoint.
    import torch.nn.init as init
    from transformers import modeling_utils
    names = [n for n in dir(init) if n.endswith('_') and not n.startswith('_')]
    saved_init = {n: getattr(init, n) for n in names}
    saved_reset = {}
    saved_post_init = modeling_utils.PreTrainedModel.post_init
    for n in names:
        setattr(init, n, lambda tensor, *args, **kwargs: tensor)
    for cls in (torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Embedding, torch.nn.LSTM,
                torch.nn.LayerNorm, torch.nn.InstanceNorm1d):
        if hasattr(cls, 'reset_parameters'):
            saved_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None
    modeling_utils.PreTrainedModel.post_init = lambda self: None
    try:
        yield
    finally:
        for n, f in saved_init.items():
            setattr(init, n, f)
        for cls, f in saved_reset.items():
            cls.reset_parameters = f
        modeling_utils.PreTrainedModel.post_init = saved_post_init


def load(kokoro_package_dir, config_path, weights_path):
    """Builds the model once: KModel(config, weights).to('cpu').eval(), the
    same construction KPipeline('a') did."""
    global _model, _torch
    KModel = _import_kmodel(kokoro_package_dir)
    import torch
    with _skip_weight_init(torch):
        _model = KModel(config=config_path, model=weights_path).to('cpu').eval()
    _torch = torch


def infer(input_ids, ref_s, speed, seed=None):
    """KModel.forward (kokoro/model.py:85-107) given the inputs C++ already
    computed: input_ids = vocab ids without the two 0 pads, ref_s = the
    voice pack's 256-float32 style row as raw little-endian bytes. Returns
    the float32 audio as raw bytes. `seed` (parity tests only) seeds torch's
    RNG first - the decoder draws random numbers on every call."""
    torch = _torch
    self = _model
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        assert len(input_ids) + 2 <= self.context_length, (len(input_ids) + 2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(self.device)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        ref_s = torch.frombuffer(bytearray(ref_s), dtype=torch.float32).reshape(1, 256).to(self.device)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu()
    return audio.numpy().tobytes()
