"""GPT-2 124M prefill: lokinfer (CPU and GPU) vs HF GPT2LMHeadModel (eager attention), all-position logits, several T
(incl. 1, odd lengths, 512, 1024 = the full context)."""
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from models import gpt2 as M  # noqa: E402

import lokinfer as lk  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return M.torch_reference()


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def model(ref, request):
    return M.build(M.weights(ref), device=request.param)


@pytest.mark.parametrize("t", [1, 7, 128, 511, 1024])
def test_gpt2_prefill_matches_hf(ref, model, t):
    ids = np.random.default_rng(t).integers(0, M.VOCAB, t).astype(np.int64)
    with torch.no_grad():
        exp = ref(torch.from_numpy(ids)[None]).logits[0].numpy()
    got = model(lk.from_numpy(ids)).numpy()
    assert got.shape == exp.shape
    err = np.abs(got - exp).max() / np.abs(exp).max()
    assert err < 2e-4, err
    assert (got.argmax(-1) == exp.argmax(-1)).mean() > 0.99


def test_gpt2_last_only(ref):
    m = M.build(M.weights(ref), last_only=True)
    ids = np.random.default_rng(5).integers(0, M.VOCAB, 33).astype(np.int64)
    with torch.no_grad():
        exp = ref(torch.from_numpy(ids)[None]).logits[0, -1:].numpy()
    np.testing.assert_allclose(m(lk.from_numpy(ids)).numpy(), exp, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_gpt2_decode_greedy_matches_hf_generate(ref, device):
    """Decode step with persistent KV caches: greedy generation must match HF generate(do_sample=False) token for
    token, and each step's logits must match HF's at that position."""
    step = M.build_decode(M.weights(ref), device=device)
    prompt = [464, 3290, 286, 4881, 318]  # "The capital of France is"
    ours = M.greedy_generate(step, prompt, 24)
    with torch.no_grad():
        hf = ref.generate(torch.tensor([prompt]), max_new_tokens=24, do_sample=False, pad_token_id=50256)[0].tolist()
    assert ours == hf[len(prompt):]
    # logits of the last step == full forward's last row (fresh model state: the cache starts empty)
    step2 = M.build_decode(M.weights(ref), device=device)
    seq = prompt + ours[:10]
    for p, t in enumerate(seq):
        got = step2(lk.from_numpy(np.array([t], np.int64)), lk.from_numpy(np.array([p], np.int64))).numpy()
    with torch.no_grad():
        exp = ref(torch.tensor([seq])).logits[0, -1:].numpy()
    assert np.abs(got - exp).max() / np.abs(exp).max() < 2e-4


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_gpt2_chunked_generate_matches_hf(ref, device):
    """One compiled model for prefill + decode (shared persistent caches): prefill the whole prompt in one call, then
    one call per token; greedy output must match HF generate token for token, and the prefill logits HF's last row."""
    model = M.build_generate(M.weights(ref), device=device)
    prompt = [464, 3290, 286, 4881, 318, 6342, 13, 383, 3139, 286]
    with torch.no_grad():
        hf = ref.generate(torch.tensor([prompt]), max_new_tokens=32, do_sample=False, pad_token_id=50256)[0].tolist()
        exp = ref(torch.tensor([prompt])).logits[0, -1:].numpy()
    got = model(lk.from_numpy(np.array(prompt, np.int64)), lk.from_numpy(np.array([0], np.int64))).numpy()
    assert np.abs(got - exp).max() / np.abs(exp).max() < 2e-4
    model2 = M.build_generate(M.weights(ref), device=device)  # fresh caches
    assert M.generate(model2, prompt, 32) == hf[len(prompt):]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_gpt2_generate_two_chunk_prefill(ref, device):
    """build_generate with the prompt prefilled in two chunks (T = 6 at pos 0, then T = 4 at pos 6: the GPU's cache
    attention path; pos 0 takes the flash prefill kernel) matches HF's last-row logits."""
    model = M.build_generate(M.weights(ref), device=device)
    prompt = [464, 3290, 286, 4881, 318, 6342, 13, 383, 3139, 286]
    model(lk.from_numpy(np.array(prompt[:6], np.int64)), lk.from_numpy(np.array([0], np.int64)))
    got = model(lk.from_numpy(np.array(prompt[6:], np.int64)), lk.from_numpy(np.array([6], np.int64))).numpy()
    with torch.no_grad():
        exp = ref(torch.tensor([prompt])).logits[0, -1:].numpy()
    assert np.abs(got - exp).max() / np.abs(exp).max() < 2e-4


def test_gpt2_generate_multi_chunk_prefill_gpu(ref):
    """GPU build_generate with later prefill chunks at pos > 0 longer than one 32-query block and not tile-aligned
    (the simdgroup-matrix flash attention over the cache, lk_attention_cache_sg), interleaved with a decode step
    (key-split decode attention): each call's last-row logits match HF's full forward over the tokens so far."""
    model = M.build_generate(M.weights(ref), device="gpu")
    ids = np.random.default_rng(3).integers(0, M.VOCAB, 300).astype(np.int64)
    pos = 0
    for n in (40, 70, 1, 117, 1, 71):
        got = model(lk.from_numpy(ids[pos:pos + n]), lk.from_numpy(np.array([pos], np.int64))).numpy()
        pos += n
        with torch.no_grad():
            exp = ref(torch.from_numpy(ids[:pos])[None]).logits[0, -1:].numpy()
        assert np.abs(got - exp).max() / np.abs(exp).max() < 2e-4, (pos, n)
