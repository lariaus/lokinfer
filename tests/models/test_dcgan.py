"""DCGAN 64x64 (CelebA weights): lokinfer G and D (CPU and GPU) vs the PyTorch tutorial modules, plus the
end-to-end sanity check D(G(z)) composed from both lokinfer graphs."""
import numpy as np
import pytest
import torch

from models import dcgan as M

import lokinfer as lk


@pytest.fixture(scope="module")
def ref():
    return M.torch_reference()


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def models(ref, request):
    g, d = ref
    return (M.build_generator(M.fold_generator(g), device=request.param),
            M.build_discriminator(M.fold_discriminator(d), device=request.param))


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


@pytest.mark.parametrize("b", [1, 3, 64])
def test_generator(ref, models, b):
    z = np.random.default_rng(b).standard_normal((b, 100, 1, 1)).astype(np.float32)
    with torch.no_grad():
        exp = ref[0](torch.from_numpy(z)).numpy()
    got = models[0](lk.from_numpy(z)).numpy()
    assert got.shape == exp.shape == (b, 3, 64, 64)
    assert rel(got, exp) < 1e-4, rel(got, exp)


@pytest.mark.parametrize("b", [1, 3, 64])
def test_discriminator(ref, models, b):
    x = np.tanh(np.random.default_rng(b + 7).standard_normal((b, 3, 64, 64))).astype(np.float32)
    with torch.no_grad():
        exp = ref[1](torch.from_numpy(x)).numpy()
    got = models[1](lk.from_numpy(x)).numpy()
    assert got.shape == exp.shape == (b,)
    np.testing.assert_allclose(got, exp, atol=1e-5, rtol=1e-4)


def test_d_of_g(ref, models):
    z = np.random.default_rng(0).standard_normal((16, 100, 1, 1)).astype(np.float32)
    img = models[0](lk.from_numpy(z)).numpy()
    p = models[1](lk.from_numpy(img)).numpy()
    with torch.no_grad():
        exp = ref[1](ref[0](torch.from_numpy(z))).numpy()
    np.testing.assert_allclose(p, exp, atol=1e-4, rtol=1e-3)
