"""ResNet-18: lokinfer graph vs torchvision (pretrained IMAGENET1K_V1), random normalized inputs, several sizes
(fully convolutional up to the global pool: odd / non-square H, W exercise every padding tail)."""
import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")

from models import resnet18 as M  # noqa: E402

import lokinfer as lk  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return M.torch_reference()


@pytest.fixture(scope="module", params=["cpu", "cpu-cnhw", "gpu"])
def model(ref, request):
    w = M.fold_weights(ref)
    return {"cpu": M.build, "cpu-cnhw": M.build_cnhw, "gpu": M.build_gpu}[request.param](w)


@pytest.mark.parametrize("n,h,w", [(1, 224, 224), (2, 225, 223), (1, 160, 160), (3, 64, 96), (1, 33, 33)])
def test_resnet18_matches_torchvision(ref, model, n, h, w):
    x = np.random.default_rng(n * 1000 + h + w).standard_normal((n, 3, h, w)).astype(np.float32)
    with torch.no_grad():
        exp = ref(torch.from_numpy(x).double().float()).numpy()
    got = model(lk.from_numpy(x)).numpy()
    assert got.shape == exp.shape
    err = np.abs(got - exp).max() / np.abs(exp).max()
    assert err < 1e-4, err
    assert (got.argmax(1) == exp.argmax(1)).all()
