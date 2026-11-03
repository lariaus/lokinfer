"""MNIST MLP: lokinfer vs PyTorch, every kernel variant, dynamic batch, plus
full test-set accuracy."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models import mnist_mlp  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return mnist_mlp.torch_reference()


@pytest.fixture(scope="module")
def weights(ref):
    return mnist_mlp.weights_from_torch(ref)


@pytest.mark.parametrize("variant", list(mnist_mlp.VARIANTS))
def test_matches_torch_dynamic_batch(ref, weights, variant):
    model = mnist_mlp.build(weights, variant)
    # Weight transposes were hoisted into init: exec has only the 3 GEMMs.
    final = model.ir_text()
    exec_part = final[final.index("exec("):]
    assert exec_part.count("kernel ") == 3 and "transpose" not in exec_part
    rng = np.random.default_rng(0)
    for b in [1, 7, 64, 1024]:
        x = ((rng.random((b, 1, 28, 28), dtype=np.float32) - mnist_mlp.MEAN) / mnist_mlp.STD).astype(np.float32)
        with torch.inference_mode():
            expect = ref(torch.from_numpy(x)).numpy()
        got = model(x).numpy()
        assert got.shape == (b, 10)
        np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
def test_test_set_accuracy(weights):
    datasets = pytest.importorskip("torchvision.datasets")
    from pathlib import Path

    root = Path.home() / ".cache" / "lokinfer" / "datasets"
    try:
        ds = datasets.MNIST(str(root), train=False, download=True)
    except Exception as e:  # offline
        pytest.skip(f"MNIST unavailable: {e}")
    x = ds.data.numpy().astype(np.float32)[:, None] / 255.0
    x = ((x - mnist_mlp.MEAN) / mnist_mlp.STD).astype(np.float32)
    y = ds.targets.numpy()
    for variant in mnist_mlp.VARIANTS:
        model = mnist_mlp.build(weights, variant)
        acc = (model(x).numpy().argmax(1) == y).mean()
        assert abs(acc - 0.9744) < 1e-4, (variant, acc)
