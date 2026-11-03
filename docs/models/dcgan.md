# DCGAN 64×64 (Generator + Discriminator) on lokinfer

The PyTorch DCGAN tutorial architecture (nz 100, ngf = ndf = 64, nc 3) with the CelebA weights of
`hussamalafandi/DCGAN_CelebA` (MIT, pinned revision). The weights are loaded strictly into our own copy of the
tutorial classes; the repo's `dcgan.py` is never executed. Plan: `impl-notes/research-models.md` §4.

Status (2026-11-01): **built and tested on CPU and GPU** (`models/dcgan.py`, `tests/models/test_dcgan.py`).

## Using it

```python
from models import dcgan as M
g, d = M.torch_reference()
G = M.build_generator(M.fold_generator(g))           # z f32[B, 100, 1, 1] -> image f32[B, 3, 64, 64]
D = M.build_discriminator(M.fold_discriminator(d))   # image f32[B, 3, 64, 64] -> probability f32[B]
# device="gpu" for either: GPU end to end, weights on the GPU. B is symbolic.
```

## Graph (NHWC inside, BN folded offline)

- G layer 1 (ConvTranspose 4×4 on a 1×1 input) is a GEMM `[B, 100] × [100, 4·4·512]` with bias + ReLU, viewed as
  `[B, 4, 4, 512]`. Layers 2–5 (ConvTranspose2d k4 s2 p1) are one GEMM `x @ W[Ci, 16·Co]` plus
  `convt2d_gather_nhwc` (each output gathers its ≤ 4 contributions, + bias, ReLU or tanh). The output goes back to
  NCHW (`nhwc_to_nchw`).
- D: four `conv2d_nhwc` k4 s2 p1 with a bias + LeakyReLU(0.2) epilogue, then the last 4×4 conv on the 4×4 map as a
  GEMV over the NHWC-flattened 8192 features (weight permuted offline), then sigmoid.
- GPU: D's first conv is an im2col straight from the NCHW input (K = 4·4·3 = 48) + a 1×1 steel conv with the leaky epilogue (padding 3 channels to 16 cost K = 256: 1.25 of 6.6 ms on the iPhone at N=64); the other convs are steel convs; MPS GEMMs, gather / layout / sigmoid kernels.

## Correctness

vs PyTorch at B ∈ {1, 3, 64}: G max relative error < 1e-4; D within 1e-5 abs / 1e-4 rel; D(G(z)) composed from both
lokinfer graphs matches PyTorch's D(G(z)). CPU and GPU.

## Performance

`bench/models/bench_m6_models.py dcgan` (lokinfer) and `bench/models/bench_other_baselines.py` (baselines). Note the
PyTorch CPU pathology recorded in research-models.md §4.4 (NNPACK on 4×4 s2 convs): the dashboard shows default eager.
