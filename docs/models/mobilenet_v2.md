# MobileNetV2 on lokinfer

ImageNet MobileNetV2 (`torchvision.models.mobilenet_v2`, IMAGENET1K_V1 weights), forward only: normalized image
`f32[N, 3, H, W]` → logits `f32[N, 1000]`. Plan: `impl-notes/research-models.md` §3. It is the runtime-overhead case
(many small kernels): the record/replay (ICB) experiment in M6 uses it.

Status (2026-11-01): **built and tested on CPU and GPU** (`models/mobilenet_v2.py`, `tests/models/test_mobilenet_v2.py`).

## Using it

```python
from models import mobilenet_v2 as M
blocks = M.fold_weights(M.torch_reference())   # BN folded; walks torchvision's own module tree
cpu = M.build(blocks)                          # or M.build(blocks, device="gpu")
logits = cpu(x)                                # x f32[N, 3, H, W], N / H / W symbolic
```

## Graph

- NHWC; BN folded offline. Per InvertedResidual block: 1×1 expand conv + ReLU6, 3×3 depthwise conv + ReLU6, 1×1
  linear projection (+ the block residual when `use_res_connect`; no activation: the linear bottleneck). Stem conv,
  1×1 → 1280 head, global avgpool, FC.
- CPU **[built]**: every expand → dw → project block is one kernel, `mbv2_block_nhwc[_res]_f32_cpu`
  (`kernels/cpu/conv2d.cpp`, `build(..., fused=True)`, the default). One task per (image, band of output rows): the
  band's expanded rows (+ the 3×3 halo) and its depthwise output stay in per-thread scratch, so the expanded tensor
  never round-trips through memory (112×112×96 at N=8: 38 MB each way). Small maps (14×14, 7×7 at N=1, where a band
  would be 1-2 rows and the halo would triple the expand work) run expand / dw / project unfused inside the same
  kernel. `fused=False` keeps the three separate kernels (`conv2d_nhwc`, `dwconv2d_nhwc`, `conv2d_nhwc`). Measured
  2026-11-01: Mac N=1 1.62 → 1.53 ms, N=8 8.4 → 6.9 ms; iPhone N=8 interleaved A/B 10.1 → 9.0 ms (N=1 within noise).
- CPU depthwise 3×3: kernel size, stride and activation are template parameters, the 18 weight vectors of an
  8-channel block stay in registers, two output pixels per iteration (independent FMA chains), padded rows read a
  zero row; tasks = (output row, channel chunk) so 7×7 / 14×14 maps still feed every thread (M2 Max single core
  15 → 60 GFLOP/s).
- GPU depthwise 3×3: one thread = 4 channels (float4) × 4 output columns, stride as a template parameter (`lk_dwconv2d_nhwc_k3s{1,2}c4`; 4.4× the one-thread-per-element kernel, which stays as the fallback); the NCHW→NHWC input conversion runs one thread per pixel.
- GPU channel padding: the steel loader needs Ci % 16 == 0, so 24-channel stages are padded to 32 (and the 3-channel
  input to 16) with zero weights / biases; padded channels stay exactly 0 through ReLU6 and the residual adds.

## Correctness

vs torchvision on random normalized inputs (same sizes as ResNet-18): max relative error < 1e-4, identical top-1, CPU
and GPU.

## Performance

`bench/models/bench_vision.py` (lokinfer) and `bench/models/bench_other_baselines.py` (baselines); dashboard.
