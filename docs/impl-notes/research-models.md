# Research: non-Kokoro port targets

Status: **research v1** (2026-11-01). It covers MNIST MLP, ResNet-18, DCGAN 64x64 G/D, GPT-2 124M and MobileNetV2:
reference code, weights, op inventories, fusion plans, PyTorch baselines and porting order. Kokoro has its own
research doc. Kokoro appears here only in the kernel-union section (§7) so the library can be planned for reuse.

Provenance tags used below: **[user]** means a user rule from `original-plan.txt` / `design-decisions.md`.
**[measured]** means numbers produced on this Mac (M2 Max, 12 cores = 8P+4E, 32 GB, macOS 27) by the scripts in
`bench/research/models/` (`common.py`, `bench.py`, `inventory.py`, `mnist_check.py`). **[inferred]** means my
recommendation. It can be overridden.

---

## 0. Summary

| Model | Reference impl | Weights (pinned) | Params | MACs @ b1 |
|---|---|---|---|---|
| MNIST MLP 784-128-64-10 | own 6-line `nn.Sequential` (key layout below) | HF `Vecrist/mnist-pytorch-digit-classifiers` @ `574952b0` `MNIST_MLPmodel_weights.pth` (MIT). **97.44 %** MNIST test acc [measured] | 0.109 M | 0.11 M |
| ResNet-18 | `torchvision.models.resnet18` | `ResNet18_Weights.IMAGENET1K_V1` (`resnet18-f37072fd.pth`, top-1 69.758) | 11.69 M | 1.814 G |
| MobileNetV2 | `torchvision.models.mobilenet_v2` | `MobileNet_V2_Weights.IMAGENET1K_V2` (`mobilenet_v2-7ebf99e0.pth`, top-1 72.154) | 3.505 M | 0.301 G |
| DCGAN G (64x64) | PyTorch DCGAN tutorial / `pytorch/examples/dcgan` `Generator` (nz=100, ngf=64, nc=3) | HF `hussamalafandi/DCGAN_CelebA` @ `0c4e26b8` `generator.pth` (MIT, CelebA 50k subset, 50 epochs) | 3.577 M | 104.6 M |
| DCGAN D (64x64) | same, `Discriminator` (ndf=64) | same repo, `discriminator.pth` | 2.766 M | 103.8 M |
| GPT-2 124M | HF `transformers` `GPT2LMHeadModel` (sdpa attention) | HF `openai-community/gpt2` @ `607a30d7` (MIT) | 124.44 M (tied) | 123.5 M / token (84.9 M blocks + 38.6 M lm_head) |

Every target has pretrained weights with an exactly matching architecture, so per **[user]** "pretrained when
available", nothing needs training. The ≤5-min training fallbacks are documented anyway (§1.1, §4.1).

Baselines are in §8. Recommended porting order is in §9.

---

## 1. MNIST MLP (784 → 128 → 64 → 10)

### 1.1 Reference and weights
- Module (state-dict keys `layers.1/3/5.{weight,bias}` match this exactly):
  ```python
  class MLP(nn.Module):
      def __init__(self):
          super().__init__()
          self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
      def forward(self, x): return self.layers(x)
  ```
- Weights: `hf_hub_download("Vecrist/mnist-pytorch-digit-classifiers", "MNIST_MLPmodel_weights.pth",
  revision="574952b057f9e9858efdcddd1404a8c105cb5d7b")`, 441 KB, MIT. The model card gives no MLP accuracy.
  I measured it on the official MNIST test set **[measured]**:
  - with `Normalize((0.1307,), (0.3081,))` (the repo app's preprocessing): **97.44 %**
  - with raw [0,1] input: 96.14 %. So the expected input is the normalized one.
- Other hub candidates were rejected: `dacorvo/mnist-mlp` has hidden 256, and `Zolisa/pytorch-mnist-tutorial`
  is 784-128-10.
- **Fallback training recipe** (only if that repo disappears) **[measured]**: `torchvision.datasets.MNIST(download=True)`
  (≈11 MB, fetched from the `ossci-datasets` S3 mirror), the same normalization, Adam lr 1e-3, batch 128,
  CrossEntropy, CPU. Results: 1 epoch → 95.4 % in 0.6 s, 4 epochs → **97.4 % in 2.6 s**, 5 epochs → 97.2 % in 3.2 s.
  That is far under the 5-min budget. Seed it and commit the resulting `.pth` + test-acc to keep it reproducible.

### 1.2 Op inventory
| # | op | in | out | weight |
|---|---|---|---|---|
| 0 | flatten (free view) | [B,1,28,28] f32 | [B,784] | – |
| 1 | linear+bias+relu | [B,784] | [B,128] | W [128,784], b [128] |
| 2 | linear+bias+relu | [B,128] | [B,64] | W [64,128], b [64] |
| 3 | linear+bias | [B,64] | [B,10] | W [10,64], b [10] |

Inference simplifications: none needed (no dropout/BN). Optional exact trick: fold the input normalization
into layer 1 (`W' = W/σ`, `b' = b − (μ/σ)·W·1`). This is exact because there is no padding, unlike convs (see §2.3).
Keep normalization out of the graph in v1 so the reference and lokinfer inputs are identical.

### 1.3 Shapes / edge cases
- B ∈ {1, 7 (odd tail), 64, 1024}. B = 1 is pure GEMV and launch-overhead bound (0.025 ms in PyTorch CPU).
- **First dynamic-dim model**: `B` dynamic, everything else static. It is the ideal first test of
  `dynamic=True` + symbolic memory planning.
- Output: logits [B,10] compared with max-abs-err. Also run a whole-test-set argmax accuracy check (must equal 97.44 %).

### 1.4 Kernels
- CPU: `gemm_bias_relu_f32_cpu`, `gemm_bias_f32_cpu` (NEON micro-kernel and an Accelerate/BNNS variant). B=1 → GEMV
  path. Stretch goal: `mlp3_f32_cpu` (the whole net in one kernel). Weights are 436 KB and fit in L2.
- GPU: the same GEMM-epilogue kernels. At B=1 the GPU is purely dispatch-bound (PyTorch MPS 0.19 ms vs CPU
  0.025 ms). It is a good probe of lokinfer's per-dispatch and per-command-buffer overhead.

---

## 2. ResNet-18 (ImageNet)

### 2.1 Reference and weights
`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` (torchvision 0.29.2; top-1 69.758 / top-5 89.078).
The file is `resnet18-f37072fd.pth`, 44.7 MB, auto-downloaded to `~/.cache/torch/hub/checkpoints`. No dataset is needed:
correctness = compare against PyTorch outputs on random and real images. Preprocessing (outside the graph): resize 256,
center-crop 224, `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`.

### 2.2 Op inventory (b=1, 224x224, NCHW shapes as in PyTorch)
| stage | ops | in → out |
|---|---|---|
| stem | conv 7x7 s2 p3 3→64 + BN + ReLU | [1,3,224,224] → [1,64,112,112] |
| | maxpool 3x3 s2 p1 | → [1,64,56,56] |
| layer1 ×2 BasicBlock | conv3x3 64→64 +BN+ReLU, conv3x3 64→64 +BN, +identity, ReLU | 56x56 |
| layer2 block0 | conv3x3 s2 64→128 +BN+ReLU, conv3x3 128→128 +BN; downsample conv1x1 s2 64→128 +BN; add; ReLU | 56→28 |
| layer2 block1 | as layer1 at 128ch | 28x28 |
| layer3 | same pattern 128→256 | 28→14 |
| layer4 | same pattern 256→512 | 14→7 |
| head | adaptive avgpool → [1,512] → linear 512→1000 + bias | → [1,1000] |

Counts: **20 conv** (1× 7x7 s2 stem, 16× 3x3 [3 of them s2], 3× 1x1 s2 downsample), 20 BN, 17 ReLU,
8 residual adds, 1 maxpool, 1 global avgpool, 1 linear. All convs are `bias=False` followed by BN.
Typical batch: 8 (also run 1, 2, 32 for scaling).

### 2.3 Inference simplifications
- **Fold BN into conv**: `w' = w·γ/√(σ²+ε)`, `b' = β − μ·γ/√(σ²+ε)`, ε=1e-5. All 20 BNs disappear and every conv gets a bias.
  Do this in the Python model-building code (or as an init-hoisted kernel) **[inferred]**.
- Fusions: conv+bias+ReLU. Second conv of each block: conv+bias+residual-add+ReLU (epilogue reads the skip tensor).
  Downsample 1x1 s2 conv+bias.
- Do **not** fold the ImageNet mean/std into the stem conv. The stem pads with zeros in *normalized* space, so folding
  is inexact at the 3-pixel border. Normalization can instead be a fused prologue of the stem kernel.
- Global avgpool + FC can be one kernel (`avgpool_gemm_bias`), or avgpool can be an epilogue of the last conv.

### 2.4 Shapes / edge cases
- 224x224 b1/b8 (benchmark). Also **non-square/odd sizes** (e.g. 225x223, 160x160, 320x320). The net is fully
  convolutional up to the adaptive avgpool, so H/W can be dynamic dims. Output dims use
  `floordiv(H + 2p − k, s) + 1`, which exercises dim expressions. Odd sizes hit edge/padding tails in every conv tile.
- Channel counts are all multiples of 64 except the 3-channel stem input. The stem (Cin=3, k=7) needs its own kernel
  (im2col-ish or special-cased). A generic NHWC conv with Cin=3 wastes SIMD lanes.

### 2.5 Fused-kernel set
| kernel | CPU | GPU |
|---|---|---|
| `conv2d_k7s2_bias_relu` (stem, Cin=3) [+ maxpool 3x3s2 fused, optional] | im2col → GEMM (Accelerate) or direct NEON | direct conv, one threadgroup per output tile |
| `conv2d_k3_bias_relu` s1/s2 | implicit-GEMM NHWC, NEON 8x12 micro-kernel or AMX via Accelerate/BNNS; Winograd F(2x2,3x3) variant for s1 | implicit-GEMM with `simdgroup_matrix` 8x8, threadgroup tiling; Winograd variant |
| `conv2d_k3_bias_add_relu` (residual epilogue) | same + epilogue | same + epilogue |
| `conv2d_k1s2_bias` (downsample) | strided GEMM | strided GEMM |
| `maxpool2d_k3s2p1` | NEON | simple |
| `avgpool_gemm_bias` (head) | GEMV | GEMV |

Layout **[inferred]**: NHWC (channels-last) internally on both devices. 1x1 conv == plain GEMM. Implicit-GEMM K
dimension = k·k·Cin is contiguous. This is what MPS/MLX/XNNPACK use. In PyTorch CPU, channels_last also helps (§8).
Accept NCHW input and fuse the NCHW→NHWC transpose into the stem kernel.

---

## 3. MobileNetV2

### 3.1 Reference and weights
`torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)` → `mobilenet_v2-7ebf99e0.pth`
(13.6 MB, top-1 72.154; V1 = 71.878, same arch). Same preprocessing as ResNet (V2 weights use resize 232).

### 3.2 Op inventory (b=1, 224x224)
- Stem: conv 3x3 s2 3→32 + BN + ReLU6 → [32,112,112].
- 17 inverted-residual blocks (t, c, n, s) = (1,16,1,1), (6,24,2,2), (6,32,3,2), (6,64,4,2), (6,96,3,1), (6,160,3,2), (6,320,1,1):
  - block 0 (t=1): dw 3x3 32ch + BN + ReLU6 → pw 1x1 32→16 + BN (no activation: linear bottleneck)
  - blocks 1–16: pw expand 1x1 Cin→6Cin + BN + ReLU6 → dw 3x3 s{1,2} + BN + ReLU6 → pw project 1x1 6Cin→Cout + BN,
    and `+ x` when s=1 and Cin==Cout (**10 residual adds**)
- Head: conv 1x1 320→1280 + BN + ReLU6 → global avgpool → dropout(0.2, removed) → linear 1280→1000.
- Counts **[measured]**: **52 conv** (1 stem, 17 depthwise 3x3 [4 of them s2], 34 pointwise), 52 BN, 35 ReLU6,
  10 adds, 1 avgpool, 1 linear. 3.505 M params, 300.8 M MACs.
- Expanded widths: 96, 144, 192, 384, 576, 960. Spatial: 112 → 56 → 28 → 14 → 7.

### 3.3 Simplifications
BN fold into every conv (all `bias=False`). Dropout removed. ReLU6 = clamp(0,6) in the epilogue.

### 3.4 Shapes / edge cases
- Channel counts **not multiples of 32/64**: 3, 16, 24, 96, 144, 160, 320, 576, 960. These exercise GEMM/conv N- and
  K-tails (24 and 144 aren't even multiples of 16).
- Many tiny, memory-bound layers (14x14 and 7x7 with ≤1k channels). Per-op overhead dominates. This makes
  MobileNetV2 the **runtime-overhead stress test**: 52 conv dispatches ≈ 70 kernels after fusion vs ResNet's ≈ 22.
- Same dynamic H/W story as ResNet.

### 3.5 Fused-kernel set
| kernel | notes |
|---|---|
| `conv2d_k3s2_bias_relu6` (stem, Cin=3) | same family as ResNet stem |
| `pwconv_bias_relu6` (expand) | = GEMM [HW, Cin]×[Cin, 6Cin] with epilogue. Reuses the GEMM family |
| `dwconv3x3_bias_relu6` s1/s2 | own kernel: NEON per-channel FMA on CPU, one thread per (pixel, 4ch) on GPU. Memory-bound |
| `pwconv_bias[_add]` (project, optional residual epilogue) | GEMM + residual epilogue (shared with ResNet's `_add`) |
| **`inverted_residual_*` fused block** (stretch, GPU first) | expand→dw→project in one kernel. The 6×-wide intermediate stays in threadgroup memory / registers, which removes the dominant memory traffic. Published fused dw+pw kernels report up to 3.7× over cuDNN on the fused modules ([FCM paper](https://arxiv.org/abs/2404.19331)). On CPU: row-band tiling so the expanded tile stays in L1/L2 |
| `avgpool_gemm_bias` | shared with ResNet |

---

## 4. DCGAN 64x64 — Generator and Discriminator

### 4.1 Reference and weights
- Architecture = PyTorch DCGAN tutorial / `pytorch/examples/dcgan` (`nz=100, ngf=ndf=64, nc=3`). `pytorch/examples` ships **no**
  checkpoints. Soumith's original `dcgan.torch` CelebA nets are Lua `.t7` files.
- **Use HF `hussamalafandi/DCGAN_CelebA`** (MIT, rev `0c4e26b8498c0ccf1ff06f47e0810ff4aaf8e14a`): `generator.pth`
  (14.3 MB) + `discriminator.pth` (11.1 MB) + `dcgan.py`. I verified that the state-dict keys `main.{0,1,3,4,...,12}` and shapes load
  **strictly** into the tutorial classes (`Generator`/`Discriminator` with `self.main = nn.Sequential(...)`) **[measured]**.
  It was trained on a 50k CelebA subset for 50 epochs.
- Alternates with the same key layout: `arzumanabbasov/DCGAN_CELEBA` (MIT, 25 epochs, bf16 AMP training). `huggan/dcgan-mnist` /
  `huggan/dcgan-celeba-faces` hold the generator only (keys `model.*`). `nateraw/cryptopunks-gan` uses a different arch (3x3 kernels, 4 channels).
- Fallback if all disappear **[inferred, not measured]**: training to visually good faces in ≤5 min isn't realistic
  (CelebA is 1.4 GB from Google Drive, which is often quota-blocked). Use random init (DCGAN `weights_init`: conv N(0,0.02), BN γ N(1,0.02), β 0)
  and set **non-trivial BN running stats** (random μ, σ² ∈ [0.5,2]) so BN folding is actually tested.
- Correctness check of semantics: D(G(z)) should be well above 0 on most z (trained pair). This gives a cheap end-to-end sanity check
  that composes both ported graphs.

### 4.2 Generator op inventory (b=1; typical b=64)
| # | op | in → out | weight |
|---|---|---|---|
| 1 | convT 4x4 s1 p0 100→512 +BN+ReLU | [1,100,1,1] → [1,512,4,4] | [100,512,4,4] |
| 2 | convT 4x4 s2 p1 512→256 +BN+ReLU | → [1,256,8,8] | [512,256,4,4] |
| 3 | convT 4x4 s2 p1 256→128 +BN+ReLU | → [1,128,16,16] | |
| 4 | convT 4x4 s2 p1 128→64 +BN+ReLU | → [1,64,32,32] | |
| 5 | convT 4x4 s2 p1 64→3 + tanh | → [1,3,64,64] | [64,3,4,4] |

5 convT, 4 BN, 4 ReLU, 1 tanh. Simplifications:
- Fold BN into convT (scale per *output* channel = dim 1 of the convT weight).
- Layer 1 on a 1x1 input **is a GEMM**: [B,100] × [100, 512·16] (+bias, ReLU), then reshape to 4x4. No conv kernel is needed.
- Layers 2–5 (k=4, s=2, p=1) decompose exactly into **4 sub-pixel (phase) 2x2 convolutions** writing interleaved outputs
  (depth-to-space). This reuses the regular conv kernel family and avoids scatter-add or zero-insertion. The same trick
  serves Kokoro's `ConvTranspose1d` (upsample rates 10 and 6, plus the depthwise k3 s2 one in `AdainResBlk1d`) in 1-D.

### 4.3 Discriminator op inventory (b=1; typical b=64)
| # | op | in → out |
|---|---|---|
| 1 | conv 4x4 s2 p1 3→64 + LeakyReLU(0.2) | [1,3,64,64] → [1,64,32,32] |
| 2 | conv 4x4 s2 p1 64→128 +BN+Leaky | → [1,128,16,16] |
| 3 | conv 4x4 s2 p1 128→256 +BN+Leaky | → [1,256,8,8] |
| 4 | conv 4x4 s2 p1 256→512 +BN+Leaky | → [1,512,4,4] |
| 5 | conv 4x4 s1 p0 512→1 + sigmoid | → [1,1,1,1] → view → [1] |

5 conv, 3 BN, 4 LeakyReLU, 1 sigmoid. The last layer is a **dot product of length 8192** per sample
(GEMV [B,8192]×[8192,1]) and needs no conv kernel. Fold BN. The final `.view(-1,1).squeeze(1)` is a free view.

### 4.4 Edge cases
- k=4 even kernels with s=2 p=1 (unlike the k=3 of ResNet/MobileNet). The Cin=3 first layer. Cout=1 last layer.
- Batch: 1, 64 (tutorial size), 128. Dynamic B again.
- **PyTorch CPU pathology [measured]**: the D at b=64 takes ~0.9–1.6 s on default eager CPU. torch 2.14 on arm64 macOS has no
  oneDNN (`mkldnn.is_available() == False`), so these 4x4 s2 convs fall to `_nnpack_spatial_convolution`. With
  `torch.backends.nnpack.flags(enabled=False)` + channels_last it drops to ~60 ms. The baseline table reports **both**:
  "default eager" and "tuned eager". Beating only the pathological number would be meaningless.

### 4.5 Fused-kernel set
- G: `gemm_bias_relu` (layer 1, shared with MLP); `convT2d_k4s2p1_bias_relu` and `_bias_tanh` (implemented as 4-phase 2x2
  implicit-GEMM conv with a depth-to-space store), NHWC.
- D: `conv2d_k4s2p1_bias_leakyrelu` (Cin=3 special case for layer 1); `gemv_sigmoid` (or `gemm_bias_sigmoid` with N=1).

---

## 5. GPT-2 124M

### 5.1 Reference and weights
HF `transformers` (5.22.1) `GPT2LMHeadModel.from_pretrained("openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
attn_implementation="sdpa")` (MIT, `model.safetensors` 548 MB f32). The tokenizer (`GPT2TokenizerFast`) is used in tests
to make realistic ids only. It is outside the graph, like Kokoro's text frontend.

Config: n_layer 12, n_embd 768, n_head 12 (**head_dim 64**), n_inner 3072, n_positions 1024, vocab 50257, LN ε 1e-5,
activation `gelu_new` (tanh approximation), `tie_word_embeddings=True`, attention scale 1/√64, causal.

### 5.2 Op inventory (B=1, sequence T; past length P for decode)
| op (per block ×12) | shapes | notes |
|---|---|---|
| embed: `wte[ids] + wpe[pos]` | ids [B,T] **i64/i32** → [B,T,768] | once, before block 0; wte [50257,768], wpe [1024,768] |
| ln_1 | [B,T,768] | LayerNorm with affine |
| c_attn (Conv1D) | [B·T,768]×[768,2304]+b | HF `Conv1D` stores W as **[in,out]** (x@W), i.e. transposed vs `nn.Linear`. Choose the layout at export |
| split q,k,v + heads | → 3×[B,12,T,64] | pure layout. Fuse into the QKV GEMM epilogue (write head-major, and write K/V straight into the KV cache) |
| causal SDPA | q[B,12,T,64], k/v[B,12,P+T,64] | scale 0.125, causal mask, softmax, ·V |
| merge heads | → [B,T,768] | fuse into attention output store |
| c_proj + residual | [B·T,768]×[768,768]+b + x | GEMM epilogue = bias + residual add |
| ln_2 | | |
| c_fc + gelu_new | [B·T,768]×[768,3072]+b, gelu-tanh | GEMM epilogue |
| c_proj + residual | [B·T,3072]×[3072,768]+b + x | |
| ln_f (once) | | |
| lm_head (once) | [B·T',768]×wteᵀ [768,50257], no bias | **tied**: same constant as the embedding table. T'=1 for generation (`logits_to_keep=1`) |

Per block: 4 GEMMs, 2 LN, 1 attention, 2 residual adds, 1 GELU. Totals: 48 GEMMs + lm_head, 25 LayerNorms, 12 attentions.
Simplifications: all dropouts (embd/attn/resid, p=0.1) removed. **lm_head reuses `wte`** (the IR must allow one external
constant to feed two kernels with different access patterns. No duplicate 154 MB copy). The `transformer.h.*.attn.bias`
causal-mask buffers are not needed (the attention kernel generates the mask).

Weights: 124.4 M × 4 B = **498 MB f32** (249 MB f16). One decode token at B=1 must stream all of them.
M2 Max ≈ 400 GB/s → **roofline ≈ 1.25 ms/token f32, ≈ 0.62 ms f16** **[inferred]**. Prefill at large T is compute-bound
(≈ 2·124 M·T FLOP + attention).

### 5.3 KV cache, scope, data-dependent shapes
Recommendation **[inferred]**: do it in two steps, both in scope for "Port GPT-2".
1. **Prefill graph** (= the PyTorch forward with `use_cache=False`): input ids [B, T] with **T dynamic (1..1024)**,
   output last-position logits [B, 50257] (plus an "all logits" variant for correctness against HF). Data-dependent shapes: none.
   T is an input dim. This is the first model with a dynamic *sequence* dim, an integer-dtype input, and attention whose
   cost depends on T².
2. **Decode-step graph with a static-capacity KV cache**: inputs `ids [B,1]`, `pos` (i32 scalar), `kcache, vcache
   [12, B, 12, 1024, 64]` (f32 or f16). The QKV kernel writes the new K/V row at `pos` **in place**. Attention reads rows
   `0..pos`. Output logits [B, 50257] (+ optional on-device `argmax` → i32 next id). Cache length is a *runtime scalar*, not a shape,
   so there are **no data-dependent shapes** and the memory plan is fully static (init-hoistable, GPU record/replay friendly).
   The loop (feed argmax back, `pos += 1`) runs in host C++/Python. A greedy-decode output must match HF `generate(do_sample=False)`
   token-for-token for ~64 tokens.
   - **IR requirement this surfaces**: graph inputs that kernels mutate in place (`inout`/donated buffers), so the cache persists across
     calls with no copy. Alternatively the cache could be model-owned state (an init-allocated buffer that persists across exec calls).
     Pick one in `design-ir.md`. I lean towards *model-owned persistent state buffers*. They also suit future streaming use.
   - Rejected alternative: a dynamic past-length dim P with concat-per-step. That costs an O(P) copy per token and turns every step into
     a dynamic-shape call with a fresh memory plan.
3. Prefill should also fill the cache (prefill graph variant with cache outputs) so generation = 1 prefill + N decode steps.

Edge cases: T ∈ {1, 7, 8, 32, 128, 511, 512, 1024}. T=1 prefill vs decode with P=0 must agree. P ∈ {0, 16, 128, 512, 1023}.
B ∈ {1, 4}. The max position is 1024 (beyond that, wpe is out of range: assert). The softmax needs the causal mask on the diagonal tile only.
Vocab 50257 is prime-ish/odd, so test lm_head N-tails. f16 variant: logits drift, so compare top-1 / top-5 agreement + tolerance.

### 5.4 Fused-kernel set
| kernel | CPU | GPU |
|---|---|---|
| `embed_gather_add` (wte[ids] + wpe[pos]) | memcpy rows | 1 thread per (token, 4 floats) |
| `layernorm` and **`add_layernorm`** (residual add + LN, outputs both the new residual and the normed value) | NEON two-pass/Welford | simdgroup reduction, 1 threadgroup per row |
| `gemm_bias_qkvsplit[_kvwrite]` | GEMM + head-major scatter epilogue | same |
| `gemm_bias_gelu_tanh` | | |
| `gemm_bias_residual` | | |
| `attention_causal_prefill_d64` | blocked (flash-style) online softmax, NEON | **flash attention** tiled, d=64 (MLX `steel_attention` design) |
| `attention_decode_d64` (q len 1) | GEMV-ish | split-K "`sdpa_vector`" style (MLX has single-pass and 2-pass split-KV variants) |
| `gemm_lmhead_tied` (x × wteᵀ, no bias) | GEMV at decode (bandwidth) | GEMV at decode |
| `argmax` (optional, i32 output) | | simdgroup reduce |

For decode at B=1 every GEMM is a GEMV and is bandwidth-bound. Reaching the roofline needs the f16 (or int8 weight-only) variants. They
make GPT-2 the natural first **f16/bf16** model. Benchmark f32 vs PyTorch first, then f16 vs PyTorch f16 / MLX.

---

## 6. Per-model "what's new" for the framework

| model | new dims/shapes | new dtypes | new op classes | new runtime/IR features |
|---|---|---|---|---|
| MNIST MLP | dynamic B | f32 | GEMM + epilogues, GEMV | dynamic batch, symbolic mem planning, CPU-vs-GPU dispatch-overhead probe |
| DCGAN D | 4-D NCHW/NHWC tensors, conv output-size dim exprs | f32 | conv2d k4s2, LeakyReLU, sigmoid, BN fold | layout transform (NCHW in → NHWC), init-time weight transforms (BN fold, layout repack) via init hoisting |
| DCGAN G | depth-to-space strides | f32 | convT (4-phase), tanh | reshape-as-view after GEMM |
| ResNet-18 | dynamic H/W (`floordiv` expr chains), odd sizes | f32 (f16 variant) | 3x3/7x7/1x1-strided conv, maxpool, avgpool, residual epilogue | multi-consumer tensors (skip connection lifetime in mem planner), real-sized weights (45 MB) |
| MobileNetV2 | non-multiple-of-16 channels | f32 | depthwise conv, ReLU6, fused inverted residual | many tiny ops → per-dispatch overhead, GPU record/replay payoff |
| GPT-2 | dynamic T, T² attention, runtime scalar `pos` | **i32/i64 ids**, f32, **f16** | gather, layernorm, attention (prefill + decode), GELU-tanh, argmax | tied/shared constants, 500 MB mmapped weights, persistent/in-place state (KV cache), scalar inputs, decode loop |
| Kokoro | data-dependent length (durations → frames) | f32 (+ i64 ids) | LSTM, conv1d dilated, convT1d, AdaIN, snake, iSTFT, cumsum/sin | data-dependent dims (`dim.read`), CPU↔GPU mixing |

---

## 7. Kernel union (all models + Kokoro) — plan for reuse

Kokoro recap (from `kokoro==0.7.16` source + `config.json`): ALBERT (hidden 768, 12 heads → **head_dim 64**, intermediate 2048,
12 layers sharing one set of weights, embedding 128→768 projection, gelu_new), text encoder (embedding 178 → conv1d k5 + channel
LayerNorm + LeakyReLU(0.2) ×3, BiLSTM 512), prosody predictor (BiLSTMs, AdaLayerNorm, AdainResBlk1d with depthwise ConvT1d k3 s2
upsample, conv1d proj), duration (sigmoid → sum → round → `repeat_interleave` alignment: **data-dependent frames**), decoder/iSTFTNet
(AdaIN1d = InstanceNorm1d + style Linear, Snake1D `x + 1/α·sin²(αx)`, conv1d k3/7/11 dilations 1/3/5, ConvTranspose1d k20 s10 and
k12 s6, SineGen harmonic source: linear interpolate + cumsum + sin, Linear+tanh, reflection pad, exp/sin magnitude-phase,
iSTFT n_fft 20 hop 5). Weight-norm folds at load.

| # | kernel family (templated over epilogue/dtype) | MLP | D | G | R18 | MV2 | GPT-2 | Kokoro | priority |
|---|---|---|---|---|---|---|---|---|---|
| K1 | **GEMM** (+ epilogues: none, bias, bias_relu, bias_relu6, bias_gelu_tanh, bias_residual, bias_tanh, bias_sigmoid, bias_leaky) — CPU NEON/AMX(Accelerate) + GPU simdgroup_matrix + MPS variant | ✓ | ✓(last) | ✓(L1) | ✓(fc, 1x1) | ✓(34 pw + fc) | ✓ | ✓ (ALBERT, Linear, LSTM input proj, AdaIN fc, 1x1 conv1d) | **P0** |
| K2 | **GEMV** (M=1 specialization of K1) | ✓ b1 | ✓ | ✓ | | | ✓ decode | ✓ LSTM recurrent, style fc | P0 |
| K3 | **implicit-GEMM conv (NHWC, 1-D/2-D, stride, dilation, padding)** + K1 epilogues | | ✓ k4s2 | ✓ (via phases) | ✓ k3/k7 | ✓ stem | | ✓ conv1d k3/5/7/11, dilation 1/3/5 | **P0** |
| K4 | transposed conv as **phase-decomposed** K3 + depth-to-space store | | | ✓ k4s2 | | | | ✓ convT1d k20s10, k12s6 | P1 |
| K5 | depthwise conv (2-D 3x3 s1/s2; 1-D convT k3s2 depthwise) + epilogue | | | | | ✓ | | ✓ (AdainResBlk1d pool) | P1 |
| K6 | fused inverted residual (pw→dw→pw) | | | | | ✓ | | | P2 (stretch) |
| K7 | pooling: maxpool 3x3s2, global avgpool (+FC fusion) | | | | ✓ | ✓ | | | P1 |
| K8 | **layernorm / add_layernorm** (row) | | | | | | ✓ | ✓ ALBERT, text-encoder channel LN, AdaLayerNorm (style-modulated variant) | **P0** |
| K9 | instance-norm / **AdaIN1d** (norm over time per channel + style affine) | | | | | | | ✓ | P0 (Kokoro) |
| K10 | **attention d=64**: non-causal + padding mask (ALBERT), causal prefill (GPT-2), decode vector (GPT-2) | | | | | | ✓ | ✓ | **P0** |
| K11 | embedding gather (+ add) | | | | | | ✓ | ✓ | P1 |
| K12 | **LSTM bidirectional** (input proj by K1, recurrent fused gates step) | | | | | | | ✓ | P0 (Kokoro) |
| K13 | elementwise / activations standalone (relu, relu6, leaky, tanh, sigmoid, gelu_tanh, snake, exp, sin) — mostly epilogues, standalone for tests and leftovers | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ snake, AdaIN | P0 |
| K14 | reductions / scans: argmax, softmax, sum, cumsum | | | | | | ✓ argmax | ✓ cumsum, duration sum | P1 |
| K15 | layout: NCHW↔NHWC, transpose, concat, pad (zero/reflection), interpolate (nearest/linear), repeat_interleave/alignment | | ✓ | ✓ | ✓ | ✓ | | ✓ | P1 |
| K16 | signal: harmonic sine source, iSTFT (n_fft 20) | | | | | | | ✓ | P0 (Kokoro) |

Reuse takeaways **[inferred]**:
1. **One templated GEMM with a pluggable epilogue (+ optional residual input) is the core.** In NHWC, 1x1 conv,
   linear, the DCGAN-G first layer, the DCGAN-D last layer and the LSTM input projections are all literally K1. Conv (K3) is K1 with an
   im2col-on-the-fly prologue and the *same* epilogue set. Write epilogues once, instantiate per (dtype, device).
2. **Attention head_dim 64 everywhere** (GPT-2 and Kokoro's ALBERT). Build one d=64 attention family with mask modes {none/padding,
   causal} and query-length modes {tiled prefill, single-query decode}.
3. **Phase-decomposed transposed conv** serves both DCGAN-G (2-D) and Kokoro (1-D). Porting DCGAN-G de-risks Kokoro's upsampler.
4. LayerNorm (GPT-2) → ALBERT LN and channel LN → AdaLayerNorm/AdaIN are one row-reduction family with different affine sources.
5. Conv-net-only kernels (maxpool, depthwise 2-D, fused inverted residual) aren't reused by Kokoro. Keep them simple
   unless a perf goal requires otherwise.

---

## 8. Baselines — reference frameworks, f32, this Mac **[measured]**

**[built]** Apples-to-apples baselines measured 2026-11-01 (M2 Max, Low Power Mode off, AC power, thermal nominal) by
`bench/models/bench_other_baselines.py`, recorded in `perf/results.json` (dashboard: `docs/perf-dashboard.md`).
Every framework runs the same weights and the same inputs: torchvision `ResNet18_Weights.IMAGENET1K_V1` /
`MobileNet_V2_Weights.IMAGENET1K_V2`, HF `hussamalafandi/DCGAN_CelebA@0c4e26b8`, HF `openai-community/gpt2@607a30d7`
(sdpa); inputs from `np.random.default_rng(0)` (`standard_normal` image / z, GPT-2 `integers(0, 50257)`); f32, eval,
no grad. Cases: ResNet-18 / MobileNetV2 `N=1 224x224`, `N=8 224x224` ([N,3,224,224] → logits [N,1000]); DCGAN G
`N=1`, `N=64` (z [N,100,1,1] → [N,3,64,64]); DCGAN D `N=1`, `N=64` ([N,3,64,64] → [N]); GPT-2 `prefill T=128`,
`prefill T=512` (one forward, `use_cache=False`, last-position logits [1,50257]) and `decode P=512` (one token at
position 512 with a 512-token KV cache → logits; cache built untimed).

Method: 3 warm-ups, then ≥ 10 runs and ≥ 2 s (≤ 1000 runs), **median** per-call ms; the GPU is synced inside every
timed call (`torch.mps.synchronize()` / `mx.eval`); inputs/outputs stay on the device. Every output is verified
against torch eager CPU, rel err = max|y − ref| / max|ref| ≤ 1e-3, or the entry is not recorded (all measured
entries: ≤ 5.3e-6). DCGAN D's sigmoid saturates to exactly 1.0 on these inputs, so D is verified on its
pre-sigmoid logits (same impl, final Sigmoid → Identity). Impls: `torch eager` (torch 2.14.1, 8 threads),
`torch eager tuned` (NNPACK off + channels_last, §4.4; not for MobileNetV2, where it makes the depthwise convs 3-8×
slower, nor GPT-2), `torch.compile` (default mode), `torch MPS`, `torch.compile MPS`, `MLX` (mlx 0.32.4, own exact
transcriptions in `bench/models/runners/other_mlx.py`, unfused eval BatchNorm, NCHW↔NHWC transposes inside the
timed call; the faster of plain and `mx.compile` is recorded, which was `mx.compile` in every case). ONNX Runtime
was not measured (not installed; needs a separate venv + export toolchain).

ms per forward, lower is better, best per device in bold:

| model | case | CPU torch eager | CPU torch eager tuned | CPU torch.compile | GPU torch MPS | GPU torch.compile MPS | GPU MLX |
|---|---|---|---|---|---|---|---|
| ResNet-18 | N=1 224x224 | **8.48** | 8.56 | 10.33 | 5.00 | 4.68 | **2.92** |
| ResNet-18 | N=8 224x224 | 45.26 | 41.47 | **41.44** | 9.71 | **6.85** | 7.44 |
| MobileNetV2 | N=1 224x224 | **45.23** | – | 279.51 | 8.59 | 3.68 | **1.76** |
| MobileNetV2 | N=8 224x224 | 126.53 | – | **95.74** | 11.42 | 5.98 | **4.50** |
| DCGAN G | N=1 | 1.31 | **1.12** | 2.03 | 1.61 | 2.03 | **0.59** |
| DCGAN G | N=64 | 15.28 | **12.25** | 15.49 | 12.02 | 11.20 | **5.63** |
| DCGAN D | N=1 | 1.22 | **1.08** | 1.86 | 1.19 | 1.28 | **1.10** |
| DCGAN D | N=64 | 429.33 | **18.74** | 429.79 | 4.09 | 3.70 | **3.23** |
| GPT-2 124M | prefill T=128 | 47.19 | – | **45.02** | 9.67 | 7.60 | **5.65** |
| GPT-2 124M | prefill T=512 | 110.61 | – | **105.40** | 21.22 | **18.70** | 19.73 |
| GPT-2 124M | decode P=512 | **19.77** | – | 20.03 | 5.40 | **3.21** | 4.14 |

Caveats: other sessions were benchmarking concurrently; the harness waits (≤ 2 min) for other processes < 200 % CPU
before each measurement and records the load it saw in each entry's notes. CPU rows varied up to ~±15 % between
repeated runs (e.g. MobileNetV2 eager N=8: 95–127 ms); GPU rows within ~±5 %. Treat differences under ~15 % on CPU
as noise. Anomalies kept as measured: torch.compile CPU on MobileNetV2 N=1 (≈ 280–320 ms, slower than N=8) and on
DCGAN D N=64 (still on the NNPACK path, ≈ 430 ms).

Rerun (overwrites only the entries it measures; ~5 min for everything, longer when the machine is busy):

```sh
.venv/bin/python bench/models/bench_other_baselines.py                      # all models, all impls
.venv/bin/python bench/models/bench_other_baselines.py --model gpt2 --impl MLX
.venv/bin/python bench/models/bench_other_baselines.py --model resnet18 --no-record   # print only
```

MLX runs as a subprocess in its own venv (mlx + numpy): set `mlx_python` in `lokinfer.local.json` or
`LOKINFER_MLX_PYTHON` (see [benchmarking.md § Local configuration](../benchmarking.md#local-configuration)).

---

## 9. Recommended porting order

| # | model | why here / what it de-risks |
|---|---|---|
| 1 | **MNIST MLP** | Smallest real model. GEMM + epilogue family (K1/K2), weight loading from `.pth`, dynamic batch, CPU vs GPU dispatch overhead (B=1: PyTorch CPU 0.025 ms vs MPS 0.19 ms is the bar for lokinfer's runtime overhead) |
| 2 | **DCGAN Discriminator** | First conv (K3) on tiny tensors, NHWC layout + init-time weight transforms (BN fold, repack), LeakyReLU/sigmoid epilogues, GEMV tail |
| 3 | **DCGAN Generator** | Transposed conv via phase decomposition (K4), the direct precursor of Kokoro's ConvTranspose1d. D(G(z)) sanity check composes both |
| 4 | **ResNet-18** | Real-size conv net: 3x3/7x7/strided convs, residual epilogue, pooling, multi-consumer lifetimes in the memory planner, dynamic H/W dim expressions, first heavy compute benchmark |
| 5 | **MobileNetV2** | Depthwise + odd channel counts (tails), many small ops → stresses runtime overhead and GPU record/replay. Fused inverted residual as an optimization stretch |
| 6 | **GPT-2** | Transformer stack: gather, LayerNorm, d=64 attention (prefill + decode), GELU, int inputs, tied 500 MB weights, persistent KV state, f16. Builds most of what Kokoro's ALBERT needs |
| 7 | Kokoro | — |

Alternative **[inferred]**: if Kokoro is urgent, move GPT-2 (prefill-only first) ahead of ResNet/MobileNetV2. Kokoro needs K1/K3/K4/K8/K10
and none of the 2-D conv-net kernels (maxpool, depthwise 2-D, inverted residual). The minimum Kokoro-critical path is MLP → DCGAN-D → DCGAN-G
→ GPT-2 prefill. ResNet-18 / MobileNetV2 are then pure conv-net goals.

---

## Sources
- MNIST MLP weights: https://huggingface.co/Vecrist/mnist-pytorch-digit-classifiers
- DCGAN weights: https://huggingface.co/hussamalafandi/DCGAN_CelebA ; alternates https://huggingface.co/arzumanabbasov/DCGAN_CELEBA ,
  https://huggingface.co/huggan/dcgan-mnist ; original Lua nets https://github.com/soumith/dcgan.torch
- torchvision model zoo weights: `torchvision.models.{ResNet18_Weights, MobileNet_V2_Weights}` (docs.pytorch.org/vision)
- GPT-2: https://huggingface.co/openai-community/gpt2
- Fused depthwise+pointwise on GPUs: https://arxiv.org/abs/2404.19331
- MLX attention kernels (sdpa_vector decode / steel_attention prefill, d=64 support), as described in the tract-metal port:
  https://docs.rs/crate/tract-metal/0.23.6/source/src/kernels/matmul/mlx_sdpa/mod.rs ; mlx-mfa: https://pypi.org/project/mlx-mfa/
