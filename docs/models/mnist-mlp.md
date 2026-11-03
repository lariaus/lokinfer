# MNIST MLP on lokinfer

A 784 → 128 → 64 → 10 multilayer perceptron (ReLU between layers), the smallest real model of the plan. It exercises
GEMM with fused bias/ReLU epilogues, dynamic batch, weight loading from a `.pth` and init-time weight preparation.

- Reference and weights: `models/mnist_mlp.py` (`torch_reference()`), pretrained
  `Vecrist/mnist-pytorch-digit-classifiers` @ `574952b0` (MIT). Input normalized with mean 0.1307, std 0.3081.
- Graph: `flatten` (a free view) → `gemm_bias_relu` → `gemm_bias_relu` → `gemm_bias`. PyTorch's `[out, in]` weights
  are transposed once by a kernel on the model's device, which the compiler hoists into init; per call only the
  three GEMMs run. The batch `B` is a symbol.
- Variants: `cpu` (hand-written NEON tiles), `cpu-accelerate` (Accelerate `sgemm` + NEON epilogue), `gpu` (GPU end to
  end: weights resident on the GPU, `transpose_f32_gpu` in init, simdgroup-matrix GEMMs; dispatch below).
- Correctness (`tests/models/test_mnist_mlp.py`): every variant matches PyTorch (B = 1, 7, 64, 1024) and reproduces the
  97.44 % test-set accuracy exactly.
- Performance (`bench/models/bench_mnist.py`; the first numbers were Low Power Mode **on**: Accelerate beat PyTorch
  CPU at every batch, 0.004 vs 0.027 ms at B = 1, 0.43 vs 0.77 ms at B = 1024; NEON won at B = 1). GPU since
  2026-11-01 (LPM off): split-K small-M kernel for M ≤ 8 + steel above (iPhone: 32×32 tiles above M = 256) — Mac GPU
  0.205 / 0.257 / 0.152 ms at B = 1 / 64 / 1024 vs torch MPS 0.306 / 0.430 / 0.390 (dashboard). iPhone 18 Pro Max via the remote executor:
  Accelerate B = 1 in 0.0016 ms.
- iPhone CPU `cpu-accelerate` tuning (2026-11-01, iPhone 18 Pro Max, nominal; dashboard 1.5 / 11.2 / 161 us at
  B = 1 / 64 / 1024 vs ExecuTorch XNNPACK 1.7 / 12.1 / 173 us), **measured and dropped**: single-threaded Accelerate
  (1.8 / 11.8 / 164 us, a wash), 2 row or column blocks on the pool (B = 64: 53 us: a pool task on an E-core
  dominates; B = 1024: 178-208 us), a pool epilogue after the multithreaded GEMM (B = 1024: 292 us; now only gelu
  epilogues go to the pool), and a fused 3-layer kernel keeping the hidden activations in scratch (1.7 / 12.8 / 173
  us). The model sits at Accelerate's SME limit (~1.4 TF/s at B = 1024); a further gain needs a different GEMM
  engine, not blocking.
- **iPhone GPU vs MLX Swift (2026-11-02, iPhone 18 Pro Max, nominal; dashboard ms): B = 1 0.152 → 0.138 (MLX Swift
  0.139), B = 64 0.189 → 0.154 (0.144, still behind), B = 1024 0.41-0.68 → 0.257 (0.264).** Mac GPU unchanged or
  faster (prof-lokinfer e2e B = 1 0.195 → 0.152-0.163, B = 64 0.218 → 0.192-0.203, B = 1024 0.37 = same kernel).
  `gemm_bias[_relu]_f32_gpu` dispatch (`kernels/gpu/kokoro_lc.cpp`, kernels in `gemm.metal`):
  - M ≤ 8: the small-M kernel now also splits K over grid z; M 9-256: steel tiles (32×32×16) with split-K. Both in
    **one dispatch**: every split stores its partial tile in a per-op scratch, the last split of a tile to arrive
    (device-scope atomic counter, reset by it, as in `attention_decode_split`) sums the partials and applies bias /
    ReLU. Splits target ~32 threadgroups on the iPhone (784→128 at B = 1: 4 → 28 threadgroups, 0.054 → 0.031 ms for
    the two ReLU layers; B = 64: 8 → 32), ~64 on the Mac; when nothing splits, the plain steel kernel runs as before.
  - M > 256: iPhone = a 32×32 BK = 32 tiled kernel with float4 loads (`gemm_bias[_relu]_k32_f32`; B = 1024
    0.248-0.269 ms vs 0.273 for the old BK = 16 tiled kernel, 0.258-0.324 steel); Mac = steel (0.37 vs 0.48 / 0.66).
  - B = 1024 was mostly a harness cost: the remote executor copied the 3.2 MB input on every timed call (Mac callers
    and the MLX Swift harness don't); it now copies once per RUN (`docs/runtime.md`), which moved B = 1024 from
    ~0.49 to ~0.27 ms by itself.
  - Env knobs for A/B (read at prepare): `LOKINFER_GEMMB_LEGACY=1` (the 2026-11-01 dispatch), `_TGS`, `_SKMAX`,
    `_MID` / `_BIG` (tile config), `_MINKC`, `_SMALLM`. Tests: `tests/kernels/test_gemm_bias_gpu.py` (every config,
    forced many-split runs, repeated calls with varying M for the counter reset).
  - **Measured and dropped** (interleaved A/B on the phone): split targets 16 / 64 / 128 (B = 64 0.159-0.171 vs 0.153);
    64×32, 32×64, 16×32, 16×64 and BK = 32 split tiles at B = 64 (0.154-0.197); the small-M kernel up to M = 64
    (0.287); steel 64×32×32 (MLX's iPhone tile), 64×32×16 and 32×32 split-capable tiles at B = 1024 (0.29-0.33); MPS
    for the three GEMMs at B = 1024 (0.44 ms without the epilogues); **a fused single-dispatch 3-layer MLP** (split-K
    layer 1, the last split of a row block runs layers 2-3, layer-2/3 weights preloaded into registers): slower on
    both devices (iPhone B = 1 0.155 vs 0.140, B = 64 0.185 vs 0.153; Mac B = 1 0.28 vs 0.19) — one threadgroup
    running the dependent tail is slower than two more dispatches (Mac: layer 1 34 us, tail ~110 us).
  - What's left is per-call latency, not kernels: iPhone GPU timestamps at B = 1 show the GPU starting 60-80 us after
    the CPU committed (encoding takes 5-9 us) and the CPU seeing completion ~30 us after the last kernel, vs ~38 us
    of kernels. Calls are bimodal (B = 1: 0.055-0.065 ms vs ~0.13 ms, interleaved; B = 1024 sessions 0.15-0.33 ms),
    so medians move ±10-40 % between sessions; compare variants only interleaved in one session.
