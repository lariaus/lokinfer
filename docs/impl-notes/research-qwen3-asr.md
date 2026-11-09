# Qwen3-ASR-1.7B — research notes

Facts gathered 2026-11-05 before the port (`qwen3-asr-port-plan.md` decides what to do with them). Model:
`Qwen/Qwen3-ASR-1.7B-hf`, pinned revision `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`, transformers 5.20.2 (native
`transformers/models/qwen3_asr`), torch 2.14.1. File references: `MOD` = `modeling_qwen3_asr.py`, `FE` =
`feature_extraction_qwen3_asr.py`, `PROC` = `processing_qwen3_asr.py`, `Q3` = `models/qwen3/modeling_qwen3.py`.
Technical report: arXiv 2601.21337 (encoder "AuT", 8× downsampling to 12.5 Hz, dynamic 1–8 s attention windows).

## 1. How the app runs it (english-practice, the reference path)

`native-server/speech_to_text/lib/qwen_asr_model.py`: `AutoModelForMultimodalLM.from_pretrained(id, dtype=float16)`
on **MPS (Mac GPU), fp16**; `processor.apply_transcription_request(audio_16k_f32, language="English")`, greedy
`generate(max_new_tokens = 128 + 8·seconds)`, `processor.decode(..., return_format="transcription_only")`. Mac CLI
build only; the iPhone sends audio to the Mac. Measured in the app: a 1:51 reading of a 277-word text.

## 2. Boundary (processor → model)

Feature extraction (`FE`:64-127), Whisper-style with 128 mels: 16 kHz mono, n_fft 400, hop 160, periodic Hann,
`torch.stft(center=True, reflect)`, last frame dropped, power spectrum, slaney mel 0–8 kHz, `log10(clamp 1e-10)`,
`max(x, global_max − 8)`, `(x + 4) / 4` (the max is over the whole clip). No dither. Clips < 0.5 s are zero-padded
to 8000 samples. `n_frames = floor(n_samples / 160)`; the mel time axis is right-padded with 0.0 (normalized
space) to a multiple of 100. No 30 s cap (padding="longest").

Processor outputs: `input_features f32[1, 128, 100·ceil(F/100)]`, `input_features_mask i32` (1 for the first F),
`input_ids i64[1, T]`, `attention_mask` (all ones). Prompt: 9 prefix ids `[151644, 8948, 198, 151645, 198, 151644,
872, 198, 151669]`, then `N_a × 151676` (`<|audio_pad|>`), then 9 suffix ids `[151670, 151645, 198, 151644, 77091,
198, 11528, 6364, 151704]` (`language English<asr_text>`). **T = N_a + 18**, audio slots contiguous from position 9.
`N_a = 13·floor(F/100) + f(F mod 100)`, `f(r) = c(c(c(r)))`, `c(x) = (x−1)//2 + 1`, `f(0) = 0` (13 tokens per
second of audio).

## 3. Audio encoder (314.3 M params)

Per 1 s chunk (`[n, 1, 128, 100]`, chunks independent, `MOD`:401-405): conv2d 1→480 3×3 s2 p1 + bias + GELU(erf)
→ `[480, 64, 50]`; conv2d 480→480 same → `[480, 32, 25]`; again → `[480, 16, 13]`; flatten `(channel, freq)` per
time step → `[13, 7680]`; `conv_out` Linear 7680→1024 **no bias**; + sinusoidal positions `[13, 1024]` (`[sin |
cos]`, inv_timescale `exp(−ln(10000)/511·i)`) that **restart every chunk**; keep the first N_a rows (only the last
chunk can be partial; its convs still see the zero-padded frames).

24 pre-LN layers (`MOD`:196-242): `x + out_proj(Attn(LN(x)))`, `x + fc2(GELU(fc1(LN(x))))`; D 1024, 16 heads,
head_dim 64, FFN 4096, all projections with bias, LayerNorm eps 1e-5. Under f16 the layer output is clamped to
±(65504 − 1000) (`MOD`:236-238). Attention is **non-causal and block-diagonal over windows of 104 rows** (8 s:
`n_window_infer` 800 / 100 × 13), the last window holds the remainder (`MOD`:168-188, 276-315). Then `ln_post`,
projector Linear 1024→1024 + bias, GELU(erf), Linear 1024→2048 + bias (`MOD`:439-450). No state across chunks or
windows.

## 4. Text decoder (Qwen3-1.7B, 1720.6 M params incl. the tied 311.2 M embedding)

28 layers, hidden 2048, 16 q heads / 8 kv heads (GQA, group 2), head_dim 128, SwiGLU `down(silu(gate)·up)` with
6144, no biases, RMSNorm eps 1e-6 computed in f32 (`Q3`:49-63), **QK-norm** (RMSNorm over head_dim, weights [128])
before RoPE, RoPE default, θ 1e6, rotate_half, full 128 dims, cos/sin in f32; plain 1-D positions 0..T−1 for every
token including audio (no M-RoPE); vocab 151936, tied lm_head; full attention in every layer.

Audio embeddings replace the `<|audio_pad|>` rows of `embed(input_ids)` (`masked_scatter`, `MOD`:545-558).
`generate()`: greedy, eos `[151643, 151645]`, no repetition penalty; prefill computes the LM head on the last row
only; the encoder runs once. A manual loop (encoder → splice → prefill → T=1 steps → argmax) matches `generate()`
token for token on CPU f32 and MPS f16.

## 5. Sizes

| part | params | f32 | f16 |
|---|---|---|---|
| encoder + projector | 317.5 M | 1.27 GB | 0.64 GB |
| LM (incl. embedding / head) | 1720.6 M | 6.88 GB | 3.44 GB |
| total | 2038.1 M | 8.15 GB | 4.08 GB |

KV cache: 57,344 values per token (114,688 B in f16). Encoder ≈ 6.2 GMAC per second of audio; LM prefill ≈ 1.41
GMAC per token; decode reads all LM weights per token (3.44 GB in f16: ≈ 9 ms floor at 380 GB/s on the M2 Max GPU,
≈ 45–55 ms on the iPhone at 60–77 GB/s — roofline estimates).

## 6. Workload shapes (prompt T = N_a + 18)

| audio | mel frames | chunks | N_a | windows | prefill T | max_new | expected tokens | final context |
|---|---|---|---|---|---|---|---|---|
| 2 s | 200 | 2 | 26 | 1 | 44 | 144 | 6–9 | ≈ 52 |
| 5 s | 500 | 5 | 65 | 1 | 83 | 168 | 13–18 | ≈ 100 |
| 10 s | 1000 | 10 | 130 | 2 | 148 | 208 | 25–36 | ≈ 180 |
| 30 s | 3000 | 30 | 390 | 4 | 408 | 368 | 75–110 | ≈ 500 |
| 60 s | 6000 | 60 | 780 | 8 | 798 | 608 | 150–220 | ≈ 980 |
| 111 s | 11100 | 111 | 1443 | 14 | 1461 | 1016 | 330–410 | ≈ 1850 |

Long audio: no documented maximum in the model card or README ("supports long audio"; raise `max_new_tokens`).
Qwen's own Python package `qwen-asr` 0.0.6 (`qwen_asr/inference/utils.py`) caps one model pass at
`MAX_ASR_INPUT_SECONDS = 1200` (20 min; 180 s with timestamps) and splits longer audio with
`split_audio_into_chunks(..., search_expand_sec=5.0)` at the lowest-energy point near each boundary. Qwen's hosted
API caps requests at 3 min (Qwen3-ASR-Toolkit splits with VAD). Some third-party apps cap segments at ~20–30 s
(SmartSub PRs #562/#563, token-budget issues in their setup). Positions: `max_position_embeddings` 65536. A 20 min
pass: N_a = 15,600, prompt 15,618, final context ≈ 20k (≤ 25,346 with the app's `max_new_tokens` formula).

## 7. Reference timings (M2 Max, 9.71 s TTS clip, indicative)

torch CPU f32: encoder 170 ms, prefill 450 ms, decode 147 ms/token, total 5.6 s. MPS f16 (app path): encoder 55
ms, prefill 93 ms, decode ≈ 15–23 ms/token, `generate()` 0.68 s.

## 8. Precision (decision: f16 on all four devices, `qwen3-asr-port-plan.md`)

- **Weights**: stored bf16. Converted to f16: 0 overflow (max |w| 153, the late RMSNorm weights), 0.017 % not
  exactly representable (tiny values in f16's subnormal range), 2049 tiny values become 0.
- **Activations** (f32 run, TTS speech): encoder residual ≤ 64, convs ≤ 15, projector ≈ 0.2; **LM residual stream
  ≈ 10,700 from layer 2 on** ("massive activations", created by layer 2's down_proj and cancelled by layer 27's) —
  6× under f16's 65504, but the f16 step there is 8.
- **f32 islands in the reference fp16 path**: RMSNorm / QK-norm math, softmax, RoPE tables; GEMM accumulation in
  f32 in the GPU libraries. The residual stream itself is f16.
- **float64 golden**: the model runs in torch float64 on CPU (2.2 s clip: 22 s, same tokens as f32; f32 vs f64
  last-row logits max diff 8.8e-5).
- **Mac measurements (torch 2.14.1, M2 Max)**: MPS GEMM 1461×2048×6144 f32 4.43 ms, f16 4.36, **bf16 8.44 (2×
  slower: bf16 emulated on Apple8 GPUs)**; MPS GEMV 1×2048×6144 f32 0.29, f16 0.22, bf16 0.19 ms. torch CPU f16 /
  bf16 GEMM is a generic slow fallback (≈ 500× slower than f32): it measures PyTorch, not the hardware.
- **Hardware**: M2 CPU has FEAT_FP16 (native f16 NEON), FEAT_FHM (FMLAL: f16 × f16 → f32), FEAT_BF16 (BFDOT /
  BFMMLA, bf16 → f32), no SME. iPhone (A20 Pro) CPU: SME2 FMOPA (f16 → f32) and BFMOPA (bf16 → f32) at the same peak
  (M4 measurement, arXiv 2409.18779). GPUs: `half` native everywhere; `bfloat` emulated on Apple7/8 (M1/M2), native
  from Apple9 (A17/M3) (metal-benchmarks GEMM gist: M4 BF16 3324 vs FP16 3407 GFLOPS); A19-class GPU matrix units
  ≈ 7.5 TFLOPS f16, bf16 about equal since OS 26.1 (tzakharko benchmark; techboards thread).
- **Accelerate has no f16 BLAS GEMM** (`cblas` is f32/f64 only); BNNS has a Float16 data type.
- **What others do with Qwen**: Qwen's releases and examples are bf16; a Qwen maintainer calls fp16 instability
  "expected" (QwenLM/Qwen3#868). Apple / edge runtimes still mostly run Qwen in fp16 with f32 islands: llama.cpp
  (f16 KV / compute, Qwen attention KQ in f32, PR #8412; bf16 Metal opt-in, PR #10218), Core ML / ANE (f16 only;
  Qwen3 ports keep norm math in f32), TensorRT-Edge-LLM Qwen3-ASR (FP16 only), MLX users casting bf16 → fp16 on
  M1/M2 (+28–50 % prefill). Unquantized MLX and GGUF Qwen3-ASR conversions default to bf16 (the checkpoint's type).
  Known fp16 failure points: RMSNorm sum of squares, attention scores, the SwiGLU product feeding down_proj.

## 9. Baselines (research 2026-11-05; all in f16 for this model)

| framework | Mac CPU | Mac GPU | iPhone CPU | iPhone GPU | notes |
|---|---|---|---|---|---|
| PyTorch eager / torch.compile | yes | MPS | — | — | native `Qwen3ASRForConditionalGeneration`; the app path is MPS fp16 |
| MLX Python (mlx-audio `stt/models/qwen3_asr`) | — | yes | — | — | `mlx-community/Qwen3-ASR-1.7B-bf16` (cast to f16) |
| MLX Swift (mlx-audio-swift) | — | — | — | yes | lists Qwen3-ASR; main branch only, no tags |
| llama.cpp (mtmd, PR #19441, 2026-04) | yes | Metal | yes | Metal | F16 GGUF via `convert_hf_to_gguf.py --outtype f16`; iOS via `build-xcframework.sh` |
| ExecuTorch (XNNPACK) | — | — | custom export | — | no Qwen3-ASR recipe (qwen3 LLM recipe + own encoder export): deferred |
| Core ML / ONNX / sherpa-onnx / MNN | | | | | int8-only community builds or no official export: skipped |

iPhone memory: 12 GB RAM; measured per-app limit **6136 MB** with `com.apple.developer.kernel.increased-memory-limit`
(lokinfer-baselines MLX logs, iOS 27.0). f32 (8.15 GB) cannot fit; f16 (4.08 GB) fits with ≈ 2 GB for KV,
activations and runtime.

Calibration (not apples to apples): speech-swift Qwen3-ASR-1.7B MLX 5-bit RTF 0.027 on an M5 Pro; Qwen3-1.7B MLX
Swift 4-bit on an A19 Pro 39.5 tok/s decode.

## 10. lokinfer facts the port relies on (verified 2026-11-05)

- `DType::F16` exists in the IR (`include/lokinfer/ir/types.h`), prints as `f16`, 2 bytes; numpy interop is
  zero-copy for `np.float16` (`python/lokinfer/dtypes.py`); the remote protocol carries the dtype id (dtype-generic).
- **No 16-bit kernel exists**: all 183 registered kernels are f32 or integer (`registered_kernels()`).
- `pytest --device=iphone` runs every compiled graph (incl. one-op kernel graphs) through the remote executor on the
  phone (`tests/conftest.py`): kernel tests can run on the iPhone without C++ test code.
- GPT-2 `build_generate` (`models/gpt2.py`): symbolic T, persistent f32 caches `[N_POS, D]` per layer
  (`lk.persistent_buffer`), chunked prefill through `attention_cache` (`kernels/cpu/decode.cpp`): the pattern the
  LM port reuses with f16 caches and GQA.
- Kokoro's data-dependent `F` symbol comes from `dim.read` of a tensor element (`python/lokinfer/ops/view.py`
  `dim_read`): the encoder's N_a uses the same mechanism (hence `n_audio i64[1]`, not a scalar).
- lokinfer already uses MPS as a library kernel (GPT-2 prefill GEMMs, `gemm_f32_gpu_mps`), so an MPS f16 GEMM is an
  acceptable "opt" kernel; Accelerate has no f16 BLAS (`cblas` f32 / f64 only), so the Mac CPU needs NEON FMLAL
  tiles or BNNS.

## Sources

Qwen3-ASR model card https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf · technical report https://arxiv.org/pdf/2601.21337
· QwenLM/Qwen3-ASR https://github.com/QwenLM/Qwen3-ASR · QwenLM/Qwen3#868 https://github.com/QwenLM/Qwen3/issues/868 ·
llama.cpp #8412 https://github.com/ggml-org/llama.cpp/pull/8412 and #10218 https://github.com/ggml-org/llama.cpp/pull/10218
· llama.cpp multimodal https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md · ggml-org GGUF
https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF · mlx-audio https://github.com/Blaizzy/mlx-audio · mlx-audio-swift
https://github.com/Blaizzy/mlx-audio-swift · ExecuTorch models https://github.com/pytorch/executorch/tree/main/examples/models
· TensorRT-Edge-LLM ASR https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/asr.html · Metal GEMM
benchmarks https://gist.github.com/philipturner/84f613a5cc745460a914d2c6ad226131 · A19 neural accelerators
https://tzakharko.github.io/apple-neural-accelerators-benchmark · M4 SME https://arxiv.org/pdf/2409.18779 · Metal
feature tables https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf · Apple memory-limit thread
https://developer.apple.com/forums/thread/834805 · Qwen3 Core ML fp16 notes
https://huggingface.co/groxaxo/qwen3-1.7b-coreml-int8 · ExecuTorch RMSNorm fp16 https://github.com/pytorch/executorch/pull/19523
