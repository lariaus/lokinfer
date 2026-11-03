# Kokoro-82M — architecture, op inventory, profiling and baselines (research notes)

Status: research, 2026-11-01. Machine: Mac M2 Max (8P+4E CPU, 38-core GPU, 32 GB), macOS 27,
Python 3.14.10, torch 2.14.1, transformers 5.20.2, kokoro 0.7.16, weights snapshot `f3ff3571…`.

Baseline under study: english-practice `native-server/kokoro_tts/kokoro_model.py::infer(ids, ref_s, speed)`,
which is `KModel.forward` minus phonemes→ids, CPU, float32, `torch.no_grad()`.

All scripts and raw results lived in a local scratch directory (not in the repo)
(`profile_kokoro.py`, `bench_eager.py`, `bench_compile.py`, `bench_onnx.py`, `bench_mlx.py`, `flops.py`,
`rng_check.py`, `lstm_check.py`, `gemm_check.py`, `gemm_k_check.py`, `results_{cpu,mps}.json`, `log_*.txt`).
They are throwaway; reproduce them in lokinfer's own `benchmarks/` when that exists.

---

## 0. TL;DR

- **One forward = ~0.665 GMAC per audio frame + ~0.066 GMAC per token** (frame = 25 ms, 40 frames/s).
  510 random tokens → F=1127 frames, 28.2 s audio, **783 GMAC (1.57 TFLOP)**. 92 % of MACs are the 1-D
  convolutions of the iSTFTNet generator (k=3/7/11, dilations 1/3/5, at 256 ch × 20F and 128 ch × 120F samples).
- **Time is the generator.** CPU eager: generator ≈ 85 % of the forward, its 128-channel stage alone ≈ 50–56 %.
  Op view (CPU): conv 56 %, instance-norm 8 %, `sin` (Snake) 6 %, elementwise mul/add/pow ≈ 9 %, randn 3.7 %.
- **PyTorch MPS has a big artificial hotspot:** `pack_padded_sequence` LSTMs run ~10× slower on MPS than plain
  LSTMs; the DurationEncoder + TextEncoder LSTMs are ~38 % of MPS forward time. Packing is a no-op for batch 1.
- **Only one random tensor matters:** `randn[1, 600F, 9]` in SineGen. `torch.rand(1,9)` (`rand_ini`) is provably
  dead (its effect is erased by the ÷300 linear downsample) and the second `randn_like` feeds an unused output.
  Verified bit-exact (`rng_check.py`).
- **Data-dependent shape:** `F = Σ_t max(1, round(Σ_50 sigmoid(·)/speed))`, `F ≤ 50·T/speed`. The one-hot
  alignment matmul is an exact gather; it can be pushed *before* linear ops (LSTM input projection, 1×1 convs)
  so they run at token rate (F/T ≈ 2.4 for real English).
- **Baselines (median, ms):**

  | case (tokens→frames) | torch CPU eager | torch.compile CPU | ORT CPU | torch MPS eager | torch.compile MPS | MLX GPU (f32) | kokoro.cpp Metal (f16) |
  |---|---|---|---|---|---|---|---|
  | rand16 (18→74, 1.85 s) | 288 | 262 | 420 | 158 | 99 | **67** | – |
  | real28 (30→86, 2.15 s) | 315 | 283 | 493 | 160 | 122 | **81** | ~350 |
  | real117 (119→278, 6.95 s) | 850 | 785 | 1570 | 381 | 299 | **260** | ~1090 |
  | rand128 (130→334, 8.35 s) | 995 | 1033 | 1872 | 466 | **358** | 390 | – |
  | rand256 (258→680, 17.0 s) | 1848 | 1643 | 3795 | 992 | **738** | 827 | – |
  | real394 (396→935, 23.4 s) | 2556 | 2303 | 5224 | 1501 | **920** | 1121 | ~3100 |
  | rand510 (512→1127, 28.2 s) | **3090** | 2793 | 6344 | 1703 | **1158** | 1404 | ~3820 |

  Best CPU: torch.compile CPU (only 5–15 % over eager). Best GPU: MLX for short inputs, torch.compile-MPS for long.
  ORT-CoreML EP crashes; MLX-CPU is ~100× slower than torch CPU; kokoro.cpp (ggml) loses on Apple.
- **Headroom (rough floors from measured GEMM throughput):** CPU ≈ 1.3–1.5 s at 510 tokens (≈ 2× over eager),
  GPU ≈ 0.3–0.45 s at 510 tokens (≈ 4× over eager MPS, ≈ 3× over the best GPU baseline).

---

## 1. Architecture walkthrough of `infer()`

### 1.1 Notation

| symbol | meaning |
|---|---|
| `N` | number of phoneme ids from the frontend, `N ≤ 510` |
| `T = N + 2` | tokens after padding with id 0 at both ends (`T ≤ 512` = ALBERT context) |
| `d_t` | per-token duration (int frames), data-dependent |
| `F = Σ d_t` | number of 25 ms frames (40 Hz). Audio seconds = F/40 |
| `2F` | F0/N curve length (80 Hz) |
| `L0 = 20F` | generator stage-0 length (800 Hz) |
| `L1 = 120F + 1` | generator stage-1 length (4.8 kHz, +1 from reflection pad / centered STFT) |
| `S = 600F` | output samples (24 kHz) |
| `s_p = ref_s[:,128:]` | prosody style (128) |
| `s_d = ref_s[:,:128]` | decoder style (128) |

Batch is always 1. Measured F/T: real English 2.34–2.87 frames/token; random ids 2.2–4.1.

Config: hidden 512, style 128, ALBERT(hidden 768, emb 128, 12 heads, ffn 2048, 12 layers **sharing one layer's
weights**, `gelu_new`, LN eps 1e-12), n_token 178, max_dur 50, iSTFTNet(upsample rates 10,6; kernels 20,12;
resblock kernels 3,7,11; dilations 1,3,5; n_fft 20; hop 5; initial ch 512).

Weights: 81.76 M params (bert 6.29 M, bert_encoder 0.39 M, predictor 16.19 M, text_encoder 5.61 M, decoder 53.28 M).

### 1.2 Inputs / preprocessing (glue)

```
input_ids  = [[0, *ids, 0]]                 int64 [1,T]
text_mask  = all False (batch 1, no padding) → every masked_fill / pack_padded_sequence is a no-op
ref_s      = frombuffer(...)                f32 [1,256]  (voice pack row pack[N-1])
```

### 1.3 PL-BERT (transformers `AlbertModel`, wrapped by `CustomAlbert`)

```
emb  = word_emb[ids] (178×128) + tok_type_emb[0] (2×128) + pos_emb[0..T) (512×128)   [1,T,128]
emb  = LayerNorm128(emb)                                                                [1,T,128]
h    = Linear 128→768 (embedding_hidden_mapping_in)                                    [1,T,768]
12× (same weights every layer):
   q,k,v = Linear 768→768 each                                     [1,12,T,64]
   a     = SDPA(q,k,v, mask = all-ones additive 0)                 [1,T,768]    (sdpa → flash-attn-for-cpu on CPU)
   h     = LayerNorm768(h + Linear 768→768(a))
   f     = Linear 2048→768( gelu_new( Linear 768→2048(h) ) )
   h     = LayerNorm768(h + f)
pooler = tanh(Linear 768→768(h[:,0]))   ← computed but UNUSED (drop it)
bert_dur = h                                                                            [1,T,768]
```

### 1.4 bert_encoder

`d_en = Linear 768→512(bert_dur).transpose → [1,512,T]`

### 1.5 Prosody predictor: DurationEncoder (`predictor.text_encoder`)

```
x = cat(d_en^T [T,512], s_p broadcast [T,128]) → [T,640]          (masked_fill no-op)
3× {
   BiLSTM(640 → 2×256)  via pack_padded_sequence   → [T,512]
   AdaLayerNorm(512): LN over channels (no affine, eps 1e-5), then (1+γ)·x + β, [γ;β] = Linear 128→1024(s_p)
   x = cat(x, s_p) → [T,640]
}
d = x → [1,T,640]
```

### 1.6 Duration head

```
x   = BiLSTM(640→2×256)(d)  (predictor.lstm, NOT packed in infer)      [1,T,512]
dur = Linear 512→50(x) → sigmoid → sum over 50 → /speed                 [T] float
pred_dur = clamp(round(dur), min=1).long()                               [T] int   ← DATA-DEPENDENT
```
Each token gets 1…⌈50/speed⌉ frames.

### 1.7 Alignment (data-dependent)

```
indices      = repeat_interleave(arange(T), pred_dur)          [F]
pred_aln_trg = zeros(T,F); pred_aln_trg[indices, arange(F)] = 1  one-hot [T,F]
en  = d^T  @ aln   : [1,640,T] @ [1,T,F] → [1,640,F]
asr = t_en @ aln   : [1,512,T] @ [1,T,F] → [1,512,F]   (after TextEncoder)
```

### 1.8 F0Ntrain (`predictor.F0Ntrain(en, s_p)`)

```
x = BiLSTM(640→2×256)(en^T)  (predictor.shared, over F frames, sequential!)   [1,F,512] → [1,512,F]
F0 branch:                                  N branch (same structure, own weights):
  AdainResBlk1d(512→512)        [512,F]
  AdainResBlk1d(512→256, up×2)  [256,2F]
  AdainResBlk1d(256→256)        [256,2F]
  Conv1d 256→1 k1 → F0_pred     [2F]        N_pred [2F]
```

**AdainResBlk1d(cin→cout, upsample?)** (StyleTTS2), `actv = LeakyReLU(0.2)`:
```
residual: AdaIN1d(cin) → leaky → [pool = depthwise ConvTranspose1d(cin,cin,k3,s2,p1,op1) if up]
          → Conv1d(cin→cout,k3,p1) → AdaIN1d(cout) → leaky → Conv1d(cout→cout,k3,p1)
shortcut: [nearest-upsample ×2 if up] → [Conv1d 1×1 cin→cout, no bias, if cin≠cout]
out = (residual + shortcut) / √2
AdaIN1d(C): InstanceNorm1d (per-channel mean/var over time, biased var, eps 1e-5, no affine)
            then (1+γ)·x̂ + β with [γ;β] = Linear 128→2C(style)
```
(Dropouts are eval-mode no-ops.)

### 1.9 TextEncoder (acoustic text encoder, `model.text_encoder`)

```
x = Embedding 178×512 (ids) → [1,512,T]
3× { Conv1d 512→512 k5 p2 (weight-norm) → LayerNorm over channels (affine γ,β) → LeakyReLU(0.2) }
x = BiLSTM(512 → 2×256) via pack_padded_sequence → [1,512,T] = t_en
```
Independent of durations → can run concurrently with BERT/predictor.

### 1.10 Decoder (`model.decoder`, style `s_d`)

```
F0d = Conv1d(1,1,k3,s2,p1)(F0_pred) → [1,1,F];   Nd = Conv1d(1,1,k3,s2,p1)(N_pred) → [1,1,F]
x   = cat(asr, F0d, Nd)                 [514,F]
x   = encode = AdainResBlk1d(514→1024)  [1024,F]
asr_res = Conv1d 512→64 k1 (asr)        [64,F]
decode[0..2]: x = AdainResBlk1d(1090→1024)( cat(x, asr_res, F0d, Nd) )   [1024,F]
decode[3]   : x = AdainResBlk1d(1090→512, up×2)( cat(...) )            [512,2F]
x = Generator(x, s_d, F0_pred)
```

### 1.11 Generator (iSTFTNet + hn-NSF harmonic source)

**Harmonic source (under no_grad, depends only on F0_pred):**
```
f0   = nearest-upsample ×300 (F0_pred) → [1,S,1]            S = 600F
fn   = f0 · [1..9]                     → [1,S,9]            (fundamental + 8 harmonics)
rad  = (fn / 24000) % 1
rand_ini = torch.rand(1,9); rand_ini[:,0]=0; rad[:,0,:] += rand_ini      ← DEAD (see §2.4)
rad_d = interpolate(rad, scale 1/300, linear)  → [1,2F,9]   (= exactly rad at each frame)
phase = cumsum(rad_d over time) · 2π                        [1,2F,9]
phase = interpolate(phase·300, scale 300, linear) → [1,S,9]
sine  = 0.1 · sin(phase)
uv    = (f0 > 10)
noise = (uv·0.003 + (1−uv)·0.1/3) · randn_like(sine)        ← THE random input [1,S,9]
sine  = sine·uv + noise
har_source = tanh(Linear 9→1 (sine))                        [1,S]
(noise branch: randn_like(uv)·0.1/3 [1,S,1] → returned but UNUSED)
har = STFT(har_source, n_fft=20, hop=5, win=hann20(periodic), center, reflect)
    → |X|, angle(X)  each [11, 120F+1] → cat → [1,22,120F+1]
```

**Upsampling stack:**
```
i=0: x = leaky(x, 0.1)                                       [512,2F]
     xs = noise_convs[0] = Conv1d(22→256, k12, s6, p3)(har)   [256,20F]
     xs = noise_res[0]  = AdaINResBlock1(256, k7, d=1,3,5)(xs)
     x  = ups[0] = ConvTranspose1d(512→256, k20, s10, p5)(x)  [256,20F]
     x  = x + xs
     x  = mean_j resblocks[j](x), j=0..2 (k=3,7,11)           [256,20F]
i=1: x = leaky(x, 0.1)
     xs = noise_convs[1] = Conv1d(22→128, k1)(har)            [128,120F+1]
     xs = noise_res[1]  = AdaINResBlock1(128, k11, d=1,3,5)(xs)
     x  = ups[1] = ConvTranspose1d(256→128, k12, s6, p3)(x)   [128,120F]
     x  = ReflectionPad1d((1,0))(x)                           [128,120F+1]
     x  = x + xs
     x  = mean_j resblocks[3+j](x), j=0..2 (k=3,7,11)         [128,120F+1]
x    = leaky(x, 0.01); x = conv_post = Conv1d(128→22, k7, p3)(x)  [22,120F+1]
spec = exp(x[:11]);  phase = sin(x[11:])        (sin of the logit, not a real angle)
audio = iSTFT(spec·e^{i·phase}, n_fft 20, hop 5, hann20, center) → [S] = 600F samples
```

**AdaINResBlock1(C, k, dil=(1,3,5))** (the generator workhorse):
```
for j in 0..2:
   xt = AdaIN1d_j1(x) ; xt = Snake(xt, α1_j)      Snake(x,α) = x + (1/α)·sin²(α·x), α per channel
   xt = Conv1d(C→C, k, dilation=dil[j], pad=dil[j]·(k−1)/2)(xt)
   xt = AdaIN1d_j2(xt); xt = Snake(xt, α2_j)
   xt = Conv1d(C→C, k, dilation=1, pad=(k−1)/2)(xt)
   x  = xt + x
```
6 convs, 6 AdaIN (6 style Linears), 6 Snakes per block; 8 such blocks (6 resblocks + 2 noise_res).

### 1.12 Measured shapes (T=130 → F=334)

Recorded with forward hooks (`shapes_128.json`): BERT `[1,130,768]`; DurEnc LSTMs `[130,640]→[130,512]`;
shared LSTM `[1,334,640]→[1,334,512]`; F0/N blocks `[512,334]→[256,668]`; decoder `[1090,334]→[1024,334]`,
decode[3] `→[512,668]`; `m_source.l_linear [1,200400,9]→[1,200400,1]`; `noise_convs[0] [22,40081]→[256,6680]`;
`ups[0] [512,668]→[256,6680]`; `ups[1] [256,6680]→[128,40080]`; stage-1 convs `[128,40081]`;
`conv_post [128,40081]→[22,40081]`; audio 200400 samples. All match the formulas above.

---

## 2. Primitive-op inventory

### 2.1 Op table (per forward, batch 1, f32 unless noted)

MAC column = closed form; "%MAC" at rand510 (F=1127, 783 GMAC total, from `flops.py`).

| op kind | where | count | shapes | MACs | data-dep? |
|---|---|---|---|---|---|
| embedding gather | ALBERT word/pos/type, TextEncoder | 4 | [T]→[T,128]×3, [T,512] | – | T |
| LayerNorm (last dim) | ALBERT emb (128), 12×2 (768) | 25 | [T,128], [T,768] | – | T |
| LayerNorm over channels, affine | TextEncoder CNN | 3 | [512,T] | – | T |
| LayerNorm no-affine + style FMA | DurEnc AdaLayerNorm | 3 | [T,512] | – | T |
| Linear / GEMM (token-rate) | ALBERT 1+72(+1 pooler), bert_encoder, duration_proj | 75 (+1 dead) | M=T, K/N ∈ {128,768,2048,512,50} | ≈ 66 M·T | T |
| SDPA (non-causal, 12×64, no mask) | ALBERT | 12 | [12,T,64] | 2·12·64·T² | T |
| gelu_new (tanh approx) | ALBERT FFN | 12 | [T,2048] | – | T |
| BiLSTM (1 layer) | DurEnc 3×(640→256), pred.lstm (640), TextEnc (512) | 5 | over T steps | 2·T·4·256·(I+256) each | T |
| BiLSTM | predictor.shared (640→256) | 1 | over **F** steps | 2·F·4·256·896 | **F** |
| sigmoid/sum/div/round/clamp | duration head | 1 each | [T,50] | – | T |
| repeat_interleave + one-hot scatter | alignment | 1 | [T]→[F], [T,F] | – | **F** |
| bmm (one-hot) | en, asr | 2 | [640,T]@[T,F], [512,T]@[T,F] | 1152·T·F (wasted) | **F** |
| Linear 128→2C (style) | AdaIN/AdaLN γ,β | 73 | GEMV, Σ rows = 49 684 | 6.36 M | no (input `ref_s` only) |
| InstanceNorm1d (per-ch, over time) | all AdaIN1d | 70 | [C, len] | – | F |
| LeakyReLU | 0.2: 22 (AdainResBlk1d) + 3 (TextEnc); 0.1: 2; 0.01: 1 | 28 | | – | F |
| Snake (mul, sin, pow, div, add) | AdaINResBlock1 | 48 | [256,20F] ×24, [128,120F+1] ×24 | – | F |
| Conv1d k3 | AdainResBlk1d convs (22) + resblocks k3 (12) + F0_conv/N_conv (2) | 36 | | 13.7 % | F |
| Conv1d k5 | TextEncoder | 3 | [512,T] | 0.3 % | T |
| Conv1d k7 | resblocks k7 (12) + noise_res[0] (6) + conv_post (1) | 19 | | 28.1 % | F |
| Conv1d k11 | resblocks k11 (12) + noise_res[1] (6) | 18 | | 49.8 % | F |
| Conv1d k1 | 1×1 shortcuts (2 F0/N + 5 decoder), F0/N proj (2), asr_res, noise_convs[1] | 11 | | 0.8 % | F |
| Conv1d k12 s6 | noise_convs[0] | 1 | [22,120F+1]→[256,20F] | 0.2 % | F |
| ConvTranspose1d depthwise k3 s2 | F0[1], N[1], decode[3] `pool` | 3 | [512,F],[512,F],[1090,F] → ×2 | ~0 | F |
| ConvTranspose1d k20 s10 | ups[0] 512→256 | 1 | [512,2F]→[256,20F] | 0.8 % | F |
| ConvTranspose1d k12 s6 | ups[1] 256→128 | 1 | [256,20F]→[128,120F] | 1.1 % | F |
| nearest upsample ×2 | AdainResBlk1d shortcuts | 3 | | – | F |
| nearest upsample ×300, linear interp ÷300, ×300 | SineGen | 3 | [2F]→[S], [S,9]→[2F,9]→[S,9] | – | F |
| cumsum | SineGen phase | 1 | [2F,9] | – | F |
| sin | Snake (48) + SineGen (1) + post phase (1) | 50 | | – | F |
| randn | SineGen noise (+1 unused) | 1 (+1) | [S,9] (+[S,1]) | – | F |
| rand | SineGen rand_ini (dead) | 1 | [1,9] | – | – |
| Linear 9→1 + tanh | m_source | 1 | [S,9] | 9·S | F |
| STFT (n_fft 20, hop 5) + abs + angle | harmonic → `har` | 1 | [S] → [22, 120F+1] | ~ | F |
| exp, sin | post | 1 each | [11,120F+1] | – | F |
| iSTFT (n_fft 20, hop 5) | output | 1 | [11,120F+1] complex → [S] | ~ | F |
| ReflectionPad1d (1,0) | after ups[1] | 1 | | – | F |
| residual add / ÷√2 / ÷3 / cat | everywhere | ~60 | | – | F |
| weight-norm recompute `g·v/‖v‖` | every weight-normed conv, **every forward** | 89 | | – | no → fold at load |

PyTorch CPU executes **146 k aten op calls at T=130 and 344 k at T=512** (torch.profiler, incl. nested);
most are LSTM per-timestep ops (thousands of addmm/sigmoid/tanh/mul calls) and the per-group loop of the
depthwise ConvTranspose (`slow_conv_transpose2d` called 2 116 times).

Total MACs fit: **MAC ≈ 0.665·F + 0.066·T GMAC** (230 GMAC at T=130/F=334; 783 GMAC at T=512/F=1127).
Per frame: stage-1 convs 377 M (57 %), stage-0 convs 220 M (33 %), decoder 34 M, ups 13 M, F0/N 4 M.

### 2.2 Dtypes

Everything is f32 except: `input_ids`/`indices`/`pred_dur` int64 (int32 is enough), `uv` bool→f32,
STFT/iSTFT complex64 internally (avoidable — see §5), `text_mask` bool (dead).

### 2.3 Weight-norm handling (fold at load)

89 modules use `torch.nn.utils.weight_norm` (legacy, `dim=0`); checkpoint stores `weight_g`, `weight_v`.
PyTorch recomputes `w = g · v / ‖v‖` on every forward (89 `_weight_norm_interface` calls ≈ 10 ms CPU).
Fold once at load: `w = g · v / ‖v‖₂` where the norm is over **all dims except dim 0**:

- Conv1d weight `[Cout, Cin/g, k]` → norm per output channel.
- **ConvTranspose1d weight `[Cin, Cout/g, k]` → norm per *input* channel** (dim 0 is Cin) — easy to get wrong.

Weight-normed modules: TextEncoder CNN (3); predictor AdainResBlk1d conv1/conv2/conv1x1/pool (16);
decoder encode/decode convs + pool + F0_conv + N_conv + asr_res (19); generator ups (2), resblock convs (36),
noise_res convs (12), conv_post (1). NOT weight-normed: noise_convs, F0_proj/N_proj, all Linears, LSTMs.

Other load-time folds worth doing:
- **all 73 style Linears into one GEMV** `[49 684 × 128]` with `+1` pre-added to the γ halves' biases;
- LSTM `b_ih + b_hh` → one bias; both directions' `W_ih` stacked `[2048, I]` → one input-projection GEMM;
- ALBERT Q,K,V → one `[2304, 768]` GEMM; drop the pooler;
- `/√2` of AdainResBlk1d into conv2 weight+bias and the 1×1 shortcut (identity shortcut needs an epilogue scale);
- resblock `mean` (`/3`): leaky_relu is positively homogeneous, so the `/3` folds into `ups[1]` and `conv_post`
  weights (consumer does `leaky(a+b+c)`);
- Snake `x + (1/α) sin²(αx) = x + (1 − cos(2αx)) / (2α)` → precompute `2α`, `1/(2α)`; one `cos` per element;
- AdaIN + style: `y = (1+γ)(x−μ)/σ + β = a·x + b` per channel, `a = (1+γ)/σ`, `b = β − a·μ`
  (μ, σ are runtime stats → `a,b` computed per forward, but it is a per-channel FMA prologue);
- STFT/iSTFT bases (22×20 real DFT·window filters) and the iSTFT window-envelope reciprocal (`1/Σw²`) as constants.

### 2.4 Randomness (all of it)

Verified with `rng_check.py` (monkey-patched RNG, same seed, max |diff| = 0.0 bit-exact):

| call | shape | where | effect on audio |
|---|---|---|---|
| `torch.rand(1, 9)` | [1,9] | `SineGen._f02sine` `rand_ini` | **none** — added to sample 0 only; the 1/300 linear downsample samples indices 300i+149/150, never 0. Still advances the RNG. |
| `torch.randn_like(sine_waves)` | **[1, 600F, 9]** | `SineGen.forward` noise | **the only effective randomness** |
| `torch.randn_like(uv)` | [1, 600F, 1] | `SourceModuleHnNSF` noise branch | **none** — `noi_source` is never used by `Generator.forward` |

No dropout is active (eval / `training=False`). So for lokinfer: one explicit input `noise: f32[600F, 9]`
(standard normal) for tests; an RNG kernel (counter-based Philox, fused into the harmonic-source kernel) for
real use. Matching torch's CPU RNG stream is not needed (seed 0 vs seed 1 audio correlation is 0.995 —
the noise is a small perturbation; amplitude 0.003 voiced / 0.033 unvoiced, then tanh, STFT, convs).

### 2.5 Numerical traps for correctness tests

1. **Duration rounding**: `round(Σ sigmoid)` flips by ±1 frame under tiny float differences → different F →
   different audio length. MLX already shows this (rand256: 409 200 vs 408 000 samples). Tests should check
   durations exactly, and the graph should accept an optional **durations override** input so the decoder can be
   tested against PyTorch with the reference durations.
2. **SineGen phase is large**: `cumsum` over 2F frames × 2π × 300 reaches ~5·10⁴ rad (real117) to ~5·10⁵ rad
   (510 tokens); f32 ulp there is 0.004–0.03 rad, so any reordering of the cumsum or of the interp arithmetic
   changes the sines visibly. Compare `har`/audio with SNR-type metrics, or replicate torch's exact op order
   (sequential cumsum, `(x·300)` before interpolation). A phase-wrapped implementation would be *more* accurate
   than the reference but not bit-close.
3. **STFT `angle`** has a ±π branch cut → isolated jumps where imag ≈ 0, real < 0.
4. InstanceNorm over up to 135 k samples: use two-level (tile partial + Chan/Welford combine) stats in f32,
   not naive Σx² − (Σx)².
5. MLX port vs torch audio correlation is only 0.88 (vs 0.995 between two torch seeds) — MLX is not a
   numerically faithful reference.

---

## 3. Data-dependent shapes

### 3.1 Where

Only one source: `pred_dur` (token-rate ints). Everything downstream of the alignment — shared LSTM, F0/N,
decoder, whole generator, harmonic source, noise input, audio length — has size F, 2F, 20F, 120F+1 or 600F.
Things that do **not** depend on F: ALBERT, DurationEncoder, duration head, TextEncoder (CNN+LSTM),
style projections. Bounds: `T ≤ F ≤ T·⌈50/speed⌉` (sum of 50 sigmoids ≤ 50) → at T=512, speed 1: F ≤ 25 600
(10.7 min audio). Upper-bound allocation for GPU-driven execution is therefore impractical (a single stage-1
activation at F=25 600 is 128·(120·25 600)·4 B ≈ 1.6 GB) → **use one GPU→CPU readback of F** (recommended),
or GPU-driven with a heuristic bound (e.g. 6·T) plus a fallback re-run.

Readback cost hiding: commit command buffer A (BERT → DurEnc → duration head → `F`), immediately commit
buffer B (TextEncoder, style GEMV — F-independent), wait only for A, read F, encode buffer C (F-shaped) while B runs.

### 3.2 Efficient alternative to the one-hot matmul

`aln` is one-hot per column, so `en[:, f] = d[:, idx[f]]` and `asr[:, f] = t_en[:, idx[f]]` **exactly**
(the f32 matmul `Σ x·1 + 0` is bit-exact with the gather). `idx[f] = token whose [start, end) contains f`,
`start = exclusive_cumsum(pred_dur)`. Compute `idx` with a scan over T plus a parallel fill (or each frame
binary-searches the T+1 prefix array). The matmuls cost 1152·T·F MACs (0.67 GMAC at 510 tokens) and an
O(T·F) dense matrix (2.3 MB at T=512/F=1127) — pure waste.

Better still, push the gather through linear ops:
- shared LSTM input projection: `W_ih·en[:,f] = (W_ih·d)[:, idx[f]]` → do the 640→2048 GEMM at **T** rows,
  gather at F (saves F/T ≈ 2.4× on that GEMM);
- `asr_res` (1×1 conv) commutes with the gather → compute at T;
- k3 convs over `asr` (encode/decode conv1) don't commute (they mix neighbouring frames across token
  boundaries) → do an implicit gather in the conv prologue (load `t_en[:, idx[f]]` instead of `asr[:, f]`), so
  `asr` [512, F] never needs to be materialized;
- channel concatenations (`cat(asr, F0d, Nd)`, `cat(x, asr_res, F0d, Nd)`) become "virtual concat" — the conv
  kernel takes several input pointers and splits its K loop.

---

## 4. Profiling (PyTorch, this Mac)

Method: `kokoro_model.infer()` unchanged (english-practice code imported read-only), `torch.manual_seed(0)` via
`seed=0`, voice `af_heart` row `pack[N-1]`, random valid vocab ids (numpy seed 0) for N = 16, 64, 128, 256, 510,
plus three real sentences phonemized by misaki (N = 28, 117, 394). Totals: `bench_eager.py`, 3 warm-up + 9 runs,
median, IQR. Module breakdown: forward pre/post hooks with `torch.mps.synchronize()` on MPS
(`profile_kokoro.py`, 2–3 warm-up + 7 runs). Op breakdown: `torch.profiler` (CPU activity) of one warm run.
torch uses 8 intra-op threads (Accelerate handles its own GEMM threading; `torch.set_num_threads(1)` does not
change sgemm speed). The machine is a working desktop (Chrome, VS Code, trustd ≈ 0.5 core): CPU IQR is 1–5 %,
MPS IQR is 1–15 % at long lengths.

### 4.1 Totals (eager, f32)

| case | T | F | audio | CPU median (IQR) | CPU RTF | MPS median (IQR) | MPS RTF |
|---|---|---|---|---|---|---|---|
| rand16 | 18 | 74 | 1.85 s | 288 ms (7) | 0.156 | 158 ms (62) | 0.085 |
| real_short28 | 30 | 86 | 2.15 s | 315 ms (5) | 0.146 | 160 ms (4) | 0.074 |
| rand64 | 66 | 170 | 4.25 s | 540 ms (12) | 0.127 | 250 ms (3) | 0.059 |
| real_medium117 | 119 | 278 | 6.95 s | 850 ms (15) | 0.122 | 381 ms (10) | 0.055 |
| rand128 | 130 | 334 | 8.35 s | 995 ms (17) | 0.119 | 466 ms (102) | 0.056 |
| rand256 | 258 | 680 | 17.0 s | 1848 ms (83) | 0.109 | 992 ms (166) | 0.058 |
| real_long394 | 396 | 935 | 23.4 s | 2556 ms (108) | 0.109 | 1501 ms (246) | 0.064 |
| rand510 | 512 | 1127 | 28.2 s | 3090 ms (145) | 0.110 | 1703 ms (192) | 0.060 |

CPU cost ≈ 2.65 ms/frame + ~90 ms fixed; MPS ≈ 1.45 ms/frame + ~50–100 ms fixed.
Ratio per frame: PyTorch CPU sustains ~250 GMAC/s (≈ 0.5 TFLOP/s) on this model.

### 4.2 By module (share of hooked total; hooks inflate absolute CPU time, shares are what matter)

| group | CPU real117 | CPU rand510 | MPS real117 | MPS rand510 |
|---|---|---|---|---|
| ALBERT (12 layers) | 5.7 % | 4.0 % | 3.5 % | 3.3 % |
| DurationEncoder (3 packed BiLSTM + AdaLN) | 1.7 % | 2.3 % | **26.6 %** | **28.7 %** |
| predictor.lstm + duration_proj | 0.4 % | 0.5 % | 1.1 % | 0.9 % |
| shared BiLSTM over F | 0.9 % | 0.9 % | 2.0 % | 1.8 % |
| F0/N AdainResBlk1d ×6 + proj | 2.1 % | 1.4 % | 3.5 % | 1.2 % |
| TextEncoder (CNN + packed BiLSTM) | 1.0 % | 0.7 % | **9.7 %** | **9.6 %** |
| Decoder encode + decode[0..3] + asr_res | 3.8 % | 2.5 % | 5.1 % | 2.2 % |
| Gen: harmonic source (SineGen) + STFT | 3.2 % | 3.5 % | 0.6 % | 0.3 % |
| **Gen stage 0** (256 ch × 20F): noise_conv0/res0, ups0, resblocks 0–2 | **22.9 %** | **20.8 %** | 15.1 % | 11.8 % |
| **Gen stage 1** (128 ch × 120F): noise_conv1/res1, ups1, resblocks 3–5 | **49.4 %** | **56.1 %** | **28.0 %** | **39.4 %** |
| Gen: conv_post + iSTFT | 0.9 % | 1.0 % | 0.6 % | 0.9 % |
| other (alignment, cat, glue) | 8.0 % | 6.4 % | 4.2 % | ~0 % |

Within stage 1 (CPU rand510): noise_res[1] (k11) 26 %, resblocks[5] (k11) 21 %, resblocks[4] (k7) 12 %,
resblocks[3] (k3) 8 % of stage — cost ∝ kernel size as expected.

**MPS LSTM finding** (`lstm_check.py`, BiLSTM 640→256, median): plain vs `pack_padded_sequence`:

| T | CPU plain | CPU packed | MPS plain | MPS packed |
|---|---|---|---|---|
| 130 | 7.4 ms | 7.9 ms | 6.9 ms | 45.9 ms |
| 512 | 27.3 ms | 27.6 ms | 16.6 ms | **159.2 ms** |
| 1127 | 78.1 ms | 60.7 ms | 30.6 ms | **346.6 ms** |

PyTorch-MPS loses ~35 % of its forward to packing that is a no-op at batch 1 (also `x_pad = torch.zeros(...)`
CPU allocations copied to MPS). PyTorch CPU LSTMs are ~50 µs/step — ~10× above what a dedicated
recurrent kernel should do (§6).

### 4.3 By op type (CPU, torch.profiler self time)

| op | rand128 (1067 ms) | rand510 (3283 ms) | real394 (2867 ms) |
|---|---|---|---|
| `_slow_conv2d_forward` (non-dilated conv1d via im2col+sgemm) | 29.3 % | 33.9 % | 32.8 % |
| `slow_conv_dilated2d` (dilated conv1d) | 19.5 % | 22.1 % | 22.3 % |
| `native_batch_norm` (InstanceNorm1d) | 9.6 % | 8.0 % | 8.4 % |
| `sin` | 6.3 % | 6.1 % | 6.6 % |
| `addmm` (ALBERT/LSTM linears) | 5.2 % | 4.3 % | 4.4 % |
| `mul` | 4.1 % | 4.3 % | 4.3 % |
| `normal_` (randn) | 3.4 % | 3.7 % | 3.5 % |
| `add` | 3.6 % | 3.4 % | 3.5 % |
| `copy_` | 5.2 % | 2.2 % | 2.9 % |
| `slow_conv_transpose2d` (ups + depthwise pool, 2116 calls) | 2.7 % | 2.5 % | 2.4 % |
| `pow` (Snake sin²) | 1.0 % | 1.0 % | 1.0 % |
| `_weight_norm_interface` | 1.0 % | 0.3 % | 0.3 % |
| SDPA (flash-attn-for-cpu) | 0.2 % | 0.6 % | 0.4 % |
| `lstm` + per-step tanh/sigmoid/add_ etc. | ~3 % | ~3 % | ~3 % |

So **~56 % convolutions, ~25 % elementwise/normalization around them, ~4 % RNG, ~5 % GEMM for ALBERT/LSTM**.
`torch.sin` alone runs at 0.5 G elem/s (128×135 000 in 32 ms) — a vectorized polynomial is ~10× faster.

### 4.4 Hardware throughput reference (measured, f32)

| op | CPU (Accelerate/AMX via torch) | MPS |
|---|---|---|
| sgemm 2048³ | 1.67 TFLOP/s | 5.2 TFLOP/s |
| sgemm 128×1408×135k (stage-1 k11 im2col shape) | 0.95 | 3.5 |
| sgemm 128×128×135k (one tap of a shifted-GEMM conv) | 1.00 | – |
| sgemm 256×256×22.5k (stage-0 tap) | 1.39–1.45 | – |
| sgemm 256×1792×22.5k (stage-0 k7 im2col) | 1.31 | 7.5 |
| sgemm 1024×3270×1127 (decoder k3 im2col) | 1.6–2.0 | 4.9 |
| torch conv1d 128ch k11 d1 L=135k | 0.80 | 5.75 |
| torch conv1d 128ch k11 d5 L=135k | 0.54 | 5.2 |
| torch conv1d 256ch k7 d3 L=22.5k | 0.85 | 4.75 |

**Floors** (rand510, 1.57 TFLOP): CPU ≈ 0.75 ms/frame stage 1 at ~1 TFLOP/s + 0.31 ms/frame stage 0 at
1.4 TFLOP/s ≈ 1.2 s, + ~0.2 s for the rest → **~1.4 s target (vs 3.09 s eager, 2.79 s torch.compile)**.
GPU ≈ 1.57 TFLOP at 5 TFLOP/s ≈ 0.31 s + LSTMs/sync → **~0.35–0.45 s target (vs 1.70 s eager MPS, 1.16 s
torch.compile MPS, 1.40 s MLX)**. Memory traffic matters as much as FLOPs: one stage-1 activation at F=1127 is
69 MB; unfused AdaIN+Snake+conv touches it ~15× per conv (≈ 1 GB per conv, 24 stage-1 convs).

### 4.5 iPhone

Not measured here (no Python on iOS; PyTorch has no Metal/MPS path on iOS; ExecuTorch/CoreML/MLX-Swift are the
on-device references). A20 Pro: 2 performance "super" cores + 4 efficiency cores, 7-core GPU with per-core
Neural Accelerators (Metal 4 tensor ops; 2× FP8 vs A19), +50 % memory bandwidth vs A19 Pro. Implications:
CPU kernels must not assume 8 P-cores; the A20 CPU exposes SME/SME2 directly (M2 Max only has Accelerate-gated
AMX); GPU f16 matmuls can use the neural accelerators through Metal 4 tensor APIs — worth a dedicated
investigation once the f32 path works.

---

## 5. Existing implementations / baselines

### 5.1 Python 3.14 / macOS arm64 wheel availability (PyPI, 2026-11-01)

| package | latest | cp314 macOS arm64 wheel? | note |
|---|---|---|---|
| torch | 2.14.1 | yes (in english-practice venv) | eager CPU/MPS + torch.compile work |
| mlx / mlx-metal | 0.32.4 | **yes** | |
| mlx-audio | 0.6.3 | pure Python | depends on mlx==0.32.4, transformers ≥5.14 |
| onnxruntime | 1.31.1 | **yes** (CPU + CoreML EP built in) | |
| kokoro-onnx | 0.6.1 | pure Python but `requires_python <3.14` | not installable; the ONNX model runs fine with plain onnxruntime |
| coremltools | 9.0 (cp ≤3.13); 9.1.dev1 has cp314 | dev only | not tested here |
| executorch | 1.5.1 | yes | not tested (a `software-mansion/Kokoro-82M-ExecuTorch` export exists) |

### 5.2 Measured here (throwaway `uv venv`s in a local scratch directory; english-practice untouched)

All with the same inputs (`inputs.npz`), 2 warm-up + 5 runs (eager: 3 + 9), medians in §0 table.

- **PyTorch eager CPU** (the official baseline) — §4.
- **torch.compile (inductor, `dynamic=True`, `decoder` + `bert` compiled, rest eager)**: CPU 0.90–1.04× eager
  time (5–15 % faster at long inputs; it keeps aten convs), first call 59 s. MPS 1.3–1.6× faster than eager MPS
  (first call 39 s). Note: eager numbers in this run vs compile run were taken at different times; noise ±5 %.
- **ONNX Runtime 1.31.1**, `onnx-community/Kokoro-82M-v1.0-ONNX` `model.onnx` (fp32, 326 MB; same durations as
  torch): CPU EP is **~2× slower** than torch eager (6.3 s at 510). **CoreML EP fails** for both `CPUOnly` and
  `CPUAndGPU`/`ALL` (MLProgram): only 1363/2255 nodes in 43 partitions, and MPSGraph aborts with
  `'mps.matmul' op contracting dimensions differ 1 & 768`. Not usable.
- **MLX (mlx-audio 0.6.3 Kokoro, weights converted f32 from the hexgrad .pth — 548 tensors)**: GPU is the best
  short-input baseline (67 ms at 16 tokens, 260 ms at 117), 1.40 s at 510. **MLX CPU is unusable** (27 s for
  16 tokens, 62 s for 64 — single-threaded convs). Its audio correlates only 0.88 with torch (different numerics).
  mlx-audio also clamps durations to 100 frames/token and does Python-loop alignment.
- **kokoro.cpp (simonfxr, ggml + Metal, F16 GGUF weights, built from the master tarball with pinned ggml
  9be3133 / highway 2607d3b, CMake+Ninja, builds cleanly)**: Metal pipeline 0.35 s (real28), 1.1 s (real117),
  3.1 s (real394), 3.8 s (rand510) — slower than torch MPS; generator dominates (2.9 s of 3.8 s). CPU (8 threads)
  0.9 / 2.7 / 9.6 / 11.6 s — ~4× slower than torch CPU. Not competitive on Apple; its write-up
  (`GPU_OPTIMIZATION.md`, Vulkan on RX 7900 XTX) has useful lessons: time-first (T, C) layout for the generator,
  fused Snake, harmonic source overlapped on CPU (−28 %), batched bi-directional LSTM (−16 %), alignment as
  `get_rows` gathers, single readback of durations.

### 5.3 Not measured (documented)

- **CoreML conversions** (ANE-focused; lokinfer excludes ANE, but they are the on-device state of the art):
  `mattmireles/kokoro-coreml` claims 30 s audio in 159 ms on M2 Ultra, 319 ms on iPhone 15 Pro Max, mostly ANE
  with GPU elementwise; 5 fixed buckets; alignment and harmonic source on the host in Swift; LSTMs on CPU.
  `tigerquoll/kokoro-82m-coreml-cpu` (fp32, CPU-only, dynamic length, native CoreML `lstm`).
  `FluidInference/kokoro-82m-coreml` (FluidAudio): 23.2× real-time vs MLX 23.8× vs PyTorch CPU 17.0×.
  To benchmark: a py3.13 venv with coremltools 9.0 (or the Swift CLIs) and `MLComputeUnits=.cpuOnly` /
  `.cpuAndGPU` for an apples-to-apples CPU/GPU comparison.
- **kokoro-ios** (MLX-Swift): claims 3.3× real-time on iPhone 13 Pro.
- **TTS.cpp** (mmwillet, ggml fork): lists Kokoro, needs a patched ggml, Metal status for Kokoro unclear.
- **kokoro-onnx package**: blocked on py3.14 by its metadata only; equivalent to the ORT run above.

### 5.4 What to beat

Design decision: compare against the best framework per device. Current bests on M2 Max (f32):
- **CPU**: torch.compile CPU ≈ torch eager (eager is the contractual baseline: 3.09 s @510, 850 ms @117).
- **GPU**: MLX for ≲ 130 tokens (67–390 ms), torch.compile-MPS for long inputs (1.16 s @510).

---

## 6. Fusion proposal

Goal: ~20 kernel *families* (~40 instantiations), ~170 dispatches per forward on GPU (vs 146–344 k aten calls).
The compiler does no fusion, so these are hand-written ops exposed in the Python frontend.

### 6.1 Where effort goes (priority order)

1. **Generator AdaIN-Snake-conv chain** (≈ 75 % CPU / ≈ 50 % GPU-eager time; ~90 % of FLOPs). Everything else
   is second order.
2. LSTMs (sequential latency; dominant on PyTorch-MPS only because of packing, but ~6 LSTM passes × T or F
   steps is a latency floor for the GPU path).
3. Harmonic source + STFT (3.5 % CPU, mostly `randn` + `sin`), ALBERT (4–6 %).
4. Decoder / F0-N AdainResBlk1d (≈ 4–6 %).

### 6.2 Kernel families

**Token-rate front end (T ≤ 512)**
1. `albert_embed`: 3 gathers + add + LayerNorm(128) [+ the 128→768 mapping as a GEMM with LN prologue].
2. `gemm_bias[_gelu|_residual_layernorm]`: generic linear with epilogues (QKV fused weight; FFN1+gelu_new;
   out-proj/FFN2 + residual + LN). CPU: Accelerate sgemm + fused epilogue on tiles; GPU: simdgroup-matrix GEMM.
3. `attention`: non-causal, 12×64, T ≤ 512, no mask (flash-style on GPU; on CPU simple tiled QK^T-softmax-V).
4. `bilstm`: input projection = (2); recurrent kernel runs both directions concurrently, fused gate math
   (`b_ih+b_hh` folded), optional fused epilogue (AdaLayerNorm + style concat for DurEnc; output gather for the
   shared LSTM). CPU: one thread per direction, `W_hh` (1024×256 f32 = 1 MB) stays in L2, NEON/AMX matvec,
   target ≤ 5 µs/step. GPU: try (a) one dispatch per K steps with a simdgroup-cooperative matvec, (b) run all
   LSTMs on the CPU in parallel with GPU work (unified memory, zero-copy) — design allows mixed graphs.
5. `duration_head`: Linear 512→50 + sigmoid + sum + /speed + round + clamp → `dur[T]`, prefix sum, `F`
   (the data-dependent dim producer) and `idx[F]` frame→token map.
6. `style_proj`: one GEMV `[49 684 × 128] · s` → all γ/β (with +1 folded). Depends only on `ref_s`.
7. `text_cnn_block`: embedding-gather prologue (first block) + conv k5 512→512 + channel-LN + leaky(0.2).

**Frame-rate**
8. **`conv1d_fused`** — the workhorse, implicit-GEMM 1-D conv with template options:
   - prologue: per-channel affine `a·x+b` (AdaIN folded with stats + style), activation ∈ {none, leaky(α),
     snake(α)}, input transform ∈ {none, gather by `idx` (alignment), nearest-up ×2, depthwise convT k3 s2
     (the `pool`), virtual concat of up to 4 inputs, sum of 3 inputs ×⅓ (resblock mean)};
   - main loop: K ∈ {1,3,5,7,11,12}, dilation ∈ {1,3,5}, stride ∈ {1,2,6};
   - epilogue: bias, residual add (with optional ×1/√2), optional second output (shortcut), and **per-tile
     partial (Σx, Σ(x−m)²) per channel** so the next AdaIN needs no extra pass.
   Instantiations: generator resblock convs (3 sizes × 2 widths), noise_res, AdainResBlk1d convs + 1×1
   shortcuts, noise_convs, F0/N projections.
9. `adain_stats_finalize`: combine partials (Chan) → μ, σ → with γ/β → per-channel (a, b). Tiny; can be folded
   into `conv1d_fused`'s prologue (each threadgroup re-reduces ≤ a few hundred partials).
10. `convT_upsample`: ups[0] (k20 s10) / ups[1] (k12 s6) as polyphase GEMMs (s phases × ⌈k/s⌉ taps), leaky(0.1)
    prologue (+ resblock ⅓-sum prologue), `+ x_source` epilogue, reflection-pad write offset for ups[1].
11. `harmonic_source_stft`: F0 ×300 nearest-up, 9 harmonics, `% 1`, frame-rate cumsum (2F×9, tiny scan),
    linear up-interp, `sin`, uv, noise (explicit input or in-kernel Philox), Linear 9→1, tanh, and the
    n_fft=20/hop=5 STFT as a 22-filter conv (reflect pad 10) + `hypot`/`atan2` → `har[22, 120F+1]`.
    Independent of everything except F0_pred → overlap with TextEncoder/decoder (CPU thread on the GPU path).
12. `post_istft`: leaky(0.01) + conv_post (128→22, k7) + `exp`/`sin` + 20-point inverse real DFT · window +
    overlap-add (each output sample gathers its 4 frames — no atomics) + `1/Σw²` envelope + trim → audio.
13. `rng_normal` (Philox 4×32 + Box-Muller), fused into (11) for production.

Not kernels anymore (folded at load or no-ops): weight-norm, pooler, masks, `pack_padded_sequence`, dropout,
`x_pad` copies, one-hot alignment matmuls, `cat`s, `/√2`, `/3`, `1+γ`.

### 6.3 Generator fusion detail

One AdaINResBlock1 step becomes **two `conv1d_fused` dispatches** instead of ~20 aten ops:
```
xt = conv_d(  snake_α1( a1·x + b1 ) )       prologue: (a1,b1) from stats(x) & γβ;  epilogue: stats(xt)
x  = x + conv_1( snake_α2( a2·xt + b2 ) )   prologue: (a2,b2) from stats(xt);      epilogue: stats(x), residual
```
Instance-norm stats are global over time, so each conv is a grid-wide barrier — the fusion boundary is one
conv. Stage-1 data traffic per conv drops from ~15 passes to ~2 (read input once with halo, write output once).

The three resblocks of a stage read the same `x` and share its stats (only γ/β/α differ): either one kernel
computing all three first convs from one loaded tile (halo = max 25 samples), or three dispatches that reuse the
stats. The stage output `mean_j` is never materialized — the consumer (`convT_upsample` / `post_istft`)
sums three inputs in its prologue.

Concurrency: noise branch (`noise_convs[i]` → `noise_res[i]`) is independent of the main branch until the add →
concurrent dispatch on GPU / separate thread group on CPU.

### 6.4 CPU strategy

- GEMM engine: Accelerate `cblas_sgemm` (the only access to AMX on M2; 1.0–2.0 TFLOP/s measured on the conv
  shapes) — conv as **K shifted GEMMs over a zero-padded time-major buffer** (no im2col: a tap is just a pointer
  offset; 1.0 TFLOP/s at K=128, 1.4 at K=256) or im2col on L2-sized time tiles. Benchmark both.
- Time-tiling: process the length in tiles (e.g. 2–8 k samples) so prologue output, GEMM and epilogue stay in
  L2; write only the final tile result + stats partials.
- Threading: Accelerate threads internally (`torch.set_num_threads` had no effect on sgemm); with 2 AMX units
  (one per P-cluster) prefer ≤ 2 concurrent GEMM streams + NEON elementwise on the other cores; measure.
- Vector math: polynomial `sin/cos/exp/tanh/sigmoid` in NEON (vForce `vvsinf` as a baseline).
- iPhone A20: 2 P-cores; SME2 kernels are an option there (not on M2).

### 6.5 GPU strategy

- Implicit-GEMM conv with `simdgroup_matrix` f32 (later f16 / Metal 4 tensor ops on A20), channels padded to
  multiples of 8/32 at load (514→520, 1090→1096, 22→24).
- Layout: evaluate time-major `[L, C]` (kokoro.cpp found it better for the generator) vs `[C, L]`.
- One readback (F) per forward; record/replay the F-independent prefix; encode the rest after the readback.
- LSTMs: see (4); measured PyTorch-MPS plain LSTM is 16.6 ms at T=512, so even naïve is fine relative to the
  current packed 159 ms.
- f16 activations halve bandwidth on the 128×120F tensors; keep f32 accumulation and f32 stats.

### 6.6 Expected wins (back-of-envelope, rand510)

| part | PyTorch CPU now | lokinfer CPU target | PyTorch MPS now | lokinfer GPU target |
|---|---|---|---|---|
| generator convs + AdaIN/Snake | ~2.6 s | ~1.2 s | ~0.9 s | ~0.30 s |
| harmonic source + STFT/iSTFT | ~0.15–0.2 s | ~0.03 s | ~0.02 s | ~0.01 s |
| LSTMs (6) | ~0.15–0.25 s | ~0.03 s | ~0.66 s | ~0.03 s |
| ALBERT + rest | ~0.2 s | ~0.1 s | ~0.1 s | ~0.05 s |
| **total** | **3.09 s** | **~1.4 s** | **1.70 s** | **~0.4 s** |

---

## 7. Sources

- kokoro 0.7.16 source (`model.py`, `modules.py`, `istftnet.py`), transformers 5.20.2 `modeling_albert.py`,
  english-practice `kokoro_model.py`, HF config/weights snapshot `f3ff3571…` (local).
- [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) (`mlx_audio/tts/models/kokoro`),
  [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16),
  [NaN report](https://huggingface.co/mlx-community/Kokoro-82M-bf16/discussions/5)
- [simonfxr/kokoro.cpp](https://github.com/simonfxr/kokoro.cpp) + `GPU_OPTIMIZATION.md`,
  [simonfxr/kokoro.cpp-GGUF](https://huggingface.co/simonfxr/kokoro.cpp-GGUF)
- [mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml),
  [tigerquoll/kokoro-82m-coreml-cpu](https://huggingface.co/tigerquoll/kokoro-82m-coreml-cpu),
  [mlboydaisuke/Kokoro-82M-CoreML](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreML),
  [FluidAudio benchmarks](https://docs.fluidinference.com/reference/benchmarks.md)
- [ovos-tts-plugin-kokoro](https://pypi.org/project/ovos-tts-plugin-kokoro/) (reports CPU faster than MPS for Kokoro)
- [kokoro-ios](https://backiee.wasmer.app/https_github_com/mlalma/kokoro-ios), [TTS.cpp fork](https://github.com/coderj001/TTS.cpp)
- A20 Pro: [MacRumors](https://macrumors.com/2026/09/09/apple-unveils-a20-pro-as-first-2nm-smartphone-chip),
  [Notebookcheck](https://www.notebookcheck.net/Apple-A20-Pro-debuts-with-7-core-GPU-32-core-Neural-Engine-and-50-more-memory-bandwidth.1395027.0.html)
- PyPI JSON API for wheel availability (mlx, mlx-metal, mlx-audio, onnxruntime, coremltools, kokoro-onnx, executorch).
