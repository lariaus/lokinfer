"""Qwen3-ASR-1.7B on lokinfer (docs/impl-notes/qwen3-asr-port-plan.md Q3, op contract qwen3-asr-f16-kernels.md).

    from models.qwen3_asr import model as M, reference as R
    sd = M.fold_weights(R.load_model())        # HF layouts -> lokinfer layouts, f16, once
"""
from __future__ import annotations

import numpy as np

ENC_LAYERS, ENC_D, ENC_HEADS, ENC_FFN, ENC_WINDOW = 24, 1024, 16, 4096, 104
LM_LAYERS, LM_D, LM_QH, LM_KVH, LM_HD, LM_FFN = 28, 2048, 16, 8, 128, 6144
VOCAB, ROPE_THETA, RMS_EPS, LN_EPS = 151936, 1e6, 1e-6, 1e-5
ENC_CLAMP = 65504.0 - 1000.0  # modeling_qwen3_asr.py: the encoder layer output is clamped under f16
CONV_C, CHUNK_ROWS = 480, 13  # conv channels; rows (time steps) per 1 s chunk after the 3 stride-2 convs
CNN_FREQ = 16  # freq bins after the convs (128 / 8)
VARIANTS = ("basic", "opt")  # graph variants; "opt" = basic + obvious fusions (plan Q5)
OPT_DEVICES = ("cpu", "gpu")  # devices whose optimized kernels have landed (Q5)


def _f16(t) -> np.ndarray:
    return np.ascontiguousarray(t.detach().float().cpu().numpy().astype(np.float16))


def fold_weights(hf_model) -> dict[str, np.ndarray]:
    """HF state -> lokinfer layouts, f16 (the checkpoint is bf16: converted once, here).

    Conv weights: HF [Co, Ci, kf, kt] (H = freq, W = time) -> [Co, kt, kf, Ci] (NHWC with H = time, W = freq).
    conv_out: HF Linear [1024, 7680] over (channel, freq) -> [7680, 1024] over (freq, channel) (our NHWC flatten).
    Linear weights: [out, in] -> [in, out] (GEMM [K, N]). The tied embedding is stored once ("embed").
    """
    m = hf_model.model
    enc, lm, pj = m.audio_tower, m.language_model, m.multi_modal_projector
    sd: dict[str, np.ndarray] = {}
    for i, conv in enumerate((enc.conv2d1, enc.conv2d2, enc.conv2d3), 1):
        sd[f"enc.conv{i}.w"] = _f16(conv.weight.permute(0, 3, 2, 1))
        sd[f"enc.conv{i}.b"] = _f16(conv.bias)
    w = enc.conv_out.weight  # [1024, 480 * 16], input index c * 16 + f
    sd["enc.conv_out.w"] = _f16(w.reshape(ENC_D, CONV_C, CNN_FREQ).permute(2, 1, 0).reshape(CNN_FREQ * CONV_C, ENC_D))
    sd["enc.pos"] = _f16(enc.positional_embedding.positional_embedding[:CHUNK_ROWS])
    for i, L in enumerate(enc.layers):
        p = f"enc.{i}."
        a = L.self_attn
        for n in ("q", "k", "v"):
            sd[p + n + ".w"] = _f16(getattr(a, f"{n}_proj").weight.t())
            sd[p + n + ".b"] = _f16(getattr(a, f"{n}_proj").bias)
        sd[p + "o.w"], sd[p + "o.b"] = _f16(a.out_proj.weight.t()), _f16(a.out_proj.bias)
        sd[p + "ln1.g"], sd[p + "ln1.b"] = _f16(L.self_attn_layer_norm.weight), _f16(L.self_attn_layer_norm.bias)
        sd[p + "ln2.g"], sd[p + "ln2.b"] = _f16(L.final_layer_norm.weight), _f16(L.final_layer_norm.bias)
        sd[p + "fc1.w"], sd[p + "fc1.b"] = _f16(L.fc1.weight.t()), _f16(L.fc1.bias)
        sd[p + "fc2.w"], sd[p + "fc2.b"] = _f16(L.fc2.weight.t()), _f16(L.fc2.bias)
    sd["enc.ln_post.g"], sd["enc.ln_post.b"] = _f16(enc.ln_post.weight), _f16(enc.ln_post.bias)
    sd["proj1.w"], sd["proj1.b"] = _f16(pj.linear_1.weight.t()), _f16(pj.linear_1.bias)
    sd["proj2.w"], sd["proj2.b"] = _f16(pj.linear_2.weight.t()), _f16(pj.linear_2.bias)
    sd["embed"] = _f16(lm.embed_tokens.weight)  # [V, 2048]: gather table AND the tied LM head
    for i, L in enumerate(lm.layers):
        p = f"lm.{i}."
        a, mlp = L.self_attn, L.mlp
        sd[p + "ln1.g"], sd[p + "ln2.g"] = _f16(L.input_layernorm.weight), _f16(L.post_attention_layernorm.weight)
        for n in ("q", "k", "v", "o"):
            sd[p + n + ".w"] = _f16(getattr(a, f"{n}_proj").weight.t())
        sd[p + "qn.g"], sd[p + "kn.g"] = _f16(a.q_norm.weight), _f16(a.k_norm.weight)
        sd[p + "gate.w"], sd[p + "up.w"] = _f16(mlp.gate_proj.weight.t()), _f16(mlp.up_proj.weight.t())
        sd[p + "down.w"] = _f16(mlp.down_proj.weight.t())
    sd["lm.norm.g"] = _f16(lm.norm.weight)
    return sd


# --- folded f16 weights on disk (no HF model object needed; memory = one tensor at a time) ---------------------------

import os  # noqa: E402

WEIGHTS_DIR = os.path.expanduser("~/.cache/lokinfer/qwen3asr/weights_f16")


def _sinusoids(length: int = CHUNK_ROWS, channels: int = ENC_D):
    """The encoder's per-chunk position table (SinusoidsPositionEmbedding, modeling_qwen3_asr.py; not a checkpoint
    tensor: a non-persistent buffer computed at init)."""
    import torch

    inc = np.log(10000) / (channels // 2 - 1)
    inv = torch.exp(-inc * torch.arange(channels // 2).float())
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


def fold_checkpoint(out_dir: str = WEIGHTS_DIR) -> str:
    """fold_weights straight from the pinned safetensors, one tensor at a time, into out_dir/<name>.npy (f16).
    Same arrays as fold_weights(R.load_model()); peak memory ~ the largest tensor. Idempotent."""
    import glob

    import torch
    from safetensors import safe_open

    if os.path.exists(os.path.join(out_dir, "DONE")):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/bcd2b5b7f32b480ab5790554cfa8347f246a14f3"))[0]

    def save(name, t):
        np.save(os.path.join(out_dir, name + ".npy"), _f16(t))

    def tr(t):
        return t.t()

    with safe_open(os.path.join(snap, "model.safetensors"), "pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            k = key.removeprefix("model.")
            if k.startswith("audio_tower.conv2d") and k.endswith(".weight"):
                save(f"enc.conv{k[18]}.w", t.permute(0, 3, 2, 1))
            elif k.startswith("audio_tower.conv2d"):
                save(f"enc.conv{k[18]}.b", t)
            elif k == "audio_tower.conv_out.weight":
                save("enc.conv_out.w", t.reshape(ENC_D, CONV_C, CNN_FREQ).permute(2, 1, 0).reshape(CNN_FREQ * CONV_C, ENC_D))
            elif k.startswith("audio_tower.layers."):
                i, rest = k[len("audio_tower.layers."):].split(".", 1)
                p = f"enc.{i}."
                m = {"self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v", "self_attn.out_proj": "o",
                     "fc1": "fc1", "fc2": "fc2", "self_attn_layer_norm": "ln1", "final_layer_norm": "ln2"}
                mod, kind = rest.rsplit(".", 1)
                if mod in ("self_attn_layer_norm", "final_layer_norm"):
                    save(p + m[mod] + (".g" if kind == "weight" else ".b"), t)
                else:
                    save(p + m[mod] + (".w" if kind == "weight" else ".b"), tr(t) if kind == "weight" else t)
            elif k.startswith("audio_tower.ln_post."):
                save("enc.ln_post." + ("g" if k.endswith("weight") else "b"), t)
            elif k.startswith("multi_modal_projector.linear_"):
                n = k[len("multi_modal_projector.linear_")]
                save(f"proj{n}." + ("w" if k.endswith("weight") else "b"), tr(t) if k.endswith("weight") else t)
            elif k == "language_model.embed_tokens.weight":
                save("embed", t)
            elif k == "language_model.norm.weight":
                save("lm.norm.g", t)
            elif k.startswith("language_model.layers."):
                i, rest = k[len("language_model.layers."):].split(".", 1)
                p = f"lm.{i}."
                names = {"input_layernorm.weight": "ln1.g", "post_attention_layernorm.weight": "ln2.g",
                         "self_attn.q_norm.weight": "qn.g", "self_attn.k_norm.weight": "kn.g"}
                if rest in names:
                    save(p + names[rest], t)
                else:
                    mod = rest.removesuffix(".weight").split(".")[-1]
                    save(p + {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate",
                              "up_proj": "up", "down_proj": "down"}[mod] + ".w", tr(t))
            elif k != "lm_head.weight":  # lm_head is tied to embed_tokens
                raise KeyError(f"unhandled checkpoint tensor {key}")
            del t
    save("enc.pos", _sinusoids())
    open(os.path.join(out_dir, "DONE"), "w").close()
    return out_dir


def load_weights(out_dir: str = WEIGHTS_DIR) -> dict[str, np.ndarray]:
    """The folded f16 weights, memory-mapped (file-backed pages: no extra RAM beyond what the kernels touch)."""
    fold_checkpoint(out_dir)
    return {f[:-4]: np.load(os.path.join(out_dir, f), mmap_mode="r") for f in os.listdir(out_dir) if f.endswith(".npy")}


# --- lokinfer graphs (Q3 basic) ------------------------------------------------------------------------------------

class _GpuOps:
    """lk.ops.f16_gpu under the contract's plain names (the CPU module lk.ops.f16 already uses them)."""

    def __getattr__(self, name):
        import lokinfer as lk

        return getattr(lk.ops.f16_gpu, f"{name}_f16_gpu")


def _ops(device: str):
    import lokinfer as lk

    return lk.ops.f16 if device == "cpu" else _GpuOps()


def _io(device: str):
    """(to_device, to_cpu) for the graph's float inputs / outputs: explicit boundaries on the GPU, identity on the
    CPU. ids / pos stay graph inputs on the CPU (the GPU kernels read them at encode time)."""
    import lokinfer as lk

    if device == "cpu":
        return (lambda x: x), (lambda x: x)
    return lk.ops.to_gpu, lk.ops.to_cpu


def _consts(sd: dict, device: str):
    import lokinfer as lk

    cache: dict = {}

    def c(key):
        if key not in cache:
            cache[key] = lk.from_numpy(sd[key], device=device) if device != "cpu" else lk.from_numpy(sd[key])
        return cache[key]
    return c


def build_encoder(sd: dict, device: str = "cpu", variant: str = "basic"):
    """features f32[n, 128, 100] (the 1 s mel chunks) + n_audio i64[1] (N_a) -> audio f16[N_a, 2048].

    The HF encoder in lokinfer ops: the conv front end chunk by chunk (audio_cnn), per-chunk positions + the first N_a
    rows (add_pos_prefix), 24 pre-LN layers (LayerNorm eps 1e-5, windowed non-causal attention of 104 rows, GELU(erf)
    FFN, the f16 clamp after the second residual), ln_post, projector (Linear, GELU, Linear)."""
    import lokinfer as lk

    assert variant in VARIANTS and (variant != "opt" or device in OPT_DEVICES), f"{variant} not built on {device}"
    O, c = _ops(device), _consts(sd, device)
    to_dev, to_cpu = _io(device)
    n = lk.Sym("n")

    def fn(features, n_audio):
        A = lk.ops.dim_read(n_audio, 0, "A")
        if variant == "opt":
            return to_cpu(_encoder_opt(O, c, to_dev(features), A))
        x = O.audio_cnn(to_dev(features), c("enc.conv1.w"), c("enc.conv1.b"), c("enc.conv2.w"), c("enc.conv2.b"),
                        c("enc.conv3.w"), c("enc.conv3.b"), c("enc.conv_out.w"))
        h = O.add_pos_prefix(x, c("enc.pos"), A)
        for i in range(ENC_LAYERS):
            p = f"enc.{i}."
            a = O.layernorm(h, c(p + "ln1.g"), c(p + "ln1.b"), LN_EPS)
            q = O.gemm_bias(a, c(p + "q.w"), c(p + "q.b"))
            k = O.gemm_bias(a, c(p + "k.w"), c(p + "k.b"))
            v = O.gemm_bias(a, c(p + "v.w"), c(p + "v.b"))
            att = O.attention_window(q, k, v, heads=ENC_HEADS, window=ENC_WINDOW)
            h = O.add(h, O.gemm_bias(att, c(p + "o.w"), c(p + "o.b")))
            a = O.layernorm(h, c(p + "ln2.g"), c(p + "ln2.b"), LN_EPS)
            f = O.gelu_erf(O.gemm_bias(a, c(p + "fc1.w"), c(p + "fc1.b")))
            h = O.add_clamp(h, O.gemm_bias(f, c(p + "fc2.w"), c(p + "fc2.b")), ENC_CLAMP)
        h = O.layernorm(h, c("enc.ln_post.g"), c("enc.ln_post.b"), LN_EPS)
        h = O.gelu_erf(O.gemm_bias(h, c("proj1.w"), c("proj1.b")))
        return to_cpu(O.gemm_bias(h, c("proj2.w"), c("proj2.b")))

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [n, 128, 100]), lk.TensorType(lk.i64, [1])])


def build_generate(sd: dict, device: str = "cpu", variant: str = "basic", debug_layers: bool = False,
                   debug_norm: bool = False):
    """Prefill chunks and decode steps of the LM, the KV cache being a caller-owned MUTABLE input:
    ids i64[T] (T <= 2048 new tokens), audio f16[A, 2048] (the rows the chunk's <|audio_pad|> ids take, in order; any
    1-row tensor when there are none), pos i64[1] (tokens already in the cache), kv f16[28, 2, P, 1024] (layers x K/V
    x positions x kv width; written in place, P = the caller's capacity, e.g. kv_cache(...)) -> f32[1, 151936] logits
    of the last new token. Qwen3: RMSNorm eps 1e-6, QK-norm then RoPE (theta 1e6), causal GQA 16 / 8 heads x 128,
    SwiGLU. One compiled model serves any capacity and any number of sequences (one kv per sequence)."""
    import lokinfer as lk
    from lokinfer.graph import current_graph
    from lokinfer.ops._def import _as_value

    assert variant in VARIANTS and (variant != "opt" or device in OPT_DEVICES), f"{variant} not built on {device}"
    O, c = _ops(device), _consts(sd, device)
    to_dev, to_cpu = _io(device)
    T, A, P = lk.Sym("T"), lk.Sym("A"), lk.Sym("P")
    W = LM_KVH * LM_HD

    def cache_view(kv, layer: int, which: int):  # kv[layer, which] as a [P, 1024] buffer (a view of the caller's)
        g = current_graph()
        return g.view(_as_value(g, kv), lk.MutableType(lk.f16, [P, W]), (2 * layer + which) * P * W)

    def last_row(x):  # debug output: the last row of a [T, D] activation, as a copy on the CPU
        g = current_graph()
        return to_cpu(O.add(g.view(_as_value(g, x), lk.TensorType(lk.f16, [1, LM_D]), (T - 1) * LM_D),
                            lk.from_numpy(np.zeros((1, LM_D), np.float16), **({} if device == "cpu" else {"device": "gpu"}))))

    def fn(ids, audio, pos, kv):
        h = O.scatter_audio(O.embedding(ids, c("embed")), ids, to_dev(audio))
        rows = []
        for i in range(LM_LAYERS):
            p = f"lm.{i}."
            kc, vc = cache_view(kv, i, 0), cache_view(kv, i, 1)
            if variant == "opt":
                h = _lm_layer_opt(O, c, i, h, pos, kc, vc)
                if debug_layers:
                    rows.append(last_row(h))
                continue
            a = O.rmsnorm(h, c(p + "ln1.g"), RMS_EPS)
            q = O.rope(O.rmsnorm_heads(O.gemm(a, c(p + "q.w")), c(p + "qn.g"), RMS_EPS), pos, ROPE_THETA)
            k = O.rope(O.rmsnorm_heads(O.gemm(a, c(p + "k.w")), c(p + "kn.g"), RMS_EPS), pos, ROPE_THETA)
            O.kv_write(kc, vc, k, O.gemm(a, c(p + "v.w")), pos)
            att = O.attention_gqa(q, kc, vc, pos, q_heads=LM_QH, kv_heads=LM_KVH)
            h = O.add(h, O.gemm(att, c(p + "o.w")))
            a = O.rmsnorm(h, c(p + "ln2.g"), RMS_EPS)
            h = O.add(h, O.gemm(O.silu_mul(O.gemm(a, c(p + "gate.w")), O.gemm(a, c(p + "up.w"))), c(p + "down.w")))
        head = O.lm_head_last_opt if variant == "opt" else O.lm_head_last
        normed = O.rmsnorm(h, c("lm.norm.g"), RMS_EPS)
        logits = to_cpu(head(normed, c("embed")))
        extra = [to_cpu(normed)] if debug_norm else []  # tests: every row's final hidden state (golden "lm_norm")
        return (logits, *extra, *rows) if (debug_layers or debug_norm) else logits

    return lk.compile(fn, input_types=[lk.TensorType(lk.i64, [T]), lk.TensorType(lk.f16, [A, LM_D]),
                                       lk.TensorType(lk.i64, [1]), lk.MutableType(lk.f16, [LM_LAYERS, 2, P, W])])


MAX_CAP = {"mac": 25_600, "iphone": 6_656}  # one 20 min pass on the Mac, one 5 min pass on the iPhone (the plan)


def kv_cache(capacity: int, device: str = "cpu"):
    """A zeroed KV cache for `capacity` positions (prompt + generated tokens): f16[28, 2, capacity, 1024], 114,688 B
    per position. CPU: a numpy array; GPU: a lokinfer GPU tensor (bound by the GPU model without a copy)."""
    import lokinfer as lk

    shape = (LM_LAYERS, 2, capacity, LM_KVH * LM_HD)
    if device == "cpu":
        return np.zeros(shape, np.float16)
    return lk.zeros(*shape, dtype=lk.f16, device="gpu")


AUDIO_START = 9  # the <|audio_pad|> run starts after the 9 prefix ids (reference.PREFIX)
EOS = (151643, 151645)


def transcribe(enc, gen, features: np.ndarray, n_audio: int, ids: np.ndarray, max_new: int, chunk: int = 2048,
               on_prefill=None, kv=None, device: str = "cpu") -> list[int]:
    """The caller contract's loop (plan "The caller contract"): encoder once, prefill in chunks of <= `chunk` ids
    (each chunk gets the audio rows its pad ids take), then greedy decode until eos or max_new tokens.
    kv: the caller's cache (kv_cache(capacity, device)); by default one sized for prompt + max_new."""
    if kv is None:
        kv = kv_cache(len(ids) + max_new, device)
    audio = enc(features, np.array([n_audio], np.int64)).numpy()
    dummy = np.zeros((1, LM_D), np.float16)
    T = len(ids)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        k0, k1 = min(max(s - AUDIO_START, 0), n_audio), min(max(e - AUDIO_START, 0), n_audio)
        rows = audio[k0:k1] if k1 > k0 else dummy
        logits = gen(np.ascontiguousarray(ids[s:e]), np.ascontiguousarray(rows), np.array([s], np.int64), kv).numpy()
    if on_prefill is not None:
        on_prefill(logits)
    toks, pos = [], T
    while True:
        tok = int(logits.argmax())
        toks.append(tok)
        if tok in EOS or len(toks) >= max_new:
            return toks
        logits = gen(np.array([tok], np.int64), dummy, np.array([pos], np.int64), kv).numpy()
        pos += 1


# --- optimized variant (Q5, docs/impl-notes/qwen3-asr-f16-opt-kernels.md) -------------------------------------------

OPT_DIR = os.path.expanduser("~/.cache/lokinfer/qwen3asr/weights_f16_opt")


def fold_opt(out_dir: str = OPT_DIR, base: str = WEIGHTS_DIR) -> str:
    """The concatenated weights the fused GEMMs take, written once to disk (memory-mapped later, never held next to
    the separate ones in RAM): LM qkv [2048, 4096] and gate|up [2048, 12288] per layer, encoder qkv [1024, 3072] (+ its
    bias [3072]). One layer at a time."""
    if os.path.exists(os.path.join(out_dir, "DONE")):
        return out_dir
    sd = load_weights(base)
    os.makedirs(out_dir, exist_ok=True)

    def save(name, a):
        np.save(os.path.join(out_dir, name + ".npy"), np.ascontiguousarray(a, np.float16))

    for i in range(LM_LAYERS):
        p = f"lm.{i}."
        save(p + "qkv.w", np.concatenate([sd[p + "q.w"], sd[p + "k.w"], sd[p + "v.w"]], axis=1))
        save(p + "gateup.w", np.concatenate([sd[p + "gate.w"], sd[p + "up.w"]], axis=1))
    for i in range(ENC_LAYERS):
        p = f"enc.{i}."
        save(p + "qkv.w", np.concatenate([sd[p + "q.w"], sd[p + "k.w"], sd[p + "v.w"]], axis=1))
        save(p + "qkv.b", np.concatenate([sd[p + "q.b"], sd[p + "k.b"], sd[p + "v.b"]]))
    open(os.path.join(out_dir, "DONE"), "w").close()
    return out_dir


def load_weights_opt() -> dict[str, np.ndarray]:
    """The base folded weights + the concatenated ones (both memory-mapped)."""
    sd = load_weights()
    fold_opt()
    sd.update({f[:-4]: np.load(os.path.join(OPT_DIR, f), mmap_mode="r") for f in os.listdir(OPT_DIR) if f.endswith(".npy")})
    return sd


def _encoder_opt(O, c, features, A):
    x = O.audio_cnn_opt(features, c("enc.conv1.w"), c("enc.conv1.b"), c("enc.conv2.w"), c("enc.conv2.b"),
                        c("enc.conv3.w"), c("enc.conv3.b"), c("enc.conv_out.w"))
    h = O.add_pos_prefix(x, c("enc.pos"), A)
    for i in range(ENC_LAYERS):
        p = f"enc.{i}."
        a = O.layernorm(h, c(p + "ln1.g"), c(p + "ln1.b"), LN_EPS)
        att = O.attention_window_qkv_opt(O.gemm_bias_opt(a, c(p + "qkv.w"), c(p + "qkv.b")),
                                          heads=ENC_HEADS, window=ENC_WINDOW)
        h = O.gemm_bias_add_opt(att, c(p + "o.w"), c(p + "o.b"), h)
        a = O.layernorm(h, c(p + "ln2.g"), c(p + "ln2.b"), LN_EPS)
        f = O.gemm_bias_gelu_opt(a, c(p + "fc1.w"), c(p + "fc1.b"))
        h = O.gemm_bias_add_clamp_opt(f, c(p + "fc2.w"), c(p + "fc2.b"), h, ENC_CLAMP)
    h = O.layernorm(h, c("enc.ln_post.g"), c("enc.ln_post.b"), LN_EPS)
    h = O.gemm_bias_gelu_opt(h, c("proj1.w"), c("proj1.b"))
    return O.gemm_bias_opt(h, c("proj2.w"), c("proj2.b"))


def _lm_layer_opt(O, c, i, h, pos, kc, vc):
    p = f"lm.{i}."
    # rmsnorm fused into the next GEMM (one dispatch per decode step on the GPU; ws = workspace, unused)
    qkv, _ = O.rmsnorm_gemm_opt(h, c(p + "ln1.g"), c(p + "qkv.w"), RMS_EPS)
    q = O.qknorm_rope_kv_opt(qkv, c(p + "qn.g"), c(p + "kn.g"), pos, kc, vc, RMS_EPS, ROPE_THETA)
    att = O.attention_gqa_opt(q, kc, vc, pos, q_heads=LM_QH, kv_heads=LM_KVH)
    h = O.gemm_add_opt(att, c(p + "o.w"), h)
    mlp, _ = O.rmsnorm_gemm_swiglu_opt(h, c(p + "ln2.g"), c(p + "gateup.w"), RMS_EPS)
    return O.gemm_add_opt(mlp, c(p + "down.w"), h)
