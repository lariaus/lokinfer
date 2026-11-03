"""GPT-2 124M prefill on lokinfer (docs/impl-notes/research-models.md §5, docs/models/gpt2.md).

Reference: HF `GPT2LMHeadModel` "openai-community/gpt2" at a pinned revision. lokinfer graph (B = 1, T symbolic):
token ids i64[T] -> logits f32[T, 50257] (all positions; `last_only=True` returns the last row f32[1, 50257]).
Pre-LN blocks: LN -> fused QKV GEMM (HF's Conv1D weights are already [in, out]) -> causal attention reading q/k/v from
the fused projection -> projection GEMM + residual -> LN -> GEMM + gelu_tanh -> GEMM + residual; final LN; the tied
LM head is a transposed-B GEMM against wte itself (no transposed copy).
CPU: Accelerate GEMMs, attention heads on the pool. GPU: MPS GEMMs, attention / bias+gelu / add+LN kernels; the
embedding gather stays on the CPU (one cpu_to_gpu sync).
"""
from __future__ import annotations

import numpy as np

import lokinfer as lk
from lokinfer import ops
from lokinfer.ops.transformer import (add_bias_layernorm2_f32_cpu, add_bias_layernorm2_sk_f32_gpu,
                                      attention_decode_kvw_f32_gpu, attention_decode_split_f32_gpu,
                                      attention_gen2_f32_gpu, attention_gen_f32_gpu,
                                      attention_qkv_causal_blk_f32_cpu, attention_qkv_sg_f32_gpu, gemm_accel_f32_cpu,
                                      gemm_auto_bias_gelu_f32_gpu, gemm_auto_f32_gpu, gemm_auto_sk_f32_gpu,
                                      gemm_splitk_f32_gpu,
                                      gemv4_f32_gpu)

HF_REPO = "openai-community/gpt2"
HF_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
N_LAYER, D, HEADS, VOCAB, N_POS = 12, 768, 12, 50257, 1024
LN_EPS = 1e-5


def torch_reference():
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(HF_REPO, revision=HF_REV, attn_implementation="eager").eval()


def weights(model) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}


def _gpu_gemms(kind: str, cw: int = 0):
    """(plain GEMM, GEMM+bias+gelu, LM-head GEMM) for the GPU graph: "mps" (prefill default: faster on GPT-2's shapes
    than steel, A/B 2026-11-01: prefill T=512 23.5 vs 26.1 ms), "steel" (lokinfer's steel GEMM with fused epilogues),
    "gemv4" (decode default: float4-column GEMVs with fused bias+gelu, Mac 2.05 vs 2.7 ms with MPS, iPhone 5.6 vs
    6.9 ms; cw = float4 columns per threadgroup, 0 = per-platform default) or "gemv" (the first GEMV, slower)."""
    if kind == "gemv4":  # decode GEMV v2 (float4 columns x K slices, fused bias+gelu); LM head stays MPS
        return (lambda x, w: gemv4_f32_gpu(x, w, cw=cw), lambda x, w, b: gemv4_f32_gpu(x, w, bias=b, act="gelu", cw=cw),
                ops.gemm_nt_f32_gpu_mps)
    if kind == "gemv":  # decode GEMVs (kokoro_tok.metal): slower than MPS on the M2 Max (5.2 vs 3.6 ms, 2026-11-01)
        return ops.gemv_f32_gpu, lambda x, w, b: ops.gemv_f32_gpu(x, w, bias=b, act="gelu"), ops.gemv_rows_f32_gpu
    if kind == "steel":
        return (ops.gemm_steel_f32_gpu, lambda x, w, b: ops.gemm_steel_f32_gpu(x, w, bias=b, act="gelu"),
                ops.gemm_nt_steel_f32_gpu)
    return ops.gemm_f32_gpu_mps, lambda x, w, b: ops.bias_gelu_f32_gpu(ops.gemm_f32_gpu_mps(x, w), b), \
        ops.gemm_nt_f32_gpu_mps


def build(sd: dict[str, np.ndarray], device: str = "cpu", last_only: bool = False, gemm: str = "mps", attn: str = "sg",
          nsg: int = 4, cpu: str = "fused", split: tuple[int, int, int, int] = (0, 0, 0, 0), act: str = "gelu",
          cattn: tuple[int, int] | None = (128, 0), sk: tuple[int, int] = (2, 4), sk_tile: int = 32):
    """attn (GPU): "sg" = simdgroup-matrix flash attention (lk_attention_qkv_sg), "flash" = the scalar flash kernel.
    sk (GPU): split-K factors of the attention / MLP output projections (N = 768), whose partials the add+LayerNorm
    kernel sums; (0, 0) = MPS; the kernel falls back to MPS when the output tiles fill the GPU (sk_tile: steel tile).
    CPU: cpu="fused" (default) = Accelerate GEMMs with fused epilogues (gemm_accel_f32_cpu; split per GEMM, 0 = the
    per-platform auto blocking), add+bias+LayerNorm in one pass, cattn = (query block, mode) of the causal blocked
    attention (None = the old full-matrix kernel); act "gelu" (NEON exp) or "gelu_vv" (vvtanhf); cpu="basic" = the unfused graph."""
    gpu = device == "gpu"
    T = lk.Sym("T")
    cpu_t = {}

    def c(key):  # CPU constant
        if key not in cpu_t:
            cpu_t[key] = lk.from_numpy(np.ascontiguousarray(sd[key]))
        return cpu_t[key]

    gpu_t = {}

    def g(key):  # GPU constant
        if key not in gpu_t:
            gpu_t[key] = lk.from_numpy(np.ascontiguousarray(sd[key]), device="gpu", copy=True)
        return gpu_t[key]

    P = "transformer."

    def fn(ids):
        h = ops.embedding_f32_cpu(c(P + "wte.weight"), ids)
        h = ops.add_f32_cpu(h, _view(c(P + "wpe.weight"), [T, D]))  # wpe[0:T]
        if not gpu and cpu == "fused":
            return _cpu_fused_blocks(h, c, T, last_only, split, act, cattn)
        if not gpu:
            for i in range(N_LAYER):
                L = f"{P}h.{i}."
                a = ops.layernorm_f32_cpu(h, c(L + "ln_1.weight"), c(L + "ln_1.bias"), eps=LN_EPS)
                qkv = ops.gemm_bias_f32_cpu_accelerate(a, c(L + "attn.c_attn.weight"), c(L + "attn.c_attn.bias"))
                att = ops.attention_qkv_f32_cpu(qkv, heads=HEADS, causal=True)
                h = ops.add_f32_cpu(h, ops.gemm_bias_f32_cpu_accelerate(att, c(L + "attn.c_proj.weight"),
                                                                        c(L + "attn.c_proj.bias")))
                a = ops.layernorm_f32_cpu(h, c(L + "ln_2.weight"), c(L + "ln_2.bias"), eps=LN_EPS)
                f = ops.gelu_tanh_f32_cpu(ops.gemm_bias_f32_cpu_accelerate(a, c(L + "mlp.c_fc.weight"),
                                                                           c(L + "mlp.c_fc.bias")))
                h = ops.add_f32_cpu(h, ops.gemm_bias_f32_cpu_accelerate(f, c(L + "mlp.c_proj.weight"),
                                                                        c(L + "mlp.c_proj.bias")))
            h = ops.layernorm_f32_cpu(h, c(P + "ln_f.weight"), c(P + "ln_f.bias"), eps=LN_EPS)
            if last_only:
                h = _view(h, [1, D], offset=(T - 1) * D)
            return ops.gemm_nt_f32_cpu_accelerate(h, c(P + "wte.weight"))

        mm, mm_gelu, mm_nt = _gpu_gemms(gemm)

        def proj_ln(x, w, res, ln_key, splits):  # (res + x @ W + b, LN of it); split-K partials summed by the LN
            if splits:
                parts = gemm_splitk_f32_gpu(x, g(w + "weight"), splits, tile=sk_tile)
                return add_bias_layernorm2_sk_f32_gpu(parts, g(w + "bias"), res, g(ln_key + "weight"),
                                                      g(ln_key + "bias"), eps=LN_EPS)
            return ops.add_bias_layernorm2_f32_gpu(mm(x, g(w + "weight")), g(w + "bias"), res, g(ln_key + "weight"),
                                                   g(ln_key + "bias"), eps=LN_EPS)

        attention = (lambda qkv, b, heads, causal: attention_qkv_sg_f32_gpu(qkv, b, heads, causal, nsg=nsg)) \
            if attn == "sg" else ops.attention_qkv_f32_gpu
        h = ops.to_gpu(h)
        ln = ops.layernorm_f32_gpu(h, g(P + "h.0.ln_1.weight"), g(P + "h.0.ln_1.bias"), eps=LN_EPS)
        for i in range(N_LAYER):
            L = f"{P}h.{i}."
            nxt = f"{P}h.{i + 1}.ln_1." if i + 1 < N_LAYER else P + "ln_f."
            att = attention(mm(ln, g(L + "attn.c_attn.weight")), g(L + "attn.c_attn.bias"), heads=HEADS, causal=True)
            h, ln = proj_ln(att, L + "attn.c_proj.", h, L + "ln_2.", sk[0])
            f = mm_gelu(ln, g(L + "mlp.c_fc.weight"), g(L + "mlp.c_fc.bias"))
            h, ln = proj_ln(f, L + "mlp.c_proj.", h, nxt, sk[1])
        if last_only:
            ln = _view(ln, [1, D], offset=(T - 1) * D)
        return ops.to_cpu(mm_nt(ln, g(P + "wte.weight")))

    return lk.compile(fn, input_types=[lk.TensorType(lk.i64, [T])])


def _cpu_fused_blocks(h, c, T, last_only, split, act, cattn=None):
    """CPU prefill blocks with fused epilogues: Accelerate GEMMs with bias (+ gelu) applied per block (gemm_accel_f32_cpu;
    split per GEMM: qkv, attn proj, fc, mlp proj; 0 = auto), the residual add + bias + next LayerNorm as one pass
    (add_bias_layernorm2_f32_cpu), the causal blocked attention (cattn = (query block, mode); None = old kernel)."""
    P = "transformer."
    sq, sp, sf, sm = split
    ln = ops.layernorm_f32_cpu(h, c(P + "h.0.ln_1.weight"), c(P + "h.0.ln_1.bias"), eps=LN_EPS)
    for i in range(N_LAYER):
        L = f"{P}h.{i}."
        nxt = f"{P}h.{i + 1}.ln_1." if i + 1 < N_LAYER else P + "ln_f."
        qkv = gemm_accel_f32_cpu(ln, c(L + "attn.c_attn.weight"), c(L + "attn.c_attn.bias"), split=sq)
        att = ops.attention_qkv_f32_cpu(qkv, heads=HEADS, causal=True) if cattn is None else \
            attention_qkv_causal_blk_f32_cpu(qkv, HEADS, qb=cattn[0], mode=cattn[1])
        o = gemm_accel_f32_cpu(att, c(L + "attn.c_proj.weight"), split=sp)
        h, ln = add_bias_layernorm2_f32_cpu(o, c(L + "attn.c_proj.bias"), h, c(L + "ln_2.weight"), c(L + "ln_2.bias"),
                                            eps=LN_EPS)
        f = gemm_accel_f32_cpu(ln, c(L + "mlp.c_fc.weight"), c(L + "mlp.c_fc.bias"), act=act, split=sf)
        o = gemm_accel_f32_cpu(f, c(L + "mlp.c_proj.weight"), split=sm)
        h, ln = add_bias_layernorm2_f32_cpu(o, c(L + "mlp.c_proj.bias"), h, c(nxt + "weight"), c(nxt + "bias"),
                                            eps=LN_EPS)
    if last_only:
        ln = _view(ln, [1, D], offset=(T - 1) * D)
    return ops.gemm_nt_f32_cpu_accelerate(ln, c(P + "wte.weight"))


def build_decode(sd: dict[str, np.ndarray], device: str = "cpu", gemm: str = "gemv4", attn: str = "split", cw: int = 0,
                 cpu_split: int | None = 0, splits: int = 0, dnsg: int = 0, single: int = 1):
    """One decode step with a persistent KV cache (model-owned state, research-models.md §5.3):
    inputs id i64[1], pos i64[1] (0-based position of this token) -> logits f32[1, 50257]. The per-layer caches
    (f32[1024, 768] for K and V) are `lk.persistent_buffer`s: they persist across calls, the step writes row `pos`
    in place and attends over rows 0..pos. Generation = feed the prompt tokens at pos 0, 1, ..., then the argmax.
    """
    gpu = device == "gpu"
    dev = "gpu" if gpu else "cpu"
    cpu_t, gpu_t = {}, {}

    def c(key):
        if key not in cpu_t:
            cpu_t[key] = lk.from_numpy(np.ascontiguousarray(sd[key]))
        return cpu_t[key]

    def g(key):
        if key not in gpu_t:
            gpu_t[key] = lk.from_numpy(np.ascontiguousarray(sd[key]), device="gpu", copy=True)
        return gpu_t[key]

    P = "transformer."
    mm, mm_gelu, mm_nt = _gpu_gemms(gemm, cw)

    def fn(tok, pos):
        caches = [(lk.persistent_buffer(lk.f32, [N_POS, D], device=dev), lk.persistent_buffer(lk.f32, [N_POS, D],
                                                                                              device=dev))
                  for _ in range(N_LAYER)]
        h = ops.add_f32_cpu(ops.embedding_f32_cpu(c(P + "wte.weight"), tok),
                            ops.embedding_f32_cpu(c(P + "wpe.weight"), pos))  # [1, D]
        if gpu:
            h = ops.to_gpu(h)
            ln = ops.layernorm_f32_gpu(h, g(P + "h.0.ln_1.weight"), g(P + "h.0.ln_1.bias"), eps=LN_EPS)
        for i, (kc, vc) in enumerate(caches):
            L = f"{P}h.{i}."
            if not gpu and cpu_split is not None:  # fused CPU step: K-split GEMVs (cpu_split tasks; 0 = auto)
                if i == 0:
                    ln = ops.layernorm_f32_cpu(h, c(L + "ln_1.weight"), c(L + "ln_1.bias"), eps=LN_EPS)
                nxt = f"{P}h.{i + 1}.ln_1." if i + 1 < N_LAYER else P + "ln_f."
                s_ = cpu_split
                qkv = gemm_accel_f32_cpu(ln, c(L + "attn.c_attn.weight"), c(L + "attn.c_attn.bias"), split=s_)
                ops.kv_write_f32_cpu(kc, vc, qkv, pos)
                att = ops.attention_decode_f32_cpu(qkv, kc, vc, pos, heads=HEADS)
                h, ln = add_bias_layernorm2_f32_cpu(gemm_accel_f32_cpu(att, c(L + "attn.c_proj.weight"), split=s_),
                                                    c(L + "attn.c_proj.bias"), h, c(L + "ln_2.weight"),
                                                    c(L + "ln_2.bias"), eps=LN_EPS)
                f = gemm_accel_f32_cpu(ln, c(L + "mlp.c_fc.weight"), c(L + "mlp.c_fc.bias"), act="gelu", split=s_)
                h, ln = add_bias_layernorm2_f32_cpu(gemm_accel_f32_cpu(f, c(L + "mlp.c_proj.weight"), split=s_),
                                                    c(L + "mlp.c_proj.bias"), h, c(nxt + "weight"), c(nxt + "bias"),
                                                    eps=LN_EPS)
                continue
            if not gpu:
                a = ops.layernorm_f32_cpu(h, c(L + "ln_1.weight"), c(L + "ln_1.bias"), eps=LN_EPS)
                qkv = ops.gemm_bias_f32_cpu_accelerate(a, c(L + "attn.c_attn.weight"), c(L + "attn.c_attn.bias"))
                ops.kv_write_f32_cpu(kc, vc, qkv, pos)
                att = ops.attention_decode_f32_cpu(qkv, kc, vc, pos, heads=HEADS)
                h = ops.add_f32_cpu(h, ops.gemm_bias_f32_cpu_accelerate(att, c(L + "attn.c_proj.weight"),
                                                                        c(L + "attn.c_proj.bias")))
                a = ops.layernorm_f32_cpu(h, c(L + "ln_2.weight"), c(L + "ln_2.bias"), eps=LN_EPS)
                f = ops.gelu_tanh_f32_cpu(ops.gemm_bias_f32_cpu_accelerate(a, c(L + "mlp.c_fc.weight"),
                                                                           c(L + "mlp.c_fc.bias")))
                h = ops.add_f32_cpu(h, ops.gemm_bias_f32_cpu_accelerate(f, c(L + "mlp.c_proj.weight"),
                                                                        c(L + "mlp.c_proj.bias")))
                continue
            nxt = f"{P}h.{i + 1}.ln_1." if i + 1 < N_LAYER else P + "ln_f."
            qkv = mm(ln, g(L + "attn.c_attn.weight"))
            if attn == "split":
                att = attention_decode_split_f32_gpu(qkv, kc, vc, pos, g(L + "attn.c_attn.bias"), heads=HEADS,
                                                     splits=splits, nsg=dnsg, single=single)
            elif attn == "kvw":
                att = attention_decode_kvw_f32_gpu(qkv, kc, vc, pos, g(L + "attn.c_attn.bias"), heads=HEADS)
            else:
                ops.kv_write_f32_gpu(kc, vc, qkv, pos, g(L + "attn.c_attn.bias"))
                att = ops.attention_decode_f32_gpu(qkv, kc, vc, pos, g(L + "attn.c_attn.bias"), heads=HEADS)
            o = mm(att, g(L + "attn.c_proj.weight"))
            h, ln = ops.add_bias_layernorm2_f32_gpu(o, g(L + "attn.c_proj.bias"), h, g(L + "ln_2.weight"),
                                                    g(L + "ln_2.bias"), eps=LN_EPS)
            f = mm_gelu(ln, g(L + "mlp.c_fc.weight"), g(L + "mlp.c_fc.bias"))
            o = mm(f, g(L + "mlp.c_proj.weight"))
            h, ln = ops.add_bias_layernorm2_f32_gpu(o, g(L + "mlp.c_proj.bias"), h, g(nxt + "weight"), g(nxt + "bias"),
                                                    eps=LN_EPS)
        if gpu:
            return ops.to_cpu(mm_nt(ln, g(P + "wte.weight")))
        if cpu_split is not None:
            return ops.gemm_nt_f32_cpu_accelerate(ln, c(P + "wte.weight"))
        h = ops.layernorm_f32_cpu(h, c(P + "ln_f.weight"), c(P + "ln_f.bias"), eps=LN_EPS)
        return ops.gemm_nt_f32_cpu_accelerate(h, c(P + "wte.weight"))

    return lk.compile(fn, input_types=[lk.TensorType(lk.i64, [1]), lk.TensorType(lk.i64, [1])])


def build_generate(sd: dict[str, np.ndarray], device: str = "cpu", kernels: str = "auto", sk: tuple[int, int] = (2, 4)):
    """Chunked prefill + decode in ONE compiled model (so prefill and decode share the persistent KV caches):
    inputs ids i64[T] (T new tokens, symbolic) and pos i64[1] (their start position) -> logits of the last new token
    f32[1, 50257]. Prefill = one call with the whole prompt at pos 0; each decode step = a call with T = 1.
    GPU kernels: "auto" (default) picks per call from T (decode: gemv4 + the key-split decode attention; prefill: MPS
    + the simdgroup-matrix flash attention, over the cache for chunks at pos > 0), "auto1" = the previous auto (decode:
    one-threadgroup-per-head kv-write attention; pos > 0 chunks: scalar flash over cache tiles), "old" = MPS + the
    scalar cache attention for every T. sk ("auto"): split-K factors of the two N = 768 projections for T > 1 (as in
    build(); (0, 0) = MPS)."""
    gpu = device == "gpu"
    dev = "gpu" if gpu else "cpu"
    T = lk.Sym("T")
    cpu_t, gpu_t = {}, {}

    def c(key):
        if key not in cpu_t:
            cpu_t[key] = lk.from_numpy(np.ascontiguousarray(sd[key]))
        return cpu_t[key]

    def g(key):
        if key not in gpu_t:
            gpu_t[key] = lk.from_numpy(np.ascontiguousarray(sd[key]), device="gpu", copy=True)
        return gpu_t[key]

    P = "transformer."

    def fn(ids, pos):
        caches = [(lk.persistent_buffer(lk.f32, [N_POS, D], device=dev),
                   lk.persistent_buffer(lk.f32, [N_POS, D], device=dev)) for _ in range(N_LAYER)]
        # positions pos .. pos+T-1: wpe rows gathered with ids = pos + arange(T) (computed by a tiny CPU kernel)
        pids = ops.arange_offset_i64_cpu(pos, T)
        h = ops.add_f32_cpu(ops.embedding_f32_cpu(c(P + "wte.weight"), ids), ops.embedding_f32_cpu(c(P + "wpe.weight"), pids))
        if gpu:
            h = ops.to_gpu(h)
            ln = ops.layernorm_f32_gpu(h, g(P + "h.0.ln_1.weight"), g(P + "h.0.ln_1.bias"), eps=LN_EPS)
        for i, (kc, vc) in enumerate(caches):
            L = f"{P}h.{i}."
            if not gpu:
                a = ops.layernorm_f32_cpu(h, c(L + "ln_1.weight"), c(L + "ln_1.bias"), eps=LN_EPS)
                qkv = ops.gemm_bias_f32_cpu_accelerate(a, c(L + "attn.c_attn.weight"), c(L + "attn.c_attn.bias"))
                ops.kv_write_rows_f32_cpu(kc, vc, qkv, pos)
                att = ops.attention_cache_f32_cpu(qkv, kc, vc, pos, heads=HEADS)
                h = ops.add_f32_cpu(h, ops.gemm_bias_f32_cpu_accelerate(att, c(L + "attn.c_proj.weight"),
                                                                        c(L + "attn.c_proj.bias")))
                a = ops.layernorm_f32_cpu(h, c(L + "ln_2.weight"), c(L + "ln_2.bias"), eps=LN_EPS)
                f = ops.gelu_tanh_f32_cpu(ops.gemm_bias_f32_cpu_accelerate(a, c(L + "mlp.c_fc.weight"),
                                                                           c(L + "mlp.c_fc.bias")))
                h = ops.add_f32_cpu(h, ops.gemm_bias_f32_cpu_accelerate(f, c(L + "mlp.c_proj.weight"),
                                                                        c(L + "mlp.c_proj.bias")))
                continue
            nxt = f"{P}h.{i + 1}.ln_1." if i + 1 < N_LAYER else P + "ln_f."
            if kernels == "old":  # scalar cache attention + MPS for every T
                qkv = ops.gemm_f32_gpu_mps(ln, g(L + "attn.c_attn.weight"))
                ops.kv_write_rows_f32_gpu(kc, vc, qkv, pos, g(L + "attn.c_attn.bias"))
                att = ops.attention_cache_f32_gpu(qkv, kc, vc, pos, g(L + "attn.c_attn.bias"), heads=HEADS)
                mm = ops.gemm_f32_gpu_mps
                mm_gelu = lambda x, w, b: ops.bias_gelu_f32_gpu(ops.gemm_f32_gpu_mps(x, w), b)  # noqa: E731
            else:  # per call: T = 1 -> decode kernels (gemv4, fused kv-write attention), T > 1 -> MPS + flash
                qkv = gemm_auto_f32_gpu(ln, g(L + "attn.c_attn.weight"))
                gen = attention_gen_f32_gpu if kernels == "auto1" else attention_gen2_f32_gpu
                att = gen(qkv, kc, vc, pos, g(L + "attn.c_attn.bias"), heads=HEADS)
                mm, mm_gelu = gemm_auto_f32_gpu, gemm_auto_bias_gelu_f32_gpu
            use_sk = kernels not in ("old", "auto1")

            def proj_ln(x, w, res, ln_key, splits):  # (res + x @ W + b, LN of it)
                if use_sk and splits:
                    parts = gemm_auto_sk_f32_gpu(x, g(w + "weight"), splits)
                    return add_bias_layernorm2_sk_f32_gpu(parts, g(w + "bias"), res, g(ln_key + "weight"),
                                                          g(ln_key + "bias"), eps=LN_EPS)
                return ops.add_bias_layernorm2_f32_gpu(mm(x, g(w + "weight")), g(w + "bias"), res,
                                                       g(ln_key + "weight"), g(ln_key + "bias"), eps=LN_EPS)

            h, ln = proj_ln(att, L + "attn.c_proj.", h, L + "ln_2.", sk[0])
            f = mm_gelu(ln, g(L + "mlp.c_fc.weight"), g(L + "mlp.c_fc.bias"))
            h, ln = proj_ln(f, L + "mlp.c_proj.", h, nxt, sk[1])
        if gpu:
            return ops.to_cpu(ops.gemm_nt_f32_gpu_mps(_view(ln, [1, D], offset=(T - 1) * D), g(P + "wte.weight")))
        h = ops.layernorm_f32_cpu(_view(h, [1, D], offset=(T - 1) * D), c(P + "ln_f.weight"), c(P + "ln_f.bias"),
                                  eps=LN_EPS)
        return ops.gemm_nt_f32_cpu_accelerate(h, c(P + "wte.weight"))

    return lk.compile(fn, input_types=[lk.TensorType(lk.i64, [T]), lk.TensorType(lk.i64, [1])])


def generate(model, prompt: list[int], n_new: int) -> list[int]:
    """Greedy generation with a build_generate() model: one prefill call, then one call per new token."""
    logits = model(lk.from_numpy(np.array(prompt, np.int64)), lk.from_numpy(np.array([0], np.int64))).numpy()
    out, pos = [], len(prompt)
    for _ in range(n_new):
        t = int(logits[0].argmax())
        out.append(t)
        logits = model(lk.from_numpy(np.array([t], np.int64)), lk.from_numpy(np.array([pos], np.int64))).numpy()
        pos += 1
    return out


def greedy_generate(step, prompt: list[int], n_new: int) -> list[int]:
    """Greedy decoding with a decode-step model: feeds the prompt at positions 0.., then argmax tokens."""
    out, logits = [], None
    for p, t in enumerate(prompt):
        logits = step(lk.from_numpy(np.array([t], np.int64)), lk.from_numpy(np.array([p], np.int64))).numpy()
    pos = len(prompt)
    for _ in range(n_new):
        t = int(logits[0].argmax())
        out.append(t)
        logits = step(lk.from_numpy(np.array([t], np.int64)), lk.from_numpy(np.array([pos], np.int64))).numpy()
        pos += 1
    return out


def _view(x, shape, offset=0):
    from lokinfer.graph import current_graph
    from lokinfer.ops._def import _as_value

    gr = current_graph()
    x = _as_value(gr, x)
    return gr.view(x, lk.TensorType(x.dtype, shape), offset)
