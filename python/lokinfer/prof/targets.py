"""Model registry for the profilers: the same models, cases and inputs as the benches (bench/models/*), so a profile
explains a dashboard number. Each target builds the lokinfer graph for a device ("cpu" / "gpu") and, for prof-torch,
the PyTorch reference on "cpu" / "mps".

A saved .lkm can be profiled too (`target_for("path/to/model.lkm")`); its inputs come from an .npz (`--inputs`).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # the repo (python/lokinfer/prof -> repo)


def _repo_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


@dataclass
class Target:
    name: str  # registry key ("kokoro") or the .lkm path
    title: str  # dashboard model name ("Kokoro-82M")
    cases: list[str]
    default_case: str
    inputs: Callable[[str], list]  # case -> lokinfer call arguments (numpy arrays / scalars)
    build: Callable[[str], object]  # device "cpu" | "gpu" -> lokinfer CompiledModel
    torch: Callable[[str], Callable[[str], Callable[[], object]]] | None = None  # device -> case -> zero-arg call
    notes: dict = field(default_factory=dict)

    def remote_inputs(self, case: str) -> list:
        """Inputs for the remote executor: every argument as a numpy array (scalars as 0-d arrays)."""
        out = []
        for a in self.inputs(case):
            if isinstance(a, np.ndarray):
                out.append(a)
            elif isinstance(a, (bool, np.bool_)):
                out.append(np.array(a, np.bool_))
            elif isinstance(a, (int, np.integer)):
                out.append(np.array(a, np.int64))
            elif isinstance(a, (float, np.floating)):
                out.append(np.array(a, np.float32))
            else:
                out.append(np.asarray(a))
        return out


def _rng_img(n: int, c: int = 3, h: int = 224, w: int = 224) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n, c, h, w)).astype(np.float32)


def _cached(fn):
    box = {}

    def get(*a):
        if a not in box:
            box[a] = fn(*a)
        return box[a]

    return get


# ----------------------------------------------------------------------------------------------------------- Kokoro

def _kokoro() -> Target:
    _repo_on_path()

    @_cached
    def ref():
        from models.kokoro import reference as R

        return R

    @_cached
    def weights():
        from models.kokoro import model as K

        return K.fold_weights(ref().load_model())

    def inputs(case):
        c = ref().load_case(case)
        ids = np.concatenate([[0], c["ids"], [0]]).astype(np.int64)
        return [ids, np.asarray(c["ref_s"], np.float32), np.float32(c["speed"]), 0]

    def build(device):
        from models.kokoro import model as K

        return K.build(weights(), gpu=(device == "gpu"))

    def torch(device):
        import importlib.util

        import torch as T
        from kokoro.model import KModel

        spec = importlib.util.spec_from_file_location("_lk_bench_kokoro", ROOT / "bench" / "models" / "bench_kokoro.py")
        sys.path.insert(0, str(ROOT / "bench"))
        bk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bk)  # english-practice's exact infer() math (torch_infer)
        snap = ref().snapshot_dir()
        model = ref().load_model() if device == "cpu" else KModel(
            config=str(snap / "config.json"), model=str(snap / "kokoro-v1_0.pth")).to(device).eval()
        T.manual_seed(0)
        return lambda case: bk.torch_infer(model, ref().load_case(case), device)

    # Default case: app59f, the median sentence of the real workload (models/kokoro/reference.py REAL_TEXTS).
    return Target("kokoro", "Kokoro-82M", ["word", "real28", "app47f", "app47m", "app59f", "app59m", "app71f", "app71m",
                                           "s6f", "s6m", "s9f", "s9m", "real117", "s20f", "s20m", "s28f", "rand510",
                                           "app59_nicole", "app59_emma", "app59_fenrir"],
                  "app59f", inputs, build, torch)


# ----------------------------------------------------------------------------------------------- the other models

def _mnist() -> Target:
    _repo_on_path()

    @_cached
    def tref():
        from models import mnist_mlp

        return mnist_mlp.torch_reference()

    def inputs(case):
        b = int(case.split("=")[1])
        return [np.random.default_rng(b).random((b, 1, 28, 28)).astype(np.float32)]

    def build(device):
        from models import mnist_mlp

        return mnist_mlp.build(mnist_mlp.weights_from_torch(tref()), "gpu" if device == "gpu" else "cpu")

    def torch(device):
        import torch as T

        m = tref().to(device)
        return lambda case: (lambda x=T.from_numpy(inputs(case)[0]).to(device): m(x))

    return Target("mnist", "MNIST MLP", ["B=1", "B=64", "B=1024"], "B=64", inputs, build, torch)


def _vision(name: str) -> Target:
    _repo_on_path()

    @_cached
    def tref():
        from models import mobilenet_v2, resnet18

        return (resnet18 if name == "resnet18" else mobilenet_v2).torch_reference()

    def inputs(case):
        return [_rng_img(int(case.split()[0].split("=")[1]))]

    def build(device):
        from models import mobilenet_v2, resnet18

        if name == "resnet18":
            w = resnet18.fold_weights(tref())
            return resnet18.build_gpu(w) if device == "gpu" else resnet18.build(w)
        return mobilenet_v2.build(mobilenet_v2.fold_weights(tref()), device=device)

    def torch(device):
        import torch as T

        m = tref().to(device)
        return lambda case: (lambda x=T.from_numpy(inputs(case)[0]).to(device): m(x))

    title = "ResNet-18" if name == "resnet18" else "MobileNetV2"
    return Target(name, title, ["N=1 224x224", "N=8 224x224"], "N=1 224x224", inputs, build, torch)


def _dcgan(which: str) -> Target:
    _repo_on_path()

    @_cached
    def tref():
        from models import dcgan

        return dcgan.torch_reference()

    def inputs(case):
        n = int(case.split("=")[1])
        rng = np.random.default_rng(0)
        if which == "g":
            return [rng.standard_normal((n, 100, 1, 1)).astype(np.float32)]
        return [np.tanh(rng.standard_normal((n, 3, 64, 64))).astype(np.float32)]

    def build(device):
        from models import dcgan

        g, d = tref()
        if which == "g":
            return dcgan.build_generator(dcgan.fold_generator(g), device=device)
        return dcgan.build_discriminator(dcgan.fold_discriminator(d), device=device)

    def torch(device):
        import torch as T

        m = tref()[0 if which == "g" else 1].to(device)
        return lambda case: (lambda x=T.from_numpy(inputs(case)[0]).to(device): m(x))

    return Target(f"dcgan_{which}", f"DCGAN {which.upper()}", ["N=1", "N=64"], "N=1", inputs, build, torch)


def _gpt2() -> Target:
    _repo_on_path()

    @_cached
    def tref():
        from models import gpt2

        return gpt2.torch_reference()

    def inputs(case):
        from models import gpt2

        t = int(case.split("=")[1])
        return [np.random.default_rng(0).integers(0, gpt2.VOCAB, t).astype(np.int64)]

    def build(device):
        from models import gpt2

        return gpt2.build(gpt2.weights(tref()), device=device, last_only=True)

    def torch(device):
        import torch as T

        m = tref().to(device)
        return lambda case: (lambda x=T.from_numpy(inputs(case)[0])[None].to(device): m(x, logits_to_keep=1))

    return Target("gpt2", "GPT-2 124M", ["prefill T=128", "prefill T=512"], "prefill T=128", inputs, build, torch)


def _gpt2_decode() -> Target:
    """GPT-2 one decode step (build_decode) at position P with a persistent KV cache, as bench_m6_models.py's
    "decode P=512" case. The cache contents do not change the work (rows 0..P are attended either way), so the
    profile does not prefill it first."""
    _repo_on_path()

    def inputs(case):
        p = int(case.split("=")[1])
        return [np.array([464], np.int64), np.array([p], np.int64)]

    def build(device):
        from models import gpt2

        return gpt2.build_decode(gpt2.weights(_gpt2_ref()), device=device)

    return Target("gpt2_decode", "GPT-2 124M", ["decode P=512"], "decode P=512", inputs, build, None)


@_cached
def _gpt2_ref():
    _repo_on_path()
    from models import gpt2

    return gpt2.torch_reference()


# ------------------------------------------------------------------------------------------------------ Qwen3-ASR

def _qwen3asr(part: str) -> Target:
    """Qwen3-ASR-1.7B optimized ("fused") graphs, bench_qwen3asr.py's cases on the golden inputs (Q1 data).
    enc: the encoder on a clip. prefill: one <= 2048-id chunk at pos 0 (the clip's whole prompt when T <= 2048).
    decode: one step attending over C keys; the cache is not prefilled (the work does not depend on its contents).
    The KV cache is the caller's mutable input, allocated on the build's device (build runs before inputs)."""
    _repo_on_path()
    box = {}

    def build(device):
        from models.qwen3_asr import model as M

        box["device"] = device
        sd = M.load_weights_opt()
        return (M.build_encoder if part == "enc" else M.build_generate)(sd, device, variant="opt")

    def clip(case):
        from models.qwen3_asr import reference as R

        return R.load_ref("s" + case.split()[1].rstrip("s"))

    def inputs(case):
        from models.qwen3_asr import model as M

        if part == "enc":
            r = clip(case)
            return [np.ascontiguousarray(r["inputs/features"]), np.ascontiguousarray(r["inputs/n_audio"])]
        if part == "prefill":
            r = clip(case)
            ids, n_audio = r["inputs/ids"][:2048], int(r["inputs/n_audio"][0])
            k1 = min(max(len(ids) - M.AUDIO_START, 0), n_audio)
            audio = (np.random.default_rng(0).standard_normal((k1, M.LM_D)) * 0.02).astype(np.float16)
            return [np.ascontiguousarray(ids), audio, np.array([0], np.int64), M.kv_cache(len(ids), box["device"])]
        c = int(case.split("=")[1])
        return [np.array([11], np.int64), np.zeros((1, M.LM_D), np.float16), np.array([c - 1], np.int64),
                M.kv_cache(c, box["device"])]

    cases = {"enc": ["enc 10s", "enc 30s", "enc 300s"], "prefill": ["prefill 10s", "prefill 30s", "prefill 300s"],
             "decode": ["decode C=170", "decode C=480", "decode C=4096", "decode C=15600"]}[part]
    return Target(f"qwen3asr_{part}", "Qwen3-ASR-1.7B", cases, cases[1], inputs, build, None)


REGISTRY: dict[str, Callable[[], Target]] = {
    "kokoro": _kokoro,
    "mnist": _mnist,
    "resnet18": lambda: _vision("resnet18"),
    "mobilenet_v2": lambda: _vision("mobilenet_v2"),
    "dcgan_g": lambda: _dcgan("g"),
    "dcgan_d": lambda: _dcgan("d"),
    "gpt2": _gpt2,
    "gpt2_decode": _gpt2_decode,
    "qwen3asr_enc": lambda: _qwen3asr("enc"),
    "qwen3asr_prefill": lambda: _qwen3asr("prefill"),
    "qwen3asr_decode": lambda: _qwen3asr("decode"),
}


def lkm_target(path: str, inputs_npz: str | None) -> Target:
    """A saved .lkm: inputs from an .npz (arrays in argument order: arr_0, arr_1, ... or sorted keys)."""
    def inputs(case):
        if not inputs_npz:
            raise SystemExit("an .lkm target needs --inputs file.npz (its arguments, in order)")
        with np.load(inputs_npz) as z:
            keys = sorted(z.files, key=lambda k: (len(k), k))
            return [z[k][()] if z[k].ndim == 0 else z[k] for k in keys]

    def build(device):
        import lokinfer as lk

        return lk.load(path)

    return Target(path, Path(path).stem, ["inputs"], "inputs", inputs, build, None, {"lkm": True})


def target_for(name: str, inputs_npz: str | None = None) -> Target:
    if name.endswith(".lkm"):
        return lkm_target(name, inputs_npz)
    if name not in REGISTRY:
        raise SystemExit(f"unknown model '{name}' (known: {', '.join(REGISTRY)}, or a .lkm path)")
    return REGISTRY[name]()
