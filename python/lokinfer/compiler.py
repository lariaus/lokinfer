"""Tracing compiler front: `compile(fn, inputs=...)` -> CompiledModel."""
from __future__ import annotations

import numpy as np

from . import _lokinfer as _C
from .dims import Sym
from .dtypes import f32, i64
from .graph import Graph, Value, tracing
from .tensor import Tensor, from_numpy
from .types import ScalarType, TensorType


def _input_type(x, idx: int, dynamic: bool):
    if isinstance(x, (TensorType, ScalarType)):
        return x
    if isinstance(x, np.ndarray):
        x = from_numpy(x)
    if isinstance(x, Tensor):
        shape = [Sym(f"D{idx}_{j}") for j in range(x.ndim)] if dynamic else list(x.shape)
        return TensorType(x.dtype, shape)
    if isinstance(x, bool):
        raise TypeError("bool scalar inputs are not supported yet")
    if isinstance(x, float):
        return ScalarType(f32)
    if isinstance(x, int):
        return ScalarType(i64)
    raise TypeError(f"unsupported input {type(x).__name__}")


def _native_input(a):
    if isinstance(a, np.ndarray):
        a = from_numpy(np.ascontiguousarray(a))
    elif hasattr(a, "__dlpack__") and not isinstance(a, Tensor):
        from .tensor import from_torch

        a = from_torch(a)
    return a._t if isinstance(a, Tensor) else a


def _native_output(o):
    """A caller-provided output: lokinfer Tensor, contiguous numpy array or torch tensor (CPU or MPS), never copied."""
    if o is None:
        return None
    if isinstance(o, Tensor):
        return o._t
    if isinstance(o, np.ndarray):
        if not o.flags.c_contiguous or not o.flags.writeable:
            raise ValueError("out= numpy arrays must be C-contiguous and writeable")
        return from_numpy(o)._t
    if hasattr(o, "__dlpack__"):
        from .tensor import from_torch

        if hasattr(o, "is_contiguous") and not o.is_contiguous():
            raise ValueError("out= torch tensors must be contiguous")
        return from_torch(o)._t
    raise TypeError(f"unsupported out= value {type(o).__name__}")


class Pending:
    """Handle of an asynchronous call (`CompiledModel.submit`). `wait()` returns what the call returns (or raises
    its error); `done()` tells whether it finished."""

    __slots__ = ("_future",)

    def __init__(self, future):
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def wait(self, timeout: float | None = None):
        return self._future.result(timeout)


class CompiledModel:
    def __init__(self, exe, model, text: str, input_types: list, single: bool, externals=()):
        self._exe = exe
        self._model = model
        self._externals = list(externals)  # (name, native tensor) for save()
        self.source_ir = text
        self.input_types = input_types
        self._single = single
        self._worker = None  # one thread per model for submit(): calls of a model run one at a time, in order

    @property
    def uses_gpu(self) -> bool:
        return self._exe.uses_gpu

    def ir_text(self) -> str:
        """IR after the compiler pipeline."""
        return self._model.ir_text()

    def save(self, path: str, *, bake_init: bool = False) -> None:
        """Writes a self-contained .lkm (captured weights are internalized). bake_init=True stores init's results
        (e.g. transposed weights) instead of their inputs, so loading does no init work (see bake_init())."""
        self._drain()
        model = _C.bake_init(self._model, self._externals)[0] if bake_init else self._model
        _C.save_model(model, str(path), self._externals)

    def bake_init(self) -> dict:
        """Runs init once and replaces its computed results (weights transformed by hoisted kernels) with constants;
        init work nothing else needs is removed, and the executable is re-created. Safe: arenas, persistent state
        and models whose init defines symbols are left alone. Returns statistics
        (baked, baked_bytes, init_ops_before, init_ops_after)."""
        self._drain()
        model, stats = _C.bake_init(self._model, self._externals)
        self._exe = _C.create_executable(model, self._externals)
        self._model = model
        return dict(stats)

    def internalize(self) -> "CompiledModel":
        """Copies the captured tensors (external constants) into the model: later writes to those tensors no longer
        affect it, and they may be freed. Re-creates the executable (init runs again). Returns self."""
        self._drain()
        if self._externals:
            model = _C.internalize_model(self._model, self._externals)
            self._exe = _C.create_executable(model, [])
            self._model = model
            self._externals = []
        return self

    @property
    def num_external_constants(self) -> int:
        return len(self._externals)

    def _call(self, args, out):
        native = [_native_input(a) for a in args]
        outs = None
        if out is not None:
            given = [out] if self._single and not isinstance(out, (list, tuple)) else list(out)
            outs = [_native_output(o) for o in given]
        results = [Tensor(t) for t in self._exe.run(native, outs)]
        if outs is not None:  # hand back the caller's own objects for the outputs it provided
            results = [g if g is not None else r for g, r in zip(given, results)]
        return results[0] if self._single else tuple(results)

    def __call__(self, *args, out=None):
        """Runs the model synchronously. out=: the output(s) to write into (one per output, None = allocate);
        lokinfer Tensors, contiguous numpy arrays or torch tensors (CPU or MPS), filled without a copy when the output
        is computed by the model (docs/python-api.md)."""
        if self._worker is not None:
            return self.submit(*args, out=out).wait()  # keep the order with pending submits
        return self._call(args, out)

    def submit(self, *args, out=None) -> Pending:
        """Starts the call on the model's worker thread and returns at once; `.wait()` gives the result. The native
        call releases the GIL, so Python keeps running meanwhile. Calls of one model run one at a time, in submit
        order; the arguments must not be written until the call is done."""
        if self._worker is None:
            from concurrent.futures import ThreadPoolExecutor

            self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lokinfer-model")
        return Pending(self._worker.submit(self._call, args, out))

    def _drain(self):
        if self._worker is not None:
            self._worker.submit(lambda: None).result()

    def __del__(self):
        w = getattr(self, "_worker", None)
        if w is not None:
            w.shutdown(wait=True)


def compile(fn, inputs=None, *, dynamic: bool = False, input_types=None, pipeline=None) -> CompiledModel:
    """Traces `fn` with symbolic values and compiles it.

    inputs:      example Tensors/arrays/scalars (static shapes unless dynamic=True)
    dynamic:     every input dim becomes its own symbol
    input_types: explicit TensorType/ScalarType list (parametric shapes, e.g. [N, N])
    pipeline:    override the pass list (default: the full pipeline)
    """
    if input_types is None:
        if inputs is None:
            raise ValueError("give inputs or input_types")
        input_types = [_input_type(x, i, dynamic) for i, x in enumerate(inputs)]
    g = Graph()
    args = [g.add_arg(t, hint=f"in{i}_") for i, t in enumerate(input_types)]
    with tracing(g):
        out = fn(*args)
    single = isinstance(out, Value)
    outs = [out] if single else list(out)
    for o in outs:
        if not isinstance(o, Value):
            raise TypeError("the traced function must return traced values")
    text = g.to_text(outs)
    passes = list(pipeline) if pipeline is not None else _C.default_pipeline()
    model = _C.compile_text(text, passes, [])
    exe = _C.create_executable(model, g.externals)
    return CompiledModel(exe, model, text, input_types, single, g.externals)


def load(path: str) -> CompiledModel:
    """Loads a .lkm written by CompiledModel.save (mmapped, zero-copy weights)."""
    model = _C.load_model(str(path))
    exe = _C.create_executable(model, [])
    return CompiledModel(exe, model, model.ir_text(), [], model.num_outputs == 1)


def compile_ir(text: str, *, externals: dict | None = None, internals: dict | None = None, pipeline=None,
               single: bool = True) -> CompiledModel:
    """Compiles textual IR directly (const.external / const.internal data by name)."""
    ext = [(k, (v if isinstance(v, Tensor) else from_numpy(np.ascontiguousarray(v)))._t) for k, v in (externals or {}).items()]
    inte = [(k, (v if isinstance(v, Tensor) else from_numpy(np.ascontiguousarray(v), copy=True))._t)
            for k, v in (internals or {}).items()]
    passes = list(pipeline) if pipeline is not None else _C.default_pipeline()
    model = _C.compile_text(text, passes, inte)
    exe = _C.create_executable(model, ext)
    return CompiledModel(exe, model, text, [], single, ext)
