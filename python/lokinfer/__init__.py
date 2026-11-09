"""lokinfer: ML inference for macOS/iOS (CPU + Metal GPU).

See docs/impl-notes/design-python-api.md.
"""
from pathlib import Path as _Path

from . import _lokinfer as _C

__version__: str = _C.__version__

# In editable (redirect) installs the .py files come from the source tree while
# the extension and metallib live in site-packages: resolve next to the .so.
_metallib = _Path(_C.__file__).with_name("lokinfer.metallib")
if _metallib.exists():
    _C.set_metallib_path(str(_metallib))

from .dtypes import DType, f32, f16, bf16, i8, u8, i16, i32, i64, bool_  # noqa: E402
from .dims import Dim, Sym, ceildiv  # noqa: E402
from .types import TensorType, ScalarType, MutableType  # noqa: E402
from .tensor import Tensor, tensor, from_numpy, from_torch, empty, zeros, randn  # noqa: E402
from .compiler import compile, CompiledModel, Pending, compile_ir, load  # noqa: E402
from . import ops  # noqa: E402
from .ops.view import dim_read  # noqa: E402


def persistent_buffer(dtype, shape, device: str = "cpu"):
    """Inside a traced graph: model-lifetime state f32[...] whose contents persist across calls (e.g. a KV cache).
    Written in place by kernels declared with `inout`; read like any tensor. Contents start uninitialized."""
    from .graph import current_graph

    g = current_graph()
    if g is None:
        raise RuntimeError("persistent_buffer() is only valid while tracing a graph (inside lk.compile)")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    return g.persistent(TensorType(dtype, shape), device)

__all__ = [
    "DType", "f32", "f16", "bf16", "i8", "u8", "i16", "i32", "i64", "bool_",
    "Dim", "Sym", "ceildiv", "TensorType", "ScalarType", "MutableType",
    "Tensor", "tensor", "from_numpy", "from_torch", "empty", "zeros", "randn",
    "compile", "compile_ir", "CompiledModel", "Pending", "load", "ops", "dim_read", "persistent_buffer",
]
