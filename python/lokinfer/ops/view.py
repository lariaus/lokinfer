"""Views: reshape / flatten as zero-cost IR `view`s (no kernel)."""
from ..dims import as_dim
from ..graph import current_graph
from ..types import TensorType
from ._def import OpError, _as_value, check


def _numel(shape):
    n = as_dim(1)
    for d in shape:
        n = n * d
    return n


def reshape(x, shape):
    """Contiguous reshape; one entry may be -1 (inferred)."""
    g = current_graph()
    if g is None:
        raise OpError("reshape is only available while tracing")
    x = _as_value(g, x)
    shape = list(shape)
    if shape.count(-1) > 1:
        raise OpError("at most one -1 in reshape")
    if -1 in shape:
        known = _numel([d for d in shape if not (isinstance(d, int) and d == -1)])
        total = _numel(x.shape)
        inferred = total // known
        check(inferred * known == total, f"cannot infer -1 reshaping {x.shape} to {shape}")
        shape[shape.index(-1)] = inferred
    check(_numel(shape) == _numel(x.shape), f"reshape changes element count: {x.shape} -> {shape}")
    return g.view(x, TensorType(x.dtype, shape))


def flatten(x, start_dim: int = 1):
    """Flattens dims [start_dim:] into one (PyTorch semantics)."""
    g = current_graph()
    x = _as_value(g, x) if g is not None else x
    rest = _numel(x.shape[start_dim:])
    return reshape(x, list(x.shape[:start_dim]) + [rest])


def dim_read(value, index: int = 0, name: str | None = None):
    """Reads element `index` of an i32/i64 tensor at run time into a new symbolic Dim (IR `dim.read`),
    e.g. the frame count produced by a kernel. Later ops can use the returned Dim in shapes and dim attrs."""
    from ..dims import Sym
    from ..dtypes import i32, i64
    from ..types import TensorType

    g = current_graph()
    if g is None:
        raise OpError("dim_read is only available while tracing")
    x = _as_value(g, value)
    check(isinstance(x.type, TensorType) and x.dtype in (i32, i64), "dim_read needs an i32/i64 tensor")
    check(isinstance(index, int) and index >= 0, "index must be a non-negative int")
    if name is None:
        name = g.fresh("R")
    g.dim_read(x, index, name)
    return Sym(name)
