"""`kernel_op`: Python-side kernel definitions (the only place op signatures
and output-type rules live; the C++ registry only maps names to code)."""
from __future__ import annotations

import atexit
import functools

from ..graph import Value, current_graph
from ..types import ScalarType, TensorType


class OpError(ValueError):
    pass


def check(cond: bool, msg: str = "invalid operands") -> None:
    if not cond:
        raise OpError(msg)


def same_shape(*vals) -> bool:
    return all(v.shape == vals[0].shape for v in vals)


def _as_value(graph, x) -> Value:
    from ..tensor import Tensor

    if isinstance(x, Value):
        if x.graph is not graph:
            raise OpError("value belongs to a different graph")
        return x
    if isinstance(x, Tensor):
        return graph.external(x)
    raise TypeError(f"expected a traced value or Tensor, got {type(x).__name__}")


_eager_cache: dict = {}
# Release cached executables before interpreter teardown (nanobind reports
# native objects still alive at exit as leaks).
atexit.register(_eager_cache.clear)


def _eager(op, args, attrs):
    from ..compiler import compile
    from ..tensor import Tensor

    for a in args:
        if not isinstance(a, Tensor):
            raise TypeError("eager kernel calls take Tensors")
    key = (op.kernel_name, tuple((a.dtype.name, a.shape, a.device) for a in args), tuple(sorted(attrs.items())))
    model = _eager_cache.get(key)
    if model is None:
        model = compile(lambda *xs: op(*xs, **attrs), inputs=list(args))
        _eager_cache[key] = model
    return model(*args)


# Every kernel defined in Python, by kernel name (checked against the C++
# registry by tests/python/test_registry.py).
KERNEL_OPS: dict = {}


def kernel_op(name: str, *, nohoist: bool = False, inplace: list[int] | None = None, inout: list[int] | None = None):
    """Declares a kernel. `rule(*operands, **attrs)` checks operands and
    returns the result type(s): TensorType, ScalarType or a tuple of them (or () for none).
    `inplace=[out_k, in_j, ...]` lets bufferization reuse operand buffers.
    `inout=[j, ...]`: the kernel writes operand j in place (a persistent buffer, e.g. a KV cache)."""

    def deco(rule):
        @functools.wraps(rule)
        def op(*args, **attrs):
            g = current_graph()
            if g is None:
                return _eager(op, args, attrs)
            vals = [_as_value(g, a) for a in args]
            out = rule(*vals, **attrs)
            single = isinstance(out, (TensorType, ScalarType))
            types = [out] if single else list(out)
            kattrs = dict(attrs)
            if nohoist:
                kattrs["nohoist"] = True
            if inplace:
                kattrs["inplace"] = list(inplace)
            if inout:
                kattrs["inout"] = list(inout)
            results = g.kernel(name, vals, kattrs, types)
            if not types:  # side-effect-only kernel (writes `inout` operands)
                return None
            return results[0] if single else tuple(results)

        op.kernel_name = name
        if name in KERNEL_OPS:
            raise OpError(f"kernel {name} defined twice")
        KERNEL_OPS[name] = op
        return op

    return deco
