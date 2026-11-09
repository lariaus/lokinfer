"""Value types used while building graphs."""
from __future__ import annotations

from dataclasses import dataclass

from .dims import Dim, as_dim
from .dtypes import DType


@dataclass(frozen=True)
class TensorType:
    dtype: DType
    shape: tuple

    def __init__(self, dtype: DType, shape):
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", tuple(as_dim(d) for d in shape))

    @property
    def rank(self) -> int:
        return len(self.shape)

    def ir(self) -> str:
        return f"tensor<{self.dtype.name}[{', '.join(str(d) for d in self.shape)}]>"

    def __repr__(self) -> str:
        return self.ir()


@dataclass(frozen=True)
class ScalarType:
    dtype: DType

    def ir(self) -> str:
        return self.dtype.name

    def __repr__(self) -> str:
        return self.ir()


@dataclass(frozen=True, init=False)
class BufferType(TensorType):
    """A raw buffer in the traced graph: memory written in place by `inout` kernels. Made by `lk.persistent_buffer`
    (model-lifetime state) or given as an input type (`lk.MutableType`, the same class): a **mutable input**, the
    caller's buffer (e.g. a KV cache) written in place, never an output, bound without a copy (IR
    `%x: buffer<...> mutable`). Kernel type rules accept it wherever a tensor of the same dtype/shape is accepted."""

    def ir(self) -> str:
        return f"buffer<{self.dtype.name}[{', '.join(str(d) for d in self.shape)}]>"


MutableType = BufferType  # input_types=[lk.MutableType(lk.f16, [L, 2, P, D])]: a mutable input
