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
    """A raw buffer in the traced graph: only `lk.persistent_buffer` makes one (model-lifetime state written in
    place by `inout` kernels). Kernel type rules accept it wherever a tensor of the same dtype/shape is accepted."""

    def ir(self) -> str:
        return f"buffer<{self.dtype.name}[{', '.join(str(d) for d in self.shape)}]>"
