"""Element types, mirroring the IR dtypes."""
from __future__ import annotations

import numpy as np


class DType:
    __slots__ = ("name", "itemsize", "np", "is_float")

    def __init__(self, name: str, itemsize: int, np_dtype, is_float: bool):
        self.name = name
        self.itemsize = itemsize
        self.np = np.dtype(np_dtype) if np_dtype is not None else None
        self.is_float = is_float

    def __repr__(self) -> str:
        return f"lokinfer.{self.name}"

    def __str__(self) -> str:
        return self.name


f32 = DType("f32", 4, np.float32, True)
f16 = DType("f16", 2, np.float16, True)
bf16 = DType("bf16", 2, None, True)  # numpy has no bfloat16
i8 = DType("i8", 1, np.int8, False)
u8 = DType("u8", 1, np.uint8, False)
i16 = DType("i16", 2, np.int16, False)
i32 = DType("i32", 4, np.int32, False)
i64 = DType("i64", 8, np.int64, False)
bool_ = DType("bool", 1, np.bool_, False)

ALL = [f32, f16, bf16, i8, u8, i16, i32, i64, bool_]
_BY_NAME = {d.name: d for d in ALL}


def by_name(name: str) -> DType:
    return _BY_NAME[name]


def from_numpy_dtype(dt) -> DType:
    dt = np.dtype(dt)
    for d in ALL:
        if d.np is not None and d.np == dt:
            return d
    raise TypeError(f"unsupported numpy dtype {dt}")
