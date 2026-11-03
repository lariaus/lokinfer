"""Elementwise kernels."""
from ..dtypes import f32
from ..types import TensorType
from ._def import check, kernel_op, same_shape


def _binary_f32(x, y):
    check(x.dtype is f32 and y.dtype is f32, "expected f32 operands")
    check(same_shape(x, y), f"shape mismatch: {x.shape} vs {y.shape}")
    return TensorType(f32, x.shape)


@kernel_op("add_f32_cpu")
def add_f32_cpu(x, y):
    return _binary_f32(x, y)


@kernel_op("mul_f32_cpu")
def mul_f32_cpu(x, y):
    return _binary_f32(x, y)


@kernel_op("add_f32_gpu")
def add_f32_gpu(x, y):
    return _binary_f32(x, y)


@kernel_op("mul_f32_gpu")
def mul_f32_gpu(x, y):
    return _binary_f32(x, y)
