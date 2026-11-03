"""Kernel signatures live in Python only; the C++ registry maps names to code.
The two must agree: every Python op has a registered kernel on the device its
name says, and every registered kernel has a Python definition."""
import lokinfer as lk
from lokinfer import _lokinfer as _C
from lokinfer import ops


def python_ops():
    from lokinfer.ops._def import KERNEL_OPS

    return dict(KERNEL_OPS)


def name_device(kernel: str) -> str:
    return "gpu" if "gpu" in kernel.split("_") else "cpu"


def test_every_python_op_is_registered_on_the_right_device():
    registered = set(_C.registered_kernels())
    for k in python_ops():
        assert k in registered, f"Python op {k} has no C++ kernel"
        assert _C.kernel_device(k) == name_device(k), k


def test_every_registered_kernel_has_a_python_definition():
    defined = set(python_ops())
    missing = [k for k in _C.registered_kernels() if k not in defined]
    assert not missing, f"registered kernels without a Python op: {missing}"
