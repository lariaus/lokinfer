"""PyTorch-like Tensor backed by native (page-aligned, unified-memory) storage."""
from __future__ import annotations

import numpy as np

from . import _lokinfer as _C
from . import dtypes as _dt
from .dtypes import DType


class Tensor:
    """A concrete tensor on "cpu" or "gpu" (Apple unified memory: a gpu tensor
    is a shared MTLBuffer, readable from the CPU after GPU work completes)."""

    __slots__ = ("_t",)

    def __init__(self, native):
        self._t = native

    # -- properties
    @property
    def shape(self) -> tuple:
        return tuple(self._t.shape)

    @property
    def dtype(self) -> DType:
        return _dt.by_name(self._t.dtype)

    @property
    def device(self) -> str:
        return self._t.device

    @property
    def ndim(self) -> int:
        return len(self._t.shape)

    @property
    def nbytes(self) -> int:
        return self._t.nbytes

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def data_ptr(self) -> int:
        return self._t.data_ptr

    # -- interop
    def numpy(self) -> np.ndarray:
        """Zero-copy numpy view (waits for pending GPU work)."""
        if self.dtype is _dt.bf16:
            raise TypeError("numpy has no bfloat16; use .torch()")
        return np.from_dlpack(self._t.array())

    def torch(self, device: str = "cpu"):
        """Zero-copy torch tensor. device="cpu" (any tensor: a CPU view of unified memory) or "mps" (GPU tensors: an
        MPS tensor sharing the MTLBuffer). Pending lokinfer GPU work is finished first."""
        import torch

        if device == "mps":
            return torch.from_dlpack(self._t.metal_array())
        if device != "cpu":
            raise ValueError("device must be 'cpu' or 'mps'")
        return torch.from_dlpack(self._t.array())

    def __dlpack__(self, **kwargs):
        return self._t.array().__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._t.array().__dlpack_device__()

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a if dtype is None else a.astype(dtype)

    def to(self, device: str) -> "Tensor":
        if device == self.device:
            return self
        return Tensor(_C.Tensor.from_array(self._t.array(), device=device, copy=True))

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def gpu(self) -> "Tensor":
        return self.to("gpu")

    def tolist(self):
        return self.numpy().tolist()

    def __getitem__(self, idx):
        return self.numpy()[idx]

    def __repr__(self) -> str:
        body = np.array2string(self.numpy(), precision=4, threshold=64) if self.dtype is not _dt.bf16 else "..."
        return f"Tensor({body}, dtype={self.dtype.name}, device={self.device})"

    def __len__(self) -> int:
        return self.shape[0]


def _wrap_array(a, device: str, copy: bool) -> Tensor:
    return Tensor(_C.Tensor.from_array(a, device=device, copy=copy))


def from_numpy(a: np.ndarray, device: str = "cpu", copy: bool = False) -> Tensor:
    """Zero-copy for contiguous CPU arrays unless copy=True or device="gpu"."""
    return _wrap_array(a, device, copy)


def from_torch(t, device: str | None = None, copy: bool = False) -> Tensor:
    """Zero-copy for contiguous CPU and MPS tensors (call torch.mps.synchronize()
    before handing over MPS data written by torch)."""
    if device is None:
        device = "gpu" if t.device.type == "mps" else "cpu"
    return _wrap_array(t.detach(), device, copy)


def tensor(data, dtype: DType | None = None, device: str = "cpu") -> Tensor:
    a = np.asarray(data, dtype=dtype.np if dtype is not None else None)
    if dtype is None and a.dtype == np.float64:
        a = a.astype(np.float32)
    return _wrap_array(np.ascontiguousarray(a), device, copy=True)


def empty(*shape, dtype: DType = _dt.f32, device: str = "cpu") -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(_C.Tensor.empty(list(shape), dtype.name, device))


def zeros(*shape, dtype: DType = _dt.f32, device: str = "cpu") -> Tensor:
    t = empty(*shape, dtype=dtype, device=device)
    t.numpy()[...] = 0
    return t


def randn(*shape, dtype: DType = _dt.f32, device: str = "cpu", seed: int | None = None) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(dtype.np)
    return _wrap_array(a, device, copy=True)
