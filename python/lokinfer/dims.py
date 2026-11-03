"""Symbolic dims. `Dim` is the native hash-consed expression handle: equal
expressions are the same handle, so `shape_a == shape_b` is exact."""
from __future__ import annotations

from ._lokinfer import Dim


def Sym(name: str) -> Dim:
    """A named symbolic dimension, e.g. `N = Sym("N")`."""
    return Dim.sym(name)


def as_dim(d) -> Dim:
    if isinstance(d, Dim):
        return d
    if isinstance(d, int):
        return Dim.const(d)
    raise TypeError(f"expected Dim or int, got {type(d).__name__}")


def ceildiv(a, b) -> Dim:
    return as_dim(a).ceildiv(b)


def dmin(a, b) -> Dim:
    return as_dim(a).min(b)


def dmax(a, b) -> Dim:
    return as_dim(a).max(b)
