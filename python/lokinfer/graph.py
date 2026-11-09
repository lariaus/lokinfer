"""Graph builder used during tracing: records ops as textual IR lines (the
text form is the canonical, fully tested interchange with the C++ side)."""
from __future__ import annotations

import contextlib
import threading

from .types import ScalarType, TensorType


class Value:
    """A symbolic value during tracing."""

    __slots__ = ("graph", "name", "type")

    def __init__(self, graph: "Graph", name: str, type_):
        self.graph = graph
        self.name = name
        self.type = type_

    @property
    def shape(self) -> tuple:
        return self.type.shape

    @property
    def dtype(self):
        return self.type.dtype

    @property
    def rank(self) -> int:
        return len(self.type.shape)

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type.ir()}"


def _fmt_attr(v) -> str:
    from .dims import Dim

    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        r = repr(v)
        return r if any(c in r for c in ".einf") else r + ".0"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, Dim):
        return f"dim<{v}>"
    if isinstance(v, (list, tuple)):
        if any(isinstance(x, float) for x in v):
            return "f[" + ", ".join(_fmt_attr(float(x)) for x in v) + "]"
        return "[" + ", ".join(str(int(x)) for x in v) + "]"
    raise TypeError(f"unsupported attribute value {v!r}")


class Graph:
    def __init__(self):
        self.args: list[Value] = []
        self.lines: list[str] = []
        self.externals: list[tuple[str, object]] = []  # (name, native tensor)
        self._external_ids: dict[int, tuple] = {}  # id -> (Tensor, Value)
        self._counter = 0

    def fresh(self, hint: str = "v") -> str:
        self._counter += 1
        return f"{hint}{self._counter}"

    def add_arg(self, type_, hint: str = "arg") -> Value:
        v = Value(self, self.fresh(hint), type_)
        self.args.append(v)
        return v

    def external(self, tensor) -> Value:
        """Captures a concrete Tensor as a const.external (zero-copy binding)."""
        # Keyed by id(); the entry holds the Tensor itself so a freed temporary's id can't be reused
        # by a different tensor during the trace (that bound the wrong constant).
        key = id(tensor)
        hit = self._external_ids.get(key)
        if hit is not None and hit[0] is tensor:
            return hit[1]
        name = f"ext{len(self.externals)}"
        t = TensorType(tensor.dtype, tensor.shape)
        v = Value(self, self.fresh("c"), t)
        self.lines.append(f'%{v.name} = const.external @"{name}" {{device = "{tensor.device}"}} : {t.ir()}')
        self.externals.append((name, tensor._t))
        self._external_ids[key] = (tensor, v)
        return v

    def kernel(self, name: str, operands: list[Value], attrs: dict, result_types: list) -> list[Value]:
        results = [Value(self, self.fresh("v"), t) for t in result_types]
        lhs = ", ".join(f"%{r.name}" for r in results)
        ops = ", ".join(f"%{o.name}" for o in operands)
        attr = ""
        if attrs:
            attr = " {" + ", ".join(f"{k} = {_fmt_attr(attrs[k])}" for k in sorted(attrs)) + "}"
        types = ", ".join(t.ir() for t in result_types)
        prefix = f"{lhs} = " if results else ""
        suffix = f" : {types}" if results else ""
        self.lines.append(f'{prefix}kernel "{name}"({ops}){attr}{suffix}')
        return results

    def persistent(self, type_, device: str) -> Value:
        """`buffer.alloc {persistent}`: model-lifetime state (contents persist across calls; hoist-init moves the
        alloc into init). Written in place by kernels declared with `inout`."""
        from .types import BufferType

        bt = BufferType(type_.dtype, type_.shape)
        v = Value(self, self.fresh("p"), bt)
        self.lines.append(f'%{v.name} = buffer.alloc {{device = "{device}", persistent = true}} : {bt.ir()}')
        return v

    def view(self, value: Value, type_, offset=0) -> Value:
        """Zero-cost reinterpretation of contiguous memory (offset in elements)."""
        v = Value(self, self.fresh("v"), type_)
        self.lines.append(f"%{v.name} = view %{value.name}[{offset}] : {type_.ir()}")
        return v

    def sync(self, value: Value, kind: str) -> Value:
        """Explicit device boundary (`gpu_to_cpu` / `cpu_to_gpu`): a zero-cost alias of
        `value` resident on the target device (unified memory, nothing is copied)."""
        v = Value(self, self.fresh("s"), value.type)
        self.lines.append(f"%{v.name} = sync {kind} %{value.name} : {value.type.ir()}")
        return v

    def dim_read(self, value: Value, index: int, symbol: str) -> None:
        self.lines.append(f"dim.read %{value.name}[{index}] -> {symbol}")

    def to_text(self, outputs: list[Value]) -> str:
        from .types import BufferType

        # buffer-typed inputs are mutable inputs (written in place by inout kernels; the verifier checks the rules)
        args = ", ".join(f"%{a.name}: {a.type.ir()}" + (" mutable" if isinstance(a.type, BufferType) else "")
                         for a in self.args)
        body = "\n".join("    " + l for l in self.lines)
        ret = ", ".join(f"%{o.name}" for o in outputs)
        types = [o.type.ir() for o in outputs]  # result types in the header, MLIR style (the parser checks them)
        res = f" -> {types[0]}" if len(types) == 1 else f" -> ({', '.join(types)})"
        return (
            "model {\n  init {\n    return\n  }\n"
            f"  exec({args}){res} {{\n{body}\n    return {ret}\n  }}\n}}\n"
        )


_state = threading.local()


def current_graph() -> Graph | None:
    return getattr(_state, "graph", None)


@contextlib.contextmanager
def tracing(graph: Graph):
    prev = current_graph()
    _state.graph = graph
    try:
        yield graph
    finally:
        _state.graph = prev
