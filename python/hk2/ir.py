"""The tile IR a kernel is traced into: structured blocks of ops on scalars and tiles.

Values carry types only. Layouts are decided after tracing (``hk2.compiler``), which is what lets one
kernel source compile for wave64 MFMA and wave32 WMMA alike.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import types as T

__all__ = ["ScalarType", "TileType", "PtrType", "Value", "Op", "Block", "Builder"]


@dataclass(frozen=True)
class ScalarType:
    dtype: T.DType

    def __repr__(self):
        return self.dtype.name


@dataclass(frozen=True)
class TileType:
    shape: Tuple[int, ...]
    dtype: T.DType

    def __repr__(self):
        return f"tile<{'x'.join(map(str, self.shape))}x{self.dtype.name}>"


@dataclass(frozen=True)
class PtrType:
    dtype: T.DType

    def __repr__(self):
        return f"ptr<{self.dtype.name}>"


_ids = itertools.count()


class Value:
    __slots__ = ("id", "type", "loc")

    def __init__(self, type, loc=None):
        self.id = next(_ids)
        self.type = type
        self.loc = loc

    @property
    def is_tile(self) -> bool:
        return isinstance(self.type, TileType)

    def __repr__(self):
        return f"%{self.id}:{self.type}"


@dataclass
class Block:
    params: List[Value] = field(default_factory=list)
    ops: List["Op"] = field(default_factory=list)


@dataclass
class Op:
    kind: str
    args: List[Value]
    results: List[Value]
    attrs: Dict[str, Any] = field(default_factory=dict)
    blocks: List[Block] = field(default_factory=list)
    loc: Optional[Tuple[str, int]] = None

    @property
    def result(self) -> Value:
        return self.results[0]

    def __repr__(self):
        a = ", ".join(f"%{v.id}" for v in self.args)
        r = ", ".join(repr(v) for v in self.results)
        at = " ".join(f"{k}={v}" for k, v in self.attrs.items() if k != "fn")
        return f"{r + ' = ' if r else ''}{self.kind}({a}) {at}".rstrip()


class Builder:
    def __init__(self):
        self.top = Block()
        self.stack: List[Block] = [self.top]
        self.loc: Optional[Tuple[str, int]] = None

    @property
    def block(self) -> Block:
        return self.stack[-1]

    def op(self, kind: str, args: Sequence[Value], result_types: Sequence[Any] = (), attrs=None,
           blocks: Sequence[Block] = ()) -> Op:
        o = Op(kind, list(args), [Value(t, self.loc) for t in result_types], dict(attrs or {}), list(blocks), self.loc)
        self.block.ops.append(o)
        return o

    def push(self, block: Block):
        self.stack.append(block)

    def pop(self) -> Block:
        return self.stack.pop()


def dump(block: Block, indent: int = 0) -> str:
    out = []
    pad = "  " * indent
    if block.params:
        out.append(pad + "^(" + ", ".join(map(repr, block.params)) + ")")
    for o in block.ops:
        out.append(pad + repr(o))
        for b in o.blocks:
            out.append(dump(b, indent + 1))
    return "\n".join(out)
