"""Distributed layouts: where each element of a tile lives (register, lane, warp).

A layout is a linear map over GF(2) from the bits of (register index, lane, warp) to the bits of a tile
coordinate. Each input bit has a *basis*: the coordinate it contributes, combined by XOR. A zero basis
means the bit does not move the coordinate: those registers, lanes or warps hold copies of the same
element (a broadcast). This is the same representation Triton's linear layouts use, restricted here to
power-of-two tiles.

Register layouts in hk2 are *disjoint*: every non-zero basis is a single bit of one dimension, and no two
non-zero bases are the same, so XOR is plain addition and an element's coordinate is
``position(lane, warp) + offset(register)``. Swizzled (XOR) maps appear only in LDS addressing.

CDNA and RDNA differ here and nowhere else in the language: wave64 vs wave32, MFMA vs WMMA operand
placement, are different basis tables (see ``hk2.arch``). A kernel names shapes, never layouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Basis = Tuple[int, ...]

__all__ = ["Layout", "blocked", "log2"]


def log2(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"hk2: {n} is not a power of two")
    return n.bit_length() - 1


def _xor(a: Basis, b: Basis) -> Basis:
    return tuple(x ^ y for x, y in zip(a, b))


@dataclass(frozen=True)
class Layout:
    shape: Tuple[int, ...]
    reg: Tuple[Basis, ...]
    lane: Tuple[Basis, ...]
    warp: Tuple[Basis, ...]
    name: str = ""

    # ------------------------------------------------------------------ sizes
    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def nregs(self) -> int:
        return 1 << len(self.reg)

    @property
    def wave(self) -> int:
        return 1 << len(self.lane)

    @property
    def nwarps(self) -> int:
        return 1 << len(self.warp)

    def __eq__(self, other) -> bool:  # the name is documentation only
        return isinstance(other, Layout) and (self.shape, self.reg, self.lane, self.warp) == (
            other.shape, other.reg, other.lane, other.warp)

    def __hash__(self) -> int:
        return hash((self.shape, self.reg, self.lane, self.warp))

    # ------------------------------------------------------------------ positions
    def _combine(self, bases: Sequence[Basis], idx: int) -> Basis:
        out = (0,) * self.rank
        for b, basis in enumerate(bases):
            if idx >> b & 1:
                out = _xor(out, basis)
        return out

    def reg_offset(self, r: int) -> Basis:
        return self._combine(self.reg, r)

    def position(self, r: int, lane: int, warp: int) -> Basis:
        return _xor(_xor(self._combine(self.reg, r), self._combine(self.lane, lane)), self._combine(self.warp, warp))

    def is_disjoint(self) -> bool:
        seen = set()
        for b in self.reg + self.lane + self.warp:
            if not any(b):
                continue
            nz = [d for d, x in enumerate(b) if x]
            if len(nz) != 1 or b[nz[0]] & (b[nz[0]] - 1) or b in seen:
                return False
            seen.add(b)
        return True

    def covers(self) -> bool:
        """Every element of the tile is held by some (register, lane, warp)."""
        need = 1
        for s in self.shape:
            need *= s
        span = {(0,) * self.rank}
        for b in self.reg + self.lane + self.warp:
            if any(b):
                span |= {_xor(p, b) for p in span}
        ok = all(all(0 <= c < s for c, s in zip(p, self.shape)) for p in span)
        return ok and len(span) == need

    # ------------------------------------------------------------------ structure
    def contig(self, dim: int) -> int:
        """How many leading registers are consecutive elements along ``dim`` (a vector's length)."""
        n = 0
        for b, basis in enumerate(self.reg):
            want = tuple((1 << b) if d == dim else 0 for d in range(self.rank))
            if basis != want:
                break
            n += 1
        return 1 << n

    def dup_lane_bits(self) -> List[int]:
        return [b for b, x in enumerate(self.lane) if not any(x)]

    def dup_warp_bits(self) -> List[int]:
        return [b for b, x in enumerate(self.warp) if not any(x)]

    def dup_reg_bits(self) -> List[int]:
        return [b for b, x in enumerate(self.reg) if not any(x)]

    def canonical_reg(self, r: int) -> int:
        """The lowest register holding the same element as register ``r`` (duplicates share a value)."""
        for b in self.dup_reg_bits():
            r &= ~(1 << b)
        return r

    # ------------------------------------------------------------------ derived layouts
    def slice(self, axis: int) -> "Layout":
        """The layout of a reduction over ``axis``: that dimension is dropped from every basis."""
        drop = lambda bs: tuple(tuple(x for d, x in enumerate(b) if d != axis) for b in bs)
        shape = tuple(s for d, s in enumerate(self.shape) if d != axis)
        return Layout(shape, drop(self.reg), drop(self.lane), drop(self.warp), f"slice({self.name},{axis})")

    def expand(self, axis: int) -> "Layout":
        ins = lambda bs: tuple(b[:axis] + (0,) + b[axis:] for b in bs)
        shape = self.shape[:axis] + (1,) + self.shape[axis:]
        return Layout(shape, ins(self.reg), ins(self.lane), ins(self.warp), f"expand({self.name},{axis})")

    def project(self, shape: Sequence[int]) -> "Layout":
        """This layout seen by an operand broadcast from ``shape`` (dimensions of size 1 are zeroed)."""
        keep = [s != 1 for s in shape]
        z = lambda bs: tuple(tuple(x if k else 0 for x, k in zip(b, keep)) for b in bs)
        return Layout(tuple(shape), z(self.reg), z(self.lane), z(self.warp), f"proj({self.name})")

    def transpose(self) -> "Layout":
        assert self.rank == 2
        t = lambda bs: tuple((b[1], b[0]) for b in bs)
        return Layout((self.shape[1], self.shape[0]), t(self.reg), t(self.lane), t(self.warp), f"T({self.name})")

    def describe(self) -> str:
        f = lambda bs: "[" + " ".join("x".join(map(str, b)) for b in bs) + "]"
        return (f"{self.name or 'layout'} {'x'.join(map(str, self.shape))}: regs {self.nregs} {f(self.reg)} "
                f"lanes {f(self.lane)} warps {f(self.warp)}")

    __repr__ = describe


def blocked(shape: Sequence[int], wave: int, nwarps: int, vec: int = 1, name: str = "blocked") -> Layout:
    """Row-major and coalesced: ``vec`` consecutive elements per lane along the last dimension, lanes
    across the last dimension then the rows, warps over the rows, and the rest repeats in registers."""
    shape = tuple(shape)
    rank = len(shape)
    queue: List[Basis] = []
    for d in reversed(range(rank)):
        for i in range(log2(shape[d])):
            queue.append(tuple((1 << i) if dd == d else 0 for dd in range(rank)))
    zero = (0,) * rank
    nv = min(log2(vec), log2(shape[-1]))
    reg = queue[:nv]
    rest = queue[nv:]
    lane = [rest.pop(0) if rest else zero for _ in range(log2(wave))]
    warp = [rest.pop(0) if rest else zero for _ in range(log2(nwarps))]
    reg += rest
    return Layout(shape, tuple(reg), tuple(lane), tuple(warp), name)
