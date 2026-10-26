"""Workgroup placement on multi-XCD GPUs: which XCD runs which workgroup.

docs/design/placement-pass.md has the why. In short: a gfx950 deals workgroup
w to the XCD of w % 8, each XCD has its own 4 MB L2, and a grid whose
workgroups share input bytes (attention's K/V of one KV head, a GEMM's A and
B panels) fetches them once per XCD that reads them and, numbered the wrong
way, holds many unrelated streams on each L2 at once. Renumbering changes
nothing a workgroup computes -- only which XCD and when -- so outputs stay
bitwise the same; the time does not (PR #55: +12-17% for attention at 128 KV
heads, -10% for a short causal grid).

Three parts, all on the traced module before the hk passes run:

  analyze(module)          each global read's sharing key over the grid axes
                           (digits: an axis whole, or `axis // r` and
                           `axis % r` for a radix r the kernel divides by),
                           weighted by bytes; the kernel's classification
                           (cluster / raster / none) and what would make
                           reordering unsafe.
  rewrite(module, plan)    a flat grid, and at kernel entry the logical
                           (x, y, z) decoded from the workgroup id through
                           `place` -- a bijection for any total -- replacing
                           every hk.block_idx.
  choose(...)              the policy hk.kernel(placement="auto") launches by.

`place` is written once over an arithmetic backend: the rewrite emits it as
arith ops, and the host evaluates the same formula (hk.check --placement,
the tests).
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .._mlir import ir
from .._mlir.dialects import arith
from .._mlir.dialects import func as func_d
from .._mlir.dialects import hk as hkd
from . import host

AXES = ("x", "y", "z")

#: XCDs (L2 domains) a part of each arch has, when the device does not say
#: (KFD's num_xcc). 1: one L2, and placement is the identity.
DEFAULT_XCDS = {"gfx950": 8, "gfx942": 8}

#: The policy's thresholds, measured on gfx950's attention (PR #55,
#: the 2511.02132 study's report, §5): cluster from five KV streams an
#: XCD; for non-uniform work (a trip count that depends on the grid), not
#: below twelve rounds of workgroups an XCD.
MIN_STREAMS = 5
MIN_ROUNDS_NONUNIFORM = 12
#: A loop whose trip count is not a constant counts this many times when
#: reads are weighted by bytes.
UNKNOWN_TRIPS = 16

#: Ops after which a different workgroup order can change progress or the
#: result: waits, arrivals and signals between workgroups, and atomics.
ORDER_DEPENDENT_OPS = frozenset({
    "hk.wg_arrive", "hk.signal", "hk.arrive", "hk.wait", "hk.post", "hk.take",
    "hk.atomic_add", "hk.atomic_scalar", "hk.scatter_add_lds",
})

#: Global reads, and which operands carry the address.
READ_OPS = frozenset({
    "hk.load", "hk.load_rows", "hk.group_load", "hk.gather_rows", "hk.load_scalar",
    "hk.stage_load", "hk.load_mx_scales", "hk.load_fp4_frags", "hk.load_fp4_frags_lds",
    "hk.load_w4",
})


# -- the XCD count -------------------------------------------------------------

_KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"
_XCD_CACHE: Dict[str, int] = {}


def xcd_count(arch: str) -> int:
    """XCDs of this process's `arch` GPUs: HK_XCDS if set, else the KFD
    topology's num_xcc when every node of that arch agrees (an MI350P or a
    DPX partition says 4, an MI300A 6), else the arch's default."""
    env = os.environ.get("HK_XCDS")
    if env:
        return int(env)
    if arch not in _XCD_CACHE:
        _XCD_CACHE[arch] = _kfd_xcds(arch) or DEFAULT_XCDS.get(arch, 1)
    return _XCD_CACHE[arch]


def _kfd_xcds(arch: str) -> Optional[int]:
    seen = set()
    for p in glob.glob(os.path.join(_KFD_NODES, "*", "properties")):
        try:
            props = dict(line.split(" ", 1) for line in open(p).read().splitlines() if " " in line)
        except OSError:
            continue
        v = int(props.get("gfx_target_version", "0") or 0)
        if not v:
            continue
        name = f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}"
        if name == arch and "num_xcc" in props:
            seen.add(int(props["num_xcc"]))
    return seen.pop() if len(seen) == 1 else None


# -- what a placement is ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Digit:
    """A grid axis whole (`kind` "whole"), or one digit of it: `axis // radix`
    ("div") or `axis % radix` ("mod"). `radix` is an int, a tensor extent
    spelled "tensor.dim" (`Digit.of("x", "mod", "k.depth")`: the KV heads of
    a (B, H_KV, N, D) `k`), or a host expression of extents and constexprs."""

    axis: str
    kind: str = "whole"
    radix: Optional[host.HostExpr] = dataclasses.field(default=None, compare=False)
    radix_text: str = ""

    @staticmethod
    def of(axis: str, kind: str = "whole", radix=None) -> "Digit":
        """A digit of grid axis `axis` ("x", "y" or "z"): whole, or `axis // radix`
        ("div") / `axis % radix` ("mod"); `radix` an int, "tensor.dim" or a host
        expression."""
        if axis not in AXES or kind not in ("whole", "div", "mod"):
            raise ValueError(f"Digit({axis!r}, {kind!r})")
        if (kind == "whole") != (radix is None):
            raise ValueError(f"Digit({axis!r}, {kind!r}): a radix for div and mod only")
        if isinstance(radix, str):
            tensor, _, dim = radix.partition(".")
            if not tensor or dim not in host.HostDim.DIMS:
                raise ValueError(f"Digit radix {radix!r}: 'tensor.dim', dim one of "
                                 f"{host.HostDim.DIMS}")
            radix = host.HostDim(tensor, dim)
        r = host.HostConst(radix) if isinstance(radix, int) else radix
        return Digit(axis, kind, r, r.emit() if r is not None else "")

    def __repr__(self) -> str:
        return self.axis if self.kind == "whole" else \
            f"{self.axis} {'//' if self.kind == 'div' else '%'} {self.radix_text}"


@dataclasses.dataclass(frozen=True)
class Placement:
    """How a kernel's workgroups are numbered.

    Placement.cluster()            one sharing cluster an XCD at a time; the key from
                                   `analyze` (the dominant read's), or `key=` digits
                                   the analysis cannot see (an index read from memory)
    Placement.chunked(chunk, tail) block_idx.x only, within each (y, z) slice and
                                   with the grid as it is: rounds of `xcds * chunk`
                                   consecutive ids an XCD (a GEMM's xcd_chunk, whose
                                   y is the split-K share); chunk="even" is an
                                   eighth of the axis an XCD
    Placement.none()               the identity

    `tail` is what happens past the last whole round: "stay" (in round-robin
    order, what the hand-written GEMM remaps did) or "spread" (dealt as
    contiguous ranges, balanced to one). chunk 0 is contiguous ranges outright.
    """

    kind: str = "none"
    key: Optional[Tuple[Digit, ...]] = None
    chunk: Union[int, str] = 0
    tail: str = "spread"

    @staticmethod
    def none() -> "Placement":
        """The hardware's order (what placement=None means)."""
        return Placement("none")

    @staticmethod
    def cluster(key: Optional[Sequence[Digit]] = None, chunk: int = 0,
                tail: str = "spread") -> "Placement":
        """One sharing cluster an XCD at a time; `key` the cluster's digits, or None
        for the analysis's. The grid is flattened and decoded at entry."""
        return Placement("cluster", tuple(key) if key is not None else None, chunk, tail)

    @staticmethod
    def chunked(chunk: Union[int, str] = 0, tail: str = "stay") -> "Placement":
        """block_idx.x renumbered in place, `chunk` consecutive ids an XCD a round
        ("even": an eighth of the axis; 0: contiguous ranges); `tail` past the
        last whole round: "stay" or "spread"."""
        return Placement("chunked", None, chunk, tail)

    def __post_init__(self):
        if self.kind not in ("none", "cluster", "chunked"):
            raise ValueError(f"Placement kind {self.kind!r}")
        if self.tail not in ("stay", "spread"):
            raise ValueError(f"Placement tail {self.tail!r}: 'stay' or 'spread'")
        if not (self.chunk == "even" or (isinstance(self.chunk, int) and self.chunk >= 0)):
            raise ValueError(f"Placement chunk {self.chunk!r}: an int >= 0 or 'even'")

    def suffix(self, xcds: int) -> str:
        """The kernel-name suffix of this placement."""
        if self.kind == "none":
            return ""
        c = "e" if self.chunk == "even" else str(self.chunk)
        t = "s" if self.tail == "stay" else "p"
        k = "cl" if self.kind == "cluster" else "ch"
        return f"_x{xcds}{k}{c}{t}"


# -- the arithmetic, once --------------------------------------------------------


class _IntOps:
    """Host ints (`place` evaluated in Python)."""

    def c(self, v):
        return v

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):    # C truncation; operands here are never negative where it matters
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b > 0) else -q

    def mod(self, a, b):
        return a - self.div(a, b) * b

    def min(self, a, b):
        return min(a, b)

    def max(self, a, b):
        return max(a, b)


def _loc(name: str):
    """A named location: the C++ translator names a value after its location."""
    return ir.Location.name(name)


class _ArithOps:
    """arith ops at the current insertion point (the rewrite), each value
    named `pl_<op>` so the generated C++ reads as what it is."""

    def __init__(self):
        self.i32 = ir.IntegerType.get_signless(32)

    def c(self, v):
        return arith.ConstantOp(self.i32, int(v), loc=_loc("pl_c")).result

    def add(self, a, b):
        return arith.AddIOp(a, b, loc=_loc("pl_add")).result

    def sub(self, a, b):
        return arith.SubIOp(a, b, loc=_loc("pl_sub")).result

    def mul(self, a, b):
        return arith.MulIOp(a, b, loc=_loc("pl_mul")).result

    def div(self, a, b):
        return arith.DivSIOp(a, b, loc=_loc("pl_div")).result

    def mod(self, a, b):
        return arith.RemSIOp(a, b, loc=_loc("pl_mod")).result

    def min(self, a, b):
        return arith.MinSIOp(a, b, loc=_loc("pl_min")).result

    def max(self, a, b):
        return arith.MaxSIOp(a, b, loc=_loc("pl_max")).result


def _contiguous(o, w, total, xcds):
    """XCD d (w % xcds) takes one contiguous range of [0, total), the first
    total % xcds XCDs one unit more; within it, in the order it is dealt."""
    x = o.c(xcds)
    d = o.mod(w, x)
    return o.add(o.add(o.mul(d, o.div(total, x)), o.min(d, o.mod(total, x))), o.div(w, x))


def place(o, w, total, xcds: int, chunk: Union[int, str, Any] = 0, tail: str = "spread"):
    """The logical unit workgroup `w` of `total` runs, workgroups being dealt
    to `xcds` XCDs round robin. A bijection on [0, total) for every total.
    `o` is the arithmetic (_IntOps for host ints, _ArithOps to emit)."""
    if xcds <= 1:
        return w
    if isinstance(chunk, int) and chunk == 0:
        return _contiguous(o, w, total, xcds)
    x = o.c(xcds)
    c = o.max(o.div(total, x), o.c(1)) if chunk == "even" else \
        (o.c(chunk) if isinstance(chunk, int) else chunk)
    per_round = o.mul(x, c)
    limit = o.mul(o.div(total, per_round), per_round)
    local = o.div(w, x)
    moved = o.add(o.add(o.mul(o.div(local, c), per_round), o.mul(o.mod(w, x), c)),
                  o.mod(local, c))
    past = o.min(o.max(o.sub(o.add(w, o.c(1)), limit), o.c(0)), o.c(1))   # 1 from `limit` on
    if tail == "stay":
        other = w
    else:
        other = o.add(limit, _contiguous(o, o.sub(w, limit), o.sub(total, limit), xcds))
    return o.add(moved, o.mul(past, o.sub(other, moved)))


class _DslOps:
    """hk device ints, inside a kernel body (renumber, xcd_range)."""

    def c(self, v):
        return v

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a // b

    def mod(self, a, b):
        return a % b

    def min(self, a, b):
        from . import min as _min  # noqa: PLC0415

        return _min(a, b)

    def max(self, a, b):
        from . import max as _max  # noqa: PLC0415

        return _max(a, b)


def renumber(w, total, *, chunk: Union[int, str, Any] = 0, tail: str = "spread", live=None):
    """`place` inside a kernel body, on device ints: the logical unit of id
    `w` of `total` as the XCDs of the GPU being traced for deal them. For a
    persistent kernel, whose units are not workgroups (a workgroup's i-th
    virtual id is w + i * grid): what Placement.chunked / cluster install at
    entry for a grid.

    `live`: only units [0, live) do work, the rest exit at once -- a grid sized
    for the most work a launch can have (a MoE's sorted blocks padded per
    expert), with the count known at run time. Ids [0, live) are then placed
    among themselves and the rest stay: the live units are dealt over every
    XCD. Placing all of [0, total) would hand them to the first XCDs' ranges
    and leave the others idle (K1 at M = 64: 18-20% slower)."""
    from . import builder as B  # noqa: PLC0415

    o, x = _DslOps(), xcd_count(B.current().arch)
    if live is None:
        return place(o, w, total, x, chunk, tail)
    return _place_live(o, w, live, x, chunk, tail)


def _place_live(o, w, live, xcds, chunk=0, tail="spread"):
    """`place` over [0, live) for ids below `live`; the identity above."""
    inside = o.min(o.max(o.sub(live, w), o.c(0)), o.c(1))       # 1 below `live`
    return o.add(w, o.mul(inside, o.sub(place(o, w, live, xcds, chunk, tail), w)))


def xcd_range(d, total):
    """(first, count): XCD `d`'s contiguous range of `total` units, the first
    total % xcds XCDs one unit more -- what contiguous `place` gives each XCD.
    For a persistent kernel that deals units to its XCDs itself."""
    from . import builder as B  # noqa: PLC0415

    return _xcd_range(_DslOps(), d, total, xcd_count(B.current().arch))


def _xcd_range(o, d, total, xcds: int):
    base, tall = o.div(total, o.c(xcds)), o.mod(total, o.c(xcds))
    return (o.add(o.mul(d, base), o.min(d, tall)),
            o.add(base, o.min(o.max(o.sub(tall, d), o.c(0)), o.c(1))))


def place_host(w: int, total: int, xcds: int, chunk: Union[int, str] = 0,
               tail: str = "spread") -> int:
    """`place` on host ints: the logical unit workgroup `w` of `total` runs on
    `xcds` XCDs (what a placed kernel computes at entry)."""
    return place(_IntOps(), w, total, xcds, chunk, tail)


# -- analysis ---------------------------------------------------------------------


@dataclasses.dataclass
class Read:
    op: str
    key: frozenset          # of Digit
    bytes: int
    indirect: bool
    line: str = ""


@dataclasses.dataclass
class Analysis:
    """What `analyze` found in one kernel."""

    reads: List[Read]
    #: cluster / raster / none, and the dominant key (cluster).
    kind: str
    key: Tuple[Digit, ...]
    #: Every digit the kernel takes of each axis.
    digits: Dict[str, List[Digit]]
    #: Why reordering is unsafe (ops that wait, arrive or add), empty if it is not.
    order_dependent: List[str]
    #: A loop's trip count depends on the grid (e.g. causal attention).
    nonuniform: bool
    #: Reads every workgroup makes of the same bytes (a broadcast), in bytes.
    broadcast_bytes: int
    note: str = ""

    def summary(self) -> str:
        """One line: the class, the key, and the flags."""
        k = ", ".join(map(repr, self.key)) or "-"
        s = f"{self.kind} key ({k})"
        if self.order_dependent:
            s += f"; order-dependent: {', '.join(sorted(set(self.order_dependent)))}"
        if self.nonuniform:
            s += "; non-uniform work"
        if self.note:
            s += f"; {self.note}"
        return s


_TILE_RE = re.compile(r"!hk\.(?:st|rt|rv|sv)<(\w+), (\d+)x(\d+)")
_GL_RE = re.compile(r"!hk\.gl<(\w+)")
_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i8": 1, "f8E4M3FN": 1, "f8E5M2": 1}


def _int_typed(v) -> bool:
    t = str(v.type)
    return t.startswith("i") and t[1:].isdigit() or t == "index" or t.startswith("!hk.coord")


def _read_bytes(op) -> int:
    gl = next((m.group(1) for o in op.operands for m in [_GL_RE.match(str(o.type))] if m), None)
    tiles = [m for v in list(op.operands) + list(op.results)
             for m in [_TILE_RE.match(str(v.type))] if m]
    if op.name == "hk.load_scalar":
        return 4
    if not tiles:
        return 64
    el, r, c = tiles[-1].group(1), int(tiles[-1].group(2)), int(tiles[-1].group(3))
    return r * c * _BYTES.get(gl or el, 2)


def _host_of(v, args: Dict[Any, str]) -> Optional[host.HostExpr]:
    """A kernel value that is a host expression (constants and tensor
    extents), or None."""
    op = v.owner
    if isinstance(op, ir.Block):
        return None
    name = op.operation.name if hasattr(op, "operation") else op.name
    o = op.operation if hasattr(op, "operation") else op
    if name == "arith.constant":
        return host.HostConst(int(ir.IntegerAttr(o.attributes["value"]).value))
    if name == "hk.extent":
        t = args.get(o.operands[0])
        if t is None:
            return None
        dim = str(o.attributes["dim"]) if "dim" in o.attributes else None
        dim = dim or re.search(r"hk\.extent %\S+ (\w+)", o.get_asm()).group(1)
        dim = dim.strip('"').split()[-1]
        return host.HostDim(t, dim) if dim in host.HostDim.DIMS else None
    ops = {"arith.addi": "+", "arith.subi": "-", "arith.muli": "*", "arith.divsi": "/",
           "arith.remsi": "%"}
    if name in ops:
        a, b = (_host_of(x, args) for x in o.operands)
        return host.HostBin(ops[name], a, b) if a is not None and b is not None else None
    return None


class _Deps:
    """The grid digits an integer value depends on, and how."""

    __slots__ = ("digits", "loop", "warp", "indirect", "other")

    def __init__(self, digits=frozenset(), loop=False, warp=False, indirect=False, other=False):
        self.digits, self.loop, self.warp = digits, loop, warp
        self.indirect, self.other = indirect, other

    def __or__(self, o: "_Deps") -> "_Deps":
        return _Deps(self.digits | o.digits, self.loop or o.loop, self.warp or o.warp,
                     self.indirect or o.indirect, self.other or o.other)


_EMPTY = _Deps()


def analyze(module: ir.Module, kernel_name: Optional[str] = None) -> Analysis:
    """The sharing structure of the kernel func in `module` (see the module
    docstring). Pure: the module is not changed."""
    f = _kernel_func(module, kernel_name)
    entry = f.regions[0].blocks[0]
    args = {}
    for i, a in enumerate(entry.arguments):
        attrs = f.arg_attrs[i] if f.arg_attrs is not None else None
        name = ir.StringAttr(ir.DictAttr(attrs)["hk.name"]).value if attrs is not None else f"arg{i}"
        args[a] = name
    memo: Dict[Any, _Deps] = {}

    def deps(v) -> _Deps:
        if v in memo:
            return memo[v]
        memo[v] = _EMPTY       # cycles through loop-carried values
        owner = v.owner
        if isinstance(owner, ir.Block):
            parent = owner.owner
            pname = parent.name if parent is not None else ""
            if pname == "scf.for":
                idx = ir.BlockArgument(v).arg_number
                if idx == 0:
                    r = _Deps(loop=True)
                else:   # an iter_arg: its init, and it changes in the loop
                    r = deps(parent.operands[2 + idx]) | _Deps(loop=True)
            else:
                r = _Deps(other=True)
            memo[v] = r
            return r
        op = owner.operation if hasattr(owner, "operation") else owner
        name = op.name
        if name == "hk.block_idx":
            r = _Deps(frozenset({Digit.of(_axis_of(op))}))
        elif name in ("hk.warp_id", "hk.lane_id"):
            r = _Deps(warp=True)
        elif name in ("arith.constant", "hk.extent"):
            r = _EMPTY
        elif name in ("arith.remsi", "arith.remui", "arith.divsi", "arith.divui",
                      "arith.floordivsi") and op.operands[0].owner is not None \
                and not isinstance(op.operands[0].owner, ir.Block) \
                and op.operands[0].owner.name == "hk.block_idx" \
                and not deps(op.operands[1]).digits:
            ax = _axis_of(op.operands[0].owner)
            radix = _host_of(op.operands[1], args)
            kind = "mod" if "rem" in name else "div"
            r = _Deps(frozenset({Digit.of(ax, kind, radix) if radix is not None
                                 else Digit.of(ax)}))
        elif name == "hk.load_scalar":
            r = _Deps(indirect=True)
            for x in op.operands:
                if _int_typed(x):
                    r = r | deps(x)
        else:
            r = _EMPTY
            for x in op.operands:
                if _int_typed(x):
                    r = r | deps(x)
        memo[v] = r
        return r

    reads: List[Read] = []
    order_dep: List[str] = []
    nonuniform = [False]

    def trips(op) -> Tuple[int, bool]:
        """Product of the enclosing loops' trip counts; and whether any of
        them depends on the grid."""
        n, grid_dep = 1, False
        p = op.parent
        while p is not None and p.name != "func.func":
            if p.name == "scf.for":
                lb, ub, st = p.operands[0], p.operands[1], p.operands[2]
                if any(deps(x).digits for x in (lb, ub, st)):
                    grid_dep = True
                c = [_const(x) for x in (lb, ub, st)]
                n *= max(1, (c[1] - c[0] + c[2] - 1) // c[2]) if None not in c and c[2] > 0 \
                    else UNKNOWN_TRIPS
            p = p.parent
        return n, grid_dep

    def visit(op):
        name = op.name
        if name in ORDER_DEPENDENT_OPS:
            order_dep.append(name)
        if name == "scf.for" and any(deps(x).digits for x in list(op.operands)[:3]):
            nonuniform[0] = True
        if name in READ_OPS:
            d = _EMPTY
            for x in op.operands:
                if _int_typed(x):
                    d = d | deps(x)
            n, _ = trips(op)
            reads.append(Read(name, d.digits, _read_bytes(op) * n, d.indirect,
                              op.get_asm().splitlines()[0][:160]))
        for reg in op.regions:
            for blk in reg.blocks:
                for o in blk.operations:
                    visit(o.operation if hasattr(o, "operation") else o)

    for o in entry.operations:
        visit(o.operation if hasattr(o, "operation") else o)

    # The axes the kernel reads at all (a workgroup is told apart by them):
    # every hk.block_idx, wherever its uses are.
    used = {_axis_of(o.operation if hasattr(o, "operation") else o) for o in entry.operations
            if (o.operation if hasattr(o, "operation") else o).name == "hk.block_idx"}
    return _classify(reads, sorted(used), order_dep, nonuniform[0])


def _const(v) -> Optional[int]:
    o = v.owner
    if isinstance(o, ir.Block) or o.name != "arith.constant":
        return None
    try:
        return int(ir.IntegerAttr(o.attributes["value"]).value)
    except Exception:  # noqa: BLE001
        return None


def _axis_of(op) -> str:
    asm = op.get_asm()
    m = re.search(r"hk\.block_idx\s+(\w)", asm)
    return m.group(1)


def _kernel_func(module: ir.Module, name: Optional[str]):
    for op in module.body.operations:
        if "hk.kernel" in op.attributes and (name is None or
                                              ir.StringAttr(op.attributes["sym_name"]).value == name):
            return op
    raise ValueError("module has no hk.kernel func")


def _norm(key: frozenset, used_axes: Sequence[str]) -> Tuple[Digit, ...]:
    """A key in a canonical order (z, y, x; div before mod), with an axis
    taken whole where it is taken whole, or by both digits of one radix."""
    out: List[Digit] = []
    for ax in ("z", "y", "x"):
        ds = [d for d in key if d.axis == ax]
        if not ds:
            continue
        kinds = {d.kind for d in ds}
        radii = {d.radix_text for d in ds if d.kind != "whole"}
        if "whole" in kinds or kinds == {"div", "mod"} and len(radii) == 1 or len(radii) > 1:
            out.append(Digit.of(ax))
        else:
            out.append(ds[0])
    return tuple(out)


def _classify(reads: List[Read], used_axes: Sequence[str], order_dep: List[str],
              nonuniform: bool) -> Analysis:
    weights: Dict[Tuple[Digit, ...], int] = {}
    broadcast = 0
    full = None
    for r in reads:
        k = _norm(r.key, used_axes)
        if not k:
            broadcast += r.bytes
            continue
        weights[k] = weights.get(k, 0) + r.bytes
    digits: Dict[str, List[Digit]] = {}
    for k in weights:
        for d in k:
            digits.setdefault(d.axis, [])
            if d not in digits[d.axis]:
                digits[d.axis].append(d)

    def is_unique(k):
        return all(Digit.of(ax) in k for ax in used_axes)

    shared = {k: w for k, w in weights.items() if not is_unique(k)}
    total = sum(weights.values()) or 1
    note = ""
    if not shared:
        return Analysis(reads, "none", (), digits, order_dep, nonuniform, broadcast,
                        "every read is the workgroup's own" if weights else "no grid-dependent reads")
    ranked = sorted(shared.items(), key=lambda kv: -kv[1])
    key, w = ranked[0]
    if w < total // 4:
        return Analysis(reads, "none", (), digits, order_dep, nonuniform, broadcast,
                        f"shared reads are {100 * w // total}% of the bytes")
    if any(r.indirect for r in reads if _norm(r.key, used_axes) == key):
        note = "the key is read through an index in memory: a heuristic"
    for k2, w2 in ranked[1:]:
        if w2 * 4 >= w and not ({d.axis for d in k2} & {d.axis for d in key}):
            return Analysis(reads, "raster", key, digits, order_dep, nonuniform, broadcast,
                            f"two keys over disjoint axes ({key!r}, {k2!r}): a raster")
    return Analysis(reads, "cluster", key, digits, order_dep, nonuniform, broadcast, note)


# -- the plan: logical digit order --------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Plan:
    """What the rewrite installs: every grid axis split into digits, the
    digits in logical order (major first), and `place`'s parameters. A
    "chunked" plan renumbers block_idx.x alone and keeps the grid."""

    kind: str                             # "cluster" or "chunked"
    order: Tuple[Digit, ...]              # major -> minor; the key's digits first
    splits: Tuple[Tuple[str, Optional[str]], ...]   # (axis, radix text or None)
    radii: Tuple[Tuple[str, Any], ...]    # (axis, radix HostExpr)
    xcds: int
    chunk: Union[int, str]
    tail: str
    key: Tuple[Digit, ...]


def make_plan(placement: Placement, analysis: Optional[Analysis], xcds: int) -> Optional[Plan]:
    """The plan for `placement` given the kernel's analysis, or None (identity)."""
    if placement.kind == "none" or xcds <= 1:
        return None
    if placement.kind == "chunked":
        return Plan("chunked", (Digit.of("x"),), (), (), xcds, placement.chunk,
                    placement.tail, ())
    key = placement.key
    if key is None:
        if analysis is None or analysis.kind != "cluster":
            raise ValueError("Placement.cluster() with no key, and the analysis found none "
                             f"({analysis.summary() if analysis else 'not analyzed'}): pass "
                             "key=(hk.placement.Digit.of(...), ...)")
        key = analysis.key
    split: Dict[str, Digit] = {}
    for d in key:
        if d.kind != "whole":
            split[d.axis] = d
    order_all: List[Digit] = []
    for ax in ("z", "y", "x"):          # original significance: z outermost, x fastest
        if ax in split:
            r = split[ax].radix
            order_all += [Digit.of(ax, "div", r), Digit.of(ax, "mod", r)]
        else:
            order_all.append(Digit.of(ax))
    keyset = set(key)
    order = tuple([d for d in order_all if d in keyset] + [d for d in order_all if d not in keyset])
    return Plan("cluster", order, tuple((ax, split[ax].radix_text) for ax in sorted(split)),
                tuple((ax, split[ax].radix) for ax in sorted(split)), xcds,
                placement.chunk, placement.tail, tuple(key))


def _sizes(o, plan: Plan, g: Dict[str, Any], radix: Callable[[Any], Any]):
    """Each digit's extent, in plan order."""
    rad = dict(plan.radii)
    out = []
    for d in plan.order:
        if d.kind == "whole":
            out.append(g[d.axis])
        elif d.kind == "mod":
            out.append(radix(rad[d.axis]))
        else:
            out.append(o.div(g[d.axis], radix(rad[d.axis])))
    return out


def decode(o, plan: Plan, w, g: Dict[str, Any], radix: Callable[[Any], Any]):
    """(x, y, z) of workgroup `w`: `place`, then the logical unit's digits in
    plan order, then each axis put back together."""
    total = o.mul(o.mul(g["x"], g["y"]), g["z"])
    u = place(o, w, total, plan.xcds, plan.chunk, plan.tail)
    sizes = _sizes(o, plan, g, radix)
    vals = []
    stride = o.c(1)
    strides = []
    for s in reversed(sizes):
        strides.append(stride)
        stride = o.mul(stride, s)
    strides.reverse()
    for s, st in zip(sizes, strides):
        vals.append(o.mod(o.div(u, st), s))
    rad = dict(plan.radii)
    axis = {}
    for ax in AXES:
        parts = {d.kind: v for d, v in zip(plan.order, vals) if d.axis == ax}
        if "whole" in parts:
            axis[ax] = parts["whole"]
        else:
            axis[ax] = o.add(o.mul(parts["div"], radix(rad[ax])), parts["mod"])
    return axis["x"], axis["y"], axis["z"]


def decode_host(plan: Plan, w: int, grid: Tuple[int, int, int],
                radii: Dict[str, int]) -> Tuple[int, int, int]:
    g = dict(zip(AXES, grid))
    return decode(_IntOps(), plan, w, g, lambda r: radii[r.emit()])


def check_bijection(plan: Plan, grid: Tuple[int, int, int], radii: Dict[str, int]) -> None:
    """Every (x, y, z) of `grid` exactly once over the flat ids, or raise."""
    gx, gy, gz = grid
    total = gx * gy * gz
    seen = set()
    for w in range(total):
        c = decode_host(plan, w, grid, radii)
        if not (0 <= c[0] < gx and 0 <= c[1] < gy and 0 <= c[2] < gz):
            raise AssertionError(f"workgroup {w} decodes to {c}, outside the grid {grid}")
        seen.add(c)
    if len(seen) != total:
        raise AssertionError(f"{total - len(seen)} of {total} grid points are run twice "
                             f"and as many never (plan {plan.order}, grid {grid})")


def applicable(plan: Plan, grid: Tuple[int, int, int], radii: Dict[str, int]) -> bool:
    """A split axis must be a whole number of its radix for the decode to be
    a bijection."""
    g = dict(zip(AXES, grid))
    for ax, r in plan.radii:
        rv = radii[r.emit()]
        if rv <= 0 or g[ax] % rv:
            return False
    return True


# -- rewrite --------------------------------------------------------------------------


def rewrite(module: ir.Module, kernel_name: str, tensors, dims, plan: Plan) -> Tuple:
    """Install `plan` in the kernel func and its grid func. Returns the new
    grid as host expressions (total, 1, 1)."""
    f = _kernel_func(module, kernel_name)
    entry = f.regions[0].blocks[0]
    names = [n for n, _ in tensors]
    targs = dict(zip(names, entry.arguments))
    old = {}
    for o in entry.operations:
        op = o.operation if hasattr(o, "operation") else o
        if op.name == "hk.block_idx":
            old[_axis_of(op)] = op
    i32 = ir.IntegerType.get_signless(32)

    def hexpr(e):
        if isinstance(e, int):
            e = host.HostConst(e)
        if isinstance(e, host.HostConst):
            return arith.ConstantOp(i32, e.const, loc=_loc("pl_c")).result
        if isinstance(e, host.HostDim):
            return hkd.ExtentOp(i32, targs[e.tensor], getattr(hkd.Extent, e.dim),
                                loc=_loc(f"pl_{e.tensor}_{e.dim}")).result
        if isinstance(e, host.HostBin):
            cls = {"+": arith.AddIOp, "-": arith.SubIOp, "*": arith.MulIOp,
                   "/": arith.DivSIOp, "%": arith.RemSIOp}[e.op]
            return cls(hexpr(e.lhs), hexpr(e.rhs), loc=_loc("pl_g")).result
        raise TypeError(f"grid dimension {e!r}")

    first = next(iter(entry.operations))
    if plan.kind == "chunked":
        with ir.Location.unknown(), ir.InsertionPoint(first):
            w = hkd.BlockIdxOp(i32, hkd.Axis.x, loc=_loc("pl_wg")).result
            x = place(_ArithOps(), w, hexpr(dims[0]), plan.xcds, plan.chunk, plan.tail)
        x.owner.attributes["hk.placement_axis"] = ir.StringAttr.get("x")
        if "x" in old:
            old["x"].results[0].replace_all_uses_with(x)
            old["x"].erase()
        return tuple(dims)
    with ir.Location.unknown(), ir.InsertionPoint(first):
        g = {ax: hexpr(d) for ax, d in zip(AXES, dims)}
        w = hkd.BlockIdxOp(i32, hkd.Axis.x, loc=_loc("pl_wg")).result
        x, y, z = decode(_ArithOps(), plan, w, g, hexpr)
    for ax, v in zip(AXES, (x, y, z)):
        # Tagged, so hk.check --placement can find and evaluate what was emitted.
        v.owner.attributes["hk.placement_axis"] = ir.StringAttr.get(ax)
        if ax in old:
            old[ax].results[0].replace_all_uses_with(v)
            old[ax].erase()
    # The grid: one axis of total workgroups.
    gfn = None
    for op in module.body.operations:
        if "hk.grid_of" in op.attributes and \
                ir.FlatSymbolRefAttr(op.attributes["hk.grid_of"]).value == kernel_name:
            gfn = op
    if gfn is not None:
        blk = gfn.regions[0].blocks[0]
        ret = list(blk.operations)[-1]
        ret = ret.operation if hasattr(ret, "operation") else ret
        gx, gy, gz = ret.operands
        with ir.Location.unknown(), ir.InsertionPoint(ret):
            t = arith.MulIOp(arith.MulIOp(gx, gy).result, gz).result
            one = arith.ConstantOp(i32, 1).result
            func_d.ReturnOp([t, one, one])
        ret.erase()
    total = _hmul(_hmul(dims[0], dims[1]), dims[2])
    return (total, 1, 1)


def _hmul(a, b):
    a = host.HostConst(a) if isinstance(a, int) else a
    b = host.HostConst(b) if isinstance(b, int) else b
    return host.HostBin("*", a, b)


# -- evaluating what was emitted --------------------------------------------------------


_EVAL = {
    "arith.addi": lambda a, b: a + b, "arith.subi": lambda a, b: a - b,
    "arith.muli": lambda a, b: a * b, "arith.divsi": _IntOps().div,
    "arith.remsi": _IntOps().mod, "arith.minsi": min, "arith.maxsi": max,
    "arith.ceildivsi": lambda a, b: -((-a) // b),
}


def emitted_decode(module: ir.Module, kernel_name: str) -> Optional[Callable]:
    """The decode the kernel func actually computes: a function
    (bx, by, bz, extents) -> (x, y, z) of the hardware's block ids, built from
    the ops tagged hk.placement_axis (an untagged axis is the hardware's), or
    None if the kernel has no placement. `extents` maps a tensor parameter to
    its (batch, depth, rows, cols)."""
    f = _kernel_func(module, kernel_name)
    entry = f.regions[0].blocks[0]
    names = {}
    for i, a in enumerate(entry.arguments):
        names[a] = ir.StringAttr(ir.DictAttr(f.arg_attrs[i])["hk.name"]).value
    tagged = {}
    for o in entry.operations:
        op = o.operation if hasattr(o, "operation") else o
        if "hk.placement_axis" in op.attributes:
            tagged[ir.StringAttr(op.attributes["hk.placement_axis"]).value] = op.results[0]
    if not tagged:
        return None

    def build(v):
        op = v.owner
        name = op.name
        if name == "arith.constant":
            c = int(ir.IntegerAttr(op.attributes["value"]).value)
            return lambda b, e: c
        if name == "hk.block_idx":
            i = AXES.index(_axis_of(op))
            return lambda b, e: b[i]
        if name == "hk.extent":
            t = names[op.operands[0]]
            dim = re.search(r"hk\.extent %\S+ (\w+)", op.get_asm()).group(1)
            i = host.HostDim.DIMS.index(dim)
            return lambda b, e: e[t][i]
        if name in _EVAL:
            fn = _EVAL[name]
            a, c = build(op.operands[0]), build(op.operands[1])
            return lambda b, e: fn(a(b, e), c(b, e))
        raise ValueError(f"the placement decode uses {name}, which hk.check cannot evaluate")

    fs = [build(tagged[a]) if a in tagged else (lambda b, e, i=i: b[i])
          for i, a in enumerate(AXES)]
    return lambda bx, by, bz, e: tuple(f((bx, by, bz), e) for f in fs)


def check_emitted(traced, extents) -> int:
    """hk.check --placement: over every workgroup id of the launch, the
    decode the kernel computes covers the grid its body numbers -- each
    (x, y, z) exactly once. Returns the number of workgroups; raises if not."""
    dec = emitted_decode(traced.module, traced.name)
    grid = tuple(x if isinstance(x, int) else x.value(extents) for x in traced.logical_grid)
    hw = tuple(x if isinstance(x, int) else x.value(extents) for x in traced.grid)
    if dec is None:
        return grid[0] * grid[1] * grid[2]
    total = grid[0] * grid[1] * grid[2]
    seen = set()
    for w in range(total):
        c = dec(w % hw[0], (w // hw[0]) % hw[1], w // (hw[0] * hw[1]), extents)
        if not all(0 <= c[i] < grid[i] for i in range(3)):
            raise AssertionError(f"{traced.name}: workgroup {w} decodes to {c}, outside {grid}")
        seen.add(c)
    if len(seen) != total:
        raise AssertionError(f"{traced.name}: {total - len(seen)} of {total} grid points run "
                             f"twice and as many never (grid {grid})")
    return total


class recording:
    """with recording() as launches: the (kernel, shapes, constexprs) of every
    hk kernel launched inside, in order."""

    def __enter__(self):
        from . import kernel as K  # noqa: PLC0415

        self._prev = K._RECORDING
        K._RECORDING = self.launches = []
        return self.launches

    def __exit__(self, *exc):
        from . import kernel as K  # noqa: PLC0415

        K._RECORDING = self._prev
        return False


def check_kernel(kernel, shapes: Sequence[Tuple[int, ...]], consts=None) -> List[str]:
    """hk.check --placement for one launch geometry: the placement `kernel`
    runs with (its own, or for placement="auto" the cluster placement it may
    pick) decodes to every grid point exactly once. Lines for a report; an
    AssertionError if a decode is not a bijection."""
    consts = dict(consts or {})
    names = [n for n, _ in kernel._tensors]
    ext = {n: (1,) * (4 - len(sh)) + tuple(sh) for n, sh in zip(names, shapes)}
    kernels = [kernel]
    if kernel.placement == "auto":
        tr = kernel.trace(**consts)
        if tr.analysis is not None and tr.analysis.kind == "cluster" \
                and not tr.analysis.order_dependent and xcd_count(tr.arch) > 1:
            kernels.append(kernel.placed())
    out = []
    for k in kernels:
        tr = k.trace(**consts)
        if tr.plan is None:
            out.append(f"{tr.name}: the hardware's numbering"
                       + (f" ({tr.analysis.summary()})" if tr.analysis else ""))
            continue
        grid = tuple(x if isinstance(x, int) else x.value(ext) for x in tr.logical_grid)
        radii = {r.emit(): r.value(ext) for _, r in tr.plan.radii}
        if not applicable(tr.plan, grid, radii):
            out.append(f"{tr.name}: not applicable at grid {grid} (an axis is not a whole "
                       f"number of its radix); placement='auto' does not pick it")
            continue
        n = check_emitted(tr, ext)
        out.append(f"{tr.name}: a bijection over {n} workgroups, grid {grid}, "
                   f"order ({', '.join(map(repr, tr.plan.order))}), {tr.plan.xcds} XCDs")
    return out


# -- policy ---------------------------------------------------------------------------


@dataclasses.dataclass
class Choice:
    """What `choose` decided for one launch geometry: `placed` (launch the
    cluster placement), `reason` (the rule that decided, in words), and the
    numbers it looked at -- `streams`, the most distinct cluster keys among
    an XCD's first resident workgroups under the default order, and `rounds`,
    workgroups an XCD over those it holds at once."""

    placed: bool
    reason: str
    streams: int = 0
    rounds: float = 0.0


def choose(analysis: Analysis, plan: Optional[Plan], grid: Tuple[int, int, int],
           radii: Dict[str, int], xcds: int, resident_per_xcd: int) -> Choice:
    """Whether hk.kernel(placement="auto") launches the clustered kernel for
    this grid (docs/design/placement-pass.md §3)."""
    if xcds <= 1:
        return Choice(False, "one L2")
    if analysis.order_dependent:
        return Choice(False, "order-dependent: " + ", ".join(sorted(set(analysis.order_dependent))))
    if analysis.kind != "cluster" or plan is None:
        return Choice(False, analysis.summary())
    if not applicable(plan, grid, radii):
        return Choice(False, "an axis is not a whole number of its radix")
    gx, gy, gz = grid
    total = gx * gy * gz
    r = max(1, resident_per_xcd)
    rad = {k: radii[v.emit()] for k, v in plan.radii}
    streams = 0
    for d in range(min(xcds, total)):
        keys = set()
        for w in range(d, min(total, d + xcds * r), xcds):
            c = {"x": w % gx, "y": (w // gx) % gy, "z": w // (gx * gy)}
            keys.add(tuple(_digit_value(dg, c, rad) for dg in plan.key))
        streams = max(streams, len(keys))
    rounds = total / (xcds * r)
    if streams < MIN_STREAMS:
        return Choice(False, f"{streams} streams an XCD (< {MIN_STREAMS})", streams, rounds)
    if analysis.nonuniform and rounds < MIN_ROUNDS_NONUNIFORM:
        return Choice(False, f"non-uniform work over {rounds:.1f} rounds an XCD "
                             f"(< {MIN_ROUNDS_NONUNIFORM})", streams, rounds)
    return Choice(True, f"{streams} streams an XCD", streams, rounds)


def _digit_value(d: Digit, c: Dict[str, int], rad: Dict[str, int]) -> int:
    v = c[d.axis]
    if d.kind == "whole":
        return v
    r = rad[d.axis]
    return v // r if d.kind == "div" else v % r


__all__ = ["Digit", "Placement", "Plan", "Analysis", "Choice", "analyze", "make_plan",
           "rewrite", "place", "place_host", "decode_host", "check_bijection", "applicable",
           "emitted_decode", "check_emitted", "check_kernel", "recording", "renumber",
           "xcd_range",
           "choose", "xcd_count", "MIN_STREAMS", "MIN_ROUNDS_NONUNIFORM"]
