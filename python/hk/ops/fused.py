"""Fused pointwise kernels: SwiGLU's activation half, and RoPE.

Neither has a reduction, which makes the out-of-bounds story the easy one: an
elementwise result does not depend on what else is in the tile, so backing the
last block up recomputes a few elements to the same answer and no masking is
needed anywhere. It also means nothing here needs `left_fill`, which is why
`silu_mul` takes any width at all and the norm kernels' `pad` thunk has no
counterpart in this file.

**The knob that mattered was not the one this file was built around.** The
obvious worry for a pointwise kernel is that one 16x64 tile per workgroup is not
enough work per wave -- at 8192x11008 that is 88064 workgroups of 32 threads
each doing two loads and a store, and the loads of one wave cannot overlap the
loads of the next because there is no next. So `silu_mul` took the same two
knobs the norm kernels do: WARPS warps per workgroup, each covering TILES column
blocks. Swept, all fifteen settings landed within 5% and the smallest won; with
tens of thousands of workgroups resident there was never a shortage of waves.
`plan` returns (1, 1) and says so.

What the same numbers did show is that the identical kernel ran at 716 GB/s
over an 8192x4096 tensor and 675 GB/s over an 8192x11008 one, doing the same
number of tiles in the same number of workgroups. The difference is the stride
a tile's 16 rows step by. And an elementwise kernel does not owe the caller its
row structure -- see `canonical`, which is where the speed actually came from.

RoPE is here rather than in a file of its own because the interesting part is
the same trick: the rotation pairs column j with column j + D/2, and there is
no cross-lane shuffle in the warp register API that would let one tile see
both. So the kernel loads *two* tiles from two column offsets -- which costs
nothing, since they are separate global reads either way -- and writes two. The
consequence is a real constraint, stated in the wrapper: D/2 must be a multiple
of the tile width.
"""

from functools import lru_cache as _lru_cache

from .. import dsl as hk
from ..types import bf16, fp16, fp32
from .norm import _overlaps

ROWS, COLS = 16, 64


# ---------------------------------------------------------------- silu_mul


#: Live fp32 tiles a warp may hold. Two per column block -- one from each
#: operand -- so TILES=4 is 8 tiles, 256 VGPRs of data, and spills. The build
#: gate would catch that, but naming the reason here is cheaper than a
#: compile.
MAX_TILES = 3



def _ftile(rows, cols):
    """The fp32 tile a row is worked in. Where tiles take a fragment shape
    (gfx950) it is laid out in 16x32 fragments, so a lane's run of a 16-bit row is 8 elements:
    one 16-byte buffer load or store where the 16x16 fragment's is 8 bytes."""
    if hk.arch.target().tile_shapes:
        return hk.RT(fp32, rows, cols, frag=(16, 32))
    return hk.RT(fp32, rows, cols)

def _start(idx, block: int, extent):
    """Where block `idx` starts, backed up so the whole tile is in bounds.

    norm's version of this returns a mask thunk as well; this one does not need
    one. An elementwise result does not depend on what else is in the tile, so
    the backed-up overlap is recomputed to the same answer and written twice
    with the same bytes.
    """
    return hk.min(idx * block, extent - block)


def _silu_mul_grid(p):
    return (hk.cdiv(p.o.cols, p.COLS * p.WARPS * p.TILES),
            hk.cdiv(p.o.rows, p.ROWS), 1)


def _silu_mul_kernel(name: str, dtype, warps: int):
    def body(a: hk.Tensor[dtype], b: hk.Tensor[dtype], o: hk.Tensor[dtype], *,
             ROWS=ROWS, COLS=COLS, WARPS=warps, TILES=1):
        # fp32 throughout: base_ops::silu is specialised for float only, and
        # the load converts on the way in for free, so there is nothing to pay
        # for the extra precision but register width.
        t = _ftile(ROWS, COLS)
        ncols = o.cols
        row_lo = _start(hk.arch.block_idx.y, ROWS, o.rows)
        base = hk.arch.block_idx.x * (WARPS * TILES)

        def block_of(i):
            """The column block warp `warp_id` covers at step `i`.

            Strided like norm's: at any step the warps of a workgroup are on
            WARPS adjacent blocks, one contiguous run, rather than WARPS places
            a group apart.
            """
            cb = base + i * WARPS if i else base
            return cb + hk.arch.warp_id() if WARPS > 1 else cb

        idx = [hk.elem_coord(0, 0, row_lo, _start(block_of(i), COLS, ncols))
               for i in range(TILES)]
        # Both loads of every tile before any of the arithmetic. Written this
        # way rather than as a load/compute/store per tile so that the loads
        # are adjacent in the generated C++ and the scheduler has TILES*2
        # outstanding, not one.
        xs = [a[i].load(t) for i in idx]
        bs = [b[i].load(t) for i in idx]
        for x, bv, i in zip(xs, bs, idx):
            x.silu(into=x)
            x *= bv
            o[i] = x.to(dtype)

    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=_silu_mul_grid, name=name)


def _strip_silu_mul_kernel(name: str, dtype, warps: int):
    """silu_mul on strips: workgroup i is row i / G, block g = i % G of it (G
    blocks a row), its warp w the columns [(g*WARPS + w)*CW, +CW), CW whole
    wave x RUN chunks. Consecutive workgroups walk along a row, not down a
    column, as the memory lies. Each operand is its own global with its own
    row stride, so a and b can be the two halves of one (rows, 2I) gate/up
    output, read in place. Without EXACT the accesses are bounded: a run at
    or past the row's end is neither read nor written."""
    def body(a: hk.Tensor[dtype], b: hk.Tensor[dtype], o: hk.Tensor[dtype], *,
             CW=512, RUN=8, EXACT=1, RESERVE=0):
        if RESERVE:
            hk.lds.reserve(RESERVE)
        S = hk.RS(fp32, 1, CW, RUN)
        per_row = hk.cdiv(o.cols, warps * CW)
        i = hk.arch.block_idx.x
        row = i // per_row
        g = i - row * per_row
        blk = g * warps + hk.arch.warp_id() if warps > 1 else g
        c0 = blk * CW
        x = a.at[row, c0].load(S, bounded=not EXACT)
        y = b.at[row, c0].load(S, bounded=not EXACT)
        x.silu(into=x)
        x *= y
        o.at[row, c0].store(x, bounded=not EXACT)

    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name, noalias=True,
                     grid=lambda p: (p.o.rows * hk.cdiv(p.o.cols, warps * p.CW), 1, 1))


# ---------------------------------------------------------------- rope


def _rotary_kernel(name: str, dtype, warps: int, positions: bool):
    """vLLM's rotary_embedding, NeoX style, on strips: token t's heads are
    rows t*H .. t*H+H-1 of x viewed (T*H, D); its angles are row pos[t] (or
    t) of cs, (P, D) fp32 holding [cos | sin]. One workgroup a token, warp w
    the heads w, w + WARPS, ...; a head is one strip row of D = 64*RUN
    columns, the halves lanes l and l + 32 apart, so

        out = x * [cos | cos] + swap_halves(x) * [-sin | sin]

    with both factors built once a warp from the one row of cs, by the same
    swap and a fill on each side of D/2."""
    def build(x, cs, pos, o, H, D, RUN):
        S = hk.RS(fp32, 1, D, RUN)
        t = hk.arch.block_idx.x
        prow = pos.at[0, t].load_scalar() if positions else t
        c = cs.at[prow, 0].load(S)                 # [cos | sin]
        sw = c.swap_halves()                       # [sin | cos]
        half = D // 2
        cosv = c.copy()
        cosv.right_fill_(half, 0.0)                # [cos | 0]
        lo = sw.copy()
        lo.left_fill_(half, 0.0)                   # [0 | cos]
        cosv += lo
        sinv = sw.copy()
        sinv.right_fill_(half, 0.0)                # [sin | 0]
        sinv *= -1.0
        hi = c.copy()
        hi.left_fill_(half, 0.0)                   # [0 | sin]
        sinv += hi
        w = hk.arch.warp_id() if warps > 1 else 0
        rows = [t * H + w + h * warps for h in range(H // warps)]
        # Every head's load before any store: x and o may be one tensor (in
        # place), so a load after a store is not one the compiler may hoist,
        # and written head by head each head waited a memory round trip.
        vs = [x.at[row, 0].load(S) for row in rows]
        for row, v in zip(rows, vs):
            p = v.swap_halves()
            v *= cosv
            p *= sinv
            v += p
            o.at[row, 0].store(v)

    if positions:
        def body(x: hk.Tensor[dtype], cs: hk.Tensor[fp32], pos: hk.Tensor[hk.i32],
                 o: hk.Tensor[dtype], *, H=32, D=128, RUN=2):
            build(x, cs, pos, o, H, D, RUN)
    else:
        def body(x: hk.Tensor[dtype], cs: hk.Tensor[fp32], o: hk.Tensor[dtype], *,
                 H=32, D=128, RUN=2):
            build(x, cs, None, o, H, D, RUN)
    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name, noalias=True,
                     grid=lambda p: (hk.cdiv(p.x.rows, p.H), 1, 1))


def _rotary16_kernel(name: str, dtype, warps: int, positions: bool):
    """_rotary_kernel with a lane's run RUN bf16 -- eight, one 16-byte
    access, or four, eight bytes: x viewed (T, H*D), a token one row, and a
    head of D columns D/RUN lanes of a chunk (at D = 128 and RUN 8, 16 lanes,
    its halves eight apart, one DPP row_ror:8 each; at RUN 4, 32 lanes, a
    v_permlane16_swap: rotate_half). A workgroup is (token, group of
    WARPS*CH chunks), warp w of it CH chunks of 64*RUN columns -- four heads
    a chunk at RUN 8, two at 4 -- and every head of a chunk turns by the same
    angles, so the factors are one chunk of `[cos | cos]` and one of
    `[sin | sin]` read with a period of D/2 out of the token's row of cs:

        out = x * cos + rotate_half(x) * sin
    """
    def build(x, cs, pos, o, D, CH, RESERVE, RUN):
        if RESERVE:
            hk.lds.reserve(RESERVE)
        cw = 64 * RUN
        S = hk.RS(fp32, 1, cw, RUN)
        t = hk.arch.block_idx.x
        prow = pos.at[0, t].load_scalar(invariant=True) if positions else t
        w = hk.arch.block_idx.y * warps + hk.arch.warp_id() if warps > 1 else hk.arch.block_idx.y
        c0 = w * CH * cw
        # The token's heads first: they do not wait on the position.
        vs = [x.at[t, c0 + j * cw].load(S) for j in range(CH)]
        cosv = cs.at[prow, 0].load(S, period=D // 2)
        sinv = cs.at[prow, D // 2].load(S, period=D // 2)
        for j, v in enumerate(vs):
            p = v.rotate_half(D)
            v *= cosv
            p *= sinv
            v += p
            o.at[t, c0 + j * cw].store(v)

    if positions:
        def body(x: hk.Tensor[dtype], cs: hk.Tensor[fp32], pos: hk.Tensor[hk.i32],
                 o: hk.Tensor[dtype], *, D=128, CH=1, RESERVE=0, RUN=8):
            build(x, cs, pos, o, D, CH, RESERVE, RUN)
    else:
        def body(x: hk.Tensor[dtype], cs: hk.Tensor[fp32], o: hk.Tensor[dtype], *,
                 D=128, CH=1, RESERVE=0, RUN=8):
            build(x, cs, None, o, D, CH, RESERVE, RUN)
    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name, noalias=True,
                     grid=lambda p: (p.x.rows, p.x.cols // (warps * p.CH * 64 * p.RUN), 1))


def _rotary_halves_kernel(name: str, dtype, warps: int, positions: bool):
    """vLLM's rotary_embedding on strips without a lane exchange, for a
    target whose strips have no swap_halves (gfx1100): a head's halves are
    two strips of D/2 columns read at 0 and D/2, and its angles the two
    halves of the token's row of cs, [cos | sin]:

        out1 = x1 * cos - x2 * sin,   out2 = x2 * cos + x1 * sin

    One workgroup a token, warp w the heads w, w + WARPS, ...; every head's
    loads before any store, so it runs in place (a lane writes only what it
    read)."""
    def build(x, cs, pos, o, H, D, RUN):
        half = D // 2
        S = hk.RS(fp32, 1, half, RUN)
        t = hk.arch.block_idx.x
        prow = pos.at[0, t].load_scalar(invariant=True) if positions else t
        w = hk.arch.warp_id() if warps > 1 else 0
        rows = [t * H + w + h * warps for h in range(H // warps)]
        x1s = [x.at[row, 0].load(S) for row in rows]
        x2s = [x.at[row, half].load(S) for row in rows]
        c = cs.at[prow, 0].load(S)
        sn = cs.at[prow, half].load(S)
        for row, x1, x2 in zip(rows, x1s, x2s):
            o1 = x1 * c
            o1 -= x2 * sn
            o2 = x2 * c
            o2 += x1 * sn
            o.at[row, 0].store(o1)
            o.at[row, half].store(o2)

    if positions:
        def body(x: hk.Tensor[dtype], cs: hk.Tensor[fp32], pos: hk.Tensor[hk.i32],
                 o: hk.Tensor[dtype], *, H=32, D=128, RUN=2):
            build(x, cs, pos, o, H, D, RUN)
    else:
        def body(x: hk.Tensor[dtype], cs: hk.Tensor[fp32], o: hk.Tensor[dtype], *,
                 H=32, D=128, RUN=2):
            build(x, cs, None, o, H, D, RUN)
    body.__name__ = name
    return hk.kernel(body, warps=warps, name=name, noalias=True,
                     grid=lambda p: (hk.cdiv(p.x.rows, p.H), 1, 1))


def _rope_grid(p):
    # x is (bh, 1, seq, D); one workgroup per (half-head column block, row
    # block, bh). The column axis only covers D/2 because each workgroup
    # handles a pair of columns j and j + D/2.
    return (p.HALF // p.COLS, hk.cdiv(p.x.rows, p.ROWS), p.x.batch)


def _rope_kernel(name: str, dtype, half: int):
    """Rotary embedding, NeoX/half-split convention.

        out[j]        = x[j] * cos[j] - x[j + D/2] * sin[j]
        out[j + D/2]  = x[j + D/2] * cos[j] + x[j] * sin[j]

    `cos` and `sin` are (seq, D/2) and are shared across heads, so they index
    with batch 0 while x indexes with the head. That is the whole reason the
    kernel takes them as separate globals instead of folding them in: they are
    broadcast along an axis x is not.
    """

    def body(x: hk.Tensor[dtype], cos: hk.Tensor[fp32], sin: hk.Tensor[fp32],
             o: hk.Tensor[dtype], *, ROWS=ROWS, COLS=COLS, HALF=half):
        t = _ftile(ROWS, COLS)
        bh = hk.arch.block_idx.z
        r = hk.arch.block_idx.y
        c = hk.arch.block_idx.x
        # The second half is HALF/COLS tiles further along the row.
        c2 = c + HALF // COLS

        lo = hk.tile_coord(bh, 0, r, c)
        hi = hk.tile_coord(bh, 0, r, c2)
        ang = hk.tile_coord(0, 0, r, c)

        xl = x[lo].load(t)
        xh = x[hi].load(t)
        cs = cos[ang].load(t)
        sn = sin[ang].load(t)

        o[lo] = (xl * cs - xh * sn).to(dtype)
        o[hi] = (xh * cs + xl * sn).to(dtype)

    body.__name__ = name
    return hk.kernel(body, warps=1, grid=_rope_grid, name=name)


_DTYPES = {"bf16": bf16, "fp16": fp16, "fp32": fp32}

#: Warp counts a silu_mul workgroup can have. Same set as the norm kernels;
#: WARPS is a launch geometry and so has to be baked into the kernel, while
#: TILES rides in as a constexpr argument.
WARP_COUNTS = tuple(1 << i for i in range(5))  # 1, 2, 4, 8, 16

KERNELS = {}
for _s, _dt in _DTYPES.items():
    for _w in WARP_COUNTS:
        KERNELS[f"silu_mul_{_s}_w{_w}"] = _silu_mul_kernel(
            f"silu_mul_{_s}_w{_w}", _dt, _w)
    KERNELS[f"rope_{_s}"] = _rope_kernel(f"rope_{_s}", _dt, COLS)
del _s, _dt, _w

#: Warp counts a strip silu_mul workgroup may have.
STRIP_WARPS = (1, 2, 4, 8)
STRIP_KERNELS = {f"strip_silu_mul_{_s}_w{_w}": _strip_silu_mul_kernel(
                     f"strip_silu_mul_{_s}_w{_w}", _dt, _w)
                 for _s, _dt in _DTYPES.items() for _w in STRIP_WARPS}

#: The rotary kernels: (dtype suffix, warps, with positions) -> Kernel; H,
#: D and RUN are constexprs.
ROTARY_WARPS = (1, 2, 4, 5, 8, 10, 16)
#: The 16-byte form's kernels; its plan is ROTARY16_PLAN.
ROTARY16_KERNELS = {f"rotary16_{_s}_w{_w}" + ("_pos" if _p else ""): _rotary16_kernel(
                        f"rotary16_{_s}_w{_w}" + ("_pos" if _p else ""), _dt, _w, _p)
                    for _s, _dt in (("bf16", hk.bf16), ("fp16", hk.fp16))
                    for _w in ROTARY_WARPS for _p in (False, True)}
ROTARY_KERNELS = {f"rotary_{_s}_w{_w}" + ("_pos" if _p else ""): _rotary_kernel(
                      f"rotary_{_s}_w{_w}" + ("_pos" if _p else ""), _dt, _w, _p)
                  for _s, _dt in _DTYPES.items() for _w in ROTARY_WARPS
                  for _p in (False, True)}
#: The rotary kernels of a target whose strips have no swap_halves (gfx1100).
ROTARY_HALVES_KERNELS = {
    f"rotary_halves_{_s}_w{_w}" + ("_pos" if _p else ""): _rotary_halves_kernel(
        f"rotary_halves_{_s}_w{_w}" + ("_pos" if _p else ""), _dt, _w, _p)
    for _s, _dt in _DTYPES.items() for _w in ROTARY_WARPS for _p in (False, True)}

#: Where the rotary kernels run on strips: gfx950 (swap_halves and
#: rotate_half are the wave64 port's).
STRIP_ARCHS = ("gfx950",)

#: The strip plan, (warps, chunks a warp), as elementwise's: one warp, one
#: 16-byte run a lane, 512 elements a workgroup. Measured on an MI355X
#: (tools/hk-bench/strip_pointwise.py, us), against tuned Triton, whose
#: search picked 512-element programs of two warps at every shape:
#:
#:                  w1 c1, 16 B   w1 c1, 8 B   rows first, best of 16   Triton
#:     2048x14336      25.1          28.2           26.0 (w1 c1)         25.3
#:     8192x14336     116.7         112.5          123.9 (w8 c8)        127.5
#:     2048x28672      59.2          57.6           63.4 (w8 c8)         60.9
#:     8192x28672     229.2         229.3          263.6 (w8 c8)        250.9
#:
#: "Rows first" is the grid this kernel had before workgroups walked along a
#: row: consecutive workgroups a row (57 KB at 14336) apart.
STRIP_PLAN = (1, 1)

#: silu_mul's strip plan on each arch it runs on strips: (warps, chunks a
#: warp), and (bytes, LDS reserve): past that many bytes moved, LDS reserved
#: to cap the workgroups a WGP holds, as the norms do past the Infinity
#: Cache. gfx1100 moved here from its 16x64 tile kernel, which refused the
#: halves of a gate/up output and fewer than 16 rows; measured for
#: hk.templates.elementwise on a W7900D (docs/templates/elementwise-rowwise.md),
#: gelu-and-mul on halves, us:
#:
#:                    tile kernel   strips w4c1   w8c1   w8c1 + 48 KB   torch.compile
#:     2048x14336        261.1         248.6      248.4      241.2         244.2
#:     8192x14336        979.3         973.9      972.2      948.3         956.2
#: gfx1201: gfx1100's plan, its LDS cap moved to the R9700's 64 MB Infinity
#: Cache. Not measured there yet.
SILU_STRIP_PLANS = {"gfx950": (STRIP_PLAN, None),
                    "gfx1100": ((8, 1), (96 << 20, 48 << 10)),
                    "gfx1201": ((8, 1), (64 << 20, 48 << 10))}


def silu_strip_plan(arch: str, moved: int):
    """(warps, chunks a warp, LDS reserve) of silu_mul's strip kernel on
    `arch` for a call that moves `moved` bytes, or None where silu_mul does
    not run on strips. hk.templates.elementwise plans rows with it."""
    got = SILU_STRIP_PLANS.get(arch)
    if got is None:
        return None
    (warps, chunks), cap = got
    return warps, chunks, (cap[1] if cap is not None and moved > cap[0] else 0)


# ---------------------------------------------------------------- wrappers

_TORCH_DTYPES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


def _suffix_of(t) -> str:
    name = _TORCH_DTYPES.get(str(t.dtype))
    if name is None:
        raise TypeError(f"hk fused kernels handle bf16/fp16/fp32, not {t.dtype}")
    return name


def _as_2d(t, what: str):
    if not t.is_contiguous():
        raise ValueError(f"{what} must be contiguous; call .contiguous()")
    cols = t.shape[-1]
    rows = t.numel() // cols if cols else 0
    if cols < COLS or rows < ROWS:
        raise ValueError(
            f"{what} is {rows}x{cols} and silu_mul needs at least {ROWS}x{COLS}. "
            f"The shape does not have to divide -- the last block backs its "
            f"address up and recomputes the overlap, which for a pointwise "
            f"kernel is exact -- but a whole tile has to fit."
        )
    r, c = canonical(rows, cols)
    # Already that shape -- `canonical` is frequently the identity, and
    # `Tensor.view` is about a microsecond of Python per operand.
    return t if r == rows and c == cols and t.dim() == 2 else t.view(r, c)


#: The row width silu_mul would rather walk, and the narrower ones it settles
#: for. See `canonical`. Measured by tools/hk-bench/fused_plan.py.
WIDTHS = (4096, 2048, 1024, 512, 256, 128, 64)


@_lru_cache(maxsize=None)
def canonical(rows: int, cols: int):
    """A (rows, cols) with the same elements that the memory system likes more.

    Memoized because `silu_mul` asks it three times per call, once per operand,
    and a launch at these sizes has only a couple of hundred microseconds of
    GPU behind it to hide Python in. The arguments are two ints and the answer
    is a pure function of them.

    silu_mul is elementwise, so the row structure is not part of the problem:
    a contiguous tensor may be viewed at any width dividing its element count
    and the kernel writes the same bytes to the same places. That makes the
    width a free parameter, and it is not a neutral one. The same kernel doing
    the same number of 16x64 tiles in the same number of workgroups ran at
    716 GB/s over an 8192x4096 tensor and 675 GB/s over an 8192x11008 one; the
    only difference between them is the stride a tile's 16 rows step by, 8192
    bytes against 22016.

    So a tensor whose own width is awkward is re-viewed at the widest power of
    two that divides its element count: 8192x11008 becomes 22016x4096. One that
    divides nothing in WIDTHS -- an odd hidden size -- keeps its own shape,
    which still works, just at the slower stride.

    Not applicable to the reductions in ops/norm.py: there the row *is* the
    problem and its width is not ours to choose.
    """
    n = rows * cols
    for w in WIDTHS:
        if n % w == 0 and n // w >= ROWS:
            return n // w, w
    return rows, cols


def plan(cols: int):
    """(WARPS, TILES) for a pointwise pass over rows of `cols` elements.

    Measured, not derived: see tools/hk-bench/fused_plan.py -- and what it
    measured refuted the hypothesis this knob was added for. Across
    2048x11008, 8192x11008 and 8192x4096 all fifteen (warps, tiles) settings
    landed within 5% of each other and the *smallest* of them, one warp holding
    one tile, won all three. A pointwise pass at these sizes already has 20k-90k
    workgroups in flight; there was never a shortage of waves to hide a load
    behind, and the only thing a wider workgroup changed was how many of its
    blocks were the clamped last one, which it then re-wrote.

    So: (1, 1). The knobs stay because the kernels are parameterized on them
    already and a narrower row or another arch could still want them, not
    because anything measured here does. `_canonical` below is what actually
    made this kernel faster.
    """
    del cols
    return 1, 1


#: (torch dtype, cols) -> (kernel, TILES). As in norm._PLAN_CACHE, and keyed
#: on the dtype object for the same reason: silu_mul at 2048x11008 is 216 us
#: of GPU, which a few microseconds of Python is a measurable fraction of --
#: it is one of the two rows where hk and torch.compile were within 1.5% of
#: each other.
_PLAN_CACHE: dict = {}


def _resolve(dtype, cols: int):
    got = _PLAN_CACHE.get((dtype, cols))
    if got is None:
        suffix = _TORCH_DTYPES.get(str(dtype))
        if suffix is None:
            raise TypeError(f"hk silu_mul handles bf16/fp16/fp32, not {dtype}")
        warps, tiles = plan(cols)
        got = _PLAN_CACHE[(dtype, cols)] = (
            KERNELS[f"silu_mul_{suffix}_w{warps}"], tiles)
    return got


def silu_mul(a, b, out=None):
    """silu(a) * b -- the second half of SwiGLU, with the gate already split.

    torch's equivalent is `F.silu(a) * b`; this does it in one pass, so the
    intermediate never reaches memory.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(
            f"silu_mul: operands differ -- {tuple(a.shape)}/{a.dtype} vs "
            f"{tuple(b.shape)}/{b.dtype}"
        )
    if out is None:
        import torch  # noqa: PLC0415  -- see norm._run
        out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    elif out.shape != a.shape or out.dtype != a.dtype:
        raise ValueError(f"silu_mul: out is {tuple(out.shape)}/{out.dtype}, expected "
                         f"{tuple(a.shape)}/{a.dtype}")
    cols = a.shape[-1] if a.dim() else 0
    plan = _strip_silu_plan(a.dtype, cols, a.numel() // cols if cols else 0)
    if plan is not None:
        return _strip_silu_mul(plan, a, b, out)
    av, bv, ov = _as_2d(a, "a"), _as_2d(b, "b"), _as_2d(out, "out")
    r, c = ov.shape
    if (r % ROWS or c % COLS) and (_overlaps(out, a) or _overlaps(out, b)):
        # A backed-up block is computed twice to the same answer only if its
        # inputs are still there the second time. In place, the workgroup
        # that covers it second can read the first one's output and apply
        # silu_mul again; so out of place, and copied back.
        return out.copy_(silu_mul(a, b))
    k, tiles = _resolve(a.dtype, av.shape[-1])
    k(av, bv, ov, TILES=tiles)
    return out


_STRIP_PLANS: dict = {}


def _strip_silu_plan(dtype, cols: int, rows: int):
    """(kernel, CW, RUN, EXACT, RESERVE) for silu_mul over `rows` rows of
    `cols` on strips, or None where it does not run on strips here."""
    from ..target import current_arch  # noqa: PLC0415

    arch = current_arch()
    nbytes = getattr(dtype, "itemsize", 2)
    plan = SILU_STRIP_PLANS.get(arch)
    big = plan is not None and plan[1] is not None and 3 * rows * cols * nbytes > plan[1][0]
    key = (dtype, cols, big)
    got = _STRIP_PLANS.get(key, False)
    if got is False:
        from ..target import get_target  # noqa: PLC0415
        from .norm import strip_run  # noqa: PLC0415

        got = None
        suffix = _TORCH_DTYPES.get(str(dtype))
        if plan is not None and suffix is not None and cols:
            warps, chunks, reserve = silu_strip_plan(arch, 3 * rows * cols * nbytes)
            run = strip_run(cols, _DTYPES[suffix].bytes)
            cw = chunks * get_target(arch).wave * run
            got = (STRIP_KERNELS[f"strip_silu_mul_{suffix}_w{warps}"], cw, run,
                   int(cols % (warps * cw) == 0), reserve)
        _STRIP_PLANS[key] = got
    return got


def _rows_view(t, what: str):
    """t as (rows, cols) without a copy: a unit column stride, and the
    leading axes compact over one row stride (a slice of the last axis, such
    as one half of a gate/up output, qualifies)."""
    if t.dim() and t.stride(-1) != 1 and t.shape[-1] > 1:
        raise ValueError(f"silu_mul: {what}'s last axis must have stride 1")
    cols = t.shape[-1] if t.dim() else 1
    if t.dim() <= 2:
        return t.view(1, cols) if t.dim() < 2 else t
    rows = t.numel() // cols if cols else 0
    try:
        return t.view(rows, cols)
    except RuntimeError as e:
        raise ValueError(f"silu_mul: {what}'s leading axes are not one row "
                         f"stride apart; call .contiguous()") from e


def _strip_silu_mul(plan, a, b, out):
    k, cw, run, exact, reserve = plan
    av, bv, ov = _rows_view(a, "a"), _rows_view(b, "b"), _rows_view(out, "out")
    if not ov.shape[0] or not ov.shape[1]:
        return out
    if _overlaps(out, a) or _overlaps(out, b):
        if not (out.data_ptr() in (a.data_ptr(), b.data_ptr())
                and out.stride() in (a.stride(), b.stride())):
            # Each element is read and written by one lane, so exactly in
            # place is safe; a partial overlap is not.
            return out.copy_(silu_mul(a.clone(), b.clone()))
    if reserve:
        k(av, bv, ov, CW=cw, RUN=run, EXACT=exact, RESERVE=reserve)
    else:
        k(av, bv, ov, CW=cw, RUN=run, EXACT=exact)
    return out


_ROTARY_CACHE: dict = {}


def _rotary_warps(heads: int) -> int:
    """Warps a token's workgroup has: four, or one per eight heads from 64
    heads up; a count that divides the heads. Measured on an MI355X (us,
    bf16, D 128; tuned Triton beside):

                    w4      w5      w8      w16     Triton
        2048x32    5.65            5.64    6.89     5.45
        2048x40    6.50    6.80    6.81             6.19
        2048x64   10.59           10.72   11.34    10.07
        8192x40   25.17   25.25   25.09            24.23
        8192x64   40.94           39.33   42.93    39.16
    """
    want = heads // 8 if heads >= 64 else 4
    return max(w for w in ROTARY_WARPS if heads % w == 0 and w <= want)


#: The 16-byte form's plan: (warps, chunks a warp) by the bytes a call moves.
#: Inside the cache one warp of two chunks, several workgroups a token;
#: past it a token a workgroup, two chunks a warp. Measured on an MI355X
#: (us, bf16, D 128, tuned Triton beside; the 4-byte kernel's plan as hk):
#:
#:                  4-byte   w1c2   whole token, c2   w1c1   Triton
#:     2048x32       5.57    5.24        5.32 (w4)      --     5.54
#:     2048x40       6.46    6.30        --             --     6.34
#:     2048x64      10.64   10.40       10.43 (w8)      --    10.09
#:     8192x32      20.25    --         20.05 (w4)    21.19   19.89
#:     8192x40      24.97    --         24.98 (w5)    24.59   24.16
#:     8192x64      39.58    --         39.37 (w8)    52.07   39.71
#:
#: An LDS reserve capping the workgroups a CU (20-80 KB, Triton's picks
#: carry waves_per_eu=2) moved none of them by more than 1%. Where a token's
#: chunks do not split into 4, 8 or 16 warps (40 heads) the eight-byte run
#: is faster: a workgroup of two warps a 512-column group, 24.72 us at
#: 8192x40, 24.53 preloaded, against w1c1's 24.76-25.1 and Triton's 24.29.
#: Every plan takes the preloaded twin past norm.STRIP_PRELOAD_PAST
#: (8192x64 39.65 -> 39.33, 2048x64 10.44 -> 10.20; docs/history/43).
ROTARY16_IN_CACHE = 64 << 20


def _rotary16_plan(tokens: int, heads: int, d: int, nbytes: int):
    """(warps, CH, RUN, preloaded) for rotary16, or None where it does not
    run: a head of 128 or 256 columns (halves 8 or 16 lanes apart at RUN 8),
    whole chunks a token."""
    from ..target import current_arch  # noqa: PLC0415
    from .norm import strip_preloads  # noqa: PLC0415

    if current_arch() not in STRIP_ARCHS or d not in (128, 256) or nbytes != 2:
        return None
    chunks = heads * d // 512
    if heads * d % 512:
        return None
    moved = 2 * tokens * heads * d * nbytes
    pre = strip_preloads(moved)
    ch = 2 if chunks % 2 == 0 else 1
    if moved <= ROTARY16_IN_CACHE:
        return 1, ch, 8, pre
    if chunks // ch in (4, 8, 16):
        return chunks // ch, ch, 8, pre
    if d == 128:
        return 2, 1, 4, pre
    return 1, 1, 8, pre


def rotary(x, cos_sin, positions=None, out=None):
    """vLLM's rotary_embedding, NeoX (half-split) style, token-major:
    `x` is (..., heads, D) with the tokens on the leading axes, `cos_sin` is
    (P, D) float32 holding [cos | sin] (vLLM's cos_sin_cache), and token t
    turns by row `positions[t]` of it -- int32, one per token -- or by row t
    without positions:

        out[..., :D/2] = x1 * cos - x2 * sin,  out[..., D/2:] = x2 * cos + x1 * sin

    In place when `out` is `x`. D is 64 * a power of two up to 16 (128 and
    256 qualify) on gfx950, 64 * a power of two up to 8 on gfx1100. On
    strips: gfx950 and gfx1100 (where a head's halves are two strips, there
    being no lane exchange there)."""
    import torch  # noqa: PLC0415

    from ..target import current_arch, get_target  # noqa: PLC0415

    arch = current_arch()
    halves = arch not in STRIP_ARCHS
    if halves and not get_target(arch).strips:
        raise NotImplementedError(f"hk.ops.rotary runs on register strips (gfx950, "
                                  f"gfx1100); on {arch} use hk.ops.rope")
    if x.dim() < 2 or not x.is_contiguous():
        raise ValueError(f"rotary: x must be a contiguous (..., heads, D) tensor, "
                         f"got {tuple(x.shape)}")
    h, d = x.shape[-2], x.shape[-1]
    run = d // 64
    if d % 64 or run not in ((1, 2, 4, 8) if halves else (1, 2, 4, 8, 16)):
        raise ValueError(f"rotary: head dim {d} is not 64 x a power of two up to "
                         f"{8 if halves else 16} (on {arch})")
    if cos_sin.dim() != 2 or cos_sin.shape[1] != d or cos_sin.dtype != torch.float32 \
            or not cos_sin.is_contiguous():
        raise ValueError(f"rotary: cos_sin must be a contiguous float32 (P, {d}), got "
                         f"{tuple(cos_sin.shape)}/{cos_sin.dtype}")
    tokens = x.numel() // (h * d) if h and d else 0
    if positions is not None and (positions.dtype != torch.int32
                                  or positions.numel() != tokens):
        raise ValueError(f"rotary: positions must be {tokens} int32s, got "
                         f"{positions.numel()} {positions.dtype}")
    if positions is None and cos_sin.shape[0] < tokens:
        raise ValueError(f"rotary: {tokens} tokens and no positions need "
                         f"{tokens} rows of cos_sin, got {cos_sin.shape[0]}")
    if out is None:
        out = torch.empty_like(x)
    if not tokens:
        return out
    plan = _rotary16_plan(tokens, h, d, x.element_size())
    if plan is not None:
        warps, ch, run, pre = plan
        k = ROTARY16_KERNELS[f"rotary16_{_suffix_of(x)}_w{warps}"
                             + ("_pos" if positions is not None else "")]
        k = k.preloaded() if pre else k
        xv, ov = x.view(tokens, h * d), out.view(tokens, h * d)
        if positions is not None:
            k(xv, cos_sin, positions.view(1, tokens), ov, D=d, CH=ch, RUN=run)
        else:
            k(xv, cos_sin, ov, D=d, CH=ch, RUN=run)
        return out
    key = (x.dtype, h, positions is not None)
    k = _ROTARY_CACHE.get(key)
    if k is None:
        suffix = _suffix_of(x)
        warps = _rotary_warps(h)
        table, kind = (ROTARY_HALVES_KERNELS, "rotary_halves") if halves else             (ROTARY_KERNELS, "rotary")
        k = _ROTARY_CACHE[key] = table[
            f"{kind}_{suffix}_w{warps}" + ("_pos" if positions is not None else "")]
    if halves:
        # A half of D columns is one wave32 chunk of D/64 a lane.
        run = d // 64
    xv, ov = x.view(tokens * h, d), out.view(tokens * h, d)
    if positions is not None:
        k(xv, cos_sin, positions.view(1, tokens), ov, H=h, D=d, RUN=run)
    else:
        k(xv, cos_sin, ov, H=h, D=d, RUN=run)
    return out


#: torch dtype -> the rope kernel for it. One f-string and one `str(dtype)`
#: per launch is not much, and a 32x2048x128 rope is 50 us.
_rope_cache: dict = {}


def rope(x, cos, sin, out=None):
    """Rotary embedding over the last axis, half-split (NeoX) convention.

    `x` is (..., seq, D) with at least one leading axis; `cos` and `sin` are
    (seq, D/2) in fp32 and are shared across every leading axis. D/2 must be a
    multiple of 64 -- head dims of 128 and 256 qualify, 64 does not, and the
    error says so rather than silently producing a rotation by the wrong angle.
    """
    import torch  # noqa: PLC0415

    if x.dim() < 3:
        raise ValueError(
            f"rope expects (batch*heads, seq, D), got {tuple(x.shape)}. Reshape "
            f"the head axis in; the kernel indexes it as the global's batch."
        )
    if not x.is_contiguous():
        raise ValueError("rope: x must be contiguous")
    seq, d = x.shape[-2], x.shape[-1]
    half = d // 2
    if d % 2 or half % COLS:
        raise ValueError(
            f"rope: head dim {d} gives a half of {half}, which is not a multiple "
            f"of {COLS}. The kernel pairs column j with column j + D/2 by loading "
            f"two tiles, so the halves have to land on tile boundaries."
        )
    if seq % ROWS:
        raise ValueError(f"rope: sequence length {seq} must be a multiple of {ROWS}")
    for nm, tbl in (("cos", cos), ("sin", sin)):
        if tuple(tbl.shape) != (seq, half):
            raise ValueError(
                f"rope: {nm} is {tuple(tbl.shape)}, expected ({seq}, {half})"
            )
        if tbl.dtype != torch.float32:
            raise TypeError(f"rope: {nm} must be float32, got {tbl.dtype}")

    if out is None:
        out = torch.empty_like(x)
    bh = x.numel() // (seq * d)
    k = _rope_cache.get(x.dtype)
    if k is None:
        k = _rope_cache[x.dtype] = KERNELS[f"rope_{_suffix_of(x)}"]
    k(x.view(bh, 1, seq, d), cos.view(1, 1, seq, half), sin.view(1, 1, seq, half),
      out.view(bh, 1, seq, d), HALF=half)
    return out


__all__ = ["KERNELS", "ROWS", "COLS", "MAX_TILES", "WARP_COUNTS", "WIDTHS",
           "canonical", "plan", "silu_mul", "rope", "rotary"]
