"""Row-wise normalizations: RMSNorm, LayerNorm, softmax.

The three share a skeleton -- reduce along a row, then rescale that row -- and
the skeleton is where all the difficulty is, so it is written once here and the
three kernels differ only in which reduction and which rescale.

Four things about that skeleton are worth reading before the code.

**The tensor does not have to tile.** include/rdna3 bounds-checks nothing, so a
partial block cannot simply be read: past the end of a torch allocation is a
fault, not zeros. The library's own answer (attn.cpp:735) is to back the
*address* up until the tile is wholly in bounds. That is exact for an
elementwise kernel -- the overlapping elements are simply computed twice with
the same answer -- but a reduction *along the axis you backed up* would count
those elements twice and give a wrong mean. So the overlap is masked with the
reduction's identity before it is folded in:

    lo  = s_min(cb * COLS, cols - COLS)   # backed-up start of block cb
    pad = cb * COLS - lo                  # 0 except in the last block
    t   = left_fill(load(x, (row, lo)), pad, identity)

Only for a reduction that is not idempotent, though. The overlap is real
elements of the same row, so a max or a min cannot be moved by seeing one of
them a second time; softmax's max pass and quantize's absmax pass skip the mask
entirely, and only the sums pay for it. The mask is also the reason `pad` is a
thunk: two of the three passes never ask for it.

Along rows there is nothing to mask: each row's normalization is independent of
every other, so recomputing a row in the backed-up last block writes the same
bytes. `lo_row` is a plain back-up.

Because the tile has to fit, this needs `rows >= ROWS` and `cols >= COLS`. A
hidden dimension smaller than 64 is not a case worth a second code path; the
wrapper says so rather than reading past the tensor.

**The accumulator is fp32 even when the tensor is bf16.** Not for the usual
accuracy reason alone: an rv's lane layout is derived from its element type, so
a bf16 tile and an fp32 vector do not agree about which lane holds which row,
and there is no such thing as a mixed-precision reduction here. The tensor is
loaded straight into an fp32 tile instead -- `kittens::load` runs the elements
through `base_types::convertor` on the way in -- and everything downstream is
fp32 until the store converts back.

That load is not free, and for a long time it was much less free than this
paragraph used to claim. An fp32 tile's lanes own every *other* column of a
16-element run, so a widening load issued one narrow access per element: 32
`global_load_u16` per 16x64 tile where two `global_load_b128` would do. The
answer is the same one the narrowing store already used in the other
direction -- read the whole run and pick, rather than ask the memory system
for a strided gather -- and it lives in `load`'s `widening` branch. It is also
why MAX_TPW's register figures moved: the narrow loads were live values.

**One warp per row block is not enough warps.** A 4096-row tensor at 16 rows
per block is 256 workgroups; this GPU has 192 SIMDs, so a one-warp workgroup
leaves 1.3 waves per SIMD and nothing to hide a load behind. The row is split
across WARPS warps of one workgroup instead and the partial reductions are
combined through LDS (lang.collective.cross_warp). That is parallelism bought
without an extra pass, and it is most of the difference between this and
torch.compile's numbers.

**The other half is the pass count.** Written the obvious way this is three
passes over memory for softmax and two for the others, because the values are
re-read rather than kept. torch inductor's generated kernel is *persistent*: it
holds the row in registers and reads it once. Per achieved byte the two are the
same speed, so the extra pass is the entire gap.

So this holds the row too, when it fits. `TPW` is how many 16x64 fp32 tiles
each warp keeps live; at WARPS=16 and TPW=4 that is a 4096-wide row held across
a 512-thread workgroup at 128 VGPRs a lane, and the kernel reads global memory
exactly once. When it does not fit, `TPW=0` selects the streaming variant and
the same code re-reads instead -- `visit()` below is deliberately the only
place that knows which.

A 16384-wide row does not fit either way: 16 rows of it is a megabyte. But *one*
row of it does, and `FOLD` is how a 16-row tile addresses a one-row problem --
the wrapper reshapes (R, C) to (R*16, C/16) so a tile's 16 rows are 16 chunks
of a single row, and `fold_rows` combines them at the end. See
lang.collective.fold_rows.

`plan` picks between the three, and it needs to know *which* of the three ops
is asking. rms and layer want the row held unfolded wherever that is legal;
softmax wants it folded at the same width, by half again. The reason is in
`_FOLD_FIRST_KINDS`, with the measurement.
"""


from typing import Any, Optional

from .. import dsl as hk
from ..dsl.collective import cross_warp, fold_rows
from ..types import bf16, fp16, fp32

#: 16 rows x 64 columns of fp32 is 32 VGPRs. 64 bf16 columns is also what makes
#: the global load issue 16-byte accesses instead of going elementwise.
ROWS, COLS = 16, 64

#: Live fp32 tiles one warp may hold in the persistent variant. Four is 128
#: VGPRs of data; the reductions' temporaries bring a quantize kernel to 167,
#: which is under the 240 cliff and leaves 9 waves per SIMD at gfx1100's
#: allocation granule of 24.
#:
#: Five was tried and rejected, and the two reasons are worth keeping apart.
#:
#: It is *nearly* legal now. Five used to spill outright -- a spill here is a
#: wrong answer rather than a slowdown, see hk.build's hard gate -- and the
#: widening global load freed enough registers that the exact variant builds:
#: rms 216/occ 7, layer 216/occ 7, softmax 240/occ 6 with four registers of
#: headroom. The *inexact* variant of softmax, which carries the clamp and the
#: mask, still does not, and plan emits TPW=MAX_TPW at widths that are not a
#: whole number of blocks. So five is not available at every width it would be
#: chosen for, which is already disqualifying.
#:
#: And it would not pay where it is available. Raising the cap changes the plan
#: for exactly one band -- 4097 to 5120 columns, 65 to 80 blocks -- which
#: otherwise falls through to FOLD. Measured at the top of that band
#: (norm_plan.py, rms 16384x5120, the shape Qwen2.5-14B would hit):
#:
#:     w4  tpw2 fold16   0.467      <- what MAX_TPW=4 picks
#:     w16 tpw5 fold1    0.512      <- what MAX_TPW=5 would pick
#:     w16 tpw0 fold1    0.606
#:
#: Ten percent the wrong way. Five tiles plus the reduction temporaries is 216
#: VGPRs and occupancy 7 where four is 167 and occupancy 9, and at this width
#: the extra occupancy is worth more than the saved pass. The cliff is between
#: four and five, not between five and six.
MAX_TPW = 4

#: 16 warps is a 512-thread workgroup. Past that the workgroup stops fitting
#: alongside anything else and the barrier starts to cost more than the
#: parallelism is worth.
MAX_WARPS = 16

_NEG_INF = float("-inf")



def _ftile(rows, cols, fold=1):
    """The fp32 tile a row is worked in. Where tiles take a fragment shape
    (gfx950) it is laid out in 16x32 fragments, so a lane's run of a 16-bit row is 8 elements:
    one 16-byte buffer load or store where the 16x16 fragment's is 8 bytes.
    Not when the row is folded onto the tile's rows (FOLD > 1): fold_rows
    reduces through a 16x16 tile whose vector is retyped onto this one's,
    and include/cdna4's vectors differ by fragment."""
    if hk.arch.target().tile_shapes and fold == 1:
        return hk.RT(fp32, rows, cols, frag=(16, 32))
    return hk.RT(fp32, rows, cols)

def _grid(p):
    # One workgroup per row block. The column loop is inside the kernel: a
    # normalization is a reduction along the row, so the whole row has to be
    # seen by one workgroup, and splitting it across workgroups would need a
    # second pass through memory to combine the partials.
    return (1, hk.cdiv(p.x.rows, p.ROWS), 1)


def _block_start(idx, block: int, extent, exact: bool = False):
    """Where block `idx` starts, backed up so the whole tile is in bounds.

    Returns `(start, backed_up_by)`. The second is how many leading elements of
    the loaded tile repeat the previous block -- zero for every block but the
    last, and zero for that one too when the extent divides. It is a thunk
    because two of the three passes do not need it, and an unused `const int`
    in the generated C++ is a line the next reader has to rule out.

    `exact` says the caller has *proved*, at trace time, that no block is ever
    backed up: the extent is a whole number of blocks and no warp is handed an
    index past the last of them. Then there is no clamp and no overlap, and the
    thunk is None -- which is not the same as a thunk returning zero. A zero
    that only the host knows about still costs the kernel a `const int` it
    compares against every column, and it costs LayerNorm a whole tile copy,
    because the only reason that copy exists is to have something maskable that
    is not the held tile. See `_mask` and `_exact`.
    """
    want = idx * block
    if exact:
        return want, None
    lo = hk.min(want, extent - block)
    return lo, lambda: want - lo


def _mask(tile, pad, ident):
    """Fill the backed-up overlap of `tile` with a reduction's identity.

    A no-op when `pad` is None -- see `_block_start` -- so that a kernel can
    write the masking unconditionally and still emit nothing for a width that
    divides.
    """
    if pad is not None:
        tile.left_fill_(pad(), ident)
    return tile


def _exact(cols: int, warps: int, tpw: int) -> bool:
    """Is every column block this plan visits wholly in bounds, unclamped?

    Two conditions, and both are decided by `plan` before anything is traced:
    the row is a whole number of blocks, and the warp-slots the kernel hands
    out land exactly on those blocks. The persistent variant hands out
    WARPS*TPW of them once; the streaming one hands out WARPS per step for
    ceil(nblk/WARPS) steps, so it needs WARPS to divide nblk or the last step
    runs off the end.
    """
    if cols % COLS:
        return False
    nblk = _ceil(cols, COLS)
    return warps * tpw == nblk if tpw else nblk % warps == 0


def _ceil(a: int, b: int) -> int:
    # Not lang.host.cdiv: that builds a *traced* host expression for the grid
    # lambda, and everything in plan() is a plain Python integer decided before
    # anything is traced.
    return -(-a // b)


def _split(nblk: int):
    """(warps, tiles per warp) for `nblk` column blocks in one workgroup.

    One warp per block, capped at MAX_WARPS. Only past sixteen blocks does a
    warp hold more than one tile, and then as evenly as ceiling division
    allows.

    **This used to round the warp count down to a power of two**, which costs
    nothing when `nblk` is itself a power of two and a great deal when it is
    not: five blocks went to four warps holding two tiles each, eight
    warp-slots for five blocks, so three eighths of the warps did redundant
    work on the clamped last block while every warp paid for a second live
    tile. Measured at the width where that happens (tools/hk-bench/
    odd_warps.py, 5120 columns, which folds to five blocks -- and is
    Qwen2.5-14B's hidden size):

                      quantize 16384x5120   rms 16384x5120   softmax 8192x5120
        w6 tpw1             0.366               0.465              0.254
        w5 tpw1             0.368               0.478              0.253
        w3 tpw2             0.385               0.487              0.258
        w4 tpw2 (was)       0.407               0.474              0.265
        torch.compile       0.371               0.780              0.335

    Note which axis explains it. w3 and w6 hand out the same six slots and w6
    is the faster of the two in all three ops, so this is not slot-counting:
    it is TPW. A second live tile is 32 more VGPRs a lane, and at this width
    the occupancy is worth more than the halved workgroup count. The old rule
    could not express "five warps" and so could not reach either.

    That quantize row is the whole reason this changed -- it was the one shape
    in the Phase 2 bench where hk lost to torch.compile.

    Five warps and not six, even though six is faster in the table: six covers
    five blocks with a sixth warp that has nothing of its own and is clamped
    onto the last block, and "round the warp count *up* past the work" is a
    rule with one supporting measurement (rms, 2.8%, with quantize and softmax
    calling it noise) and no mechanism. Five covers the blocks exactly. The
    price is that rms at this one width gives up 1% against the old w4 tpw2,
    which is what the other two ops gain eleven and four and a half percent to
    pay for.

    The cost is kernels: WARPS is a constexpr, so admitting every count from 1
    to 16 takes the shipped table from 45 norm kernels to 144. They are
    compiled lazily and cached on content hash, so a process still pays only
    for the widths it sees.
    """
    warps = min(MAX_WARPS, nblk)
    return warps, _ceil(nblk, warps)


def _fold_split(nblk: int):
    """(warps, tiles per warp) for a row folded onto a tile's rows.

    _split's one-warp-a-block, except on a wave64 (CDNA) target from eight
    blocks up, where four tiles a warp and a quarter of the warps wins. A
    16x64 fp32 tile there is 16 VGPRs a lane, not 32, so the live tiles that
    cost gfx1100 its occupancy are cheap. Measured on an MI355X
    (norm_plan_sweep, bf16, x torch):

                         w8 tpw1 (split)   w2 tpw4   best
        rms   16384x8192      0.90           0.97    1.01 (w1 tpw8)
        layer 16384x8192      0.84           1.01    1.02 (w1 tpw8)
        softmax 16384x8192    0.76           0.93    0.93
        rms     512x8192      1.08           1.07    1.10 (w4 tpw2)

    Below eight blocks the alternatives were within 3% either way. Past 64
    a quarter of the blocks is more warps than a workgroup has (a 131072- or
    262144-wide softmax asked for a 32- and a 64-warp kernel), and the row
    goes to _split.
    """
    if 8 <= nblk <= 4 * MAX_WARPS and nblk % 4 == 0:
        from ..target import current_arch, get_target  # noqa: PLC0415

        if get_target(current_arch()).wave == 64:
            return nblk // 4, 4
    return _split(nblk)


#: Below this many warps the folded view is not worth taking. Folding divides
#: the blocks a workgroup covers by 16, so a row narrow enough would fold down
#: to a one- or two-warp workgroup that still pays fold_rows' LDS round trip.
#: Four is where the measurement starts -- 4096 columns fold to four blocks,
#: which is the narrowest folded case in the sweep. It gates only the
#: fold-first branch; a row too wide to hold unfolded folds at any width,
#: because there the alternative is streaming.
_FOLD_MIN_WARPS = 4

#: Which ops take the folded view *before* trying to hold the row unfolded.
#: Both read memory exactly once; what differs is how the work is spread. At
#: 16384x4096 the unfolded plan is 1024 workgroups of 16 warps holding four
#: tiles each, the folded one 16384 workgroups of four warps holding one --
#: same tiles, four times the warps, a quarter of the live registers in any of
#: them, and the register count is what buys occupancy.
#:
#: That used to be the whole argument, and it used to cover all three ops.
#: It does not any more. Since `load` grew its widening fast path -- a 16-bit
#: global read into an fp32 tile now issues two `global_load_b128` per 16x64
#: tile where it used to issue 32 `global_load_u16` -- the unfolded hold is
#: cheaper in registers (the narrow loads were live values: TPW=4 went 207
#: VGPR to 167, occupancy 7 to 9) and far cheaper in issue slots. It now wins
#: for rms and layer wherever it is legal. Measured (norm_plan.py, burst
#: timing, ms):
#:
#:                       folded w4 tpw1   unfolded w16 tpw4
#:     rms   16384x4096      0.402              0.361
#:     layer 16384x4096      0.378              0.354
#:     rms    8192x4096      0.201              0.172
#:     rms    6144x4096      0.117              0.087
#:     softmax 8192x4096     0.191              0.290   <- the exception
#:
#: softmax was the exception, and not by a little: folded was 1.5x there, and
#: the ordering held at 2048x4096 too (0.051 vs 0.059). Its passes were the
#: arithmetic-bound ones -- two `v_exp_f32` an element, quarter rate on
#: gfx1100 -- so occupancy was worth more to it than the saved issue slots.
#: The online softmax computes one exp an element, and the exception went
#: with the second (scoreboard harness, ms, folded / unfolded):
#:
#:     softmax 2048x4096     0.0317 / 0.0317
#:     softmax 8192x4096     0.1978 / 0.1860
#:
#: So no op folds first by kind any more. What still folds first is a short
#: tensor (_FOLD_ROWS_BELOW), for every kind.
_FOLD_FIRST_KINDS: tuple = ()

#: Below this many rows a foldable row is folded first, whatever the op: the
#: unfolded plan is a workgroup per sixteen rows, too few to fill the card,
#: where folded is one a row. The weighted norms measured the crossing
#: (_WEIGHTED_UNFOLD_ROWS) and softmax ties at 2048.
_FOLD_ROWS_BELOW = 4096


def plan(cols: int, kind: str = "rms", rows: Optional[int] = None):
    """(WARPS, TPW, FOLD) for a normalization over rows of `cols` elements,
    `rows` of them if known (fewer than _FOLD_ROWS_BELOW fold first).

    Three outcomes, all about the row:

      persistent          TPW in 1..MAX_TPW, FOLD=1. A whole 16-row slab held
                          in registers across the workgroup, so the row is
                          read exactly once.
      persistent + fold   FOLD=ROWS. The caller reshapes so one workgroup
                          covers one row and a tile's 16 rows are 16 chunks of
                          it. Also read once, at a quarter of the live tiles
                          per warp. Tried first only for the kinds in
                          _FOLD_FIRST_KINDS; otherwise it is the fallback for
                          a row too wide to hold unfolded.
      streaming           TPW=0, FOLD=1. Neither fits; re-read the row.

    `kind` is the one of "rms" / "layer" / "softmax" being planned, and it is
    here because the three do not want the same answer -- see
    _FOLD_FIRST_KINDS for the measurement that says so.

    **This function used to ask a fourth question and no longer does.** It
    took the bytes that had to stay live for the re-read to hit, and when they
    fit in the 96 MB Infinity Cache it capped TPW and refused to fold, on the
    argument that paying registers to avoid a second read is a straight loss
    when the second read is a cache hit at better than a terabyte a second.
    The argument is still sound and the conclusion was still wrong, for the
    same reason quant.plan's copy of it was: it priced the hold at what the
    hold used to cost. With the widening load the held variants win inside the
    cache as well, by more than they win outside it (norm_plan.py, every case
    resident; `plan` here is what the cache branch chose):

        rms   4096x4096   plan w16 tpw0 f1 0.088   w16 tpw4 f1 0.063
        layer 4096x4096   plan w16 tpw0 f1 0.086   w16 tpw4 f1 0.066
        rms   6144x4096   plan w16 tpw0 f1 0.117   w16 tpw4 f1 0.087
        softmax 2048x4096 plan w16 tpw0 f1 0.056   w4  tpw1 f16 0.051

    Twenty-four to twenty-nine percent. So the regime question is gone, and
    with it the `working_set` argument and _CACHE_MAX_TPW. What remains true
    of the old branch is only its negative half: the *flat* rule it replaced
    (TPW=0, FOLD=1 everywhere, resident or not) was wrong by forty percent at
    1024 columns, and nothing below would pick it.

    Powers of two only for WARPS, so that the number of distinct kernels a
    process compiles stays small -- every (kind, dtype, WARPS, TPW, FOLD) is a
    separate hipcc invocation the first time it is seen.

    **What this still gets wrong**, from the sweep that fixed the rest of it
    (norm_plan.py, 16 cases; the rule names the winner or ties it in 12):

        rms 32768x2048   plan w16 tpw2 f1  0.358   w2 tpw0 f16  0.335   -6%
        rms 65536x1024   plan w16 tpw1 f1  0.367   w1 tpw0 f16  0.356   -3%
        softmax 8192x5120 plan w4 tpw2 f16 0.289   w2 tpw3 f16  0.275   -5%

    The first two are the same shape of miss: at a narrow width a *folded
    streaming* workgroup wins, and this function never emits one -- FOLD is
    only ever reached with a TPW that holds. The obvious patch is a fourth
    branch, and it is not taken, because 1024 columns disagrees with itself:
    at 16384 rows the plan's w16 tpw1 f1 is the outright winner (0.058 against
    0.064 for w1 tpw0 f16), and at 65536 rows it loses. Same width, opposite
    answers, so the width cannot decide it -- and what distinguishes the two
    is total size, which is the regime question this function just finished
    deleting on four cases' worth of evidence. Three percent is not enough to
    reinstate it on two.

    The third is _split's doing: 5 blocks over powers-of-two warps is 4x2,
    and the winner is 2x3. Admitting non-powers-of-two for WARPS would fix it
    and would roughly double the kernel count.
    """
    foldable = cols % ROWS == 0 and cols // ROWS >= COLS
    short = rows is not None and rows < _FOLD_ROWS_BELOW
    if (kind in _FOLD_FIRST_KINDS or short) and foldable:
        fw, ftpw = _fold_split(_ceil(cols // ROWS, COLS))
        if ftpw <= MAX_TPW and fw >= _FOLD_MIN_WARPS:
            return fw, ftpw, ROWS
    warps, tpw = _split(_ceil(cols, COLS))
    if tpw <= MAX_TPW:
        return warps, tpw, 1
    # Too wide to hold unfolded. Fold anyway if it is legal at all -- even a
    # two-warp folded workgroup beats re-reading the row, which is why
    # _FOLD_MIN_WARPS does not apply here.
    if foldable:
        fw, ftpw = _fold_split(_ceil(cols // ROWS, COLS))
        if ftpw <= MAX_TPW:
            return fw, ftpw, ROWS
    return warps, 0, 1


def _row_reduce_kernel(name: str, dtype, kind: str, warps: int,
                       bounded: bool = False, split: int = 0):
    """A kernel that reduces each row and rescales it.

    `kind` picks the arithmetic:
      rms    -- y = x * rsqrt(mean(x^2) + eps)
      layer  -- y = (x - mean(x)) * rsqrt(var(x) + eps)
      softmax-- y = exp(x - max(x)) / sum(exp(x - max(x)))

    `bounded` takes the bounded loads and stores, the buffer accesses the
    weighted norms use; every tile here is backed up into bounds, so they
    read and write the same elements. `split` is HK_RT_HALF_SPLIT, which only
    the bounded forms have.
    """
    def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype], *, ROWS=ROWS, COLS=COLS,
             WARPS=warps, TPW=0, FOLD=1, EXACT=0, EPS=1e-6):
        t = _ftile(ROWS, COLS, FOLD)
        vt = hk.col_vec(t)

        def put(idx, v):
            if bounded:
                o[idx].store(v, bounded=True)
            else:
                o[idx] = v
        ncols = x.cols
        nblk = hk.cdiv(ncols, COLS)
        row_lo, _ = _block_start(hk.arch.block_idx.y, ROWS, x.rows)
        # How many elements of the *original* row this workgroup covers. With
        # FOLD the tensor has been reshaped, so the tensor's column count is a
        # sixteenth of the row length and the means below would be 16x too big.
        rowlen = ncols * FOLD if FOLD != 1 else ncols

        def cb_of(step):
            """The column block this warp handles at `step`.

            Strided rather than blocked, so that at any moment the warps of a
            workgroup are reading 16 adjacent blocks -- one contiguous 4 KB
            run per row -- instead of 16 places a row apart.
            """
            return step * WARPS + hk.arch.warp_id() if WARPS > 1 else step

        def tile(cb):
            col_lo, pad = _block_start(cb, COLS, ncols, EXACT)
            idx = hk.elem_coord(0, 0, row_lo, col_lo)
            return x[idx].load(t, bounded=bounded), pad, idx

        # Over-provisioned blocks -- WARPS*TPW can exceed nblk, and a warp
        # whose step runs past the end gets cb >= nblk -- are not a special
        # case. _block_start clamps them to the last in-bounds block, so they
        # re-read data another warp already has; the mask then makes them the
        # identity for the reduction, and the write pass writes the same bytes
        # to the same place. Both are exactly what backing up already does for
        # a non-dividing tail.
        held = [tile(cb_of(s)) for s in range(TPW)] if TPW else None
        steps = None if TPW else hk.cdiv(nblk, WARPS)

        def visit(fn):
            """Call `fn(v, pad, idx)` once per column block this warp owns.

            Persistent: the tiles are already in registers and this costs
            nothing. Streaming: a device loop that re-reads them. Keeping the
            two behind one call is what stops the variants from drifting --
            everything below this point is written once.

            The one rule `fn` has to obey: **do not write to `v` except in the
            last pass.** In the streaming variant `v` is a fresh load and
            scribbling on it is harmless; in the persistent variant it is the
            tile the write pass will store. `left_fill_` writes in place -- it
            has to, see hk.dsl.values.Tile._fill -- so masking `v` directly
            would zero the overlap columns of the *output*, and another warp's
            copy of those same columns would then race with them. A shape that
            divides never shows it, because there the mask is empty. Mask a
            derived tile instead; every reduction below already makes one.
            """
            if TPW:
                for rec in held:
                    fn(*rec)
            else:
                for s in range(steps):
                    fn(*tile(cb_of(s)))

        def combine(vec, op):
            """Finish a per-warp partial: across the workgroup's warps, then
            across the tile's rows when they are chunks of one row."""
            vec = cross_warp(vec, op, WARPS)
            return fold_rows(vec, op, WARPS) if FOLD != 1 else vec

        if kind == "softmax":
            # Online: a running max and a sum of exp under it, rescaled when
            # the max moves, then the warps' (max, sum) pairs combined. The
            # max is unmasked -- a backed-up block re-reads real columns of
            # the same row, and a max cannot be moved by an element it has
            # already seen -- the sum is masked, since a sum is not
            # idempotent: shift first, mask second, so the overlap is -inf
            # and exp(-inf) = 0.
            #
            # Held, it is one pass and one exp an element: each tile becomes
            # its own exp(v - m_tile) in place, and the write scales it by
            # exp(m_tile - M) / S, a vector. Three passes and two exps an
            # element before (max; sum of exp; exp again to write), and
            # v_exp_f32 is quarter rate on gfx1100. Streamed, it is two
            # passes over memory instead of three.
            if TPW:
                tms = []
                for v, pad, idx in held:
                    mt = hk.full(vt, _NEG_INF)
                    mt.max_(v.max(axis=1))
                    v -= mt
                    v.exp_()
                    st = hk.zeros(vt)
                    st += (_mask(v.copy(), pad, 0.0) if pad is not None
                           else v).sum(axis=1)
                    tms.append((mt, st))
                m = tms[0][0].copy()
                for mt, _ in tms[1:]:
                    m.max_(mt)
                s = hk.zeros(vt)
                for mt, st in tms:
                    w = mt - m
                    w.exp_()
                    s += st * w
            else:
                m = hk.full(vt, _NEG_INF)
                s = hk.zeros(vt)

                # In place throughout: this runs inside visit()'s device
                # loop, where a rebound name would not be carried round it.
                def online(v, pad, idx):
                    nonlocal m, s
                    m_old = m.copy()
                    m.max_(v.max(axis=1))
                    alpha = m_old - m
                    alpha.exp_()
                    s *= alpha
                    p = _mask(v - m, pad, _NEG_INF)
                    p.exp_()
                    s += p.sum(axis=1)

                visit(online)
            g_m = combine(m, "max")
            w = m - g_m
            w.exp_()
            s *= w
            s = combine(s, "add")

            # The write. Unmasked -- a repeated column is written twice with
            # the same value, which is the point of backing up -- and stored
            # from fp32: the store narrows in registers, where .to() first
            # moved every element across the wave halves.
            if TPW:
                for (v, pad, idx), (mt, _) in zip(held, tms):
                    scale = mt - g_m
                    scale.exp_()
                    scale /= s
                    v *= scale
                    put(idx, v)
            else:
                def write(v, pad, idx):
                    p = v - g_m
                    p.exp_()
                    p /= s
                    put(idx, p)

                visit(write)
            return

        sq = hk.zeros(vt)
        mean = hk.zeros(vt) if kind == "layer" else None

        def accumulate(v, pad, idx):
            nonlocal sq, mean
            # A sum is not idempotent, so unlike the max above this one does
            # have to mask the overlap. What it must not do is mask `v` -- see
            # visit() -- so the mask lands on a tile derived from it. The two
            # kinds differ in which tile that is, and it is worth the branch:
            # each ends up holding exactly one temporary.
            if kind == "layer" and pad is not None:
                # LayerNorm needs the plain sum as well, so it needs a
                # maskable copy of the held tile: mask it, sum it, then square
                # it in place and sum that.
                m_t = _mask(v.copy(), pad, 0.0)
                mean += m_t.sum(axis=1)
                m_t *= m_t
                sq += m_t.sum(axis=1)
            else:
                # No copy. RMSNorm never needed one -- the multiply allocates
                # the tile it writes to, so the mask lands on a temporary for
                # free, and masking after squaring is the same sum because 0
                # squared is 0. LayerNorm gets here too when EXACT says there
                # is nothing to mask, and then its plain sum can read the held
                # tile directly; that copy is the whole cost of the `else`
                # branch not being taken, 32 VGPRs and 32 moves per tile.
                if kind == "layer":
                    mean += v.sum(axis=1)
                sq += _mask(v * v, pad, 0.0).sum(axis=1)

        visit(accumulate)
        sq = combine(sq, "add")
        if kind == "layer":
            mean = combine(mean, "add")

        # E[x^2], and for LayerNorm var = E[x^2] - E[x]^2.
        #
        # That identity cancels: on a row with mean 100 and variance 1 the two
        # terms agree to four digits and fp32 keeps about seven, so the
        # variance carries ~1e-3 of relative error. torch uses Welford and does
        # not. It is accepted here because the alternative is a third pass over
        # the row (mean, then deviations) for an input distribution -- a
        # normalized activation with a large offset -- that a LayerNorm is
        # placed to prevent. If a model ever needs it, the fix is the third
        # pass, not a wider accumulator.
        sq /= rowlen
        if kind == "layer":
            mean /= rowlen
            sq -= mean * mean
        sq += EPS
        scale = sq.rsqrt()

        def write(v, pad, idx):
            if kind == "layer":
                v -= mean
            v *= scale
            put(idx, v.to(dtype))

        visit(write)

    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     defines={"HK_RT_HALF_SPLIT": split} if split else None)


def _rms_weighted_kernel(name: str, dtype, warps: int, residual: bool,
                         split: int = 0):
    """The RMSNorm a model runs: y = x * rsqrt(mean(x^2) + eps) * w, and with
    `residual` vLLM's fused_add_rms_norm -- x + r is what is normalised, and
    it is written back to `r_out` -- one read of each operand.

    `split` is HipKittens' HK_RT_HALF_SPLIT for this kernel: see
    _HALF_SPLIT for which plans take it.

    `_row_reduce_kernel`'s schedule. With FOLD the caller views the row as
    FOLD chunks -- one workgroup a row, a tile's rows its chunks -- and the
    weight the same way, so a weight tile loaded at the data tile's
    coordinates is the weight of every element in it.

    In place -- and the fused add always is -- no warp may read an element
    after any warp has written it. The persistent variant reads every tile
    before the cross-warp sum's barrier. The streaming one re-reads in its
    write pass, and that is safe only for a block no other warp covers: an
    inexact plan's backed-up last block overlaps the block before it, and an
    over-provisioned warp is clamped onto it, so a re-read there could see a
    neighbour's output and add the residual a second time (3x4100 came back
    with its last 64 columns several times too large). So the blocks that
    overlap are all read before any of them is written; see visit_writing().
    """
    @hk.func
    def core(x, w, o, r, r_out, ROWS, COLS, WARPS, TPW, FOLD, EXACT, EPS):
        t = _ftile(ROWS, COLS, FOLD)
        vt = hk.col_vec(t)
        rv = hk.row_vec(t)
        ncols = x.cols
        nblk = hk.cdiv(ncols, COLS)
        rowlen = ncols * FOLD if FOLD != 1 else ncols
        # Rows bounded rather than backed up: a decode step normalises one
        # row per request, fewer than a tile's sixteen, and a row past the
        # tensor reads zero and is never written.
        row_lo = hk.arch.block_idx.y * ROWS

        def cb_of(step):
            return step * WARPS + hk.arch.warp_id() if WARPS > 1 else step

        # A folded weight tile is loaded with its data tile, so its latency
        # hides under the reduction's instead of following it -- unless the
        # warp holds several tiles, where the extra tiles would spill.
        hold_w = FOLD != 1 and TPW <= 1

        # Every load and store is bounded, the weight's too though its
        # sixteen rows always fill the tile: the bounded forms are the
        # straight-line buffer accesses, widening and narrowing in registers.
        # Loading 16-bit tiles and copy()ing instead moves every element
        # across the wave halves (tools/hk-bench/norm_stages.py, one 1x4096
        # row: 0.86 us of a 3.5 us kernel).
        def weight(widx):
            return w[widx].load(t, bounded=True) if FOLD != 1 else w[widx].load(rv)

        def load(col_lo, pad=None):
            idx = hk.elem_coord(0, 0, row_lo, col_lo)
            v = x[idx].load(t, bounded=True)
            if r is not None:
                v += r[idx].load(t, bounded=True)
            widx = hk.elem_coord(0, 0, 0, col_lo)
            return v, pad, idx, weight(widx) if hold_w else widx

        def tile(cb):
            return load(*_block_start(cb, COLS, ncols, EXACT))

        held = [tile(cb_of(s)) for s in range(TPW)] if TPW else None
        steps = None if TPW else hk.cdiv(nblk, WARPS)

        def visit(fn):
            if TPW:
                for rec in held:
                    fn(*rec)
            else:
                for s in range(steps):
                    fn(*tile(cb_of(s)))

        def visit_writing(fn):
            """visit() for the write pass of a streamed, inexact plan.

            The blocks are aligned to the row's end instead, so the one that
            is clamped is block 0, onto [0, COLS), and the one it overlaps is
            block 1: both in the first step. The first step is read, the reads
            retired and a barrier passed before any of it is written; after
            it, each warp takes only blocks that exist, all whole, so nothing
            it reads is anyone else's to write. A one-warp workgroup reads
            blocks 0 and 1 together, in a scope of their own so that the
            pair's registers are not held across the loop.

            Aligned to the start, the overlapping pair is split across two
            steps whenever the block count is one more than a multiple of
            WARPS. Holding the last step's tile across the loop instead cost
            the fused add two waves of occupancy on gfx1100, three on gfx950,
            and spilled its folded form."""
            def at_end(cb):
                return load(hk.min(hk.max(ncols - (nblk - cb) * COLS, 0), ncols - COLS))

            if WARPS > 1:
                first = at_end(cb_of(0))
                hk.arch.vm_wait(0)
                hk.arch.barrier()
                fn(*first)
            else:
                with hk.scope():
                    first = [at_end(0), at_end(1)]
                    hk.arch.vm_wait(0)
                    for rec in first:
                        fn(*rec)
            for cb in range(cb_of(0) + max(WARPS, 2), nblk, WARPS):
                fn(*at_end(cb))

        sq = hk.zeros(vt)

        def accumulate(v, pad, idx, g):
            nonlocal sq
            sq += _mask(v * v, pad, 0.0).sum(axis=1)

        visit(accumulate)
        sq = cross_warp(sq, "add", WARPS)
        if FOLD != 1:
            sq = fold_rows(sq, "add", WARPS)
        sq /= rowlen
        sq += EPS
        scale = sq.rsqrt()

        def write(v, pad, idx, g):
            if r_out is not None:
                r_out[idx].store(v, bounded=True)
            v *= scale
            v *= g if hold_w else weight(g)
            o[idx].store(v, bounded=True)

        (visit if TPW or EXACT else visit_writing)(write)

    if residual:
        def body(x: hk.Tensor[dtype], r: hk.Tensor[dtype], w: hk.Tensor[dtype],
                 o: hk.Tensor[dtype], r_out: hk.Tensor[dtype], *, ROWS=ROWS,
                 COLS=COLS, WARPS=warps, TPW=0, FOLD=1, EXACT=0, EPS=1e-6):
            core(x, w, o, r, r_out, ROWS, COLS, WARPS, TPW, FOLD, EXACT, EPS)
    else:
        def body(x: hk.Tensor[dtype], w: hk.Tensor[dtype], o: hk.Tensor[dtype],
                 *, ROWS=ROWS, COLS=COLS, WARPS=warps, TPW=0, FOLD=1, EXACT=0,
                 EPS=1e-6):
            core(x, w, o, None, None, ROWS, COLS, WARPS, TPW, FOLD, EXACT, EPS)

    body.__name__ = name
    # Ties to even for the residual: a sum of two bf16s is exact in fp32, so
    # its tie bit is set about one time in ten (measured: 10.7% of a 300x4096
    # add), not "essentially never" as HipKittens' default assumes, and ties
    # away from zero would grow the residual stream's magnitude a little at
    # every layer. Rounded like torch and vLLM, it matches them bit for bit.
    defines = {"HK_BF16_ROUND": 2} if residual else {}
    if split:
        defines["HK_RT_HALF_SPLIT"] = split
    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     defines=defines or None)


_DTYPES = {"bf16": bf16, "fp16": fp16, "fp32": fp32}

#: HipKittens' HK_RT_HALF_SPLIT for the weighted norms' folded plans. A 16-bit
#: row tile holds each row twice, in lanes l and l + 16, and the packed fp32 ->
#: 16-bit store completes the same 32-byte run in both lanes, so by default
#: every bounded load and store moves each run twice. 3 has each wave half read
#: its own 16 bytes, take the other 16 from its partner over v_permlanex16,
#: and write only its own. Measured (tools/hk-bench/norm_layout.py, one
#: process, output bit-identical to the mirrored kernel, us):
#:
#:                                   plan            mirrored   split
#:     rms_norm 2048x4096            w4 tpw1 fold16     27.17   24.72
#:     fused_add_rms_norm 2048x4096  w4 tpw1 fold16     46.12   40.15
#:     rms_norm 1x4096 / 32x4096     w4 tpw1 fold16  3.84/3.91  3.67/3.77
#:     rms_norm 8192x4096            w16 tpw4           193.1   197.9
#:     fused_add_rms_norm 8192x4096  w16 tpw4           349.0   370.3
#:
#: So the folded one-tile plans take it and the unfolded hold does not: with
#: four tiles a warp the exchange's VALU work costs more than the halved
#: memory traffic returns. Only 16-bit globals have a run to split; fp16 is
#: the same layout and code path as the bf16 that was measured.
_HALF_SPLIT = 3
_HALF_SPLIT_DTYPES = ("bf16", "fp16")

#: Softmax's folded one-tile plan, the same way, with one more step: its
#: loads and stores were the unbounded ones, which have no split. Bounded --
#: the buffer accesses the weighted norms use, and every tile here is backed
#: up into bounds, so the same elements -- and with the store split
#: (norm_layout.py, us, bit-identical):
#:
#:                          plan            shipped  bounded  bounded+split 2
#:     softmax 2048x4096    w4 tpw1 fold16    29.99    23.69    23.22
#:     softmax 8192x4096    w16 tpw4          185.99   197.13   197.06
#:
#: The load split (3) measured 23.89 at 2048, so softmax takes the stores
#: only; the unfolded plan stays as it was.
_SOFTMAX_SPLIT = 2

#: How many warps a workgroup may have. `plan` only ever returns one of these,
#: and it can return any of them: `_split` hands out one warp per column block
#: up to MAX_WARPS, so a row of five blocks asks for five warps. See _split for
#: what restricting this to powers of two cost.
WARP_COUNTS = tuple(range(1, MAX_WARPS + 1))

#: (kind, dtype suffix, warps) -> Kernel. Nothing is compiled until called, and
#: TPW/FOLD ride in as constexpr arguments so that one entry here still covers
#: every column count.
KERNELS = {}
for _s, _dt in _DTYPES.items():
    for _k in ("rms", "layer", "softmax"):
        for _w in WARP_COUNTS:
            _n = f"{_k}_{_s}_w{_w}"
            KERNELS[_n] = _row_reduce_kernel(_n, _dt, _k, _w)
    if _s in _HALF_SPLIT_DTYPES:
        for _w in WARP_COUNTS:
            _n = f"softmax_{_s}_w{_w}_hb"
            KERNELS[_n] = _row_reduce_kernel(_n, _dt, "softmax", _w, bounded=True,
                                             split=_SOFTMAX_SPLIT)
    for _k, _res in (("rmsw", False), ("addrmsw", True)):
        for _w in WARP_COUNTS:
            _n = f"{_k}_{_s}_w{_w}"
            KERNELS[_n] = _rms_weighted_kernel(_n, _dt, _w, _res)
            if _s in _HALF_SPLIT_DTYPES:
                KERNELS[_n + "_h"] = _rms_weighted_kernel(_n + "_h", _dt, _w, _res,
                                                         split=_HALF_SPLIT)
del _s, _dt, _k, _w, _n, _res


# ---------------------------------------------------------------- strips
#
# Everything above works in the register layouts WMMA gives: a 16-row tile
# whose 16-bit form holds each row twice and whose fp32 form gives a lane
# alternate columns. A row kernel wants neither. The kernels below hold the
# row in a strip (hk.RS, kittens::rs): each lane owns runs of RUN contiguous
# elements, one 16-byte access each, nothing replicated -- Triton's blocked
# layout, and FlyDSL's. One workgroup a row; warp w holds columns
# [w*CW, (w+1)*CW), CW = CHUNKS*32*RUN, read once.
#
# What that bought, at 4096 columns, against the fastest of Triton and FlyDSL
# each tuned, Inductor and torch (tools/hk-bench/scoreboard.py, us; the
# tile column is this file's tile kernels, the strip column what hk.ops now
# runs):
#
#                         rows 1        32            2048          8192
#     rms_norm   tile     5.1           3.9           24.8          190.1
#                strip    2.4           2.3           18.4          169.9
#                best     2.5           2.5           19.0          174.1
#     fused add  tile     6.0           4.5           40.1          347.8
#                strip    2.5-3.2       2.5-3.0       36.2          339.0
#                best     2.6           2.8           35.8          328.0
#     softmax    tile     --            --            23.1          184.6
#                strip    --            --            18.7          168.6
#                best     --            --            19.1          172.5
#
# Most of the tile kernels' one-row time was not the tile path at all; see
# kittens::touch_kernargs. At 8192 rows every arm sits at the HBM roof.

#: Elements of one row a lane may hold. 64 is 64 VGPRs of data, and a
#: weighted norm holds as many again of weight; past it the row is wider than
#: 16 warps x 64 x 32 = 32768 elements and goes to the tile kernels' streaming
#: plan.
STRIP_MAX_PER_LANE = 64
#: On wave64 the kinds that hold only the row may hold twice that where a
#: plan asks for it: a row of 8192 in one wave, 128 a lane, as FlyDSL's
#: tuned softmax does.
STRIP_MAX_PER_LANE_WAVE64 = {"rms": 128, "layer": 128, "softmax": 128}

#: Elements of a row a lane aims to hold: two 16-byte runs of bf16, one for
#: the in-place fused add. Fewer a lane is more warps on the row. Measured
#: across 1024-8192 columns and 1-8192 rows (tools/hk-bench/strip_plan.py,
#: us), two runs win or tie for the read-only kernels nearly everywhere --
#:
#:                          8/lane    16/lane    32/lane
#:     rms_norm  2048x2048   10.33      9.62      11.23
#:     rms_norm  8192x1024   19.07     18.32      20.95
#:     softmax   2048x4096   20.23     18.82      20.00
#:     softmax   8192x2048   37.00     36.22      37.49
#:
#: -- with misses of 1-3% at a few shapes (8 a lane at 2048x1024; at
#: 8192x4096 and 5120 the two orders swap between runs). The fused add, which
#: reads and writes both of its row operands, is slowest at two runs of all
#: three in the cache-resident regime, by 5-19%, and takes one:
#:
#:     fused add 2048x1024   10.81     12.04      10.89
#:     fused add 2048x2048   19.14     21.17      19.85
#:     fused add 2048x4096   36.26     37.22      35.60
#:
#: Holding two or four rows a strip (ROWS) won nowhere by more than noise but
#: the fused add at 2048x1024 (7%), and it is not taken.
_STRIP_PER_LANE = {"addrmsw": 8}
_STRIP_PER_LANE_DEFAULT = 16

#: The same target on wave64 (gfx950), every kind: 32 elements, four
#: 16-byte runs of bf16, the warps then chosen to split the row's chunks
#: exactly where a split near the target does (_wave64_split): 4096 bf16
#: columns are 2 warps of 4 chunks, 5120 2 of 5, 8192 4 of 4. Measured on an
#: MI355X (tools/hk-bench/strip_plan.py, us, the five kinds at 8, 16, 32 and
#: 64 a lane): within 5% of the fastest at all 30 of kind x {4096, 5120,
#: 8192} x {2048, 8192} rows and within 1% at 23, where gfx11's 16 (8 for
#: the fused add) was 7-16% behind at 2048x8192, and 32 a lane split by
#: ceiling (5120's ten chunks as three warps of four) 8-30% at 2048x5120:
#:
#:                          16/lane    32/lane      64/lane
#:     rmsnorm   2048x8192   11.17     10.32 w4       9.99
#:     softmax   2048x8192   12.10     11.30 w4      11.82
#:     layernorm 2048x5120    7.11      8.93 w3c4     6.86 w2c5  (the rule)
#:     rms_norm  8192x8192   38.32     38.49 w4      38.77
_STRIP_PER_LANE_WAVE64 = 32

#: gfx950's measured per-lane targets by (kind, columns, rows band): the
#: target, and whether a call past the cache keeps the occupancy cap. The
#: bands are "short" (under _STRIP_AT_ROWS rows), "mid" and "long" (from
#: _STRIP_LONG_ROWS). The best target
#: follows the width more than the rows: 4096-5120 columns want more warps a
#: row, 8192 fewer (a row in one or two waves, the cross-warp combine gone
#: or halved); the fused add past the cache wants 8-16 a lane and no cap.
#: Elsewhere the default and the cap. Measured on an MI355X in the
#: scoreboard's own cases, every arm beside, bf16, us (the default, then the
#: target taken, then the fastest other arm):
#:
#:                          default   taken          fastest other
#:     rms_norm  2048x4096    5.73    16:  5.48      5.62 inductor
#:     rms_norm  2048x8192   10.54    64: 10.28     10.97 triton_tuned
#:     rmsnorm   2048x4096    5.81     8:  5.32      5.50 inductor
#:     rmsnorm   2048x8192   10.86   128: 10.01     10.27 triton_tuned
#:     layernorm 2048x8192   11.36   128: 10.47     10.70 triton_tuned
#:     layernorm 8192x4096   20.63   128: 20.07     19.62 triton_tuned
#:     softmax   2048x4096    6.41   128:  6.19      6.15 flydsl_tuned
#:     softmax   8192x4096   21.07   128: 20.38     20.33 flydsl_tuned
#:     softmax     32x4096    3.90   128:  3.87      3.81 flydsl
#:     fused add 8192x8192   77.35    8 uncapped: 75.69   75.53 triton_tuned
_STRIP_PER_LANE_AT = {
    ("softmax", 4096, "short"): (128, True),
    ("rmsw", 4096, "mid"): (16, True), ("rmsw", 5120, "mid"): (16, True),
    ("rmsw", 8192, "mid"): (64, True),
    ("rms", 4096, "mid"): (8, True), ("rms", 5120, "mid"): (16, True),
    ("rms", 8192, "mid"): (128, True),
    ("layer", 5120, "mid"): (128, True), ("layer", 8192, "mid"): (128, True),
    ("softmax", 4096, "mid"): (128, True), ("softmax", 5120, "mid"): (128, True),
    ("rmsw", 4096, "long"): (16, True), ("rmsw", 5120, "long"): (16, True),
    ("rms", 4096, "long"): (8, True), ("rms", 5120, "long"): (16, True),
    ("layer", 4096, "long"): (128, True), ("layer", 5120, "long"): (128, True),
    ("softmax", 4096, "long"): (128, True), ("softmax", 5120, "long"): (128, True),
    ("addrmsw", 4096, "long"): (16, False), ("addrmsw", 5120, "long"): (8, False),
    ("addrmsw", 8192, "long"): (8, False),
}
_STRIP_AT_ROWS, _STRIP_LONG_ROWS = 2048, 8192
_STRIP_PER_LANE_ARCHS = ("gfx950",)


def _wave64_split(chunks: int, per: int, most: int):
    """(warps, chunks a warp) for `chunks` chunks at about `per` a warp: of
    the exact splits (warps dividing chunks, at most MAX_WARPS, at most
    `most` chunks a warp) the one whose chunks a warp are nearest `per`, the
    larger on a tie; a ceiling split where none is."""
    exact = [(abs(chunks // w - per), -(chunks // w), w)
             for w in range(1, MAX_WARPS + 1) if chunks % w == 0 and chunks // w <= most]
    if exact:
        _, neg, w = min(exact)
        return w, -neg
    warps = min(MAX_WARPS, _ceil(chunks, per))
    return warps, _ceil(chunks, warps)

#: Past the Infinity Cache a strip kernel streams HBM, and there it runs
#: faster with fewer rows in flight than its registers allow: LDS reserved
#: so that a WGP (128 KB of LDS) holds two workgroups. 8192x4096, us
#: (scoreboard timing, 50 calls a round; tuned Triton beside):
#:
#:                   no cap   5 a WGP   3 a WGP   2 a WGP   Triton
#:     rms_norm  w8   177.0    177.4     175.8     173.9     173.7
#:     softmax   w8   177.1    178.7     175.5     173.7     174.5
#:     fused add w16  350.1    353.3     348.1     346.5     337.4
#:
#: Inside the cache it is the other way round -- the same cap took a
#: 2048x4096 rms_norm from 18.4 to 22.3 us -- so it applies only when the
#: bytes a call moves exceed the cache. Tuned Triton gets the same effect
#: from its register allocation: its 8192-row fused add, at waves_per_eu=4,
#: allocates 241 VGPRs, one workgroup a WGP.
_STRIP_CACHE_BYTES = 96 << 20
_STRIP_RESERVE = 48 << 10

#: The same cap on gfx950, past its 256 MB cache: 53 KB of a CU's 160, three
#: workgroups a CU. 8192x8192 bf16, us (one process; tuned Triton beside,
#: whose pick there carries a waves_per_eu):
#:
#:                  no cap   32 KB   53 KB   80 KB   Triton
#:     rmsnorm       39.19   38.79   37.83   45.07   37.67
#:     layernorm     39.96   38.95   38.81   54.04   37.64
#:     rms_norm      38.75   38.52   37.77   48.80   38.19
#:     softmax       39.91   39.37   38.88   58.94   50.41
#:     fused add     76.95   76.83   77.53   76.00     --
#:
#: Inside the cache it costs, as on gfx11: 8192x4096 rmsnorm 20.13 uncapped,
#: 22.63 at 53 KB. 8192x8192's in and out are exactly the cache's 256 MiB,
#: so the cap starts at the cache's size rather than past it.
_STRIP_RESERVE_AT = {"gfx1100": (_STRIP_CACHE_BYTES, _STRIP_RESERVE),
                     "gfx950": ((256 << 20) - 1, 53 << 10)}

#: Row operands each kind reads or writes once: the bytes a call moves are
#: this times rows x cols x the element size.
_STRIP_STREAMS = {"addrmsw": 4}


def strip_run(cols: int, nbytes: int) -> int:
    """Elements in a lane's run: a 16-byte access wherever the row's length
    keeps the runs aligned, halved until it does."""
    r = 16 // nbytes
    while r > 1 and cols % r:
        r //= 2
    return r


def _wave() -> int:
    from ..target import current_arch, get_target  # noqa: PLC0415

    return get_target(current_arch()).wave


def strip_plan(cols: int, kind: str, nbytes: int, wave: Optional[int] = None,
               per_lane: Optional[int] = None):
    """(WARPS, CHUNKS, RUN, EXACT) for a strip kernel, or None where the row
    is too wide to hold.

    RUN from strip_run; CHUNKS, the wave*RUN-column chunks one warp holds,
    from the per-lane target; WARPS to cover the row, at most MAX_WARPS, the
    chunks then rebalanced so that no warp holds more than it must. EXACT when
    the warps cover the row with nothing left over: then nothing is bounded.
    `wave` is the target's (32 on gfx11, 64 on CDNA), the current one's if
    not given; `per_lane` overrides wave64's per-lane target."""
    wave = wave or _wave()
    run = strip_run(cols, nbytes)
    span = wave * run
    chunks = _ceil(cols, span)
    if wave == 64:
        most = STRIP_MAX_PER_LANE
        if per_lane and per_lane > most:
            most = STRIP_MAX_PER_LANE_WAVE64.get(kind, most)
        warps, per = _wave64_split(chunks, max(1, (per_lane or _STRIP_PER_LANE_WAVE64) // run),
                                   most // run)
        if per * run > most:
            return None
        return warps, per, run, int(warps * per * span == cols)
    target = _STRIP_PER_LANE.get(kind, _STRIP_PER_LANE_DEFAULT)
    per = max(1, target // run)
    warps = min(MAX_WARPS, _ceil(chunks, per))
    per = _ceil(chunks, warps)
    if per * run > STRIP_MAX_PER_LANE:
        return None
    return warps, per, run, int(warps * per * span == cols)


def _strip_kernel(name: str, dtype, kind: str, warps: int):
    """A row kernel on a strip: one workgroup a row, the row read once.

    `kind`: rms, layer, softmax (unweighted, as _row_reduce_kernel), rmsw
    (times a weight) and addrmsw (vLLM's fused_add_rms_norm: x + r is
    normalised and written back to r_out). Rows are any count, columns any
    count: without EXACT every access is bounded -- a run at or past the row's
    end reads zero and is not written -- and the reductions that a zero would
    move are masked.
    """
    @hk.func
    def core(x, w, o, r, r_out, ROWS, CHUNKS, RUN, EXACT, RESERVE, EPS):
        if RESERVE:
            hk.lds.reserve(RESERVE)
        cw = CHUNKS * hk.arch.target().wave * RUN
        S = hk.RS(fp32, ROWS, cw, RUN)
        row = hk.arch.block_idx.x * ROWS if ROWS > 1 else hk.arch.block_idx.x
        c0 = hk.arch.warp_id() * cw if warps > 1 else 0
        bounded = not EXACT
        # The weight first: its loads then issue beside the row's rather than
        # after the reduction has waited for them.
        g = w.at[0, c0].load(hk.row_vec(S), bounded=bounded) if w is not None else None
        v = x.at[row, c0].load(S, bounded=bounded)
        if r is not None:
            v += r.at[row, c0].load(S, bounded=bounded)
            # Before the combine's barrier, though gfx11's s_barrier then
            # waits for it (s_waitcnt_vscnt 0): issued after the barrier it
            # lands on the kernel's tail, which measured 0.1-0.2 us slower at
            # 1-32 rows and no faster at 2048 and 8192.
            r_out.at[row, c0].store(v, bounded=bounded)
        n = x.cols
        # Columns of this warp's strip inside the row; past them the bounded
        # load left zeros.
        tail = None if EXACT else n - c0
        if kind == "softmax":
            if tail is not None:
                v.right_fill_(tail, _NEG_INF)
            m = cross_warp(v.max(axis=1), "max", warps)
            v -= m
            v.exp_()
            s = cross_warp(v.sum(axis=1), "add", warps)
            inv = hk.full(hk.col_vec(S), 1.0)
            inv /= s
            v *= inv
        elif kind == "layer":
            # Two passes over registers, not E[x^2] - E[x]^2: the row is
            # held, so the deviations cost a reduction and not a pass over
            # memory, and they do not cancel.
            mean = cross_warp(v.sum(axis=1), "add", warps)
            mean /= n
            v -= mean
            if tail is not None:
                v.right_fill_(tail, 0.0)
            var = cross_warp((v * v).sum(axis=1), "add", warps)
            var /= n
            var += EPS
            v *= var.rsqrt()
        else:
            sq = cross_warp((v * v).sum(axis=1), "add", warps)
            sq /= n
            sq += EPS
            v *= sq.rsqrt()
            if g is not None:
                v *= g
        o.at[row, c0].store(v, bounded=bounded)

    if kind == "addrmsw":
        def body(x: hk.Tensor[dtype], r: hk.Tensor[dtype], w: hk.Tensor[dtype],
                 o: hk.Tensor[dtype], r_out: hk.Tensor[dtype], *, ROWS=1, CHUNKS=1,
                 RUN=8, EXACT=1, RESERVE=0, EPS=1e-6):
            core(x, w, o, r, r_out, ROWS, CHUNKS, RUN, EXACT, RESERVE, EPS)
    elif kind == "rmsw":
        def body(x: hk.Tensor[dtype], w: hk.Tensor[dtype], o: hk.Tensor[dtype], *,
                 ROWS=1, CHUNKS=1, RUN=8, EXACT=1, RESERVE=0, EPS=1e-6):
            core(x, w, o, None, None, ROWS, CHUNKS, RUN, EXACT, RESERVE, EPS)
    else:
        def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype], *, ROWS=1, CHUNKS=1, RUN=8,
                 EXACT=1, RESERVE=0, EPS=1e-6):
            core(x, None, o, None, None, ROWS, CHUNKS, RUN, EXACT, RESERVE, EPS)

    body.__name__ = name
    # Ties to even for the residual, as _rms_weighted_kernel says why.
    defines = {"HK_BF16_ROUND": 2} if kind == "addrmsw" else None
    return hk.kernel(body, warps=warps, grid=_strip_grid, name=name, defines=defines)


def _strip_grid(p):
    return (hk.cdiv(p.x.rows, p.ROWS), 1, 1)


#: (kind, dtype suffix, warps) -> strip Kernel; CHUNKS, RUN and EXACT are
#: constexprs, so one entry covers every width.
STRIP_KERNELS = {}
for _s, _dt in _DTYPES.items():
    for _k in ("rms", "layer", "softmax", "rmsw", "addrmsw"):
        for _w in WARP_COUNTS:
            _n = f"strip_{_k}_{_s}_w{_w}"
            STRIP_KERNELS[_n] = _strip_kernel(_n, _dt, _k, _w)
del _s, _dt, _k, _w, _n


def _strips_here() -> bool:
    """Strips are include/rdna3's and hk's wave64 port's (Target.strips); on
    any other target the tile kernels run."""
    from ..target import current_arch, get_target  # noqa: PLC0415

    return get_target(current_arch()).strips


_RESERVE_HERE: list = []


def _reserve_here():
    """(cache bytes, LDS reserve) of this target's occupancy cap, or None."""
    if not _RESERVE_HERE:
        from ..target import current_arch  # noqa: PLC0415

        _RESERVE_HERE.append(_STRIP_RESERVE_AT.get(current_arch()))
    return _RESERVE_HERE[0]


_PER_LANE_HERE: list = []


def _per_lane_here():
    """_STRIP_PER_LANE_AT where it was measured, else nothing."""
    if not _PER_LANE_HERE:
        from ..target import current_arch  # noqa: PLC0415

        _PER_LANE_HERE.append(_STRIP_PER_LANE_AT if current_arch() in _STRIP_PER_LANE_ARCHS
                              else {})
    return _PER_LANE_HERE[0]


def _resolve_strip(kind: str, dtype, cols: int, rows: int):
    """(kernel, CHUNKS, RUN, EXACT, RESERVE), or None for a row too wide for a
    strip. RESERVE caps the workgroups a WGP (a CU on gfx950) holds when the
    call streams HBM: see _STRIP_RESERVE and _STRIP_RESERVE_AT."""
    nbytes = getattr(dtype, "itemsize", 2)
    cap = _reserve_here()
    streams = cap is not None and rows * cols * nbytes * _STRIP_STREAMS.get(kind, 2) > cap[0]
    at, capped = None, True
    if nbytes == 2:
        band = ("short" if rows < _STRIP_AT_ROWS else
                "long" if rows >= _STRIP_LONG_ROWS else "mid")
        at, capped = _per_lane_here().get((kind, cols, band), (None, True))
    streams = streams and capped
    key = ("strip", kind, dtype, cols, streams, at)
    got = _PLAN_CACHE.get(key, False)
    if got is False:
        suffix = _TORCH_DTYPES.get(str(dtype))
        if suffix is None:
            raise TypeError(f"hk norms handle bf16/fp16/fp32, not {dtype}")
        p = strip_plan(cols, kind, _DTYPES[suffix].bytes, per_lane=at) if _strips_here() else None
        got = None
        if p is not None:
            warps, chunks, run, exact = p
            got = (STRIP_KERNELS[f"strip_{kind}_{suffix}_w{warps}"], chunks, run, exact,
                   cap[1] if streams else 0)
        _PLAN_CACHE[key] = got
    return got


# ---------------------------------------------------------------- wrappers

_TORCH_DTYPES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


def _suffix_of(t) -> str:
    name = _TORCH_DTYPES.get(str(t.dtype))
    if name is None:
        raise TypeError(f"hk norms handle bf16/fp16/fp32, not {t.dtype}")
    return name


def _shape_of(t):
    """(rows, cols) of the normalization `t` describes."""
    if not t.is_contiguous():
        raise ValueError("hk kernels take contiguous tensors; call .contiguous()")
    cols = t.shape[-1]
    rows = t.numel() // cols if cols else 0
    if cols < COLS or rows < ROWS:
        raise ValueError(
            f"shape {tuple(t.shape)} gives a {rows}x{cols} normalization and hk's "
            f"norm kernels need at least {ROWS}x{COLS}. The tail handling backs "
            f"the block up until it is in bounds, which needs a full tile to "
            f"exist; it does not need the shape to divide."
        )
    return rows, cols


def _view(t, rows: int, cols: int, fold: int):
    """The 2D view the kernel reads.

    With fold != 1 this is the reshape that puts one row per workgroup: row r
    of the original becomes rows r*fold .. r*fold+fold-1 of the view, which is
    exactly the `fold` consecutive tile rows that fold_rows combines. It is a
    view of the same contiguous memory, so it costs nothing.

    Nothing in bytes. `Tensor.view` is about a microsecond of Python and these
    kernels run for sixty, and the common case -- a 2D tensor, FOLD 1 -- asks
    for the shape it already has: `rows` was computed as `numel // cols` and
    `cols` as `shape[-1]`, so for `t.dim() == 2` the requested shape is `t`'s
    own by construction. Hand back `t`.
    """
    if fold == 1 and t.dim() == 2:
        return t
    return t.view(rows * fold, cols // fold)


#: (kind, torch dtype, cols) -> (kernel, TPW, FOLD, EXACT).
#:
#: Everything in that tuple is a function of the three key parts and nothing
#: else, and computing it costs a `plan` call, an `_exact` call, an f-string,
#: a `str(dtype)` and two dict probes. That is small, and it is not small
#: compared to what it is in front of: a 4096x4096 rmsnorm is sixty
#: microseconds of GPU and the bare pybind launcher under all of this is
#: 5.3 us. So each piece of the Python side that can be answered once per
#: (op, dtype, width) rather than once per call is worth removing. See
#: Kernel.__call__ for the other half.
#:
#: The key holds the `torch.dtype` object rather than hk's suffix for it,
#: because deriving the suffix is `str(dtype)` plus a lookup and a dtype is
#: already hashable. The suffix lookup then runs only on a miss -- including
#: its TypeError, which is raised before anything is cached, so an unsupported
#: dtype raises on every call and not just the first. `_suffix_of` stays for
#: the benches, which hold a tensor and want the name.
_PLAN_CACHE: dict = {}


def _resolve(kind: str, dtype, cols: int, rows: int):
    short = rows < _FOLD_ROWS_BELOW
    key = (kind, dtype, cols, short)
    got = _PLAN_CACHE.get(key)
    if got is None:
        suffix = _TORCH_DTYPES.get(str(dtype))
        if suffix is None:
            raise TypeError(f"hk norms handle bf16/fp16/fp32, not {dtype}")
        warps, tpw, fold = plan(cols, kind, rows if short else None)
        hb = (kind == "softmax" and fold != 1 and tpw == 1
              and suffix in _HALF_SPLIT_DTYPES)
        got = _PLAN_CACHE[key] = (
            KERNELS[f"{kind}_{suffix}_w{warps}" + ("_hb" if hb else "")], tpw, fold,
            int(_exact(cols // fold, warps, tpw)),
        )
    return got


#: "strip" runs the norms on the strip kernels wherever a row fits one;
#: "tile" keeps them on the tile kernels above, for measuring the one against
#: the other. Read once; a tool may assign it.
LAYOUT = __import__("os").environ.get("HK_NORM_LAYOUT", "strip")


def _flat(t, rows: int, cols: int):
    return t if t.dim() == 2 else t.view(rows, cols)


def _overlaps(a, b) -> bool:
    """Do two contiguous tensors share any bytes?"""
    pa, pb = a.data_ptr(), b.data_ptr()
    return pa < pb + b.numel() * b.element_size() and pb < pa + a.numel() * a.element_size()


def _run_strip(kind: str, x, out, eps: float) -> bool:
    """Run `kind` on a strip kernel if one holds the row; False if not."""
    if not x.is_contiguous():
        raise ValueError("hk kernels take contiguous tensors; call .contiguous()")
    cols = x.shape[-1] if x.dim() else 0
    rows = x.numel() // cols if cols else 0
    if not rows:
        return True
    got = _resolve_strip(kind, x.dtype, cols, rows)
    if got is None:
        return False
    k, chunks, run, exact, reserve = got
    xv, ov = _flat(x, rows, cols), _flat(out, rows, cols)
    if kind == "softmax":
        k(xv, ov, CHUNKS=chunks, RUN=run, EXACT=exact, RESERVE=reserve)
    else:
        k(xv, ov, CHUNKS=chunks, RUN=run, EXACT=exact, RESERVE=reserve, EPS=eps)
    return True


def _run(kind: str, x, out, eps: float):
    if out is None:
        # Imported here and not at the top because `hk` traces, builds and
        # compiles without torch -- tests/hk/ir and tests/hk/codegen run on a
        # machine with no torch and no GPU. Inside the branch and not at the
        # top of the function because `import` is a sys.modules probe and a
        # store every call, and the caller that passes `out` does not need it.
        import torch  # noqa: PLC0415
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError(
            f"{kind}: out is {tuple(out.shape)}/{out.dtype}, "
            f"expected {tuple(x.shape)}/{x.dtype}"
        )
    if LAYOUT == "strip" and _run_strip(kind, x, out, eps):
        return out
    return _run_tile(kind, x, out, eps)


def _run_tile(kind: str, x, out, eps: float):
    rows, cols = _shape_of(x)
    k, tpw, fold, exact = _resolve(kind, x.dtype, cols, rows)
    if (not (tpw or exact) or (fold == 1 and rows % ROWS)) and _overlaps(x, out):
        # Blocks backed up into bounds can be covered twice: by two
        # workgroups (a ragged last row block) or by two warps (a streamed,
        # inexact last column block), and in place one of them may re-read
        # what the other has already written -- a row normalized twice,
        # which softmax is not idempotent under. So out of place and copied
        # back. The fused add, in place by contract, orders its reads and
        # writes instead (_rms_weighted_kernel).
        import torch  # noqa: PLC0415
        return out.copy_(_run_tile(kind, x, torch.empty_like(x), eps))
    # The constexprs go through the call rather than `specialize(...)`: the
    # call path is keyed on them and hits the trace/build/module caches, where
    # a fresh specialization would re-trace and re-resolve the module on every
    # launch. See Kernel.specialize.
    #
    # Written as two calls rather than one with a `**extra` dict because the
    # dict is the expensive part -- Kernel.__call__ keys its launcher cache on
    # the kwargs tuple, and spelling the kwargs out lets CPython build that
    # tuple directly instead of building a dict for this frame to unpack and
    # the callee to rebuild.
    xv, ov = _view(x, rows, cols, fold), _view(out, rows, cols, fold)
    if kind == "softmax":
        k(xv, ov, TPW=tpw, FOLD=fold, EXACT=exact)
    else:
        k(xv, ov, TPW=tpw, FOLD=fold, EXACT=exact, EPS=eps)
    return out


def rmsnorm(x, eps: float = 1e-6, out=None):
    """x * rsqrt(mean(x^2) + eps), over the last axis.

    No learned weight: a weight multiply is elementwise and belongs in whatever
    fuses next, and adding it here would mean a second global and a broadcast
    that the caller may not want.
    """
    return _run("rms", x, out, eps)


def layernorm(x, eps: float = 1e-5, out=None):
    """(x - mean) * rsqrt(var + eps), over the last axis, unweighted.

    `var` is the biased estimate (divide by N), which is what torch's
    F.layer_norm uses.
    """
    return _run("layer", x, out, eps)


def softmax(x, out=None):
    """Softmax over the last axis, max-shifted."""
    return _run("softmax", x, out, 0.0)


#: Rows from which the weighted norms stop folding. Folded is one workgroup a
#: row, which is what keeps a decode step's handful of rows from being one
#: workgroup with fifteen of its sixteen rows empty; unfolded is a sixteenth
#: of the workgroups holding four times the tiles, which wins once there are
#: enough of them to fill the machine. Measured (scoreboard.py, 4096 columns,
#: ms, folded / unfolded):
#:
#:                    rows 1          32              2048            8192
#:     rms_norm       0.0052/0.0087   0.0049/0.0096   0.0277/0.0333   0.2033/0.1964
#:     fused add      0.0069/0.0210   0.0055/0.0215   0.0505/0.0713   0.4030/0.3653
_WEIGHTED_UNFOLD_ROWS = 4096

#: The residual variant's own TPW cap, unfolded and with its last block backed
#: up: four tiles a warp spill there (measured at 3584 columns), where exact
#: it is 207 VGPRs.
_ADD_TPW_INEXACT = 3


def _tpw_cap(kind: str, fold: int, exact: bool) -> int:
    """The most tiles a warp of `kind` may hold; plan()s never ask for more."""
    if kind == "addrmsw" and fold == 1 and not exact:
        return _ADD_TPW_INEXACT
    return MAX_TPW


def _resolve_weighted(kind: str, dtype, cols: int, rows: int):
    """(kernel, TPW, FOLD, EXACT). HK_NORM_FOLD=on/off overrides the row
    rule, for measuring the one plan against the other."""
    import os  # noqa: PLC0415

    mode = os.environ.get("HK_NORM_FOLD", "auto")
    want = mode == "on" or (mode == "auto" and rows < _WEIGHTED_UNFOLD_ROWS)
    key = (kind, dtype, cols, want)
    got = _PLAN_CACHE.get(key)
    if got is None:
        suffix = _TORCH_DTYPES.get(str(dtype))
        if suffix is None:
            raise TypeError(f"hk norms handle bf16/fp16/fp32, not {dtype}")
        fold = ROWS if want and cols % ROWS == 0 and cols // ROWS >= COLS else 1
        warps, tpw = (_fold_split if fold != 1 else _split)(_ceil(cols // fold, COLS))
        exact = _exact(cols // fold, warps, tpw)
        if tpw > _tpw_cap(kind, fold, exact):
            tpw = 0
            exact = _exact(cols // fold, warps, tpw)
        half = fold != 1 and tpw == 1 and suffix in _HALF_SPLIT_DTYPES
        got = _PLAN_CACHE[key] = (
            KERNELS[f"{kind}_{suffix}_w{warps}" + ("_h" if half else "")], tpw,
            fold, int(exact))
    return got


def _rows_cols(x):
    if not x.is_contiguous():
        raise ValueError("hk kernels take contiguous tensors; call .contiguous()")
    cols = x.shape[-1]
    if cols < COLS and LAYOUT != "strip":
        raise ValueError(f"rows of {cols} elements; the tile norms need "
                         f"at least {COLS}")
    return (x.numel() // cols if cols else 0), cols


def _weighted_strip(kind: str, x, rows: int, cols: int):
    if LAYOUT != "strip" or not rows:
        return None
    return _resolve_strip(kind, x.dtype, cols, rows)


def rms_norm(x, weight, eps: float = 1e-6, out=None):
    """x * rsqrt(mean(x^2) + eps) * weight over the last axis -- the RMSNorm
    a model has. Any number of rows, including fewer than a tile's."""
    rows, cols = _rows_cols(x)
    if out is None:
        import torch  # noqa: PLC0415
        out = torch.empty_like(x)
    got = _weighted_strip("rmsw", x, rows, cols)
    if got is not None:
        k, chunks, run, exact, reserve = got
        k(_flat(x, rows, cols), weight.view(1, cols), _flat(out, rows, cols),
          CHUNKS=chunks, RUN=run, EXACT=exact, RESERVE=reserve, EPS=eps)
        return out
    if not rows:
        return out
    k, tpw, fold, exact = _resolve_weighted("rmsw", x.dtype, cols, rows)
    shape = (rows * fold, cols // fold)
    k(x.view(shape), weight.view(fold, cols // fold), out.view(shape),
      TPW=tpw, FOLD=fold, EXACT=exact, EPS=eps)
    return out


def fused_add_rms_norm(x, residual, weight, eps: float = 1e-6):
    """vLLM's fused_add_rms_norm, in place: `residual` becomes x + residual
    and `x` its RMSNorm times `weight`. One read of x, residual and weight,
    one write of each result."""
    rows, cols = _rows_cols(x)
    if residual.shape != x.shape or not residual.is_contiguous():
        raise ValueError(f"residual is {tuple(residual.shape)}, expected a "
                         f"contiguous {tuple(x.shape)}")
    got = _weighted_strip("addrmsw", x, rows, cols)
    if got is not None:
        k, chunks, run, exact, reserve = got
        xv, rv = _flat(x, rows, cols), _flat(residual, rows, cols)
        k(xv, rv, weight.view(1, cols), xv, rv, CHUNKS=chunks, RUN=run,
          EXACT=exact, RESERVE=reserve, EPS=eps)
        return x, residual
    if not rows:
        return x, residual
    k, tpw, fold, exact = _resolve_weighted("addrmsw", x.dtype, cols, rows)
    shape = (rows * fold, cols // fold)
    xv, rv = x.view(shape), residual.view(shape)
    k(xv, rv, weight.view(fold, cols // fold), xv, rv, TPW=tpw, FOLD=fold,
      EXACT=exact, EPS=eps)
    return x, residual


__all__ = [
    "KERNELS", "STRIP_KERNELS", "ROWS", "COLS", "MAX_TPW", "MAX_WARPS",
    "WARP_COUNTS", "plan", "strip_plan", "rmsnorm", "layernorm", "softmax",
    "rms_norm", "fused_add_rms_norm",
]
