"""T3 cross-entropy from the docs only: register strips (hk.RS), one workgroup a
row, an online logsumexp per warp, warps combined with hk.collective.cross_warp;
below 256 rows the row is split over several workgroups and a second one-warp
kernel combines the partials. The target logit is read from the chunk that holds
it, every other column filled with 0, and summed. All math in the DSL; torch only
allocates, and reinterprets the int64 labels as int32 pairs (`view`, no copy).

DX log -- what had to be looked up, and where:
- The shape of a row reduction: examples/03_softmax.py (max_, neg_infty, a vec
  broadcast, cross_warp "max"/"add") and examples/02_rmsnorm.py (a strip a warp).
  The strip layout and "bounded=True reads zero past the last column":
  language.md "Register strips". Strip width a step: chapter 37 (32 elements a
  lane on wave64 = 4 runs), then measured.
- `copy(into=)`, carrying a tile through a runtime loop in place: examples/10.
- That a helper with a runtime `for`/`if` must be `@hk.func`: language.md "@hk.func".
  Returning a tuple of vecs from one works (not said, tried).
- hk.math has `log`: dir(hk.math) (no list in the docs). `hk.full(ty, value)`,
  `hk.as_vec`, `View.store_scalar`: dir(hk) / dir(hk.View) and help().
- wg_arrive for a one-launch split: language.md "Namespaces" + chapter 11.
GAP: load_scalar() reads int32 only (help(): "One int32 element"); language.md
     says only `.load_scalar()`. There is no way to read ONE bf16/fp32 element (the
     target logit at a runtime column). Worked around: load the strip chunk that
     contains the column, `left_fill_`/`right_fill_` everything else to 0, `sum`.
     ~15 min to find and design.
GAP: no int64 dtype, and nothing says how to take int64 indices (torch's default
     for labels / token ids). Worked around with `labels.view(torch.int32)` and the
     low word at column 2*t (little-endian; -100 stays -100). ~5 min.
GAP: writing one value a row from a strip is undocumented and the two error
     messages point at each other: `loss.at[0, row].store(v)` says "a strip's
     col_vec ... stores with store_scalar"; `store_scalar(v)` then says it "needs a
     uniform vector ... uniformity comes from hk.collective.fold_rows" -- even for
     the output of cross_warp or a strip row-reduction, which ARE uniform (a
     butterfly). The way out, `hk.as_vec(v, S, uniform=True)`, is in no doc; found
     with dir(hk) + help(hk.as_vec), and it is an unchecked assertion. ~15 min.
GAP: `right_fill_(idx, value)` / `left_fill_` have no docstring and language.md
     only names them ("fills by column"). Which side is inclusive (>= idx / < idx),
     and whether idx may be a device Int, were guessed from HipKittens' convention
     and confirmed only by the results being right. ~5 min.
GAP: masking a ragged tail for a max / logsumexp is not covered: `bounded=True`
     reads 0, which is right for a sum and wrong for a max or a sum of exp. Used
     `right_fill_(ncols - c, -1e30)` on the last chunk (did not try float('-inf')).
GAP: (wrong) help(hk.RS) says "`cols` is a whole number of 32*run chunks"; on
     gfx950 it is 64*run (language.md has it right: wave*run).
GAP: Vec has exp_ but no log_ (in place); `hk.math.log(v, into=v)` instead. Small.
GAP: a 1-D tensor's coordinates: language.md "fills the trailing ones" reads as if
     a 1-D (T,) tensor were rows; it is cols (`loss.at[0, row]`), learned from
     examples/10 (`mean.at[0, r]`).
GAP: performance guidance for a long-row reduction: none beyond chapter 37's
     "32 elements a lane". On gfx950 warps 1-16 x 1-8 chunks a step were all within
     5.4-6.1 TB/s at T>=2048 (8 warps x 2 chunks picked); prefetching the next chunk
     into a second strip (double buffer) gained nothing. For small T (64 rows on 256
     CUs) the docs offer only wg_arrive (chapter 11, "ships as a capability for
     kernels whose grid really is too small"). On gfx950 it is correct but 8-15x
     SLOWER here (T64: 67-131 us vs 7.8 us for two launches; T512: 198-392 us vs 29
     us), which nothing warns about (chapter 11 measured gfx1100's decode GEMV). Two
     kernels (partials, then a one-warp combine) are what worked: T64 15 -> 7.8 us.
     ~40 min of sweeps.

Result (device us, docs_only vs torch.compile): gfx950 1.98-3.49x (5.5-6.1 TB/s
from T=512 up; 1.5-2.2x the Triton baseline); gfx1100 1.32-3.63x (at the W7900D's
bandwidth from T=2048 up; a tie with Triton, 2.3x it at T64). The plan was tuned
on gfx950 only; the same kernels ran unchanged on gfx1100 (wave 32 -> 256-column
chunks).
"""

import torch

import hk
from hk.target import current_arch

KIND = "docs_only"
LABEL = "from the docs: register strips, online logsumexp, a row split over workgroups when T is small"

WAVE = hk.get_target(current_arch()).wave
CHUNK = WAVE * 8                       # one strip chunk: wave lanes x a bf16 run of 8
NEG = -1.0e30                          # "minus infinity" that exp() and max() survive

S1 = hk.RS(hk.fp32, 1, CHUNK)
CV1 = hk.col_vec(S1)
_kernels = {}


@hk.func
def _row_lse(logits, row, first, step, cw, S, warps):
    """This warp's online (max, sum exp) over chunks first, first+step, ... of the
    row, then combined across the workgroup's warps: (max, sum) uniform vectors."""
    CV = hk.col_vec(S)
    ncols = logits.cols
    m = hk.full(CV, NEG)
    mp = hk.full(CV, NEG)
    acc = hk.zeros(S)
    for i in range(first, hk.cdiv(ncols, cw), step):
        c = i * cw
        x = logits.at[row, c].load(S, bounded=True)
        if c + cw > ncols:
            x.right_fill_(ncols - c, NEG)
        m.copy(into=mp)
        m.max_(x.max(axis=1))
        acc *= (mp - m).exp()
        acc += (x - m).exp()
    mall = hk.collective.cross_warp(m, "max", warps)
    s = hk.zeros(CV)
    s += acc.sum(axis=1)
    s *= (m - mall).exp()
    return mall, hk.collective.cross_warp(s, "add", warps)


def _target(logits, row, lab):
    """logits[row, lab] as a uniform vector (lab >= 0): the 16-byte-aligned chunk
    holding it, every other column filled with 0, summed."""
    cl = (lab // CHUNK) * CHUNK
    t = logits.at[row, cl].load(S1, bounded=True)
    t.right_fill_(lab - cl + 1, 0.0)
    t.left_fill_(lab - cl, 0.0)
    tv = hk.zeros(CV1)
    tv += t.sum(axis=1)
    return tv


@hk.func
def _finish(logits, labels, pm, ps, row, split):
    """The row's `split` partials combined, minus the target: the loss, uniform."""
    lab = labels.at[0, row * 2].load_scalar()       # low word of the int64 label
    res = hk.zeros(CV1)
    if lab >= 0:
        m = pm.at[row, 0].load(S1, bounded=True)
        m.right_fill_(split, NEG)
        s = ps.at[row, 0].load(S1, bounded=True)
        mx = hk.full(CV1, NEG)
        mx.max_(m.max(axis=1))
        s *= (m - mx).exp()
        res += s.sum(axis=1)
        hk.math.log(res, into=res)
        res += mx
        res -= _target(logits, row, lab)
    return hk.as_vec(res, S1, uniform=True)


def _fused(warps, k):
    """One workgroup a row."""
    key = ("fused", warps, k)
    if key in _kernels:
        return _kernels[key]
    cw = k * CHUNK
    S = hk.RS(hk.fp32, 1, cw)
    CV = hk.col_vec(S)

    @hk.kernel(warps=warps, grid=lambda p: (p.logits.rows, 1, 1), name=f"xent_w{warps}_k{k}")
    def xent(logits: hk.Tensor[hk.bf16], labels: hk.Tensor[hk.i32], loss: hk.Tensor[hk.fp32]):
        row, wid = hk.arch.block_idx.x, hk.arch.warp_id()
        mall, sall = _row_lse(logits, row, wid, warps, cw, S, warps)
        lab = labels.at[0, row * 2].load_scalar()       # low word of the int64 label
        res = hk.zeros(CV)
        if lab >= 0:
            res += mall + sall.log()
            res -= _target(logits, row, lab)
        # a 1-row strip's row reductions are butterflies: every lane holds the answer
        loss.at[0, row].store_scalar(hk.as_vec(res, S, uniform=True))

    _kernels[key] = xent
    return xent


def _split(warps, k, split):
    """`split` workgroups a row, each writing its (max, sum) partial; then a
    one-warp kernel a row combines them and subtracts the target."""
    key = ("split", warps, k, split)
    if key in _kernels:
        return _kernels[key]
    cw = k * CHUNK
    S = hk.RS(hk.fp32, 1, cw)

    @hk.kernel(warps=warps, grid=lambda p: (split, p.logits.rows, 1),
               name=f"xent_part_w{warps}_k{k}_s{split}")
    def part(logits: hk.Tensor[hk.bf16], pm: hk.Tensor[hk.fp32], ps: hk.Tensor[hk.fp32]):
        sp, row, wid = hk.arch.block_idx.x, hk.arch.block_idx.y, hk.arch.warp_id()
        mall, sall = _row_lse(logits, row, sp * warps + wid, split * warps, cw, S, warps)
        pm.at[row, sp].store_scalar(hk.as_vec(mall, S, uniform=True))
        ps.at[row, sp].store_scalar(hk.as_vec(sall, S, uniform=True))

    @hk.kernel(warps=1, grid=lambda p: (p.logits.rows, 1, 1), name=f"xent_comb_s{split}")
    def comb(logits: hk.Tensor[hk.bf16], labels: hk.Tensor[hk.i32], pm: hk.Tensor[hk.fp32],
             ps: hk.Tensor[hk.fp32], loss: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        loss.at[0, row].store_scalar(_finish(logits, labels, pm, ps, row, split))

    _kernels[key] = (part, comb)
    return part, comb


def plan(t, v):
    """(warps, chunks a warp's step, workgroups a row). Measured on gfx950 (see the DX
    log): a row a workgroup is at 5.5-6 TB/s from T=512 up, flat across 2-8 warps;
    below ~256 rows the grid is too small for 256 CUs, so the row is split."""
    if t < 256:
        nchunks = -(-v // (2 * CHUNK))
        split = max(1, min(16, -(-512 // t), nchunks // 2))
        if split > 1:
            return 4, 2, split
    return 8, 2, 1


def solve(logits, labels):
    t, v = logits.shape
    loss = torch.empty(t, device=logits.device, dtype=torch.float32)
    lab32 = labels.view(torch.int32)               # (2T,) little-endian: low word first
    warps, k, split = plan(t, v)
    if split == 1:
        _fused(warps, k)(logits, lab32, loss)
        return loss
    part, comb = _split(warps, k, split)
    pm = torch.empty(t, split, device=logits.device, dtype=torch.float32)
    ps = torch.empty_like(pm)
    part(logits, pm, ps)
    comb(logits, lab32, pm, ps, loss)
    return loss
