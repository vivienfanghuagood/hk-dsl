"""T2 from the docs only: fused residual add + RMSNorm + per-token int8 quant,
one workgroup per row on register strips (hk.RS), the row read once and held:
x + residual -> bf16 r (stored in place into `residual`) -> sum of squares
across warps -> y = r * w * rstd -> absmax across warps -> scale, q = y / scale.

DX log -- what had to be looked up, and where (about 1 h 15 min in all):
- The layout: language.md "Register strips" plus examples/02_rmsnorm.py's
  rmsnorm_strip_kernel (one workgroup a row, warp w owns [w*CW, (w+1)*CW),
  cross_warp for the row sum). That example is the right template; found fast.
- The per-arch chunk (wave*8): language.md "Layout" bullet and
  examples/02_rmsnorm.py (`get_target(current_arch())`), as in t1's strip attempt.
- Rounding of x + residual on gfx950: history/37 ("Stores round bf16 to nearest
  even"); on gfx1100: history/18 and history/38 (see the GAP below).
- How many warps a row: history/27 and history/37 (the shipped plan is "32 elements
  a lane" on wave64); then measured myself, see below.
- Runtime lookups: dir(hk), dir(hk.math), help(hk.RS), help(hk.View) (store_scalar),
  help(hk.Vec), help(hk.as_vec), help(hk.collective.cross_warp), and five error messages.
GAP: int8 output is not documented anywhere a kernel author reads. language.md lists
     `hk.i8` as a dtype and says quantize runs on strips, and nothing else; there is no
     round(), no clamp(), and the semantics of fp32 -> int8 (round? truncate? saturate?)
     are not stated. `y.to(hk.i8)` raises "there is no i8 strip: rs holds bf16, fp16
     and fp32 elements". Guessed from "a strip ... converts its element type on the way"
     that storing the fp32 strip into an `hk.Tensor[hk.i8]` converts; it does, and
     rounds to nearest (checked by the q off-by-one count, the same as torch.compile's).
     Whether it saturates is still unknown (not needed: |y / scale| <= 127). ~15 min.
GAP: one value per row (the scale) out of a strip has no documented spelling.
     language.md mentions only `.load_scalar()`. `scale.at[row, 0].store(s)` raises
     "store of a strip's col_vec: ... stores with store_scalar"; `store_scalar(s)` raises
     "store_scalar needs a uniform vector ... Uniformity comes from
     hk.collective.fold_rows ... sixteen rows" -- the two messages send you to each
     other, and the second talks about 16-row tiles although a one-row strip's col_vec
     is in every lane (language.md says the butterfly "leaves the answer in every lane").
     Worked around with `hk.as_vec(s, S, uniform=True)` found via help(hk.as_vec), whose
     docstring does not say when uniform=True is legitimate. ~15 min.
GAP: (wrong, by omission) `t.to(hk.bf16)` in registers rounds ties AWAY from zero on
     RDNA (gfx1100) and to even on gfx950; stores round to even on both. Nothing in
     language.md says so. bf16(x + residual) is a tie on ~10% of elements, so this
     kernel was bit-exact on gfx950 and wrong (r off by one ulp on 10.7% of elements)
     on gfx1100. The fix, `defines={"HK_BF16_ROUND": 2}`, is only in history/18 and
     history/38; language.md documents `defines` but lists no knob. ~20 min.
GAP: (wrong) help(hk.RS) says the runs start at `j*32*run + l*run` and "cols is a whole
     number of 32*run chunks": true on gfx11 only; on gfx950 the chunk is 64*run
     (language.md has it right).
GAP: Vec has no reflected division: `127.0 / amax` raises "unsupported operand type(s)
     for /: 'float' and 'Vec'"; used `y /= s` (which is also the reference's y/scale).
     `1.0 - t` is likewise missing (help() lists __radd__/__rmul__ but no __rsub__).
GAP: no guidance on warps per row for a fused op. history/37's shipped rule (32 elements
     a lane = 4 chunks a warp) was 25% slower here at 1-256 rows than 2 chunks a warp.
     Swept 1/2/4/5/8 chunks a warp on gfx950 (us, CUDA graph of 50 calls):
       T1-H4096 2.69/2.70/3.44/-/4.65   T2048-H4096 13.0/12.2/12.4/-/12.3
       T8192-H4096 39.7/37.5/39.1/-/39.6   T2048-H8192 25.1/23.4/23.4/-/26.3
     so the plan is 2 chunks a warp (H/1024 warps on gfx950). ~15 min.
GAP: how a 1-D tensor maps to (rows, cols) is ambiguous ("fills the trailing ones");
     passed weight.view(1, H) to be sure, and loaded it as a one-row strip of the same
     type rather than as a row_vec, whose load spelling is not shown anywhere.
GAP: on gfx1100 a kernel called as `k(...)` launches on the null stream (runtime.md "The
     stream", one paragraph about the pybind entry point). The first gfx1100 run of this
     file "passed" and reported device_us of 0.1-2.2 us and host_us above the card's
     bandwidth (147 MB in 128 us on a W7900D): the suite's CUDA graph captured nothing
     and its events did not wait for the kernel, with no error anywhere. getting-started.md
     and language.md never mention it; a docs-only user calling `k(...)` as the first page
     teaches gets silently wrong timings on RDNA. Fixed by launching through
     `hk.torch_op(k)` there (needs a unique `name=` per specialisation, since ops are
     memoised by name, and the constexprs baked in as defaults). ~20 min.
- A note, not a gap: the reference clamps the scale at 1e-10; `Vec.maximum(1e-10)` takes
  a Python float (found by trying), so an all-zero padding row gives q = 0, not NaN.
- Correctness: r bit-exact against the reference on gfx950 (and on gfx1100 with the
  define); q matches the reference's to within one step, as many off by one as torch.compile.
- Measured (device us, docs_only / torch.compile / triton), gfx950: T1 2.65/3.21/2.87,
  T16 2.88/3.58/3.15, T37-H5120 3.55/4.89/4.15, T256 3.30/4.32/3.64, T2048 11.5/15.9/13.1,
  T8192 37.6/48.6/38.1, T4096-H5120 26.9/34.6/38.6, T2048-H8192 22.1/27.4/24.2
  (x TC 1.21-1.38, geomean 1.29). gfx1100 (via hk.torch_op): T1 4.61/5.40/5.21,
  T16 4.69/5.98/5.29, T37 5.23/6.61/6.13, T256 7.59/11.5/10.0, T2048 45.2/51.0/46.5,
  T8192 329/334/329, T4096-H5120 119/215/213, T2048-H8192 73.6/170/169 (x TC 1.01-2.31).
- Seen in passing: the suite's hk_ops baseline (fused_add_rms_norm + quantize) is
  "wrong" on 7 of 8 shapes on gfx950 under the current check (q more than 1.25 steps
  from y/scale).
"""

import functools

import torch

import hk
from hk.target import current_arch

KIND = "docs_only"
LABEL = "from the docs: one row a workgroup on register strips, add+norm+absmax+quant fused"

WAVE = hk.get_target(current_arch()).wave
CHUNK = WAVE * 8                      # one strip chunk: wave lanes x runs of 8


CDNA = current_arch() in ("gfx942", "gfx950")   # whose pybind launch takes torch's stream


@functools.lru_cache(maxsize=None)
def _kernel(cw: int, warps: int, eps: float):
    S = hk.RS(hk.fp32, 1, cw)

    # HK_BF16_ROUND=2: in-register fp32->bf16 rounds ties to even on RDNA too (default there is
    # ties away, which disagrees with torch on ~10% of a residual sum's elements; chapter 18/38)
    @hk.kernel(warps=warps, grid=lambda p: (p.x.rows, 1, 1), defines={"HK_BF16_ROUND": 2},
               name=f"dx_t2_docs_{cw}_{warps}_{abs(hash(eps)) % 10**8}")
    def add_rms_quant(x: hk.Tensor[hk.bf16], res: hk.Tensor[hk.bf16], w: hk.Tensor[hk.bf16],
                      q: hk.Tensor[hk.i8], scale: hk.Tensor[hk.fp32],
                      EPS: hk.Constexpr[float] = eps):
        row, c0 = hk.arch.block_idx.x, hk.arch.warp_id() * cw
        v = x.at[row, c0].load(S)
        v += res.at[row, c0].load(S)
        rb = v.to(hk.bf16)                                   # r = bf16(x + residual)
        res.at[row, c0].store(rb)
        r = rb.to(hk.fp32)
        sq = hk.collective.cross_warp((r * r).sum(axis=1), "add", warps)
        y = r * w.at[0, c0].load(S)
        y *= hk.math.rsqrt(sq / x.cols + EPS)
        amax = hk.collective.cross_warp(y.abs().max(axis=1), "max", warps)
        s = (amax / 127.0).maximum(1e-10)                    # an all-zero (padding) row: q = 0
        if hk.arch.warp_id() == 0:
            scale.at[row, 0].store_scalar(hk.as_vec(s, S, uniform=True))
        y /= s
        q.at[row, c0].store(y)                               # fp32 -> int8 on the way out

    # RDNA's pybind entry launches on the null stream (runtime.md "The stream"): not on
    # torch's stream, not in a CUDA graph. hk.torch_op launches on torch's stream there.
    return add_rms_quant if CDNA else hk.torch_op(add_rms_quant)


def _plan(h: int):
    chunks = h // CHUNK
    # two chunks a warp: swept 1/2/4/5/8 on gfx950 (2 best or within 1% at every shape; 4,
    # chapter 37's "32 elements a lane", was 25% slower at decode sizes)
    for cpw in (2, 1):
        if chunks % cpw == 0:
            return cpw * CHUNK, chunks // cpw
    return CHUNK, chunks


def supports(x, residual, weight, eps):
    if x.shape[-1] % CHUNK:
        return f"H must be a multiple of {CHUNK} (one strip chunk)"
    return True


def solve(x, residual, weight, eps):
    t, h = x.shape
    cw, warps = _plan(h)
    q = torch.empty((t, h), device=x.device, dtype=torch.int8)
    scale = torch.empty((t, 1), device=x.device, dtype=torch.float32)
    _kernel(cw, warps, float(eps))(x, residual, weight.view(1, h), q, scale)
    return q, scale, residual
