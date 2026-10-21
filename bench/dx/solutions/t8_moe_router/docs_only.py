"""T8 from the docs only: MoE router (softmax, top-k, renormalise) on register strips.

One warp a token: the row of E logits is a one-row fp32 strip with runs of 1 (lane l
owns columns l, l + wave, ...), read with a bounded load and filled with -inf past E.
The softmax is never formed: the top-k of the probabilities is the top-k of the logits,
and after renormalising, w_i = exp(l_i - l_max) / sum_topk exp(l_j - l_max), the full
softmax denominator cancels. Each of the k rounds is a row max, the lowest column that
holds it (a min over a column-index strip, so ties give distinct ids), and that column
masked to -1e30. The k ids and weights are built into strips (column i = round i) and
stored with one bounded store each, into the (T, k) int32 / fp32 outputs.

DX log -- what had to be looked up, and where (about 1 h in all):
- Strips, bounded loads past the last column, the -inf fill for a max over a ragged
  row: language.md "Register strips" and history/27 ("softmax's max fills the tail with
  -inf first"). The E=8 row is narrower than one wave, so it is a 64-column strip
  (32 on gfx11) of which 8 are real.
- Runtime lookups: dir(hk.arch) (lane_id), help(hk.Tile) (step, right_fill_, min),
  help(hk.Reduction) (.value()), help(hk.as_vec), and the error messages.
GAP: no top-k, argmax, iota/column-index, compare or select is documented. Everything
     is assembled from pieces whose semantics had to be guessed:
     - a column index: `hk.arch.lane_id()` (language.md lists "lane ids" in the arch
       namespace, nothing more) added to a strip -- that a per-lane Int broadcasts into a
       strip is not documented -- plus `right_fill_(j*wave, j*wave)` for each further chunk;
     - `right_fill_(idx, value)`: language.md says only "fills by column (right_fill_,
       left_fill_)"; that it sets the columns >= idx was a guess, confirmed by the output;
     - equality and selection: only `step()` ("1 where the element is positive") exists,
       so v == m is `((v - m) * 1e30 + 1).step()` and "column == idx" is
       `(0.5 - |iota - idx|).step()`. ~25 min for the whole selection scheme.
GAP: (wrong) help(Tile.step) says "Tiles only"; it works on strips.
GAP: no reflected subtraction/division on tiles and vectors (`1.0 - t`, `c / v` raise);
     spelled `-(...) + 0.5` and `iota - hit * 4096.0 + 4096.0`.
GAP: int32 output from a strip is undocumented (strips hold only bf16/fp16/fp32, said
     by an error in t2, not by the docs); storing the fp32 strip of ids into an
     `hk.Tensor[hk.i32]` converts, which was a guess that worked.
GAP: `t.max(axis=1)` returns a Reduction, not a Vec; for `m[:, None]` and `(m - m0).exp()`
     it needs `.value()`, found only via help(); language.md shows reductions only as
     `acc += t.sum(axis=1)` or straight into cross_warp.
GAP: performance of strip reductions on wave64 is not documented. The kernel is
     instruction-bound, not memory-bound: for E=128, k=8 the ISA (extracted by hand from
     Build.so_path with llvm-objcopy + clang-offload-bundler + llvm-objdump; no documented
     way to see a kernel's disassembly) is 1108 instructions a wave, 398 of them s_nop
     (DPP hazards in the butterflies) for 16 cross-lane reductions. Nothing in the docs
     says how to give a tiny row less than a whole wave, which is what would cut that.
GAP: as in t2: on gfx1100 `k(...)` launches on the null stream, so CUDA-graph capture
     and torch's events silently miss it; launched through `hk.torch_op(k)` there.
Performance iterations (gfx950, us, CUDA graph of 50 calls):
- rows a strip 1/2/4 x warps 1/2/4/8: more rows a strip only adds butterflies
  (T4096-E128: 5.93 at 1 row/4 warps, 6.31 at 2 rows, 7.42 at 4); kept 1 row, 4 warps.
- storing each id and weight with `store_scalar` as soon as it is known (no out strips,
  `if row < rows` guards since store_scalar has no bounded form) was slower everywhere
  (T4096-E128 6.99, T16384-E128 18.8 vs 17.3); reverted.
- Result (device us, docs_only / torch.compile / triton), gfx950: T16-E8 1.87/18.3/1.98,
  T4096-E8 2.59/35.9/2.21, T16-E64 2.66/22.2/4.11, T4096-E64 5.17/42.3/4.38,
  T1-E128 2.92/19.1/5.25, T16-E128 2.94/30.1/5.33, T256-E128 3.08/31.2/5.53,
  T4096-E128 5.92/58.0/5.69, T16384-E128 18.3/144/9.93 (x TC 6.6-13.9; vs Triton 0.54-1.81).
  gfx1100 (via hk.torch_op): 4.12/25.0/4.27, 5.61/54.0/4.60, 4.17/25.4/5.32,
  11.1/74.2/7.63, 4.35/26.8/7.56, 4.35/32.0/7.96, 4.49/34.2/7.19, 13.0/120/13.3,
  42.3/380/36.7 (x TC 6.1-9.6; vs Triton 0.69-1.83). Losing to Triton from 4096 tokens.
- Not tried (out of the time box): one reduction a round instead of two, by packing the
  column into the low mantissa bits of the bf16 logit (exact for |logit| >= 2^-70, not
  for exact zeros).
"""

import functools

import torch

import hk
from hk.target import current_arch

KIND = "docs_only"
LABEL = "from the docs: a strip row per token, k rounds of max + argmin-index on the logits"

WAVE = hk.get_target(current_arch()).wave
NEG = float("-inf")


CDNA = current_arch() in ("gfx942", "gfx950")   # whose pybind launch takes torch's stream


@functools.lru_cache(maxsize=None)
def _kernel(cols: int, k: int, warps: int, e: int):
    S = hk.RS(hk.fp32, 1, cols, run=1)      # run 1: lane l owns columns l, l+wave, ...

    @hk.kernel(warps=warps, grid=lambda p: (hk.cdiv(p.logits.rows, warps), 1, 1),
               name=f"dx_t8_docs_{cols}_{k}_{warps}_{e}")
    def router(logits: hk.Tensor[hk.bf16], ids: hk.Tensor[hk.i32], w: hk.Tensor[hk.fp32],
               E: hk.Constexpr[int] = e):
        row = hk.arch.block_idx.x * warps + hk.arch.warp_id()
        v = logits.at[row, 0].load(S, bounded=True)
        if E < cols:
            v.right_fill_(E, NEG)
        iota = hk.zeros(S)                       # the column index, as fp32
        for j in range(1, cols // WAVE):
            iota.right_fill_(j * WAVE, float(j * WAVE))
        iota += hk.arch.lane_id()
        m0 = v.max(axis=1).value()
        wsum = hk.zeros(hk.col_vec(S))
        out_w = hk.zeros(S)
        out_i = hk.zeros(S)
        for i in range(k):
            m = m0 if i == 0 else v.max(axis=1).value()
            hit = ((v - m[:, None]) * 1e30 + 1.0).step()          # 1 where v == m
            idx = (iota - hit * 4096.0 + 4096.0).min(axis=1).value()  # lowest such column
            e = (m - m0).exp()
            wsum += e
            slot = hk.zeros(S)
            slot.right_fill_(i, 1.0)
            slot.right_fill_(i + 1, 0.0)                           # 1 at column i only
            out_w += slot * e[:, None]
            out_i += slot * idx[:, None]
            if i + 1 < k:
                sel = (-(iota - idx[:, None]).abs() + 0.5).step()     # 1 at column idx
                v -= sel * 1e30
        out_w /= wsum[:, None]
        w.at[row, 0].store(out_w, bounded=True)
        ids.at[row, 0].store(out_i, bounded=True)

    # RDNA's pybind entry launches on the null stream (runtime.md "The stream"): not on
    # torch's stream, not in a CUDA graph. hk.torch_op launches on torch's stream there.
    return router if CDNA else hk.torch_op(router)


def _plan(t: int, e: int):
    cols = max(WAVE, -(-e // WAVE) * WAVE)
    return cols, 4


def supports(logits, k):
    if not 0 < k <= logits.shape[-1] <= 4096:
        return "needs 0 < k <= E <= 4096"
    return True


def solve(logits, k):
    t, e = logits.shape
    cols, warps = _plan(t, e)
    ids = torch.empty((t, k), device=logits.device, dtype=torch.int32)
    w = torch.empty((t, k), device=logits.device, dtype=torch.float32)
    _kernel(cols, k, warps, e)(logits, ids, w)
    return ids, w
