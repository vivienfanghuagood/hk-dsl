"""T1 written from the docs only, the way the first page teaches: one warp per
16x64 register tile (docs/getting-started.md "Your first kernel"), fp32 math.

DX log -- what had to be looked up, and where:
- The kernel shape, grid and tile_coord: getting-started.md (the `add` kernel).
- The two halves as strided views without a copy: language.md "Memory"
  ("a slice of the last axis -- the two halves of a fused (M, 2I) output -- is
  accepted without a copy"). Found; good.
- Ragged M (37 rows): language.md "Memory": `x[idx].load(ty, bounded=True)`.
GAP: the bounded *store* spelling is not shown anywhere; `o[idx] = v` takes no
     keyword, so `o[idx].store(v, bounded=True)` is a guess from the strip section's
     "bounded=True ... writes nothing there" (it works).
GAP: no list of the math functions. language.md says only "hk.math: elementwise
     functions"; that gelu exists at all is learned from "Errors and locations"
     ("gelu on a bf16 tile raises while tracing"). Which GELU it is (tanh or erf)
     is not documented: the tanh form had to be confirmed by testing against torch
     (hk.math.__all__ is the real list; there is no tanh, no sigmoid, no erf).
GAP: gelu is fp32-only (said only in the error section), so the tiles are loaded
     as fp32 (`load(hk.RT(hk.fp32, ...))` converts bf16 on the way in: language.md's
     first example comment) and converted back with `.to(hk.bf16)`.
GAP: nothing says this first-page tiling is slow at prefill sizes (0.6x
     torch.compile on gfx950 at 4096x14336) or that register strips (hk.RS) are the
     layout for streaming kernels; see docs_only_strips.py for that attempt.
"""

import torch

import hk

KIND = "docs_only"
LABEL = "from the docs: 16x64 register tiles, as the first page teaches"

T = hk.RT(hk.fp32, 16, 64)


@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
def gelu_and_mul(gate: hk.Tensor[hk.bf16], up: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    g = gate[idx].load(T, bounded=True)
    u = up[idx].load(T, bounded=True)
    o[idx].store((hk.math.gelu(g) * u).to(hk.bf16), bounded=True)


def solve(x):
    i = x.shape[-1] // 2
    out = torch.empty((x.shape[0], i), device=x.device, dtype=x.dtype)
    gelu_and_mul(x[:, :i], x[:, i:], out)
    return out
