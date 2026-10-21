"""T1 from the docs, second attempt: register strips (hk.RS), which a user finds
by reading language.md past the first page ("Register strips": "a kernel that
only streams rows wants each lane to own contiguous runs of a row").

DX log:
- The strip layout and its rule "cols is a whole number of wave*run-column
  chunks": language.md "Register strips". The wave is 32 on gfx11 and 64 on
  gfx950, so the strip width is a per-architecture constant; how to ask for it
  outside a kernel came from examples/02_rmsnorm.py (`get_target(current_arch())`)
  and examples/05_attention.py (`hk.arch.target().wave` inside one).
- `bounded=True` on strip loads/stores covers both the ragged last row and the
  ragged last column block: language.md, same section.
GAP: no guidance on how many warps or columns a workgroup should take for a
     streaming op; WARPS=4 x one chunk each is a guess (hk.ops picks per shape).
GAP: the same gaps as docs_only.py about hk.math (no list, gelu's form undocumented).
"""

import torch

import hk
from hk.target import current_arch

KIND = "docs_only"
LABEL = "from the docs: register strips (language.md), 4 warps x one chunk"

WARPS = 4
CW = hk.get_target(current_arch()).wave * 8          # one chunk: wave lanes x runs of 8
S = hk.RS(hk.fp32, 1, CW)


@hk.kernel(warps=WARPS, grid=lambda p: (hk.cdiv(p.o.cols, CW * WARPS), p.o.rows, 1))
def gelu_and_mul(gate: hk.Tensor[hk.bf16], up: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    row = hk.arch.block_idx.y
    col = (hk.arch.block_idx.x * WARPS + hk.arch.warp_id()) * CW
    g = gate.at[row, col].load(S, bounded=True)
    u = up.at[row, col].load(S, bounded=True)
    o.at[row, col].store(hk.math.gelu(g) * u, bounded=True)


def solve(x):
    i = x.shape[-1] // 2
    out = torch.empty((x.shape[0], i), device=x.device, dtype=x.dtype)
    gelu_and_mul(x[:, :i], x[:, i:], out)
    return out
