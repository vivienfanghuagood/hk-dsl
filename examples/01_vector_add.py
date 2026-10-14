"""Elementwise add of two bf16 tensors: the smallest complete kernel.

One workgroup of one warp per 16x64 tile. The grid is a function of the
output's shape; a tile coordinate addresses the same tile in all three
tensors; `+` on the two register tiles is one `kittens::add`. Rows and
columns have to be multiples of the tile here -- `x[idx].load(t,
bounded=True)` is how a kernel takes a ragged last row (see hk/ops/norm.py).

    python3 examples/01_vector_add.py
    python3 examples/01_vector_add.py --dump cpp
"""

import _common

import hk

ROWS, COLS = 16, 64
T = hk.RT(hk.bf16, ROWS, COLS)


@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, COLS), hk.cdiv(p.o.rows, ROWS), p.o.batch))
def add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                        hk.arch.block_idx.x)
    o[idx] = a[idx].load(T) + b[idx].load(T)


def main():
    args = _common.parse(__doc__)
    if _common.static(args, add):
        return
    _common.require_gpu()
    import torch

    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    o = torch.empty_like(a)
    print("vector add, 8192 x 8192 bf16")
    add(a, b, o)
    # Both round the fp32 sum to bf16 once, so they agree exactly.
    _common.check("hk add vs torch", o, a + b, rel=0.0)
    if args.no_bench:
        return
    best = _common.compare({
        "hk (this kernel)": lambda: add(a, b, o),
        "torch.add": lambda: torch.add(a, b, out=o),
    }, args.rounds)
    _common.report("time", best, baseline="torch.add",
                   work=3 * a.numel() * a.element_size(), unit="TB/s")


if __name__ == "__main__":
    main()
