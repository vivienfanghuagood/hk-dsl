"""for / if / while on device values: what becomes scf, what is carried, and
the C++ that comes out.

A kernel body is ordinary Python, rewritten so that a loop or a branch whose
bound or condition is a device value -- here a count read from memory with
`load_scalar` -- compiles to scf.for / scf.if / scf.while. One that only
involves Python values is unrolled or decided at trace time.

Both kernels compute the same thing: for each batch b, the sum of x[b]'s
first n[b] 16x16 column blocks.

  * `prefix_sum_for`: a runtime `for` over n. `col` is a Python int before
    the loop and rebound inside it, so it is carried through the loop (an
    iter_arg) and becomes a device integer.
  * `prefix_sum_while`: a runtime `while` over the row, with a runtime `if`
    that takes only the first n blocks. `k` is carried through the while.

The accumulator is a tile, and a tile is updated in place (`acc += ...`):
rebinding one inside a runtime loop is refused, because every iteration has
to write the same registers.

    python3 examples/06_control_flow.py
    python3 examples/06_control_flow.py --dump cpp
"""

import _common

import hk

T_IN = hk.RT(hk.bf16, 16, 16, "row")
T_ACC = hk.RT(hk.fp32, 16, 16, "row")


@hk.kernel(grid=lambda p: (1, 1, p.x.batch))
def prefix_sum_for(x: hk.Tensor[hk.bf16], n: hk.Tensor[hk.i32], o: hk.Tensor[hk.fp32]):
    b = hk.arch.block_idx.z
    count = n.at[0, b].load_scalar()        # a device value: the loop below is runtime
    acc = hk.zeros(T_ACC)
    col = 0
    for i in range(count):                  # scf.for, with col as an iter_arg
        acc += x[hk.elem_coord(b, 0, 0, col)].load(T_IN).to(hk.fp32)
        col += 16
    o[hk.elem_coord(b, 0, 0, 0)] = acc


@hk.kernel(grid=lambda p: (1, 1, p.x.batch))
def prefix_sum_while(x: hk.Tensor[hk.bf16], n: hk.Tensor[hk.i32], o: hk.Tensor[hk.fp32]):
    b = hk.arch.block_idx.z
    count = n.at[0, b].load_scalar()
    acc = hk.zeros(T_ACC)
    k = 0
    while k < x.cols:                       # scf.while, with k carried
        if k < count * 16:                  # scf.if on a device comparison
            acc += x[hk.elem_coord(b, 0, 0, k)].load(T_IN).to(hk.fp32)
        k += 16
    o[hk.elem_coord(b, 0, 0, 0)] = acc


def _control_lines(kernel) -> str:
    """The loops and branches of the generated C++, to see what Python became."""
    src = kernel.source(scaffold="bare")
    body = src[src.index("__global__"):]
    keep = ("for (", "while (", "if (", "break;", "int col =", "int k =", " col = ", " k = ")
    return "\n".join(line for line in body.splitlines() if any(t in line for t in keep))


def main():
    args = _common.parse(__doc__)
    if args.dump or args.build:
        for kernel in (prefix_sum_for, prefix_sum_while):
            _common.static(args, kernel)
        return
    _common.require_gpu()
    import torch

    batch, blocks = 6, 12
    x = torch.randn(batch, 1, 16, 16 * blocks, device="cuda", dtype=torch.bfloat16)
    n = torch.tensor([[0, 1, 3, 7, 11, 12]], device="cuda", dtype=torch.int32)
    blocks_of = x.float().reshape(batch, 16, blocks, 16)
    want = torch.stack([blocks_of[i, :, :int(c)].sum(1) for i, c in enumerate(n[0])])
    want = want.reshape(batch, 1, 16, 16)
    print(f"prefix block sums, {batch} batches of 16 x {16 * blocks}, n = {n[0].tolist()}")
    for kernel in (prefix_sum_for, prefix_sum_while):
        o = torch.empty(batch, 1, 16, 16, device="cuda", dtype=torch.float32)
        kernel(x, n, o)
        _common.check(kernel.name, o, want, rel=1e-5)
    for kernel in (prefix_sum_for, prefix_sum_while):
        print(f"\n  the control flow {kernel.name} compiled to:")
        print("\n".join("    " + line.strip() for line in _control_lines(kernel).splitlines()))


if __name__ == "__main__":
    main()
