"""C = A·Bᵀ through LDS on the WMMA unit, with a derived LDS wait.

A is M x K and B is N x K, both K-contiguous (the "NT" layout of a linear
layer's weight); C is M x N in bf16, accumulated in fp32. One workgroup of
four warps per 64 x 64 block of C, each warp owning a 32 x 32 quarter. Per
32-wide step of K:

  * the whole workgroup copies a 64 x 32 tile of A and of B into LDS
    (`hk.group(...).load`), then a barrier makes it visible to every warp;
  * each warp issues its LDS reads without waiting (`wait=False`): its A
    rows, and its B rows in two halves;
  * it waits for A and the first half of B only, and runs that half's
    `acc[0] += at @ bt[0].T` while the second half is still in flight; then
    it waits for the rest and runs the other half. LDS returns in order, so
    retiring the first half means waiting until only the second half's reads
    are outstanding -- the hk-lds-pipeline pass derives that count, and
    `--dump ir` shows it on each wait as `{n = ...}`;
  * a second barrier keeps the next copy from overwriting a tile a slower
    warp is still reading.

This is the shape of a GEMM, not a fast one: one buffer, nothing overlapped.
hk.ops.gemm.matmul, which it is compared against, double-buffers through a
register staging pipeline and is tuned per shape; read it next (hk/ops/gemm.py).
M and N have to be multiples of 64 and K of 32 here.

    python3 examples/04_gemm.py
    python3 examples/04_gemm.py --dump ir | grep lds_wait_for
"""

import _common

import hk

BM, BN, BK = 64, 64, 32
WARPS = 4                                  # 2 x 2 warps, 32 x 32 of C each
ST_A = hk.ST(hk.bf16, BM, BK)
ST_B = hk.ST(hk.bf16, BN, BK)
T_A = hk.RT(hk.bf16, 32, BK, "row")       # a warp's 32 rows of A
T_B = hk.RT(hk.bf16, 16, BK, "row")       # half of its 32 rows of B
T_C = hk.RT(hk.fp32, 32, 16, "col")       # WMMA accumulates into col layout


@hk.kernel(warps=WARPS, grid=lambda p: (hk.cdiv(p.c.cols, BN), hk.cdiv(p.c.rows, BM), 1))
def gemm(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
    g = hk.group(WARPS)
    As = hk.lds.alloc(ST_A, name="As")
    Bs = hk.lds.alloc(ST_B, name="Bs")
    row0 = hk.arch.block_idx.y * BM
    col0 = hk.arch.block_idx.x * BN
    wid = hk.arch.warp_id()
    wr, wc = wid // 2, wid % 2

    acc = [hk.zeros(T_C) for _ in range(2)]
    for k0 in range(0, a.cols, BK):                       # runtime loop over K
        g.load(As, a.at[row0, k0])                        # the workgroup: global -> LDS
        g.load(Bs, b.at[col0, k0])
        hk.arch.barrier()
        at = As.subtile(32, BK, wr, 0).load(T_A, wait=False)
        bt = [Bs.subtile(16, BK, 2 * wc + h, 0).load(T_B, wait=False) for h in range(2)]
        hk.lds.wait_for(at, bt[0])                        # bt[1] still in flight: n > 0
        acc[0] += at @ bt[0].T                            # mma_ABt into acc[0]
        hk.lds.wait_for(bt[1])                            # nothing behind it: n = 0
        acc[1] += at @ bt[1].T
        hk.arch.barrier()
    for h in range(2):                                    # fp32 -> bf16 on the way out
        c.at[row0 + wr * 32, col0 + wc * 32 + 16 * h] = acc[h]


def main():
    args = _common.parse(__doc__)
    if _common.static(args, gemm):
        return
    _common.require_gpu()
    import torch

    m = n = k = 4096
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    print(f"GEMM C = A·Bᵀ, M = N = K = {m}, bf16 in, fp32 accumulate")
    gemm(a, b, c)
    _common.check("this kernel vs fp32 torch", c, a.float() @ b.float().T, rel=1e-2)
    if args.no_bench:
        return
    contenders = {"hk (this kernel)": lambda: gemm(a, b, c),
                  "hk.ops.gemm.matmul": lambda: hk.ops.gemm.matmul(a, b, c),
                  "torch.matmul": lambda: torch.matmul(a, b.T)}
    from hk.target import current_arch, get_target  # noqa: PLC0415

    if get_target(current_arch()).wave == 64 and current_arch() != "gfx950":
        # hk.ops.gemm has a gfx950 form (hk.ops.gemm_cdna), not a gfx942 one.
        del contenders["hk.ops.gemm.matmul"]
    best = _common.compare(contenders, args.rounds)
    _common.report("time", best, baseline="torch.matmul", work=2.0 * m * n * k,
                   unit="TFLOP/s")


if __name__ == "__main__":
    main()
