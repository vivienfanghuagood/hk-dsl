"""examples/04_gemm.py through LDS, with every LDS read waited (wait=True).

The example keeps half of B's reads in flight across the first mma with
wait=False; on a CDNA target hk refuses that until hk-lds-pipeline counts
ds_reads per target (see docs/design/cdna.md). Everything else is the same
kernel: four warps, a 64 x 64 block of C, the workgroup copies A and B
global -> LDS, each warp reads its subtiles and runs mma_ABt.

On gfx950 that exercises: st_16x32 shared tiles derived from the operands they
feed, group<4>::load (include/cdna4's buffer_load ... lds, no VGPRs), the
unwaited ds_read_b128 of include/cdna4's load(rt, st), and the wait + bind hk
puts after it.

    HK_ARCH=gfx950 python3 tools/cdna-probes/gemm_lds.py           # GPU
    HK_ARCH=gfx942 python3 tools/cdna-probes/gemm_lds.py --build   # compile-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import _common  # noqa: E402

import hk  # noqa: E402

BM, BN, BK = 64, 64, 32
WARPS = 4
ST_A = hk.ST(hk.bf16, BM, BK)
ST_B = hk.ST(hk.bf16, BN, BK)
T_A = hk.RT(hk.bf16, 32, BK, "row")
T_B = hk.RT(hk.bf16, 16, BK, "row")
T_C = hk.RT(hk.fp32, 32, 16, "col")


@hk.kernel(warps=WARPS, grid=lambda p: (hk.cdiv(p.c.cols, BN), hk.cdiv(p.c.rows, BM), 1))
def gemm_lds(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
    g = hk.group(WARPS)
    As = hk.lds.alloc(ST_A, name="As")
    Bs = hk.lds.alloc(ST_B, name="Bs")
    row0 = hk.arch.block_idx.y * BM
    col0 = hk.arch.block_idx.x * BN
    wid = hk.arch.warp_id()
    wr, wc = wid // 2, wid % 2

    acc = [hk.zeros(T_C) for _ in range(2)]
    for k0 in range(0, a.cols, BK):
        g.load(As, a.at[row0, k0])
        g.load(Bs, b.at[col0, k0])
        hk.arch.barrier()
        at = As.subtile(32, BK, wr, 0).load(T_A)
        for h in range(2):
            bt = Bs.subtile(16, BK, 2 * wc + h, 0).load(T_B)
            acc[h] += at @ bt.T
        hk.arch.barrier()
    for h in range(2):
        c.at[row0 + wr * 32, col0 + wc * 32 + 16 * h] = acc[h]


def main():
    args = _common.parse(__doc__)
    if _common.static(args, gemm_lds):
        return
    _common.require_gpu()
    import torch

    torch.manual_seed(0)
    for m, n, k in ((64, 64, 32), (512, 256, 256), (4096, 4096, 4096)):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        gemm_lds(a, b, c)
        _common.check(f"gemm_lds {m}x{n}x{k} vs fp32 torch", c,
                      a.float() @ b.float().T, rel=1e-2)
    if args.no_bench:
        return
    best = _common.compare({
        "hk gemm_lds (unpipelined)": lambda: gemm_lds(a, b, c),
        "torch.matmul (hipBLASLt)": lambda: torch.matmul(a, b.T, out=c),
    }, args.rounds)
    _common.report("time", best, baseline="torch.matmul (hipBLASLt)",
                   work=2 * m * n * k, unit="TFLOP/s")


if __name__ == "__main__":
    main()
