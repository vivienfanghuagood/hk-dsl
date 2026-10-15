"""The smallest matrix kernel that reaches the MFMA unit: C = A·Bᵀ with no LDS.

One warp per 32 x 32 block of C. Per 32-wide step of K it loads a 32 x 32
tile of A and two 16 x 32 tiles of B straight from global into registers and
runs `acc[h] += a @ b[h].T` -- mma_ABt. On gfx950 the operand tiles become
`rt<bf16, ..., row, rt_16x32>` and each 16x16 output fragment is one
v_mfma_f32_16x16x32_bf16; on gfx942 they stay 16x16 and the instruction is
v_mfma_f32_16x16x16_bf16. Not a fast GEMM -- every warp re-reads its operands
from global -- a reachability check for the lowering.

    HK_ARCH=gfx950 python3 tools/cdna-probes/mma_reg.py            # GPU: check vs torch
    HK_ARCH=gfx950 python3 tools/cdna-probes/mma_reg.py --dump cpp
    HK_ARCH=gfx942 python3 tools/cdna-probes/mma_reg.py --build    # compile-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import _common  # noqa: E402

import hk  # noqa: E402

BM, BN, BK = 32, 32, 32
T_A = hk.RT(hk.bf16, BM, BK, "row")
T_B = hk.RT(hk.bf16, 16, BK, "row")
T_C = hk.RT(hk.fp32, BM, 16, "col")


@hk.kernel(warps=1, grid=lambda p: (hk.cdiv(p.c.cols, BN), hk.cdiv(p.c.rows, BM), 1))
def mma_reg(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
    row0 = hk.arch.block_idx.y * BM
    col0 = hk.arch.block_idx.x * BN
    acc = [hk.zeros(T_C) for _ in range(2)]
    for k0 in range(0, a.cols, BK):
        at = a.at[row0, k0].load(T_A)
        for h in range(2):
            bt = b.at[col0 + 16 * h, k0].load(T_B)
            acc[h] += at @ bt.T
    for h in range(2):
        c.at[row0, col0 + 16 * h] = acc[h]


def main():
    args = _common.parse(__doc__)
    if _common.static(args, mma_reg):
        return
    _common.require_gpu()
    import torch

    torch.manual_seed(0)
    for m, n, k in ((64, 64, 32), (256, 512, 128), (2048, 2048, 1024)):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        mma_reg(a, b, c)
        _common.check(f"mma_reg {m}x{n}x{k} vs fp32 torch", c,
                      a.float() @ b.float().T, rel=1e-2)


if __name__ == "__main__":
    main()
