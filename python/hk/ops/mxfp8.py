"""MXFP8 GEMM on gfx950: C = (sa . A)(sb . B)^T in bf16.

A (M, K) and B (N, K) are fp8 e4m3, K-contiguous (F.linear's layout); sa (M,
K/32) and sb (N, K/32) their E8M0 scales, one byte a row and 32 columns (the
OCP MX layout -- a 32x32-block weight scale, as DeepSeek-V4.1 ships, is its
rows repeated: `expand_block_scales`). On v_mfma_scale_f32_16x16x128_f8f6f4
through `hk.mma_ABt_mx`.

The first form: 128 x 128 blocks of C, four warps of 64 x 64, K in 128-wide
steps through one LDS buffer each for A and B, scales read from global per
step. M and N multiples of 128, K of 128.
"""

from __future__ import annotations

from .. import dsl as hk
from ..types import bf16, fp32, fp8e4m3, i8

BM = BN = 128
BK = 128
WARPS = 4


def mxfp8_kernel(name: str = "mxfp8_gemm_128x128"):
    st_a = hk.ST(fp8e4m3, BM, BK)
    st_b = hk.ST(fp8e4m3, BN, BK)
    t_a = hk.RT(fp8e4m3, 64, BK, "row")
    t_b = hk.RT(fp8e4m3, 64, BK, "row")
    t_c = hk.RT(fp32, 64, 64, "col")

    def body(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], sa: hk.Tensor[i8],
             sb: hk.Tensor[i8], c: hk.Tensor[bf16]):
        g = hk.group(WARPS)
        As = hk.lds.alloc(st_a, name="As")
        Bs = hk.lds.alloc(st_b, name="Bs")
        row0 = hk.arch.block_idx.y * BM
        col0 = hk.arch.block_idx.x * BN
        wid = hk.arch.warp_id()
        wr, wc = wid // 2, wid % 2
        acc = hk.zeros(t_c)
        for k0 in range(0, a.cols, BK):
            g.load(As, a.at[row0, k0])
            g.load(Bs, b.at[col0, k0])
            s_a = hk.load_mx_scales(sa, 64, BK // 32, row0 + wr * 64, k0 // 32)
            s_b = hk.load_mx_scales(sb, 64, BK // 32, col0 + wc * 64, k0 // 32)
            g.wait_for(As, Bs)
            hk.arch.barrier()
            ta = As.subtile(64, BK, wr, 0).load(t_a)
            tb = Bs.subtile(64, BK, wc, 0).load(t_b)
            hk.mma_ABt_mx(ta, tb, acc, s_a, s_b, into=acc)
            hk.arch.barrier()
        c.at[row0 + wr * 64, col0 + wc * 64] = acc

    body.__name__ = name
    return hk.kernel(body, arch="gfx950", warps=WARPS, name=name,
                     grid=lambda p: (p.c.cols // BN, p.c.rows // BM, 1))


_KERNEL = None


def gemm(a, b, sa, sb, out=None):
    """bf16 (M, N) = (sa . a)(sb . b)^T; see the module docstring."""
    import torch  # noqa: PLC0415

    global _KERNEL
    m, k = a.shape
    n = b.shape[0]
    if m % BM or n % BN or k % BK:
        raise ValueError(f"mxfp8.gemm: M={m} and N={n} multiples of {BM}, K={k} of {BK}")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if _KERNEL is None:
        _KERNEL = mxfp8_kernel()
    _KERNEL(a, b, sa.view(torch.int8), sb.view(torch.int8), out)
    return out


_PP = {}


def gemm_pp(a, b, sa, sb, out=None, *, block_n: int = 256, **sched):
    """The same product on gfx950's pingpong (hk.ops.gemm_cdna, mx=True):
    256 x block_n blocks of eight warps, K-tiles of 128 double-buffered
    through LDS. N a multiple of block_n, K of 128; any M."""
    import torch  # noqa: PLC0415

    from . import gemm_cdna  # noqa: PLC0415

    m, k = a.shape
    n = b.shape[0]
    if n % block_n or k % 128:
        raise ValueError(f"mxfp8.gemm_pp: N={n} a multiple of {block_n}, K={k} of 128")
    key = (block_n, tuple(sorted(sched.items())))
    kern = _PP.get(key)
    if kern is None:
        name = "mxfp8_pp_" + "_".join([str(block_n)] + [f"{k_}{v}" for k_, v in key[1]])
        kern = _PP[key] = gemm_cdna.gemm_kernel(name, mx=True, k_step=128,
                                                block_n=block_n, arch="gfx950", **sched)
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    kern(a, b, sa.view(torch.int8), sb.view(torch.int8), out)
    return out


def expand_block_scales(s, block_rows: int = 32):
    """A weight's 32x32-block E8M0 scales [N/32, K/32] as one row each [N, K/32]."""
    return s.repeat_interleave(block_rows, dim=0).contiguous()
