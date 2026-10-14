"""Row softmax: a max, an exponential, a sum and a normalisation, each a pass
over the row.

o = exp(x - max) / sum(exp(x - max)) over the last axis, one workgroup per 16
rows split across WARPS warps. Three passes, each combined across the warps
through LDS: the row max, the sum of exponentials against it, then the
normalised output. A reduction leaves one value per row (a column vector);
`m[:, None]` broadcasts it back along the row. hk.ops.norm.softmax, which
this is compared against, is the planned and tuned version.

    python3 examples/03_softmax.py
    python3 examples/03_softmax.py --dump ir
"""

import _common

import hk

T = hk.RT(hk.fp32, 16, 64)


def softmax_kernel(warps: int = 4):
    @hk.kernel(warps=warps, grid=lambda p: (1, hk.cdiv(p.x.rows, 16), p.x.batch))
    def softmax(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16],
                WARPS: hk.Constexpr[int] = warps):
        row0, wid = hk.arch.block_idx.y * 16, hk.arch.warp_id()
        blocks = hk.cdiv(x.cols, 64)

        m = hk.neg_infty(hk.col_vec(T))
        for cb in range(wid, blocks, WARPS):
            m.max_(x.at[row0, cb * 64].load(T).max(axis=1))
        m = hk.collective.cross_warp(m, "max", WARPS)

        s = hk.zeros(hk.col_vec(T))
        for cb in range(wid, blocks, WARPS):
            s += (x.at[row0, cb * 64].load(T) - m[:, None]).exp().sum(axis=1)
        s = hk.collective.cross_warp(s, "add", WARPS)

        for cb in range(wid, blocks, WARPS):
            e = (x.at[row0, cb * 64].load(T) - m[:, None]).exp()
            o.at[row0, cb * 64] = (e / s[:, None]).to(hk.bf16)

    return softmax


def main():
    args = _common.parse(__doc__)
    kernel = softmax_kernel()
    if _common.static(args, kernel):
        return
    _common.require_gpu()
    import torch

    rows, cols = 8192, 4096
    x = 4 * torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)
    print(f"softmax, {rows} x {cols} bf16")
    kernel(x, o)
    _common.check("this kernel vs fp32 torch", o, torch.softmax(x.float(), -1), rel=2e-2)
    if args.no_bench:
        return
    best = _common.compare({
        "hk (this kernel)": lambda: kernel(x, o),
        "hk.ops.norm.softmax": lambda: hk.ops.norm.softmax(x, out=o),
        "torch.softmax": lambda: torch.softmax(x, -1),
    }, args.rounds)
    _common.report("time", best, baseline="torch.softmax",
                   work=2 * x.numel() * x.element_size(), unit="TB/s")


if __name__ == "__main__":
    main()
