"""Does the compiler sink a scaled MFMA into a lane-divergent `if` on gfx950?

K1's MoE stage 2 did its top-k sum with `scatter(..., add=True)`. It was 0.28
off under the hipcc in rocm/pytorch 7.2.3 (roc-7.2.3) and exact under
ROCm 10's. The scatter's per-row `if (row valid)` is the only block that uses
the MFMA results, and roc-7.2.3's MachineSink moves the V_MFMA_SCALE
instructions into it. The plain MFMAs (v_mfma_f32_16x16x32_bf16) stay put,
and ROCm 10 moves neither: the same LLVM IR through each llc differs at that
pass alone.

An MFMA under a partial EXEC still reads all 64 lanes. This probe's sunk
MFMAs compute the right values, because their operands were complete before
the `if`. In K1 the sunk block also held the copies that assemble the FP4
weight fragment out of AGPRs. Under the partial EXEC those left the masked
lanes' operand registers holding the previous tile's values, so every row
came out wrong.

hk's guard is detail::hk_pinned in include/hk/cdna.cuh. It is an empty asm
that takes the scatter's tile before the `if`, so whatever computed the tile
stays above it. HK_UNPINNED builds without it. hk.runtime.cocheck reports an
MFMA inside an `s_and_saveexec` region (a warning).

`kernel(steps)` is one warp in K1's stage-2 shape. Its inputs are FP4 weights w [n, K]
in AITER's layout and x [64, K] fp8, each with E8M0 byte scales (K = 128 * steps).
For each 16-row tile t of x:

    acc = mma_ABt_mx(w, x[16t:16t+16], a_fmt="fp4")      # [n, 16], col layout
    out.scatter(transpose(acc), ids, 16 * t, sort_topk=TOPK, add=True)

ids are AITER's MoE sort ids (token | choice << 24). Rows whose token is >= M
are padding and add nothing.

    HK_ARCH=gfx950 python3 tools/cdna-probes/mfma_scale_sink.py      # the ISA, both variants
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import hk  # noqa: E402
from hk import bf16, fp8e4m3, fp32, i8, i32  # noqa: E402

ARCH = "gfx950"
TOPK = 2
ROWS = 64


def kernel(steps: int = 2, pinned: bool = True, n: int = 32):
    K = 128 * steps

    def mx_sink(w: hk.Tensor[i8], ws: hk.Tensor[i8], x: hk.Tensor[fp8e4m3], xs: hk.Tensor[i8],
                ids: hk.Tensor[i32], out: hk.Tensor[bf16]):
        tw = hk.load_fp4_frags(hk.RT(fp8e4m3, n, K), w, 0, 0, group_rows=n)
        s_w = hk.load_mx_scales(ws, n, K // 32, 0, 0, layout="aiter", group_rows=n)
        accs = []
        for t in range(ROWS // 16):     # every tile's product first, then the stores: K1's epilogue
            tx = x[hk.elem_coord(0, 0, 16 * t, 0)].load(hk.RT(fp8e4m3, 16, K))
            s_x = hk.load_mx_scales(xs, 16, K // 32, 16 * t, 0)
            accs.append(hk.mma_ABt_mx(tw, tx, hk.zeros(hk.RT(fp32, n, 16, "col")), s_w, s_x,
                                      a_fmt="fp4"))
        for t in range(ROWS // 16):
            out.scatter(hk.transpose(accs[t]), ids, 16 * t, col=0, sort_topk=TOPK, add=True)

    return hk.kernel(mx_sink, arch=ARCH, warps=1,
                     name=f"mx_sink_{steps}_{n}" + ("" if pinned else "_unpinned"),
                     grid=lambda p: (1, 1, 1), max_vgprs=512,
                     defines=None if pinned else {"HK_UNPINNED": 1})


def source(steps: int = 2, pinned: bool = True) -> str:
    from hk.codegen import scaffold  # noqa: PLC0415

    k = kernel(steps, pinned)
    return k.with_defines(scaffold.device(k.trace()))


def main() -> int:
    from hk.runtime import cocheck  # noqa: PLC0415
    from hk.runtime import compile as hc  # noqa: PLC0415

    bad = 0
    for pinned in (True, False):
        b = hc.build(source(2, pinned), ARCH, name=f"mx_sink_{'pinned' if pinned else 'unpinned'}",
                     kind="co")
        for k in cocheck.check(Path(b.so_path)):
            print(("pinned  " if pinned else "unpinned"), k.line())
            bad += pinned and any("partial EXEC" in w for w in k.warnings)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
