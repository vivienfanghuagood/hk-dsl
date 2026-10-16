"""Does LLVM hoist an MFMA above the wait that guards its operand on gfx950?

hk binds a tile's registers after every s_waitcnt that retires its LDS reads
(lds_wait_for), because the ds_reads are inline asm whose outputs LLVM takes
as ready when the asm ends, and the MFMA builtin is pure: nothing in the IR
keeps it below a separate s_waitcnt asm. On gfx1100 it was measured to
hoist. This probe asks the same of gfx950.

For each register-pressure point (BK: the reduction depth of a step; NB: the
B tiles in flight behind A) it builds an LDS GEMM step loop twice, as hk
generates it and with every bind removed (lds_wait_for<N>(...) ->
lds_wait<N>()), disassembles both, and walks the instruction stream: an MFMA
that reads a ds_read's destination registers while that ds_read is not yet
retired by a covering `s_waitcnt lgkmcnt(n)` (one that leaves at most n
newer LDS ops outstanding) is a hazard -- it computes on registers the LDS
has not written. The walk is linear, with the loop body visited twice so a
read issued at the bottom and consumed at the top is seen.

    HK_ARCH=gfx950 python3 tools/cdna-probes/bind_hazard.py [--run]

--run also launches each variant on the GPU (in rocm/pytorch) and checks it
against torch, best of 3, so a hazard in the ISA can be seen as a wrong
answer.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402

ARCH = "gfx950"


def gemm_steps(bk: int, nb: int):
    """One warp, C[32 x 16*nb] = A[32 x K] B[16*nb x K]^T, K in steps of bk:
    each step DMAs A and B into LDS, reads A and every B subtile without
    waiting, and retires them one at a time as the MFMAs need them."""
    st_a, st_b = hk.ST(hk.bf16, 32, bk), hk.ST(hk.bf16, 16 * nb, bk)
    t_a, t_b = hk.RT(hk.bf16, 32, bk, "row"), hk.RT(hk.bf16, 16, bk, "row")
    t_c = hk.RT(hk.fp32, 32, 16, "col")

    @hk.kernel(arch=ARCH, warps=1, grid=lambda p: (1, 1, 1), name=f"bh_bk{bk}_nb{nb}")
    def k(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
        As = hk.lds.alloc(st_a, name="As")
        Bs = hk.lds.alloc(st_b, name="Bs")
        g = hk.group(1)
        accs = [hk.zeros(t_c) for _ in range(nb)]
        for k0 in range(0, a.cols, bk):
            g.load(As, a.at[0, k0])
            g.load(Bs, b.at[0, k0])
            hk.arch.barrier()
            at = As.load(t_a, wait=False)
            bts = [Bs.subtile(16, bk, h, 0).load(t_b, wait=False) for h in range(nb)]
            hk.lds.wait_for(at, bts[0])
            accs[0] += at @ bts[0].T
            for h in range(1, nb):
                hk.lds.wait_for(bts[h])
                accs[h] += at @ bts[h].T
            hk.arch.barrier()
        for h in range(nb):
            c.at[0, 16 * h] = accs[h]

    return k


_BIND = re.compile(r"kittens::lds_wait_for<(\d+)>\([^)]*\);")


def unbound(src: str) -> str:
    return _BIND.sub(r"kittens::lds_wait<\1>();", src)


def objdump(path: Path) -> str:
    exe = hc.rocm_path() / "llvm" / "bin" / "llvm-objdump"
    return subprocess.run([str(exe), "-d", str(path)], capture_output=True,
                          text=True, check=True).stdout


_REG = re.compile(r"\b([va])(?:\[(\d+):(\d+)\]|(\d+)(?!\d))")


def regs(operand: str):
    out = set()
    for m in _REG.finditer(operand):
        kind = m.group(1)
        lo, hi = (int(m.group(2)), int(m.group(3))) if m.group(2) else (int(m.group(4)),) * 2
        out |= {(kind, r) for r in range(lo, hi + 1)}
    return out


def hazards(asm: str):
    """(hazards, mfmas, ds_reads) over the kernel's instruction stream."""
    lines = [l.split("//")[0].strip() for l in asm.splitlines()]
    insts = [l for l in lines if re.match(r"^(ds_read|v_mfma|s_waitcnt|s_cbranch|s_branch)", l)]
    found, mfmas, reads = [], 0, 0
    # Walk twice: what is pending at the loop's bottom is pending at its top.
    pending = []
    for rnd in range(2):
        for ins in insts:
            op, _, rest = ins.partition(" ")
            if op.startswith("ds_read"):
                pending.append(regs(rest.split(",")[0]))
                reads += rnd == 0
            elif op == "s_waitcnt":
                m = re.search(r"lgkmcnt\((\d+)\)", rest)
                if m:
                    keep = int(m.group(1))
                    pending = pending[len(pending) - keep:] if keep else []
            elif op.startswith("v_mfma"):
                mfmas += rnd == 0
                srcs = set()
                for o in rest.split(",")[1:4]:
                    srcs |= regs(o)
                for p in pending:
                    if p & srcs:
                        found.append(ins)
                        break
    return found, mfmas, reads


def run(tr, build, bk, nb, trials=3):
    """Worst relative error of the variant against torch over `trials` runs."""
    import torch  # noqa: PLC0415

    from hk.runtime import codeobj  # noqa: PLC0415

    k = codeobj.CodeObjectKernel(tr, build)
    g = torch.Generator(device="cuda").manual_seed(bk * 100 + nb)
    K = bk * 16
    a = torch.randn(32, K, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(16 * nb, K, device="cuda", dtype=torch.bfloat16, generator=g)
    want = a.float() @ b.float().T
    worst = 0.0
    for _ in range(trials):
        c = torch.zeros(32, 16 * nb, device="cuda", dtype=torch.bfloat16)
        shapes, ptrs, rows = codeobj.tensor_args(k, (a, b, c), tr.name)
        k.launch(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream,
                 shapes, ptrs, rows)
        torch.cuda.synchronize()
        err = ((c.float() - want).abs().max() / want.abs().max()).item()
        worst = err if err != err or err > worst else worst   # NaN sticks
    return worst


def main():
    points = [(32, 2), (64, 2), (64, 4), (128, 2), (128, 4), (64, 8), (128, 8)]
    launch = "--run" in sys.argv
    print(f"{'BK':>4} {'NB':>3} {'variant':8} {'VGPR+AGPR':>10} {'ds_read':>7} "
          f"{'mfma':>5} {'hazards':>7}" + ("  max rel err (3 runs)" if launch else ""))
    for bk, nb in points:
        k = gemm_steps(bk, nb)
        try:
            tr = k.trace()
        except Exception as e:  # noqa: BLE001
            print(f"{bk:4d} {nb:3d} trace failed: {str(e).splitlines()[0][:100]}")
            continue
        src = scaffold.device(tr)
        for variant, s in (("bound", src), ("unbound", unbound(src))):
            try:
                b = hc.build(s, ARCH, name=f"{tr.name}_{variant}", kind="co")
            except Exception as e:  # noqa: BLE001
                print(f"{bk:4d} {nb:3d} {variant:8} build failed: {str(e).splitlines()[0][:90]}")
                continue
            (r,) = b.kernels
            h, m, d = hazards(objdump(b.so_path))
            err = f"  {run(tr, b, bk, nb):.3g}" if launch else ""
            print(f"{bk:4d} {nb:3d} {variant:8} {r.vgpr:>4}+{r.agpr:<5} {d:7d} {m:5d} {len(h):7d}"
                  + err + (f"  e.g. {h[0][:60]}" if h and not launch else ""))


if __name__ == "__main__":
    main()
