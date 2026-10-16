"""The ds_read count hk-lds-pipeline derives on CDNA, held to the ISA.

A counted wait is only right if the count is: a load counted high makes
every wait issued while it is in flight too shallow, which is a wrong answer
some of the time. On CDNA the count is include/cdna{3,4}'s and depends on the
fragment, so each kernel here is one asynchronous LDS load of one kind -- an
mma operand in each layout, a plain 16x16 tile in each -- built for the
target, disassembled, and its ds_reads counted against `lds_loads` in the IR.
Compile-only: hipcc and llvm-objdump, no GPU.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402


def _objdump() -> str:
    for c in (shutil.which("llvm-objdump"), str(hc.rocm_path() / "llvm" / "bin" / "llvm-objdump")):
        if c and Path(c).exists():
            return c
    pytest.skip("no llvm-objdump")


def _kernel(arch, kind):
    st = hk.ST(hk.bf16, 32, 32)
    acc_t = hk.RT(hk.fp32, 32, 32, "col")
    row = hk.RT(hk.bf16, 32, 32, "row")
    col = hk.RT(hk.bf16, 32, 32, "col")
    small = {"plain_row": hk.RT(hk.bf16, 32, 32, "row"),
             "plain_col": hk.RT(hk.bf16, 32, 32, "col")}

    @hk.kernel(arch=arch, warps=1, grid=lambda p: (1, 1, 1), name=f"cnt_{kind}")
    def k(a: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(st, name="s")
        hk.group(1).load(s, a.at[0, 0])
        hk.arch.barrier()
        if kind == "operand_row":                  # A and B of mma_ABt
            x = s.load(row, wait=False)
            hk.lds.wait_for(x)
            acc = hk.zeros(acc_t)
            acc += x @ x.T
            c.at[0, 0] = acc
        elif kind == "operand_col":                # B of mma_AB
            y = s.load(col, wait=False)
            hk.lds.wait_for(y)
            acc = hk.zeros(acc_t)
            acc += a.at[0, 0].load(row) @ y
            c.at[0, 0] = acc
        else:                                      # not an operand: 16x16
            z = s.load(small[kind], wait=False)
            hk.lds.wait_for(z)
            c.at[0, 0] = z.to(hk.fp32).to(hk.bf16)

    return k


CASES = [("gfx950", "operand_row", 2), ("gfx950", "operand_col", 4),
         ("gfx950", "plain_row", 4), ("gfx950", "plain_col", 4),
         ("gfx942", "operand_row", 4), ("gfx942", "operand_col", None),
         ("gfx942", "plain_row", 4)]


@pytest.mark.parametrize("arch,kind,want", CASES)
def test_the_derived_count_is_the_isa_count(arch, kind, want):
    k = _kernel(arch, kind)
    if want is None:
        # include/cdna3 reads a col tile with ordinary loads: nothing to
        # keep in flight, and hk says so instead of guessing a count.
        with pytest.raises(Exception, match="nothing to keep in flight"):
            k.trace()
        return
    tr = k.trace()
    (n,) = {int(m) for m in re.findall(r"lds_loads = (\d+)", tr.dump())}
    assert n == want
    b = hc.build(scaffold.device(tr), arch, name=tr.name, kind="co")
    asm = subprocess.run([_objdump(), "-d", str(b.so_path)], capture_output=True,
                         text=True, check=True).stdout
    assert len(re.findall(r"\bds_read", asm)) == n
