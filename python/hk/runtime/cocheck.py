"""hk's two gates, on any AMDGPU code object: resources, and waits.

hk refuses to return a kernel that spills, and its IR keeps the waits that
inline-asm LDS reads need. Neither check needs hk's source. Both can be read
off a compiled code object: the resources from its AMDGPU metadata note, the
waits from its disassembly. This runs them on any code object -- hk's,
HipKittens', Triton's, FlyDSL's -- so a kernel from anywhere is held to the
same rules:

* **Resources.** Scratch (a private segment) or a spill fails: in a library
  that places s_waitcnt by hand, scratch traffic reorders against it. The
  VGPR, AGPR and SGPR counts and the occupancy they allow on the target are
  reported (gfx90a+ charges the AGPRs to the same file; the note's
  `.vgpr_count` is already that sum there).
* **Waits.** An instruction that reads a VGPR an LDS or vector-memory load
  is still writing -- before an s_waitcnt retires the load -- fails. The
  hardware does not interlock on these registers; the compiler inserts the
  waits for loads it emitted itself and nothing does for loads written as
  inline asm, which is what the bind after hk's waits is for. Measured on
  gfx950 (tools/cdna-probes/bind_hazard.py): with the bind removed, LLVM
  hoists the MFMAs above the waits and every configuration returns NaN.
* **EXEC** (a warning). An MFMA inside a lane-divergent `if` (an
  `s_and_saveexec` region). The MFMA reads every lane's operands, and whatever
  was computed beside it in that block ran only in the active lanes. That is
  wrong unless the condition is uniform when it runs. hk's attention kernels
  branch on a VGPR-laundered constant (`c4_live`), which is uniform at run
  time and is reported here too. roc-7.2.3 sinks V_MFMA_SCALE into a
  scatter-add's per-row `if` when only that block uses the result
  (tests/hk/codegen/test_mfma_scale_sink.py). K1 lost its operand copies to
  this, and its stage 2 came out 0.28 off.

The wait scan is a linear walk of each kernel's instructions, done twice so
a load at the bottom of a loop that is consumed at its top is seen. It
models the counters as in-order queues of loads (`lgkmcnt(n)` / `dscnt`
leaves the newest n LDS loads in flight, `vmcnt(n)` / `loadcnt` the newest
n vector-memory loads, copies into LDS among them), which over-approximates what is in flight when
scalar loads or stores share a counter: a report can be a false alarm, a
clean pass is only as good as the linear walk is for the kernel's control
flow.

    python -m hk.runtime.cocheck kernel.co [more.co ...]
    python -m hk.runtime.cocheck --bundle lib.so     # a host binary with an embedded HIP fat binary

Exits 1 if any kernel fails either gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..target import TARGETS


def _tool(name: str) -> str:
    from .compile import rocm_path  # noqa: PLC0415

    p = rocm_path() / "llvm" / "bin" / name
    return str(p) if p.exists() else name


# ---------------------------------------------------------------- resources


@dataclass
class Kernel:
    name: str
    arch: str = ""
    vgpr: int = 0
    agpr: int = 0
    sgpr: int = 0
    scratch: int = 0
    vgpr_spill: int = 0
    sgpr_spill: int = 0
    wave: int = 0
    occupancy: Optional[int] = None
    hazards: List[str] = field(default_factory=list)
    #: Reads of a load's register by the next instruction's address: the old
    #: value, by issue order; and MFMAs under a partial EXEC (exec_hazards).
    #: Warnings, not failures.
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.scratch or self.vgpr_spill or self.sgpr_spill or self.hazards)

    def line(self) -> str:
        occ = f" occ={self.occupancy}" if self.occupancy is not None else ""
        agpr = f" ({self.agpr} AGPR)" if self.agpr else ""
        status = "OK  " if self.ok else "FAIL"
        out = (f"{status} {self.name}: vgpr={self.vgpr}{agpr} sgpr={self.sgpr}"
               f" scratch={self.scratch} spill={self.vgpr_spill}/{self.sgpr_spill}{occ}"
               f" wait-hazards={len(self.hazards)}"
               + (f" warnings={len(self.warnings)}" if self.warnings else ""))
        for h in self.hazards[:3]:
            out += f"\n       {h}"
        for h in self.warnings[:1]:
            out += f"\n       (warning) {h}"
        return out


def resources(co: Path) -> Dict[str, Kernel]:
    """Per kernel symbol, the counts in the code object's metadata note."""
    return parse_notes(subprocess.run([_tool("llvm-readelf"), "--notes", str(co)],
                                      capture_output=True, text=True, check=True).stdout)


def parse_notes(out: str) -> Dict[str, Kernel]:
    """`llvm-readelf --notes`, read: under `amdhsa.kernels`, an entry per
    kernel at "  - .", its own keys at four spaces, nested lists deeper."""
    m = re.search(r"amdhsa\.target:\s+\S*--(gfx\w+)", out)
    arch = m.group(1) if m else ""
    entries: List[Dict[str, str]] = []
    inside = False
    for line in out.splitlines():
        if line.startswith("amdhsa.kernels:"):
            inside = True
            continue
        if inside and re.match(r"^\S", line):
            break
        if not inside:
            continue
        m = re.match(r"^  - \.(\w+):\s*(.*)$", line) or re.match(r"^    \.(\w+):\s*(.*)$", line)
        if not m:
            continue
        if line.startswith("  - "):
            entries.append({})
        if entries:
            entries[-1][m.group(1)] = m.group(2).strip()
    t = TARGETS.get(arch)
    kernels: Dict[str, Kernel] = {}
    for e in entries:
        k = Kernel(name=e.get("name", "?"), arch=arch,
                   vgpr=int(e.get("vgpr_count", 0)), agpr=int(e.get("agpr_count", 0)),
                   sgpr=int(e.get("sgpr_count", 0)),
                   scratch=int(e.get("private_segment_fixed_size", 0)),
                   vgpr_spill=int(e.get("vgpr_spill_count", 0)),
                   sgpr_spill=int(e.get("sgpr_spill_count", 0)),
                   wave=int(e.get("wavefront_size", 0)))
        if t is not None and k.vgpr:
            # gfx90a+'s .vgpr_count already includes the AGPRs.
            k.occupancy = t.occupancy(k.vgpr)
        kernels[k.name] = k
    return kernels


# ---------------------------------------------------------------- waits

_REG = re.compile(r"\b([va])(?:\[(\d+):(\d+)\]|(\d+)(?!\d))")
_LDS_LOAD = re.compile(r"^ds_(read|load|bpermute|permute|swizzle)|^ds_\w*_rtn_")
_VMEM_LOAD = re.compile(r"^(buffer|global|flat|scratch)_load")
_STORE = re.compile(r"^(buffer|global|flat|scratch)_store|^ds_(write|store)")
_VMEM_ATOMIC = re.compile(r"^(buffer|global|flat)_atomic")


def _regs(operand: str) -> Set[Tuple[str, int]]:
    out: Set[Tuple[str, int]] = set()
    for m in _REG.finditer(operand):
        lo, hi = ((int(m.group(2)), int(m.group(3))) if m.group(2)
                  else (int(m.group(4)), int(m.group(4))))
        out |= {(m.group(1), r) for r in range(lo, hi + 1)}
    return out


def _split_operands(rest: str) -> List[str]:
    return [o.strip() for o in rest.split(",")] if rest.strip() else []


@dataclass
class Inst:
    addr: int
    text: str

    @property
    def mnem(self) -> str:
        return self.text.partition(" ")[0]

    @property
    def ops(self) -> List[str]:
        return _split_operands(self.text.partition(" ")[2])


def _branch_target(ins: Inst, next_addr: int) -> Optional[int]:
    """Where an s_branch / s_cbranch_* goes: the next instruction's address
    plus its signed 16-bit dword offset."""
    if not (ins.mnem == "s_branch" or ins.mnem.startswith("s_cbranch")):
        return None
    m = re.match(r"(-?\d+)", ins.text.partition(" ")[2].strip())
    if not m:
        return None
    return next_addr + 4 * int(m.group(1))


class _Walker:
    """Loads in flight, as two in-order queues, and the reads that beat them."""

    def __init__(self, stores_in_vmcnt: bool, limit: int):
        self.stores_in_vmcnt = stores_in_vmcnt
        self.limit = limit
        self.found: List[str] = []
        self.benign: List[str] = []
        self.prev: Optional[str] = None

    def step(self, st, ins: Inst):
        st = self._step(st, ins)
        self.prev = ins.text
        return st

    def _step(self, st, ins: Inst):
        lds, vmem = st
        mnem, ops, rest = ins.mnem, ins.ops, ins.text.partition(" ")[2]
        if mnem == "s_endpgm":
            return [], []
        if mnem.startswith("s_wait"):
            for kind, q in (("lgkmcnt", 0), ("dscnt", 0), ("vmcnt", 1), ("loadcnt", 1)):
                m = re.search(rf"{kind}\((\d+)\)", rest) or (
                    re.match(r"(\d+)", rest.strip()) if mnem == f"s_wait_{kind}" else None)
                if m:
                    keep = int(m.group(1))
                    if q == 0:
                        lds = lds[len(lds) - keep:] if keep else []
                    else:
                        vmem = vmem[len(vmem) - keep:] if keep else []
            return lds, vmem
        if mnem.startswith("s_"):
            return lds, vmem
        atomic_ret = bool(_VMEM_ATOMIC.match(mnem) and re.search(r"\b(glc|sc0)\b", rest))
        # an atomic without a return reads all its operands, as a store does
        store = bool(_STORE.match(mnem)) or bool(_VMEM_ATOMIC.match(mnem) and not atomic_ret)
        srcs: Set = set()
        for o in (ops if store else ops[1:]):
            srcs |= _regs(o)
        if srcs:
            for q in (lds, vmem):
                for dst, load in q:
                    if dst & srcs:
                        h = f"{ins.text}   <- {load}"
                        # A memory op taking its address from the load just
                        # issued reads the old value: the load cannot have
                        # returned one instruction later. Latency, not a
                        # contract -- an asm output that should have been
                        # early-clobber (=&v) -- so it is reported apart.
                        out = (self.benign if load == self.prev and
                               (_LDS_LOAD.match(mnem) or _VMEM_LOAD.match(mnem))
                               else self.found)
                        if h not in out and len(out) < self.limit:
                            out.append(h)
                        break
        if not store and ops:
            # A write to a register a load is still filling never comes out
            # of a compiler; it is the walk crossing a path that did not run.
            dst = _regs(ops[0])
            if dst and not (_LDS_LOAD.match(mnem) or _VMEM_LOAD.match(mnem)):
                lds = [(d - dst, l) for d, l in lds]
                vmem = [(d - dst, l) for d, l in vmem]
        if _LDS_LOAD.match(mnem) and ops:
            lds = lds + [(_regs(ops[0]), ins.text)]
        elif mnem.startswith("ds_"):
            # every LDS op takes its place in lgkmcnt's order, stores and atomics too. Left out,
            # a compiler's lgkmcnt(11) after a run of LDS stores read as leaving eleven loads in
            # flight (a false alarm on every RDNA GEMM that stages through LDS).
            lds = lds + [(set(), ins.text)]
        elif _VMEM_LOAD.match(mnem) and ops and re.search(r"\blds\b", rest):
            # gfx950's DMA into LDS writes no register but takes its place in
            # vmcnt's order: without it, a load into registers issued before
            # the copies looks unretired by the wait that does retire it.
            vmem = vmem + [(set(), ins.text)]
        elif _VMEM_LOAD.match(mnem) and ops:
            vmem = vmem + [(_regs(ops[0]), ins.text)]
        elif _VMEM_ATOMIC.match(mnem) and ops:
            # an atomic takes a place in vmcnt's order; one that returns (glc / sc0) also
            # writes its first operand. Left out, the compiler's vmcnt(n) after a run of
            # atomics read as too weak a wait (a false alarm on every scatter-add).
            if atomic_ret or self.stores_in_vmcnt:
                vmem = vmem + [(_regs(ops[0]) if atomic_ret else set(), ins.text)]
        elif self.stores_in_vmcnt and mnem.startswith(("buffer_store", "global_store",
                                                       "flat_store", "scratch_store")):
            vmem = vmem + [(set(), ins.text)]
        return lds, vmem


def wait_hazards(insts: Sequence[Inst], arch: str = "", limit: int = 50,
                 benign: Optional[List[str]] = None) -> List[str]:
    """Reads of a VGPR/AGPR a load is still writing, as 'reader  <- load'.

    One walk in program order, then each loop (a backward branch) walked
    again from the state at its branch, so a load at a loop's bottom that is
    consumed at its top is seen."""
    w = _Walker(stores_in_vmcnt=arch.startswith("gfx9"), limit=limit)
    st = ([], [])
    at_branch = {}
    index = {ins.addr: i for i, ins in enumerate(insts)}
    for i, ins in enumerate(insts):
        st = w.step(st, ins)
        nxt = insts[i + 1].addr if i + 1 < len(insts) else ins.addr + 4
        tgt = _branch_target(ins, nxt)
        if tgt is not None and tgt <= ins.addr and tgt in index:
            at_branch[i] = (index[tgt], st)
    for j, (i, st) in at_branch.items():
        w.prev = None
        for ins in insts[i:j + 1]:
            st = w.step(st, ins)
    if benign is not None:
        benign.extend(w.benign)
    return w.found


def exec_hazards(insts: Sequence[Inst], limit: int = 50) -> List[str]:
    """MFMAs issued under a partial EXEC: between an `s_and_saveexec` that opens a
    lane-divergent `if` and the `s_or exec, exec, ..` that closes it.

    The MFMA reads all 64 lanes' operands; an operand computed in the same block (a copy out of
    AGPRs, a conversion) was written in the active lanes only, so the rest are stale and the
    product is wrong for every row (K1's scatter-add stage 2 on gfx950: 0.28 off). No source
    puts an MFMA there. A compiler sinks it into the only block that uses its result:
    roc-7.2.3's MachineSink does that to V_MFMA_SCALE, ROCm 10's does not. Reported whether or
    not an operand happens to be computed there. The scan counts regions linearly, so divergent
    loops (closed by an `s_or` with no `s_and_saveexec`) are not seen. A warning: the scan
    cannot tell a condition that is uniform at run time (hk's `_live` branches) from a
    per-lane one."""
    found: List[str] = []
    depth, opened = 0, ""
    for ins in insts:
        m = ins.mnem
        if m.startswith("s_and_saveexec"):
            depth += 1
            opened = ins.text
        elif m in ("s_or_b64", "s_or_b32") and ins.ops[:2] in (["exec", "exec"],
                                                             ["exec_lo", "exec_lo"]):
            depth = max(depth - 1, 0)
        elif depth and m.startswith("v_mfma") and len(found) < limit:
            found.append(f"{ins.text}   <- under a partial EXEC: {opened}")
    return found


def disassemble(co: Path) -> Dict[str, List[Inst]]:
    """Per function symbol, its instructions with their addresses."""
    out = subprocess.run([_tool("llvm-objdump"), "-d", str(co)],
                         capture_output=True, text=True, check=True).stdout
    funcs: Dict[str, List[Inst]] = {}
    cur: Optional[List[Inst]] = None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            cur = funcs.setdefault(m.group(1), [])
            continue
        if cur is None:
            continue
        text, _, comment = line.partition("//")
        text = text.strip()
        a = re.match(r"\s*([0-9A-Fa-f]+):", comment)
        if text and a and re.match(r"^[a-z]", text):
            cur.append(Inst(int(a.group(1), 16), re.sub(r"\s+", " ", text)))
    return funcs


def check(co: Path) -> List[Kernel]:
    ks = resources(co)
    code = disassemble(co)
    for name, k in ks.items():
        insts = code.get(name, [])
        k.hazards = wait_hazards(insts, k.arch, benign=k.warnings)
        k.warnings.extend(exec_hazards(insts))
    return list(ks.values())


# ---------------------------------------------------------------- fat binaries


def unbundle(host: Path, outdir: Path) -> List[Path]:
    """The amdgcn code objects of a host binary's embedded HIP fat binary."""
    fat = outdir / "fatbin"
    subprocess.run([_tool("llvm-objcopy"), f"--dump-section=.hip_fatbin={fat}", str(host),
                    str(outdir / "stripped")], check=True, capture_output=True)
    lst = subprocess.run([_tool("clang-offload-bundler"), "--list", "--type=o",
                          f"--input={fat}"], capture_output=True, text=True, check=True).stdout
    outs = []
    for i, tgt in enumerate(t for t in lst.split() if "amdgcn" in t):
        o = outdir / f"co{i}.co"
        subprocess.run([_tool("clang-offload-bundler"), "--unbundle", "--type=o",
                        f"--targets={tgt}", f"--input={fat}", f"--output={o}"], check=True)
        outs.append(o)
    return outs


def lds_copy_hazards(binary: Path, bundled: bool = True) -> List[str]:
    """What hk.runtime.compile.build refuses: an instruction that reads a register an LDS load
    is still filling, before the s_waitcnt that retires the load, as 'reader  <- load'.

    hk's LDS reads are inline asm, and LLVM takes an asm output as ready when the asm ends. hk
    places the lgkmcnt wait itself and binds the fragment after it, so nothing computes on it
    early. But under register pressure the compiler may copy the destination right after the
    read, before hk's wait: a v_mov to assemble an operand tuple, or a v_accvgpr_write to park it
    in an AGPR. The copy then holds whatever the registers had, and the kernel is silently wrong
    with ScratchSize 0. Measured under ROCm 10 (hipcc 23):
    - an MXFP8 GEMM with byte scales, 128x256 at K=1152, was 0.83 off;
    - hk.ops.gemm_cdna_dp's 128x256 setprio split-K configurations hold 24 such copies.
    Only loads from LDS count: vector-memory loads here are the compiler's own, and it waits for
    them (a report on one is this linear walk crossing a path that did not run). `bundled`: a host
    .so carrying a HIP fat binary, else one code object."""
    with tempfile.TemporaryDirectory(prefix="hk-cocheck-") as d:
        cos = unbundle(binary, Path(d)) if bundled else [binary]
        found: List[str] = []
        for co in cos:
            code = disassemble(co)
            for name, k in resources(co).items():
                for h in wait_hazards(code.get(name, []), k.arch):
                    if "<- ds_" in h:
                        found.append(f"{name}: {h}")
        return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hk.runtime.cocheck",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--bundle", action="store_true",
                    help="the files are host binaries with an embedded HIP fat binary")
    a = ap.parse_args(argv)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in a.files:
            cos = unbundle(f, Path(tmp)) if a.bundle else [f]
            for co in cos:
                ks = check(co)
                print(f"{f}{'' if co == f else ' [' + co.name + ']'}: {len(ks)} kernel(s), "
                      f"{ks[0].arch if ks else '?'}")
                for k in ks:
                    print("  " + k.line())
                    bad += not k.ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
