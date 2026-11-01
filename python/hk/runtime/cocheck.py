"""hk's gates, on any AMDGPU code object: resources, waits, and MFMA hazards.

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
* **MFMA hazards.** An instruction that reads or overwrites an MFMA's result
  (or overwrites a register it reads as srcC) fewer wait states after it,
  on some path into it, than the MFMA's window: `mfmahaz`. LLVM before
  llvm/llvm-project#225989 padded the shortest path into a block too
  little when it found a longer one first (ROCm 7.2.3 to 10.1); a gfx950
  paged decode computed wrong scores from it (ROCm/llvm-project#5153).
* **VOPD interlock hazard (gfx11.0, gfx11.5).** A dual-issue instruction
  (`v_dual_X :: v_dual_Y`) whose X and Y SRC0, or VSRC1, are distinct VGPRs
  with the same parity: `vopd_parity_hazard`. The forwarding drops the
  interlock and an operand written just before is read stale; LLVM fixed the
  pairing in llvm/llvm-project#220348, which no ROCm release has yet.
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

Exits 1 if any kernel fails a gate.
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..target import TARGETS


def _tool(name: str) -> str:
    """ROCm's LLVM tool `name`, where hipcc's clang is (compile.llvm_tool); FileNotFoundError
    naming where it looked if it is nowhere. llc may be missing (TheRock ships none): its
    callers probe with it and get "llc" back, which does not exist, and say so."""
    from .compile import llvm_tool  # noqa: PLC0415

    if name == "llc":
        return llvm_tool(name, required=False) or name
    return llvm_tool(name)


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
    #: MFMA results touched too early on some path (mfmahaz).
    mfma: List[str] = field(default_factory=list)
    #: VOPDs with gfx11's interlock hazard (vopd_parity_hazard).
    vopd: List[str] = field(default_factory=list)
    #: Reads of a load's register by the next instruction's address: the old
    #: value, by issue order; and MFMAs under a partial EXEC (exec_hazards).
    #: Warnings, not failures.
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.scratch or self.vgpr_spill or self.sgpr_spill or self.hazards
                    or self.mfma or self.vopd)

    def line(self) -> str:
        occ = f" occ={self.occupancy}" if self.occupancy is not None else ""
        agpr = f" ({self.agpr} AGPR)" if self.agpr else ""
        status = "OK  " if self.ok else "FAIL"
        out = (f"{status} {self.name}: vgpr={self.vgpr}{agpr} sgpr={self.sgpr}"
               f" scratch={self.scratch} spill={self.vgpr_spill}/{self.sgpr_spill}{occ}"
               f" wait-hazards={len(self.hazards)}"
               + (f" mfma-hazards={len(self.mfma)}" if self.mfma else "")
               + (f" vopd-pairs={len(self.vopd)}" if self.vopd else "")
               + (f" warnings={len(self.warnings)}" if self.warnings else ""))
        for h in self.hazards[:3]:
            out += f"\n       {h}"
        for h in self.mfma[:3]:
            out += f"\n       {h}"
        for h in self.vopd[:3]:
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
#: Distinct in-flight states the wait scan follows into one instruction (paths that differ in
#: what they have in flight); past it, the first ones.
_MAX_PATHS = 16


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


def _wait_keeps(mnem: str, rest: str) -> Tuple[Optional[int], Optional[int]]:
    """How many LDS and vector-memory loads a wait leaves in flight; None for a
    counter it does not wait on.

    gfx9-11 spell one s_waitcnt with named fields, `lgkmcnt(n)` and `vmcnt(n)`.
    gfx12 splits the counters into their own instructions, an immediate each
    (`s_wait_dscnt 0x3`, `s_wait_loadcnt 0x0`), plus two combined forms whose
    immediate packs loadcnt or storecnt in bits 13:8 and dscnt in bits 5:0
    (`s_wait_loadcnt_dscnt 0x302`: three loads, two LDS ops). The immediates
    are hex; reading `0x3` as its leading digit would make every wait a full
    drain and hide every hazard behind it."""
    lds = vmem = None
    m = re.search(r"lgkmcnt\((\d+)\)", rest)
    if m:
        lds = int(m.group(1))
    m = re.search(r"vmcnt\((\d+)\)", rest)
    if m:
        vmem = int(m.group(1))
    imm = rest.strip().split(None, 1)[0].rstrip(",") if rest.strip() else ""
    try:
        v = int(imm, 0)
    except ValueError:
        return lds, vmem
    if mnem == "s_wait_dscnt":
        lds = v
    elif mnem == "s_wait_loadcnt":
        vmem = v
    elif mnem == "s_wait_loadcnt_dscnt":
        vmem, lds = (v >> 8) & 0x3F, v & 0x3F
    elif mnem == "s_wait_storecnt_dscnt":
        lds = v & 0x3F
    return lds, vmem


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
            keep_lds, keep_vmem = _wait_keeps(mnem, rest)
            if keep_lds is not None:
                lds = lds[len(lds) - keep_lds:] if keep_lds else []
            if keep_vmem is not None:
                vmem = vmem[len(vmem) - keep_vmem:] if keep_vmem else []
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

    One walk in program order over the states that reach each instruction: a
    forward branch carries its state to its target, and the instruction after
    an s_branch is reached by jumps only (walking straight on from the code
    before it reported that code's loads against a block they never reach).
    Then each loop (a backward branch) walked again from the states at its
    branch, so a load at a loop's bottom that is consumed at its top is seen."""
    w = _Walker(stores_in_vmcnt=arch.startswith("gfx9"), limit=limit)

    def key(st):
        return tuple(tuple((tuple(sorted(d)), l) for d, l in q) for q in st)

    def merge(sts):
        out = {}
        for st in sts:
            out.setdefault(key(st), st)
        return list(out.values())[:_MAX_PATHS]

    def step(sts, ins):
        prev = w.prev
        out = []
        for st in sts:
            w.prev = prev
            out.append(w._step(st, ins))
        w.prev = ins.text
        return merge(out)

    sts = [([], [])]
    at_branch = {}
    ahead: Dict[int, list] = {}
    index = {ins.addr: i for i, ins in enumerate(insts)}
    for i, ins in enumerate(insts):
        if ins.addr in ahead:
            sts = merge(sts + ahead.pop(ins.addr))
        sts = step(sts, ins)
        nxt = insts[i + 1].addr if i + 1 < len(insts) else ins.addr + 4
        tgt = _branch_target(ins, nxt)
        if tgt is not None and tgt > ins.addr:
            ahead.setdefault(tgt, []).extend(sts)
        if tgt is not None and tgt <= ins.addr and tgt in index:
            at_branch[i] = (index[tgt], sts)
        if ins.mnem == "s_branch":
            sts = []
    for j, (i, sts) in at_branch.items():
        w.prev = None
        for ins in insts[i:j + 1]:
            sts = step(sts, ins)
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
        k.mfma = mfma_hazards_in(insts, k.arch)
        k.vopd = [i.text for i in insts if vopd_parity_hazard(i)] if k.arch.startswith(
            VOPD_HAZARD_ARCHS) else []
        k.warnings.extend(exec_hazards(insts))
    return list(ks.values())


# ---------------------------------------------------------------- scratch frames

#: The reason the spill gate records for a frame it lets through.
UNUSED_FRAME = "reserved-unused scratch frame"
_FLAT_MEM = re.compile(r"^flat_(load|store|atomic)")


def kernel_descriptor(co: Path, name: str) -> Dict[str, int]:
    """The scratch fields of kernel `name`'s descriptor (`name.kd`, 64 bytes): the private
    segment size, compute_pgm_rsrc2's SCRATCH_EN, and kernel_code_properties' enables of the
    private segment buffer and flat scratch init user SGPRs."""
    syms = subprocess.run([_tool("llvm-readelf"), "-sW", str(co)], capture_output=True, text=True,
                          check=True).stdout
    secs = subprocess.run([_tool("llvm-readelf"), "-SW", str(co)], capture_output=True, text=True,
                          check=True).stdout
    val = ndx = None
    for line in syms.splitlines():
        p = line.split()
        if len(p) >= 8 and p[-1] == name + ".kd":
            val, ndx = int(p[1], 16), int(p[6])
    if val is None:
        raise ValueError(f"no kernel descriptor {name}.kd in {co}")
    base = None
    for line in secs.splitlines():
        m = re.match(r"\s*\[\s*(\d+)\]\s+\S+\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m and int(m.group(1)) == ndx:
            base = int(m.group(3), 16) - int(m.group(2), 16)
    if base is None:
        raise ValueError(f"no section {ndx} for {name}.kd in {co}")
    raw = co.read_bytes()[base + val: base + val + 64]
    props = struct.unpack_from("<H", raw, 56)[0]
    return {"private_segment_fixed_size": struct.unpack_from("<I", raw, 4)[0],
            "scratch_en": struct.unpack_from("<I", raw, 52)[0] & 1,
            "private_segment_buffer": props & 1,
            "flat_scratch_init": (props >> 5) & 1}


def frame_verdict(insts: Sequence[str], kd: Dict[str, int]) -> Tuple[bool, str]:
    """Whether a kernel's scratch frame is reserved and never accessed: every one of
    - no scratch_* instruction;
    - no private segment buffer user SGPRs (so no buffer_* can address the private segment);
    - no flat scratch: neither its init user SGPRs nor an operand naming flat_scratch;
    - no flat_* memory instruction (a flat address may be private: without an address analysis
      every one counts).
    Returns (unused, why not)."""
    for text in insts:
        op = text.split(" ", 1)[0]
        if op.startswith("scratch_"):
            return False, f"scratch access: {text}"
        if _FLAT_MEM.match(op):
            return False, f"flat access (may be private): {text}"
        if "flat_scratch" in text:
            return False, f"flat_scratch operand: {text}"
    if kd.get("private_segment_buffer"):
        return False, "the private segment buffer user SGPRs are enabled"
    if kd.get("flat_scratch_init"):
        return False, "flat scratch init user SGPRs are enabled"
    return True, ""


def unused_scratch_frame(binary: Path, bundled: bool = True) -> Dict[str, Tuple[bool, str]]:
    """Per kernel symbol of a binary: frame_verdict on its code and descriptor."""
    with tempfile.TemporaryDirectory(prefix="hk-cocheck-") as d:
        cos = unbundle(binary, Path(d)) if bundled else [binary]
        out: Dict[str, Tuple[bool, str]] = {}
        for co in cos:
            code = disassemble(co)
            for name in resources(co):
                out[name] = frame_verdict([i.text for i in code.get(name, [])],
                                          kernel_descriptor(co, name))
        return out


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


def mfma_hazards_in(insts: Sequence[Inst], arch: str) -> List[str]:
    """One function's MFMA hazards (mfmahaz.deficits), reported."""
    from . import mfmahaz  # noqa: PLC0415

    prog = mfmahaz.from_objdump(insts)
    return mfmahaz.report(prog, mfmahaz.deficits(prog, arch))


def mfma_hazards(binary: Path, bundled: bool = True) -> List[str]:
    """What hk.runtime.compile.build refuses on CDNA: an MFMA result read or
    overwritten too few wait states after the MFMA on some path, as
    'kernel: where: reader ... needs N more'. `bundled` as lds_copy_hazards."""
    with tempfile.TemporaryDirectory(prefix="hk-cocheck-") as d:
        cos = unbundle(binary, Path(d)) if bundled else [binary]
        found: List[str] = []
        for co in cos:
            code = disassemble(co)
            for name, k in resources(co).items():
                found += [f"{name}: {h}" for h in mfma_hazards_in(code.get(name, []), k.arch)]
        return found


#: The targets with the VOPD interlock hazard: gfx11.0 and gfx11.5 (not gfx11.7, not gfx12).
VOPD_HAZARD_ARCHS = ("gfx110", "gfx115", "gfx11-generic")
_VGPR = re.compile(r"^v(\d+)$")


def _vopd_sources(half: str) -> List[str]:
    """SRC0 and VSRC1 of one VOPD component (`v_dual_add_f32 v0, v14, v22` -> [v14, v22]):
    fmamk carries its literal between them, fmaak after them."""
    op, _, rest = half.partition(" ")
    srcs = _split_operands(rest)[1:]
    if op.endswith("fmamk_f32"):
        return [srcs[0], srcs[2]] if len(srcs) > 2 else srcs[:1]
    return srcs[:2]


def vopd_parity_hazard(ins: Inst) -> Optional[str]:
    """Why a VOPD instruction (`v_dual_X ... :: v_dual_Y ...`) has gfx11's VOPD interlock hazard,
    or None: X's and Y's SRC0, or their VSRC1, are distinct VGPRs with the same register-number
    parity. The destination-buffer forwarding then drops the interlock for one of them, and an
    operand written just before is read stale. LLVM requires only different banks (the number mod
    4) there; llvm/llvm-project#220348 requires different parity on gfx11.0 and gfx11.5, and no
    ROCm release has it yet. On gfx1100 under a partial EXEC, 88-93 of 100 launches of the window
    in tools/rdna3-probes/vopd_interlock_parity.hip read a wrong value, and none with the parities
    made different; a bounded strip RMSNorm computed wrong values from one such pair."""
    x, sep, y = ins.text.partition(" :: ")
    if not sep or not x.startswith("v_dual_"):
        return None
    for i, (a, b) in enumerate(zip(_vopd_sources(x), _vopd_sources(y))):
        ma, mb = _VGPR.match(a), _VGPR.match(b)
        if ma and mb and ma.group(1) != mb.group(1) and \
                int(ma.group(1)) % 2 == int(mb.group(1)) % 2:
            return f"X and Y {'VSRC1' if i else 'SRC0'} {a} and {b} have the same parity"
    return None


def vopd_hazards(binary: Path, bundled: bool = True) -> List[str]:
    """What hk.runtime.compile.build rebuilds on gfx11.0 / gfx11.5: every VOPD with the interlock
    hazard (`vopd_parity_hazard`), as 'kernel: instruction   <- why'. `bundled` as
    lds_copy_hazards."""
    with tempfile.TemporaryDirectory(prefix="hk-cocheck-") as d:
        cos = unbundle(binary, Path(d)) if bundled else [binary]
        found: List[str] = []
        for co in cos:
            for name, insts in disassemble(co).items():
                for i in insts:
                    why = vopd_parity_hazard(i)
                    if why:
                        found.append(f"{name}: {i.text}   <- {why}")
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
