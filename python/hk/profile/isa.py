"""The code object's instructions: where the main loop is, and what is in it.

No GPU needed. `code_object(path)` takes the bytes of a kernel's AMDGPU ELF
out of what hk built -- a device-only `kernel.co`, or the offload bundle in
a host extension's `.hip_fatbin` section -- `disassemble()` runs ROCm's
llvm-objdump on it, and `analyze()` splits each kernel's instructions into
the prologue, the loops (a branch back to an earlier address) and what
follows the last loop, and counts each part's instruction mix by class
(`CLASSES`).

A thread trace (hk.profile's ATT pass) adds, per instruction, how often it
ran and the cycles a wave spent on it -- `analyze(..., att=...)`; the main
loop is then the loop the waves spent the most cycles in, not the one with
the most matrix instructions in its text.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Instruction classes, in the order a mix is printed. Every instruction is
#: exactly one of them (`classify`).
CLASSES = (
    "mfma",          # v_mfma_*, v_smfmac_*: the CDNA matrix core
    "wmma",          # v_wmma_*, v_swmmac_*: the RDNA matrix instructions
    "valu",          # every other v_* (including v_accvgpr moves)
    "valu_trans",    # v_exp/log/rcp/rsq/sqrt/sin/cos: the transcendental unit
    "salu",          # scalar ALU
    "smem",          # s_load/s_buffer_load/s_store: scalar memory
    "vmem_load",     # buffer/global/flat/scratch loads into VGPRs
    "vmem_lds",      # buffer/global loads straight into LDS (gfx950's DMA)
    "vmem_store",    # buffer/global/flat/scratch stores
    "vmem_atomic",   # buffer/global/flat atomics
    "ds_read",       # LDS -> VGPRs
    "ds_write",      # VGPRs -> LDS
    "ds_other",      # ds_swizzle, ds_bpermute, LDS atomics
    "waitcnt",       # s_waitcnt* (split by counter in `waits`)
    "barrier",       # s_barrier*
    "branch",        # s_branch, s_cbranch_*, s_setpc/swappc
    "nop",           # s_nop, v_nop: hazard padding
    "other",         # s_setprio, s_delay_alu, s_sendmsg, s_endpgm, ...
)

_TRANS = re.compile(r"^v_(exp|log|rcp|rsq|sqrt|sin|cos)_")
_SMEM = re.compile(r"^s_(load|buffer_load|store|buffer_store|dcache|memtime|memrealtime|"
                   r"scratch_load|scratch_store|atc_probe|prefetch_data)")
_VMEM = re.compile(r"^(buffer|global|flat|scratch|tbuffer|image)_(\w+)")
_BRANCH = re.compile(r"^s_(branch|cbranch_\w+|setpc_b64|swappc_b64|call_b64)$")
_MMA_SHAPE = re.compile(r"_(\d+)x(\d+)x(\d+)")


def classify(mnemonic: str, operands: str = "") -> str:
    """The class (`CLASSES`) of one instruction, from its mnemonic (and, for
    a buffer load, whether it writes LDS)."""
    m = mnemonic
    if m.startswith(("v_mfma", "v_smfmac")):
        return "mfma"
    if m.startswith(("v_wmma", "v_swmmac")):
        return "wmma"
    if m == "v_nop" or m == "s_nop":
        return "nop"
    if m.startswith("v_"):
        return "valu_trans" if _TRANS.match(m) else "valu"
    if m.startswith("ds_"):
        if m.startswith(("ds_read", "ds_load")):
            return "ds_read"
        if m.startswith(("ds_write", "ds_store")):
            return "ds_write"
        return "ds_other"
    vm = _VMEM.match(m)
    if vm:
        op = vm.group(2)
        if "atomic" in op:
            return "vmem_atomic"
        if op.startswith(("store", "format_store")) or "_store" in m:
            return "vmem_store"
        if re.search(r"(^|\s)lds(\s|$)", operands):
            return "vmem_lds"
        return "vmem_load"
    if m.startswith("s_waitcnt"):
        return "waitcnt"
    if m.startswith("s_barrier"):
        return "barrier"
    if _BRANCH.match(m):
        return "branch"
    if _SMEM.match(m):
        return "smem"
    if m.startswith("s_"):
        if m.startswith(("s_setprio", "s_delay_alu", "s_sendmsg", "s_endpgm", "s_sleep",
                         "s_sched", "s_clause", "s_inst_prefetch", "s_icache", "s_trap",
                         "s_set_inst_prefetch", "s_ttrace", "s_code_end", "s_wait_",
                         "s_version", "s_setreg", "s_getreg", "s_denorm", "s_round")):
            return "other"
        return "salu"
    return "other"


def mma_flops(mnemonic: str, wave: int = 64) -> int:
    """FLOPs (multiply + add, so 2 a product) one wave's matrix instruction
    does: 2*M*N*K from its name (v_mfma_f32_16x16x32_bf16 -> 16384,
    v_wmma_f32_16x16x16_bf16 -> 8192). 0 for anything else. The legacy
    multi-block MFMAs (4x4x4 with 16 blocks, 32x32x2 with 2) are counted by
    their block count."""
    if not mnemonic.startswith(("v_mfma", "v_smfmac", "v_wmma", "v_swmmac")):
        return 0
    m = _MMA_SHAPE.search(mnemonic)
    if not m:
        return 0
    a, b, k = (int(x) for x in m.groups())
    blocks = 1
    if mnemonic.startswith("v_mfma"):
        tail = mnemonic[m.end():].lstrip("_")
        if a == 4 and b == 4:
            blocks = 16
        elif (a, b) in ((32, 32), (16, 16)) and _MULTI_K.get(tail) == k:
            blocks = 2 if a == 32 else 4
    return 2 * a * b * k * blocks


#: The legacy MFMAs that compute several independent blocks rather than
#: reduce along K: by dtype suffix, the K at which a 32x32 one is 2 blocks
#: and a 16x16 one 4 (CDNA ISA, "MFMA" tables). The gfx950 ones
#: (v_mfma_f32_16x16x32_bf16, ...) are all single-block.
_MULTI_K = {"f32": 1, "f16": 4, "bf16": 2, "bf16_1k": 4, "i8": 4}


# ------------------------------------------------------------------ waits

_CNT = re.compile(r"(vmcnt|lgkmcnt|expcnt|vscnt|vm_vsrc|va_vdst)\((\d+)\)")


def wait_kind(mnemonic: str, operands: str) -> str:
    """What an s_waitcnt waits for: "vmem" (vmcnt or vscnt: global loads /
    stores, and gfx950's DMA into LDS), "lgkm" (lgkmcnt: LDS, scalar loads,
    messages), "vmem+lgkm" (both at once), "exp" or "other"."""
    if mnemonic in ("s_waitcnt_vscnt", "s_waitcnt_vmcnt"):
        return "vmem"
    if mnemonic in ("s_waitcnt_lgkmcnt",):
        return "lgkm"
    if mnemonic == "s_waitcnt_expcnt":
        return "exp"
    cnts = {k for k, _ in _CNT.findall(operands)}
    if not cnts and mnemonic == "s_waitcnt" and operands.strip() in ("0", "0x0"):
        return "vmem+lgkm"          # s_waitcnt 0: everything
    vm = bool(cnts & {"vmcnt", "vscnt"})
    lg = "lgkmcnt" in cnts
    if vm and lg:
        return "vmem+lgkm"
    if vm:
        return "vmem"
    if lg:
        return "lgkm"
    if "expcnt" in cnts:
        return "exp"
    return "other"


# ------------------------------------------------------------------ code objects


def _elf_sections(data: bytes) -> Dict[str, Tuple[int, int]]:
    """{name: (file offset, size)} of a 64-bit little-endian ELF."""
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    raw = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name, _typ, _flags, _addr, offset, size = struct.unpack_from("<IIQQQQ", data, off)
        raw.append((name, offset, size))
    stroff = raw[e_shstrndx][1]
    out = {}
    for name, offset, size in raw:
        end = data.index(b"\0", stroff + name)
        out[data[stroff + name:end].decode(errors="replace")] = (offset, size)
    return out


_BUNDLE = b"__CLANG_OFFLOAD_BUNDLE__"


def _unbundle(blob: bytes, arch: Optional[str]) -> bytes:
    """The device ELF for `arch` out of a clang offload bundle."""
    if blob[:4] == b"CCOB":
        raise ValueError("compressed offload bundle (CCOB): unbundle it with "
                         "clang-offload-bundler; hk builds uncompressed ones")
    if not blob.startswith(_BUNDLE):
        raise ValueError("no clang offload bundle in .hip_fatbin")
    n = struct.unpack_from("<Q", blob, len(_BUNDLE))[0]
    pos = len(_BUNDLE) + 8
    found = []
    for _ in range(n):
        off, size, tl = struct.unpack_from("<QQQ", blob, pos)
        pos += 24
        triple = blob[pos:pos + tl].decode(errors="replace")
        pos += tl
        if "amdgcn" in triple and size:
            found.append((triple, blob[off:off + size]))
    if not found:
        raise ValueError("the offload bundle has no amdgcn code object")
    for triple, co in found:
        if arch is None or triple.endswith(arch) or f"-{arch}" in triple:
            return co
    raise ValueError(f"no code object for {arch} in the bundle "
                     f"({', '.join(t for t, _ in found)})")


def code_object(path, arch: Optional[str] = None) -> bytes:
    """The AMDGPU code object hk built at `path`: a device-only `.co` as it
    is, or the one embedded in a host `.so`'s `.hip_fatbin`."""
    data = Path(path).read_bytes()
    secs = _elf_sections(data)
    if ".hip_fatbin" in secs:
        off, size = secs[".hip_fatbin"]
        return _unbundle(data[off:off + size], arch)
    return data


def objdump() -> Optional[str]:
    """ROCm's llvm-objdump (found as hk.runtime.compile.llvm_tool finds it), or None."""
    try:
        from ..runtime.compile import llvm_tool  # noqa: PLC0415

        return llvm_tool("llvm-objdump", required=False)
    except Exception:  # noqa: BLE001 -- no hipcc at all: PATH is all there is
        return shutil.which("llvm-objdump")


@dataclass
class Inst:
    """One instruction: its address in the code object, mnemonic, operands,
    class (`CLASSES`), and for a branch its target address."""

    addr: int
    mnemonic: str
    operands: str = ""
    cls: str = ""
    target: Optional[int] = None
    size: int = 4

    def text(self) -> str:
        return f"{self.mnemonic} {self.operands}".strip()


@dataclass
class Function:
    """A kernel's instructions, in address order."""

    name: str
    insts: List[Inst] = field(default_factory=list)


_FUNC = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:\s*$")
_LINE = re.compile(r"^\s+(\S+)\s*(.*?)\s*//\s*([0-9a-fA-F]+):\s*((?:[0-9A-Fa-f]{8}\s*)+)"
                   r"(?:<([^>+]+)(?:\+0x([0-9a-fA-F]+))?>)?\s*$")


def parse_objdump(text: str) -> List[Function]:
    """llvm-objdump -d's text -> [Function]."""
    funcs: List[Function] = []
    cur: Optional[Function] = None
    starts: Dict[str, int] = {}
    pending: List[Tuple[Inst, str, int]] = []
    for line in text.splitlines():
        m = _FUNC.match(line)
        if m:
            cur = Function(name=m.group(2))
            starts[cur.name] = int(m.group(1), 16)
            funcs.append(cur)
            continue
        if cur is None:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        mn, ops, addr, words, sym, off = m.groups()
        inst = Inst(addr=int(addr, 16), mnemonic=mn, operands=ops.strip(),
                    size=4 * len(words.split()))
        inst.cls = classify(mn, inst.operands)
        if inst.cls == "branch" and sym:
            pending.append((inst, sym, int(off, 16) if off else 0))
        cur.insts.append(inst)
    for inst, sym, off in pending:
        if sym in starts:
            inst.target = starts[sym] + off
    return funcs


def disassemble(co: bytes, arch: str) -> List[Function]:
    """The kernels in a code object, disassembled (needs llvm-objdump)."""
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    tool = objdump()
    if tool is None:
        raise FileNotFoundError(
            "llvm-objdump not found: not beside hipcc's clang, not in ROCM_PATH's llvm/bin, "
            "not on PATH. Set ROCM_PATH to the ROCm install that has llvm/bin/llvm-objdump.")
    fd, path = tempfile.mkstemp(suffix=".co")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(co)
        proc = subprocess.run([tool, "-d", f"--mcpu={arch}", path], capture_output=True,
                              text=True, timeout=120)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise RuntimeError(f"llvm-objdump failed: {proc.stderr.strip()[:400]}")
    return [f for f in parse_objdump(proc.stdout) if f.insts]


def from_att(code: List[list]) -> Tuple[List[Function], Dict[int, Dict[str, int]]]:
    """A thread trace's code.json rows -> ([Function], {addr: stats}).

    A row is [text, _, index, source/name, code object id, address, hits,
    latency, stall, idle]; a row whose text starts with ';' opens a kernel.
    The stats are summed over the traced waves: `hits` executions, `latency`
    the cycles from issue to the next instruction's, `stall` the cycles it
    waited to issue, `idle` the cycles the wave had nothing to issue."""
    funcs: List[Function] = []
    stats: Dict[int, Dict[str, int]] = {}
    cur: Optional[Function] = None
    for row in code:
        if not row:
            continue
        text = str(row[0])
        if text.startswith(";"):
            name = str(row[3]) if len(row) > 3 and row[3] else text.lstrip("; ").strip()
            cur = Function(name=name)
            funcs.append(cur)
            continue
        if len(row) < 10:
            continue
        addr = int(row[5])
        parts = text.split(None, 1)
        mn, ops = parts[0], (parts[1] if len(parts) > 1 else "")
        inst = Inst(addr=addr, mnemonic=mn, operands=ops, cls=classify(mn, ops))
        if inst.cls == "branch":
            inst.target = _att_target(mn, ops, addr)
        if cur is None:
            cur = Function(name="?")
            funcs.append(cur)
        cur.insts.append(inst)
        stats[addr] = {"hits": int(row[6]), "latency": int(row[7]), "stall": int(row[8]),
                       "idle": int(row[9])}
    for f in funcs:
        f.insts.sort(key=lambda i: i.addr)
        for a, b in zip(f.insts, f.insts[1:]):
            a.size = max(4, b.addr - a.addr)
    return funcs, stats


def _att_target(mn: str, ops: str, addr: int) -> Optional[int]:
    """A SOPP branch's target from its printed simm16 (words past the next
    instruction), as the thread trace's disassembly prints it."""
    if mn in ("s_setpc_b64", "s_swappc_b64", "s_call_b64"):
        return None
    tok = ops.split()[0] if ops.split() else ""
    try:
        v = int(tok, 0)
    except ValueError:
        return None
    if v >= 0x8000:
        v -= 0x10000
    return addr + 4 + 4 * v


# ------------------------------------------------------------------ analysis


def _empty_mix() -> Dict[str, int]:
    return {c: 0 for c in CLASSES}


_WIDTH = (("dwordx4", 16), ("b128", 16), ("dwordx3", 12), ("b96", 12), ("dwordx2", 8),
          ("b64", 8), ("dword", 4), ("b32", 4), ("short", 2), ("u16", 2), ("i16", 2),
          ("b16", 2), ("d16", 2), ("byte", 1), ("u8", 1), ("i8", 1), ("b8", 1))


def access_bytes(mnemonic: str) -> Optional[int]:
    """Bytes one lane moves in a vector-memory or LDS instruction, from its
    mnemonic (buffer_load_dwordx4 -> 16, global_load_b64 -> 8,
    ds_read_b128 -> 16); None where it does not say."""
    m = mnemonic.split("_", 2)[-1] if mnemonic.startswith(("buffer_", "global_", "flat_",
                                                           "scratch_", "ds_")) else ""
    for tag, n in _WIDTH:
        if tag in m:
            return n
    return None


def mix(insts: Iterable[Inst]) -> Dict[str, Any]:
    """{class: count} over static instructions, the waits by what they wait
    for, the bytes a lane each memory instruction moves ({class: {bytes:
    count}}), and the matrix FLOPs one pass over them does (a wave's)."""
    out: Dict[str, Any] = _empty_mix()
    waits: Dict[str, int] = {}
    widths: Dict[str, Dict[int, int]] = {}
    flops = 0
    n = 0
    for i in insts:
        n += 1
        out[i.cls] += 1
        if i.cls == "waitcnt":
            k = wait_kind(i.mnemonic, i.operands)
            waits[k] = waits.get(k, 0) + 1
        if i.cls.startswith(("vmem_", "ds_")):
            b = access_bytes(i.mnemonic)
            if b:
                w = widths.setdefault(i.cls, {})
                w[b] = w.get(b, 0) + 1
        flops += mma_flops(i.mnemonic)
    res = {"instructions": n, "classes": {k: v for k, v in out.items() if v},
           "waits": waits, "mma_flops": flops}
    if widths:
        res["bytes_per_lane"] = {c: {str(b): v for b, v in sorted(w.items())}
                                 for c, w in widths.items()}
    mma = out["mfma"] + out["wmma"]
    if mma:
        # What a matrix instruction has to share its issue slots with: at
        # one wave a SIMD, every one of these is matrix-core idle time.
        res["per_matrix_instruction"] = round((n - mma) / mma, 3)
    return res


@dataclass
class Loop:
    start: int
    end: int           # address of the backward branch
    index: Tuple[int, int] = (0, 0)   # instruction indices [first, last]


def loops(f: Function) -> List[Loop]:
    """Every backward branch's range, outermost first."""
    pos = {i.addr: n for n, i in enumerate(f.insts)}
    out = []
    for n, i in enumerate(f.insts):
        if i.cls == "branch" and i.target is not None and i.target <= i.addr \
                and i.target in pos:
            out.append(Loop(start=i.target, end=i.addr, index=(pos[i.target], n)))
    out.sort(key=lambda lp: (lp.start, -lp.end))
    return out


def _att_sum(insts: Iterable[Inst], stats: Dict[int, Dict[str, int]]) -> Dict[str, Any]:
    """Thread-trace cycles over a range, total and by class / wait kind."""
    tot = {"hits": 0, "latency": 0, "stall": 0, "idle": 0}
    by: Dict[str, Dict[str, int]] = {}
    for i in insts:
        s = stats.get(i.addr)
        if not s:
            continue
        key = i.cls
        if i.cls == "waitcnt":
            key = "waitcnt:" + wait_kind(i.mnemonic, i.operands)
        b = by.setdefault(key, {"hits": 0, "latency": 0, "stall": 0, "idle": 0})
        for k in tot:
            tot[k] += s[k]
            b[k] += s[k]
    return {"total": tot, "by_class": by}


def analyze(f: Function, *, att: Optional[Dict[int, Dict[str, int]]] = None,
            waves_traced: Optional[int] = None) -> Dict[str, Any]:
    """One kernel's ISA summary (the JSON's `isa`): the whole kernel's mix,
    the loops and the main one, the prologue before it and the epilogue
    after it, and with a thread trace the cycles each part took."""
    ls = loops(f)
    res: Dict[str, Any] = {"function": f.name, "instructions": len(f.insts),
                           "total": mix(f.insts), "loops": []}

    def cycles(lo: int, hi: int) -> int:
        if not att:
            return 0
        return sum(att.get(i.addr, {}).get("latency", 0) for i in f.insts[lo:hi + 1])

    for lp in ls:
        a, b = lp.index
        body = f.insts[a:b + 1]
        ent = {"start": hex(lp.start), "end": hex(lp.end), "instructions": len(body),
               "mix": mix(body)}
        if att:
            hits = att.get(f.insts[b].addr, {}).get("hits", 0)
            ent["iterations"] = hits          # summed over the traced waves
            ent["cycles"] = cycles(a, b)
        res["loops"].append(ent)

    main = None
    if ls:
        if att:
            # Innermost-preferring: a loop's cycles include its inner loops',
            # so take the largest, then any loop inside it holding >= 80% of it.
            order = sorted(range(len(ls)), key=lambda j: -res["loops"][j]["cycles"])
            main = order[0]
            for j in order[1:]:
                if ls[j].start >= ls[main].start and ls[j].end <= ls[main].end and \
                        res["loops"][j]["cycles"] >= 0.8 * res["loops"][main]["cycles"]:
                    main = j
        else:
            def weight(j):
                m = res["loops"][j]["mix"]
                return (m["classes"].get("mfma", 0) + m["classes"].get("wmma", 0),
                        m["instructions"])
            main = max(range(len(ls)), key=weight)
    total_cycles = cycles(0, len(f.insts) - 1) if att else 0
    if main is not None:
        a, b = ls[main].index
        # Hot sibling loops (a K loop compiled in two copies, a tail loop):
        # the main loop is then the span from the first to the last of them.
        merged = [main]
        for j, lp in enumerate(ls):
            if j == main or (lp.index[0] <= a and lp.index[1] >= b) or \
                    (lp.index[0] >= a and lp.index[1] <= b):
                continue
            if att:
                hot = res["loops"][j]["cycles"] >= 0.25 * max(1, res["loops"][main]["cycles"])
            else:
                mm = res["loops"][main]["mix"]["classes"]
                mj = res["loops"][j]["mix"]["classes"]
                nmain = mm.get("mfma", 0) + mm.get("wmma", 0)
                hot = nmain > 0 and mj.get("mfma", 0) + mj.get("wmma", 0) >= 0.5 * nmain
            if hot:
                merged.append(j)
                a, b = min(a, lp.index[0]), max(b, lp.index[1])
        if len(merged) > 1:
            body = f.insts[a:b + 1]
            res["main_loop"] = {"start": hex(f.insts[a].addr), "end": hex(f.insts[b].addr),
                                "instructions": len(body), "mix": mix(body),
                                "which": merged, "merged_loops": len(merged)}
            if att:
                res["main_loop"]["cycles"] = cycles(a, b)
        else:
            res["main_loop"] = dict(res["loops"][main], which=main)
        # After the loop: a software pipeline's drain (the last tiles'
        # matrix instructions) and then the epilogue proper -- what runs
        # after the last matrix instruction: the accumulator's conversion,
        # a template's epilogue math, the stores.
        n = len(f.insts)
        last_mma = max((j for j in range(b + 1, n) if f.insts[j].cls in ("mfma", "wmma")),
                       default=None)
        e0 = (last_mma + 1) if last_mma is not None else b + 1
        parts = [("main_loop", a, b), ("prologue", 0, a - 1)]
        if last_mma is not None:
            parts.append(("drain", b + 1, last_mma))
        parts.append(("epilogue", e0, n - 1))
        for key, lo, hi in parts[1:]:
            body = f.insts[lo:hi + 1] if hi >= lo else []
            res[key] = {"instructions": len(body), "mix": mix(body)}
            if body:
                res[key]["start"] = hex(body[0].addr)
        if att:
            for key, lo, hi in parts:
                part = f.insts[lo:hi + 1] if hi >= lo else []
                s = _att_sum(part, att)
                res[key]["att"] = s
                res[key]["cycle_share"] = (round(s["total"]["latency"] / total_cycles, 4)
                                           if total_cycles else None)
    else:
        res["main_loop"] = None
        if att:
            s = _att_sum(f.insts, att)
            res["straight_line"] = {"att": s}
    if att:
        res["att"] = {"total_cycles": total_cycles, "waves_traced": waves_traced}
    return res


def regions(res: Dict[str, Any]) -> List[Tuple[int, str]]:
    """[(first address, part)] in address order from analyze()'s result:
    prologue, main_loop, drain, epilogue (whichever exist)."""
    out: List[Tuple[int, str]] = [(0, "prologue")]
    ml = res.get("main_loop")
    if not ml:
        return [(0, "kernel")]
    out.append((int(ml["start"], 16), "main_loop"))
    for part in ("drain", "epilogue"):
        if (res.get(part) or {}).get("start"):
            out.append((int(res[part]["start"], 16), part))
    out.sort()
    return out


def matrix_ops_per_wave(f: Function, att: Dict[int, Dict[str, int]],
                        waves_traced: int) -> Tuple[float, float]:
    """(matrix instructions, matrix FLOPs) one wave executes on average, from
    a thread trace's execution counts: what a counter cannot say on gfx1100,
    which has no WMMA counter."""
    if not waves_traced:
        return 0.0, 0.0
    n = fl = 0.0
    for i in f.insts:
        if i.cls in ("mfma", "wmma"):
            h = att.get(i.addr, {}).get("hits", 0)
            n += h
            fl += h * mma_flops(i.mnemonic)
    return n / waves_traced, fl / waves_traced


__all__ = ["CLASSES", "classify", "mma_flops", "wait_kind", "code_object", "disassemble",
           "parse_objdump", "from_att", "mix", "loops", "analyze", "Inst", "Function",
           "matrix_ops_per_wave", "objdump"]
