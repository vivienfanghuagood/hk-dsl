"""What hk2 knows about each GPU: wave size, LDS, buffer descriptors, and the matrix-core atoms.

Adding an architecture is adding a row here. Each matrix atom is an LLVM intrinsic plus the basis tables
of its A, B and C fragments (see ``hk2.layout``); the compiler builds every operand and accumulator
layout from these, so a kernel's source does not change between CDNA and RDNA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from . import types as T
from .layout import Basis, Layout, log2

__all__ = ["Arch", "MmaAtom", "get_arch", "ARCHES"]


@dataclass(frozen=True)
class MmaAtom:
    m: int
    n: int
    k: int
    dtype: T.DType          # operand element type
    intrinsic: str
    op_ty: str              # LLVM type of one operand fragment as the intrinsic takes it
    op_elt: str             # its element type ("bfloat", "i16", "half")
    c_ty: str               # accumulator fragment type
    a_reg: Tuple[Basis, ...]
    a_lane: Tuple[Basis, ...]
    b_reg: Tuple[Basis, ...]
    b_lane: Tuple[Basis, ...]
    c_reg: Tuple[Basis, ...]
    c_lane: Tuple[Basis, ...]
    tail: str = ""          # trailing immediate operands (cbsz, abid, blgp on MFMA)

    @property
    def a_regs(self) -> int:
        return 1 << len(self.a_reg)

    @property
    def b_regs(self) -> int:
        return 1 << len(self.b_reg)

    @property
    def c_regs(self) -> int:
        return 1 << len(self.c_reg)


@dataclass(frozen=True)
class Arch:
    name: str
    family: str             # "cdna" or "rdna"
    wave: int
    lds_bytes: int
    buffer_flags: int       # dword 3 of a raw buffer resource
    atoms: Dict[str, MmaAtom] = field(default_factory=dict)

    def mma(self, dtype: T.DType) -> MmaAtom:
        try:
            return self.atoms[dtype.name]
        except KeyError:
            raise NotImplementedError(f"hk2: no {dtype.name} matrix instruction on {self.name}") from None


R2 = lambda *bs: tuple(bs)

# ---------------------------------------------------------------- CDNA (wave64, MFMA)
# 16x16 accumulators: lane l holds rows 4*(l/16) + i (i < 4) of column l % 16.
_MFMA_C = dict(c_reg=R2((1, 0), (2, 0)), c_lane=R2((0, 1), (0, 2), (0, 4), (0, 8), (4, 0), (8, 0)), c_ty="<4 x float>",
               tail=", i32 0, i32 0, i32 0")


def _mfma(k, dtype, intrinsic, op_ty, op_elt):
    kv = k // 4  # K elements per lane
    a_reg = tuple((0, 1 << i) for i in range(log2(kv)))
    a_lane = R2((1, 0), (2, 0), (4, 0), (8, 0), (0, kv), (0, 2 * kv))
    b_reg = tuple((1 << i, 0) for i in range(log2(kv)))
    b_lane = R2((0, 1), (0, 2), (0, 4), (0, 8), (kv, 0), (2 * kv, 0))
    return MmaAtom(16, 16, k, dtype, intrinsic, op_ty, op_elt, a_reg=a_reg, a_lane=a_lane, b_reg=b_reg, b_lane=b_lane,
                   **_MFMA_C)


_GFX950 = {"bfloat16": _mfma(32, T.bf16, "llvm.amdgcn.mfma.f32.16x16x32.bf16", "<8 x bfloat>", "bfloat"),
           "float16": _mfma(32, T.f16, "llvm.amdgcn.mfma.f32.16x16x32.f16", "<8 x half>", "half")}
_GFX942 = {"bfloat16": _mfma(16, T.bf16, "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", "<4 x i16>", "i16"),
           "float16": _mfma(16, T.f16, "llvm.amdgcn.mfma.f32.16x16x16f16", "<4 x half>", "half")}

# ---------------------------------------------------------------- RDNA3 (wave32, WMMA)
# A and B: lanes 16-31 repeat lanes 0-15, each lane holds all 16 K. C: lane l holds rows 2i + l/16.
_W11 = dict(a_reg=R2((0, 1), (0, 2), (0, 4), (0, 8)), a_lane=R2((1, 0), (2, 0), (4, 0), (8, 0), (0, 0)),
            b_reg=R2((1, 0), (2, 0), (4, 0), (8, 0)), b_lane=R2((0, 1), (0, 2), (0, 4), (0, 8), (0, 0)),
            c_reg=R2((2, 0), (4, 0), (8, 0)), c_lane=R2((0, 1), (0, 2), (0, 4), (0, 8), (1, 0)), c_ty="<8 x float>")
_GFX11 = {"bfloat16": MmaAtom(16, 16, 16, T.bf16, "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16", "<16 x i16>",
                              "i16", **_W11),
          "float16": MmaAtom(16, 16, 16, T.f16, "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16", "<16 x half>",
                             "half", **_W11)}

# ---------------------------------------------------------------- RDNA4 (wave32, WMMA, no lane repeat)
# A: lane l holds row l % 16, K 8*(l/16) + j. C: lane l holds rows 8*(l/16) + i.
_W12 = dict(a_reg=R2((0, 1), (0, 2), (0, 4)), a_lane=R2((1, 0), (2, 0), (4, 0), (8, 0), (0, 8)),
            b_reg=R2((1, 0), (2, 0), (4, 0)), b_lane=R2((0, 1), (0, 2), (0, 4), (0, 8), (8, 0)),
            c_reg=R2((1, 0), (2, 0), (4, 0)), c_lane=R2((0, 1), (0, 2), (0, 4), (0, 8), (8, 0)), c_ty="<8 x float>")
_GFX12 = {"bfloat16": MmaAtom(16, 16, 16, T.bf16, "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16", "<8 x i16>",
                              "i16", **_W12),
          "float16": MmaAtom(16, 16, 16, T.f16, "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16", "<8 x half>",
                             "half", **_W12)}

_CDNA_FLAGS = 0x00020000
_RDNA_FLAGS = 0x31004000

ARCHES: Dict[str, Arch] = {}
for _n, _fam, _lds, _atoms in (("gfx942", "cdna", 64 << 10, _GFX942), ("gfx950", "cdna", 160 << 10, _GFX950),
                               ("gfx1100", "rdna", 64 << 10, _GFX11), ("gfx1101", "rdna", 64 << 10, _GFX11),
                               ("gfx1102", "rdna", 64 << 10, _GFX11), ("gfx1151", "rdna", 64 << 10, _GFX11),
                               ("gfx1200", "rdna", 64 << 10, _GFX12), ("gfx1201", "rdna", 64 << 10, _GFX12)):
    ARCHES[_n] = Arch(_n, _fam, 64 if _fam == "cdna" else 32, _lds,
                      _CDNA_FLAGS if _fam == "cdna" else _RDNA_FLAGS, _atoms)


def get_arch(name: str) -> Arch:
    try:
        return ARCHES[name.split(":")[0]]
    except KeyError:
        raise NotImplementedError(f"hk2: unknown GPU {name}; supported: {', '.join(ARCHES)}") from None


# ---------------------------------------------------------------- layouts built from atoms
def warp_grid(m: int, n: int, nwarps: int, atom: MmaAtom) -> Tuple[int, int]:
    """Split ``nwarps`` over an m x n output: as square as the tile allows, never below one atom per warp."""
    best = None
    for i in range(log2(nwarps) + 1):
        wm, wn = 1 << i, nwarps >> i
        if m // wm < atom.m or n // wn < atom.n:
            continue
        score = (abs(log2(m // wm) - log2(n // wn)), -wm)
        if best is None or score < best[0]:
            best = (score, (wm, wn))
    if best is None:  # more warps than atoms: the spare warps repeat the work
        wm = max(1, min(nwarps, m // atom.m))
        wn = max(1, min(nwarps // wm, n // atom.n))
        return wm, wn
    return best[1]


def _rep(base: int, count: int, dim: int, rank: int = 2):
    return tuple(tuple((base << i) if d == dim else 0 for d in range(rank)) for i in range(log2(count)))


def _warps(wm, wn, nwarps, m_step, n_step, a_like: Optional[str]):
    """Warp bases: N first, then M; an operand ignores the other dimension (its warps share data)."""
    out = []
    for i in range(log2(wn)):
        out.append((0, 0) if a_like == "a" else (0, n_step << i))
    for i in range(log2(wm)):
        out.append((0, 0) if a_like == "b" else (m_step << i, 0))
    while len(out) < log2(nwarps):
        out.append((0, 0))
    return tuple(out)


def acc_layout(atom: MmaAtom, m: int, n: int, nwarps: int) -> Layout:
    wm, wn = warp_grid(m, n, nwarps, atom)
    tm, tn = m // wm, n // wn
    reg = atom.c_reg + _rep(atom.n, tn // atom.n, 1) + _rep(atom.m, tm // atom.m, 0)
    return Layout((m, n), reg, atom.c_lane, _warps(wm, wn, nwarps, tm, tn, None), f"mma_c{wm}x{wn}")


def a_layout(atom: MmaAtom, m: int, k: int, n: int, nwarps: int) -> Layout:
    wm, wn = warp_grid(m, n, nwarps, atom)
    tm = m // wm
    reg = atom.a_reg + _rep(atom.k, k // atom.k, 1) + _rep(atom.m, tm // atom.m, 0)
    return Layout((m, k), reg, atom.a_lane, _warps(wm, wn, nwarps, tm, 0, "a"), f"mma_a{wm}x{wn}")


def b_layout(atom: MmaAtom, m: int, k: int, n: int, nwarps: int) -> Layout:
    wm, wn = warp_grid(m, n, nwarps, atom)
    tn = n // wn
    reg = atom.b_reg + _rep(atom.k, k // atom.k, 0) + _rep(atom.n, tn // atom.n, 1)
    return Layout((k, n), reg, atom.b_lane, _warps(wm, wn, nwarps, 0, tn, "b"), f"mma_b{wm}x{wn}")
