"""Attention schedules: flash attention forward, (B, H, N, D) bf16.

The same protocol as the GEMMs' (hk.schedules.Schedule), for a problem that
is a sequence length and a head dim rather than M x N x K:

    AttnRdna   gfx1100's WMMA kernel   hk.ops.attn.attn_kernel
    AttnCdna   gfx950's pingpong       hk.ops.attn_cdna.attn_kernel

`head_dim`, `causal` and `scale` are the problem, not the schedule: they are
arguments of kernel(), check() and launch(), and a schedule is the knobs
that choose how a given problem is computed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple

from ._base import Schedule, _arch, _dtype_name, knob

__all__ = ["AttnSchedule", "AttnRdna", "AttnCdna", "lookup_attention", "space_attention"]

_KERNELS: Dict[Tuple, Any] = {}


@dataclass(frozen=True)
class AttnSchedule(Schedule):
    """What an attention schedule does: check a problem, build its kernel,
    run it. The problem is `n` (query rows; `n_kv` keys, default n),
    `head_dim` and `causal`."""

    def check(self, n: int, head_dim: int, arch: Optional[str] = None, *,
              causal: bool = False, n_kv: Optional[int] = None,
              dtype: Any = "bf16") -> List[str]:
        """Why this schedule cannot run attention over `n` queries (and
        `n_kv` keys) of `head_dim` on `arch`: [] if it can."""
        arch = _arch(arch)
        n_kv = n if n_kv is None else n_kv
        if arch not in self.arches:
            return [f"{self.family}: builds for {', '.join(self.arches)}, not {arch}"]
        why = []
        if _dtype_name(dtype) != "bf16":
            why.append(f"{self.family}: bf16 only, not {_dtype_name(dtype)}")
        if n < 1 or n_kv < 1:
            return why + [f"n={n}, n_kv={n_kv}: an empty sequence has nothing to launch"]
        if causal and n_kv != n:
            why.append(f"causal cross attention (n_kv={n_kv} != n={n}): the diagonal "
                       f"conventions differ; not a kernel hk has")
        why += self.check_kernel(arch, head_dim=head_dim, causal=causal)
        why += self._check_shape(n, n_kv, head_dim, arch)
        return why

    def check_kernel(self, arch: Optional[str] = None, *, head_dim: int = 128,
                     causal: bool = False) -> List[str]:
        """The kernel half of check(): whether it traces and builds."""
        arch = _arch(arch)
        if arch not in self.arches:
            return [f"{self.family}: builds for {', '.join(self.arches)}, not {arch}"]
        from ..ops import attn  # noqa: PLC0415

        if head_dim not in attn.HEAD_DIMS:
            return [f"head_dim={head_dim}: hk's attention has a schedule for "
                    f"{attn.HEAD_DIMS}"]
        return self._check_struct(arch, head_dim)

    def _check_struct(self, arch: str, head_dim: int = 128) -> List[str]:
        return []

    def _check_shape(self, n, n_kv, head_dim, arch) -> List[str]:
        return []

    def kernel(self, head_dim: int = 128, causal: bool = False,
               scale: Optional[float] = None, *, arch: Optional[str] = None):
        """The hk.Kernel for this schedule and problem: q, k, v, o, all
        (B, H, N, D) bf16 (K and V may have fewer heads: GQA)."""
        if scale is not None and scale == head_dim ** -0.5:
            scale = None
        if arch is None or arch == _current():
            got = self._shared(head_dim, causal, scale)
            if got is not None:
                return got
        key = (self, head_dim, causal, scale, arch)
        if key not in _KERNELS:
            _KERNELS[key] = self._make(head_dim, causal, scale, arch)
        return _KERNELS[key]

    def _shared(self, head_dim, causal, scale):
        return None

    def _make(self, head_dim, causal, scale, arch):
        raise NotImplementedError

    def launch(self, q, k, v, *, causal: bool = False, scale: Optional[float] = None,
               out=None):
        """Attention with this schedule: (B, H, N, D) bf16, contiguous; checked
        first (an illegal schedule raises hk.schedules.Illegal)."""
        import torch  # noqa: PLC0415

        from ._base import Illegal  # noqa: PLC0415

        n, d = q.shape[-2], q.shape[-1]
        why = self.check(n, d, causal=causal, n_kv=k.shape[-2])
        if why:
            raise Illegal(self, why, arch=_current())
        if out is None:
            out = torch.empty_like(q)
        self.kernel(d, causal, scale)(q, k, v, out)
        return out

    @classmethod
    def base_for_problem(cls, n: int, head_dim: int, causal: bool, arch: str) -> "AttnSchedule":
        """The family's starting point for a problem: what hk ships for it."""
        return cls()

    @classmethod
    def space(cls, n: int, head_dim: int, arch: Optional[str] = None, *,
              causal: bool = False, around: Optional["AttnSchedule"] = None,
              radius: int = 1, axes=None, legal: bool = True) -> Iterator["AttnSchedule"]:
        """The family's legal schedules for the problem (see
        Schedule.space for around, radius and axes)."""
        import dataclasses  # noqa: PLC0415
        import itertools  # noqa: PLC0415

        arch = _arch(arch)
        if arch not in cls.arches:
            return
        ax = cls.axes()
        if axes:
            bad = set(axes) - set(ax)
            if bad:
                raise ValueError(f"{cls.family} has no knob(s) {sorted(bad)}; it has {list(ax)}")
        if around is not None:
            if not isinstance(around, cls):
                return
            gen = around.neighbors(radius, list(axes) if axes else None)
        else:
            b = cls.base_for_problem(n, head_dim, causal, arch)
            lists = {}
            for name, a in ax.items():
                if axes and name in axes:
                    v = axes[name]
                    lists[name] = (list(a.values) if v == "all" else
                                   list(v) if isinstance(v, (list, tuple)) else [v])
                elif a.seed is not None:
                    lists[name] = list(a.seed)
                else:
                    lists[name] = [getattr(b, name)]
            names = list(lists)
            gen = (dataclasses.replace(b, **dict(zip(names, combo)))
                   for combo in itertools.product(*(lists[x] for x in names)))
        seen = set()
        for s in gen:
            if s in seen:
                continue
            seen.add(s)
            if not legal or not s.check(n, head_dim, arch, causal=causal):
                yield s


def _current() -> str:
    from ..target import current_arch  # noqa: PLC0415

    return current_arch()


def _scale_suffix(scale: Optional[float]) -> str:
    if scale is None:
        return ""
    return f"_s{struct.unpack('<Q', struct.pack('<d', float(scale)))[0]:016x}"


#: AttnRdna's knobs hk.ops.attn's tuners search and record (its DEFAULT).
_TUNED = ("vt_d_chunk", "qk_tiles", "pv_tiles")


@dataclass(frozen=True)
class AttnRdna(AttnSchedule):
    """gfx1100's flash attention (hk.ops.attn; history chapter 3): `warps`
    waves of `q_block` query rows each, the KV axis in `kv_block` steps, K
    staged and V transposed through LDS. q_block x warps is the query tile:
    a query shorter than it has no block to back up into."""

    family: ClassVar[str] = "attn_rdna"
    arches: ClassVar[Tuple[str, ...]] = ("gfx1100", "gfx1201")

    q_block: int = knob(16, values=(16,), doc="query rows a wave owns (16: the kernel's "
                                             "operand tiles are one fragment)")
    kv_block: int = knob(32, values=(32, 64), doc="the KV step: K/V rows a block")
    warps: int = knob(12, values=(8, 12, 16), doc="waves a workgroup (q_block x warps rows)")
    vt_d_chunk: int = knob(16, values=(16,), doc="head-dim width of V^T a wave stages "
                                                 "(16 only: wider needs a register subtile)")
    qk_tiles: int = knob(2, values=(1, 2, 3, 4), seed=(1, 2, 3, 4),
                         doc="16x16 LDS reads of K in flight before the wait that retires them")
    pv_tiles: int = knob(2, values=(1, 2, 3, 4), seed=(1, 2, 3, 4),
                         doc="16x16 LDS reads of V^T in flight before their wait")
    setprio: bool = knob(True, values=(True, False),
                         doc="s_setprio around each MMA group (kept: within noise)")
    max_vgprs: int = knob(256, gate=True, doc="build gate")
    min_occupancy: int = knob(1, gate=True, doc="build gate")

    def legacy(self) -> Dict[str, Any]:
        """The tuner's dict (hk.ops.attn.DEFAULT's keys)."""
        return {k: getattr(self, k) for k in _TUNED}

    def _extra(self) -> Dict[str, Any]:
        return {k: v for k, v in self.changed().items() if k not in _TUNED}

    def name_for(self, head_dim: int, causal: bool, scale: Optional[float] = None) -> str:
        """The kernel's name, as hk.ops.attn names it."""
        from .. import autotune  # noqa: PLC0415
        from ..ops import attn  # noqa: PLC0415

        base = f"attn_fwd_d{head_dim}{'_causal' if causal else ''}" + _scale_suffix(scale)
        sched = self.legacy()
        if sched != attn.DEFAULT:
            base += f"_{autotune.label_of(sched)}"
        extra = self._extra()
        return base + (f"_{autotune.label_of(extra)}" if extra else "")

    def _shared(self, head_dim, causal, scale):
        from ..ops import attn  # noqa: PLC0415

        if scale is None and not self._extra() and (head_dim, causal) in attn.TUNERS:
            return attn.TUNERS[(head_dim, causal)]._kernel_for(
                {**attn.DEFAULT, **self.legacy()})
        return None

    def _make(self, head_dim, causal, scale, arch):
        from ..ops import attn  # noqa: PLC0415

        k = attn.attn_kernel(self.name_for(head_dim, causal, scale), head_dim=head_dim,
                             causal=causal, scale=scale, **self.knobs())
        if arch:
            k._arch = arch
        return k

    def _check_struct(self, arch: str, head_dim: int = 128) -> List[str]:
        from ..target import get_target  # noqa: PLC0415

        why = []
        for nm in ("q_block", "kv_block", "vt_d_chunk"):
            v = getattr(self, nm)
            if v <= 0 or v % 16:
                why.append(f"{nm}={v}: every extent is in 16x16 WMMA fragments")
        if why:
            return why
        if self.q_block != 16:
            why.append(f"q_block={self.q_block}: the kernel's Q, S and O tiles are one "
                       f"16-row fragment a wave (q_block 32 fails to trace: mma_ABt "
                       f"shapes)")
        if self.kv_block < 32:
            why.append(f"kv_block={self.kv_block}: V^T's LDS tile is then 16 wide, whose "
                       f"32-byte swizzle the granule reads do not take")
        if why:
            return why
        if self.kv_block % (16 * self.qk_tiles):
            why.append(f"qk_tiles={self.qk_tiles} must tile the {self.kv_block}-row KV block")
        if head_dim % (16 * self.pv_tiles):
            why.append(f"pv_tiles={self.pv_tiles} must tile the {head_dim}-wide head")
        if self.vt_d_chunk != 16:
            why.append(f"vt_d_chunk={self.vt_d_chunk}: only 16 is implemented")
        if head_dim // self.vt_d_chunk > self.warps:
            why.append(f"head_dim/vt_d_chunk = {head_dim // self.vt_d_chunk} V^T chunks "
                       f"but only {self.warps} warps: one chunk a warp is the only split")
        lds = 2 * self.kv_block * head_dim * 2
        cap = get_target(arch).lds_per_workgroup
        if lds > cap:
            why.append(f"LDS: {lds} B; {arch} has {cap}")
        if not why:
            why += _rdna_registers(self, head_dim)
        return why

    def _check_shape(self, n, n_kv, head_dim, arch) -> List[str]:
        why = []
        if n < self.q_block * self.warps:
            why.append(f"n={n} is below one query tile ({self.q_block} x {self.warps} = "
                       f"{self.q_block * self.warps} rows): the last block backs up, and "
                       f"there is nothing to back into")
        if n_kv < self.kv_block:
            why.append(f"n_kv={n_kv} is below one KV block ({self.kv_block})")
        return why

    @classmethod
    def base_for_problem(cls, n, head_dim, causal, arch):
        from ..ops import attn  # noqa: PLC0415

        if (head_dim, causal) in attn.TUNERS:
            return cls(**attn.TUNERS[(head_dim, causal)].schedule_for(attn._tune_key(n)))
        return cls()


def _rdna_registers(s: AttnRdna, head_dim: int) -> List[str]:
    """What the samples found (tools/hk-bench/schedules_legality.py
    sample-attn: every point of the space at head_dim 64 and 128, causal or
    not): a 64-row KV block at head_dim 128 holds S and V's staging beside O
    and spilled in 99 of 108 builds; at 32 rows every build is 203-233 VGPRs."""
    if head_dim * s.kv_block > 128 * 32:
        return [f"registers: kv_block={s.kv_block} at head_dim {head_dim} -- S, the staged "
                f"V and O do not fit beside each other (99 of 108 sampled builds "
                f"spilled); kv_block 32"]
    return []


@dataclass(frozen=True)
class AttnCdna(AttnSchedule):
    """gfx950's flash attention (hk.ops.attn_cdna; history chapter 34):
    HipKittens' CDNA4 GQA forward, eight warps of 32 query rows, 64-row KV
    blocks double-buffered by the DMA, two halves a barrier apart. The tile
    is fixed; the knobs are the schedule's choices around it."""

    family: ClassVar[str] = "attn_cdna"
    arches: ClassVar[Tuple[str, ...]] = ("gfx950",)

    lazy: float = knob(8.0, values=(0.0, 4.0, 8.0, 12.0), seed=(0.0, 8.0),
                       doc="log2 headroom before O is rescaled (FlashAttention 4's threshold)")
    sched: bool = knob(True, values=(True, False), seed=(True, False),
                       doc="HipKittens' interleave of MFMAs with VALU and exp2")
    setprio: bool = knob(True, values=(True, False), seed=(True, False),
                         doc="s_setprio around the PV phase")
    lpt: bool = knob(True, values=(True, False),
                     doc="causal: longest query blocks first")
    skip: bool = knob(True, values=(True, False),
                      doc="causal: drop the math of the last two steps past a warp's queries")
    pairs: bool = knob(True, values=(True, False),
                       doc="causal: a workgroup walks the longest and the shortest query block left")

    def name_for(self, head_dim: int, causal: bool, scale: Optional[float] = None) -> str:
        """The kernel's name, as hk.ops.attn_cdna.kernel_for names it."""
        name = f"attn_fwd_d{head_dim}{'_causal' if causal else ''}_cdna" + _scale_suffix(scale)
        for kk, vv in sorted(self.changed().items()):
            val = int(vv) if isinstance(vv, bool) else vv
            name += "_" + "".join(ch if ch.isalnum() else "p" for ch in f"{kk}{val}")
        return name

    def _shared(self, head_dim, causal, scale):
        from ..ops import attn_cdna  # noqa: PLC0415

        return attn_cdna.kernel_for(head_dim, causal, scale, **self.changed())

    def _make(self, head_dim, causal, scale, arch):
        from ..ops import attn_cdna  # noqa: PLC0415

        return attn_cdna.attn_kernel(self.name_for(head_dim, causal, scale),
                                     head_dim=head_dim, causal=causal, scale=scale,
                                     arch=arch, **self.knobs())

    def _check_struct(self, arch: str, head_dim: int = 128) -> List[str]:
        if self.lazy < 0:
            return [f"lazy={self.lazy}: a headroom in log2 units, not negative"]
        return []

    def canonical(self):
        return self


def lookup_attention(n: int, head_dim: int, causal: bool = False,
                     arch: Optional[str] = None, *,
                     scale: Optional[float] = None) -> AttnSchedule:
    """The schedule hk.attention runs for `n` queries of `head_dim`: gfx950's
    kernel as attn_cdna ships it, gfx1100's at its tuner's record for the
    length (hk.ops.attn._tune_key) -- or, with a `scale` other than
    1/sqrt(head_dim), the default, as hk.ops.attn builds those."""
    arch = _arch(arch)
    if arch == "gfx950":
        return AttnCdna()
    if scale is not None and scale != head_dim ** -0.5:
        return AttnRdna()
    return AttnRdna.base_for_problem(n, head_dim, causal, arch)


def space_attention(n: int, head_dim: int, arch: Optional[str] = None, *,
                    causal: bool = False, around: Optional[AttnSchedule] = None,
                    radius: int = 1, axes=None) -> Iterator[AttnSchedule]:
    """Legal attention schedules for the problem on `arch` (see
    Schedule.space)."""
    arch = _arch(arch)
    if around is not None:
        yield from type(around).space(n, head_dim, arch, causal=causal, around=around,
                                      radius=radius, axes=axes)
        return
    for cls in (AttnRdna, AttnCdna):
        if arch in cls.arches:
            yield from cls.space(n, head_dim, arch, causal=causal, axes=axes)
