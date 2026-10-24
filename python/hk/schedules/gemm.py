"""GEMM schedules: C = A @ B.T, A (M, K) and B (N, K) bf16, fp32 accumulate.

Four kernel families and one composite, each a frozen dataclass whose
fields are every knob its kernel factory takes:

    Rdna     gfx1100's WMMA GEMM            hk.ops.gemm.gemm_kernel
    Decode   the decode GEMV, M <= 64       hk.ops.decode_gemm.decode_gemm_kernel
    PingPong gfx950's 8-wave pingpong       hk.ops.gemm_cdna.gemm_kernel
    Dp       gfx950's 4-warp ring of stages hk.ops.gemm_cdna_dp.gemm_kernel
    Columns  two schedules side by side on column ranges of C (a tail, or
             the columns that fill whole waves of CUs and the rest)

A split-K schedule (`splits` > 1, PingPong and Dp) is two launches -- the
GEMM's fp32 shares into a scratch, the reduction into C -- and its steps()
say so. See docs/design/schedules.md for the design and docs/schedules.md
for using it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple

from . import _regs
from ._base import Schedule, _arch, family, families, from_record, knob

__all__ = ["Rdna", "Decode", "PingPong", "Dp", "Columns", "GemmSchedule", "Step",
           "lookup", "from_candidate", "space", "CUS"]


# ---------------------------------------------------------------- steps


@dataclass
class Step:
    """One launch of a GEMM call.

    kernel    "rdna" (hk.ops.gemm), "pp" (gemm_cdna's pingpong), "dp"
              (gemm_cdna_dp's four-warp blocks), "decode" (decode_gemm),
              "reduce" (gemm_cdna_dp's strip reduction of split-K shares)
    sched     the kernel's schedule, as the module that owns it spells it
    cols      the columns [lo, hi) of C it covers
    epilogue  whether the epilogue runs in it
    into      "out" (C) or "scratch" (fp32 split-K shares)
    route     the candidate name the route table gave (gfx950)
    schedule  the hk.schedules object it came from (None for "reduce" and
              for steps made by hand)
    """

    kernel: str
    sched: Dict[str, Any]
    cols: Tuple[int, int]
    epilogue: bool
    into: str = "out"
    route: str = ""
    schedule: Any = field(default=None, compare=False, repr=False)

    def __str__(self) -> str:
        e = " +epilogue" if self.epilogue else ""
        r = f" [{self.route}]" if self.route else ""
        return f"{self.kernel}{r} cols {self.cols[0]}:{self.cols[1]} -> {self.into}{e}"


# ---------------------------------------------------------------- common


#: Kernels this module built (not the ones hk.ops shares), by
#: (family, schedule, epilogue tag, arch).
_KERNELS: Dict[Tuple, Any] = {}
#: Prepared launches, by (schedule, M, N, K, device).
_RUNNERS: Dict[Tuple, Callable] = {}

#: CUs a wave of one-a-CU workgroups fills on an MI355X (gemm_cdna.CUS).
CUS = 256

_GATES = ("max_vgprs", "min_occupancy")


def _current() -> str:
    from ..target import current_arch  # noqa: PLC0415

    return current_arch()


def _target_of(arch: str):
    from ..target import get_target  # noqa: PLC0415

    return get_target(arch)


def _lds_cap(arch: str) -> int:
    from ..target import get_target  # noqa: PLC0415

    return get_target(arch).lds_per_workgroup


@dataclass(frozen=True)
class GemmSchedule(Schedule):
    """What every GEMM schedule does, beyond Schedule's protocol: the
    launches of a call (steps), its kernels, and running it."""

    def steps(self, m: int, n: int, k: int) -> List[Step]:
        """The launches of a call at M x N x K, in order, with the one(s)
        that would carry an epilogue (hk.templates.gemm)."""
        return self._steps(m, n, k, 0)

    def _steps(self, m: int, n: int, k: int, lo: int) -> List[Step]:
        splits = getattr(self, "splits", 1)
        cols = (lo, lo + n)
        if splits > 1:
            return [Step(self.family, self.knobs(), cols, False, "scratch", schedule=self),
                    Step("reduce", {"splits": splits}, cols, True)]
        return [Step(self.family, self.knobs(), cols, True, schedule=self)]

    @property
    def name(self) -> str:
        """The kernel's name: the schedule, spelled as hk.ops spells it."""
        return self._name()

    def _name(self) -> str:
        raise NotImplementedError

    def kernel(self, *, epilogue=None, arch: Optional[str] = None):
        """The hk.Kernel this schedule builds (for a split-K schedule, the
        GEMM that writes the fp32 shares). `epilogue` is hk.templates.gemm's
        bound epilogue (hk.ops._epilogue); `arch` defaults to the current one.
        Traced on first use: .source() is its C++, .build() compiles it."""
        if (epilogue is None and (arch is None or arch == _current())
                and self._shared_ok()):
            return self._shared_kernel()
        key = (self.family, self, getattr(epilogue, "tag", None), arch)
        got = _KERNELS.get(key)
        if got is None:
            got = _KERNELS[key] = self._make(epilogue, arch)
        return got

    def _shared_ok(self) -> bool:
        return all(getattr(self, g) == type(self).__dataclass_fields__[g].default
                   for g in _GATES)

    def _shared_kernel(self):
        return self._make(None, None)

    def _make(self, epilogue, arch):
        raise NotImplementedError

    def kernels(self, m: int, n: int, k: int, *, epilogue=None,
                arch: Optional[str] = None) -> List[Tuple[Step, Any, Dict[str, Any]]]:
        """[(step, kernel, constexprs)] a call at M x N x K launches, with
        `epilogue` (hk.templates.gemm's) on the step that carries it: what
        k.source(**constexprs) / k.build(**constexprs) take, with no GPU."""
        from ..ops import gemm_cdna_dp  # noqa: PLC0415

        out = []
        for st in self.steps(m, n, k):
            lo, hi = st.cols
            if st.kernel == "reduce":
                if epilogue is not None:
                    kern, consts = gemm_cdna_dp.reduce_epilogue_kernel(
                        st.sched["splits"], hi - lo, epilogue, arch=arch, m=m)
                else:
                    kern, consts = _plain_reduce(st.sched["splits"], m, hi - lo, arch)
                out.append((st, kern, consts))
                continue
            e = epilogue if st.epilogue else None
            out.append((st, st.schedule.kernel(epilogue=e, arch=arch), {}))
        return out

    def predict(self, arch: Optional[str] = None) -> Dict[str, Any]:
        """What check() predicts the build will report: VGPRs (arch VGPRs;
        an upper estimate), LDS bytes, waves per SIMD."""
        return _regs.predict(self, _arch(arch))

    def _check_knobs(self, arch: str) -> List[str]:
        return self._check_struct(arch) or self._check_resources(arch)

    def _check_struct(self, arch: str) -> List[str]:
        """The knobs' arithmetic, before any resource is predicted."""
        return []

    def _check_resources(self, arch: str) -> List[str]:
        p = self.predict(arch)
        why = []
        cap = _lds_cap(arch)
        if p["lds"] > cap:
            why.append(f"LDS: {p['lds']} B a workgroup (as hk-lds-alloc places it, each "
                       f"buffer aligned to its size); {arch} has {cap}")
        simd = _target_of(arch).vgprs_per_simd
        if p.get("estimate", 0) > simd:
            why.append(f"registers: the register tiles alone are an estimated "
                       f"{p['estimate']} VGPRs, more than the {simd} a SIMD has "
                       f"(hk-check-budgets refuses it while tracing)")
        if p["vgpr"] is not None:
            limit = p["limit"] if self.max_vgprs >= 256 else min(p["limit"], self.max_vgprs)
            if p["vgpr"] > limit:
                what = "VGPRs" if arch != "gfx950" else "VGPRs + AGPRs"
                why.append(f"registers: up to {p['vgpr']} {what} predicted, over the "
                           f"{limit} a wave has{' (max_vgprs)' if limit < p['limit'] else ''}"
                           f": it would spill, which the build refuses -- "
                           f"{p['hint']}")
            elif self.min_occupancy > 1 and p["occupancy"] < self.min_occupancy:
                why.append(f"occupancy: up to {p['vgpr']} registers predicted is "
                           f"{p['occupancy']} waves a SIMD, under min_occupancy="
                           f"{self.min_occupancy}")
        return why

    def launch(self, a, b, out=None):
        """C = A @ B.T with this schedule (every launch of the call); `b` is
        (N, K). Returns `out`, made if not given (bf16 (M, N)). Needs a GPU
        of an arch in `arches`. The schedule is checked first: an illegal
        one raises hk.schedules.Illegal with check()'s reasons."""
        import torch  # noqa: PLC0415

        m, k = a.shape
        n = b.shape[0]
        key = (self, m, n, k, a.device)
        go = _RUNNERS.get(key)
        if go is None:
            self.require(m, n, k)
            go = _RUNNERS[key] = _runner(self, m, n, k, a.device)
        if out is None:
            out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        go(a, b, out)
        return out

    __call__ = launch


def _plain_reduce(splits: int, m: int, n: int, arch: Optional[str]):
    """The reduction gemm_cdna_dp.reduce launches into a contiguous C."""
    from ..ops import gemm_cdna_dp as dp  # noqa: PLC0415
    from ..target import get_target  # noqa: PLC0415

    tgt = get_target(_arch(arch))
    name = f"gemm_cdna_dp_sreduce_k{splits}"
    key = ("reduce", name, arch)
    if key not in _KERNELS:
        _KERNELS[key] = dp.strip_reduce_kernel(name, splits, dp.STRIP_REDUCE_WARPS, arch=arch)
    cw = dp.STRIP_REDUCE_CHUNKS * tgt.wave * 4
    return _KERNELS[key], dict(CW=cw, RUN=4,
                               EXACT=int(m * n % (dp.STRIP_REDUCE_WARPS * cw) == 0))


def _runner(s: GemmSchedule, m: int, n: int, k: int, device):
    """go(a, b, out): every launch of s at this shape, the scratch made."""
    from ..ops import decode_gemm, gemm_cdna_dp  # noqa: PLC0415

    calls = []
    contiguous_a = False
    for st in s.steps(m, n, k):
        lo, hi = st.cols
        whole = (lo, hi) == (0, n)
        if st.kernel == "reduce":
            calls.append(("reduce", lo, hi, whole, st.sched["splits"]))
            continue
        kern = st.schedule.kernel()
        if st.into == "scratch":
            ws = gemm_cdna_dp.scratch_for(device, st.sched["splits"], m, hi - lo)
            calls.append(("scratch", lo, hi, whole, (kern, ws)))
            continue
        extra = ()
        if st.kernel == "decode":
            contiguous_a = True
            if st.sched["splits"] > 1:
                extra = decode_gemm.scratch_for(device, st.sched["splits"],
                                                st.sched["tokens"], hi - lo)
        elif st.kernel == "pp":
            from ..ops import gemm_cdna  # noqa: PLC0415

            extra = gemm_cdna.sk_args(st.sched, device, m, hi - lo, k)
        calls.append(("out", lo, hi, whole, (kern, extra)))

    if len(calls) == 1 and calls[0][3]:
        kern, extra = calls[0][4]

        def go1(a, b, out):
            if contiguous_a and not a.is_contiguous():
                a = a.contiguous()
            kern(a, b, out, *extra)

        return go1

    def go(a, b, out):
        if contiguous_a and not a.is_contiguous():
            a = a.contiguous()
        ws = None
        for what, lo, hi, whole, arg in calls:
            bb = b if whole else b[lo:hi]
            oo = out if whole else out[:, lo:hi]
            if what == "scratch":
                kern, ws = arg
                kern(a, bb, ws)
            elif what == "reduce":
                gemm_cdna_dp.reduce(ws, oo)
            else:
                kern, extra = arg
                kern(a, bb, oo, *extra)

    return go


# ---------------------------------------------------------------- gfx1100


@dataclass(frozen=True)
class Rdna(GemmSchedule):
    """gfx1100's WMMA GEMM (hk.ops.gemm.gemm_kernel; history chapter 28):
    a block_m x block_n block of C per workgroup of `warps` waves in a
    warp_rows x (warps / warp_rows) grid, K in k_step tiles through LDS."""

    family: ClassVar[str] = "rdna"
    arches: ClassVar[Tuple[str, ...]] = ("gfx1100",)

    block_m: int = knob(128, values=(16, 32, 64, 128, 256), seed=(64, 128),
                        doc="rows of C a workgroup computes: LDS traffic per WMMA "
                            "against the registers of a deeper K-tile")
    block_n: int = knob(128, values=(64, 128, 256),
                        doc="columns of C a workgroup computes; N must be a multiple")
    k_step: int = knob(64, values=(16, 32, 64, 128), seed=(32, 64),
                       doc="K a tile covers: global latency per barrier against LDS; "
                           "K must be a multiple")
    dot_slice: int = knob(16, values=(16, 32),
                          doc="K a register operand covers per LDS read")
    warps: int = knob(8, values=(4, 8), doc="waves a workgroup")
    warp_rows: int = knob(4, values=(1, 2, 4, 8), doc="rows of the warp grid")
    n_split: int = knob(2, values=(1, 2, 4), seed=(2, 4),
                        doc="accumulators the warp tile's columns are cut into: "
                            "register pressure against ds_read width")
    wgm: int = knob(8, values=(1, 4, 8, 16), seed=(4, 8, 16),
                    doc="L2 swizzle: block-rows walked together, how long a B panel stays resident")
    write_pos: int = knob(2, values=(0, 1, 2), seed=(1, 2),
                          doc="where the staging drain lands against the math (two LDS buffers)")
    lds_buffers: int = knob(1, values=(1, 2), seed=(1, 2),
                            doc="1: half the LDS, three workgroups a WGP, two barriers a "
                                "tile; 2: one barrier, two workgroups")
    stagger: int = knob(0, values=(0, 1, 2),
                        doc="workgroup w starts K at tile w*stagger (FlyDSL's; a loss here)")
    stream_a: bool = knob(False, values=(False, True),
                          doc="hold A one 16-row fragment at a time (FlyDSL's four-wave kernel)")
    setprio: bool = knob(False, values=(False, True),
                         doc="s_setprio around each WMMA group (off: 3.5-4.3% faster)")
    sched: int = knob(0, values=(0, 1, 2, 4),
                      doc="sched_group_barrier: one staging load then `sched` WMMAs")
    max_vgprs: int = knob(256, gate=True, doc="build gate: refuse a build over this many VGPRs")
    min_occupancy: int = knob(1, gate=True, doc="build gate: refuse a build under this many waves/SIMD")

    def _name(self) -> str:
        from ..ops import gemm as rdna  # noqa: PLC0415

        return rdna._sched_name(self.knobs())

    def legacy(self) -> Dict[str, Any]:
        """The dict hk.ops.gemm spells this schedule as (schedule_for's and
        the TUNER's): DEFAULT's keys plus any knob off its default."""
        from ..ops import gemm as rdna  # noqa: PLC0415

        kn = self.knobs()
        ch = self.changed()
        return {k: v for k, v in kn.items() if k in rdna.DEFAULT or k in ch}

    def _shared_kernel(self):
        from ..ops import gemm as rdna  # noqa: PLC0415

        return rdna.TUNER._kernel_for(self.legacy())

    def _make(self, epilogue, arch):
        from ..ops import _epilogue  # noqa: PLC0415
        from ..ops import gemm as rdna  # noqa: PLC0415

        k = rdna.gemm_kernel(_epilogue.name(self._name(), epilogue), **self.knobs(),
                             epilogue=epilogue)
        if arch:
            k._arch = arch
        return k

    def _check_struct(self, arch: str) -> List[str]:
        why = []
        if self.warps < 1 or self.warps > 16:
            why.append(f"warps={self.warps}: 1 to 16 waves a workgroup")
            return why
        if self.warp_rows < 1 or self.warps % self.warp_rows:
            why.append(f"warp_rows={self.warp_rows} does not divide warps={self.warps} "
                       f"into a grid")
            return why
        cols = self.warps // self.warp_rows
        reg_m, reg_n = self.block_m // self.warp_rows, self.block_n // cols
        if reg_m * self.warp_rows != self.block_m or reg_m % 16:
            why.append(f"block_m={self.block_m} over {self.warp_rows} warp rows is not "
                       f"whole 16-row fragments")
        if reg_n * cols != self.block_n or reg_n % 16:
            why.append(f"block_n={self.block_n} over {cols} warp columns is not whole "
                       f"16-column fragments")
        elif self.n_split < 1 or reg_n % self.n_split or (reg_n // self.n_split) % 16:
            why.append(f"n_split={self.n_split} must cut the {reg_n}-wide warp tile into "
                       f"whole 16-column chunks")
        if self.dot_slice % 16 or self.dot_slice < 16:
            why.append(f"dot_slice={self.dot_slice}: a multiple of 16 (the WMMA's K)")
        elif self.k_step % self.dot_slice:
            why.append(f"dot_slice={self.dot_slice} must divide k_step={self.k_step}")
        if self.write_pos not in (0, 1, 2):
            why.append(f"write_pos={self.write_pos}: 0, 1 or 2")
        if self.lds_buffers not in (1, 2):
            why.append(f"lds_buffers={self.lds_buffers}: 1 or 2")
        if self.stream_a and reg_m // 16 < 2:
            why.append(f"stream_a streams a {reg_m}-row warp tile by 16-row fragments; "
                       f"it needs at least two")
        if self.stagger < 0 or self.sched < 0:
            why.append("stagger and sched are counts, not negative")
        two = 2 * (self.block_m + self.block_n) * self.k_step * 2
        if two > _lds_cap(arch):
            # hk.ops.gemm's factory sizes its LDS check for two buffers
            # whatever lds_buffers is (_derive), so it refuses these too.
            why.append(f"{self.block_m}x{self.block_n}x{self.k_step}: the factory checks "
                       f"two LDS buffers' {two} B against {arch}'s {_lds_cap(arch)}, "
                       f"whatever lds_buffers is")
        threads = 32 * self.warps
        for what, rows in (("A", self.block_m), ("B", self.block_n)):
            if rows * self.k_step * 2 % (16 * threads):
                why.append(f"a {rows}x{self.k_step} tile of {what} is not a whole number "
                           f"of 16-byte loads for {threads} lanes")
        return why

    def _check_shape(self, m, n, k, arch) -> List[str]:
        why = []
        if k % self.k_step:
            why.append(f"K={k} is not a multiple of k_step={self.k_step}")
        if n % self.block_n:
            why.append(f"N={n} is not a multiple of block_n={self.block_n} "
                       f"(only M may be ragged)")
        return why

    def canonical(self) -> "Rdna":
        s = self
        if s.lds_buffers == 1 and s.write_pos != 2:
            # One buffer has nothing to overlap the write with: write_pos
            # only places the two-buffer schedule's drain.
            s = dataclasses.replace(s, write_pos=2)
        return s

    @classmethod
    def base_for(cls, m, n, k, arch):
        from ..ops import gemm as rdna  # noqa: PLC0415

        if m <= rdna.THIN["block_m"]:
            return cls(**rdna.THIN)
        return cls(**{kk: v for kk, v in rdna.DEFAULT.items()})


@dataclass(frozen=True)
class Decode(GemmSchedule):
    """The decode GEMV (hk.ops.decode_gemm; history chapter 9): W straight to
    registers, one 16-row block of W a workgroup, its K split across `warps`
    waves (and across `splits` workgroups), combined through LDS. M <= 16 x
    tokens. gfx1100's for M <= 32 in hk.templates.gemm; gfx950's "decode" route."""

    family: ClassVar[str] = "decode"
    arches: ClassVar[Tuple[str, ...]] = ("gfx1100", "gfx950")

    tokens: int = knob(1, values=(1, 2, 4), doc="16-row token tiles: M <= 16 x tokens")
    slices: int = knob(4, values=(1, 2, 4, 8), seed=(2, 4),
                       doc="16-deep K slices a load: bytes in flight against registers")
    warps: int = knob(16, values=(1, 2, 4, 8, 16), seed=(4, 8, 16),
                      doc="waves sharing a 16-row block of W: the waves that cover latency")
    combine_depth: int = knob(0, values=(0, 2, 4, 8), seed=(0, 4),
                              doc="partials the combine reads before it waits (0: all)")
    resident: int = knob(0, values=(0, 1, 2, 4), seed=(0, 2),
                         doc="cap on workgroups a WGP/CU holds, by sizing the combine's LDS")
    splits: int = knob(1, values=(1, 2, 4, 8), seed=(1, 2, 4),
                       doc="K cut across workgroups, the last to arrive sums the shares")
    max_vgprs: int = knob(256, gate=True, doc="build gate")
    min_occupancy: int = knob(1, gate=True, doc="build gate")

    def _name(self) -> str:
        return (f"decode_gemm_t{self.tokens}_s{self.slices}_w{self.warps}"
                f"_c{self.combine_depth}_r{self.resident}"
                + (f"_k{self.splits}" if self.splits > 1 else ""))

    def legacy(self) -> Dict[str, Any]:
        return dict(tokens=self.tokens, slices=self.slices, warps=self.warps,
                    combine_depth=self.combine_depth, resident=self.resident,
                    splits=self.splits)

    def _steps(self, m, n, k, lo):
        return [Step("decode", self.legacy(), (lo, lo + n), True, schedule=self)]

    def _shared_kernel(self):
        from ..ops import decode_gemm  # noqa: PLC0415

        return decode_gemm.kernel_for(**self.legacy())

    def _make(self, epilogue, arch):
        from ..ops import _epilogue, decode_gemm  # noqa: PLC0415

        return decode_gemm.decode_gemm_kernel(_epilogue.name(self._name(), epilogue),
                                              **self.knobs(), epilogue=epilogue, arch=arch)

    def _check_struct(self, arch: str) -> List[str]:
        from ..target import get_target  # noqa: PLC0415

        why = []
        for kn in ("tokens", "slices", "warps", "splits"):
            if getattr(self, kn) < 1:
                why.append(f"{kn}={getattr(self, kn)}: at least 1")
        if self.warps > 16:
            why.append(f"warps={self.warps}: at most 16 waves a workgroup")
        if self.combine_depth < 0 or self.resident < 0:
            why.append("combine_depth and resident are counts, not negative")
        if why:
            return why
        tgt = get_target(arch)
        if tgt.wave == 64 and self.slices % 2:
            why.append(f"slices={self.slices}: on {arch} a load is whole 32-deep MFMA "
                       f"operands, so slices is even")
        part = 16 * self.tokens * 16 * 4
        if self.resident:
            if self.warps == 1:
                why.append("resident sizes the combine's LDS, and a one-warp workgroup "
                           "has no combine")
            else:
                budget = tgt.lds_pool_bytes // self.resident - (16 if self.splits > 1 else 0)
                slots = max(self.warps, budget // part)
                if slots * part > tgt.lds_per_workgroup:
                    why.append(f"resident={self.resident} needs {slots * part} B of LDS a "
                               f"workgroup; one may hold {tgt.lds_per_workgroup}")
        return why

    def _check_shape(self, m, n, k, arch) -> List[str]:
        why = []
        if m > 16 * self.tokens:
            why.append(f"M={m} is more than tokens={self.tokens} 16-row tiles hold "
                       f"({16 * self.tokens})")
        if n % 16:
            why.append(f"N={n} is not a multiple of 16 (a workgroup takes 16 rows of W)")
        step = 16 * self.slices * self.warps * self.splits
        if k % step:
            why.append(f"K={k} is not a multiple of 16 x slices x warps x splits = {step} "
                       f"(every warp takes whole rounds)")
        return why

    def canonical(self) -> "Decode":
        s = self
        if s.combine_depth >= s.warps and s.combine_depth:
            s = dataclasses.replace(s, combine_depth=0)
        return s

    @classmethod
    def base_for(cls, m, n, k, arch):
        from ..ops import decode_gemm  # noqa: PLC0415

        s = decode_gemm.schedule_for(min(max(m, 1), 16 * decode_gemm.MAX_TOKENS), k)
        if s is None:
            t = decode_gemm.tokens_for(min(max(m, 1), 16 * decode_gemm.MAX_TOKENS))
            return cls(**decode_gemm.SCHEDULES[t], tokens=t)
        return cls(**s)


# ---------------------------------------------------------------- gfx950


_NT = ("", "a", "b", "ab")


@dataclass(frozen=True)
class PingPong(GemmSchedule):
    """gfx950's pingpong GEMM (hk.ops.gemm_cdna.gemm_kernel; history chapters
    33 and 41): HipKittens' 8-wave CDNA4 schedule, a block in four quadrants,
    two halves of the workgroup a barrier apart so one wave of every SIMD is
    at the matrix core while the other is at memory."""

    family: ClassVar[str] = "pp"
    arches: ClassVar[Tuple[str, ...]] = ("gfx950",)

    block_m: int = knob(256, values=(128, 160, 192, 256, 320), seed=(128, 256),
                        doc="rows of C a workgroup computes: a half is whole 16-row "
                            "fragments for each warp row (160: warps_m=1, 192/320: 1 or 2)")
    block_n: int = knob(256, values=(128, 160, 192, 256, 320), seed=(128, 256),
                        doc="columns of C a workgroup computes; N must be a multiple "
                            "(160: warps_n=1, 192/320: 1 or 2)")
    k_step: int = knob(64, values=(32, 64, 128), doc="K a tile covers")
    warps_m: int = knob(2, values=(1, 2, 4, 8), doc="rows of the 8-warp grid")
    warps_n: int = knob(4, values=(1, 2, 4, 8), doc="columns of the 8-warp grid")
    wgm: int = knob(8, values=(4, 8, 16), doc="L2 swizzle: block-rows walked together")
    xcd_chunk: int = knob(64, values=(16, 32, 64), doc="consecutive workgroups kept on one XCD")
    setprio: bool = knob(True, values=(True, False), doc="raised priority around the MFMAs")
    transposed_c: bool = knob(False, values=(False, True),
                              doc="accumulate C^T: one 8-byte store a fragment (1.2% slower)")
    early_store: bool = knob(True, values=(True, False),
                             doc="store the first two quadrants as soon as they are final")
    hipkittens_waits: bool = knob(False, doc="HipKittens' hand-counted vmcnts (refused "
                                             "without late_b; a no-op with it)")
    late_b: bool = knob(True, values=(True, False),
                        doc="read the next tile's first B one phase later, after its wait")
    splits: int = knob(1, values=(1, 2, 4), seed=(1, 2, 4),
                       doc="split-K: workgroups along K, fp32 shares then a reduction")
    lds_shape: Optional[str] = knob(None, values=(None, "16x64"), seed=(None, "16x64"),
                                    doc="st_16x64 LDS layout: whole-line copies")
    nt: str = knob("", values=("", "b"), seed=("", "b"),
                   doc="nontemporal copies of A ('a') and/or B ('b'): stream past the caches")
    stream_k: int = knob(0, values=(0, 1, 2), seed=(0, 1, 2),
                         doc="persistent: sk_grid workgroups run a plan of (block, K "
                             "range) segments -- 1 every block's K cut into equal runs "
                             "(stream-K), 2 whole waves of blocks then the rest cut; a cut "
                             "block summed in fp32 by its last workgroup")
    sk_grid: int = knob(256, values=(256,),
                        doc="stream-K's workgroups: one a CU of an MI355X")
    max_vgprs: int = knob(256, gate=True, doc="build gate")
    min_occupancy: int = knob(1, gate=True, doc="build gate")

    def _name(self) -> str:
        from ..ops import gemm_cdna  # noqa: PLC0415

        return gemm_cdna._sched_name(self.knobs())

    def _shared_ok(self) -> bool:
        return super()._shared_ok() and not self.hipkittens_waits

    def _shared_kernel(self):
        from ..ops import gemm_cdna  # noqa: PLC0415

        return gemm_cdna.kernel_for(self.knobs())

    def _make(self, epilogue, arch):
        from ..ops import _epilogue, gemm_cdna  # noqa: PLC0415

        return gemm_cdna.gemm_kernel(_epilogue.name(self._name(), epilogue), **self.knobs(),
                                     epilogue=epilogue, arch=arch)

    def _check_struct(self, arch: str) -> List[str]:
        why = []
        if self.warps_m * self.warps_n != 8:
            why.append(f"the pingpong pairs two halves of four warps; warps_m x warps_n is "
                       f"{self.warps_m}x{self.warps_n} = {self.warps_m * self.warps_n}, not 8")
            return why
        half_m, half_n = self.block_m // 2, self.block_n // 2
        reg_m, reg_n = half_m // self.warps_m, half_n // self.warps_n
        if (self.block_m % 2 or self.block_n % 2 or reg_m % 16 or reg_n % 16
                or reg_m * self.warps_m != half_m or reg_n * self.warps_n != half_n):
            why.append(f"{self.block_m}x{self.block_n} over {self.warps_m}x{self.warps_n} "
                       f"warps does not give whole 16-row accumulators in each quadrant")
        if self.k_step % 32 or self.k_step < 32:
            why.append(f"k_step={self.k_step}: a multiple of 32 (the MFMA's K)")
        for what, rows in (("A", half_m), ("B", half_n)):
            # Whole copies a warp: a half that is not whole copies a thread
            # (96 x 64 is 1.5 for 512 threads) has its first warps copy the
            # rest (hk/cdna.cuh's group_load_async).
            if rows * self.k_step * 2 % (16 * 64):
                why.append(f"half a block of {what} ({rows}x{self.k_step}) is not a whole "
                           f"number of 16-byte copies for a 64-lane warp")
        if self.lds_shape not in (None, "16x64"):
            why.append(f"lds_shape={self.lds_shape!r}: None or '16x64'")
        elif self.lds_shape == "16x64" and self.k_step % 64:
            why.append(f"lds_shape='16x64' with k_step={self.k_step}: st_16x64 is read "
                       f"64 deep, so k_step is a multiple of 64")
        if self.nt not in _NT:
            why.append(f"nt={self.nt!r}: one of {_NT}")
        if self.hipkittens_waits and not self.late_b:
            why.append("hipkittens_waits without late_b: hk-lds-pipeline refuses it -- the "
                       "fourth memory phase reads Bs[2] while its copy is in flight")
        if self.splits < 1:
            why.append(f"splits={self.splits}: at least 1")
        if self.stream_k not in (0, 1, 2):
            why.append(f"stream_k={self.stream_k!r}: 0 (off), 1 (stream-K) or 2 (whole "
                       f"waves, then stream-K)")
        if self.stream_k:
            if self.splits != 1:
                why.append(f"stream_k with splits={self.splits}: stream-K cuts K itself "
                           f"(fp32 shares summed in the kernel); splits must be 1")
            if self.transposed_c:
                why.append("stream_k with transposed_c: a cut block's shares are stored "
                           "and summed in the untransposed accumulators' layout")
            if self.sk_grid < 8 or self.sk_grid % 8:
                why.append(f"sk_grid={self.sk_grid}: a whole number of workgroups for each "
                           f"of the 8 XCDs")
        return why

    def _check_shape(self, m, n, k, arch) -> List[str]:
        why = []
        if k % (self.k_step * self.splits):
            why.append(f"K={k} is not a multiple of k_step x splits = "
                       f"{self.k_step * self.splits}")
        if n % self.block_n:
            why.append(f"N={n} is not a multiple of block_n={self.block_n} (a tail of N "
                       f"is a Columns schedule)")
        if self.stream_k:
            if k % (2 * self.k_step):
                why.append(f"stream_k with K={k}: its unit of work is a pair of K-tiles, "
                           f"so K is a multiple of 2 x k_step = {2 * self.k_step}")
            units = -(-m // self.block_m) * (n // max(self.block_n, 1)) * (
                k // (2 * self.k_step))
            if (units + 1) * self.sk_grid >= 2**31:
                why.append(f"stream_k at {m}x{n}x{k}: {units} units x sk_grid="
                           f"{self.sk_grid} overflows the kernel's 32-bit arithmetic")
        why += _scratch_ok(self.splits, m, n)
        return why

    def canonical(self) -> "PingPong":
        s = self
        if s.hipkittens_waits and s.late_b:
            # late_b's waits are HipKittens' own: the same kernel.
            s = dataclasses.replace(s, hipkittens_waits=False)
        if not s.stream_k and s.sk_grid != 256:
            s = dataclasses.replace(s, sk_grid=256)
        return s

    @classmethod
    def base_for(cls, m, n, k, arch):
        from ..ops import gemm_cdna  # noqa: PLC0415

        return cls(**gemm_cdna.schedule_for(m, n, k))


@dataclass(frozen=True)
class Dp(GemmSchedule):
    """gfx950's four-warp GEMM (hk.ops.gemm_cdna_dp.gemm_kernel; history
    chapter 41): smaller blocks for the shapes the pingpong has too few of,
    copies `stages - 1` K-tiles ahead in a ring of LDS buffers, one barrier
    a tile."""

    family: ClassVar[str] = "dp"
    arches: ClassVar[Tuple[str, ...]] = ("gfx950",)

    block_m: int = knob(128, values=(16, 32, 64, 128, 256), seed=(32, 64, 128),
                        doc="rows of C a workgroup computes")
    block_n: int = knob(128, values=(32, 48, 64, 96, 128, 224, 256), seed=(64, 96, 128, 224),
                        doc="columns of C a workgroup computes; N must be a multiple")
    k_step: int = knob(64, values=(32, 64, 128, 256), seed=(64, 128),
                       doc="K a tile covers")
    stages: int = knob(3, values=(2, 3, 4, 5), seed=(2, 3, 4),
                       doc="LDS buffers in the ring: copies stages-1 tiles ahead")
    warps_m: int = knob(2, values=(1, 2, 4), doc="rows of the warp grid")
    warps_n: int = knob(2, values=(1, 2, 4), doc="columns of the warp grid")
    wgm: int = knob(8, values=(4, 8, 16), doc="L2 swizzle: block-rows walked together")
    xcd_chunk: int = knob(32, values=(0, 16, 32, 64), seed=(0,),
                          doc="consecutive workgroups kept on one XCD (0: an eighth of the grid)")
    setprio: bool = knob(False, values=(False, True), doc="raised priority around the MFMAs")
    splits: int = knob(1, values=(1, 2, 4, 8), seed=(1, 2, 4, 8),
                       doc="split-K: workgroups along K, fp32 shares then a reduction")
    pipelined: bool = knob(True, values=(True, False),
                           doc="read tile t+1's operands during tile t, into a second register set")
    lds_shape: Optional[str] = knob(None, values=(None, "16x64"), seed=(None, "16x64"),
                                    doc="st_16x64 LDS layout: whole-line copies, 64-deep reads")
    nt: str = knob("", values=("", "b"), seed=("", "b"),
                   doc="nontemporal copies of A ('a') and/or B ('b'): stream past the caches")
    max_vgprs: int = knob(256, gate=True, doc="build gate")
    min_occupancy: int = knob(1, gate=True, doc="build gate")

    def _name(self) -> str:
        from ..ops import gemm_cdna_dp  # noqa: PLC0415

        return gemm_cdna_dp._sched_name(self.knobs())

    def _shared_kernel(self):
        from ..ops import gemm_cdna_dp  # noqa: PLC0415

        return gemm_cdna_dp.kernel_for(self.knobs())

    def _make(self, epilogue, arch):
        from ..ops import _epilogue, gemm_cdna_dp  # noqa: PLC0415

        return gemm_cdna_dp.gemm_kernel(_epilogue.name(self._name(), epilogue),
                                        **self.knobs(), epilogue=epilogue, arch=arch)

    def _check_struct(self, arch: str) -> List[str]:
        why = []
        if self.warps_m < 1 or self.warps_n < 1 or self.warps_m * self.warps_n > 16:
            why.append(f"warps {self.warps_m}x{self.warps_n}: 1 to 16 waves")
            return why
        if self.warps_m * self.warps_n < 2:
            why.append("dp with one warp: a one-wave workgroup issues every copy's "
                       "addresses itself, and spilled in 9 of 14 sampled builds even "
                       "at small tiles; use warps_m x warps_n of 2 or more")
            return why
        threads = 64 * self.warps_m * self.warps_n
        copies = (self.block_m + self.block_n) * self.k_step * 2 // (16 * threads)
        if threads == 128 and self.stages >= 3 and (
                copies >= 20 or (self.stages >= 5 and copies >= 12)):
            why.append(f"dp with two warps, {copies} copies a thread a tile in "
                       f"{self.stages} stages: the copies' addresses spill SGPRs (28 of "
                       f"64 such sampled builds did); use four warps, fewer stages or a "
                       f"smaller k_step")
        reg_m, reg_n = self.block_m // self.warps_m, self.block_n // self.warps_n
        if (reg_m % 16 or reg_n % 16 or reg_m * self.warps_m != self.block_m
                or reg_n * self.warps_n != self.block_n or not reg_m or not reg_n):
            why.append(f"{self.block_m}x{self.block_n} over {self.warps_m}x{self.warps_n} "
                       f"warps does not give whole 16-row accumulators")
        for what, rows in (("A", self.block_m), ("B", self.block_n)):
            if rows * self.k_step * 2 % (16 * threads):
                why.append(f"a {rows}x{self.k_step} tile of {what} is not a whole number "
                           f"of 16-byte copies for {threads} threads")
        sk = 64 if self.lds_shape == "16x64" else 32
        if self.k_step % sk or self.k_step < sk:
            why.append(f"k_step={self.k_step}: whole {sk}-deep slices"
                       + (" (st_16x64 reads a whole 64-wide subtile)" if sk == 64 else ""))
        if self.stages < 2:
            why.append(f"stages={self.stages}: a ring of two buffers or more")
        if self.lds_shape not in (None, "16x64"):
            why.append(f"lds_shape={self.lds_shape!r}: None or '16x64'")
        if self.nt not in _NT:
            why.append(f"nt={self.nt!r}: one of {_NT}")
        if self.splits < 1:
            why.append(f"splits={self.splits}: at least 1")
        if self.xcd_chunk < 0:
            why.append(f"xcd_chunk={self.xcd_chunk}: 0 (an eighth of the grid) or a count")
        return why

    def _check_shape(self, m, n, k, arch) -> List[str]:
        why = []
        if k % (self.k_step * self.splits):
            why.append(f"K={k} is not a multiple of k_step x splits = "
                       f"{self.k_step * self.splits}")
        if n % self.block_n:
            why.append(f"N={n} is not a multiple of block_n={self.block_n}")
        why += _scratch_ok(self.splits, m, n)
        return why

    @classmethod
    def base_for(cls, m, n, k, arch):
        # gemm_cdna_dp.DEFAULT, which the dataclass's defaults are.
        return cls()


def _scratch_ok(splits: int, m: int, n: int) -> List[str]:
    if splits > 1 and m * n * splits * 4 > 2**30:
        return [f"splits={splits}: the fp32 shares of {m}x{n} would be "
                f"{m * n * splits * 4 / 2**30:.1f} GiB of scratch (at most 1 GiB is kept "
                f"per shape; a shape that large has blocks enough not to split)"]
    return []


# ---------------------------------------------------------------- columns


@dataclass(frozen=True)
class Columns(GemmSchedule):
    """Two schedules side by side: `head` on C's first `cut` columns, `tail`
    on the rest (each its own launches). `cut` is a column count, "block"
    (N less N's remainder by the head's block_n: the pingpong's last 128
    columns of a 151936-wide lm_head), or "wave" (the columns whose
    256x256 blocks fill whole waves of the MI355X's CUs; route "ppw")."""

    family: ClassVar[str] = "cols"
    arches: ClassVar[Tuple[str, ...]] = ("gfx950", "gfx1100")

    head: Any = None
    tail: Any = None
    cut: Any = "block"

    def __post_init__(self):
        if not isinstance(self.head, GemmSchedule) or not isinstance(self.tail, GemmSchedule):
            raise TypeError("Columns(head=, tail=) takes two GEMM schedules")

    @classmethod
    def axes(cls):
        return {}

    def _name(self) -> str:
        return f"cols({self.head.name} | {self.tail.name} @ {self.cut})"

    def __str__(self) -> str:
        return f"cols(head={self.head}, tail={self.tail}, cut={self.cut!r})"

    def to_record(self, *, full: bool = False) -> Dict[str, Any]:
        return {"family": "cols", "cut": self.cut, "head": self.head.to_record(full=full),
                "tail": self.tail.to_record(full=full)}

    @classmethod
    def _from_record_dict(cls, d, strict):
        bad = set(d) - {"head", "tail", "cut"}
        if bad and strict:
            raise ValueError(f"cols has no knob(s) {sorted(bad)}; it has head, tail, cut")
        return cls(head=from_record(d["head"], strict=strict),
                   tail=from_record(d["tail"], strict=strict), cut=d.get("cut", "block"))

    def cut_at(self, m: int, n: int) -> int:
        """The columns the head takes at M x N."""
        if self.cut == "block":
            bn = getattr(self.head, "block_n", 0)
            return n - n % bn if bn else 0
        if self.cut == "wave":
            mb = -(-m // 256)
            if mb > CUS:
                return 0
            per = CUS // mb
            return (n // 256 // per) * per * 256
        return int(self.cut)

    def _steps(self, m, n, k, lo):
        c = self.cut_at(m, n)
        return self.head._steps(m, c, k, lo) + self.tail._steps(m, n - c, k, lo + c)

    def check(self, m, n, k, arch=None, dtype="bf16"):
        arch = _arch(arch)
        why = []
        if not (isinstance(self.cut, int) or self.cut in ("block", "wave")):
            return [f"cut={self.cut!r}: a column count, 'block' or 'wave'"]
        if m < 1 or n < 1 or k < 1:
            return [f"M={m}, N={n}, K={k}: every extent must be at least 1"]
        c = self.cut_at(m, n)
        if not 0 < c < n:
            why.append(f"cut={self.cut!r} gives the head {c} of N={n} columns: both parts "
                       f"must have some")
            return why
        why += [f"head: {r}" for r in self.head.check(m, c, k, arch, dtype)]
        why += [f"tail: {r}" for r in self.tail.check(m, n - c, k, arch, dtype)]
        return why

    def check_kernel(self, arch=None):
        return ([f"head: {r}" for r in self.head.check_kernel(arch)]
                + [f"tail: {r}" for r in self.tail.check_kernel(arch)])

    def kernel(self, *, epilogue=None, arch=None):
        raise TypeError("a Columns schedule is several kernels: kernels(m, n, k)")

    def predict(self, arch=None):
        return {"head": self.head.predict(arch), "tail": self.tail.predict(arch)}

    def neighbors(self, radius: int = 1, axes=None) -> Iterator["Columns"]:
        yield self
        for h in self.head.neighbors(radius, axes):
            if h != self.head:
                yield dataclasses.replace(self, head=h)
        for t in self.tail.neighbors(radius, axes):
            if t != self.tail:
                yield dataclasses.replace(self, tail=t)

    @classmethod
    def space(cls, m, n, k, arch=None, *, around=None, radius=1, axes=None, base=None,
              legal=True):
        arch = _arch(arch)
        if around is not None:
            if isinstance(around, Columns):
                for s in around.neighbors(radius):
                    if not legal or not s.check(m, n, k, arch):
                        yield s
            return
        if arch != "gfx950":
            return
        # The two splits the routes use: the pingpong on N's whole 256-wide
        # blocks and the tail rule's candidates on the rest ("block"), and
        # the pingpong on the columns that fill whole waves of the CUs and
        # the routes' remainders on the rest ("wave").
        from ..ops import gemm_cdna  # noqa: PLC0415

        seen = set()
        wave = cls(head=PingPong(), tail=PingPong(), cut="wave").cut_at(m, n)
        for cut, c, names in (("block", n - n % 256, [t for _, t in gemm_cdna.TAIL] + ["pp128"]),
                              ("wave", wave, ["pp128", "dp128x0", "dp64", "dp128"])):
            if not 0 < c < n:
                continue
            head = PingPong(**gemm_cdna.schedule_for(m, c, k))
            for t in names:
                if not gemm_cdna.fits(t, m, n - c, k):
                    continue
                s = cls(head=head, tail=from_candidate(t, m, n - c, k), cut=cut)
                if s in seen:
                    continue
                seen.add(s)
                if not legal or not s.check(m, n, k, arch):
                    yield s


# ---------------------------------------------------------------- routes


def from_candidate(name: str, m: int, n: int, k: int) -> GemmSchedule:
    """The schedule hk.ops.gemm_cdna's route candidate `name` runs at
    M x N x K -- exactly what gemm_cdna.run launches, as data: the
    pingpong's block resolved to the shape (gemm_cdna.schedule_for), its
    N tail (gemm_cdna.tail_for) and "ppw"'s wave cut as Columns, "decode" as
    decode_gemm.schedule_for's. A route table's own candidates (names the
    code does not have) are schedule records, taken as they are."""
    from ..ops import decode_gemm, gemm_cdna, gemm_cdna_dp  # noqa: PLC0415

    if name not in gemm_cdna.CANDIDATES:
        extra = gemm_cdna.extra_candidates()
        if name in extra:
            return extra[name]
        raise KeyError(f"no gfx950 route candidate {name!r}")
    kind, sched = gemm_cdna.CANDIDATES[name]
    if kind == "decode":
        s = decode_gemm.schedule_for(m, k)
        if s is None:
            raise ValueError(f"the decode GEMV does not take M={m}, K={k}")
        return Decode(**s)
    if kind == "ppw":
        cut = gemm_cdna.wave_cut(m, n)
        return Columns(head=from_candidate("pp", m, cut, k),
                       tail=from_candidate(sched["rem"], m, n - cut, k), cut="wave")
    if kind == "dp":
        return Dp(**{**gemm_cdna_dp.DEFAULT, **sched})
    s = {**gemm_cdna.schedule_for(m, n, k), **sched}
    wide = n - n % 256
    if s.get("splits", 1) == 1 and n % 256 and wide:
        head = PingPong(**{**gemm_cdna.schedule_for(m, wide, k), **sched})
        return Columns(head=head, tail=from_candidate(gemm_cdna.tail_for(m, k), m, n - wide, k),
                       cut="block")
    return PingPong(**s)


def lookup(m: int, n: int, k: int, arch: Optional[str] = None, *,
           op: str = "matmul") -> GemmSchedule:
    """The schedule hk runs today for C = A @ B.T at M x N x K on `arch`:
    op="matmul" what hk.ops.matmul launches (gfx950: gemm_cdna.route's
    candidate; gfx1100: hk.ops.gemm.schedule_for, the tuned records
    included), op="template" what hk.templates.gemm plans (on gfx1100 the
    decode GEMV for M <= 32)."""
    arch = _arch(arch)
    if arch == "gfx950":
        from ..ops import gemm_cdna  # noqa: PLC0415

        return from_candidate(gemm_cdna.route(m, n, k), m, n, k)
    from ..ops import decode_gemm  # noqa: PLC0415
    from ..ops import gemm as rdna  # noqa: PLC0415

    if op == "template":
        from ..templates.gemm import RDNA_DECODE_MAX_M  # noqa: PLC0415

        if m <= RDNA_DECODE_MAX_M and decode_gemm.fits(m, n, k):
            return Decode(**decode_gemm.schedule_for(m, k))
    elif op != "matmul":
        raise ValueError(f"op={op!r}: 'matmul' or 'template'")
    return Rdna(**rdna.schedule_for(m, n, k))


def space(m: int, n: int, k: int, arch: Optional[str] = None, *,
          families=None, around: Optional[Schedule] = None, radius: int = 1,
          axes=None, legal: bool = True) -> Iterator[GemmSchedule]:
    """Legal GEMM schedules for M x N x K on `arch`, family by family (see
    Schedule.space for `around`, `radius` and `axes`). `families` limits
    them: names ("dp", "pp", ...) or classes. With `around`, only that
    schedule's family (its neighbourhood)."""
    arch = _arch(arch)
    if around is not None:
        yield from type(around).space(m, n, k, arch, around=around, radius=radius,
                                      axes=axes, legal=legal)
        return
    fams = [c for c in families_for(arch)
            if families is None or c.family in families or c in families]
    for cls in fams:
        ax = None
        if axes:
            ax = {a: v for a, v in axes.items() if a in cls.axes()}
        yield from cls.space(m, n, k, arch, axes=ax, legal=legal)


def families_for(arch: str) -> List[type]:
    """The GEMM families that run on `arch`, the composite last."""
    return [c for c in (Rdna, Decode, PingPong, Dp, Columns) if arch in c.arches]
