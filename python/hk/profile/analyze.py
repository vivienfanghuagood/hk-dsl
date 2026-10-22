"""From measurements to a verdict: the roofline, the grid, the stalls.

Pure functions over dicts -- no GPU, no profiler -- so the verdict logic is
tested on recorded counter fixtures (tests/hk/profile). `kernel_report()`
takes one kernel's raw material (hk.profile.run.collect's record, its build
row, its ISA analysis) and returns the per-kernel JSON; `findings()` and
`verdict()` read that JSON.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from . import counters as _counters

#: Per arch: the ceilings a kernel is measured against, and where each is
#: from. The same numbers hk.integration.headroom.PEAKS (matrix FLOP/s,
#: DRAM bytes/s) and hk.check.CARDS (the cache-resident rate and the cache's
#: size) use; tests/hk/profile holds them equal.
CEILINGS: Dict[str, Dict[str, Any]] = {
    "gfx950": {
        "matrix_flops": {"bf16": 2.48e15, "fp16": 2.48e15, "fp8": 4.96e15},
        "dram_bytes_s": 6.6e12,
        "cache_bytes_s": 9.1e12,
        "cache_bytes": 256 * 2**20,
        "source": "MI355X measured (hk.integration.headroom.PEAKS: a register-only "
                  "v_mfma bf16 loop on every CU, 2.47-2.48 PFLOP/s; a 1 GiB read past the "
                  "256 MB Infinity Cache, 6.54-6.60 TB/s); cache-resident 9.1 TB/s "
                  "(hk.check.CARDS). fp8 is 2x bf16 (datasheet ratio, not measured)",
    },
    "gfx1100": {
        "matrix_flops": {"bf16": 122.6e12, "fp16": 122.6e12},
        "dram_bytes_s": 864e9,
        "cache_bytes_s": 2.43e12,
        "cache_bytes": 96 * 2**20,
        "source": "W7900D: the bf16 WMMA peak and GDDR6 at 18 Gbps x 384 bits "
                  "(hk.integration.headroom.PEAKS); the fastest cache-resident rate "
                  "measured, 2.43 TB/s at <= 96 MB (hk.check.CARDS). A measured WMMA "
                  "loop reaches 100.5 TFLOP/s (hk.target.gfx1100.CEILINGS)",
    },
}

#: The stable names of a finding's `kind` (the JSON contract; docs/profile.md).
KINDS = (
    "memory_bound",       # the bytes past the L2 run at >= MEMORY_BOUND of the ceiling
    "compute_bound",      # matrix FLOP/s at >= COMPUTE_BOUND of the peak
    "latency_bound",      # neither roof: the waves wait (see stalls)
    "tail",               # the grid is a fraction of a round of the CUs' slots
    "underfilled",        # fewer workgroups than slots: CUs idle the whole kernel
    "epilogue",           # cycles after the main loop that nothing overlaps
    "prologue",           # cycles before the main loop
    "stall",              # one reason a large share of the main loop waits on
    "occupancy",          # achieved waves well under what the resources allow
    "issue_bound",        # few waves a SIMD, and the main loop's other instructions
                          # issue between the matrix ones with nothing to cover them
    "lds_bank_conflict",  # LDS cycles lost to bank conflicts
    "l2_refetch",         # >= 1.8x the call's bytes cross the L2
    "narrow_access",      # global loads/stores under 16 B a lane in a kernel that waits
                          # on vector memory
    "small",              # a kernel too short for its launch and ramp
    "noisy",              # the timed calls spread > 5%: the time is not to be trusted
    "spill",              # scratch in use
)

MEMORY_BOUND = 0.6
#: Past the L2 the rate is measured against the cache-resident one, which
#: only a copy reaches: a higher bar.
FABRIC_BOUND = 0.7
COMPUTE_BOUND = 0.6

#: Thread-trace instruction classes -> the stall reason they stand for.
REASONS = {
    "mfma": "matrix_core", "wmma": "matrix_core",
    "waitcnt:vmem": "vmem_wait", "waitcnt:lgkm": "lgkm_wait",
    "waitcnt:vmem+lgkm": "vmem_lgkm_wait", "waitcnt:exp": "other_wait",
    "waitcnt:other": "other_wait",
    "barrier": "barrier",
    "vmem_load": "vmem_issue", "vmem_lds": "vmem_issue", "vmem_store": "vmem_issue",
    "vmem_atomic": "vmem_issue",
    "ds_read": "lds_issue", "ds_write": "lds_issue", "ds_other": "lds_issue",
    "valu": "valu", "valu_trans": "valu",
    "salu": "scalar", "smem": "scalar",
    "nop": "nop", "branch": "other", "other": "other",
}

#: What each reason means, for the summary and docs/profile.md.
REASON_TEXT = {
    "matrix_core": "at MFMA/WMMA instructions: in a loop's cycles, issuing or waiting for "
                   "the matrix pipe; in the matrix core's idle time, waiting for operands",
    "vmem_wait": "s_waitcnt on vmcnt: global loads (or gfx950's DMA into LDS) not back",
    "lgkm_wait": "s_waitcnt on lgkmcnt: LDS reads/writes (or scalar loads) not back",
    "vmem_lgkm_wait": "s_waitcnt on both counters",
    "other_wait": "s_waitcnt on expcnt or another counter",
    "barrier": "s_barrier: waiting for the workgroup's other waves",
    "vmem_issue": "at buffer/global loads and stores: the address/data path is busy",
    "lds_issue": "at ds_read/ds_write: the LDS queue or bandwidth (or bank conflicts)",
    "valu": "at vector ALU instructions (their dependencies, or the VALU busy)",
    "scalar": "at scalar ALU and scalar-memory instructions",
    "nop": "s_nop: hazard padding the compiler inserted",
    "other": "branches, s_setprio, s_delay_alu and other control",
    "idle": "no instruction to issue (instruction fetch, a wave not yet started)",
    "no_instruction": "between instructions: none ready to issue (instruction fetch, a "
                      "dependency the hardware tracks without an s_waitcnt)",
}


def ceilings(arch: str) -> Dict[str, Any]:
    return CEILINGS.get(arch, {})


# ------------------------------------------------------------------ the grid


def grid_fit(arch: str, agent: Dict[str, Any], grid: Dict[str, Any],
             build: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """How the workgroups fill the GPU: workgroups a slot (a CU on CDNA, a
    WGP -- the LDS pool -- on RDNA) holds at once and what binds it, the
    slots, and the rounds the grid takes. From the dispatch's resources (the
    compiler's, from the build row, where hk built it)."""
    from ..target import get_target  # noqa: PLC0415

    try:
        t = get_target(arch)
    except KeyError:
        return {}
    wave = agent.get("wave") or t.wave
    wg_threads = grid.get("workgroup") or 0
    wgs = grid.get("workgroups") or 0
    if not wg_threads or not wgs:
        return {}
    waves_per_wg = -(-wg_threads // wave)
    vgpr, agpr = grid.get("vgpr") or 0, grid.get("agpr") or 0
    lds = grid.get("lds_bytes") or 0
    occ_regs = None
    inst = (build or {}).get("instantiations") or []
    if inst:
        i0 = inst[0]
        vgpr, agpr = i0.get("vgpr", vgpr), i0.get("agpr", agpr)
        occ_regs = i0.get("occupancy_by_regs")
        lds = max(lds, (build or {}).get("lds_bytes") or 0, i0.get("lds") or 0)
    if occ_regs is None:
        occ_regs = t.occupancy(t.charged_vgprs(vgpr, agpr)) if vgpr else t.max_waves_per_simd
    pool_simds = t.simds_per_lds_pool
    simds = agent.get("simds") or 0
    slots = simds // pool_simds if simds else 0
    lim = {
        "waves": (t.max_waves_per_simd * pool_simds) // waves_per_wg,
        "registers": (occ_regs * pool_simds) // waves_per_wg,
        "lds": (t.lds_pool_bytes // lds) if lds else 10**9,
    }
    per_slot = max(0, min(lim.values()))
    binding = " and ".join(k for k in lim if lim[k] == per_slot)
    out: Dict[str, Any] = {
        "workgroups": wgs, "threads_per_workgroup": wg_threads,
        "waves_per_workgroup": waves_per_wg, "lds_bytes": lds, "vgpr": vgpr, "agpr": agpr,
        "slot": "CU" if pool_simds and arch.startswith("gfx9") else "WGP",
        "slots": slots, "workgroups_per_slot": per_slot, "bound_by": binding,
        "theoretical_waves_per_simd": min(t.max_waves_per_simd,
                                          per_slot * waves_per_wg // pool_simds)
        if pool_simds else None,
    }
    if slots and per_slot:
        cap = slots * per_slot
        rounds = wgs / cap
        out["rounds"] = round(rounds, 3)
        full = math.floor(rounds + 1e-9)
        frac = rounds - full
        if rounds < 1:
            out["idle_share"] = round(1 - rounds, 3)
            out["last_round"] = {"workgroups": wgs, "of": cap}
        elif frac > 1e-6:
            n = math.ceil(rounds)
            # Workgroups of equal length, issued round by round: the share of
            # slot-time the last round leaves empty.
            out["idle_share"] = round((n - rounds) / n, 3)
            out["last_round"] = {"workgroups": wgs - full * cap, "of": cap}
        else:
            out["idle_share"] = 0.0
    return out


# ------------------------------------------------------------------ stalls


def stalls_from_att(region: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A thread-trace region summary (isa.analyze's `att`) -> {reason:
    {"cycles", "stall", "share"}} sorted by cycles, plus `idle`."""
    if not region or "by_class" not in region:
        return None
    total = region["total"]["latency"] or 0
    if not total:
        return None
    out: Dict[str, Dict[str, float]] = {}
    for cls, s in region["by_class"].items():
        r = REASONS.get(cls, "other")
        e = out.setdefault(r, {"cycles": 0, "stall": 0})
        e["cycles"] += s["latency"]
        e["stall"] += s["stall"]
    idle = region["total"].get("idle", 0)
    for e in out.values():
        e["share"] = round(e["cycles"] / total, 4)
    res = dict(sorted(out.items(), key=lambda kv: -kv[1]["cycles"]))
    return {"cycles": total, "idle": idle, "by_reason": res}


# ------------------------------------------------------------------ one kernel


def _dtype_of(isa: Optional[Dict[str, Any]], c: Dict[str, Any]) -> str:
    if c.get("raw_mops"):
        m = c["raw_mops"]
        best = max(m, key=lambda k: m[k])
        return {"SQ_INSTS_VALU_MFMA_MOPS_F8": "fp8", "SQ_INSTS_VALU_MFMA_MOPS_F16": "fp16"}.get(
            best, "bf16")
    return (isa or {}).get("matrix_dtype") or "bf16"


def kernel_report(arch: str, agent: Dict[str, Any], rec: Dict[str, Any], *,
                  build: Optional[Dict[str, Any]] = None,
                  isa: Optional[Dict[str, Any]] = None,
                  call_bytes: Optional[float] = None, flops: Optional[float] = None,
                  bytes_: Optional[float] = None, single: bool = False) -> Dict[str, Any]:
    """The per-kernel JSON (docs/profile.md#the-json): time, grid, roofline,
    matrix core, occupancy, stalls, LDS, caches, ISA, findings, verdict.

    rec      hk.profile.run.collect's kernel record (durations, grid, raw
             counters, thread trace)
    build    the hk build row of this kernel, if hk built it
    isa      isa.analyze()'s summary of its code
    call_bytes  the call's footprint (inputs and outputs once): which memory
             ceiling applies (the cache's if it fits)
    flops, bytes_  the kernel's work if the caller knows it better than the
             counters (a single-kernel call's --flops/--bytes)
    single   the call launches only this kernel (call_bytes is then its bytes
             when the counters have none)
    """
    ceil = ceilings(arch)
    raw = rec.get("counters") or {}
    m = _counters.derive(arch, raw, agent, rec.get("counters_by_pass")) if raw else {}
    dur_ns = rec.get("median_ns") or 0
    dur_s = dur_ns * 1e-9
    out: Dict[str, Any] = {
        "name": rec.get("name"), "hk": build is not None, "launches_per_call":
        max(1, (rec.get("launches") or 1) // max(1, rec.get("iters") or 1)),
        "time": {"us": round(dur_ns / 1e3, 2), "min_us": round(min(rec.get("durations_ns")
                                                                    or [dur_ns]) / 1e3, 2),
                 "noise": rec.get("noise"), "samples": len(rec.get("durations_ns") or [])},
    }
    if build:
        out["build"] = {k: build.get(k) for k in ("kernel", "consts", "warps", "lds_bytes",
                                                   "source", "binary") if k in build}
        if build.get("instantiations"):
            out["build"]["resources"] = build["instantiations"][0]
    g = grid_fit(arch, agent, rec.get("grid") or {}, build)
    out["grid"] = g or {"workgroups": (rec.get("grid") or {}).get("workgroups")}
    if m.get("cycles") and dur_s:
        # The counter pass's own duration is what its cycles were counted in.
        cp = (rec.get("counter_pass_ns") or {}).get("core")
        top = agent.get("max_clock_mhz") or 0
        # GRBM_GUI_ACTIVE includes the dispatch's ramp: for a kernel of a few
        # microseconds it overstates the cycles, so no clock is read there.
        if cp and cp >= 50e3:
            mhz = m["cycles"] / (cp * 1e-9) / 1e6
            if not top or mhz <= top * 1.05:
                m["clock_mhz"] = round(mhz, 0)
    # ---- roofline
    roof: Dict[str, Any] = {}
    fl = flops
    fl_src = "given" if flops else None
    if fl is None and m.get("matrix_flops"):
        fl, fl_src = m["matrix_flops"], "counters (SQ_INSTS_VALU_MFMA_MOPS_* x 512)"
    if fl is None and isa and isa.get("att_matrix_flops_per_wave") and m.get("waves"):
        fl = isa["att_matrix_flops_per_wave"] * m["waves"]
        fl_src = "thread trace (matrix instructions a wave ran x FLOPs each) x SQ_WAVES"
    if fl is None and isa and isa.get("att_matrix_flops_per_wave") and g.get("workgroups"):
        fl = isa["att_matrix_flops_per_wave"] * g["workgroups"] * g["waves_per_workgroup"]
        fl_src = "thread trace (matrix instructions a wave ran x FLOPs each) x the grid's waves"
    dtype = (isa or {}).get("matrix_dtype") or "bf16"
    peak = (ceil.get("matrix_flops") or {}).get(dtype)
    if fl and dur_s:
        roof["flops"] = fl
        roof["flops_source"] = fl_src
        roof["flop_s"] = fl / dur_s
        roof["matrix_dtype"] = dtype
        if peak:
            roof["of_compute_peak"] = round(fl / dur_s / peak, 4)
            roof["compute_peak"] = peak
            # The peak is at the part's top clock; under a power limit the
            # kernel ran slower, and its own peak was lower.
            top = agent.get("max_clock_mhz")
            if m.get("clock_mhz") and top:
                roof["clock_mhz"] = m["clock_mhz"]
                roof["of_compute_peak_at_clock"] = round(
                    fl / dur_s / (peak * min(1.0, m["clock_mhz"] / top)), 4)
    # Memory. Two readings, because a byte past the L2 may come from the MALL
    # (Infinity Cache) or from DRAM and no counter says which:
    #  - the bytes past the L2 (counters) against the cache-resident rate:
    #    how hard the kernel drives the fabric behind the L2;
    #  - the call's own bytes (inputs and outputs once, a single-kernel call)
    #    against DRAM, when they do not fit the cache: DRAM moved at least
    #    that much.
    # Bound on memory if DRAM is at >= MEMORY_BOUND for sure, or the fabric
    # at >= FABRIC_BOUND.
    by = bytes_
    by_src = "given" if bytes_ else None
    if by is None and m.get("fetch_bytes") is not None:
        by = m["fetch_bytes"] + (m.get("write_bytes") or 0)
        by_src = "counters: bytes past the L2, read + written"
    own = call_bytes if single else None
    if by is None and own:
        by, by_src = own, "the call's inputs and outputs, once each (no counters)"
    if by and dur_s:
        roof["bytes"] = by
        roof["bytes_source"] = by_src
        roof["bytes_s"] = by / dur_s
        if ceil:
            roof["of_dram"] = round(by / dur_s / ceil["dram_bytes_s"], 4)
            roof["of_cache"] = round(by / dur_s / ceil["cache_bytes_s"], 4)
            if own and own > ceil["cache_bytes"]:
                roof["dram_bytes_min"] = own
                roof["of_dram_min"] = round(own / dur_s / ceil["dram_bytes_s"], 4)
            if own and by_src.startswith("counters") and by >= 1.8 * own:
                roof["refetch"] = round(by / own, 2)
            # The reading nearer its bar is of_memory_peak; memory_ceiling
            # names it.
            cand = [("cache", roof["of_cache"], ceil["cache_bytes_s"], FABRIC_BOUND)]
            if roof.get("of_dram_min") is not None:
                cand.append(("dram", roof["of_dram_min"], ceil["dram_bytes_s"], MEMORY_BOUND))
            elif not by_src.startswith("counters") and own:
                # No counters, the call's bytes: DRAM unless they fit the cache.
                cand = ([("cache", roof["of_cache"], ceil["cache_bytes_s"], MEMORY_BOUND)]
                        if own <= ceil["cache_bytes"] else
                        [("dram", roof["of_dram"], ceil["dram_bytes_s"], MEMORY_BOUND)])
            which = max(cand, key=lambda c: c[1] / c[3])
            roof["memory_ceiling"] = which[0]
            roof["of_memory_peak"] = which[1]
            roof["memory_peak"] = which[2]
            roof["memory_bound_at"] = which[3]
        if fl:
            roof["intensity"] = round(fl / by, 2)
            if peak and ceil:
                roof["ridge"] = round(peak / ceil["dram_bytes_s"], 2)
    out["roofline"] = roof
    # ---- matrix core
    mc: Dict[str, Any] = {}
    has_matrix = bool((m.get("instructions") or {}).get("mfma") or m.get("mfma_busy")
                      or (isa or {}).get("matrix_dtype"))
    if not has_matrix:
        pass
    elif m.get("mfma_busy") is not None:
        mc = {"busy": m["mfma_busy"], "source": "SQ_VALU_MFMA_BUSY_CYCLES / (cycles x SIMDs)"}
    elif roof.get("of_compute_peak") is not None and arch.startswith("gfx11"):
        mc = {"busy": roof["of_compute_peak"],
              "source": "derived: achieved WMMA FLOP/s over the peak (gfx1100 has no WMMA "
                        "counter)"}
    out["matrix_core"] = mc
    # ---- occupancy
    occ: Dict[str, Any] = {}
    if g.get("theoretical_waves_per_simd") is not None:
        occ["theoretical_waves_per_simd"] = g["theoretical_waves_per_simd"]
        occ["bound_by"] = g.get("bound_by")
    if m.get("achieved_waves_per_simd") is not None:
        occ["achieved_waves_per_simd"] = m["achieved_waves_per_simd"]
        if occ.get("theoretical_waves_per_simd"):
            occ["achieved_of_theoretical"] = round(
                m["achieved_waves_per_simd"] / occ["theoretical_waves_per_simd"], 3)
    if m.get("cu_busy") is not None:
        occ["cu_busy"] = m["cu_busy"]
    if m.get("se_busy") is not None:
        occ["se_busy"] = m["se_busy"]
    out["occupancy"] = occ
    # ---- stalls
    st: Dict[str, Any] = {}
    if m.get("wave_time"):
        st["counters"] = m["wave_time"]
    if isa:
        region = (isa.get("main_loop") or {}).get("att") or \
            (isa.get("straight_line") or {}).get("att")
        a = stalls_from_att(region)
        if a:
            a["region"] = "main loop" if (isa.get("main_loop") or {}).get("att") \
                else "whole kernel"
            st["att"] = a
            st["top"] = [{"reason": k, "share": v["share"]}
                         for k, v in list(a["by_reason"].items())[:4]]
    if isa and isa.get("matrix_idle"):
        st["matrix_idle"] = isa["matrix_idle"]
    if not st.get("top") and m.get("wave_time"):
        wt = m["wave_time"]
        st["top"] = [{"reason": k, "share": wt[k]} for k in
                     sorted(("dependency_wait", "issue_wait", "lds_issue_wait"),
                            key=lambda k: -(wt.get(k) or 0)) if wt.get(k) is not None]
    out["stalls"] = st
    # ---- LDS, caches, latency
    lds: Dict[str, Any] = {}
    for k in ("lds_bank_conflict", "lds_addr_conflict", "lds_unaligned_stall"):
        if m.get(k) is not None:
            lds[k.replace("lds_", "")] = m[k]
    if (m.get("instructions") or {}).get("lds") is not None:
        lds["instructions"] = m["instructions"]["lds"]
    out["lds"] = lds
    cache: Dict[str, Any] = {}
    if m.get("l2_hit") is not None:
        cache["l2_hit"] = m["l2_hit"]
    cache["mall_hit"] = None
    if m.get("fetch_bytes") is not None:
        cache["fetch_bytes"] = m["fetch_bytes"]
    if m.get("write_bytes") is not None:
        cache["write_bytes"] = m["write_bytes"]
    out["cache"] = cache
    if m.get("latency_cycles"):
        out["latency_cycles"] = m["latency_cycles"]
    if m.get("instructions"):
        out["instructions"] = m["instructions"]
        if m.get("instructions_per_wave"):
            out["instructions_per_wave"] = m["instructions_per_wave"]
    if m.get("clock_mhz"):
        out["clock_mhz"] = m["clock_mhz"]
    if isa:
        out["isa"] = isa
    if raw:
        out["counters"] = {k: v for k, v in sorted(raw.items())}
    out["findings"] = findings(arch, out)
    out["verdict"] = verdict(arch, out)
    return out


# ------------------------------------------------------------------ findings


def _pct(x: Optional[float]) -> str:
    return "?" if x is None else f"{x * 100:.0f}%"


def findings(arch: str, k: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What the numbers say, each {kind, cost, text, evidence}: `cost` is
    the share of the kernel's time the finding accounts for, where it can be
    estimated (None where it is a bound, not a loss). Sorted by cost."""
    out: List[Dict[str, Any]] = []
    roof, g, occ, st = k.get("roofline") or {}, k.get("grid") or {}, k.get("occupancy") or {}, \
        k.get("stalls") or {}
    isa = k.get("isa") or {}
    us = (k.get("time") or {}).get("us") or 0
    mem = roof.get("of_memory_peak")
    comp = roof.get("of_compute_peak")
    dram_name = "HBM" if arch.startswith("gfx9") else "DRAM"
    past = "past the L2" in (roof.get("bytes_source") or "")
    where = (f"the cache-resident rate{' (past the L2)' if past else ''}"
             if roof.get("memory_ceiling") == "cache" else dram_name)
    if mem is not None and mem >= (roof.get("memory_bound_at") or MEMORY_BOUND):
        if roof.get("memory_ceiling") == "dram" and roof.get("dram_bytes_min"):
            text = (f"memory-bound: {_pct(mem)} of {dram_name} at least (the call's "
                    f"{roof['dram_bytes_min'] / 1e6:.0f} MB in {us:.1f} us, "
                    f"{roof['dram_bytes_min'] / (us * 1e-6) / 1e12:.2f} TB/s)")
        else:
            text = (f"memory-bound: {roof['bytes_s'] / 1e12:.2f} TB/s "
                    + ("past the L2 " if "past the L2" in (roof.get("bytes_source") or "")
                       else "") + f"= {_pct(mem)} of {where}")
        out.append({"kind": "memory_bound", "cost": None, "text": text,
                    "evidence": {"of_memory_peak": mem, "memory_ceiling":
                                 roof.get("memory_ceiling"), "bytes": roof.get("bytes"),
                                 "bytes_source": roof.get("bytes_source"),
                                 "dram_bytes_min": roof.get("dram_bytes_min")}})
    if roof.get("refetch"):
        out.append({"kind": "l2_refetch", "cost": None,
                    "short": f"L2 re-fetch: {roof['refetch']:.1f}x the call's bytes cross it",
                    "text": f"the L2 re-fetches: {roof['bytes'] / 1e6:.0f} MB cross it, "
                            f"{roof['refetch']:.1f}x the call's inputs and outputs "
                            f"({_pct(roof.get('of_cache'))} of the cache-resident rate): tiles "
                            f"shared by workgroups on different XCDs/L2s, or evicted before "
                            f"their reuse",
                    "evidence": {"bytes": roof["bytes"], "refetch": roof["refetch"],
                                 "of_cache": roof.get("of_cache")}})
    busy = (k.get("matrix_core") or {}).get("busy")
    if (comp is not None and comp >= COMPUTE_BOUND) or (busy is not None and
                                                        busy >= COMPUTE_BOUND):
        text = "compute-bound: "
        if busy is not None:
            text += f"matrix core {_pct(busy)} busy, "
        text += (f"{roof['flop_s'] / 1e12:.0f} TFLOP/s = {_pct(comp)} of the "
                 f"{roof.get('matrix_dtype')} peak" if comp is not None else "")
        if roof.get("of_compute_peak_at_clock") is not None:
            text += (f" ({_pct(roof['of_compute_peak_at_clock'])} of it at the "
                     f"{roof['clock_mhz'] / 1e3:.2f} GHz measured)")
        out.append({"kind": "compute_bound", "cost": None, "text": text,
                    "evidence": {"of_compute_peak": comp, "matrix_core_busy": busy,
                                 "clock_mhz": roof.get("clock_mhz")}})
    # The grid.
    idle = g.get("idle_share")
    if g.get("rounds") is not None and idle:
        lr = g.get("last_round") or {}
        slot = g.get("slot", "CU")
        if g["rounds"] < 1 and g["workgroups"] < g["slots"]:
            cu_idle = round(1 - g["workgroups"] / g["slots"], 4)
            out.append({"kind": "underfilled", "cost": cu_idle,
                        "text": f"underfilled: {g['workgroups']} workgroups for {g['slots']} "
                                f"{slot}s: {_pct(cu_idle)} of the {slot}s have none",
                        "evidence": {k2: g.get(k2) for k2 in ("workgroups", "slots",
                                                              "workgroups_per_slot", "rounds")}})
        elif g["rounds"] < 1:
            pass        # every slot has work; fewer waves than fit is occupancy's to say
        elif g["rounds"] < 16 and idle >= 0.02:
            ev = {k2: g.get(k2) for k2 in ("workgroups", "slots", "workgroups_per_slot",
                                           "rounds", "idle_share")}
            busy = occ.get("cu_busy")
            if busy is not None:
                ev["cu_busy"] = busy
            text = (f"tail: {g['workgroups']} workgroups = {g['rounds']:.2f} rounds of "
                    f"{g['slots']} {slot}s x {g['workgroups_per_slot']}; the last fills "
                    f"{lr.get('workgroups')}/{lr.get('of')}, leaving {_pct(idle)} of the "
                    f"slot-time idle")
            if busy is not None:
                text += f" (CUs measured busy {_pct(busy)} of the kernel)"
            out.append({"kind": "tail", "cost": idle, "text": text, "evidence": ev})
    # The epilogue and prologue, from the thread trace: the matrix core's
    # idle time while the waves are there (timeline), else their share of
    # the waves' cycles.
    per_slot = g.get("workgroups_per_slot")
    mi = st.get("matrix_idle")
    for part in ("epilogue", "prologue"):
        p = isa.get(part) or {}
        share = p.get("cycle_share")
        parts = [part]
        if part == "epilogue" and isa.get("drain"):
            # After the main loop: the drain (the last tiles' MFMAs and, with
            # early stores, part of the epilogue's math) and the epilogue.
            parts = ["drain", "epilogue"]
            if share is not None and (isa["drain"].get("cycle_share") is not None):
                share = round(share + isa["drain"]["cycle_share"], 4)
        by_region = (mi or {}).get("idle_by_region", {})
        idle_there = (round(sum(by_region.get(x, 0.0) for x in parts), 4)
                      if mi else None)
        if (idle_there if idle_there is not None else share or 0) < 0.03:
            continue
        cls: Dict[str, int] = {}
        n_inst = 0
        for x in parts:
            n_inst += (isa.get(x) or {}).get("instructions") or 0
            for c, v in (((isa.get(x) or {}).get("mix") or {}).get("classes") or {}).items():
                cls[c] = cls.get(c, 0) + v
        valu = cls.get("valu", 0) + cls.get("valu_trans", 0)
        stores = cls.get("vmem_store", 0)
        loads = cls.get("vmem_load", 0) + cls.get("vmem_lds", 0)
        what = (f"{n_inst} instructions: {valu} VALU, "
                + (f"{stores} stores" if part == "epilogue" else f"{loads} loads"))
        where_ = ("after the main loop" + (" (drain + epilogue)" if len(parts) > 1 else "")
                  if part == "epilogue" else "before the main loop")
        if idle_there is not None:
            text = (f"{part}: the matrix core idles {_pct(idle_there)} of the kernel while "
                    f"the waves are {where_} ({what})")
            cost = idle_there
        else:
            text = f"{part}: {_pct(share)} of a wave's cycles are {where_} ({what})"
            cost = share if not (per_slot and per_slot > 1) else share / per_slot
        if per_slot == 1:
            text += ("; at one workgroup a slot no other workgroup's main loop overlaps it, "
                     "so it adds to the kernel's time")
        out.append({"kind": part, "cost": round(cost, 4), "text": text,
                    "evidence": {"cycle_share": share, "matrix_idle_share": idle_there,
                                 "regions": parts, "valu": valu, "stores": stores,
                                 "loads": loads, "workgroups_per_slot": per_slot}})
    # Why the matrix core idles, by what the waves are at (timeline).
    if mi:
        for reason, v in list(mi["idle"].items())[:4]:
            if v < 0.04:
                continue
            out.append({"kind": "stall", "cost": round(v, 4), "reason": reason,
                        "text": f"the matrix core idles {_pct(v)} of the kernel with its waves "
                                f"at {reason} ({REASON_TEXT.get(reason, '')})",
                        "evidence": {"reason": reason, "share": v,
                                     "region": "matrix-idle cycles (traced CU)"}})
    # Stalls in the main loop.
    att = st.get("att")
    if att and not mi:
        for reason, v in list(att["by_reason"].items())[:3]:
            if reason == "matrix_core" or v["share"] < 0.1:
                continue
            region_share = (isa.get("main_loop") or {}).get("cycle_share") or 1.0
            out.append({"kind": "stall", "cost": round(v["share"] * region_share, 4),
                        "reason": reason,
                        "text": f"{att['region']}: {_pct(v['share'])} of its cycles at "
                                f"{reason} ({REASON_TEXT.get(reason, '')})",
                        "evidence": {"reason": reason, "share": v["share"],
                                     "region": att["region"]}})
    elif not att and st.get("counters"):
        wt = st["counters"]
        dw = wt.get("dependency_wait")
        if dw is not None and dw >= 0.3:
            out.append({"kind": "stall", "cost": None, "reason": "dependency_wait",
                        "text": f"waves spend {_pct(dw)} of their time waiting on a "
                                f"dependency (s_waitcnt, barrier, a result): the counters "
                                f"cannot say which; run with the thread trace",
                        "evidence": {"dependency_wait": dw,
                                     "issue_wait": wt.get("issue_wait")}})
    # Few waves a SIMD and a main loop heavy in non-matrix instructions:
    # their issue is the matrix core's idle time.
    tw = occ.get("theoretical_waves_per_simd")
    ml = isa.get("main_loop") or {}
    per = (ml.get("mix") or {}).get("per_matrix_instruction")
    if tw is not None and tw <= 2 and per is not None:
        # The matrix core's idle time in the main loop with its waves issuing
        # other instructions (an epilogue's VALU is the epilogue finding's).
        in_loop = ((mi or {}).get("idle_in") or {}).get("main_loop")
        src = in_loop if in_loop is not None else ((mi or {}).get("idle") or {})
        issue = sum(v for r, v in src.items()
                    if r in ("vmem_issue", "lds_issue", "valu", "scalar", "nop", "other"))
        if (mi and issue >= 0.15) or (not mi and per >= 1.5 and tw <= 1):
            cls = (ml.get("mix") or {}).get("classes") or {}
            nm = cls.get("mfma", 0) + cls.get("wmma", 0)
            what = ", ".join(f"{cls[c]} {c}" for c in ("vmem_lds", "vmem_load", "ds_read",
                                                       "ds_write", "valu", "salu")
                             if cls.get(c))
            text = (f"issue-bound: {tw} wave(s) a SIMD (bound by {g.get('bound_by')}), and "
                    f"the main loop issues {per:.1f} other instructions per matrix one "
                    f"({what} against {nm})")
            if mi:
                text += (f"; the matrix core idles {_pct(issue)} of the kernel with its waves "
                         f"issuing them")
            out.append({"kind": "issue_bound", "cost": round(issue, 4) if mi else None,
                        "text": text,
                        "evidence": {"waves_per_simd": tw, "bound_by": g.get("bound_by"),
                                     "per_matrix_instruction": per,
                                     "matrix_idle_issuing": round(issue, 4) if mi else None}})
    # Narrow global accesses in a kernel that waits on memory.
    whole = (ml or {}).get("mix") or (isa.get("total") or {})
    bpl = whole.get("bytes_per_lane") or {}
    narrow = {c: w for c, w in bpl.items() if c in ("vmem_load", "vmem_store")
              and any(int(b) < 16 for b in w)}
    waits_mem = 0.0
    if att:
        waits_mem = sum((att["by_reason"].get(r) or {}).get("share", 0)
                        for r in ("vmem_issue", "vmem_wait"))
    elif st.get("counters"):
        waits_mem = (st["counters"].get("issue_wait") or 0)
    if narrow and (mem is None or mem < MEMORY_BOUND) and waits_mem >= 0.3:
        desc = "; ".join(f"{c.replace('vmem_', '')}s " + ", ".join(
            f"{n} of {b} B" for b, n in sorted(w.items(), key=lambda kv: int(kv[0])))
            for c, w in narrow.items())
        out.append({"kind": "narrow_access", "cost": None,
                    "short": "narrow global accesses (" + "; ".join(
                        f"{c.replace('vmem_', '')}s " + "/".join(f"{b} B" for b in sorted(
                            w, key=int)) for c, w in narrow.items()) + " a lane)",
                    "text": f"narrow global accesses: {desc} a lane, against 16 B "
                            f"(dwordx4) at most; the waves wait on vector memory "
                            f"{_pct(waits_mem)} of their cycles -- wider, contiguous runs "
                            f"move the same bytes in fewer requests",
                    "evidence": {"bytes_per_lane": narrow, "vmem_wait_share": round(waits_mem, 4),
                                 "of_memory_peak": mem}})
    # Occupancy that the grid does not explain.
    aot = occ.get("achieved_of_theoretical")
    if aot is not None and aot < 0.7 and not idle:
        out.append({"kind": "occupancy", "cost": None,
                    "short": f"occupancy {occ['achieved_waves_per_simd']:.1f} of "
                             f"{occ['theoretical_waves_per_simd']} waves/SIMD",
                    "text": f"occupancy: {occ['achieved_waves_per_simd']:.2f} waves/SIMD "
                            f"achieved of {occ['theoretical_waves_per_simd']} the resources "
                            f"allow ({_pct(aot)})",
                    "evidence": dict(occ)})
    lbc = (k.get("lds") or {}).get("bank_conflict")
    if lbc is not None and lbc >= 0.05:
        out.append({"kind": "lds_bank_conflict", "cost": None,
                    "text": f"LDS bank conflicts: {_pct(lbc)} of the LDS's active cycles",
                    "evidence": {"bank_conflict": lbc}})
    noise = (k.get("time") or {}).get("noise")
    if noise is not None and noise > 0.05:
        out.append({"kind": "noisy", "cost": None,
                    "short": f"timing noisy ({_pct(noise)} spread)",
                    "text": f"the timing is noisy: the calls' spread is {_pct(noise)} of the "
                            f"median (min {(k.get('time') or {}).get('min_us')} us): another "
                            f"job on the GPU, or its clocks moving; re-run before trusting the "
                            f"time",
                    "evidence": {"noise": noise}})
    if us and us < 8:
        out.append({"kind": "small", "cost": None,
                    "text": f"{us:.1f} us: launch and ramp are a large part of a kernel this "
                            f"short",
                    "evidence": {"us": us}})
    res = (k.get("build") or {}).get("resources") or {}
    if res.get("scratch") or res.get("spill"):
        out.append({"kind": "spill", "cost": None,
                    "text": f"scratch {res.get('scratch')} B, {res.get('spill')} spills",
                    "evidence": {"scratch": res.get("scratch"), "spill": res.get("spill")}})
    has_bound = any(f["kind"] in ("memory_bound", "compute_bound") for f in out)
    if not has_bound and (mem is not None or comp is not None):
        text = "latency-bound (neither roof): " + ", ".join(
            x for x in ((f"{_pct(comp)} of the matrix peak" if comp is not None else ""),
                        (f"{_pct(mem)} of {where}" if mem is not None else "")) if x)
        out.append({"kind": "latency_bound", "cost": None, "text": text,
                    "evidence": {"of_compute_peak": comp, "of_memory_peak": mem}})
    out.sort(key=lambda f: (f["kind"] not in ("memory_bound", "compute_bound",
                                              "latency_bound"),
                            -(f["cost"] or 0)))
    return out


def verdict(arch: str, k: Dict[str, Any]) -> Dict[str, Any]:
    """The one line: what bounds the kernel, then what it loses, largest
    first -- {line, bound, losses: [kind...]}."""
    fs = k.get("findings") or findings(arch, k)
    roof = k.get("roofline") or {}
    mc = (k.get("matrix_core") or {}).get("busy")
    bound = next((f for f in fs if f["kind"] in ("memory_bound", "compute_bound",
                                                 "latency_bound")), None)
    parts = []
    if bound is not None:
        parts.append(bound["text"])
    elif roof.get("of_compute_peak") is not None:
        parts.append(f"{_pct(roof['of_compute_peak'])} of the matrix peak")
    if mc is not None and not any("matrix core" in p for p in parts) and             (bound is None or bound["kind"] != "memory_bound"):
        parts.append(f"matrix core {_pct(mc)} busy")
    losses = [f for f in fs if f.get("cost")]
    # An issue-bound finding is the sum of its stalls: say it, not them.
    if any(f["kind"] == "issue_bound" for f in losses):
        losses = [f for f in losses if not (f["kind"] == "stall" and f.get("reason") in (
            "vmem_issue", "lds_issue", "valu", "scalar", "nop", "other"))]
    for f in losses[:3]:
        if f["kind"] == "tail":
            g = k.get("grid") or {}
            lr = g.get("last_round") or {}
            parts.append(f"tail: {g.get('rounds'):.2f} rounds of {g.get('slots')} "
                         f"{g.get('slot', 'CU')}s, the last fills {lr.get('workgroups')}/"
                         f"{lr.get('of')} ({_pct(f['cost'])} idle)")
        elif f["kind"] == "underfilled":
            parts.append(f"underfilled: {_pct(f['cost'])} of the "
                         f"{(k.get('grid') or {}).get('slot', 'CU')}s have no workgroup")
        elif f["kind"] == "issue_bound":
            ev = f["evidence"]
            parts.append(f"issue-bound at {ev['waves_per_simd']} wave(s) a SIMD ("
                         f"{ev['bound_by']}): the matrix core idles {_pct(f['cost'])} while its "
                         f"waves issue the loop's {ev['per_matrix_instruction']:.1f} other "
                         f"instructions per matrix one")
        elif f["kind"] in ("epilogue", "prologue"):
            ev = f["evidence"]
            if ev.get("matrix_idle_share") is not None:
                where_ = ("after the main loop" if f["kind"] == "epilogue"
                          else "before the main loop")
                parts.append(f"matrix core idle {where_} {_pct(ev['matrix_idle_share'])}"
                             + (" (one workgroup a CU: not hidden)"
                                if ev.get("workgroups_per_slot") == 1 else ""))
            else:
                parts.append(f"{f['kind']} {_pct(ev['cycle_share'])} of a wave's cycles"
                             + (" unhidden" if ev.get("workgroups_per_slot") == 1 else ""))
        elif f["kind"] == "stall" and "matrix-idle" in str(f["evidence"].get("region")):
            parts.append(f"matrix core idle at {f['evidence']['reason']} "
                         f"{_pct(f['evidence']['share'])}")
        elif f["kind"] == "stall":
            parts.append(f"waves at {f['evidence']['reason']} {_pct(f['evidence']['share'])} "
                         f"of their cycles ({f['evidence']['region']})")
    for f in fs:
        if f["kind"] in ("noisy", "narrow_access", "l2_refetch", "small", "spill",
                         "lds_bank_conflict",
                         "occupancy") and len(parts) < 6:
            parts.append(f.get("short") or f["text"])
    return {"line": "; ".join(parts) if parts else "no verdict: nothing was measured",
            "bound": bound["kind"] if bound else None,
            "losses": [f["kind"] for f in losses]}


__all__ = ["CEILINGS", "KINDS", "REASONS", "REASON_TEXT", "ceilings", "grid_fit",
           "stalls_from_att", "kernel_report", "findings", "verdict"]
