"""Which hardware counters each arch is asked for, and what they mean.

rocprofv3 collects one group of counters per run of the program (a *pass*):
a group must fit the counter registers of each block, and a group that does
not makes rocprofv3 abort the program ("Request exceeds the capabilities of
the hardware to collect"). The groups below were each run on the part they
are for (MI355X, ROCm 7.2.3; W7900D, ROCm 7.2.4) and fit; `derive()` turns
their raw sums into the metrics hk.profile reports.

How a raw value is read, measured rather than taken from the counter's
description where the two disagree:

* GRBM_GUI_ACTIVE is summed over an MI355X's 8 XCDs: a kernel's cycles are
  it divided by `Num_Xcc`.
* SQ_WAVE_CYCLES, SQ_WAIT_ANY, SQ_WAIT_INST_ANY, SQ_ACTIVE_INST_* and
  SQ_WAIT_INST_LDS are in quad-cycles (x4 gives wave-cycles), summed over
  the SIMDs.
* SQ_BUSY_CU_CYCLES (gfx950) reads as cycles summed over CUs, though its
  description says quad-cycles: x4 puts a full GEMM's CUs at 317% busy, x1
  at 79% -- the 1.75-wave grid's expected 87.5% less its ramp.
* SQ_INSTS_* count wave instructions (a 16x16x32 bf16 GEMM's SQ_INSTS_MFMA
  is exactly 2MNK / 16384); SQ_INSTS_VALU_MFMA_MOPS_* count FLOPs / 512.
* Bytes past the L2 on gfx950 are TCC_EA0_RDREQ_DRAM_32B and
  TCC_EA0_WRREQ_WRITE_DRAM_32B x 32: a 4096x14336 bf16 elementwise kernel
  reads exactly its two inputs and writes its output by them, where
  rocprofv3's FETCH_SIZE gives half the reads (it counts the 128-byte
  requests as 64).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Per arch: [(pass name, [counters])]. Each pass is a run of the program.
PASSES: Dict[str, List[Tuple[str, List[str]]]] = {
    "gfx950": [
        ("core", ["GRBM_GUI_ACTIVE", "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CU_CYCLES",
                  "SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_INSTS_SALU", "SQ_INSTS_LDS",
                  "TCC_EA0_RDREQ_DRAM_32B", "TCC_EA0_WRREQ_WRITE_DRAM_32B",
                  "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"]),
        ("mix", ["GRBM_GUI_ACTIVE", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_SMEM",
                 "SQ_VALU_MFMA_BUSY_CYCLES", "SQ_INSTS_VALU_MFMA_MOPS_BF16",
                 "SQ_INSTS_VALU_MFMA_MOPS_F16", "SQ_INSTS_VALU_MFMA_MOPS_F8",
                 "SQ_INSTS_BRANCH", "TCC_BUBBLE_sum", "TCC_HIT_sum", "TCC_MISS_sum"]),
        ("wait", ["GRBM_GUI_ACTIVE", "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_ACTIVE_INST_ANY",
                  "SQ_ACTIVE_INST_VALU", "SQ_ACTIVE_INST_VMEM", "SQ_ACTIVE_INST_LDS",
                  "SQ_ACTIVE_INST_SCA", "SQ_ACTIVE_INST_MISC"]),
        ("lds", ["GRBM_GUI_ACTIVE", "SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE",
                 "SQ_LDS_ADDR_CONFLICT", "SQ_LDS_UNALIGNED_STALL", "SQ_WAIT_INST_LDS",
                 "SQ_IFETCH", "SQ_INSTS_VALU_MFMA_MOPS_F32"]),
        ("latency", ["VmemLatency", "LdsLatency"]),
    ],
    "gfx1100": [
        ("core", ["GRBM_GUI_ACTIVE", "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES",
                  "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_LDS", "SQ_INSTS_SMEM",
                  "GL2C_HIT_sum", "GL2C_MISS_sum", "GL2C_EA_RDREQ_32B_sum",
                  "GL2C_EA_RDREQ_64B_sum"]),
        ("wait", ["GRBM_GUI_ACTIVE", "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_WAIT_INST_LDS",
                  "SQ_INSTS_TEX_LOAD", "SQ_INSTS_TEX_STORE", "SQ_INSTS_FLAT",
                  "SQ_INST_CYCLES_VMEM", "GL2C_EA_RDREQ_96B_sum", "GL2C_EA_RDREQ_128B_sum",
                  "GL2C_EA_WRREQ_64B_sum", "GL2C_MC_WRREQ_sum"]),
        ("lds", ["GRBM_GUI_ACTIVE", "SQC_LDS_BANK_CONFLICT", "SQC_LDS_IDX_ACTIVE",
                 "TA_TA_BUSY_max"]),
        ("latency", ["LdsLatency"]),
    ],
}

#: What each arch's counters cannot attribute, said in the report.
CANNOT: Dict[str, List[str]] = {
    "gfx950": [
        "MALL (Infinity Cache) hits: the L2's misses all count as requests to the fabric "
        "(TCC_EA0_RDREQ_DRAM = TCC_EA0_RDREQ); a MALL hit and an HBM read look alike, so "
        "`dram_bytes` is bytes past the L2",
        "stall reasons by instruction: the counters give the share of wave time waiting "
        "for a dependency (s_waitcnt, barrier) or to issue, not which; the thread trace "
        "(att) attributes them per instruction on one CU",
    ],
    "gfx1100": [
        "matrix-core (WMMA) activity: no counter counts WMMA instructions or busy cycles; "
        "hk.profile counts them from the thread trace (executions x FLOPs each) or reports "
        "the achieved FLOP/s against the peak instead",
        "MFMA-style per-type instruction counts: only VALU, SALU, LDS, SMEM, texture "
        "load/store and FLAT are counted",
        "Infinity Cache (MALL) hits: GL2C_EA/MC requests are past the L2, whether the "
        "MALL or GDDR6 served them",
        "per-type active cycles (SQ_ACTIVE_INST_*), VMEM latency and CU busy: not exposed "
        "on gfx11; the thread trace attributes the waits",
    ],
}

#: Raw counters in quad-cycles (x4 = cycles).
QUAD = ("SQ_WAVE_CYCLES", "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY", "SQ_ACTIVE_INST_ANY",
        "SQ_ACTIVE_INST_VALU", "SQ_ACTIVE_INST_VMEM", "SQ_ACTIVE_INST_LDS",
        "SQ_ACTIVE_INST_SCA", "SQ_ACTIVE_INST_MISC", "SQ_WAIT_INST_LDS",
        "SQ_INST_CYCLES_VMEM")

_MOPS = ("SQ_INSTS_VALU_MFMA_MOPS_BF16", "SQ_INSTS_VALU_MFMA_MOPS_F16",
         "SQ_INSTS_VALU_MFMA_MOPS_F8", "SQ_INSTS_VALU_MFMA_MOPS_F32",
         "SQ_INSTS_VALU_MFMA_MOPS_F6F4", "SQ_INSTS_VALU_MFMA_MOPS_I8",
         "SQ_INSTS_VALU_MFMA_MOPS_XF32")


def passes(arch: str) -> List[Tuple[str, List[str]]]:
    """The counter groups for `arch` ([] where none are known)."""
    return list(PASSES.get(arch, []))


def _r(x: Optional[float], n: int = 4) -> Optional[float]:
    return None if x is None else round(x, n)


def derive(arch: str, raw: Dict[str, float], agent: Dict[str, Any]) -> Dict[str, Any]:
    """{metric: value} from one kernel's raw counters (medians over the
    profiled dispatches) and the agent's facts (Cu_Count, Simd_Count,
    Num_Xcc). A metric whose counters are missing is absent."""
    g = raw.get
    cus = float(agent.get("cus") or 0) or None
    simds = float(agent.get("simds") or 0) or None
    xcc = float(agent.get("xcc") or 1)
    ses = float(agent.get("ses") or 0) or None
    out: Dict[str, Any] = {}
    cyc = g("GRBM_GUI_ACTIVE")
    cycles = cyc / xcc if cyc else None
    out["cycles"] = _r(cycles, 0)

    def q(name):
        v = g(name)
        return None if v is None else v * (4 if name in QUAD else 1)

    wave_cycles = q("SQ_WAVE_CYCLES")
    if g("SQ_WAVES") is not None:
        out["waves"] = int(g("SQ_WAVES"))
    if wave_cycles and cycles and simds:
        out["achieved_waves_per_simd"] = _r(wave_cycles / cycles / simds, 3)
    if g("SQ_BUSY_CU_CYCLES") is not None and cycles and cus:
        out["cu_busy"] = _r(g("SQ_BUSY_CU_CYCLES") / (cycles * cus), 3)
    if g("SQ_BUSY_CYCLES") is not None and cycles and ses and arch.startswith("gfx11"):
        out["se_busy"] = _r(g("SQ_BUSY_CYCLES") / (cycles * ses), 3)
    if g("SQ_VALU_MFMA_BUSY_CYCLES") is not None and cycles and simds:
        out["mfma_busy"] = _r(g("SQ_VALU_MFMA_BUSY_CYCLES") / (cycles * simds), 3)
    mops = [g(n) for n in _MOPS if g(n) is not None]
    if mops:
        out["matrix_flops"] = sum(mops) * 512
    insts = {}
    for k, name in (("valu", "SQ_INSTS_VALU"), ("mfma", "SQ_INSTS_MFMA"),
                    ("salu", "SQ_INSTS_SALU"), ("lds", "SQ_INSTS_LDS"),
                    ("smem", "SQ_INSTS_SMEM"), ("vmem_load", "SQ_INSTS_VMEM_RD"),
                    ("vmem_store", "SQ_INSTS_VMEM_WR"), ("branch", "SQ_INSTS_BRANCH"),
                    ("vmem_load", "SQ_INSTS_TEX_LOAD"), ("vmem_store", "SQ_INSTS_TEX_STORE"),
                    ("flat", "SQ_INSTS_FLAT")):
        if g(name) is not None:
            insts[k] = int(g(name))
    if insts:
        out["instructions"] = insts
        if out.get("waves"):
            out["instructions_per_wave"] = {k: _r(v / out["waves"], 1) for k, v in insts.items()}
    # Bytes past the L2. On gfx950 in 32-byte units (a 64-byte request
    # counts 2, a 128-byte one 4): rocprofv3's FETCH_SIZE expression counts a
    # 128-byte read as 64 bytes there -- half of a streaming kernel's known
    # input -- while TCC_EA0_RDREQ_DRAM_32B x 32 is exactly it.
    if arch.startswith("gfx9"):
        if g("TCC_EA0_RDREQ_DRAM_32B") is not None:
            out["fetch_bytes"] = g("TCC_EA0_RDREQ_DRAM_32B") * 32
        if g("TCC_EA0_WRREQ_WRITE_DRAM_32B") is not None:
            out["write_bytes"] = g("TCC_EA0_WRREQ_WRITE_DRAM_32B") * 32
        hit, miss = g("TCC_HIT_sum"), g("TCC_MISS_sum")
    else:
        parts = [(g("GL2C_EA_RDREQ_32B_sum"), 32), (g("GL2C_EA_RDREQ_64B_sum"), 64),
                 (g("GL2C_EA_RDREQ_96B_sum"), 96), (g("GL2C_EA_RDREQ_128B_sum"), 128)]
        if all(v is not None for v, _ in parts):
            out["fetch_bytes"] = sum(v * s for v, s in parts)
        wr, w64 = g("GL2C_MC_WRREQ_sum"), g("GL2C_EA_WRREQ_64B_sum")
        if wr is not None and w64 is not None:
            out["write_bytes"] = (wr - w64) * 32 + w64 * 64
        hit, miss = g("GL2C_HIT_sum"), g("GL2C_MISS_sum")
    if hit is not None and miss is not None and hit + miss > 0:
        out["l2_hit"] = _r(hit / (hit + miss), 3)
    # Where the waves' time went, as shares of their wave-cycles.
    if wave_cycles:
        sh = {}
        for k, name in (("dependency_wait", "SQ_WAIT_ANY"), ("issue_wait", "SQ_WAIT_INST_ANY"),
                        ("lds_issue_wait", "SQ_WAIT_INST_LDS"), ("active", "SQ_ACTIVE_INST_ANY"),
                        ("active_valu", "SQ_ACTIVE_INST_VALU"),
                        ("active_vmem", "SQ_ACTIVE_INST_VMEM"),
                        ("active_lds", "SQ_ACTIVE_INST_LDS"),
                        ("active_scalar", "SQ_ACTIVE_INST_SCA"),
                        ("active_branch_msg", "SQ_ACTIVE_INST_MISC")):
            v = q(name)
            if v is not None:
                sh[k] = _r(v / wave_cycles, 3)
        if sh:
            out["wave_time"] = sh
    bc = g("SQ_LDS_BANK_CONFLICT") if g("SQ_LDS_BANK_CONFLICT") is not None \
        else g("SQC_LDS_BANK_CONFLICT")
    act = g("SQ_LDS_IDX_ACTIVE") if g("SQ_LDS_IDX_ACTIVE") is not None \
        else g("SQC_LDS_IDX_ACTIVE")
    if bc is not None and act:
        out["lds_bank_conflict"] = _r(bc / act, 4)
        out["lds_active_cycles"] = act
    if g("SQ_LDS_ADDR_CONFLICT") is not None and act:
        out["lds_addr_conflict"] = _r(g("SQ_LDS_ADDR_CONFLICT") / act, 4)
    if g("SQ_LDS_UNALIGNED_STALL") is not None and act:
        out["lds_unaligned_stall"] = _r(g("SQ_LDS_UNALIGNED_STALL") / act, 4)
    lat = {}
    for k, name in (("vmem", "VmemLatency"), ("lds", "LdsLatency"), ("smem", "SmemLatency"),
                    ("ifetch", "InstrFetchLatency")):
        if g(name) is not None:
            lat[k] = _r(g(name), 1)
    if lat:
        out["latency_cycles"] = lat
    if g("TA_TA_BUSY_max") is not None and cycles:
        out["ta_busy"] = _r(g("TA_TA_BUSY_max") / cycles, 3)
    return out


__all__ = ["PASSES", "CANNOT", "passes", "derive", "QUAD"]
