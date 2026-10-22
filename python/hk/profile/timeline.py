"""When the matrix core sat idle, what were the waves doing?

A thread trace records, for every wave on the traced CU, each instruction's
start cycle, its stall and its duration. Per SIMD this module rebuilds when
the matrix pipe was busy -- each MFMA/WMMA occupies it for FLOPs / (the
SIMD's FLOPs a cycle) cycles from its issue -- and attributes every cycle it
was not to what the SIMD's waves were at in that cycle (split evenly
between them): an s_waitcnt on vmcnt, a barrier, the epilogue's VALU, no
wave at all. That is the answer to "the matrix core is 65% busy: why not
100%", per reason and per part of the kernel (prologue, main loop, drain,
epilogue).

A model, not a counter: the pipe occupancy per instruction is the peak
rate's (exact for back-to-back MFMAs, which issue 16 cycles apart on gfx950
at 16x16x32 bf16), and only the traced CU's SIMDs are seen.
"""

from __future__ import annotations

import bisect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Matrix FLOPs one SIMD does a cycle at the peak, per arch and dtype:
#: gfx950 2.5 PFLOP/s bf16 = 1024 SIMDs x 2.4 GHz x 1024; gfx1100 122.6
#: TFLOP/s = 192 SIMD32s x 2.5 GHz x 256.
SIMD_FLOPS_PER_CYCLE: Dict[str, Dict[str, int]] = {
    "gfx950": {"bf16": 1024, "fp16": 1024, "fp8": 2048},
    "gfx942": {"bf16": 512, "fp16": 512, "fp8": 1024},
    "gfx1100": {"bf16": 256, "fp16": 256},
}


def matrix_idle(waves: Sequence[Dict[str, Any]], *, arch: str,
                label: Callable[[int], Tuple[str, str]],
                pipe_cycles: Callable[[int], int],
                is_matrix: Callable[[int], bool]) -> Optional[Dict[str, Any]]:
    """waves: [{"cu", "simd", "begin", "end", "instructions": [[t, type,
    stall, dur, code_index], ...]}]; label(code_index) -> (reason, region);
    pipe_cycles(code_index) -> cycles the matrix pipe is busy for it.

    Returns {"simds", "span_cycles", "busy", "idle": {reason: share of the
    span}, "idle_by_region": {region: share}, "no_wave": share} summed over
    the traced SIMDs, or None when there is nothing to attribute."""
    by_simd: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for w in waves:
        if not w.get("instructions"):
            continue
        by_simd.setdefault((w.get("cu", 0), w.get("simd", 0)), []).append(w)
    if not by_simd:
        return None
    lo = min(w["begin"] for ws in by_simd.values() for w in ws)
    hi = max(w["end"] for ws in by_simd.values() for w in ws)
    span_total = 0
    busy_total = 0
    idle: Dict[str, float] = {}
    idle_region: Dict[str, float] = {}
    no_wave = 0.0
    any_matrix = False
    for _key, ws in by_simd.items():
        # The pipe's busy intervals on this SIMD.
        iv: List[Tuple[int, int]] = []
        # Each wave's instruction starts and ends, for "what is it at, at t".
        tracks = []
        for w in ws:
            ins = w["instructions"]
            starts = [e[0] for e in ins]
            tracks.append((w["begin"], w["end"], starts, ins))
            for e in ins:
                if is_matrix(e[4]):
                    any_matrix = True
                    s = e[0] + (e[2] or 0)
                    iv.append((s, s + max(1, pipe_cycles(e[4]))))
        iv.sort()
        merged: List[List[int]] = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        span = hi - lo
        span_total += span
        busy = sum(min(b, hi) - max(a, lo) for a, b in merged if b > lo and a < hi)
        busy_total += busy
        # The idle segments: [lo, hi) minus the busy intervals, cut at every
        # wave's instruction boundaries so each piece has one state a wave.
        gaps = []
        cur = lo
        for a, b in merged:
            if a > cur:
                gaps.append((cur, min(a, hi)))
            cur = max(cur, b)
        if cur < hi:
            gaps.append((cur, hi))
        for a, b in gaps:
            if b <= a:
                continue
            cuts = {a, b}
            for (wb, we, starts, ins) in tracks:
                if we <= a or wb >= b:
                    continue
                cuts.add(max(a, wb))
                cuts.add(min(b, we))
                i = bisect.bisect_right(starts, a)
                j = bisect.bisect_left(starts, b)
                for k in range(max(0, i - 1), min(len(ins), j + 1)):
                    t0 = ins[k][0]
                    t1 = t0 + (ins[k][3] or 0)
                    if a < t0 < b:
                        cuts.add(t0)
                    if a < t1 < b:
                        cuts.add(t1)
            pts = sorted(cuts)
            for x, y in zip(pts, pts[1:]):
                states = []
                for (wb, we, starts, ins) in tracks:
                    if not (wb <= x < we):
                        continue
                    k = bisect.bisect_right(starts, x) - 1
                    if k < 0:
                        states.append(("no_instruction", "prologue"))
                        continue
                    e = ins[k]
                    if x < e[0] + (e[3] or 0):
                        states.append(label(e[4]))
                    else:
                        nxt = ins[k + 1][4] if k + 1 < len(ins) else e[4]
                        states.append(("no_instruction", label(nxt)[1]))
                d = y - x
                if not states:
                    no_wave += d
                    continue
                share = d / len(states)
                for reason, region in states:
                    idle[reason] = idle.get(reason, 0.0) + share
                    idle_region[region] = idle_region.get(region, 0.0) + share
    if not any_matrix or not span_total:
        return None
    return {
        "simds": len(by_simd), "span_cycles": hi - lo,
        "busy": round(busy_total / span_total, 4),
        "idle": {k: round(v / span_total, 4) for k, v in
                 sorted(idle.items(), key=lambda kv: -kv[1])},
        "idle_by_region": {k: round(v / span_total, 4) for k, v in
                           sorted(idle_region.items(), key=lambda kv: -kv[1])},
        "no_wave": round(no_wave / span_total, 4),
        "model": "pipe busy FLOPs/(SIMD peak FLOPs a cycle) cycles from each matrix "
                 "instruction's issue; idle cycles split evenly over the SIMD's waves' "
                 "current instruction",
    }


__all__ = ["matrix_idle", "SIMD_FLOPS_PER_CYCLE"]
