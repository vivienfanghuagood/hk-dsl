"""An analytic estimate of a GEMM candidate's time: the order the search
compiles and times candidates in, never a decision.

A schedule space holds thousands of points and a compile is seconds, so the
search compiles a few dozen; which few dozen is this model's job. It knows
what can be known without a GPU:

  * **wave quantization** -- workgroups over the CUs that run them at once;
    a grid of 1.75 waves pays for two (the gate_up-at-1024 case of
    docs/design/tuning-goal.md);
  * **padding** -- a 256-row block at M = 300 computes 512 rows;
  * **tile efficiency** -- operand bytes per MFMA fall as a block grows
    (bm * bn / (bm + bn));
  * **split-K** -- more workgroups for an fp32 round trip and a second launch;
  * **stream-K** -- no part wave, for a slower main loop (the runs of an XCD
    are at different K, so its L2 serves them less) and the cut blocks'
    fp32 shares;
  * **the memory floor** -- every weight byte is read once, which is the
    whole story at decode sizes.

It is calibrated on nothing but the cards' datasheets and the shape of the
kernels; the tuning split (bench/tune) is where its ordering was checked
against measurements, and the measured rungs decide.
"""

from __future__ import annotations

import math
from typing import Any, Dict

#: CUs, dense bf16 FLOP/s, sustained HBM bytes/s, LDS a CU, launch cost (s).
CARDS = {
    "gfx950": dict(cus=256, peak=2.3e15, bw=6.5e12, lds=163840, launch=3e-6, c0=40.0),
    "gfx1100": dict(cus=96, peak=1.0e14, bw=8.0e11, lds=65536, launch=4e-6, c0=20.0),
}


#: Stream-K's main loop against the data-parallel one's, per block.
SK_SLOW = 1.3


def _tile_eff(bm: int, bn: int, c0: float) -> float:
    i = bm * bn / (bm + bn)
    return i / (i + c0)


def estimate(kind: str, s: Dict[str, Any], m: int, n: int, k: int, arch: str) -> float:
    """Seconds, roughly, for candidate (kind, full schedule) at M x N x K."""
    c = CARDS[arch]
    bytes_once = (m * k + n * k + m * n) * 2
    if kind == "decode":
        tokens = s.get("tokens", 1)
        splits = s.get("splits", 1)
        t = n * k * 2 / c["bw"] * (1.0 + 0.08 * (tokens - 1))
        # Too few workgroups to stream the weights at the card's rate: one a
        # 16-row block of W, times the K split.
        wgs = (n // 16) * splits
        fill = min(1.0, wgs / (2 * c["cus"]))
        return t / max(fill, 0.05) + c["launch"] * (1 + (splits > 1) * 0.5)
    if kind == "ppw":
        return float("inf")             # composite; a seed, never generated
    bm, bn = s["block_m"], s["block_n"]
    splits = s.get("splits", 1)
    mb = math.ceil(m / bm)
    nb = math.ceil(n / bn)
    wgs = mb * nb * splits
    if kind == "pp":
        conc = 1
        # A warp's accumulator: operand rows a warp reads per element fall
        # as it grows; 64 x 32 (the 256 x 256 block, 2 x 4 warps) is the
        # reference. 16 x 80 (256 x 160 on 8 x 1) measured 0.76 of it.
        wm, wn = s.get("warps_m", 2), s.get("warps_n", 4)
        rm, rn = max(bm // 2 // wm, 1), max(bn // 2 // wn, 1)
        r = rm * rn / (rm + rn)
        eff = (r / (r + 20.0)) / ((64 * 32 / 96) / (64 * 32 / 96 + 20.0))
    elif kind == "dp":
        lds = s.get("stages", 3) * (bm + bn) * s.get("k_step", 64) * 2
        conc = max(1, min(4, c["lds"] // max(lds, 1)))
        warps = s.get("warps_m", 2) * s.get("warps_n", 2)
        eff = 0.80 if warps >= 4 else 0.65
    else:                               # rdna
        lds = s.get("lds_buffers", 1) * (bm + bn) * s.get("k_step", 64) * 2
        conc = max(1, min(3, (2 * c["lds"]) // max(lds, 1) // 2 + 1))
        eff = 1.0 if s.get("warps", 8) >= 8 else 0.85
    slots = c["cus"] * conc
    waves = math.ceil(wgs / slots)
    per_cu = c["peak"] / c["cus"]
    wg_flops = 2.0 * bm * bn * (k / splits)
    t_wave = conc * wg_flops / (per_cu * _tile_eff(bm, bn, c["c0"]) * eff)
    t_compute = waves * t_wave
    sk = s.get("stream_k", 0) if kind == "pp" else 0
    shares = 0
    if sk:
        blocks = mb * nb
        whole = (blocks // slots) * slots if sk == 2 else 0
        cut = blocks - whole
        # The cut blocks' K spread evenly over every CU, at SK_SLOW the
        # rate (measured 1.2-1.5x on the tuning split), and each run's
        # second block a prologue and a share more.
        t_compute = (whole // slots) * t_wave + (cut / slots) * t_wave * SK_SLOW
        if cut:
            t_compute += t_wave * min(1.0, 4.0 / max(k // 128, 1))
            shares = min(cut * 2, 2 * slots)
    # A is read once per column block, B once per row block, through L2: count
    # the re-reads at an eighth.
    traffic = bytes_once + (m * k * 2 * (nb - 1) + n * k * 2 * (mb - 1)) / 8
    if splits > 1:
        traffic += m * n * 4 * splits * 2
    traffic += shares * bm * bn * 4 * 2
    t_mem = traffic / c["bw"]
    launches = 1 + (splits > 1)
    return max(t_compute, t_mem) + launches * c["launch"]
