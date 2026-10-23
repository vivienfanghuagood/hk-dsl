"""A grouped GEMM in Triton, tuned per shape: T4's "tuned Triton" baseline.

out[o[g]:o[g+1]] = x[o[g]:o[g+1]] @ w[g].T, w as (E, N, K). The kernel is the
tutorial matmul (masked tiles, grouped launch order) over a virtual list of
row tiles: each expert's rows are cut into BLOCK_M tiles, `tile_end[g]` is the
running count of tiles through expert g (three small torch ops on the
offsets, no host synchronisation), and a program finds its expert by a binary
search over it. The grid is sized for the most tiles any routing can make
(cdiv(rows, BLOCK_M) + E); programs past the routing's count return.

The tuning space is bench/tune/triton_gemm.py's (block sizes, warps, stages,
`waves_per_eu`, and on CDNA the MFMA size and `kpack`), pruned to the
problem. Each config is timed queued behind a GPU spin, the median of a few
rounds, on the first call for a (rows, N, K, E) -- so a shape is tuned once,
on the routing it first sees. (triton.autotune is not used: the tile list
depends on BLOCK_M, which its grid callback cannot give each config.)
"""

from __future__ import annotations

import statistics

import torch
import triton
import triton.language as tl

LDS = {"gfx950": 160 * 1024, "gfx1100": 64 * 1024}
_PICK = {}


def _arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def configs(arch: str):
    out = []
    cdna = not arch.startswith("gfx11")
    for bm in (32, 64, 128, 256):
        for bn in (64, 128, 256):
            for bk in ((64, 128) if cdna else (32, 64)):
                for w in (4, 8):
                    for s in ((2,) if cdna else (1, 2)):
                        if (bm + bn) * bk * 2 * s > LDS.get(arch, 64 * 1024):
                            continue
                        if bm * bn // (w * (64 if cdna else 32)) > 256:
                            continue
                        if bm * bn < 64 * 128 and w == 8:
                            continue
                        for wpe in ((0, 2) if cdna else (0,)):
                            kw = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8,
                                  "waves_per_eu": wpe}
                            if cdna:
                                kw.update(matrix_instr_nonkdim=16, kpack=1)
                            out.append(triton.Config(kw, num_warps=w, num_stages=s))
    return out


def prune(cfgs, rows: int, n: int, k: int, e: int):
    """Rows per expert bound BLOCK_M; N bounds BLOCK_N; K bounds BLOCK_K."""
    per = max(1, rows // max(1, e))
    keep = [c for c in cfgs
            if c.kwargs["BLOCK_M"] <= max(32, 2 * triton.next_power_of_2(per))
            and c.kwargs["BLOCK_N"] <= max(64, triton.next_power_of_2(n))
            and c.kwargs["BLOCK_K"] <= max(32, k)]
    return keep or cfgs


def _bench(fn, rounds=3, calls=3):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        torch.cuda._sleep(5_000_000)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(calls):
            fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / calls)
    return statistics.median(ts)


@triton.jit
def _body(x_ptr, w_ptr, o_ptr, offs_ptr, tend_ptr, ROWS, N, K, E,
          s_xm, s_we, s_wn, s_om,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(ROWS, BLOCK_M) + E
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    t = first_m + (pid % group) % gsize
    pid_n = (pid % group) // gsize
    total = tl.load(tend_ptr + E - 1)
    if t < total:
        lo = 0
        hi = E - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if tl.load(tend_ptr + mid) > t:
                hi = mid
            else:
                lo = mid + 1
        g = lo
        before = tl.load(tend_ptr + g - 1, mask=g > 0, other=0)
        row0 = tl.load(offs_ptr + g) + (t - before) * BLOCK_M
        rend = tl.load(offs_ptr + g + 1)
        rm = row0 + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a = x_ptr + rm[:, None].to(tl.int64) * s_xm + rk[None, :]
        b = w_ptr + g.to(tl.int64) * s_we + rn[None, :].to(tl.int64) * s_wn + rk[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            ka = tl.load(a, mask=(rm[:, None] < rend) & (rk[None, :] < K - k * BLOCK_K), other=0.0)
            kb = tl.load(b, mask=(rn[None, :] < N) & (rk[:, None] < K - k * BLOCK_K), other=0.0)
            acc += tl.dot(ka, kb)
            a += BLOCK_K
            b += BLOCK_K
        c = o_ptr + rm[:, None].to(tl.int64) * s_om + rn[None, :]
        tl.store(c, acc.to(tl.bfloat16), mask=(rm[:, None] < rend) & (rn[None, :] < N))


def tile_end(offsets, block_m: int):
    """Running count of BLOCK_M row tiles through each expert: (E,) int32."""
    sizes = offsets[1:] - offsets[:-1]
    return torch.cumsum((sizes + (block_m - 1)) // block_m, 0, dtype=torch.int32)


def _launch(cfg, x, w, out, offsets, rows, n, k, e):
    bm, bn = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"]
    grid = ((triton.cdiv(rows, bm) + e) * triton.cdiv(n, bn),)
    _body[grid](x, w, out, offsets, tile_end(offsets, bm), rows, n, k, e,
                x.stride(0), w.stride(0), w.stride(1), out.stride(0),
                num_warps=cfg.num_warps, num_stages=cfg.num_stages, **cfg.kwargs)


def tune(x, w, offsets, out, rows, n, k, e):
    """Time every pruned config on these inputs; keep the fastest."""
    best, best_t = None, float("inf")
    for cfg in prune(configs(_arch()), rows, n, k, e):
        try:
            t = _bench(lambda cfg=cfg: _launch(cfg, x, w, out, offsets, rows, n, k, e))
        except Exception:  # noqa: BLE001 -- a config that does not compile is skipped
            continue
        if t < best_t:
            best, best_t = cfg, t
    if best is None:
        raise RuntimeError("no Triton config ran")
    _PICK[(rows, n, k, e)] = best
    return best


def grouped_mm(x, w, offsets, out=None):
    """out (rows, N) = the grouped GEMM; tuned on the first call per (rows, N, K, E)."""
    rows, k = x.shape
    e, n, _ = w.shape
    out = torch.empty(rows, n, device=x.device, dtype=x.dtype) if out is None else out
    cfg = _PICK.get((rows, n, k, e)) or tune(x, w, offsets, out, rows, n, k, e)
    _launch(cfg, x, w, out, offsets, rows, n, k, e)
    return out


def picked(rows: int, n: int, k: int, e: int):
    """The config tuned for a shape, as a dict (for a picks file), or None."""
    cfg = _PICK.get((rows, n, k, e))
    if cfg is None:
        return None
    return dict(cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)


def preset(rows: int, n: int, k: int, e: int, pick: dict) -> None:
    """Use `pick` (a picked() dict) for a shape instead of tuning it."""
    kw = dict(pick)
    w, s = kw.pop("num_warps"), kw.pop("num_stages")
    _PICK[(rows, n, k, e)] = triton.Config(kw, num_warps=w, num_stages=s)
