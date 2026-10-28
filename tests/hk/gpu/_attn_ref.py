"""Attention references for the GPU tests, in float64, and the check every
attention test applies.

float64 rather than torch's fp32 SDPA: on ROCm an fp32
F.scaled_dot_product_attention is not always the math backend, and the same
case measured 2.4e-3 to 4.1e-3 against it from run to run (10-28, gfx950) --
a reference that moves by more than the kernel's own error cannot hold a
tolerance at the kernel's error. Explicit softmax in float64 is exact to well
below what a bf16 kernel can resolve.

`check` holds an output to two bounds. The one that matters is per row: an
output row (the last axis) may be off by at most ULPS bf16 ulps of that row's
largest value. A bound relative to the whole output's largest value cannot
see a wrong row of small values. A gfx950 decode that computed some scores
wrong (an MFMA hazard LLVM padded too little, ROCm/llvm-project#5153) was
16-30 ulps off in its worst row, and only 0.46-4.1e-2 relative to the
output's largest value -- mostly under the 2e-2 these tests used to ask,
and under TOL in two cases of three.

HK_TOL_RECORD=<file> appends "<test id> <arch> <rel> <ulps>" for every check,
so the bounds can be re-derived from measurements.
"""

from __future__ import annotations

import os

import torch

#: Bounds, measured 2026-10-28 over every check of the tests that use them
#: (test_attn, test_sdpa, test_paged, test_paged_fused, test_paged_prefill,
#: test_attn_k3, test_hipkittens, test_schedules, test_vllm_backend_decode,
#: test_fenced): worst on gfx950, CI and vLLM images / gfx1100:
#: - ULPS, the worst row's error in bf16 ulps of its largest value: 2.31 / 1.32.
#:   The miscomputing decode: 16-30.
#: - TOL, max |got - ref| / max |ref|: 7.1e-3 / 5.3e-3.
ULPS = 4.0
TOL = 1e-2


def attention64(q, k, v, causal=False, scale=None, q_chunk=1024):
    """softmax(q k^T * scale) v for (B, H, N, D) q and (B, H_kv, N_kv, D) k, v
    (GQA: a q head reads kv head h // (H / H_kv)), in float64, returned as
    float32. `causal` is top-left aligned: query i sees keys 0..i."""
    b, h, n, d = q.shape
    nk = k.shape[2]
    per = h // k.shape[1]
    scale = d ** -0.5 if scale is None else scale
    out = torch.empty(b, h, n, d, device=q.device, dtype=torch.float32)
    pos_k = torch.arange(nk, device=q.device)[None, :]
    for bi in range(b):
        for hi in range(h):
            kk = k[bi, hi // per].double()
            vv = v[bi, hi // per].double()
            for i in range(0, n, q_chunk):
                j = min(i + q_chunk, n)
                s = (q[bi, hi, i:j].double() @ kk.T) * scale
                if causal:
                    pos_q = torch.arange(i, j, device=q.device)[:, None]
                    s = s.masked_fill(pos_k > pos_q, float("-inf"))
                out[bi, hi, i:j] = (torch.softmax(s, dim=-1) @ vv).float()
    return out


def softmax_v64(qq, kk, vv, scale, keep=None):
    """One head: softmax(qq kk^T * scale) vv in float64 (2D operands),
    `keep` a boolean mask of allowed (query, key) pairs. float32 out."""
    s = (qq.double() @ kk.double().T) * scale
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    return (torch.softmax(s, dim=-1) @ vv.double()).float()


def check(got, want) -> float:
    """Assert `got` is within ULPS and TOL of `want` (recorded under
    HK_TOL_RECORD); returns the relative error."""
    got, want = got.float(), want.float()
    assert torch.isfinite(got).all().item(), "output has non-finite elements"
    r, u = rel(got, want), ulps(got, want)
    record(r, u)
    assert u < ULPS and r < TOL, (
        f"worst row off by {u:.2f} bf16 ulps of its largest value (bound {ULPS}); "
        f"max |got - want| / max |want| = {r:.2e} (bound {TOL})")
    return r


def rel(got, want) -> float:
    """max |got - want| / max |want|."""
    got, want = got.float(), want.float()
    # an all-zero answer (every key masked out) is held to absolute zero
    return ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


def ulps(got, want) -> float:
    """The worst output row's error in bf16 ulps of that row's largest
    |want|: max over rows of max_j |got - want| / ulp(max_j |want|), a row
    being the last axis. A row of zeros is held to zero."""
    d = want.shape[-1]
    g, w = got.float().reshape(-1, d), want.float().reshape(-1, d)
    m = w.abs().amax(-1).clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(m)) - 7)
    return ((g - w).abs().amax(-1) / ulp).max().item()


def record(err: float, ulp: float = float("nan")) -> None:
    path = os.environ.get("HK_TOL_RECORD")
    if not path:
        return
    test = os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]
    arch = os.environ.get("HK_ARCH", "?")
    with open(path, "a") as f:
        f.write(f"{test} {arch} {err:.6e} {ulp:.4f}\n")
