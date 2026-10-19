"""Triton kernels for the scoreboard, each in two forms.

As a Triton user first writes them: the matmul is the official tutorial's
with its configs, the softmax the tutorial's, and the norms one program per
row with the row in a single block -- the shape every Triton RMSNorm in the
wild has. `matmul_abt`, `rms_norm`, `fused_add_rms_norm`, `softmax`.

Tuned: the same kernel bodies, autotuned over what a Triton user tuning for
gfx1100 would sweep (`TUNED_SPACE` says what), each candidate timed the way
the scoreboard times (`_warm_bench`). `matmul_abt_tuned`,
`rms_norm_tuned`, `fused_add_rms_norm_tuned`, `softmax_tuned`;
`tuning(fn)` reports what a tuned launcher searched and what it picked.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _mm_configs():
    out = []
    for bm, bn, bk, w, s in ((128, 128, 32, 4, 2), (128, 128, 64, 8, 2),
                             (256, 128, 32, 8, 2), (128, 256, 32, 8, 2),
                             (64, 128, 64, 4, 2), (128, 64, 64, 4, 2),
                             (128, 128, 64, 4, 1)):
        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                                  "GROUP_M": 8}, num_warps=w, num_stages=s))
    return out


#: What the tuned launchers sweep, and what they deliberately do not.
#: `matrix_instr_nonkdim` and `kpack` choose among MFMA instructions, which
#: CDNA has and gfx1100 does not: RDNA3's WMMA is 16x16x16 only, so they have
#: nothing to select here. `num_stages` past 2 does not fit the 64 KB of LDS a
#: workgroup gets at these tiles.
TUNED_SPACE = {
    "matmul": "BLOCK_M, BLOCK_N in {64,128,256}, BLOCK_K in {32,64,128}, "
              "num_warps in {4,8}, num_stages in {1,2}, waves_per_eu in {0,2}; "
              "pruned where the staged tiles pass 64 KB of LDS or the fp32 "
              "accumulator passes 128 VGPRs a lane",
    "row": "num_warps in {1,2,4,8,16} x waves_per_eu in {0,1,2,4}, keyed on rows "
           "and columns",
}
TUNED_BENCH = "warm, back to back behind a GPU sleep, best of 5 rounds of 10"
LDS_BYTES = 64 * 1024


def _warm_bench(fn, quantiles=None, rounds=5, calls=10):
    """Rank candidates the way scoreboard.py scores them: warm, back to back,
    best round.

    Triton's own do_bench flushes the cache before every call and keeps the
    median, which ranks a different regime. A 2048x4096 bf16 softmax is 33 MB
    and lives in the 96 MB Infinity Cache when it is run back to back; ranked
    cold, the pick ran 1.4x slower here than the untuned kernel. The GPU sleep
    in front of each round keeps the host ahead of the GPU, so a two-microsecond
    kernel is timed rather than its launch.
    """
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        torch.cuda._sleep(20_000_000)
        beg, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        beg.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        best = min(best, beg.elapsed_time(end) / calls)
    return [best] * len(quantiles) if quantiles else best


def _mm_configs_tuned():
    out = []
    for bm in (64, 128, 256):
        for bn in (64, 128, 256):
            for bk in (32, 64, 128):
                for w in (4, 8):
                    for s in (1, 2):
                        if (bm + bn) * bk * 2 * s > LDS_BYTES:
                            continue
                        if bm * bn // (w * 32) > 128:
                            continue
                        for wpe in (0, 2):
                            out.append(triton.Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                                 "GROUP_M": 8, "waves_per_eu": wpe},
                                num_warps=w, num_stages=s))
    return out


def _prune_mm(configs, named_args, **kwargs):
    """Tiles wider than the problem compute rows nobody keeps; drop them,
    unless that would leave nothing."""
    m, n = named_args["M"], named_args["N"]
    keep = [c for c in configs
            if c.kwargs["BLOCK_M"] <= max(m, 64) and c.kwargs["BLOCK_N"] <= max(n, 64)]
    return keep or configs


def _row_configs_tuned():
    return [triton.Config({"waves_per_eu": wpe}, num_warps=w)
            for w in (1, 2, 4, 8, 16) for wpe in (0, 1, 2, 4)]


def _autotune(configs, key, **kw):
    """triton.autotune, searching in every process under `_warm_bench`.

    Not `cache_results`: its disk key does not include the benchmark, so a pick
    made under another timing would be served without a search."""
    return triton.autotune(configs=configs, key=key, do_bench=_warm_bench, **kw)


def _matmul_abt_body(a_ptr, b_ptr, c_ptr, M, N, K, s_am, s_ak, s_bn, s_bk, s_cm,
                     s_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsize
    pid_n = (pid % group) // gsize
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm[:, None] * s_am + rk[None, :] * s_ak
    b = b_ptr + rn[None, :] * s_bn + rk[:, None] * s_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        ka = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_K),
                     other=0.0)
        kb = tl.load(b, mask=(rn[None, :] < N) & (rk[:, None] < K - k * BLOCK_K),
                     other=0.0)
        acc += tl.dot(ka, kb)
        a += BLOCK_K * s_ak
        b += BLOCK_K * s_bk
    c = c_ptr + rm[:, None] * s_cm + rn[None, :] * s_cn
    tl.store(c, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


_matmul_abt = triton.autotune(configs=_mm_configs(), key=["M", "N", "K"])(
    triton.jit(_matmul_abt_body))
_matmul_abt_tuned = _autotune(_mm_configs_tuned(), ["M", "N", "K"],
                              prune_configs_by={"early_config_prune": _prune_mm})(
    triton.jit(_matmul_abt_body))


def _matmul(kernel, a, b, out):
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty(m, n, device=a.device, dtype=a.dtype) if out is None else out
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"])         # noqa: E731
                         * triton.cdiv(n, meta["BLOCK_N"]),)
    kernel[grid](a, b, out, m, n, k, a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1), out.stride(0), out.stride(1))
    return out


def matmul_abt(a, b, out=None):
    """a @ b.T, b given as (N, K) -- F.linear's layout."""
    return _matmul(_matmul_abt, a, b, out)


def matmul_abt_tuned(a, b, out=None):
    """matmul_abt, autotuned over the gfx1100 space."""
    return _matmul(_matmul_abt_tuned, a, b, out)


def _rmsnorm_body(x_ptr, r_ptr, w_ptr, o_ptr, ro_ptr, M, N, eps,
                  RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
    if RESIDUAL:
        x += tl.load(r_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(ro_ptr + row * N + cols, x.to(ro_ptr.dtype.element_ty), mask=m)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * N + cols, (x * rstd * w).to(o_ptr.dtype.element_ty),
             mask=m)


_rmsnorm = triton.jit(_rmsnorm_body)
_rmsnorm_tuned = _autotune(_row_configs_tuned(), ["M", "N", "RESIDUAL"])(
    triton.jit(_rmsnorm_body))
_add_searched = set()


def _norm(kernel, x, r, w, o, ro, eps, residual, **launch):
    n = x.shape[-1]
    m = x.numel() // n
    kernel[(m,)](x, r, w, o, ro, m, n, eps, RESIDUAL=residual,
                 BLOCK=triton.next_power_of_2(n), **launch)


def rms_norm(x, w, eps=1e-6, out=None):
    out = torch.empty_like(x) if out is None else out
    n = x.shape[-1]
    _norm(_rmsnorm, x, x, w, out, out, eps, False, num_warps=8 if n >= 2048 else 4)
    return out


def rms_norm_tuned(x, w, eps=1e-6, out=None):
    out = torch.empty_like(x) if out is None else out
    _norm(_rmsnorm_tuned, x, x, w, out, out, eps, False)
    return out


def fused_add_rms_norm(x, r, w, eps=1e-6):
    n = x.shape[-1]
    _norm(_rmsnorm, x, r, w, x, r, eps, True, num_warps=8 if n >= 2048 else 4)
    return x, r


def fused_add_rms_norm_tuned(x, r, w, eps=1e-6):
    # In place, so the search runs on copies: no candidate sees another's
    # sums, and a timed call is the kernel alone. `restore_value` would clone
    # and copy both operands inside every timed call -- four copy kernels to
    # one norm, which is what the ranking would then be measuring.
    key = (x.numel() // x.shape[-1], x.shape[-1])
    if key not in _add_searched:
        xs, rs = x.clone(), r.clone()
        _norm(_rmsnorm_tuned, xs, rs, w, xs, rs, eps, True)
        _add_searched.add(key)
    _norm(_rmsnorm_tuned, x, r, w, x, r, eps, True)
    return x, r


def _softmax_body(x_ptr, o_ptr, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(o_ptr + row * N + cols,
             (e / tl.sum(e, axis=0)).to(o_ptr.dtype.element_ty), mask=m)


_softmax = triton.jit(_softmax_body)
_softmax_tuned = _autotune(_row_configs_tuned(), ["M", "N"])(triton.jit(_softmax_body))


def softmax(x, out=None):
    out = torch.empty_like(x) if out is None else out
    n = x.shape[-1]
    _softmax[(x.numel() // n,)](x, out, x.numel() // n, n,
                                BLOCK=triton.next_power_of_2(n),
                                num_warps=8 if n >= 2048 else 4)
    return out


def softmax_tuned(x, out=None):
    out = torch.empty_like(x) if out is None else out
    n = x.shape[-1]
    _softmax_tuned[(x.numel() // n,)](x, out, x.numel() // n, n,
                                      BLOCK=triton.next_power_of_2(n))
    return out


# ---------------------------------------------------------------- the LLM rows
#
# The rest of an LLM layer's row and pointwise ops, for scoreboard.py's llm-*
# groups: RMSNorm without a weight, LayerNorm (unweighted, as hk's), int8
# quantize (per row, absmax / 127), the three elementwise ops, silu_and_mul on
# a gate/up output (T, 2I) -> (T, I), and NeoX RoPE on (T, H, D) with each
# token's angles. Each as a Triton user writes it, and autotuned over the
# space TUNED_SPACE names, ranked by _warm_bench.

def _rows_unweighted_body(x_ptr, o_ptr, M, N, eps, KIND: tl.constexpr,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
    if KIND == 0:
        y = x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps))
    else:
        mean = tl.sum(x, axis=0) / N
        d = tl.where(m, x - mean, 0.0)
        y = d * (1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + eps))
    tl.store(o_ptr + row * N + cols, y.to(o_ptr.dtype.element_ty), mask=m)


_rows_unweighted = triton.jit(_rows_unweighted_body)
_rows_unweighted_tuned = _autotune(_row_configs_tuned(), ["M", "N", "KIND"])(
    triton.jit(_rows_unweighted_body))


def _unweighted(kernel, x, out, eps, kind, **launch):
    n = x.shape[-1]
    m = x.numel() // n
    kernel[(m,)](x, out, m, n, eps, KIND=kind, BLOCK=triton.next_power_of_2(n), **launch)
    return out


def rmsnorm(x, eps=1e-6, out=None):
    out = torch.empty_like(x) if out is None else out
    return _unweighted(_rows_unweighted, x, out, eps, 0,
                       num_warps=8 if x.shape[-1] >= 2048 else 4)


def rmsnorm_tuned(x, eps=1e-6, out=None):
    out = torch.empty_like(x) if out is None else out
    return _unweighted(_rows_unweighted_tuned, x, out, eps, 0)


def layernorm(x, eps=1e-5, out=None):
    out = torch.empty_like(x) if out is None else out
    return _unweighted(_rows_unweighted, x, out, eps, 1,
                       num_warps=8 if x.shape[-1] >= 2048 else 4)


def layernorm_tuned(x, eps=1e-5, out=None):
    out = torch.empty_like(x) if out is None else out
    return _unweighted(_rows_unweighted_tuned, x, out, eps, 1)


def _quant_body(x_ptr, q_ptr, s_ptr, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    tl.store(s_ptr + row, amax / 127.0)
    q = x * (127.0 / amax)
    # Round half to even, as hk and torch.round do: HIP's libdevice has no
    # rint, and adding then subtracting 1.5 * 2^23 rounds an fp32 |q| < 2^22
    # to an integer under the default rounding mode. |q| <= 127 here.
    q = (q + 12582912.0) - 12582912.0
    tl.store(q_ptr + row * N + cols, q.to(tl.int8), mask=m)


_quant = triton.jit(_quant_body)
_quant_tuned = _autotune(_row_configs_tuned(), ["M", "N"])(triton.jit(_quant_body))


def _quantize(kernel, x, q, s, **launch):
    n = x.shape[-1]
    m = x.numel() // n
    kernel[(m,)](x, q, s, m, n, BLOCK=triton.next_power_of_2(n), **launch)
    return q, s


def quantize(x, q, s):
    return _quantize(_quant, x, q, s, num_warps=8 if x.shape[-1] >= 2048 else 4)


def quantize_tuned(x, q, s):
    return _quantize(_quant_tuned, x, q, s)


def _ew_configs_tuned():
    return [triton.Config({"BLOCK": b, "waves_per_eu": wpe}, num_warps=w)
            for b in (1024, 2048, 4096, 8192, 16384) for w in (4, 8, 16)
            for wpe in (0, 2) if b >= w * 64 * 4]


def _ew_body(a_ptr, b_ptr, o_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m)
    if OP == 0:
        y = a + tl.load(b_ptr + offs, mask=m)
    elif OP == 1:
        y = a * tl.load(b_ptr + offs, mask=m)
    else:
        y = tl.exp(a.to(tl.float32)).to(a.dtype)
    tl.store(o_ptr + offs, y, mask=m)


_ew = triton.jit(_ew_body)
_ew_tuned = _autotune(_ew_configs_tuned(), ["n", "OP"])(triton.jit(_ew_body))
EW_OPS = {"add": 0, "mul": 1, "exp": 2}


def elementwise(op, a, b, out, BLOCK=1024):
    """The tutorial's vector add, for add, mul and exp (b is ignored)."""
    n = a.numel()
    _ew[(triton.cdiv(n, BLOCK),)](a, b, out, n, OP=EW_OPS[op], BLOCK=BLOCK)
    return out


def elementwise_tuned(op, a, b, out):
    n = a.numel()
    _ew_tuned[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](a, b, out, n, OP=EW_OPS[op])
    return out


def _silu_mul_configs_tuned():
    return [triton.Config({"BLOCK": b, "waves_per_eu": wpe}, num_warps=w)
            for b in (512, 1024, 2048, 4096, 8192) for w in (2, 4, 8, 16)
            for wpe in (0, 2) if b >= w * 64]


def _silu_mul_body(x_ptr, o_ptr, M, I, BLOCK: tl.constexpr):
    """vLLM's silu_and_mul: x is (M, 2I), gate then up; out (M, I)."""
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = cols < I
    g = tl.load(x_ptr + row * 2 * I + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * 2 * I + I + cols, mask=m, other=0.0).to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * u
    tl.store(o_ptr + row * I + cols, y.to(o_ptr.dtype.element_ty), mask=m)


_silu_mul = triton.jit(_silu_mul_body)
_silu_mul_tuned = _autotune(_silu_mul_configs_tuned(), ["M", "I"])(
    triton.jit(_silu_mul_body))


def silu_and_mul(x, out, BLOCK=1024):
    m, i2 = x.shape
    _silu_mul[(triton.cdiv(i2 // 2, BLOCK), m)](x, out, m, i2 // 2, BLOCK=BLOCK, num_warps=4)
    return out


def silu_and_mul_tuned(x, out):
    m, i2 = x.shape
    i = i2 // 2
    _silu_mul_tuned[lambda meta: (triton.cdiv(i, meta["BLOCK"]), m)](x, out, m, i)
    return out


def _rope_configs_tuned():
    return [triton.Config({"BLOCK_H": bh, "waves_per_eu": wpe}, num_warps=w)
            for bh in (1, 2, 4, 8, 16, 32, 64) for w in (1, 2, 4, 8)
            for wpe in (0, 2)]


def _rope_body(x_ptr, c_ptr, s_ptr, o_ptr, T, H, HALF: tl.constexpr,
               BLOCK_H: tl.constexpr):
    """NeoX RoPE on (T, H, 2*HALF): token t's angles c[t], s[t] (HALF each)."""
    t = tl.program_id(0)
    hs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    j = tl.arange(0, HALF)
    mh = (hs < H)[:, None]
    c = tl.load(c_ptr + t * HALF + j)[None, :]
    s = tl.load(s_ptr + t * HALF + j)[None, :]
    base = t * H * 2 * HALF + hs[:, None] * 2 * HALF + j[None, :]
    x1 = tl.load(x_ptr + base, mask=mh, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF, mask=mh, other=0.0).to(tl.float32)
    tl.store(o_ptr + base, (x1 * c - x2 * s).to(o_ptr.dtype.element_ty), mask=mh)
    tl.store(o_ptr + base + HALF, (x2 * c + x1 * s).to(o_ptr.dtype.element_ty), mask=mh)


_rope = triton.jit(_rope_body)
_rope_tuned = _autotune(_rope_configs_tuned(), ["T", "H", "HALF"])(triton.jit(_rope_body))


def rope(x, cos, sin, out):
    t, h, d = x.shape
    bh = min(triton.next_power_of_2(h), 8)
    _rope[(t, triton.cdiv(h, bh))](x, cos, sin, out, t, h, HALF=d // 2, BLOCK_H=bh,
                                   num_warps=4)
    return out


def rope_tuned(x, cos, sin, out):
    t, h, d = x.shape
    _rope_tuned[lambda meta: (t, triton.cdiv(h, meta["BLOCK_H"]))](
        x, cos, sin, out, t, h, HALF=d // 2)
    return out


TUNED_SPACE.update({
    "elementwise": "BLOCK in {1024..16384} x num_warps in {4,8,16} x waves_per_eu "
                   "in {0,2}, keyed on the element count",
    "silu_mul": "BLOCK in {512..8192} x num_warps in {2,4,8,16} x waves_per_eu in "
                "{0,2}, keyed on rows and I",
    "rope": "BLOCK_H (heads a program) in {1..64} x num_warps in {1,2,4,8} x "
            "waves_per_eu in {0,2}, keyed on tokens, heads, head dim",
})

_TUNED = {matmul_abt_tuned: (_matmul_abt_tuned, "matmul"),
          rms_norm_tuned: (_rmsnorm_tuned, "row"),
          fused_add_rms_norm_tuned: (_rmsnorm_tuned, "row"),
          softmax_tuned: (_softmax_tuned, "row"),
          rmsnorm_tuned: (_rows_unweighted_tuned, "row"),
          layernorm_tuned: (_rows_unweighted_tuned, "row"),
          quantize_tuned: (_quant_tuned, "row"),
          elementwise_tuned: (_ew_tuned, "elementwise"),
          silu_and_mul_tuned: (_silu_mul_tuned, "silu_mul"),
          rope_tuned: (_rope_tuned, "rope")}


def tuning(fn):
    """What a tuned launcher searched and what it picked, after it has run:
    {"space": ..., "candidates": n, "chosen": "..."}. The tutorial launchers
    report their fixed configs."""
    if fn in _TUNED:
        tuner, kind = _TUNED[fn]
        chosen = getattr(tuner, "best_config", None)
        return {"level": "tuned", "space": TUNED_SPACE[kind], "bench": TUNED_BENCH,
                "candidates": len(tuner.configs), "chosen": str(chosen) if chosen else None}
    if fn is matmul_abt:
        return {"level": "tutorial", "space": "the matmul tutorial's configs",
                "candidates": len(_matmul_abt.configs),
                "chosen": str(getattr(_matmul_abt, "best_config", None))}
    return {"level": "as written", "space": "num_warps 8 at N >= 2048, else 4",
            "candidates": 1, "chosen": None}
