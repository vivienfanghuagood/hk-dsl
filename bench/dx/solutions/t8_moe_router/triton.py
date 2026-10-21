"""Triton baseline: a block of rows per program, the softmax over E in
registers, then k rounds of argmax-and-mask, renormalised (the shape of vLLM's
topk_softmax kernels)."""

import torch
import triton
import triton.language as tl

KIND = "baseline"
LABEL = "Triton: rows x E in registers, k rounds of argmax"


@triton.jit
def _router(x_ptr, ids_ptr, w_ptr, T, E: tl.constexpr, K: tl.constexpr,
            BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_E)
    rmask = rows < T
    x = tl.load(x_ptr + rows[:, None] * E + cols[None, :],
                mask=rmask[:, None] & (cols[None, :] < E), other=float("-inf")).to(tl.float32)
    mx = tl.max(x, axis=1)
    p = tl.exp(x - mx[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    kk = tl.arange(0, BLOCK_K)
    out_w = tl.zeros([BLOCK_T, BLOCK_K], tl.float32)
    out_i = tl.zeros([BLOCK_T, BLOCK_K], tl.int32)
    for j in tl.static_range(K):
        best = tl.max(p, axis=1)
        idx = tl.argmax(p, axis=1)
        out_w = tl.where(kk[None, :] == j, best[:, None], out_w)
        out_i = tl.where(kk[None, :] == j, idx[:, None].to(tl.int32), out_i)
        p = tl.where(cols[None, :] == idx[:, None], -1.0, p)
    out_w = out_w / tl.sum(out_w, axis=1)[:, None]
    omask = rmask[:, None] & (kk[None, :] < K)
    tl.store(w_ptr + rows[:, None] * K + kk[None, :], out_w, mask=omask)
    tl.store(ids_ptr + rows[:, None] * K + kk[None, :], out_i, mask=omask)


def solve(logits, k):
    t, e = logits.shape
    ids = torch.empty((t, k), device=logits.device, dtype=torch.int32)
    w = torch.empty((t, k), device=logits.device, dtype=torch.float32)
    block_t = 16
    _router[(triton.cdiv(t, block_t),)](logits, ids, w, t, E=e, K=k, BLOCK_T=block_t,
                                        BLOCK_E=triton.next_power_of_2(e),
                                        BLOCK_K=triton.next_power_of_2(k), num_warps=4)
    return ids, w
