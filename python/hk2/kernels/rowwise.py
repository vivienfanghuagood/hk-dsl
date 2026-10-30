"""Row-wise kernels: RMSNorm, LayerNorm, softmax. One program normalizes BM rows of a 2-D tensor; the row
fits in one tile (BN = the next power of two of the row length, the tail masked)."""
import torch

import hk2 as hk


@hk.jit
def _rmsnorm(x, w, out, eps, BM: hk.constexpr = 1, BN: hk.constexpr = 4096):
    m0 = hk.program_id(0) * BM
    t = hk.load(x, (m0, 0), (BM, BN)).to(hk.f32)
    r = hk.rsqrt((t * t).sum(1) / x.shape[1] + eps)
    g = hk.load(w, 0, BN).to(hk.f32)
    hk.store(out, (m0, 0), t * r[:, None] * g[None, :])


@hk.jit
def _layernorm(x, w, b, out, eps, BM: hk.constexpr = 1, BN: hk.constexpr = 4096):
    m0 = hk.program_id(0) * BM
    n = x.shape[1]
    t = hk.load(x, (m0, 0), (BM, BN)).to(hk.f32)
    mean = t.sum(1) / n
    cols = hk.arange(BN)[None, :]
    d = hk.where(cols < n, t - mean[:, None], 0.0)
    r = hk.rsqrt((d * d).sum(1) / n + eps)
    g = hk.load(w, 0, BN).to(hk.f32)
    beta = hk.load(b, 0, BN).to(hk.f32)
    hk.store(out, (m0, 0), d * r[:, None] * g[None, :] + beta[None, :])


@hk.jit
def _softmax(x, out, BM: hk.constexpr = 1, BN: hk.constexpr = 4096):
    m0 = hk.program_id(0) * BM
    t = hk.load(x, (m0, 0), (BM, BN), other=float("-inf")).to(hk.f32)
    e = hk.exp(t - t.max(1)[:, None])
    hk.store(out, (m0, 0), e / e.sum(1)[:, None])


def _plan(x: torch.Tensor):
    n = x.shape[-1]
    bn = 1 << max(n - 1, 1).bit_length()
    bm = max(1, min(16, 2048 // bn))  # small rows: several per program
    warps = 8 if bn >= 4096 else 4
    return bm, bn, warps


def _rows(x):
    return x.reshape(-1, x.shape[-1])


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x2 = _rows(x)
    out = torch.empty_like(x2)
    bm, bn, w = _plan(x2)
    _rmsnorm[(hk.cdiv(x2.shape[0], bm),)](x2, weight, out, eps, BM=bm, BN=bn, num_warps=w)
    return out.view_as(x)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x2 = _rows(x)
    out = torch.empty_like(x2)
    bm, bn, w = _plan(x2)
    _layernorm[(hk.cdiv(x2.shape[0], bm),)](x2, weight, bias, out, eps, BM=bm, BN=bn, num_warps=w)
    return out.view_as(x)


def softmax(x: torch.Tensor) -> torch.Tensor:
    x2 = _rows(x)
    out = torch.empty_like(x2)
    bm, bn, w = _plan(x2)
    _softmax[(hk.cdiv(x2.shape[0], bm),)](x2, out, BM=bm, BN=bn, num_warps=w)
    return out.view_as(x)
