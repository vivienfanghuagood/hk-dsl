"""GEMM: ``linear(x, w, bias=None, act=None)`` = act(x @ w.T + bias), and ``matmul(a, b)`` = a @ b.

One kernel, autotuned per problem on the GPU it runs on (``hk2.autotune``). Programs are ordered in groups of
GROUP_M row blocks so that neighbouring workgroups share B panels in L2.
"""
import torch

import hk2 as hk


@hk.func
def _act(y, ACT):
    if ACT == "silu":
        y = y / (1.0 + hk.exp(-y))
    elif ACT == "relu":
        y = hk.maximum(y, 0.0)
    elif ACT == "gelu":  # tanh approximation
        t = 0.7978845608028654 * (y + 0.044715 * y * y * y)
        y = 0.5 * y * (2.0 - 2.0 / (hk.exp(2.0 * t) + 1.0))
    return y


@hk.jit
def _gemm(a, b, c, bias, BM: hk.constexpr = 128, BN: hk.constexpr = 128, BK: hk.constexpr = 64,
          GROUP_M: hk.constexpr = 8, TRANS_B: hk.constexpr = True, HAS_BIAS: hk.constexpr = False,
          ACT: hk.constexpr = ""):
    M, N, K = a.shape[0], c.shape[1], a.shape[1]
    n_m, n_n = hk.cdiv(M, BM), hk.cdiv(N, BN)
    pid = hk.xcd_swizzle(hk.program_id(0), n_m * n_n)
    group = GROUP_M * n_n
    first = pid // group * GROUP_M
    rows = hk.minimum(n_m - first, GROUP_M)
    pm = first + (pid % group) % rows
    pn = (pid % group) // rows
    m0, n0 = pm * BM, pn * BN
    acc = hk.zeros((BM, BN))
    for k in range(0, K, BK):
        x = hk.load(a, (m0, k), (BM, BK))
        if TRANS_B:
            y = hk.load(b, (n0, k), (BN, BK)).T
        else:
            y = hk.load(b, (k, n0), (BK, BN))
        acc = hk.dot(x, y, acc)
    if HAS_BIAS:
        acc = acc + hk.load(bias, n0, BN).to(hk.f32)[None, :]
    hk.store(c, (m0, n0), _act(acc, ACT))


_CONFIGS = [hk.Config(BM=256, BN=256, BK=64, num_warps=8), hk.Config(BM=256, BN=128, BK=64, num_warps=8),
            hk.Config(BM=128, BN=256, BK=64, num_warps=8), hk.Config(BM=128, BN=128, BK=64, num_warps=4),
            hk.Config(BM=128, BN=128, BK=32, num_warps=4), hk.Config(BM=64, BN=128, BK=64, num_warps=4),
            hk.Config(BM=64, BN=64, BK=64, num_warps=4)]
_tuned = hk.autotune(configs=_CONFIGS, key=["a", "b", "TRANS_B", "HAS_BIAS", "ACT"])(_gemm)


def _grid(meta):
    a, c = meta["a"], meta["c"]
    return (hk.cdiv(a.shape[0], meta["BM"]) * hk.cdiv(c.shape[1], meta["BN"]),)


def _run(a, b, n, trans_b, bias, act, out_dtype):
    lead = a.shape[:-1]
    a2 = a.reshape(-1, a.shape[-1])
    c = torch.empty(a2.shape[0], n, device=a.device, dtype=out_dtype or a.dtype)
    _tuned[_grid](a2, b, c, bias if bias is not None else c, TRANS_B=trans_b, HAS_BIAS=bias is not None,
                  ACT=act or "")
    return c.view(*lead, n)


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None, act: str = None,
           out_dtype: torch.dtype = None) -> torch.Tensor:
    """act(x @ w.T + bias), w of shape (N, K) as torch.nn.Linear stores it. act: None, 'silu', 'relu', 'gelu'."""
    return _run(x, w, w.shape[0], True, bias, act, out_dtype)


def matmul(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype = None) -> torch.Tensor:
    """a @ b for a (..., K) and b (K, N)."""
    return _run(a, b, b.shape[1], False, None, None, out_dtype)
