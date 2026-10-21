"""What t6 (soft-capping) and t7 (sliding window) share: the inputs, the fp32
golden reference, and the check."""

import torch

from bench.dx.common import assert_close


def make_inputs(shape, device="cuda", seed=0, qk_std=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    b, hq, hkv, n, d = shape["B"], shape["Hq"], shape["Hkv"], shape["N"], shape["D"]

    def r(h, std):
        return (std * torch.randn(b, h, n, d, generator=g, device=device)).to(torch.bfloat16)

    return {"q": r(hq, qk_std), "k": r(hkv, qk_std), "v": r(hkv, 1.0)}


def attention_ref(q, k, v, scale, cap=None, window=None, chunk=1024):
    """fp32, causal, GQA; one KV head's query group and a chunk of queries at a
    time, over only the keys that chunk can see."""
    b_, hq, n, d = q.shape
    hkv = k.shape[1]
    grp = hq // hkv
    out = torch.empty(b_, hq, n, d, dtype=torch.float32, device=q.device)
    for b in range(b_):
        for h in range(hkv):
            kh, vh = k[b, h].float(), v[b, h].float()
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                lo = 0 if window is None else max(0, s - window + 1)
                qc = q[b, h * grp:(h + 1) * grp, s:e].float()            # (grp, c, d)
                sc = (qc @ kh[lo:e].T) * scale                           # (grp, c, e-lo)
                if cap is not None:
                    sc = cap * torch.tanh(sc / cap)
                i = torch.arange(s, e, device=q.device)[:, None]
                j = torch.arange(lo, e, device=q.device)[None, :]
                mask = j > i
                if window is not None:
                    mask |= (i - j) >= window
                sc.masked_fill_(mask, float("-inf"))
                out[b, h * grp:(h + 1) * grp, s:e] = torch.softmax(sc, -1) @ vh[lo:e]
    return out


def check(out, ref, inputs):
    return assert_close("out", out, ref, rtol=3e-2, atol=3e-2, dtype=torch.bfloat16,
                        shape=inputs["q"].shape)


def flops(shape, window=None):
    """Causal (and windowed) attention's useful flops: 4*d per (query, visible key)."""
    n = shape["N"]
    if window is None or window >= n:
        pairs = n * (n + 1) // 2
    else:
        pairs = window * (window + 1) // 2 + (n - window) * window
    return 4 * shape["B"] * shape["Hq"] * shape["D"] * pairs
