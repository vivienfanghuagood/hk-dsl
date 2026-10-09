"""`F.scaled_dot_product_attention`, on the generated kernel where it fits.

This is Phase 6's point of contact with the ecosystem. Every Radeon attention
path worth replacing -- diffusers, comfyUI, vLLM, SGLang -- goes through
`F.scaled_dot_product_attention`, which on ROCm lands in aotriton's `attn_fwd`.
Redirecting it is one line::

    import torch.nn.functional as F
    from hk.ops import sdpa
    F.scaled_dot_product_attention = sdpa.scaled_dot_product_attention

or, reversibly, `sdpa.patch()` and `sdpa.unpatch()`.

The shape rules are `hk.ops.attn`'s, not repeated here: one list of rules, two
policies. `hk.attention` raises when a call does not fit, this falls back to
torch. Falling back is silent on purpose -- a framework that printed a warning
per step would print one per diffusion step -- and `HK_SDPA_VERBOSE=1` prints
each distinct reason once.

**Why this goes through `torch.ops` and not through the pybind module.** The
pybind path launches on the default stream, which is fine in a benchmark and
wrong inside a framework: a graph capture would refuse it, and an overlapped
copy would race it. `hk.torch_op` builds the same kernel body behind a
`TORCH_LIBRARY` registration that launches on torch's current stream. The first
call for a given (head_dim, causal, scale) pays a compile; `python3 -m
hk.autotune --aot` and `sdpa.warm()` are how a server pays it at startup
instead.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from . import attn as _attn

__all__ = [
    "attention", "scaled_dot_product_attention", "supported",
    "why_unsupported", "op_for", "warm", "patch", "unpatch", "patched",
]

NAMESPACE = "hk"

_VERBOSE = os.environ.get("HK_SDPA_VERBOSE", "") not in ("", "0")
_SEEN = set()
_OPS = {}
_PATCHED = None


def op_for(head_dim: int, causal: bool, scale: Optional[float] = None,
           n: int = 4096):
    """The `torch.ops.hk.*` entry for one (head_dim, causal, scale, length).

    `n` only picks the tuned schedule, so it changes which kernel runs and
    never what it computes. Memoised here as well as in `hk.torch_op` because
    this is on the launch path and `_kernel_for` is a couple of dict lookups
    that need not happen per call.
    """
    key = (head_dim, causal, scale, _attn._tune_key(n))
    op = _OPS.get(key)
    if op is None:
        from ..runtime.torch_ext import torch_op  # noqa: PLC0415

        op = _OPS[key] = torch_op(
            _attn._kernel_for(head_dim, causal, scale, n), ns=NAMESPACE
        )
    return op


def why_unsupported(q, k, v, attn_mask=None, dropout_p: float = 0.0,
                    is_causal: bool = False, enable_gqa: bool = False) -> str:
    """Why this exact call cannot run on the kernel, or '' if it can."""
    import torch  # noqa: PLC0415

    if attn_mask is not None:
        return "attn_mask is not supported"
    if dropout_p != 0.0:
        return "dropout is not supported (this is an inference kernel)"
    if torch.is_grad_enabled() and any(
        getattr(t, "requires_grad", False) for t in (q, k, v)
    ):
        return "no backward: an input requires grad"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return "device tensors only"
    if q.dim() == 4 and k.dim() == 4 and k.shape[1] != q.shape[1] \
            and not enable_gqa:
        # torch itself rejects this, so falling back keeps its message rather
        # than inventing a second one that says the same thing differently.
        return "head counts differ but enable_gqa is False"
    # Contiguity is the caller's to fix (see `scaled_dot_product_attention`),
    # so ask about everything else on contiguous stand-ins.
    return _attn._why(q.contiguous(), k.contiguous(), v.contiguous(),
                      bool(is_causal))


def supported(q, k, v, attn_mask=None, dropout_p: float = 0.0,
              is_causal: bool = False, enable_gqa: bool = False) -> bool:
    """Would this exact call run on the kernel?"""
    return why_unsupported(q, k, v, attn_mask, dropout_p, is_causal,
                           enable_gqa) == ""


def attention(q, k, v, *, causal: bool = False, scale: Optional[float] = None,
              out=None):
    """The kernel with no fallback: raises if the call does not fit.

    Use this where a wrong dispatch should be loud -- benchmarks, tests, any
    integration where quietly running aotriton would be mistaken for a win.
    """
    import torch  # noqa: PLC0415

    why = _attn._why(q, k, v, causal)
    if why:
        raise ValueError(f"hk sdpa: {why}")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous():
        raise ValueError(f"hk sdpa: out is {tuple(out.shape)}/{out.dtype}, "
                         f"expected a contiguous {tuple(q.shape)}/{q.dtype}")
    op_for(q.shape[-1], causal, scale, q.shape[-2])(q, k, v, out)
    return out


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    """`F.scaled_dot_product_attention`, argument for argument.

    Signature-identical to torch's, defaults included, so it can replace it by
    assignment. Anything the kernel does not cover is forwarded to torch.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415,N812

    # The globals are plain row-major (B, H, N, D) with no stride support, and
    # the op refuses a non-contiguous tensor rather than reading it wrong, so
    # the copy happens here. A framework that hands over the usual
    # `.view(B, N, H, D).transpose(1, 2)` pays it; one that keeps (B, H, N, D)
    # contiguous does not, because .contiguous() is then a no-op.
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    why = why_unsupported(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa)
    if why:
        if _VERBOSE and why not in _SEEN:
            _SEEN.add(why)
            print(f"hk sdpa: falling back to torch -- {why}")
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    out = torch.empty_like(q)
    op_for(q.shape[-1], bool(is_causal), scale, q.shape[-2])(q, k, v, out)
    return out


def warm(shapes: List[Tuple[int, bool, int]] = (), *,
         scale: Optional[float] = None, verbose: bool = False) -> List[str]:
    """Build the ops a server will need, before it needs them.

    `shapes` is (head_dim, causal, seq_len) triples; the default covers both
    head dims, causal and not, at the tuned lengths. Returns the op names, so
    a startup script can log what it paid for. Compiles are serial here --
    `python3 -m hk.autotune --aot -j 32` is the parallel one, and it is the
    better place to pay this if the schedules are already recorded.
    """
    if not shapes:
        shapes = [(d, c, n) for d in _attn.HEAD_DIMS for c in (False, True)
                  for n in (4096, 16384, 65536)]
    names = []
    for head_dim, causal, n in shapes:
        op_for(head_dim, causal, scale, n)
        name = _attn._kernel_for(head_dim, causal, scale, n).name
        if name not in names:
            names.append(name)
        if verbose:
            print(f"warmed d{head_dim} causal={causal} n={n} -> {name}")
    return names


def patch() -> None:
    """Redirect `F.scaled_dot_product_attention` to this module.

    Idempotent, and it keeps the original so `unpatch` can put it back -- a
    second `patch()` that captured our own wrapper as "the original" would make
    unpatching a no-op and the fallback path infinitely recursive.
    """
    global _PATCHED
    import torch.nn.functional as F  # noqa: PLC0415,N812

    if _PATCHED is not None:
        return
    _PATCHED = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = scaled_dot_product_attention


def unpatch() -> None:
    global _PATCHED
    import torch.nn.functional as F  # noqa: PLC0415,N812

    if _PATCHED is None:
        return
    F.scaled_dot_product_attention = _PATCHED
    _PATCHED = None


def patched() -> bool:
    return _PATCHED is not None
