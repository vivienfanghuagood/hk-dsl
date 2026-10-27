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
from typing import Dict, List, Optional, Tuple

from . import attn as _attn

__all__ = [
    "attention", "scaled_dot_product_attention", "supported",
    "why_unsupported", "op_for", "warm", "patch", "unpatch", "patched",
]

NAMESPACE = "hk"

_VERBOSE = os.environ.get("HK_SDPA_VERBOSE", "") not in ("", "0")
#: gfx1100's non-causal head_dim-128 calls on hk.ops.attn_k3 (the DiT kernel, K3) rather than
#: hk.ops.attn's. On by default on gfx1100 (route `sdpa_k3` of hk.integration.switches);
#: HK_SDPA_K3=0 or HK_DISABLE=1 turns it off. Decided once, at the first call (None until then).
_K3: Optional[bool] = None


def _k3_on() -> bool:
    global _K3
    if _K3 is None:
        from ..integration import switches  # noqa: PLC0415

        _K3 = switches.enabled("sdpa_k3")
    return _K3
_K3_FITS: Dict[Tuple, bool] = {}
_SEEN = set()
_OPS = {}
_PATCHED = None


def op_for(head_dim: int, causal: bool, scale: Optional[float] = None,
           n: int = 4096, bnhd: bool = False, mapping: str = "block_first"):
    """The `torch.ops.hk.*` entry for one (head_dim, causal, scale, length).

    `n` only picks the tuned schedule (gfx1100's; gfx950's kernel has one),
    so it changes which kernel runs and never what it computes. Memoised here as well as in `hk.torch_op` because
    this is on the launch path and `_kernel_for` is a couple of dict lookups
    that need not happen per call.
    """
    key = (head_dim, causal, scale, _attn._tune_key(n), bnhd, mapping)
    op = _OPS.get(key)
    if op is None:
        from ..runtime.torch_ext import torch_op  # noqa: PLC0415

        op = _OPS[key] = torch_op(
            _attn._kernel_for(head_dim, causal, scale, n, bnhd=bnhd, mapping=mapping),
            ns=NAMESPACE
        )
    return op


_RUNS = {}


def _run_for(head_dim: int, causal: bool, scale: Optional[float] = None,
             n: int = 4096, bnhd: bool = False, mapping: str = "block_first"):
    """`op_for`'s op as a launch: without the dispatcher outside torch.compile
    (`torch_ext.launcher`). What this module's calls go through."""
    key = (head_dim, causal, scale, _attn._tune_key(n), bnhd, mapping)
    run = _RUNS.get(key)
    if run is None:
        from ..runtime.torch_ext import launcher  # noqa: PLC0415

        run = _RUNS[key] = launcher(
            _attn._kernel_for(head_dim, causal, scale, n, bnhd=bnhd, mapping=mapping),
            ns=NAMESPACE)
    return run


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
    # so ask about everything else without materialising anything: a call that
    # is going to fall back must not pay for three copies first.
    return _attn._why(q, k, v, bool(is_causal), require_contiguous=False)


#: Verdicts, keyed on everything `why_unsupported` reads.
#:
#: A vision tower calls SDPA once per layer with the same shape -- 24 times for
#: CLIP ViT-L -- and the check is ten-odd Python predicates. On a tower this
#: kernel has no head_dim for, those predicates are the *entire* cost the patch
#: adds, and they measured as 6% of a 0.19 ms SigLIP attention. Memoised, the
#: unsupported path costs one tuple hash.
#:
#: Bounded for the same reason `FALLBACKS` is. Grad-enabled is in the key
#: because it is process state, not argument state, and a cached "yes" taken
#: under `no_grad` would be wrong the moment someone trains.
_VERDICTS: Dict[Tuple, str] = {}
_VERDICT_CAP = 512


def _verdict(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa) -> str:
    import torch  # noqa: PLC0415

    if attn_mask is not None or dropout_p != 0.0:
        return why_unsupported(q, k, v, attn_mask, dropout_p, is_causal,
                               enable_gqa)
    key = (q.shape, k.shape, v.shape, q.dtype, k.dtype, v.dtype,
           q.is_cuda, k.is_cuda, v.is_cuda, bool(is_causal), bool(enable_gqa),
           torch.is_grad_enabled() and any(
               getattr(t, "requires_grad", False) for t in (q, k, v)))
    got = _VERDICTS.get(key)
    if got is None:
        got = why_unsupported(q, k, v, None, 0.0, is_causal, enable_gqa)
        if len(_VERDICTS) < _VERDICT_CAP:
            _VERDICTS[key] = got
    return got


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
    _run_for(q.shape[-1], causal, scale, q.shape[-2],
             mapping=_attn._mapping(q, k, causal))(q, k, v, out)
    return out


#: How many calls each branch took, since the process started or `reset_stats`.
#:
#: A patch that silently falls back looks exactly like a patch that works: the
#: answers are right either way and the only difference is the time, which on a
#: vision tower is a few hundred microseconds inside a multi-second request.
#: So the drop-in counts. `hk.integration.vllm.probe()` reads this, and the
#: vLLM verification reads it before and after a generate to prove which branch
#: a real server took rather than inferring it from a stopwatch.
STATS = {"kernel": 0, "fallback": 0, "bnhd": 0, "k3": 0}

#: (q shape, k shape) -> calls, of the calls hk.ops.attn_k3 took (the first 16 shapes).
K3_SHAPES: Dict[Tuple, int] = {}

#: Shapes that fell back, with the reason, so a run that was slower than hoped
#: can say which call it was. Bounded: a serving process must not grow a dict
#: because someone sent it a new sequence length.
FALLBACKS: Dict[Tuple, str] = {}
_FALLBACK_CAP = 64


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = 0
    K3_SHAPES.clear()
    FALLBACKS.clear()
    _VERDICTS.clear()


def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                 dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    """`F.scaled_dot_product_attention`, argument for argument.

    Signature-identical to torch's, defaults included, so it can replace it by
    assignment. Anything the kernel does not cover is forwarded to torch.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415,N812

    # Decide first, copy second. The globals are plain row-major (B, H, N, D)
    # with no stride support, so a call the kernel takes has to materialise a
    # `.transpose(1, 2)` view -- but a call it *rejects* must not, because
    # torch is about to be handed the original strided tensors anyway and the
    # copies would be pure loss. vLLM's ViT path is exactly this case: it
    # permutes (b, s, h, d) to (b, h, s, d) for every tower, and the towers
    # this kernel has no head_dim for were measured 16% slower than no patch
    # when the copy came first.
    why = _verdict(query, key, value, attn_mask, dropout_p, is_causal,
                   enable_gqa)
    if why:
        STATS["fallback"] += 1
        if len(FALLBACKS) < _FALLBACK_CAP:
            FALLBACKS[(tuple(query.shape), str(query.dtype))] = why
        if _VERBOSE and why not in _SEEN:
            _SEEN.add(why)
            print(f"hk sdpa: falling back to torch -- {why}")
        # torch's own function: once `patch()` has run, the module attribute
        # is this one, and calling it would recurse until Python gives up.
        torch_sdpa = _PATCHED if _PATCHED is not None else F.scaled_dot_product_attention
        return torch_sdpa(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    STATS["kernel"] += 1
    if not is_causal and _k3_on() and _k3_fits(query, key, value, scale):
        from . import attn_k3  # noqa: PLC0415

        STATS["k3"] += 1
        shp = (tuple(query.shape), tuple(key.shape))
        if shp in K3_SHAPES or len(K3_SHAPES) < 16:
            K3_SHAPES[shp] = K3_SHAPES.get(shp, 0) + 1
        if attn_k3.is_bnhd(query, key, value):
            STATS["bnhd"] += 1
        return attn_k3.attention(query, key, value)
    if _bnhd_views(query, key, value):
        # (B, H, N, D) views of (B, N, H, D) memory -- what an engine that
        # projects into (B, N, H, D) and transposes for SDPA passes. The
        # kernel reads that memory where it is (no copies in, none out) and
        # the output is handed back as the same kind of view.
        b, h, n, d = query.shape
        out = torch.empty((b, n, h, d), device=query.device, dtype=query.dtype)
        flat = [t.transpose(1, 2).reshape(t.shape[0], 1, t.shape[2], -1)
                for t in (query, key, value, out.transpose(1, 2))]
        _run_for(d, bool(is_causal), scale, n, bnhd=True)(*flat)
        STATS["bnhd"] += 1
        return out.transpose(1, 2)
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    out = torch.empty_like(q)
    _run_for(q.shape[-1], bool(is_causal), scale, q.shape[-2],
             mapping=_attn._mapping(q, k, bool(is_causal)))(q, k, v, out)
    return out


def _k3_fits(q, k, v, scale) -> bool:
    key = (q.shape, k.shape, v.shape, q.stride(), k.stride(), v.stride(), q.dtype, scale)
    got = _K3_FITS.get(key)
    if got is None:
        from . import attn_k3  # noqa: PLC0415

        got = not _attn._native_cdna() and attn_k3.why(q, k, v, False, scale) == ""
        if len(_K3_FITS) < _VERDICT_CAP:
            _K3_FITS[key] = got
    return got


def _bnhd_views(*ts) -> bool:
    """Whether every tensor is a (B, H, N, D) view of contiguous (B, N, H, D)
    memory, on gfx1100 (gfx950's kernel has no such variant)."""
    if _attn._native_cdna():
        return False
    return all(t.dim() == 4 and not t.is_contiguous() and t.transpose(1, 2).is_contiguous()
               for t in ts)


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
