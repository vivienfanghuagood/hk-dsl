"""Cross-entropy over a large vocabulary, forward and backward, for training (RDNA, strips).

Megatron-LM's vocab-parallel cross-entropy at one tensor-parallel rank, in two passes over the
logits instead of its five fp32 elementwise passes, and with no fp32 copy of them:

* forward: per row (token), lse = max + log(sum exp(x - max)) in fp32, from the bf16 logits held
  in registers; loss = lse - x[target] (the target's logit gathered on the host side of the call).
* backward: grad = (exp(x - lse) - onehot(target)) * grad_loss, computed in fp32 and rounded to
  bf16 once -- the value Megatron's path produces (its fp32 softmax and subtraction, then the cast
  to the logits' dtype). The target entry's subtraction is done in fp32, never on a rounded p.
  Transformer Engine's fused kernel before 2.19 stored bf16(p) and subtracted 1 in bf16, so a
  target with p >= 0.998 got a zero gradient; Megatron-LM disables it for that
  ("stability issues"). Here the target's gradient is (p - 1) * g in fp32, written last.
"""

from __future__ import annotations

from typing import Optional

from .. import dsl as hk
from ..dsl.collective import cross_warp
from ..dsl.values import as_vec
from ..types import bf16, fp32

_NEG_INF = float("-inf")
WARPS = 16
RUN = 8


def _chunks(v: int, warps: int = WARPS, run: int = RUN) -> int:
    per = 32 * run
    return -(-v // (warps * per))


def _fwd_kernel(warps: int = WARPS):
    def body(x: hk.Tensor[bf16], lse: hk.Tensor[fp32], *, CHUNKS: hk.Constexpr = 1,
             RUN: hk.Constexpr = 8):
        cw = CHUNKS * hk.arch.target().wave * RUN
        S = hk.RS(fp32, 1, cw, RUN)
        row = hk.arch.block_idx.x
        c0 = hk.arch.warp_id() * cw
        v = x.at[row, c0].load(S, bounded=True)
        v.right_fill_(x.cols - c0, _NEG_INF)
        m = cross_warp(v.max(axis=1), "max", warps)
        v -= m
        v.exp_()
        s = cross_warp(v.sum(axis=1), "add", warps)
        out = s.log() + m
        # lse broadcast across one chunk of its row (N, wave * RUN): the backward reads it so
        B = hk.RS(fp32, 1, hk.arch.target().wave * RUN, RUN)
        bl = hk.zeros(B)
        bl += as_vec(out, B)
        with hk.if_(hk.arch.warp_id() == 0):
            lse.at[row, 0].store(bl)

    return hk.kernel(body, warps=warps, name=f"ce_fwd_w{warps}",
                     grid=lambda p: (p.x.rows, 1, 1), noalias=True)


def _bwd_kernel(warps: int = WARPS):
    def body(x: hk.Tensor[bf16], lse: hk.Tensor[fp32], gl: hk.Tensor[fp32],
             grad: hk.Tensor[bf16], *, CHUNKS: hk.Constexpr = 1, RUN: hk.Constexpr = 8):
        cw = CHUNKS * hk.arch.target().wave * RUN
        S = hk.RS(fp32, 1, cw, RUN)
        B = hk.RS(fp32, 1, hk.arch.target().wave * RUN, RUN)
        row = hk.arch.block_idx.x
        c0 = hk.arch.warp_id() * cw
        # lse and the loss gradient of this row, each broadcast across one chunk of a row
        # (the host lays them out so): a max over it is the value, in every lane
        l = lse.at[row, 0].load(B).max(axis=1)
        g = gl.at[row, 0].load(B).max(axis=1)
        v = x.at[row, c0].load(S, bounded=True)
        v -= as_vec(l, S)
        v.exp_()
        v *= as_vec(g, S)
        grad.at[row, c0].store(v, bounded=True)

    return hk.kernel(body, warps=warps, name=f"ce_bwd_w{warps}",
                     grid=lambda p: (p.x.rows, 1, 1), noalias=True,
                     defines={"HK_BF16_ROUND": 2})


_KERNELS: dict = {}


def _kernels():
    if not _KERNELS:
        _KERNELS["fwd"], _KERNELS["bwd"] = _fwd_kernel(), _bwd_kernel()
    return _KERNELS["fwd"], _KERNELS["bwd"]


def _bcast(v, width: int):
    return v.reshape(-1, 1).expand(-1, width).contiguous()


class _CE:
    fn = None


def _function():
    if _CE.fn is not None:
        return _CE.fn
    import torch  # noqa: PLC0415

    class CrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, target, ignore_index):
            n, v = logits.shape
            fwd, _ = _kernels()
            w = 32 * RUN
            lse_b = torch.empty(n, w, device=logits.device, dtype=torch.float32)
            fwd(logits, lse_b, CHUNKS=_chunks(v), RUN=RUN)
            keep = target != ignore_index
            tgt = torch.where(keep, target, 0)
            xt = logits.gather(1, tgt.view(-1, 1)).view(-1).float()
            ctx.save_for_backward(logits, tgt, keep, lse_b)
            return torch.where(keep, lse_b[:, 0] - xt, 0.0)

        @staticmethod
        def backward(ctx, gl):
            logits, target, keep, lse_b = ctx.saved_tensors
            n, v = logits.shape
            _, bwd = _kernels()
            grad = torch.empty_like(logits)
            gl = torch.where(keep, gl.float(), 0.0)    # an ignored row's gradient is zero
            bwd(logits, lse_b, _bcast(gl, lse_b.shape[1]), grad, CHUNKS=_chunks(v), RUN=RUN)
            # the target's entry: (p - 1) * g in fp32, then one rounding (Megatron's order)
            xt = logits.gather(1, target.view(-1, 1)).view(-1).float()
            pt = torch.exp(xt - lse_b[:, 0])
            gt = torch.where(keep, (pt - 1.0) * gl, 0.0)
            grad.scatter_(1, target.view(-1, 1), gt.to(grad.dtype).view(-1, 1))
            return grad, None, None

    _CE.fn = CrossEntropy
    return CrossEntropy


def cross_entropy(logits, target, *, ignore_index: int = -100):
    """Per-token cross-entropy of bf16 `logits` (N, V) against int64 `target` (N,): fp32 loss
    (N,), differentiable in logits (bf16 gradient). No reduction and no label smoothing (Megatron
    applies its loss mask to the per-token loss afterwards); a row whose target is `ignore_index`
    has loss 0 and gradient 0. RDNA (strips); any V."""
    import torch  # noqa: PLC0415

    if logits.dim() != 2 or logits.dtype != torch.bfloat16 or logits.stride(1) != 1:
        raise ValueError(f"cross_entropy: bf16 (N, V) logits contiguous along V, got "
                         f"{tuple(logits.shape)}/{logits.dtype}")
    if logits.shape[1] % 8 or logits.stride(0) % 8:
        raise ValueError(f"cross_entropy: V={logits.shape[1]} and the row stride must be multiples of "
                         f"8 (16-byte rows; Megatron pads its vocabulary to a multiple of 128)")
    if target.shape != logits.shape[:1]:
        raise ValueError(f"cross_entropy: target {tuple(target.shape)} for logits "
                         f"{tuple(logits.shape)}")
    return _function().apply(logits, target.long(), int(ignore_index))


def vocab_parallel_cross_entropy(logits, target, tp_group=None):
    """Megatron-LM's fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group)
    at one tensor-parallel rank: logits (S, B, V) bf16 and target (S, B) -> loss (S, B) fp32. A
    larger tensor-parallel group is refused (the vocabulary would be split across ranks)."""
    import torch  # noqa: PLC0415

    if tp_group is not None and torch.distributed.is_initialized() \
            and torch.distributed.get_world_size(tp_group) > 1:
        raise ValueError("hk vocab_parallel_cross_entropy: one tensor-parallel rank only")
    s, b, v = logits.shape
    loss = cross_entropy(logits.reshape(s * b, v), target.reshape(s * b))
    return loss.view(s, b)


__all__ = ["cross_entropy", "vocab_parallel_cross_entropy"]
