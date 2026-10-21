"""Triton baseline: one program per row, an online logsumexp over column
blocks, then the label's logit (the structure of Liger's and Unsloth's
cross-entropy forward)."""

import torch
import triton
import triton.language as tl

KIND = "baseline"
LABEL = "Triton: one program per row, online logsumexp (Liger-style)"


@triton.jit
def _ce_fwd(x_ptr, lab_ptr, loss_ptr, V, stride, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lab = tl.load(lab_ptr + row)
    base = x_ptr + row * stride
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for c0 in range(0, V, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        s = tl.where(m_new == float("-inf"), 0.0, s)
        m = m_new
    mx = tl.max(m, axis=0)
    total = tl.sum(s * tl.exp(m - mx), axis=0)
    lse = mx + tl.log(total)
    tgt = tl.load(base + tl.maximum(lab, 0)).to(tl.float32)
    loss = tl.where(lab == IGNORE, 0.0, lse - tgt)
    tl.store(loss_ptr + row, loss)


def solve(logits, labels):
    t, v = logits.shape
    loss = torch.empty(t, device=logits.device, dtype=torch.float32)
    _ce_fwd[(t,)](logits, labels, loss, v, logits.stride(0), IGNORE=-100, BLOCK=4096,
                  num_warps=8)
    return loss
