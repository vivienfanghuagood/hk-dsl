"""T3 cross_entropy: the reference, the inputs and the check. task.md is the problem."""

import torch
import torch.nn.functional as F

from bench.dx.common import Mismatch, assert_close

SHAPES = [
    dict(name="T64-V151936", T=64, V=151936),
    dict(name="T512-V151936", T=512, V=151936),
    dict(name="T2048-V151936", T=2048, V=151936),
    dict(name="T8192-V151936", T=8192, V=151936),
    dict(name="T4096-V128256", T=4096, V=128256),
    dict(name="T8192-V32000", T=8192, V=32000),
]
QUICK = ("T64-V151936", "T2048-V151936", "T8192-V32000")
SMALL = [dict(name="small", T=16, V=4096), dict(name="small-ragged", T=7, V=1000)]
IGNORE = -100


def make_inputs(shape, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    t, v = shape["T"], shape["V"]
    logits = (2 * torch.randn(t, v, generator=g, device=device)).to(torch.bfloat16)
    labels = torch.randint(0, v, (t,), generator=g, device=device, dtype=torch.int64)
    drop = torch.rand(t, generator=g, device=device) < 0.05
    labels[drop] = IGNORE
    if t > 1:
        labels[-1] = IGNORE  # always at least one ignored token
    return {"logits": logits, "labels": labels}


def reference(logits, labels):
    """Golden: fp64, a row at a time so the copy stays small."""
    out = torch.zeros(logits.shape[0], dtype=torch.float64, device=logits.device)
    for s in range(0, logits.shape[0], 1024):
        x = logits[s:s + 1024].double()
        lab = labels[s:s + 1024]
        lse = torch.logsumexp(x, -1)
        tgt = x.gather(1, lab.clamp_min(0)[:, None])[:, 0]
        out[s:s + 1024] = torch.where(lab == IGNORE, 0.0, lse - tgt)
    return out


def torch_impl(logits, labels):
    """What a PyTorch user writes (Hugging Face upcasts the logits first)."""
    return F.cross_entropy(logits.float(), labels, reduction="none")


def check(out, ref, inputs):
    msg = assert_close("loss", out, ref, rtol=1e-4, atol=2e-3, dtype=torch.float32)
    ign = inputs["labels"] == IGNORE
    if ign.any() and (out[ign] != 0).any():
        raise Mismatch(f"loss: {int((out[ign] != 0).sum())} ignored tokens have a nonzero loss")
    return msg


def work(shape):
    return {"bytes": shape["T"] * shape["V"] * 2 + shape["T"] * 12}
