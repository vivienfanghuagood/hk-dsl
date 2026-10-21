"""T8 moe_router: the reference, the inputs and the check. task.md is the problem."""

import torch

from bench.dx.common import Mismatch

SHAPES = ([dict(name=f"T{t}-E8-k2", T=t, E=8, k=2) for t in (16, 4096)]
          + [dict(name=f"T{t}-E64-k8", T=t, E=64, k=8) for t in (16, 4096)]
          + [dict(name=f"T{t}-E128-k8", T=t, E=128, k=8) for t in (1, 16, 256, 4096, 16384)])
QUICK = ("T16-E8-k2", "T16-E128-k8", "T4096-E128-k8")
SMALL = [dict(name="small", T=33, E=64, k=8), dict(name="small-e8", T=5, E=8, k=2)]


def make_inputs(shape, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    logits = (2 * torch.randn(shape["T"], shape["E"], generator=g, device=device)).to(torch.bfloat16)
    return {"logits": logits, "k": int(shape["k"])}


def reference(logits, k):
    p = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(p, k, dim=-1)
    return ids.to(torch.int32), w / w.sum(-1, keepdim=True), p


def torch_impl(logits, k):
    p = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(p, k, dim=-1)
    return ids.to(torch.int32), w / w.sum(-1, keepdim=True)


def check(out, ref, inputs):
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise Mismatch(f"expected (ids, weights), got {type(out).__name__}")
    ids, w = out
    _, w_ref, p = ref
    t, e = inputs["logits"].shape
    k = inputs["k"]
    if ids.dtype != torch.int32 or tuple(ids.shape) != (t, k):
        raise Mismatch(f"ids: {ids.dtype} {tuple(ids.shape)}, expected int32 {(t, k)}")
    if w.dtype != torch.float32 or tuple(w.shape) != (t, k):
        raise Mismatch(f"weights: {w.dtype} {tuple(w.shape)}, expected float32 {(t, k)}")
    il = ids.long()
    if ((il < 0) | (il >= e)).any():
        raise Mismatch("ids: out of [0, E)")
    srt = il.sort(-1).values
    if (srt[:, 1:] == srt[:, :-1]).any():
        raise Mismatch("ids: a row repeats an expert")
    got_p = p.gather(1, il)
    top_p = torch.topk(p, k, -1).values
    bad = (got_p.sort(-1, descending=True).values - top_p).abs() > 1e-6
    if bad.any():
        r = int(bad.any(-1).nonzero()[0])
        raise Mismatch(f"ids: {int(bad.any(-1).sum())} rows are not a top-{k}, first row {r}: "
                       f"{ids[r].tolist()} vs {torch.topk(p[r], k).indices.tolist()}")
    want_w = got_p / got_p.sum(-1, keepdim=True)
    err = (w - want_w).abs().max().item()
    if err > 1e-4:
        raise Mismatch(f"weights: max abs err {err:.2e} against the renormalised probabilities")
    if (w[:, 1:] > w[:, :-1] + 1e-6).any():
        raise Mismatch("weights: a row is not in descending order")
    return f"weights: max abs err {err:.2e}"


def work(shape):
    return {"bytes": shape["T"] * (shape["E"] * 2 + shape["k"] * 8)}
