"""hk.templates.rowwise: the router's softmax, then k rounds of "the largest
probability, the first column holding it, take it out", then the k weights
renormalised -- each a row reduction. The column index is an operand (an
arange made once, in setup: a property of the model's expert count)."""

import torch

import hk
from hk import templates as T

KIND = "template"
LABEL = "hk.templates.rowwise: softmax + k rounds of row_max/row_min, (T, k) outputs"


def router(k):
    @T.rowwise(outputs=(T.per_row(hk.i32, k), T.per_row(hk.fp32, k)), name=f"router_k{k}")
    def route(logits, col):
        p = (logits - T.row_logsumexp(logits)).exp()
        ids, ws = [], []
        for _ in range(k):
            m = T.row_max(p)
            i = T.row_min(col + (m - p).step() * 1e9)     # the first column holding the max
            ids.append(i)
            ws.append(m)
            p = p - (1.0 - (col - i).abs().step()) * 2.0   # out of the running
        s = ws[0]
        for w in ws[1:]:
            s = s + w
        return ids, [w / s for w in ws]
    return route


_ROUTERS = {}


def setup(logits, k):
    r = _ROUTERS.get(k) or _ROUTERS.setdefault(k, router(k))
    col = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
    return lambda: r(logits, col)


def solve(logits, k):
    return setup(logits, k)()
