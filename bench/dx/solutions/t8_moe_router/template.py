"""hk.templates.rowwise: the router's top-k, then the k weights renormalised.
Softmax is monotonic, so the k largest probabilities are the k largest logits,
and renormalised over the k they are a softmax of those k logits: k rounds of
"the largest logit, the first column holding it, take it out", each a row
reduction, then exp(l_i - l_max) over their sum. The column index is an operand
(an arange made once, in setup: a property of the model's expert count)."""

import torch

import hk
from hk import templates as T

KIND = "template"
LABEL = "hk.templates.rowwise: k rounds of row_max/row_min, (T, k) outputs"


def router(k):
    @T.rowwise(outputs=(T.per_row(hk.i32, k), T.per_row(hk.fp32, k)), name=f"router_k{k}")
    def route(logits, col):
        x, ids, top = logits, [], []
        for _ in range(k):
            m = T.row_max(x)
            i = T.row_min(col + (m - x).step() * 1e9)      # the first column holding the max
            ids.append(i)
            top.append(m)
            x = x - (1.0 - (col - i).abs().step()) * 1e30  # out of the running
        ws = [(m - top[0]).exp() for m in top]
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
