"""hk.templates.rowwise: logsumexp minus the target logit, per row. A row of
151936 does not fit in registers, so the template streams it, once: the
logsumexp is online (a running max and a rescaled sum). `pick` gathers the
label's logit (int64 labels read in place; -100 clamps to column 0 and the
loss is zeroed)."""

import hk
from hk import templates as T

KIND = "template"
LABEL = "hk.templates.rowwise: row_logsumexp - pick(label), streamed for long rows"


@T.rowwise(outputs=T.per_row(hk.fp32))
def cross_entropy(logits, label):
    loss = T.row_logsumexp(logits) - T.pick(logits, label)
    return loss * (label != -100).to_int()


def solve(logits, labels):
    return cross_entropy(logits, labels)
