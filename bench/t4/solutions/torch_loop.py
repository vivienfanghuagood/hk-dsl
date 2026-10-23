"""The trivial T4 solution: torch.mm per expert (hipBLASLt on ROCm).

It shows the interface and how score.py treats a solution; it is correct and
scores a ratio, and it FAILS the audit by design -- its GEMMs are not hk's.
"""

import torch


def solve(x, w, offsets, out):
    o = offsets.tolist()          # the group bounds to the host: one sync a call
    for g in range(w.shape[0]):
        if o[g + 1] > o[g]:
            torch.mm(x[o[g]:o[g + 1]], w[g].t(), out=out[o[g]:o[g + 1]])
    return out
