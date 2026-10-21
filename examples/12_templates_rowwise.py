"""Templates: write the math of an element or of a row, and let
hk.templates.elementwise / rowwise do the rest.

    gelu_and_mul = hk.templates.elementwise(lambda g, u: hk.math.gelu(g) * u)

    @hk.templates.rowwise
    def rms_norm(x, w, *, EPS=1e-6):
        return x * (T.row_mean(x * x) + EPS).rsqrt() * w

A hook gets fp32 register strips (the load widens, the store rounds to
nearest even); `T.row_sum / row_max / row_mean / row_logsumexp` reduce the
whole row. The template picks the plan per GPU -- the run a lane reads, the
warps, the stretch each covers, the grid order, an occupancy cap past the
cache, a multi-pass stream for a row too wide to hold -- from the plans
hk.ops measured. `k.plan(*args)` prints it. Five kernels below, each checked
against torch and timed against torch.compile and, where it has one, the
hk.ops kernel for the same job. docs/templates/elementwise-rowwise.md is the
guide.

    python3 examples/12_templates_rowwise.py
    python3 examples/12_templates_rowwise.py --dump cpp       # the gelu kernel's C++
"""

import _common

import hk
from hk import templates as T

# ---------------------------------------------------------------- the kernels

gelu_and_mul = T.elementwise(lambda g, u: hk.math.gelu(g) * u)


@T.rowwise
def rms_norm(x, w, *, EPS=1e-6):
    return x * (T.row_mean(x * x) + EPS).rsqrt() * w


@T.rowwise(outputs=2)                                  # y, and the new residual
def add_rms_norm(x, r, w, *, EPS=1e-6):
    h = x + r
    return h * (T.row_mean(h * h) + EPS).rsqrt() * w, h


@T.rowwise
def softmax(x):
    e = (x - T.row_max(x)).exp()
    return e / T.row_sum(e)


@T.rowwise(outputs=T.per_row(hk.fp32))                 # one loss a row
def cross_entropy(logits, label):
    loss = T.row_logsumexp(logits) - T.pick(logits, label)
    return loss * (label != -100).to_int()             # ignore_index


def kernels():
    """The kernels at one shape each, as `--dump` / `--build` and the test
    suite see them: the template's kernel with the plan's constexprs."""
    B, M = hk.bf16, T.meta
    out = []
    for name, t, args in (
            ("gelu_and_mul", gelu_and_mul, (M((64, 14336), B, (28672, 1)),) * 2),
            ("rms_norm", rms_norm, (M((64, 4096), B), M((4096,), B))),
            ("add_rms_norm", add_rms_norm, (M((64, 4096), B), M((64, 4096), B), M((4096,), B))),
            ("softmax", softmax, (M((64, 8192), B),)),
            ("cross_entropy", cross_entropy, (M((64, 151936), B), M((64,), hk.i32)))):
        k, c = t.kernel(*args)
        out.append((name, k.specialize(**c)))
    return out


def main():
    args = _common.parse(__doc__)
    if _common.static(args, kernels()[0][1]):
        return
    _common.require_gpu()
    import torch
    import torch.nn.functional as F

    bf = torch.bfloat16
    m, i, n, v = 4096, 14336, 4096, 151936
    x = torch.randn(m, 2 * i, device="cuda", dtype=bf)
    g, u = x[:, :i], x[:, i:]
    print(f"gelu_and_mul on the halves of a {m}x{2 * i} gate/up output\n  {gelu_and_mul.plan(g, u)}")
    _common.check("template vs fp32 torch", gelu_and_mul(g, u),
                  F.gelu(g.float(), approximate="tanh") * u.float(), rel=2e-2)

    h = torch.randn(m, n, device="cuda", dtype=bf)
    r = torch.randn_like(h)
    w = torch.randn(n, device="cuda", dtype=bf)
    hf = h.float()
    print(f"rms_norm, {m}x{n}\n  {rms_norm.plan(h, w)}")
    _common.check("template vs fp32 torch", rms_norm(h, w),
                  hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float(), rel=2e-2)
    y, s = add_rms_norm(h, r, w)
    _common.check("fused add: the residual", s, h + r, rel=0)
    print(f"softmax, {m}x{n}\n  {softmax.plan(h)}")
    _common.check("template vs fp32 torch", softmax(h), torch.softmax(hf, -1), rel=2e-2)

    logits = (2 * torch.randn(512, v, device="cuda")).to(bf)
    labels = torch.randint(0, v, (512,), device="cuda")
    labels[::20] = -100
    print(f"cross_entropy, 512x{v}\n  {cross_entropy.plan(logits, labels)}")
    _common.check("template vs fp32 torch", cross_entropy(logits, labels),
                  F.cross_entropy(logits.float(), labels, reduction="none"), rel=1e-3)
    if args.no_bench:
        return

    out = torch.empty(m, i, device="cuda", dtype=bf)
    yo = torch.empty_like(h)
    c_gelu = torch.compile(lambda x: F.gelu(x[:, :i], approximate="tanh") * x[:, i:])
    c_rms = torch.compile(lambda x, w: (x.float() * torch.rsqrt(
        x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(bf))
    c_sm = torch.compile(lambda x: torch.softmax(x.float(), -1).to(bf))
    c_ce = torch.compile(lambda l, y: F.cross_entropy(l.float(), y, reduction="none"))
    for title, fns in (
            ("gelu_and_mul", {"template": lambda: gelu_and_mul(g, u, out=out),
                              "torch.compile": lambda: c_gelu(x)}),
            ("rms_norm", {"template": lambda: rms_norm(h, w, out=yo),
                          "hk.ops.rms_norm": lambda: hk.ops.rms_norm(h, w, out=yo),
                          "torch.compile": lambda: c_rms(h, w)}),
            ("softmax", {"template": lambda: softmax(h, out=yo),
                         "hk.ops.softmax": lambda: hk.ops.softmax(h, out=yo),
                         "torch.compile": lambda: c_sm(h)}),
            ("cross_entropy", {"template": lambda: cross_entropy(logits, labels),
                               "torch.compile": lambda: c_ce(logits, labels)})):
        best = _common.compare(fns, args.rounds)
        _common.report(title, best, baseline="torch.compile")


if __name__ == "__main__":
    main()
