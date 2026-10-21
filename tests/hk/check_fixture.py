"""A kernel file the way `python -m hk.check FILE.py:NAME` reads one: the
kernel, `reference`, `inputs` and `SHAPES`. tests/hk/ir/test_check.py and
tests/hk/gpu/test_check.py run hk.check over it; docs/check.md shows it.

GELU(tanh)-and-mul on 16x64 register tiles, as the first page of the docs
teaches a kernel (and a deliberately wrong twin, for the failure path).
"""

import hk

ROWS, COLS = 16, 64


def _grid(p):
    return (hk.cdiv(p.o.cols, COLS), hk.cdiv(p.o.rows, ROWS), 1)


@hk.kernel(grid=_grid)
def gelu_mul(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    t = hk.RT(hk.fp32, ROWS, COLS)
    x = a[idx].load(t)
    x.gelu(into=x)                 # hk.math.gelu is the tanh form
    x *= b[idx].load(t)
    o[idx] = x.to(hk.bf16)


@hk.kernel(grid=_grid)
def gelu_mul_wrong(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    """Writes every tile but the last column block: hk.check's NaN fill
    makes the unwritten elements mismatches."""
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    t = hk.RT(hk.fp32, ROWS, COLS)
    x = a[idx].load(t)
    x.gelu(into=x)
    x *= b[idx].load(t)
    if hk.arch.block_idx.x < o.cols // COLS - 1:
        o[idx] = x.to(hk.bf16)


@hk.kernel(grid=lambda p: (p.o.rows * hk.cdiv(p.o.cols, p.CW), 1, 1))
def gelu_mul_strip(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16], *,
                   CW=512):
    """llms.txt's kernel: the same op on register strips, 16-byte runs."""
    S = hk.RS(hk.fp32, 1, CW, 8)
    per_row = hk.cdiv(o.cols, CW)
    row = hk.arch.block_idx.x // per_row
    c0 = (hk.arch.block_idx.x - row * per_row) * CW
    x = a.at[row, c0].load(S, bounded=True)
    x.gelu(into=x)
    x *= b.at[row, c0].load(S, bounded=True)
    o.at[row, c0].store(x, bounded=True)


def gelu_mul_fn(a, b):
    """The same kernel behind a function: allocates and returns the output."""
    import torch

    o = torch.empty_like(a)
    gelu_mul(a, b, o)
    return o


def reference(a, b):
    import torch.nn.functional as F

    return F.gelu(a, approximate="tanh") * b


SHAPES = [(64, 1024), (512, 4096)]


def inputs(device, shape=(64, 1024)):
    import torch

    a = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    b = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    return a, b, torch.empty_like(a)


def fn_inputs(device, shape=(64, 1024)):
    return inputs(device, shape)[:2]
