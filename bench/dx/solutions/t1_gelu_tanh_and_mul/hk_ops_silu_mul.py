"""hk's tuned SiLU-and-mul on the same shapes: the closest kernel hk.ops has.
It computes silu(gate) * up, NOT gelu -- a proxy for what a tuned hk kernel
moving the same bytes takes, timed but not checked.

On gfx950 it takes the two halves of x as strided views; on gfx1100 it refuses
them (a cross-GPU API difference the goal's G4 is about), so there the halves
are copied to contiguous tensors in setup(), outside the timing."""

import torch

import hk

KIND = "proxy"
PROXY = "silu(gate)*up, not gelu: hk.ops has no GELU-and-mul"
LABEL = "hk.ops.silu_mul (proxy: SiLU, not GELU)"


def setup(x):
    i = x.shape[-1] // 2
    out = torch.empty((x.shape[0], i), device=x.device, dtype=x.dtype)
    gate, up = x[:, :i], x[:, i:]
    try:
        hk.ops.silu_mul(gate, up, out)
    except Exception:  # noqa: BLE001 -- gfx1100: strided halves refused
        gate, up = gate.contiguous(), up.contiguous()
    return lambda: hk.ops.silu_mul(gate, up, out)


def solve(x):
    return setup(x)()
