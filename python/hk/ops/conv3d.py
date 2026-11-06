"""A conv3d as calls of a few output frames along time: no hk kernel, the engine's own conv.

On gfx1100 MIOpen (ROCm 7.2.1, bf16) runs a conv3d of a full-resolution video decoder input
with 6 or more frames on its naive reference kernel, `naive_conv_ab_nonpacked_fwd_ncdhw`:
Wan's VAE decoder at 832x480 (96 channels, 3x3x3) takes 9.5 s a call there, and 95 ms as two
calls of 2 output frames; inputs of 3-5 frames already get a real solver. Each call here is the
caller's conv on a slice of the input, so the result is the same conv: bit-identical where
MIOpen picks a real solver for both, and closer to fp32 than the naive kernel's where it does not.

    y = frames(conv, x, weight, bias)    # conv(x, weight, bias) is the engine's conv3d

Only for a conv without time padding, stride or dilation (a causal conv that pads time itself,
as Wan's WanCausalConv3d does); anything else is one call.
"""

from __future__ import annotations

from typing import Callable, Optional

#: Output frames per call.
FRAMES = 2


def frames(conv: Callable, x, weight, bias: Optional[object] = None, n: int = FRAMES,
           time_stride: int = 1, time_dilation: int = 1, time_padding: int = 0):
    """conv(x, weight, bias) as calls of `n` output frames along x's time axis (dim 2 of
    N, C, T, H, W). One call where the conv strides, dilates or pads time, or where it has
    at most `n` output frames."""
    import torch  # noqa: PLC0415

    kt = weight.shape[2]
    out = x.shape[2] - kt + 1
    if (x.dim() != 5 or time_stride != 1 or time_dilation != 1 or time_padding != 0
            or out <= n):
        return conv(x, weight, bias)
    return torch.cat([conv(x[:, :, s:min(s + n, out) + kt - 1], weight, bias)
                      for s in range(0, out, n)], 2)
