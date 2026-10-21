# T1: GELU(tanh)-and-mul (Gemma's MLP activation)

Write a GPU kernel for the activation between the fused gate/up projection and
the down projection of a Gemma MLP (vLLM's `GeluAndMul(approximate="tanh")`).

## Semantics

Input `x`, shape `(M, 2*I)`, bfloat16, contiguous. The first `I` columns are the
gate, the last `I` the up projection:

```
gate = x[:, :I]
up   = x[:, I:]
out  = gelu_tanh(gate) * up                  # shape (M, I), bfloat16
gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g**3)))
```

i.e. `torch.nn.functional.gelu(gate, approximate="tanh") * up`. Compute in fp32
and round once to bfloat16 at the end (rounding the GELU to bf16 first, as
eager torch does, also passes).

## Interface

`solve(x) -> out`, a new `(M, I)` bfloat16 tensor. Do not modify `x`.

## Shapes

`M` is the number of tokens: decode batches (1, 16, 37 -- not a multiple of 16)
and prefill chunks (128, 1024, 4096). `I` is the intermediate size: 9216
(Gemma-2-2B), 14336 (Gemma-2-9B), 21504 (Gemma-3-27B).

| name | M | I |
|---|---:|---:|
| M1-I14336 | 1 | 14336 |
| M16-I14336 | 16 | 14336 |
| M37-I9216 | 37 | 9216 |
| M128-I14336 | 128 | 14336 |
| M1024-I14336 | 1024 | 14336 |
| M4096-I14336 | 4096 | 14336 |
| M4096-I9216 | 4096 | 9216 |
| M1024-I21504 | 1024 | 21504 |

Inputs are `torch.randn` in bfloat16 (seeded; `reference.py:make_inputs`).

## Correctness

Against the fp32 reference: `|out - ref| <= 1e-2 + 2e-2 * |ref|` elementwise,
dtype bfloat16, shape `(M, I)`.

## What is measured

Time per call against torch eager, `torch.compile` (default mode,
`dynamic=False`), a Triton kernel, and hk's tuned `hk.ops.silu_mul` as a proxy
(same memory traffic, SiLU rather than GELU -- hk has no GELU-and-mul). The op
moves `3 * M * I * 2` bytes; at prefill sizes it is bandwidth-bound.
