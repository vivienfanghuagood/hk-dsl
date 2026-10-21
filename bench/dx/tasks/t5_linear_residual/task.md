# T5: linear + residual add

Write a GPU kernel for a transformer's output projections with the residual
add fused into the GEMM's epilogue: attention's `o_proj` and the MLP's
`down_proj`, each followed by `hidden = hidden + proj(x)`.

## Semantics

Inputs: `x` `(M, K)` bfloat16, `weight` `(N, K)` bfloat16 (a `torch.nn.Linear`
weight, K-contiguous), `residual` `(M, N)` bfloat16. All contiguous.

```
out = x @ weight.T + residual      # accumulate in fp32, add the residual, round once; (M, N) bfloat16
```

i.e. `F.linear(x, weight) + residual`.

## Interface

`solve(x, weight, residual) -> out`, a new tensor. Do not modify the inputs.
Work on `weight` alone may go in `setup()` (bench/dx/README.md).

## Shapes

`M` tokens in {16, 128, 1024, 4096}; `o_proj` of Qwen3-8B / Llama-3-8B
(K = N = 4096), `down_proj` of Qwen3-8B (K = 12288, N = 4096) and of Llama-3-8B
(K = 14336, N = 4096).

| name | M | K | N |
|---|---:|---:|---:|
| oproj-M16 | 16 | 4096 | 4096 |
| oproj-M128 | 128 | 4096 | 4096 |
| oproj-M1024 | 1024 | 4096 | 4096 |
| oproj-M4096 | 4096 | 4096 | 4096 |
| down-M16-K12288 | 16 | 12288 | 4096 |
| down-M128-K12288 | 128 | 12288 | 4096 |
| down-M1024-K12288 | 1024 | 12288 | 4096 |
| down-M4096-K12288 | 4096 | 12288 | 4096 |
| down-M128-K14336 | 128 | 14336 | 4096 |
| down-M4096-K14336 | 4096 | 14336 | 4096 |

Inputs: `x = randn`, `weight = randn / sqrt(K)`, `residual = randn`, bf16, seeded.

## Correctness

`|out - ref| <= 2e-2 + 2e-2 * |ref|` against the fp32 reference; bfloat16,
shape `(M, N)`.

## What is measured

Against torch eager (GEMM, then add), `torch.compile`, a Triton GEMM with the
add fused (the Triton matmul tutorial, autotuned), and `hk.ops.matmul` followed
by the add in torch. `2*M*N*K` flops.
