# T4: linear + bias + GELU(tanh)

Write a GPU kernel for a linear layer with its bias and activation fused into
the GEMM's epilogue: the MLP up-projection of a GELU model (GPT-2/BERT/ViT
style fc1), at the projection sizes of Qwen3-8B and Llama-3-8B.

## Semantics

Inputs: `x` `(M, K)` bfloat16, `weight` `(N, K)` bfloat16 (a `torch.nn.Linear`
weight: each row is an output feature, K-contiguous), `bias` `(N,)` bfloat16.
All contiguous.

```
out = gelu_tanh(x @ weight.T + bias)        # accumulate in fp32; out (M, N) bfloat16
gelu_tanh(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z**3)))
```

i.e. `F.gelu(F.linear(x, weight, bias), approximate="tanh")`.

## Interface

`solve(x, weight, bias) -> out`. Do not modify the inputs. `weight` and `bias`
are a model's parameters: work on them that does not depend on `x` (a
transpose, a re-layout) may go in `setup()` and is not timed (bench/dx/README.md).

## Shapes

`M` tokens in {16, 128, 1024, 4096}, `K = 4096`, `N` in {12288 (Qwen3-8B's
intermediate), 14336 (Llama-3-8B's)}.

| name | M | K | N |
|---|---:|---:|---:|
| M16-K4096-N12288 | 16 | 4096 | 12288 |
| M128-K4096-N12288 | 128 | 4096 | 12288 |
| M1024-K4096-N12288 | 1024 | 4096 | 12288 |
| M4096-K4096-N12288 | 4096 | 4096 | 12288 |
| M16-K4096-N14336 | 16 | 4096 | 14336 |
| M128-K4096-N14336 | 128 | 4096 | 14336 |
| M1024-K4096-N14336 | 1024 | 4096 | 14336 |
| M4096-K4096-N14336 | 4096 | 4096 | 14336 |

Inputs: `x = randn`, `weight = randn / sqrt(K)`, `bias = 0.5 * randn`, bf16, seeded.

## Correctness

`|out - ref| <= 2e-2 + 2e-2 * |ref|` against the fp32 reference (fp32 GEMM,
bias, GELU); bfloat16, shape `(M, N)`.

## What is measured

Against torch eager (hipBLASLt GEMM, then bias and GELU), `torch.compile`
(the GEMM plus a fused Triton epilogue), a Triton GEMM with the epilogue fused
(the Triton matmul tutorial, autotuned), and `hk.ops.matmul` followed by the
epilogue in torch (hk has no GEMM with an epilogue). `2*M*N*K` flops.
