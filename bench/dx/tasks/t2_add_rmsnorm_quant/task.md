# T2: fused residual add + RMSNorm + per-token dynamic int8 quantization

Write a GPU kernel for the start of a transformer layer in a W8A8 model: add
the residual, RMSNorm the sum, and quantize the normed row to int8 with one
scale per token (vLLM's `fused_add_rms_norm` followed by
`dynamic_per_token_scaled_int8_quant`, or its fused
`rms_norm_dynamic_per_token_quant` with a residual).

## Semantics

Inputs: `x` `(T, H)` bfloat16, `residual` `(T, H)` bfloat16, `weight` `(H,)`
bfloat16, all contiguous; `eps` a Python float (1e-6).

```
r      = bf16(x + residual)                          # the updated residual (sum in fp32, rounded once)
y      = r * rsqrt(mean(r**2, axis=-1) + eps) * weight   # in fp32, from the bf16 r
scale  = max(|y|, axis=-1) / 127                     # fp32, shape (T, 1)
q      = clamp(round(y / scale), -128, 127) as int8
```

Outputs, in this order: `(q, scale, r)` -- `q` int8 `(T, H)`, `scale` float32
`(T, 1)`, `r` bfloat16 `(T, H)`.

## Interface

`solve(x, residual, weight, eps) -> (q, scale, r)`. As in vLLM, you may write
`r` into `residual` in place and use `x` as scratch (`MUTABLE_INPUTS`); returning
`residual` itself as `r` is fine.

## Shapes

`T` tokens (decode 1-37, prefill 256-8192) x hidden `H`: 4096 (Llama-3-8B,
Qwen3-8B), 5120 (Qwen3-14B), 8192 (Llama-3-70B).

| name | T | H |
|---|---:|---:|
| T1-H4096 | 1 | 4096 |
| T16-H4096 | 16 | 4096 |
| T37-H5120 | 37 | 5120 |
| T256-H4096 | 256 | 4096 |
| T2048-H4096 | 2048 | 4096 |
| T8192-H4096 | 8192 | 4096 |
| T4096-H5120 | 4096 | 5120 |
| T2048-H8192 | 2048 | 8192 |

Inputs: `x`, `residual` are `torch.randn` bf16; `weight` is `1 + 0.1*randn`.

## Correctness

- `r`: within one bf16 ulp of the reference (`rtol=2**-8`, `atol=1e-6`).
- `scale`: `rtol=1e-2` against the reference's fp32 scale, shape `(T, 1)`, float32.
- `q`: int8, every element within 1.25 steps of `y / scale` for the solution's
  own `scale` (`|q - y_ref / scale| <= 1.25`): exact rounding is within 0.5;
  rounding the normed value to bf16 on the way, as vLLM's kernel does
  (`x * rstd`, then `* weight`), adds up to 0.75 at |q| near 127.

## What is measured

Against torch eager, `torch.compile`, a Triton kernel (one program per row, all
three steps fused), and `hk.ops.fused_add_rms_norm` followed by
`hk.ops.quantize` (two hk kernels: the tuned building blocks hk has).
The op reads `2*T*H*2 + H*2` bytes and writes `T*H*(2+1) + 4*T`.
