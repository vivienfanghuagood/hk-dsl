# T6: causal GQA prefill attention with tanh logit soft-capping (Gemma-2)

Write a GPU kernel for Gemma-2's prefill attention: causal, grouped-query,
with the attention logits soft-capped by `cap * tanh(s / cap)` before the
softmax (`attn_logit_softcapping = 50.0`).

## Semantics

Inputs: `q` `(B, Hq, N, D)`, `k` and `v` `(B, Hkv, N, D)`, all bfloat16 and
contiguous (`F.scaled_dot_product_attention`'s layout); `Hq` is a multiple of
`Hkv` and query head `h` reads KV head `h // (Hq // Hkv)`. `scale` and `cap` are
Python floats.

```
s[i, j] = scale * dot(q[i], k[j])
s[i, j] = cap * tanh(s[i, j] / cap)                 # soft-capping
s[i, j] = -inf  where j > i                         # causal
out[i]  = sum_j softmax_j(s[i, :]) * v[j]           # in fp32; out (B, Hq, N, D) bfloat16
```

`scale` is Gemma-2's `query_pre_attn_scalar ** -0.5`: `144 ** -0.5` for
Gemma-2-27B (D = 128), `256 ** -0.5` for Gemma-2-9B (D = 256) -- not `D ** -0.5`.

## Interface

`solve(q, k, v, scale, cap) -> out`. Do not modify the inputs.

## Shapes

| name | B | Hq | Hkv | N | D | model |
|---|---:|---:|---:|---:|---:|---|
| 27b-B1-N1024 | 1 | 32 | 16 | 1024 | 128 | Gemma-2-27B |
| 27b-B1-N4096 | 1 | 32 | 16 | 4096 | 128 | Gemma-2-27B |
| 27b-B1-N8192 | 1 | 32 | 16 | 8192 | 128 | Gemma-2-27B |
| 27b-B4-N2048 | 4 | 32 | 16 | 2048 | 128 | Gemma-2-27B |
| 9b-B1-N2048 | 1 | 16 | 8 | 2048 | 256 | Gemma-2-9B |
| 9b-B1-N8192 | 1 | 16 | 8 | 8192 | 256 | Gemma-2-9B |

Inputs: `q, k = 3 * randn`, `v = randn`, bf16, seeded -- large enough that the
logits reach the cap (a kernel without the soft-cap fails the check).

## Correctness

`|out - ref| <= 3e-2 + 3e-2 * |ref|` against an fp32 reference; bfloat16,
shape `(B, Hq, N, D)`.

## What is measured

Against torch eager (Hugging Face's eager Gemma-2 attention: scores
materialised, softmax in fp32; with the scores in fp32 too, because HF's
bf16 scores are 7% off at these logits and fail the check), `torch.compile` of `flex_attention` with the
soft-cap as a `score_mod` and a causal block mask (what torch offers for a
modified score), and, as *proxies* without the soft-cap (lower bounds on the
time, not the same math), `hk.attention(causal=True)` and torch's SDPA.
