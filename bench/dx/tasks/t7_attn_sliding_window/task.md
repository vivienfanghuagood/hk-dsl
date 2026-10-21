# T7: causal sliding-window GQA prefill attention (Mistral)

Write a GPU kernel for Mistral-style prefill attention: causal, grouped-query,
and each query sees only the last `window` keys (itself included). Past the
window, whole key blocks can be skipped, so a good kernel's time grows
linearly in the sequence, not quadratically.

## Semantics

Inputs: `q` `(B, Hq, N, D)`, `k` and `v` `(B, Hkv, N, D)`, bfloat16, contiguous
(SDPA's layout); query head `h` reads KV head `h // (Hq // Hkv)`. `scale` and
`window` are Python numbers (`window = 4096`).

```
s[i, j] = scale * dot(q[i], k[j])
visible(i, j) = (j <= i) and (i - j < window)        # flash-attn's window_size=(window-1, 0)
out[i]  = sum_{visible j} softmax_j(s[i, :]) * v[j]   # in fp32; out (B, Hq, N, D) bfloat16
```

## Interface

`solve(q, k, v, scale, window) -> out`. Do not modify the inputs.

## Shapes

Mistral-7B's attention: `Hq = 32`, `Hkv = 8`, `D = 128`, `scale = 128 ** -0.5`,
`window = 4096`.

| name | B | N |
|---|---:|---:|
| B1-N2048 | 1 | 2048 |
| B1-N4096 | 1 | 4096 |
| B1-N8192 | 1 | 8192 |
| B1-N16384 | 1 | 16384 |
| B2-N8192 | 2 | 8192 |

At N <= 4096 the window covers every causal key, so it is plain causal
attention; past it, the window matters. Inputs: `randn` in bf16, seeded.

## Correctness

`|out - ref| <= 3e-2 + 3e-2 * |ref|` against an fp32 reference; bfloat16,
shape `(B, Hq, N, D)`.

## What is measured

Against torch eager (SDPA with a boolean window mask, built once per length),
`torch.compile` of `flex_attention` with the window as a `mask_mod` (its block
mask skips the blocks outside the window), and `hk.attention(causal=True)`:
exact where N <= window (an `hk_ops` baseline there), a *proxy* doing the
full causal work past it.
