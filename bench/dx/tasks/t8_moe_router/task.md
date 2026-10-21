# T8: MoE router -- softmax, top-k, renormalise

Write a GPU kernel for a mixture-of-experts layer's routing step (vLLM's
`topk_softmax` with `renormalize=True`): per token, a softmax over the expert
logits, the k largest probabilities, and those k weights renormalised to sum
to one.

## Semantics

Input: `logits` `(T, E)` bfloat16, contiguous (the router GEMM's output);
`k` a Python int.

```
p        = softmax(logits.float(), axis=-1)
w, ids   = topk(p, k)                  # the k largest, in descending order
w        = w / sum(w)                  # renormalised
```

Outputs, in this order: `ids` int32 `(T, k)` and `weights` float32 `(T, k)`,
each row sorted by descending weight. With ties (bf16 logits can tie), any
valid top-k is accepted.

## Interface

`solve(logits, k) -> (ids, weights)`. Do not modify the input.

## Shapes

`T` tokens x (`E` experts, `k`): (8, 2) Mixtral-8x7B, (64, 8) OLMoE-1B-7B,
(128, 8) Qwen3-30B-A3B / Qwen3-235B-A22B.

| name | T | E | k |
|---|---:|---:|---:|
| T16-E8-k2 | 16 | 8 | 2 |
| T4096-E8-k2 | 4096 | 8 | 2 |
| T16-E64-k8 | 16 | 64 | 8 |
| T4096-E64-k8 | 4096 | 64 | 8 |
| T1-E128-k8 | 1 | 128 | 8 |
| T16-E128-k8 | 16 | 128 | 8 |
| T256-E128-k8 | 256 | 128 | 8 |
| T4096-E128-k8 | 4096 | 128 | 8 |
| T16384-E128-k8 | 16384 | 128 | 8 |

Inputs: `logits = 2 * randn` in bf16, seeded.

## Correctness

Per row: the ids are distinct and in `[0, E)`; the reference's fp32
probabilities at the returned ids, sorted, equal its top-k probabilities
(`atol=1e-6`, so the set is a valid top-k); the weights are those
probabilities renormalised (`atol=1e-4`) and in descending order.

## What is measured

Against torch eager, `torch.compile`, and a Triton kernel (one program per
block of rows, the softmax and k rounds of argmax in registers). hk.ops has no
router kernel (`hk.ops.moe` is the expert GEMM and the block alignment).
These are launch-bound at decode sizes: compare `device_us` and `host_us`.
