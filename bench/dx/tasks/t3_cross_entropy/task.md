# T3: cross-entropy forward over a large vocabulary

Write a GPU kernel for the per-token loss of a language model's training or
evaluation step: a reduction over a very long row (up to 151936 columns) plus a
gather, with padding tokens ignored.

## Semantics

Inputs: `logits` `(T, V)` bfloat16, contiguous; `labels` `(T,)` int64, each in
`[0, V)` or `-100` (ignored, as `ignore_index=-100`).

```
loss[t] = logsumexp(logits[t, :]) - logits[t, labels[t]]     # in fp32
loss[t] = 0                                                   # where labels[t] == -100
```

i.e. `torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")`.
Output: `loss` `(T,)` float32.

## Interface

`solve(logits, labels) -> loss`. Do not modify the inputs.

## Shapes

`T` tokens x vocabulary `V`: 32000 (Llama-2, Mistral), 128256 (Llama-3), 151936
(Qwen2/Qwen3).

| name | T | V |
|---|---:|---:|
| T64-V151936 | 64 | 151936 |
| T512-V151936 | 512 | 151936 |
| T2048-V151936 | 2048 | 151936 |
| T8192-V151936 | 8192 | 151936 |
| T4096-V128256 | 4096 | 128256 |
| T8192-V32000 | 8192 | 32000 |

Inputs: `logits = 2 * randn` in bf16; `labels` uniform in `[0, V)` with 5% set to
`-100` (seeded).

## Correctness

`|loss - ref| <= 2e-3 + 1e-4 * |ref|` elementwise against an fp64 reference;
float32, shape `(T,)`; exactly 0 where the label is -100.

## What is measured

Against torch eager, `torch.compile`, and a Triton kernel (one program per
row, an online logsumexp over column blocks, as Liger's / Unsloth's). hk.ops
has no cross-entropy and no logsumexp. The op reads `T*V*2` bytes.
