# Against the engine, which is the number that matters

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Everything [above](06-phase-7-paged-decode.md) measures a decode step against vLLM's Triton attention. That
is the wrong target twice over: the point of this IR is to patch kernels an
engine is *missing* a good one for and make the request faster, so the
baseline is the engine as it ships, and Triton attention is the best-tuned
kernel on the ROCm board rather than a gap.

A profile says where the time is. Qwen3-8B, batch 16, context 4096, by device
time (`tools/hk-bench/vllm_profile.py`):

```
57.6%  hipBLASLt GEMM
39.1%  Triton prefill attention
 1.0%  silu-and-mul
 0.9%  Triton decode attention        <- what the whole section above is about
 0.6%  fused_add_rms_norm
 0.4%  rms_norm
 0.2%  rotary embedding
 0.1%  reshape_and_cache
```

The decode attention is **0.9%** of this workload. The prefill attention in
front of **39%** is served by the dense kernel, which [Phase 4](03-phase-4-attention.md) already put at
parity with its handwritten C++ and 2.9-3.2x over aotriton -- and the backend
routes prefill to it for any prompt that fills a query tile.

Total request time, arms alternating in separate processes, stock vLLM
choosing its own backend as the 1.00x (`tools/hk-bench/vllm_e2e.py`, 72 output
tokens, two rounds, min; the two hk rounds agreed to 0.3%):

```
   shape    stock ms      hk ms   speedup
  1024x1      3261.2     3753.4    0.87x
  1024x4      4077.5     4565.0    0.89x
 1024x16      7515.4     7730.0    0.97x
  4096x1      4491.8     4589.1    0.98x
  4096x4      9085.3     7906.2    1.15x
 4096x16     27455.9    21196.0    1.29x
```

**1.29x end to end on the prefill-heavy shape**, and the ordering across the
table is exactly what the profile predicts: the win tracks prefill's share of
the request, so it grows with context and with batch, and at short context
with a small batch -- where the request is almost all decode -- the 1.15x
decode deficit shows through as 0.87x.

What that says about the tool rather than the kernel: the thing worth patching
was not the kernel that was hardest to beat, it was the one with 39% of the
time in front of it, and finding that took a profile rather than an argument.
The remaining 57.6% is GEMM, which is where the next look goes.

Four facts about vLLM's plugin surface are written down in
`vllm_backend.py` because each cost a run to find: `AttentionBackendEnum` is
closed and `register_backend` overrides rather than adds; the reserved
`CUSTOM` member is rejected by `RocmPlatform`'s allowlist, so the override has
to target a member the platform accepts (`TRITON_ATTN`, which is also ROCm's
default -- hence the separate `HK_VLLM_BACKEND=1` opt-in); `get_name()` must
answer with the member it overrides because vLLM round-trips the name through
the enum; and short prompts cannot use the dense kernel and cannot be padded
into it, because zero-padded keys are not masked out, they are keys with score
zero and `exp(0)` is a real weight in the softmax sum.

Scope, stated rather than discovered: no paged *prefill* kernel, so prefix
caching and chunked prefill are refused rather than silently attending over
part of a context. The backend also declares that it cannot run inside a full
CUDA graph -- it plans its KV split on the host from `max_seq_len`, which a
graph would freeze at the capture batch -- so with graphs on, vLLM captures
the model piecewise and runs this attention eagerly between the pieces.

---

← [Phase 7](06-phase-7-paged-decode.md) · [Development record](README.md) · [The linear layers](08-linear-layers.md) →
