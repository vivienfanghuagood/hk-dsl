# MoE in SGLang: the routing kernel that failed, hk's expert GEMM, and a leaner launch

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

sglang-radeon's README says MoE models do not run in SGLang on RDNA3. Its
reason is that Triton's fused-MoE kernel faults on gfx1100
([triton#10808](https://github.com/triton-lang/triton/issues/10808)). This
chapter finds a different cause on this stack and fixes it. It then puts hk's
MoE GEMM ([chapter 21](21-moe.md)) under SGLang's MoE layers. On the way,
the one-token case exposed the cost of hk's Python launch, which this
chapter also cuts.

## What stopped the models

The setup was OLMoE-1B-7B in stock SGLang 0.5.18, with sglang-radeon and
its prebuilt kernel ([chapter 24](24-sglang.md)'s pod).
- **The failure is in routing.** OLMoE fails in `sgl_kernel.topk_softmax`
  with `unspecified launch failure`, before any expert GEMM runs. The op
  fails on its own, in a process that ran nothing else.
  `sgl_kernel.moe_align_block_size` runs.
- **Triton's MoE kernel runs.** With the top-k computed in torch, OLMoE ran
  on SGLang's own Triton fused-MoE kernel (triton 3.5.1) and wrote coherent
  text. The torch top-k computes what SGLang's own `fused_topk_torch_native`
  does.
- **So on this stack the routing kernel is what stops MoE.** The Triton
  fault the README cites did not show on OLMoE. Whether it shows with
  another model or another Triton was not tested.

## The patch

`hk.integration.sglang_moe` rebinds two names by identity in every SGLang
module that imported them, as the vLLM patch does:
- **`topk_softmax`.** It becomes softmax and top-k in torch, written into
  the caller's tensors. Softcapping and a correction bias are handled as
  sgl_kernel takes them.
- **`invoke_fused_moe_kernel`.** It becomes `hk.ops.moe`'s GEMM where that
  takes the call: bf16, no scales, zero points or bias, and none of SGLang's
  fused epilogues (TMA, `c_sorted`, fused sum or add, SwiGLU, LoRA, masks).
  `try_get_optimal_moe_config` asks for 16-row blocks up to 256 tokens a
  step. It sets them in the up and the down config together, because the
  two GEMMs share one sort. Everything else goes to SGLang's Triton kernel.

SGLang sorts as vLLM does, with the same `tokens * top_k` sentinel, and its
`moe_align_block_size` fills the whole list. hk's GEMM therefore runs on
SGLang's sort unchanged.

`HK_SGLANG_MOE=1` turns both changes on through the `sglang.srt.plugins`
entry point; `HK_SGLANG_MOE=topk` makes only the routing change. Calling
SGLang's layer outside a server takes the server's setup:
`runtime_context.publish(ServerArgs(...))` and a tensor-parallel group of
one (`tools/hk-bench/sglang_moe_ab.py`).

## The layer

`tools/hk-bench/sglang_moe_ab.py` runs the whole layer, SGLang's
`fused_experts_impl`: the sort, both GEMMs, the activation and the sum. It
uses random bf16 weights of the model's shape. The arms differ only in the
rebinding, and each arm's output is held to the other's before timing. The
baseline is SGLang's default Triton config, since it ships none for this
GPU.

| tokens | OLMoE, stock ms | hk | x | Qwen3-30B-A3B shapes, stock ms | hk | x |
|---|---|---|---|---|---|---|
| 1 | 0.437 | 0.279 | 1.56 | 0.282 | 0.274 | 1.03 |
| 2 | 0.699 | 0.278 | 2.51 | 0.543 | 0.281 | 1.93 |
| 8 | 1.169 | 0.680 | 1.72 | 0.998 | 0.629 | 1.59 |
| 32 | 1.682 | 1.014 | 1.66 | 1.773 | 1.402 | 1.27 |
| 128 | 2.124 | 1.304 | 1.63 | 1.989 | 1.657 | 1.20 |
| 256 | 2.240 | 1.749 | 1.28 | 2.391 | 1.883 | 1.27 |
| 512 | 2.170 | 2.683 | 0.81 | 2.497 | 2.472 | 1.01 |
| 2048 | 4.867 | 7.984 | 0.61 | 3.913 | 6.347 | 0.62 |

The crossover lies between 256 and 512 tokens on both shapes, so the
16-row ceiling stays at 256, as in vLLM. Against Triton's output the hk
layer's relative error is at most 5.8e-3.

## At one token: the launch, not the kernel

The first end-to-end runs lost at one token. `bench_one_batch` decoded
OLMoE at 16.2 ms a step with hk's GEMM against 15.4-15.8 on Triton's, even
though the layer's GPU time had dropped 1.56x. Eager decode of a 1B-active
model at batch 1 is host-bound: the hk layer took 0.27 ms of GPU and as
much host time to issue. So what decided the step was what each launch costs
the CPU.
- **The GEMM call was not the problem.** Issuing one expert GEMM cost
  51.5 µs of host time through SGLang's Triton path and 40.3 µs through hk's
  stand-in.
- **The rest of the hk path was.** The whole layer cost 229 µs stock and
  253 µs with hk. The profile put the difference in hk's launch: the torch
  op alone was 27.3 µs, and the views and the slice around it cost the
  rest.
- **The launch was Python, field by field.** `codeobj`'s launch packed the
  argument with one `struct.pack_into` per pointer, extent and stride. It
  built a new ctypes buffer and `extra` array for every launch, and entered
  a `torch.cuda.device` context even when the device was current. It now
  packs through one precompiled struct, reuses a per-thread buffer and skips
  the context switch when the device is already current. The MoE stand-in
  also stopped asking `takes` twice and stopped slicing a list it would keep
  whole.

| host µs per call, one token | before | after |
|---|---|---|
| hk torch op alone | 27.3 | 17.8 |
| hk MoE stand-in | 40.3 | 25.6 |
| SGLang's Triton MoE launch | 51.5 | 52.5 |
| whole layer, stock / hk | 229.0 / 252.7 | 238.5 / 197.5 |

hk's full GPU suite passed on the new launch: 508 tests and one skip.

## In one process

`bench_one_batch` ran OLMoE with 512 tokens in and 64 out. Both arms use
the torch top-k; they differ in the expert GEMMs. Each was run twice,
alternating, after the change above.

| batch | prefill, Triton GEMMs ms | hk | decode step, Triton GEMMs ms | hk | x |
|---|---|---|---|---|---|
| 1 | 47.2-48.2 | 47.2-47.8 | 16.08-16.12 | 15.01-15.13 | 1.07 |
| 8 | 203.6-203.9 | 203.2-203.6 | 21.89-24.23 | 17.21-17.38 | 1.27-1.40 |
| 32 | 765.4-766.5 | 765.7-766.1 | 33.38-34.60 | 26.02-26.22 | 1.28-1.32 |

Prefill is unchanged, as it should be: 512 or more tokens a step stay on
Triton. Greedy decoding of SGLang's three correctness prompts, 48 tokens
each, came out identical in both arms.

## Served

The servers ran `launch_server` with `HK_SGLANG_MOE=topk` (Triton's GEMMs)
or `HK_SGLANG_MOE=1`, with the plugin applied in the launcher and the
scheduler. Load came from `bench_serving` with random ids, 512 tokens in and
128 out. Every request completed, with no errors in either log.

| concurrency | output tok/s, Triton GEMMs | hk | x | mean TTFT ms, Triton | hk | mean TPOT ms, Triton | hk | x |
|---|---|---|---|---|---|---|---|---|
| 1 | 61.6 | 65.5 | 1.06 | 36.4 | 34.5 | 16.07 | 15.10 | 1.06 |
| 8 | 345.6 | 457.2 | 1.32 | 142.3 | 132.2 | 22.15 | 16.55 | 1.34 |
| 32 | 836.2 | 1057.6 | 1.26 | 328.4 | 311.1 | 35.82 | 27.92 | 1.28 |

These are mean TTFTs, not medians. At concurrency 8 the arms batched their
prefills differently, so the medians came out at 139.8 against 59.1 ms while
the means came out at 141.7 against 137.3. A median that halves while
prefill runs the same kernels is batching, not speed.

The first serving runs were biased the other way as well. The Triton arm
went first and paid Triton's JIT for new shapes inside its measurement, and
its concurrency-32 TTFT fell from 229 to 167 ms once the cache was warm.
Every number above comes from warm caches.

## What this does not cover

- **Prefill.** Steps of more than 256 tokens stay on Triton's GEMM. It ties
  hk at 512 tokens on Qwen3-30B-A3B's shapes, is 1.24x faster on OLMoE's,
  and is 1.4-1.6x faster from 1024 tokens up.
- **Where the routing fix belongs.** It belongs in sgl_kernel or
  sglang-radeon, not in a kernel library. hk replaces the top-k whenever
  `HK_SGLANG_MOE` is set, at the cost of a few small torch kernels a layer.
- **One MoE model end to end.** Qwen3-30B-A3B in bf16 does not fit on one
  48 GB card, so it was measured at the layer only.
- **The baseline's configs.** SGLang's Triton runs its default configs,
  since it ships none for this GPU. A tuned config would narrow the layer
  gap.
- **Eager only**, as in [chapter 24](24-sglang.md).

## Reproducing

```bash
# the layer, both shapes, hk at every token count
HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/sglang_moe_ab.py --block16-max 100000

# served: Triton's GEMMs, then hk's
HIP_VISIBLE_DEVICES=0 HK_SGLANG_MOE=topk python -m sglang.launch_server \
    --model-path allenai/OLMoE-1B-7B-0125-Instruct --attention-backend triton \
    --disable-cuda-graph --dtype bfloat16 --port 30000
HIP_VISIBLE_DEVICES=0 HK_SGLANG_MOE=1 python -m sglang.launch_server ...
python -m sglang.bench_serving --backend sglang --port 30000 --dataset-name random-ids \
    --random-input-len 512 --random-output-len 128 --random-range-ratio 1.0 \
    --num-prompts 32 --max-concurrency 8
```
