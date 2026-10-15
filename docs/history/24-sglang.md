# SGLang, second attempt: a volume, the right torch, and the GEMV F.linear leaves behind

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 23](23-integration-path.md) ended its SGLang attempt with two
requirements for the next one:
- a pod whose models and venvs live on a persistent volume;
- a torch whose HIP major version matches the prebuilt gfx1100 kernel's.

This chapter meets both. Under SGLang, hk's linear layers make Qwen3-8B
serve 1.49-1.81x faster.

## The pod

- **The volume.** The pod mounts a 300 GiB claim of the `disk-local`
  storage class at `/workspace`. It is a hostPath under `/disk/ssd2` on the
  node: a separate 3.5 TB NVMe, not the root disk whose eviction threshold
  took the last pod. Every cache points into the volume (`HF_HOME`,
  `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `TMPDIR`), and
  so do the models and venvs. `WaitForFirstConsumer` binds the volume
  through the scheduler, so the pod is pinned with `nodeSelector`; a
  `nodeName` pod would wait for it forever.
- **No GPU request.** All eight of the node's GPUs were allocated, two to the
  development pod. The pod is privileged, sees every GPU, and runs on GPU 0
  only, never at the same time as the other pod.
- **DNS.** The cluster DNS resolved no external name, though egress was open.
  The pod resolves through the nodes' nameservers (`dnsPolicy: None`). From
  there, download.pytorch.org and hf-mirror.com deliver 66-70 MB/s and PyPI's
  file host about 170 KB/s, so pip goes through a domestic mirror.
- **The torch.** `sglang-kernel-rdna3` 0.4.6.post1 links `libamdhip64.so.7`
  and torch 2.9.1's ABI. The first attempt's torch 2.9.1+rocm6.4 ships HIP 6,
  so the kernel would have loaded a second HIP runtime into the process.
  sglang-radeon validates on `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`,
  which failed to pull through the node's registry mirror. An image already
  on the node carries the same build, torch 2.9.1+gitff65f5b with HIP
  7.2.53211, plus ROCm 7.2.1 and vLLM 0.16.1.
- **The venv.** The image's Python is itself a venv (`/opt/venv`), so
  `--system-site-packages` would have reached `/usr`'s site-packages and no
  torch. A `.pth` file appends `/opt/venv`'s site-packages after the new
  venv's own. SGLang's packages therefore win, and torch comes from the
  image.

sglang-radeon's installer put in SGLang 0.5.18, the prebuilt kernel and
plugin 0.1.5. `sglang-rdna3 doctor` passed every check except one shim for
MiniMax-H3's video model. The image also carries aiter, which SGLang leaves
off unless `SGLANG_USE_AITER` is set. The runs set
`USE_ROCM_AITER_ROPE_BACKEND=0`, so apex's RoPE stays on its native kernel
as sglang-radeon pins it.

## What SGLang's linear runs, and what hk's table says about it

Without aiter, SGLang's `UnquantizedLinearMethod.apply` on ROCm is
`F.linear`, which is torch 2.9.1's hipBLASLt. vLLM's table was measured
against vLLM's own skinny GEMMs, which tie hk's GEMV at M ≤ 3 and so keep
those Ms. Against `F.linear` they are not ties. `route_tune.py --engine
sglang` measures against `F.linear` and writes
`routes_sglang_linear_gfx1100.json`:

| Qwen3-8B, M = 1 | F.linear µs | hk GEMV µs | x | M = 64: x |
|---|---|---|---|---|
| qkv 6144x4096 | 154 | 67 | 2.30 | 1.73 |
| o 4096x4096 | 101 | 55 | 1.84 | 1.61 |
| gate_up 24576x4096 | 597 | 311 | 1.92 | 1.86 |
| down 4096x12288 | 264 | 169 | 1.56 | 1.35 |
| lm_head 151936x4096 | 1908 | 1520 | 1.26 | 1.18 |

qkv's 50 MB of weights at 864 GB/s is 58 µs, so the GEMV runs at 87% of the
bandwidth and F.linear at 38%. Every shape gives M 1-64 to the GEMV. Above
M = 64 the table is run for run the vLLM table's, the prefill GEMM's comb
included.
`apply_linear()` now routes by this table where there is one and by vLLM's
otherwise, and says which.

## In one process

`sglang.bench_one_batch` with tp 1 runs the model in its own process, so a
wrapper applies the patch before it starts. The setup was Qwen3-8B in bf16,
512 tokens in and 64 out, Triton attention, eager. Each arm ran in a fresh
process on GPU 0, one after another.

That benchmark warms only its first batch size. The first stock run
therefore paid Triton's JIT inside its batch 8 and 32 prefills: 2352 and
5376 ms, against 912 and 3969-4117 once the cache was warm. The stock arm
was rerun twice and the warm runs are the baseline.

| batch | prefill, stock ms | hk | x | decode step, stock ms | hk, vLLM's table | hk, SGLang's table | x |
|---|---|---|---|---|---|---|---|
| 1 | 147.3-148.0 | 110.2-110.3 | 1.34 | 44.24-44.28 | 44.25 | 25.06-25.64 | 1.75 |
| 8 | 911.0-912.5 | 744.4-745.2 | 1.22 | 46.57-46.83 | 26.50 | 26.57-26.58 | 1.76 |
| 32 | 3969-4117 | 3114-3120 | 1.30 | 54.81-55.06 | 34.52 | 34.51-34.63 | 1.59 |

With vLLM's table, batch 1 stays on F.linear and gains nothing. With
SGLang's, no call fell back (`{'hk': 576, 'hk_decode': 31680,
'fallback': 0}`).

The batch 1 step can be checked against the table. Added up from the
timings above, its linear layers cost 42.1 ms on F.linear and 23.2 ms on
hk. The step went from 44.3 to 25.1 ms. Even eager, the step is GPU-bound,
and its linear layers are nearly all of it.

## Served

The plugin is `hk.integration._sglang_plugin`, registered under
`sglang.srt.plugins` and gated on `HK_SGLANG_LINEAR=1`. SGLang loaded it in
the launcher and in the scheduler process, and both printed
`[hk] linear -> hk (...; routes measured against sglang F.linear)`.

The setup was `launch_server` with sglang-radeon's flags, then
`bench_serving` with random ids, 512 tokens in and 128 out. Each server was
warmed with 16 requests before it was timed. All requests completed on
both servers, with no errors in either log.

| concurrency | output tok/s, stock | hk | x | TTFT ms, stock | hk | TPOT ms, stock | hk | x |
|---|---|---|---|---|---|---|---|---|
| 1 | 22.75 | 41.27 | 1.81 | 93.5 | 51.9 | 43.53 | 24.00 | 1.81 |
| 8 | 159.52 | 270.65 | 1.70 | 534.6 | 431.7 | 46.08 | 26.14 | 1.76 |
| 32 | 456.02 | 678.08 | 1.49 | 606.2 | 489.7 | 54.41 | 34.26 | 1.59 |

## Accuracy

At M 1, 8 and 32 on all five shapes, hk's linear and F.linear land equally
far from an fp64 matmul, to three digits: 1.8-3.5e-3 at worst and 3-4e-4 RMS
relative, the rounding of a bf16 output. SGLang's correctness mode runs
three prompts greedily, 48 tokens each. Two matched stock exactly. The third
took the other side of a near-tie at its ninth token: stock went on "Italy
is Rome", hk "France is Paris", repeated.

## What this does not cover

- **Eager only.** sglang-radeon disables CUDA graphs on RDNA3 because HIP
  graph capture hangs there. The op is graph-safe, and graph-mode numbers
  wait on that fix.
- **MoE.** SGLang's Triton fused-MoE faults on gfx1100, so MoE models do not
  run in SGLang on this card at all. hk's MoE GEMM, which runs in vLLM
  ([chapter 21](21-moe.md)), is the obvious candidate. It is not wired into
  SGLang.
- **One model, one GPU, tp 1.** Another model needs its own table before
  its shapes route (a minute and a half for Qwen3-8B).
- **Attention** is SGLang's Triton backend in both arms. The SDPA rebinding
  (`apply()`) is still not verified inside SGLang.

## Reproducing

```bash
# the table, on the GPU the server will use
HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/route_tune.py --engine sglang \
    --model Qwen/Qwen3-8B --out python/hk/tuned/routes_sglang_linear_gfx1100.json

# served, each arm its own server
HIP_VISIBLE_DEVICES=0 [HK_SGLANG_LINEAR=1] python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --attention-backend triton --disable-cuda-graph \
    --dtype bfloat16 --mem-fraction-static 0.8 --port 30000
python -m sglang.bench_serving --backend sglang --port 30000 --dataset-name random-ids \
    --random-input-len 512 --random-output-len 128 --random-range-ratio 1.0 \
    --num-prompts 32 --max-concurrency 8
```
