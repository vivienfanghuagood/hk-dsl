# hk in SGLang

hk replaces some of SGLang's kernels on a Radeon W7900 / RX 7900 (gfx1100)
with faster ones. You install it next to SGLang; there is nothing to
configure. For text-to-image and video, see
[SGLang-Diffusion](#sglang-diffusion-text-to-image-and-video) below: it needs
one more command.

## Enable hk in one command

In the Python environment that runs SGLang (a ROCm build for RDNA3):

```bash
pip install hk-dsl && python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B \
    --attention-backend triton --disable-cuda-graph
```

That is all: SGLang loads hk's plugin by itself, in the launcher and in every
scheduler process; later starts need only the `launch_server` part. Triton
attention without CUDA graphs is how SGLang runs on RDNA3. Then send a request:

```bash
curl localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Check that it is on.** Before you start a server, ask hk which routes it
will apply on this GPU:

```bash
python3 -m hk.integration --status
```

```
routes (hk is on by default; HK_DISABLE=1 turns every route off, HK_<ROUTE>=0 one of them):
...
sglang_linear     HK_SGLANG_LINEAR     on   default on (gfx1100)
sglang_norm       HK_SGLANG_NORM       on   default on (gfx1100)
sglang_moe        HK_SGLANG_MOE        on   default on (gfx1100)
...
```

Once the server runs, its log has one `[hk]` line per SGLang process naming
the routes it applied:

```
[hk] SGLang routes on: sglang_linear (default on (gfx1100)), sglang_norm (default on (gfx1100)), sglang_moe (default on (gfx1100)); ...
```

No `[hk]` line at all means SGLang did not load the plugin, or hk has no
SGLang route for your GPU: see [troubleshooting](#troubleshooting).

## Supported GPUs and models

| GPU | arch | hk in SGLang |
|---|---|---|
| Radeon PRO W7900, W7900D, RX 7900 | gfx1100 (RDNA3) | on by default |
| Radeon AI PRO R9700, RX 9070 | gfx1201 (RDNA4) | not yet: use [vLLM](vllm.md) |
| Instinct MI300X, MI325X, MI350X, MI355X | gfx942, gfx950 (CDNA) | off |

What hk takes over depends on what your model has. Each route sends a call to
hk only where hk measured faster, and leaves every other call to SGLang:

| your model has | route | checked end to end on |
|---|---|---|
| bf16 linear layers | `sglang_linear` | Qwen3-8B |
| Gemma-style RMSNorm (where SGLang would run its torch fallback) | `sglang_norm` | Qwen3.8-27B |
| bf16 MoE experts, softmax top-k (hk also fixes SGLang's top-k on gfx1100) | `sglang_moe` | OLMoE-1B-7B |

`--status` prints what the version you installed does. Which SGLang builds hk
was tested with is the [support matrix](support-matrix.md); the full per-GPU
route table is [supported hardware](supported-hardware.md#engine-routes-on-each).

## Turn it off

```bash
HK_DISABLE=1 python3 -m sglang.launch_server ...         # stock SGLang: hk patches nothing
HK_SGLANG_LINEAR=0 python3 -m sglang.launch_server ...   # one route off: HK_<ROUTE>=0
```

With `HK_DISABLE=1` the log says `[hk] SGLang: off (HK_DISABLE=1), stock`.
`pip uninstall hk-dsl` removes the plugin entirely.

## Troubleshooting

**No `[hk]` line in the server's log.** On a GPU other than gfx1100 that is
expected: hk has no SGLang route there (`--status` shows them `off`). On
gfx1100, check that `pip show hk-dsl` works with the Python that runs SGLang,
and that `SGLANG_PLUGINS`, if you set it, includes `hk`.

**`[hk] WARNING: sglang: route ... OFF -- engine contract FAILED`.** Your
SGLang changed a function that route replaces, so hk leaves that route off and
keeps the others. SGLang still runs correctly. The versions hk was tested with
are in the [support matrix](support-matrix.md).

**The first requests are slow.** hk compiles each kernel the first time it is
used (`[hk] JIT: compiling kernel ...` in the log) and keeps it in its cache
(`~/.cache/hk`, or `HK_CACHE_DIR`; in a container, put it on a volume). A
model whose linear shapes hk never measured sends them to SGLang until you
measure them once:

```bash
python3 -m hk.integration tune --model Qwen/Qwen3-8B --engine sglang
```

## SGLang-Diffusion (text to image and video)

SGLang-Diffusion runs its model in worker processes that load no plugins, so
hk needs one install command in that environment:

```bash
python3 -m hk.integration sglang-diffusion install     # once per environment
python3 -m hk.integration sglang-diffusion status      # installed, and which routes are on
```

After that, hk's attention and linear kernels are on by default on gfx1100 in
every SGLang-Diffusion process; generate as you always do. Each patched
process prints a line with how many calls took hk's kernels (read the
worker's line: the parent's counts stay 0).

```bash
HK_DISABLE=1 ...                                       # stock
HK_SGLD_LINEAR=0 ...                                   # attention only (HK_SGLD_ATTN=0: linear only)
python3 -m hk.integration sglang-diffusion uninstall   # remove it
```

Checked on Qwen-Image 2.1. The lossy ConvRot INT8 route stays opt-in
(`HK_SGLD_CONVROT=1`).

## Reference

- [SGLang routes](sglang-routes.md): what each route replaces, what it was
  checked against, and the SGLang-Diffusion boot in detail.
- [Integration](integration.md): the default-on rule, every switch and
  environment variable, the engine contract, and adding a kernel of your own.
