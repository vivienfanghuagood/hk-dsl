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
| Radeon AI PRO R9700, RX 9070 | gfx1201 (RDNA4) | language models: not yet, use [vLLM](vllm.md); [SGLang-Diffusion](#sglang-diffusion-text-to-image-and-video): on by default |
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
hk needs one install step in that environment (an SGLang-Diffusion ROCm
environment for a Radeon W7900 / RX 7900, gfx1100, or an R9700 / RX 9070,
gfx1201; there is no prebuilt SGLang image for gfx1201, and hk was checked
there with SGLang 0.5.23). Then generate as usual. hk then takes, by default:

| route | switch | what | models | gfx1100 | gfx1201 |
|---|---|---|---|---|---|
| `sgld_attn` | `HK_SGLD_ATTN` | the DiT's attention (`--attention-backend torch_sdpa`) | Qwen-Image 2.1, Wan2.1 | on | on (from 0.9.14) |
| `sgld_linear` | `HK_SGLD_LINEAR` | the DiT's projections where hk measured faster (with a bias: from 0.9.12) | Qwen-Image 2.1, Wan2.1 | on | off |
| `sgld_rope` | `HK_SGLD_ROPE` | the complex RoPE, bit for bit | Qwen-Image 2.1 | on | off |
| `sgld_qknorm` | `HK_SGLD_QKNORM` | q / k RMSNorm + RoPE as one launch (from 0.9.12) | Qwen-Image 2.1 | on | off |
| `sgld_vaeconv` | `HK_SGLD_VAECONV` | the VAE's conv3d as one conv2d (from 0.9.14; 2-frame calls before), which MIOpen would run on its slow reference kernel | Wan2.1 | on | on (from 0.9.14) |
| `sgld_ditnorm` | `HK_SGLD_DITNORM` | the DiT block norms (residual + LayerNorm + scale / shift) as one launch each (from 0.9.16) | Wan2.1 | on | off |

`HK_<SWITCH>=0` turns one off. The lossy ConvRot
INT8 route stays opt-in (`HK_SGLD_CONVROT=1`). What each one replaces and what
it measured: [SGLang routes](sglang-routes.md#sglang-diffusion-text-to-image-and-video).

For Qwen-Image 2.1:

```bash
pip install -U hk-dsl && python3 -m hk.integration sglang-diffusion install   # once per environment
SGLANG_USE_AITER=0 MIOPEN_FIND_MODE=FAST sglang generate --model-path Qwen/Qwen-Image-2.1 \
    --prompt "a ginger cat asleep on a stack of books" --height 1024 --width 1024 \
    --num-inference-steps 30 --attention-backend torch_sdpa --num-gpus 1 --save-output --output-path out
```

The image is written to `out/`. `-U` matters: an older hk-dsl already in the
environment is not upgraded by a plain `pip install`, and its boot is the one
that runs.

**Check that it is on.** `python3 -m hk.integration sglang-diffusion status`
says whether it is installed and which routes are on. During generation, the
log has a `[hk] sglang-diffusion: routes on: sgld_attn ..., sgld_linear ...`
line, and the worker that runs the model prints how many calls took hk's
kernels:

```
[hk] sglang-diffusion: attention: hk 614 (614 in place, 614 on attn_k3), torch 68 (...); linear: hk ...; rope: hk ...
```

Read the worker's line: the parent process prints one too, at exit, whose
counts stay 0. With `HK_DISABLE=1` there is no `[hk]` line at all. A route
that does not apply to the model counts 0, and so does `rope` where `qknorm`
is on: it does the RoPE in its own launch (`qknorm: hk 1984, engine 0`).

**Wan2.1 (text to video)** is the same, with a video model:

```bash
SGLANG_USE_AITER=0 MIOPEN_FIND_MODE=FAST sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cat walking on the grass" --num-frames 81 --height 480 --width 832 \
    --num-inference-steps 20 --attention-backend torch_sdpa --num-gpus 1 --save-output --output-path out
```

The worker's line then counts the DiT norms (`ditnorm: hk 3640`) and the VAE's
conv3d calls hk took (`vaeconv: conv2d 692`) beside the attention calls. The
line comes every `HK_SGLD_REPORT_S` seconds (60), and the worker exits without
a last one, so the VAE count can miss the decode; a decode of seconds, not
many minutes, is the sign it ran. Some ROCm builds of SGLang send
Wan's fused norm layers to their CUDA kernel, and the first denoising step
fails with `No module named 'cuda'`, with or without hk: that build needs
SGLang's fix for it first. `sglang generate` exits 0 even when generation
fails, so check that `out/` has the video and the log no `Generation failed`.

```bash
HK_DISABLE=1 sglang generate ...                       # stock
HK_SGLD_VAECONV=0 sglang generate ...                  # one route off: HK_<SWITCH>=0
python3 -m hk.integration sglang-diffusion uninstall   # remove it
```

The first generation also compiles hk's kernels (a few seconds each, kept in
`~/.cache/hk`).

## Reference

- [SGLang routes](sglang-routes.md): what each route replaces, what it was
  checked against, and the SGLang-Diffusion boot in detail.
- [Integration](integration.md): the default-on rule, every switch and
  environment variable, the engine contract, and adding a kernel of your own.
