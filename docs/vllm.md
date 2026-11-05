# hk in vLLM

hk replaces some of vLLM's kernels on Radeon GPUs with faster ones. You
install it next to vLLM; there is nothing to configure.

## Enable hk in one command

In the Python environment that runs vLLM (a ROCm build of vLLM and torch):

```bash
pip install hk-dsl
```

That is all: vLLM loads hk's plugin by itself, in every worker. Start vLLM as
you always do:

```bash
vllm serve Qwen/Qwen3-8B
```

**Check that it is on.** Before you start a server, ask hk which routes it
will apply on this GPU:

```bash
python3 -m hk.integration --status
```

```
routes (hk is on by default; HK_DISABLE=1 turns every route off, HK_<ROUTE>=0 one of them):
...
vllm_int4         HK_VLLM_INT4         on   default on (gfx1201)
vllm_gptq         HK_VLLM_GPTQ         on   default on (gfx1201)
vllm_gemm         HK_VLLM_GEMM         on   default on (gfx1201)
vllm_moe          HK_VLLM_MOE          on   default on (gfx1201)
vllm_prefill      HK_VLLM_PREFILL      on   default on (gfx1201)
...
```

Once the server runs, its log has one `[hk]` line per vLLM process, naming
the routes it applied, and for the linear layers how many took hk's kernel:

```
[hk] vLLM routes on: vllm_int4 (default on (gfx1201)), vllm_gptq (default on (gfx1201)), vllm_gemm (default on (gfx1201)), vllm_moe (default on (gfx1201)), vllm_prefill (default on (gfx1201)); ...
[hk] linear layers at load: 145 layers through hk's op; unrouted at load: none
```

No `[hk]` line at all means vLLM did not load the plugin: see
[troubleshooting](#troubleshooting).

**The first start takes longer.** hk compiles its kernels on first use and
times the linear-layer shapes of your model it has no measurement for, for up
to 60 seconds (`HK_AUTOTUNE_BUDGET`), before vLLM sizes its cache. Both are
kept in hk's cache (`~/.cache/hk`, or `HK_CACHE_DIR`), so later starts skip
them. In a container, put `HK_CACHE_DIR` on a volume.

## Supported GPUs and models

| GPU | arch | hk in vLLM |
|---|---|---|
| Radeon PRO W7900, W7900D, RX 7900 | gfx1100 (RDNA3) | on by default |
| Radeon AI PRO R9700, RX 9070 | gfx1201 (RDNA4) | on by default |
| Instinct MI355X, MI350X | gfx950 (CDNA4) | DeepSeek-V4.1's layers and AWQ linear layers ([reference](vllm-routes.md)) |
| Instinct MI300X, MI325X | gfx942 (CDNA3) | off |

What hk takes over depends on what your model has. Each route sends a call to
hk only for the shapes where hk measured faster, and leaves every other call
to vLLM's own kernel:

| your model has | route | gfx1100 | gfx1201 | checked end to end on |
|---|---|---|---|---|
| bf16 linear layers | `vllm_gemm` | on | on | Qwen3-8B, Qwen3-1.7B, Qwen3.5-9B |
| AWQ 4-bit linear layers | `vllm_int4` | on | on | Qwen3-8B-AWQ, Qwen3-14B-AWQ |
| GPTQ 4-bit linear layers (symmetric, no act-order) | `vllm_gptq` | on | on | Qwen3.8-27B-GPTQ-Int4 |
| bf16 MoE experts | `vllm_moe` | on | on | OLMoE-1B-7B |
| AWQ 4-bit MoE experts | `vllm_moe_wna16` | on | on | Qwen3.6-35B-A3B-AWQ |
| attention prefill (head size 64, 128 or 256) | `vllm_prefill` | on | on | MiniCPM5-2B, Qwen3-8B, Qwen3.8-27B, MiniCPM-V-4.7 |
| a vision encoder, with `--mm-encoder-attn-backend TORCH_SDPA` | `vllm_sdpa` | on | off | InternVL2-2B |

The table is for the current release; `--status` prints what the version you
installed does. The full per-GPU table, generated from hk's switches, is
[supported hardware](supported-hardware.md#engine-routes-on-each); which vLLM
builds hk was tested with is the [support matrix](support-matrix.md). Serving
results are on [the scorecard](performance/scorecard.md).

## Turn it off

```bash
HK_DISABLE=1 vllm serve Qwen/Qwen3-8B          # stock vLLM: hk patches nothing
HK_VLLM_GEMM=0 vllm serve Qwen/Qwen3-8B        # one route off: HK_<ROUTE>=0
```

With `HK_DISABLE=1` the log says `[hk] vLLM: off (HK_DISABLE=1), stock`.
`pip uninstall hk-dsl` removes the plugin entirely.

## Troubleshooting

**No `[hk]` line in the server's log.** vLLM did not load the plugin.
Check that `pip show hk-dsl` works with the Python that runs vLLM, and that
`VLLM_PLUGINS`, if you set it, includes `hk_attention`. If the line says
`no route on`, `--status` says why: usually a GPU hk has no routes for, or an
`HK_*` variable left set.

**`[hk] WARNING: vllm: route ... OFF -- engine contract FAILED`.** Your vLLM
changed a function that route replaces, so hk leaves that route off and keeps
the others. vLLM still runs correctly. The versions hk was tested with are in
the [support matrix](support-matrix.md).

**The first request, or the first start, is slow.** That is hk compiling and
measuring once ([above](#enable-hk-in-one-command)); the log says
`[hk] JIT: compiling kernel ...` and `[hk] linear routes at load: measured ...`.
To pay it before serving, measure your model ahead of time with the same
`HK_CACHE_DIR` the server will use:

```bash
python3 -m hk.integration tune --model Qwen/Qwen3-8B      # add --tp N for tensor parallel
```

## Reference

- [vLLM routes](vllm-routes.md): what each route replaces, the shapes it takes
  and declines, its counters and its own switches.
- [Integration](integration.md): the default-on rule, every switch and
  environment variable, the engine contract, and adding a kernel of your own.
