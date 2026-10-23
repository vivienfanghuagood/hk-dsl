# The matched stack: the vLLM 0.31 record on triton-rocm 3.7.1, and the allocation that slows Triton's decode

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 38](38-rdna-losses.md) found two Tritons in the vLLM venv of the
engines' pod: triton-rocm 3.7.1, which torch 2.13.0+rocm7.2 pins, and an
upstream Triton 3.8.0 over the same `triton` package, whose decode kernel
is up to 40% slower at 32-64 requests. Every stock arm measured in that
venv ran 3.8.0. If 3.8.0 made stock slower, hk's end-to-end gains since
then were inflated. This chapter rebuilds the stack with triton-rocm 3.7.1
in a venv of its own, leaving the original alone, and re-measures the
end-to-end record there. It also re-checks the decode routing, and finds why
Triton's decode kernel is slower inside vLLM than on the bench, which had
been open since [chapter 20](20-vllm-031.md).

The answer: the record holds. On the matched stack, with main's hk, stock
vLLM is chapter 20's stock to within 0.9% per request, and every published
end-to-end claim on the W7900D holds or grows -- both patches on Qwen3-8B
1.215-1.855x per request at 1-32 requests (published 1.20-1.75x), AWQ up to
3.78x and 4.47x (3.80x, 4.43x), OLMoE's MoE patch 1.02-1.41x (1.02-1.41x),
the R2 gate on Qwen2.5-7B 1.18-1.32x at 16-128 requests (1.17-1.32x).
Interleaved in one run, stock under 3.8.0 is 0-2.5% slower per request than
under 3.7.1 -- its prefill slower, its decode step faster -- so the chapters
measured against it were inflated by at most 2.5%, and nowhere did a claim
shrink. Triton's decode kernel is up to 2.5x slower in vLLM because vLLM's
cache is one allocation of 26.75 GiB: past 2 GiB the same pages read 2.5x
slower to Triton's kernel and no slower to hk's walk.

## What the venv had

`/workspace/venv-vllm031` was built on October 15 by `vllm031_setup.sh`
([chapter 24](24-sglang.md)'s rebuild). The dist-info directories date its
steps (UTC):
- 05:26:27, triton-rocm 3.7.1, installed with torch 2.13.0+rocm7.2 from
  download.pytorch.org's rocm7.2 index (torch's `Requires-Dist:
  triton-rocm==3.7.1`);
- 05:28:23, triton 3.8.0, during `pip install -r reqs.txt`: xgrammar 0.2.7,
  one of vLLM's common requirements, requires `triton` with no version, and
  triton-rocm is a different distribution name, so pip took the newest
  `triton` from PyPI. Its files went over triton-rocm's in
  `site-packages/triton` (89 files of 3.7.1 that 3.8.0 does not ship, AMD
  headers and three libraries, were left beside them);
- 06:13, vLLM 0.31.0 built from source against that torch; 06:16 amdsmi;
  October 16, FlyDSL 0.4.0 ([chapter 30](30-flydsl-routes-gate-contracts.md)).
  Nothing else changed after.

So 3.8.0 was not installed later over a working stack: the venv had it
from the start, and every measurement in it imported it -- chapter 24's
check, and chapters 30, 31, 32, 35, 36 and 38 (the performance gate's
baseline among them). The first
vLLM 0.31 build ([chapters 20](20-vllm-031.md)-[22](22-int4.md)), on
Python 3.10 in a pod since evicted, cannot be checked; its kernel numbers
match 3.7.1 (Triton's decode on hk's cache at 4096 x 32: 0.6907 ms a
layer there, 0.6874 under 3.7.1 below, 0.9618 under 3.8.0), and so do its
stock end-to-end times (below).

Which Triton is vLLM 0.31's? Three answers:
- torch 2.13.0+rocm7.2's pin, triton-rocm 3.7.1 -- what `pip install` of
  the torch vLLM 0.31 builds against brings;
- vLLM 0.31.0's own ROCm image (`docker/Dockerfile.rocm_base`) builds
  Triton from ROCm/triton `release/internal/3.8.x` at 669b31a, with torch
  from ROCm/pytorch `release/2.13`. vLLM vendors `triton_kernels` from the
  same commit, and under 3.7.1 they fail to import (`cannot import name
  'is_hip_gfx1250'`; vLLM logs it and carries on -- only its MXFP4 MoE and
  `matmul_ogs` paths use them, none of the models here);
- upstream Triton 3.8.0, what the venv had, by way of xgrammar.

This chapter measures the first: the Triton the torch pins. The third is
the original venv. The second was not available here (the cluster cannot
build ROCm/triton's LLVM).

torch is the one vLLM's extensions were built against: torch's files date
from 05:27, vLLM's `_C.abi3.so` from 06:13 with `Found Torch:` that path in
its build log, and its libraries need `libtorch.so` and `libc10.so` by
soname with no RUNPATH into either venv.

## A venv of its own

The original was copied (`cp -a`, 17 GB) to `/workspace/venv-vllm031-tri371`;
in the copy only, the shebangs and `activate` were pointed at the copy,
`triton` 3.8.0 was uninstalled and the triton-rocm 3.7.1 wheel reinstalled
(`--no-deps --force-reinstall`). `pip freeze` differs from the original in
one line, and vLLM imports `triton` 3.7.1 there.

The wheel is chapter 38's, `triton_rocm-3.7.1-cp312-cp312-linux_x86_64.whl`,
sha256 `b200ade2…`, the one cp312 3.7.1 wheel the rocm7.2 index lists
(index hash and file agree). Its RECORD and the original venv's record of
3.7.1 are the same file but for `--target`'s paths; the `libtriton.so`
inside the wheel does not match that RECORD's hash (477,383,385 bytes
against the 460,782,192 RECORD says): pip copies a wheel's RECORD rather
than hashing what it installs, so the published wheel carries a stale hash,
and "the same hash as the venv's record" (chapter 38) compared two copies
of it. The 3.7.1 files 3.8.0 left in the original are the wheel's, byte
for byte, but for `.pyc` files compiled at install.

After all of it, the original's tree -- every path, size and mtime -- and
its `libtriton.so` hash were unchanged. Runs from it below (`venv_ab.py`,
`paged_ab.py`, `paged_alloc.py`) used `PYTHONDONTWRITEBYTECODE=1` and caches
outside it.

The work ran in two parts. The kernel check, the decode routing and the
engine probe (October 20) used hk at 2e59199 (chapter 38's tree) on GPU 3.
The end-to-end record, the allocation bench and the two-venv comparison
(October 22) used hk at main's c35a5e7 -- the code 0.5.0 ships -- on GPU 0,
idle-checked before each run and held under the CI's GPU 0 lock. Each tree's
`hk._mlir` was built for the venv's Python in a directory of its own, the
second linked `-Bsymbolic` as main builds it. One W7900D, alone, every run.

**The kernel.** `paged_ab.py` (chapter 20's) under each venv, ms a layer
with every layer's cache cold, graph clock:

| ctx x batch | Triton on hk's cache, 3.7.1 | 3.8.0 | vLLM 0.16's call, 3.7.1 | 3.8.0 | hk's walk, either |
|---|---|---|---|---|---|
| 1024 x 16 | 0.1201 | 0.1219 | 0.1217 | 0.1203 | 0.1055 |
| 1024 x 32 | 0.1804 | 0.2640 | 0.1836 | 0.2933 | 0.1916 |
| 4096 x 16 | 0.3737 | 0.3892 | 0.3981 | 0.3824 | 0.3775 |
| 4096 x 32 | 0.6874 | 0.9618 | 0.6875 | 1.0762 | 0.7461 |

The copy runs 3.7.1's kernel, and the original 3.8.0's.

## The record on the matched stack

Everything in this section ran in the matched venv (vLLM 0.31.0, torch
2.13.0+rocm7.2, triton-rocm 3.7.1, FlyDSL 0.4.0, Python 3.12; one W7900D)
with hk at c35a5e7, at the shapes of the chapters that published them.
`python -m hk.integration ab --json`, CUDA graphs and torch.compile with
prefix caching and chunked prefill on, as vLLM ships (`HK_EAGER=0
HK_VLLM_DEFAULTS=1`), unless a table says otherwise; two rounds, arms
alternating in fresh engines, each round the best of two repetitions. A
ratio is the median over rounds of stock's time over the arm's, and a
verdict is `ab`'s against the band the rounds measured (2% at almost every
row). Every arm's greedy tokens were identical to stock's at every shape of
every run in this section (72/72; 128/128 and 160/160 where the run was
longer).

The published tables took each arm's best round; `ab` pairs rounds and
takes the median. With two rounds whose spread was 0.1-0.5% at most rows,
the two readings differ by less than the rounding of the published ratios.

### Qwen3-8B

bf16; chapter 20's arms and shapes, and chapter 38's 64 requests. Total time
per request batch as stock over the arm; then the decode step, the slope
between the long run and the short one (72 and 32 tokens; 128 and 64 at 64
requests):

| ctx x batch | stock ms | TRITON_ATTN | hk | gemm | hk+gemm | step: stock ms | hk | hk+gemm | hk+gemm, chapter 20 (3.7.1) | chapter 38 (3.8.0) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1 | 1990.5 | 1.169x | 1.181x | 1.030x | 1.223x | 24.53 | 20.31 | 20.26 | 1.22x | -- |
| 1024 x 4 | 2767.7 | 1.081x | 1.117x | 1.076x | 1.215x | 26.14 | 22.35 | 22.16 | 1.20x | -- |
| 1024 x 16 | 7460.5 | 1.021x | 1.068x | 1.462x | 1.609x | 50.84 | 46.60 | 25.61 | 1.52x | 1.587x |
| 1024 x 32 | 12135.0 | 0.990x | 1.112x | 1.356x | 1.559x | 61.54 | 48.82 | 29.73 | 1.47x | 1.539x |
| 1024 x 64 | 25421.3 | -- | 1.147x | 1.306x | 1.557x | 78.64 | 57.70 | 39.95 | -- | 1.530x |
| 4096 x 1 | 3635.2 | 1.291x | 1.495x | 1.054x | 1.615x | 36.68 | 20.98 | 21.02 | 1.59x | -- |
| 4096 x 4 | 7113.9 | 0.984x | 1.252x | 1.140x | 1.482x | 38.19 | 24.81 | 24.76 | 1.44x | -- |
| 4096 x 16 | 26512.3 | 1.019x | 1.356x | 1.228x | 1.807x | 72.99 | 55.15 | 35.16 | 1.70x | 1.837x |
| 4096 x 32 | 51720.5 | 1.009x | 1.430x | 1.190x | 1.855x | 113.50 | 69.30 | 49.35 | 1.75x | 1.862x |

Every hk, gemm and hk+gemm ratio is a win past its band, in total, prefill
and decode, except gemm's decode step at one and four requests, a tie
(0.986-1.005x): vLLM runs its own `wvSplitK` at M ≤ 5 and the table leaves it
there, as in chapter 20. TRITON_ATTN ties stock at 1024 x 32 and 4096 x
4-32 and wins elsewhere; its prefill loses (0.78-0.95x), its decode step
wins.

- **Stock is chapter 20's stock.** Per request batch it is 0.1-0.9% faster
  than chapter 20 recorded under triton-rocm 3.7.1 at all eight shapes
  (1990.5 against 1992 ms at 1024 x 1, 51720.5 against 51762 at 4096 x 32),
  its decode step within 1.2%.
- **Both patches: 1.215-1.855x per request at 1-32 requests** (chapter 20:
  1.20-1.75x). At 1 and 4 requests within 3% of chapter 20; at 16 and 32,
  6% higher. hk's kernels changed since chapter 20 (the GEMM schedule of
  chapter 28, FlyDSL's rows of chapters 30-31, the decode walk of chapter
  38); stock did not. The claim grew.
- **With chapter 38's decode, 1.557-1.855x at 16-64 requests** (chapter 38,
  against Triton 3.8.0's stock: 1.530-1.862x). Shape by shape within 1.8%
  of chapter 38: 1.3-1.8% higher at 1024 tokens, 0.4-1.6% lower at 4096.
  hk alone: 1.068-1.430x (chapter 38: 1.060-1.434x).
- **hk against TRITON_ATTN**, the backend it replaces, paired by round:
  1.01x, 1.03x, 1.05x and 1.12x at 1024 tokens and 1-32 requests; 1.16x,
  1.27x, 1.33x and 1.42x at 4096 (chapter 20: 1.01-1.39x). At 16 requests
  hk's decode step is 1.0-1.2% behind TRITON_ATTN's (chapter 20: 4.5% and
  6.7%); at 32, 1.20x and 1.52x ahead.
- **The linear patch alone**: 1.03-1.46x per request; a tie in the decode
  step at 1-4 requests and 1.23-1.68x at 16-64. Chapter 36's `ab` example
  read 1.444x and 1.205x at 1024 and 4096 x 16 under 3.8.0; here 1.462x and
  1.228x.

The linear patch's counters, per engine: hk's decode GEMV took 3,132-3,564
captured calls, FlyDSL's GEMM 144-792 (the shipped table's FlyDSL rows,
with FlyDSL 0.4.0 in the venv), hk's GEMM 0-1,080, and vLLM's own op
1,188-1,908 (M ≤ 5).

**Graph mode at 96 and 128 requests, and eager** (chapter 30's
configurations; Qwen3-8B, `stock` against `hk+gemm` on the shipped table):

| mode, ctx x batch | stock ms | hk+gemm | decode step: stock ms | hk+gemm ms | ratio | published (3.8.0) |
|---|---|---|---|---|---|---|
| graphs, defaults off, 512 x 96, 160 tokens | 22947.8 | 1.441x | 75.07 | 47.28 | 1.588x | 1.343x, decode 1.413x |
| graphs, defaults off, 512 x 128, 160 tokens | 29467.5 | 1.483x | 94.98 | 54.97 | 1.728x | 1.442x, decode 1.651x |
| eager, 1024 x 1 | 2041.6 | 1.217x | 25.21 | 20.93 | 1.204x | |
| eager, 1024 x 4 | 2870.2 | 1.206x | 27.38 | 23.54 | 1.163x | |
| eager, 1024 x 16 | 7578.9 | 1.574x | 51.92 | 27.14 | 1.913x | |
| eager, 4096 x 1 | 3716.3 | 1.607x | 37.47 | 21.71 | 1.726x | |
| eager, 4096 x 4 | 7210.9 | 1.453x | 39.69 | 26.30 | 1.509x | |
| eager, 4096 x 16 | 23655.6 | 1.558x | 74.12 | 36.71 | 2.019x | |

The published graph-mode ratios are chapter 30's `+FlyDSL` per request and
chapter 31's decode step on the shipped table; eager, chapter 30 had
1.18-1.57x at 1-16 requests, here 1.21-1.61x. At 96 and 128 requests part
of the gain is stock's: its request is 2.1% and 1.0% slower than chapter
30's stock under 3.8.0 (22947.8 against 22471.3 ms, 29467.5 against
29184.1) and its decode step 4.7% and 2.9% slower (75.07 against 71.72,
94.98 against 92.32), the direction the two-venv run below finds for the
decode step. The rest is hk's: chapter 38's walk came after chapter 30.

### 4-bit: Qwen3-8B-AWQ and Qwen3-14B-AWQ

Chapter 22's arms and shapes. fp16, `VLLM_USE_TRITON_AWQ=1` in both arms,
so stock runs Triton's AWQ kernel; `int4` is `HK_VLLM_INT4=1`, hk's W4A16
kernel up to 64 tokens a call (4,464 captured calls and 0 fallbacks a run
on the 8B model). Total and decode step as stock over the arm, against
chapter 22's (triton-rocm 3.7.1, as recorded):

| model, ctx x batch | stock ms | int4 | chapter 22 | step: stock ms | int4 ms | ratio | chapter 22 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B-AWQ, 1024 x 1 | 4948.2 | 3.782x | 3.80x | 66.15 | 14.95 | 4.424x | 4.46x |
| Qwen3-8B-AWQ, 1024 x 4 | 5696.7 | 2.764x | 2.74x | 66.33 | 15.25 | 4.351x | 4.33x |
| Qwen3-8B-AWQ, 1024 x 16 | 9289.0 | 1.669x | 1.66x | 70.05 | 19.13 | 3.664x | 3.70x |
| Qwen3-8B-AWQ, 1024 x 32 | 14538.8 | 1.299x | 1.30x | 81.54 | 34.78 | 2.345x | 2.27x |
| Qwen3-8B-AWQ, 4096 x 1 | 6680.8 | 2.189x | 2.20x | 78.05 | 27.01 | 2.890x | 2.92x |
| Qwen3-8B-AWQ, 4096 x 4 | 10411.2 | 1.547x | 1.56x | 78.50 | 26.98 | 2.909x | 2.87x |
| Qwen3-8B-AWQ, 4096 x 16 | 29018.4 | 1.143x | 1.15x | 94.56 | 40.34 | 2.346x | 2.34x |
| Qwen3-8B-AWQ, 4096 x 32 | 55096.5 | 1.067x | 1.07x | 132.71 | 86.89 | 1.528x | 1.57x |
| Qwen3-14B-AWQ, 1024 x 1 | 8859.3 | 4.472x | 4.43x | 117.99 | 21.27 | 5.548x | 5.52x |
| Qwen3-14B-AWQ, 1024 x 16 | 17075.5 | 1.694x | 1.69x | 122.90 | 25.52 | 4.819x | 4.71x |
| Qwen3-14B-AWQ, 4096 x 1 | 11513.7 | 2.516x | 2.50x | 132.35 | 34.52 | 3.834x | 3.80x |
| Qwen3-14B-AWQ, 4096 x 16 | 51077.2 | 1.157x | 1.16x | 147.83 | 47.97 | 3.087x | 2.87x |

Every row a win; prefill, vLLM's path in both arms, a tie at every shape
(0.991-1.017x). Per request every ratio is within 1% of chapter 22's;
stock within 1.7%. The two headline numbers are 3.78x (3.80x) and 4.47x (4.43x).

### MoE: OLMoE-1B-7B

Chapter 21's arms and shapes, bf16; `moe` is hk's expert GEMM up to 256
tokens a step, `hk+moe` adds hk's attention backend. Chapter 21 measured
`hk+moe` with Triton's decode kernel at every batch, the routing chapter 20
then reverted above eight requests; its 16- and 32-request `hk+moe` rows
had not been measured since. Here the routing is as it ships (hk's walk from
16 requests):

| ctx x batch | stock ms | moe | hk+moe | step: stock ms | moe ms | hk+moe ms | chapter 21: moe / hk+moe |
|---|---|---|---|---|---|---|---|
| 1024 x 1 | 762.2 | 1.409x | 1.929x | 9.96 | 6.85 | 4.82 | 1.41x / 1.92x |
| 1024 x 4 | 1020.5 | 1.275x | 1.572x | 12.14 | 9.02 | 6.97 | 1.28x / 1.57x |
| 1024 x 16 | 2235.8 | 1.107x | 1.388x | 22.69 | 19.72 | 14.45 | 1.11x / 1.19x |
| 1024 x 32 | 3365.6 | 1.044x | 1.332x | 29.60 | 27.67 | 19.15 | 1.04x / 1.17x |
| 3072 x 1 | 1104.7 | 1.251x | 2.234x | 13.60 | 10.48 | 5.17 | 1.24x / 2.22x |
| 3072 x 4 | 1874.0 | 1.179x | 1.678x | 18.56 | 14.58 | 9.16 | 1.18x / 1.67x |
| 3072 x 16 | 5065.3 | 1.050x | 1.533x | 38.69 | 35.20 | 20.46 | 1.05x / 1.24x |
| 3072 x 32 | 8651.7 | 1.019x | 1.469x | 56.03 | 53.72 | 30.70 | 1.02x / 1.25x |

`moe` is within 0.01 of chapter 21's at every shape, a win everywhere but
3072 x 32 (1.019x, a tie in a 2% band); its prefill ties stock
(0.997-1.004x). `hk+moe` matches chapter 21 at 1-4 requests and is
1.33-1.53x at 16-32 (chapter 21, on the reverted routing: 1.17-1.25x).
Stock is within 0.3% of chapter 21's.

### A model no table had measured: Qwen2.5-7B-Instruct

[Chapter 35](35-any-model.md)'s run, the R2 gate: `gemm` from an empty
record directory, so its first start measures the shapes it lacks within
60 s, as shipped; three rounds. The run was made twice on the matched stack,
on chapter 38's tree (October 20) and on c35a5e7 (October 22); the table
is the second, the first in brackets:

| mode, ctx x batch | stock ms | gemm | prefill | decode step: stock ms | gemm | chapter 35 (3.8.0): total / decode |
|---|---|---|---|---|---|---|
| defaults, 1024 x 1 | 1756.0 | 1.032x (1.033x) | 1.314x | 21.40 | 0.999x | 1.039x / 1.005x |
| defaults, 1024 x 4 | 2463.5 | 1.077x (1.079x) | 1.265x | 22.41 | 0.994x | 1.083x / 1.004x |
| defaults, 1024 x 16 | 6189.9 | 1.319x (1.316x) | 1.293x | 35.74 | 1.363x | 1.318x / 1.383x |
| defaults, 1024 x 32 | 10093.8 | 1.306x (1.300x) | 1.301x | 37.99 | 1.346x | 1.300x / 1.376x |
| defaults, 4096 x 1 | 3172.6 | 1.062x (1.062x) | 1.247x | 31.16 | 0.998x | 1.065x / 1.000x |
| defaults, 4096 x 4 | 6345.4 | 1.156x (1.151x) | 1.274x | 31.99 | 0.998x | 1.151x / 0.995x |
| defaults, 4096 x 16 | 22250.2 | 1.230x (1.217x) | 1.288x | 48.10 | 1.258x | 1.205x / 1.283x |
| defaults, 4096 x 32 | 42347.0 | 1.213x (1.204x) | 1.304x | 54.81 | 1.221x | 1.193x / 1.216x |
| graphs, defaults off, 512 x 96, 160 tokens | 17895.3 | 1.184x (1.177x) | 1.307x | 43.48 | 1.004x | 1.173x / 1.015x |
| graphs, defaults off, 512 x 128, 160 tokens | 25030.5 | 1.319x (1.311x) | 1.308x | 68.68 | 1.359x | 1.313x / 1.358x |

Per request: 1.21-1.32x at 16-32 requests, 1.18x and 1.32x at 96 and 128,
1.03-1.16x at 1-4 (chapter 35: 1.19-1.32x, 1.17x / 1.31x, 1.04-1.15x).
Every total a win; the decode step ties stock at 1-4 requests and at 96
(vLLM's skinny GEMM at M ≤ 5; at 96 the measured routes keep three of four
shapes on vLLM's op), as chapter 35 found; one prefill ratio, 1024 x 1, is
noise in both runs (1.317x and 1.314x, bands past 15%). Greedy tokens
identical at every shape.

## What 3.8.0 did to stock

`tools/hk-bench/venv_ab.py` runs `ab`'s stock arm under each venv's Python
in turn, round by round with the order rotated, and judges the second
against the first with `ab`'s verdicts. Qwen3-8B, vLLM's defaults, graphs,
72 tokens against 32, two rounds; ratio = time under 3.7.1 over time under
3.8.0 (below 1, 3.8.0 is the slower):

| ctx x batch | stock ms, 3.7.1 | total | prefill | decode step, 3.7.1 ms | 3.8.0 ms | decode |
|---|---|---|---|---|---|---|
| 1024 x 16 | 7433.4 | 1.000x tie | 0.978x loss | 51.15 | 50.03 | 1.022x win |
| 1024 x 32 | 12108.2 | 1.008x tie | 0.986x tie | 61.48 | 58.17 | 1.057x win |
| 4096 x 16 | 26548.8 | 0.975x loss | 0.943x loss | 72.84 | 71.14 | 1.024x win |
| 4096 x 32 | 51722.8 | 0.991x tie | 0.941x loss | 113.33 | 100.88 | 1.123x win |

Under 3.8.0 stock's request was 0-2.5% slower: its prefill 1.4-5.9% slower
and its decode step 2-12% faster. Stock on gfx1100 is `ROCM_ATTN`, whose
custom kernel refuses the card and falls back to Triton's, so both moves are
Triton's; which kernel of the fallback moved which way was not taken
apart. Greedy tokens identical (72/72). So a ratio of the time per request
measured against 3.8.0's stock was inflated by at most 2.5%, at 4096 x 16,
and decode-step ratios were deflated by up to 11% at 4096 x 32. The tables
above agree: on the matched stack hk+gemm reads 1.6% lower than chapter
38's at 4096 x 16 and 1.3-1.8% higher at 1024 tokens, and chapter 30's 96- and
128-request rows read higher.

## The decode routing under 3.7.1

hk's backend runs Triton's kernel on hk's cache up to eight requests and
hk's walk from 16 up, because in chapters 20 and 38 giving Triton the
larger batches made the captured step slower, though on the bench Triton's
kernel was the faster at most of them. Chapter 38 measured that step under
3.8.0, and said chapter 20 had seen the same under 3.7.1. Measured again,
in one run with the arms interleaved: `hk` as shipped against `hk+user`, the
same backend with an `HK_VLLM_USER` module that sets
`vllm_backend.TRITON_DECODE_ALSO = ((9, 64),)` (what
`HK_TRITON_DECODE_ALSO=9-64` does), so Triton's kernel takes every decode
batch. Qwen3-8B, vLLM's defaults, graphs on, two rounds; 72 tokens against
32 (128 against 64 at 48 and 64 requests); chapter 38's tree. The routing
counters say Triton's kernel was in the captured graphs (324 captured
calls, 0 of hk's). Decode step, ms (median over rounds):

| ctx x batch | hk's walk | Triton's kernel | Triton / hk | stock |
|---|---|---|---|---|
| 1024 x 16 | 46.53 | 50.06 | 1.076x | 51.04 |
| 1024 x 24 | 47.32 | 50.15 | 1.060x | 53.37 |
| 1024 x 32 | 49.12 | 58.20 | 1.185x | 61.57 |
| 1024 x 48 | 55.36 | 62.61 | 1.131x | 67.96 |
| 1024 x 64 | 59.02 | 70.79 | 1.199x | 79.60 |
| 4096 x 16 | 56.38 | 72.84 | 1.292x | 72.14 |
| 4096 x 24 | 61.82 | 73.82 | 1.194x | 80.46 |
| 4096 x 32 | 68.19 | 102.98 | 1.510x | 112.18 |

Under 3.7.1, too, the step is faster on hk's walk at every batch from 16 to
64, by 6-51%; every row is past its band, outputs are identical (72/72,
128/128), and prefill is the same in both arms. The default stays: Triton
up to eight requests, hk from 16. `python -m hk.integration headroom
--graphs` puts a number on the call: with 15-22 requests at 4096 tokens
(graphs for 16 and 24), Triton's kernel took 0.877 ms a call in the step
(chapter 38, under 3.8.0: 1.93) and hk's walk 0.449; at 48 x 1024, 0.516
against 0.319. On the bench, with only the batch's pages allocated, Triton
3.7.1 takes 0.4985 ms at 24 x 4096 and 0.2944 at 48 x 1024.

## Why Triton's kernel is slower in the engine

A probe (`HK_VLLM_USER`-style, in vLLM's own process, eager so the calls
go through Python) timed the backend's Triton call on the engine's own
tensors at the first decode step of 32 requests at 4096 tokens, three
layers, then changed one thing at a time (median of 20 calls, ms):

| the call | ms |
|---|---|
| as the backend makes it | 1.82-1.83 |
| captured in a CUDA graph and replayed | 1.75 |
| the block table copied to rows 144 wide (vLLM's are 136) | 1.84-1.85 |
| the query cloned contiguous (it was) | 1.82-1.84 |
| the pages it reads copied into a cache of their own, the table renumbered | **0.76** |
| the same, pages in table order | 0.77 |
| hk's walk, on the engine's cache | 0.766 |

The engine's cache is one allocation of 26.75 GiB, every layer a view into
it; the batch read blocks 1-4128 of the layer's 6087, in order (a fresh
engine hands them out sequentially). Copied out, the same pages and the same
strides (65536, 8192, 256, 1), the same kernel runs 2.4x faster, with
output identical to the bit. Graph capture, the table's width and the
query's layout do nothing. hk's walk does not care where the pages are.

`tools/hk-bench/paged_alloc.py` (new) reproduces it without vLLM: one
layer's pages carved out of an allocation of a given size, at its start or
its end, Triton's kernel as the backend calls it and hk's walk on the same
pages, CUDA-graph replays, median of 20, outputs checked. At 32 requests
of 4096 tokens under 3.7.1, ms a call:

| allocation | 0.50 GiB | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 4.00 | 26.00 |
|---|---|---|---|---|---|---|---|---|---|
| Triton 3.7.1, pages at the start | 0.6906 | 0.6922 | 0.6909 | 0.6901 | 0.6917 | 0.6923 | **1.7465** | 1.7412 | 1.7450 |
| the same, pages at the end | -- | 0.6916 | 0.6924 | 0.6923 | 0.6904 | 0.6903 | **1.7426** | 1.7449 | 1.7499 |
| hk's walk, either | 0.740-0.744 | | | | | | | | |

A step, not a slope: up to 1.75 GiB Triton's kernel runs the compact speed,
from 2 GiB (2^31 bytes) on 2.5x slower, wherever in the allocation the pages
sit. hk's walk is 0.740-0.744 ms at every size. With the cache in a 26 GiB
allocation, as vLLM's is, both Tritons and hk at the decode batches (ms a
call; compact from `paged_ab.py` above):

| ctx x batch | Triton 3.7.1, compact | 26 GiB | Triton 3.8.0, compact | 26 GiB | hk's walk, 26 GiB |
|---|---|---|---|---|---|
| 1024 x 16 | 0.1201 | 0.1857 | 0.1219 | 0.4179 | 0.0754 |
| 1024 x 24 | -- | 0.2415 | -- | 0.5522 | 0.1079 |
| 1024 x 32 | 0.1804 | 0.4621 | 0.2640 | 0.8843 | 0.2040 |
| 1024 x 48 | -- | 0.5209 | -- | 1.0711 | 0.3021 |
| 1024 x 64 | -- | 0.7775 | -- | 1.4845 | 0.4063 |
| 4096 x 16 | 0.3737 | 0.8629 | 0.3892 | 1.8618 | 0.3842 |
| 4096 x 24 | -- | 0.9409 | -- | 2.0033 | 0.5789 |
| 4096 x 32 | 0.6874 | 1.7351 | 0.9618 | 3.3028 | 0.7423 |
| 4096 x 64 | -- | 2.7483 | -- | 5.5553 | 1.4995 |

The 26 GiB column is the engine's: 0.9409 ms at 24 x 4096 against the 0.877
headroom measured in the step at 15-22 requests, 0.5209 at 48 x 1024
against 0.516, and under 3.8.0 2.0033 against chapter 38's 1.93. In vLLM's
allocation hk's walk is 1.6-2.5x faster than 3.7.1's kernel from 16
requests up, and 3.5-5.6x faster than 3.8.0's. At 1-8 requests, where
Triton's 3-D kernel runs, the two stay close in the large allocation
(Triton's time over hk's 0.83-1.18x, hk behind at 1024 x 1 and 4096 x 8),
so up to eight requests stay on Triton. (The same sweeps' compact rows are left out: run after a 26 GiB
allocation had been made and freed, both kernels read those pages up to
24% slower than `paged_ab.py` and the threshold run do.)

What the other bench tools measured, chapter 20's `paged_ab.py` and
chapter 38's `paged_variants.py` included, was the compact case: they
allocate the pages a batch reads and nothing more. That is the paradox of
chapters 20 and 38. It is not graph capture and not the Triton version:
past 2 GiB of allocation, Triton's 2-D kernel runs 2.5x slower under 3.7.1
and 3.4x under 3.8.0, and hk's walk does not. Which part of the memory
system makes the difference was not established; a 2^31-byte boundary
points at 32-bit offsets somewhere between the allocation and the loads
(hk's walk addresses pages through buffer resources from each page's own
base), but nothing here tested that.

## What changes in the record

Nothing published shrinks. Where a number moves it is up, and the docs now
carry the matched numbers with their stack:
- README's "Results at a glance": Qwen3-8B per request 1.20-1.75x becomes
  1.22-1.85x at 1-32 requests; the note that the RDNA3 vLLM tables await a
  re-run is gone.
- [Performance](../performance.md) A1: the Qwen3-8B table is this chapter's;
  chapter 38's before-and-after table, chapter 30's graph-mode rows, chapter
  35's Qwen2.5-7B table and chapter 21's MoE table gain the matched numbers
  or are replaced by them; the AWQ table gains them. The stacks table lists
  the matched venv. The "pending re-validation" notes are gone; the
  performance gate's baseline is still 3.8.0's (below).
- [Integration](../integration.md): the decode routing's reason now gives
  the 3.7.1 numbers and the allocation, and the supported-versions line
  names triton-rocm 3.7.1 and how to check it (from the draft of this
  chapter).

## The venv

Restore triton-rocm 3.7.1 in `/workspace/venv-vllm031` -- `pip uninstall
triton`, then the 3.7.1 wheel with `--no-deps --force-reinstall`, as the
copy was made -- or retire it for the copy, and pin it in
`vllm031_setup.sh` by installing the triton-rocm wheel after the
requirements. Reasons: 3.7.1 is what torch 2.13.0+rocm7.2 pins and what
chapters 20-22 measured; the copy differs from the original in that one
`pip freeze` line and runs everything the record runs, outputs identical;
under 3.8.0 the Triton kernels hk's backend and TRITON_ATTN call are slower
(1.1-1.9x in vLLM's allocation at 16-64 requests, 1-8% at 1-8), while
stock's request moves by only 0-2.5%. vLLM 0.31's own image ships yet a
third Triton (ROCm's 3.8.x), so neither is "vLLM's" in every sense, but
3.7.1 is the one that matches the torch the venv has. After restoring, the
performance gate's gfx1100 baseline (`tests/perf/baselines/gfx1100.json`,
measured in this venv under 3.8.0, chapters 31 and 38) needs one refresh,
since its Triton arms change with the Triton. This chapter did not change
the original venv.

## How it was run

```
# pod hfang-hk-vllm, W7900D GPU 0 (HIP_VISIBLE_DEVICES=0); PATH=/workspace/venv-vllm031-tri371/bin:$PATH
# hk at c35a5e7, hk._mlir built for the venv's Python (scripts/build_hk_mlir.sh, HK_MLIR_BUILD of its own)
export HK_EAGER=0 HK_VLLM_DEFAULTS=1
AB="python3 -m hk.integration ab --tokens 72 --short 32 --reps 2"
$AB --model Qwen/Qwen3-8B --arms stock,TRITON_ATTN,hk,gemm,hk+gemm --context 1024 --batch 1,4,16,32 --json e1_1k.json
$AB --model Qwen/Qwen3-8B --arms stock,TRITON_ATTN,hk,gemm,hk+gemm --context 4096 --batch 1,4,16,32 --json e1_4k.json
$AB --model Qwen/Qwen3-8B --arms stock,hk,gemm,hk+gemm --context 1024 --batch 64 --tokens 128 --short 64 --json e2.json
VLLM_USE_TRITON_AWQ=1 $AB --model Qwen/Qwen3-8B-AWQ --dtype float16 --arms stock,int4 --context 1024,4096 --batch 1,4,16,32 --json awq8.json
VLLM_USE_TRITON_AWQ=1 $AB --model Qwen/Qwen3-14B-AWQ --dtype float16 --arms stock,int4 --context 1024,4096 --batch 1,16 --json awq14.json
$AB --model allenai/OLMoE-1B-7B-0125-Instruct --arms stock,moe,hk+moe --context 1024,3072 --batch 1,4,16,32 --json olmoe.json
HK_TUNE_DIR=$(mktemp -d) $AB --model Qwen/Qwen2.5-7B-Instruct --arms stock,gemm --context 1024,4096 --batch 1,4,16,32 --reps 3 --json r2_def.json
env -u HK_VLLM_DEFAULTS HK_TUNE_DIR=$(mktemp -d) $AB --model Qwen/Qwen2.5-7B-Instruct --arms stock,gemm --context 512 --batch 96,128 --tokens 160 --short 128 --reps 3 --json r2_graph.json
env -u HK_VLLM_DEFAULTS $AB --model Qwen/Qwen3-8B --arms stock,hk+gemm --context 512 --batch 96,128 --tokens 160 --short 128 --json e3.json
env -u HK_VLLM_DEFAULTS -u HK_EAGER $AB --model Qwen/Qwen3-8B --arms stock,hk+gemm --context 1024,4096 --batch 1,4,16 --short 24 --json eager.json
python3 tools/hk-bench/venv_ab.py --py tri371=/workspace/venv-vllm031-tri371/bin/python \
    --py tri380=/workspace/venv-vllm031/bin/python --model Qwen/Qwen3-8B --context 1024,4096 --batch 16,32 --json vab.json
# each venv, PYTHONDONTWRITEBYTECODE=1 and caches outside the original
python3 tools/hk-bench/paged_alloc.py --batch 1,4,8,16,24,32,48,64 --context 1024 --width 38 --gib 0,26 --no-ends
python3 tools/hk-bench/paged_alloc.py --batch 1,4,8,16,24,32,64 --context 4096 --gib 0,26 --no-ends
python3 tools/hk-bench/paged_alloc.py --batch 32 --context 4096 --gib 0,0.75,1,1.25,1.5,1.75,2,4,26
```
