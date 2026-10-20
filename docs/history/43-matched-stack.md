# The matched stack: the end-to-end record on triton-rocm 3.7.1, and the allocation that slows Triton's decode

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
stock decode steps (112.2 ms at 4096 x 32 there and below).

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

This chapter measures the first, as asked: the Triton the torch pins. The
third is the original venv. The second was not available here (the
cluster cannot build ROCm/triton's LLVM).

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
`paged_ab.py`) used `PYTHONDONTWRITEBYTECODE=1` and caches outside it.
hk was origin/main at 2e59199, `hk._mlir` built for cp312 in a directory
of its own; every run used GPU 3 alone.

**The kernel.** `paged_ab.py` (chapter 20's) under each venv, ms a layer
with every layer's cache cold, graph clock:

| ctx x batch | Triton on hk's cache, 3.7.1 | 3.8.0 | vLLM 0.16's call, 3.7.1 | 3.8.0 | hk's walk, either |
|---|---|---|---|---|---|
| 1024 x 16 | 0.1201 | 0.1219 | 0.1217 | 0.1203 | 0.1055 |
| 1024 x 32 | 0.1804 | 0.2640 | 0.1836 | 0.2933 | 0.1916 |
| 4096 x 16 | 0.3737 | 0.3892 | 0.3981 | 0.3824 | 0.3775 |
| 4096 x 32 | 0.6874 | 0.9618 | 0.6875 | 1.0762 | 0.7461 |

The copy runs 3.7.1's kernel, and the original 3.8.0's.

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
32 (128 against 64 at 48 and 64 requests). The routing counters say
Triton's kernel was in the captured graphs (324 captured calls, 0 of
hk's). Decode step, ms (median over rounds):

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
pages, outputs checked:

ALLOC_TABLE

What the other bench tools measured, chapter 20's `paged_ab.py` and
chapter 38's `paged_variants.py` included, was the compact case: they
allocate the pages a batch reads and nothing more. That is the paradox of
chapters 20 and 38. It is not graph capture and not the Triton version: in
an allocation the size of vLLM's, Triton's 2-D kernel runs ALLOC_SLOWDOWN
slower, and hk's walk does not. Which part of the memory system makes the
difference -- the page table's fragments for a large allocation, say -- was
not established.
