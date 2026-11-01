# Install

What hk-dsl needs, which toolchains it is tested with, and how to install it
from a wheel or from source. Then: [the quickstart](quickstart.md).

- [What you need](#what-you-need)
- [From PyPI](#from-pypi)
- [From source](#from-source)
- [The examples and tests: a clone](#the-examples-and-tests-a-clone)

## What you need


You need Linux x86_64 and ROCm's hipcc to compile; a supported GPU (a Radeon
W7900 / RX 7900, gfx1100; an R9700, gfx1201; an Instinct MI355X, gfx950:
[supported hardware](supported-hardware.md) says what runs on each) and torch
to launch. torch is not a dependency of the package — install the
ROCm build you already use. hk finds hipcc on `PATH` (or `HK_HIPCC`, and the
HIP headers under `ROCM_PATH`, [the runtime](runtime.md#environment-variables)).
The GPU tests run with ROCm 7.2 (7.2.4 on gfx1100, 7.2.3 on gfx950) and, on
gfx950, also with the ROCm 10 of vLLM's ROCm image; the no-GPU CI compiles with
ROCm 7.1's hipcc ([testing](testing.md#ci)).

Which hipcc (AMD clang 20 to 24) is tested with which GPU, and what is known to break in 0.8.0 and on
main, is the [toolchain matrix](toolchains.md). In short: clang 22 (ROCm 7.2.x) is what the GPU tests
run on every supported GPU; **hk 0.8.0 does not compile anything under clang 24 (ROCm 10.1)** -- use
ROCm 7.2.x or 10.0 with the 0.8.0 wheel, or main.

In TheRock's layout hipcc is a launcher in `/usr/local/bin` and ROCm is under
`site-packages/_rocm_sdk_devel`, with none of its LLVM tools on `PATH`; hk finds
them beside hipcc's clang ([the runtime](runtime.md#environment-variables)).

## From PyPI

For CPython 3.10 or 3.12 on glibc 2.35 or newer (Ubuntu 22.04
and later):

```bash
pip install hk-dsl
```

A wheel is built per Python interpreter, because it carries the MLIR bindings
for one CPython. 0.8.0 has two on PyPI:

| wheel | glibc |
|---|---|
| `hk_dsl-0.8.0-cp310-cp310-manylinux_2_35_x86_64.whl` | 2.35+ |
| `hk_dsl-0.8.0-cp312-cp312-manylinux_2_35_x86_64.whl` | 2.35+ |

Both are built on glibc 2.35 (cp312 in vLLM's ROCm image), so one wheel per
interpreter serves Ubuntu 22.04, vLLM's ROCm images and anything newer, and
each wheel's glibc is checked against its tag ([building](building.md#wheels)).
0.7.0 also had a cp312 `manylinux_2_38` wheel; 0.6.0 and earlier had only
that one for cp312. The same wheels are attached to each
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases). For
another interpreter, build one ([building](building.md#wheels)).

It carries the HipKittens headers, the bindings and the shipped tuning
records, so nothing else is needed at run time but hipcc.

The release on PyPI is 0.8.0. hk is on by default in vLLM and SGLang for the
routes validated on your GPU (`HK_DISABLE=1` turns them all off, `HK_<ROUTE>=0`
one of them, [integration](integration.md#on-by-default-and-turning-it-off)).
0.8.0 adds workgroup placement as a pass (`hk.kernel(placement=)`),
`hk.map_cpp` and i32 strips, strip ops and element exchange for megakernel
roles, and gfx1201 (RDNA4) on hardware, which 0.7.0 lacked
([changelog](../CHANGELOG.md)). What main adds after 0.8.0, which no release
has yet, the API reference marks *New after 0.8.0*.

## From source

The backend is the `third_party/HipKittens` submodule, so
clone with `--recursive` (or run `git submodule update --init`). The dialect
needs MLIR, built once:

```bash
git clone --recursive https://github.com/vivienfanghuagood/hk-dsl.git && cd hk-dsl
scripts/build_mlir.sh       # MLIR 23.1.3 only, X86 target, Python bindings (~20 min on 64 cores)
scripts/build_hk_mlir.sh    # the dialect, passes, translation and hk._mlir (~1 min)
scripts/build_wheel.sh      # optional: a wheel for this interpreter, headers included
```

Keep `JOBS` × ~2 GB under the memory limit (a 48 Gi pod is
`JOBS=6 LINK_JOBS=1`). Then either install the wheel, or use the checkout in
place by putting `python/` on `PYTHONPATH` (or `pip install -e .`). The
scripts' variables are in [building](building.md#mlir-and-the-dialect).

## The examples and tests: a clone

The wheel carries the package only. The [examples](examples.md) and the test
suite are in the repository: clone it at the tag of the release you installed,
so the examples match the API you have:

```bash
git clone --recursive --branch v0.8.0 https://github.com/vivienfanghuagood/hk-dsl.git
```

(`main` documents, and needs, what is newer than the release: the API
reference marks it *New after 0.8.0*.) From the clone:

```bash
python3 -m pytest tests/hk/ir tests/hk/codegen -q    # no GPU, no torch
python3 -m pytest tests/hk/gpu -q                    # an AMD GPU and torch
ninja -C $HK_MLIR_BUILD check-hk                     # lit (a build from source): the dialect, passes, translation
```

A tier that cannot run on this machine is skipped with a reason. The suite
builds for gfx1100 unless `HK_ARCH` says otherwise, so on another GPU run it
with `HK_ARCH` set to that GPU's arch (`HK_ARCH=gfx950` on an MI355X). The
tiers and what each catches are in [testing](testing.md).
