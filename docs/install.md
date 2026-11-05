# Install

```bash
pip install hk-dsl
```

That is the whole install for most users. Then:

- **To make vLLM or SGLang faster**, run it in the environment that runs the
  engine, and read [hk in vLLM](vllm.md) or [hk in SGLang](sglang.md). The
  engine's image already has everything else hk needs.
- **To write kernels**, go on to [the quickstart](quickstart.md).

## What you need

- **Linux x86_64 with glibc 2.35 or newer** (Ubuntu 22.04 and later), and
  **CPython 3.10 to 3.13**: there is one `manylinux_2_35` wheel for each.
  Another interpreter has no wheel: [build from source](#from-source).
- **ROCm's hipcc**, to compile kernels: on `PATH`, or named by `HK_HIPCC`
  (hk finds the HIP headers under `ROCM_PATH`). Which hipcc works with which
  GPU is [the toolchain matrix](toolchains.md); clang 22 (ROCm 7.2) is what
  CI's GPU tier runs on every supported GPU.
- **A ROCm build of torch**, to launch kernels on torch tensors. It is not a
  dependency of the package: install the one you already use.
- **A supported GPU** to run on: a Radeon W7900 / RX 7900 (gfx1100), an R9700
  / RX 9070 (gfx1201) or an Instinct MI355X (gfx950).
  [Supported hardware](supported-hardware.md) says what runs on each; tracing
  and building need no GPU.

The wheel carries the HipKittens headers, the MLIR bindings and the shipped
tuning records, so nothing else is needed at run time. Check it:

```bash
python3 -c "import hk; print(hk.__version__)"
python3 -m hk.integration --status      # the engine routes hk turns on for this GPU
```

What each release adds and changes is in the [changelog](../CHANGELOG.md); the
[API reference](api/README.md) marks what `main` has beyond the release
*New after* it. The
same wheels are attached to each
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases).

## From source

The backend is the `third_party/HipKittens` submodule, so clone with
`--recursive` (or run `git submodule update --init`). The dialect needs MLIR,
built once:

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
git clone --recursive --branch v$(python3 -c "import hk; print(hk.__version__)") \
    https://github.com/vivienfanghuagood/hk-dsl.git
```

(`main` documents, and needs, what is newer than the release.) From the clone:

```bash
python3 -m pytest tests/hk/ir tests/hk/codegen -q    # no GPU, no torch
python3 -m pytest tests/hk/gpu -q                    # an AMD GPU and torch
ninja -C $HK_MLIR_BUILD check-hk                     # lit (a build from source): the dialect, passes, translation
```

A tier that cannot run on this machine is skipped with a reason. The suite
builds for gfx1100 unless `HK_ARCH` says otherwise, so on another GPU run it
with `HK_ARCH` set to that GPU's arch (`HK_ARCH=gfx950` on an MI355X). The
tiers and what each catches are in [testing](testing.md).
