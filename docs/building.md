# Building

What has to be built, how, and what a wheel carries. For the short version,
see [getting started](getting-started.md).

- [The submodule](#the-submodule)
- [MLIR and the dialect](#mlir-and-the-dialect)
- [Wheels](#wheels)
- [Using a checkout without a wheel](#using-a-checkout-without-a-wheel)
- [Layout](#layout)

## The submodule

This repository is `hk`, the language and its compiler. Its backend, the
HipKittens C++ tile library with its RDNA3 and RDNA4 ports, is the
`third_party/HipKittens` submodule, pinned to the commit the passes were
validated against. Clone with `--recursive`, or run `git submodule update
--init`.

The submodule tracks the `rdna` branch of
`https://github.com/vivienfanghuagood/HipKittens.git` (`.gitmodules`). The pin
matters: the passes encode facts about those headers, down to how many
`ds_read`s a load issues (`python/hk/runtime/compile.py`).

## MLIR and the dialect

The backend is the stock hipcc compiling the translated C++ against
HipKittens' `include/rdna3`, so nothing at run time needs LLVM. The dialect
does, once:

```bash
git clone --recursive https://github.com/vivienfanghuagood/hk-dsl.git && cd hk-dsl
scripts/build_mlir.sh       # MLIR 23.1.3 only, X86 target, Python bindings (~20 min on 64 cores)
scripts/build_hk_mlir.sh    # the dialect, passes, translation and hk._mlir (~1 min)
scripts/build_wheel.sh      # optional: a wheel for this interpreter, headers included
```

`build_mlir.sh` takes `LLVM_TAG`, `PYTHON`, `HK_LLVM_ROOT`, `JOBS` and
`LINK_JOBS`; keep `JOBS` × ~2 GB under the memory limit (a 48 Gi pod is
`JOBS=6 LINK_JOBS=1`). The bindings land in `python/hk/_mlir`, about 36 MB.

From the scripts themselves:

* **`scripts/build_mlir.sh`** builds MLIR only, for the host target only (hk
  lowers to HipKittens C++ and hipcc does the GPU code generation, so nothing
  needs AMDGPU), Release with assertions, with the Python bindings for
  `$PYTHON`. `LLVM_TAG` defaults to `llvmorg-23.1.3` and `HK_LLVM_ROOT` to
  `/dev/shm/hk-llvm`; source, build and install directories default to
  `$HK_LLVM_ROOT/llvm-project`, `build-cpXY` and `install-cpXY` and can be set
  with `LLVM_SRC`, `LLVM_BUILD` and `LLVM_INSTALL`. It needs `nanobind>=2.9,<3.0`,
  `numpy` and `PyYAML` for `$PYTHON`. With no network, put a checkout at
  `$HK_LLVM_ROOT/llvm-project` first; the clone is skipped when one is there.
  `JOBS` defaults to `nproc`, which inside a container is the node's core
  count, not the cgroup's.
* **`scripts/build_hk_mlir.sh`** builds `mlir/` against that install and
  installs the `hk._mlir` package into `python/hk/_mlir`. The build directory
  is `$HK_MLIR_BUILD` (default `$HK_LLVM_ROOT/hk-build-cpXY`); `LIT` names the
  `lit` to use for the tests. Afterwards `ninja -C $HK_MLIR_BUILD check-hk`
  runs the lit tests ([testing](testing.md)).

Bindings are per CPython version, so a machine that runs two interpreters
builds both scripts twice, once with each `PYTHON`.

## Wheels

**Wheels.** `scripts/build_wheel.sh` builds one per interpreter, because
`hk._mlir` is nanobind extension modules for one CPython:
`hk_dsl-0.2.0-cp310-cp310-linux_x86_64.whl` (11.2 MB) and
`hk_dsl-0.2.0-cp312-cp312-linux_x86_64.whl` (11.3 MB) -- named `hipkittens` when
they were measured, before the split. Each
carries the package, include/ (the generated kernels include it at JIT
time) and the bindings. A wheel built for another interpreter installs and
then fails at import, so the script refuses to build one. The link-time
`libHKPythonCAPI.so` symlink is dropped, because a wheel would carry it as a
second 34 MB copy. Installed outside the checkout, each wheel
traces, translates, compiles against its own headers and launches on the
W7900D, with a result equal to torch's bit for bit.

```bash
scripts/build_wheel.sh                                   # python3, wheel in dist/
PYTHON=/opt/venv/bin/python3 ISOLATION=0 scripts/build_wheel.sh
pip install dist/hk_dsl-0.3.1-cp310-cp310-linux_x86_64.whl
```

Releases carry the cp310 and cp312 wheels as assets
(`gh release create vX.Y.Z dist/*.whl`), and a manylinux copy of each, because
PyPI refuses the `linux_x86_64` tag. `scripts/manylinux_tag.sh` makes them: it
retags each wheel with the tag `auditwheel` says it already satisfies,
`manylinux_2_35` for the cp310 wheel built on Ubuntu 22.04 and `manylinux_2_38`
for the cp312 one built in the Ubuntu 24.04 pod. Nothing is grafted and the
binaries are not touched, since the wheels link only libc, libm, libgcc_s and
libstdc++. Installed into a fresh venv, the retagged cp310 wheel pulls its
three dependencies from PyPI, traces, translates and builds outside the
checkout.

```bash
PYTHON=/path/with/auditwheel/python3 scripts/manylinux_tag.sh dist/*linux_x86_64.whl
gh release upload vX.Y.Z dist/manylinux/*.whl
```

`.github/workflows/pypi.yml` uploads a release's manylinux wheels to PyPI. It
builds nothing, because a hosted runner has neither the ROCm nor the MLIR
toolchain. It runs by hand with `gh workflow run pypi.yml -f tag=vX.Y.Z`, and
on every published release once the repository variable `PYPI_PUBLISH` is
`true`. It uses the `PYPI_API_TOKEN` secret of the `pypi` environment, or PyPI
trusted publishing when that secret is not set. The repository has neither
yet. 0.3.1, the first version on PyPI, was uploaded by hand with twine from
the same manylinux wheels.

What is in the wheel (`scripts/build_wheel.sh`, `pyproject.toml`):

* `python/hk`, the package;
* `hk/include`: HipKittens' headers from the submodule and hk's own from
  `include/`, merged;
* `hk/_mlir`, built for that interpreter by `scripts/build_hk_mlir.sh` with the
  same `PYTHON` — the script checks the extension's `SOABI` and stops if it
  does not match;
* `hk/tuned/*.json`, the shipped tuning records and routing tables, so a fresh
  install picks the measured schedule on its first call
  ([autotune](runtime.md#autotune)).

`OUT` changes the output directory (default `dist/`), and `ISOLATION=0`
builds with the interpreter's own setuptools (>= 64), for a machine without an
index to fetch one from. The package's dependencies are `pybind11>=2.10`,
`numpy` and `typing_extensions`. torch is deliberately not one: hk traces and
compiles without it, only launching needs it, and on a ROCm machine it is
already installed by something with strong opinions about which build. The
wheel also registers the `vllm.general_plugins` entry point, which does
nothing unless `HK_PATCH_VLLM=1` ([integration](integration.md#turning-them-on)).

Wheels are Linux x86_64 only, one CPython each, and are built by hand today
on the machines that have the toolchain.

## Using a checkout without a wheel

With `hk._mlir` built, put `python/` on `PYTHONPATH` — which is what
`examples/` and `tools/hk-bench/` do — or run `pip install -e .` from the repo
root. In a checkout the generated C++ finds its headers in `include/` and the
submodule; `HK_INCLUDE_DIR` overrides that ([compiling](runtime.md#compiling)).

## Layout

```
hk/dsl/         the language: @kernel / @func, values and operators, the AST
                rewrite, control flow, hk.arch / lds / math / group / collective
hk/types.py     dtypes and the tile / vector / tensor types a kernel is written in
hk/compiler/    the MLIR context, the pass pipeline, the translation entry point
hk/_mlir/       the built bindings (hk dialect + MLIR core), not checked in
hk/target/      gfx1100 / gfx1201 facts, as data
hk/codegen/     the module boundary around a translated kernel (scaffold.py)
hk/runtime/     arch detection, hipcc + content-hash cache, the resource gate,
                and the torch.ops registration (torch_ext.py)
hk/autotune/    schedule search: build the space, keep what passes the gate, time
hk/tuned/       the schedules that search picked, and the measured routing
                tables, as shipped data
hk/ops/         kernels written in the DSL, plus the SDPA drop-in (sdpa.py)
hk/integration/ patching a running vLLM or SGLang; engine.py is what every
                patch shares (op, routes, counters, CUDA-graph contract)
mlir/           the hk dialect (ODS), its passes, the C++ translation,
                hk-opt / hk-translate, lit tests
include/        hk's own headers (pyutils/hk_bind.cuh, the fast pybind entry)
third_party/HipKittens/
                the backend: kittens.cuh and include/rdna3, include/rdna4;
                also the C++ kernels hk is measured against (kernels/rdna3)
scripts/        building MLIR, the dialect and the wheels
tools/hk-bench/ the benchmarks and engine harnesses behind every number here
examples/       complete kernels, each checked against torch and timed
docs/           these pages, and the development record (docs/history)
```

The `hk/` paths are under `python/`. `tests/hk/` is the test suite
([testing](testing.md)).
