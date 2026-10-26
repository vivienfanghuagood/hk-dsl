# Building

What has to be built, how, and what a wheel carries. For the short version,
see [getting started](getting-started.md).

- [The submodule](#the-submodule)
- [MLIR and the dialect](#mlir-and-the-dialect)
- [Wheels](#wheels)
- [Using a checkout without a wheel](#using-a-checkout-without-a-wheel)
- [CI](#ci)
- [The performance gate](#the-performance-gate)
- [Layout](#layout)

## The submodule

This repository is `hk`, the language and its compiler. Its backend, the
HipKittens C++ tile library with its RDNA3 and RDNA4 ports and its CDNA3 and
CDNA4 headers, is the `third_party/HipKittens` submodule, pinned to the commit the passes were
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
`hk_dsl-<version>-cp310-cp310-linux_x86_64.whl` and its cp312 twin, about
11 MB each (11.2 and 11.3 MB at 0.2.0, when they were still named
`hipkittens`). Each carries the package, include/ (the generated kernels
include it at JIT time) and the bindings. A wheel built for another interpreter installs and
then fails at import, so the script refuses to build one. The link-time
`libHKPythonCAPI.so` symlink is dropped, because a wheel would carry it as a
second 34 MB copy. Installed outside the checkout, each wheel
traces, translates, compiles against its own headers and launches on the
W7900D, with a result equal to torch's bit for bit.

```bash
scripts/build_wheel.sh                                   # python3, wheel in dist/
PYTHON=/opt/venv/bin/python3 ISOLATION=0 scripts/build_wheel.sh
pip install dist/hk_dsl-0.4.0-cp310-cp310-linux_x86_64.whl
```

Releases carry the cp310 and cp312 wheels as assets
(`gh release create vX.Y.Z dist/*.whl`), and a manylinux copy of each, because
PyPI refuses the `linux_x86_64` tag. `scripts/manylinux_tag.sh` makes them: it
retags each wheel with the tag `auditwheel` says it already satisfies. Both
wheels are built on glibc 2.35 and so are `manylinux_2_35`: cp310 on Ubuntu
22.04 with its python3, cp312 in the vLLM ROCm image (Ubuntu 22.04, gcc 11.4,
`/usr/bin/python3` 3.12). Nothing is grafted and the
binaries are not touched, since the wheels link only libc, libm, libgcc_s and
libstdc++.

**Build cp312 on glibc 2.35, not newer.** A wheel's tag is the glibc it was
built against, and a newer one is not accepted by an older system. 0.3.1
through 0.7.0 built cp312 in rocm/pytorch on Ubuntu 24.04: `libHKPythonCAPI`
needed `GLIBC_2.38` (`__isoc23_strtol`, which glibc 2.38's headers substitute
for `strtol`), so the wheel was `manylinux_2_38`, and `pip install hk-dsl`
found no cp312 wheel in the vLLM ROCm images, which have glibc 2.35. 0.7.0 got
an extra cp312 `manylinux_2_35` wheel, built in the vLLM image, and from 0.8.0
it is the only cp312 wheel; pip on a newer glibc takes it too. Building in the
vLLM image takes three changes from the rocm/pytorch recipe: run it with
`--entrypoint bash`; use `/usr/bin/python3` with explicit
`-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++`; and, with
cmake from pip, put its `cmake/data/bin` on `PATH`, because the `bin/cmake`
wrapper names an interpreter the image lacks.

**`scripts/check_wheel_glibc.py`** checks the result. For every ELF file in a
wheel it reads the symbol versions it requires (`readelf -V`). It fails if the
highest `GLIBC_2.N` is above the wheel's `manylinux_2_M` tag, since such a
wheel installs on glibc 2.M and then fails at import. With auditwheel
importable, it also checks every `GLIBCXX`, `CXXABI` and `GCC` version against
the tag's policy. On 0.7.0's wheels, the cp310 `manylinux_2_35` and both cp312
wheels (`manylinux_2_38` at `GLIBC_2.38`, `manylinux_2_35` at `GLIBC_2.34`)
pass; the 2_38 wheel renamed to 2_35 fails. CI runs it on the IR leg against
the cp310 wheel it builds, and `tests/hk/ir/test_check_wheel_glibc.py` tests
it on wheels made from a system binary. Installed into a fresh venv, the retagged cp310 wheel pulls its
three dependencies from PyPI, traces, translates and builds outside the
checkout.

```bash
PYTHON=/path/with/auditwheel/python3 scripts/manylinux_tag.sh dist/*linux_x86_64.whl
/path/with/auditwheel/python3 scripts/check_wheel_glibc.py dist/*.whl dist/manylinux/*.whl
gh release upload vX.Y.Z dist/manylinux/*.whl
```

`.github/workflows/pypi.yml` uploads a release's manylinux wheels to PyPI. It
builds nothing, because a hosted runner has neither the ROCm nor the MLIR
toolchain. It runs by hand with `gh workflow run pypi.yml -f tag=vX.Y.Z`, and
on every published release once the repository variable `PYPI_PUBLISH` is
`true`. It uses the `PYPI_API_TOKEN` secret of the `pypi` environment, or PyPI
trusted publishing when that secret is not set. The repository has neither
yet. 0.3.1, the first version on PyPI, through 0.7.0 were uploaded by hand with
twine from the same manylinux wheels. The release steps, and the checks that
must pass before a tag, are in [releasing](releasing.md).

**Known issue: the workflow cannot upload yet.** Run for v0.6.0 (run
40186344465), it downloaded the release's wheels and then PyPI refused the
exchange: "no corresponding publisher (Publisher with matching claims was not
found)". The claims it presented were repository `vivienfanghuagood/hk-dsl`,
workflow `.github/workflows/pypi.yml`, environment `pypi`. Fixing it is the
PyPI project owner's step, on pypi.org: hk-dsl, Settings, Publishing, add a
GitHub trusted publisher with those three values (or put a `PYPI_API_TOKEN`
secret in the repository's `pypi` environment). Until then a release goes up
with twine from the manylinux assets, after checking their sha256 against the
GitHub release.

`pyproject.toml` and `hk.__version__` say the last release's version until
the next release raises both, so a wheel built from main between releases
carries the released version's number.

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
index to fetch one from. An older setuptools does not fail. Ubuntu 22.04's
system 59.6 ignores `[project]` and builds `UNKNOWN-0.0.0-*.whl`, so build the
cp310 wheel from a venv with a current setuptools. The package's dependencies are `pybind11>=2.10`,
`numpy` and `typing_extensions`. torch is deliberately not one: hk traces and
compiles without it, only launching needs it, and on a ROCm machine it is
already installed by something with strong opinions about which build. The
wheel also registers two entry points, `vllm.general_plugins` and
`sglang.srt.plugins`, each of which does nothing until one of its switches
is set ([integration](integration.md#on-by-default-and-turning-it-off)).

Wheels are Linux x86_64 only, one CPython each, and are built by hand today
on the machines that have the toolchain.

### Reproducible wheels

The same commit, built with the same toolchain, gives the same wheel byte for
byte, wherever the checkout and the build directory are. 0.7.0's rc and tag
wheels, built from the same tree, did not: the cp310 build ran in a `mktemp -d`
directory, whose name ended up in two `__FILE__` strings in
`libHKPythonCAPI.so` and in the GNU build-id of the three libraries linked
against it. The zip layer differed too: every entry's timestamp was the
build time.

What makes a build reproducible, and where it is done:

* **Paths.** `mlir/CMakeLists.txt` compiles with `-ffile-prefix-map` for the
  checkout (`/hk/src`), the build directory (`/hk/build`), the MLIR install
  (`/hk/mlir`) and nanobind (`/hk/nanobind`), each both as given and with
  symlinks resolved, so neither assertion messages (`__FILE__`) nor debug info
  name a build-machine path. The configure output lists the maps (`-- hk:
  -ffile-prefix-map=...`). `-DHK_REPRODUCIBLE_PATHS=OFF` turns them off.
  Arguments to `scripts/build_hk_mlir.sh` go to the cmake configure, so extra
  flags need no edit: `scripts/build_hk_mlir.sh -DCMAKE_CXX_FLAGS=...`
  (`CFLAGS` and `CXXFLAGS` in the environment work too, on a fresh build
  directory).
* **RPATH.** The bindings are linked with their install RUNPATH, `$ORIGIN`
  (`mlir/python/CMakeLists.txt`). Linked with the build tree's absolute
  `_mlir_libs` path instead, the three libraries that link `libHKPythonCAPI`
  were patched by `cmake --install`, in place: `.dynstr` kept the old string's
  length, so the layout followed the length of the build directory's path, and
  the build-id, hashed at link time, kept the old path itself. That, not code,
  was the "only the build-id differs" of 0.7.0.
* **Timestamps.** `scripts/build_wheel.sh` sets `SOURCE_DATE_EPOCH` to the
  checkout's last commit time unless it is already set, and wheel uses it for
  every zip entry. `scripts/manylinux_tag.sh` rewrites `RECORD`, which takes
  the same `SOURCE_DATE_EPOCH`, so export it for that step too. Without git
  (a worktree mounted into a container, whose `.git` names a host path) the
  script cannot find the commit: pass `SOURCE_DATE_EPOCH` in.
* **File modes.** The wheel records each file's mode, and a checkout made under
  `umask 077` has 0600 files. `build_wheel.sh` makes every staged file 0644, or
  0755 if it was executable, and builds under `umask 022`.
* **Order.** wheel writes the entries sorted by path and `RECORD` in the same
  order, so neither depends on the file system. The wheel carries no bytecode
  (`__pycache__` is dropped from the stage, and setuptools compiles none).

What stays fixed by the toolchain rather than by hk, and so has to be the same
for two builds to match:

* the compiler and linker, whose versions are in each library's `.comment`
  (GCC 11.4 from Ubuntu 22.04 for both, on the host for cp310 and in the vLLM
  image for cp312),
  and the system headers, named as they are (`/usr/include/c++/11/...`);
* the MLIR install. Its static libraries carry the paths of its own build,
  `/dev/shm/hk-llvm/...` for the CI's MLIR 23.1.3, in about 300 assertion
  strings. They are the same for every hk build against that install, and only
  a rebuild of MLIR (with LLVM's `-DLLVM_USE_RELATIVE_PATHS_IN_FILES=ON`)
  removes them;
* nanobind's, setuptools' and wheel's versions.

A release build that checks itself: build twice, from two worktrees of the tag
in different directories with different build directories, and compare.

```bash
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct vX.Y.Z); export SOURCE_DATE_EPOCH
# ... scripts/build_hk_mlir.sh + scripts/build_wheel.sh + scripts/manylinux_tag.sh, twice ...
sha256sum a/*.whl b/*.whl                       # equal pairs: done
python3 -c 'import sys, zipfile; a, b = (zipfile.ZipFile(p) for p in sys.argv[1:]); \
  print([n for n in a.namelist() if a.read(n) != b.read(n)])' a/X.whl b/X.whl
```

If the sha256s differ, the second line names the files that do. For a `.so`,
`readelf -n` shows whether the build-id differs and `strings -a | grep /`
finds a path that leaked; for the zip layer, compare `ZipInfo.date_time` and
`external_attr`. The maintainers' release script (`release-build.sh`, not in
this repository) does all of this, builds cp310 on the host and cp312 in the
vLLM image, runs the glibc check, and writes `SHA256SUMS`; a rebuild of the same tag checks
against it with `sha256sum -c SHA256SUMS`.

## Using a checkout without a wheel

With `hk._mlir` built, put `python/` on `PYTHONPATH` — which is what
`examples/` and `tools/hk-bench/` do — or run `pip install -e .` from the repo
root. In a checkout the generated C++ finds its headers in `include/` and the
submodule; `HK_INCLUDE_DIR` overrides that ([compiling](runtime.md#compiling)).

## CI

CI is two workflows. `ci.yml`, on GitHub's runners, on every push and pull
request, builds the dialect against a prebuilt MLIR 23.1.3 and runs the two
tiers that need no GPU. `gpu.yml`, on two self-hosted runners on the MI355X
workstation, on pushes to main only, runs the GPU tier on a gfx950 and a
gfx1100; its jobs, `gfx950` and `gfx1100`, are main's required checks. How,
what each costs, and how to make a new MLIR archive:
[testing](testing.md#ci) and [the GPU tier](testing.md#the-gpu-tier).

## The performance gate

Moved to [testing](testing.md#the-performance-gate).

## Layout

```
hk/dsl/         the language: @kernel / @func, values and operators, the AST
                rewrite, control flow, hk.arch / lds / math / group / collective
hk/types.py     dtypes and the tile / vector / tensor types a kernel is written in
hk/compiler/    the MLIR context, the pass pipeline, the translation entry point
hk/_mlir/       the built bindings (hk dialect + MLIR core), not checked in
hk/target/      gfx1100 / gfx1201 / gfx942 / gfx950 facts, as data
hk/codegen/     the module boundary around a translated kernel (scaffold.py)
hk/runtime/     arch detection, hipcc + content-hash cache, the resource gate,
                the code-object checker (cocheck.py), and the torch.ops
                registration (torch_ext.py)
hk/autotune/    schedule search: build the space, keep what passes the gate, time;
                and the linear routes' measurement (routes.py)
hk/tuned/       the schedules that search picked, and the measured routing
                tables, as shipped data
hk/ops/         kernels written in the DSL, the SDPA drop-in (sdpa.py), and
                HipKittens' CDNA4 kernels wrapped as hk ops (hipkittens.py)
hk/integration/ patching a running vLLM or SGLang; engine.py is what every
                patch shares (op, routes, counters, CUDA-graph contract)
mlir/           the hk dialect (ODS), its passes, the C++ translation,
                hk-opt / hk-translate, lit tests
include/        hk's own headers: pyutils/hk_bind.cuh (the fast pybind entry)
                and hk/cdna.cuh (what CDNA needs beside HipKittens)
third_party/HipKittens/
                the backend: kittens.cuh and include/rdna3, rdna4, cdna3,
                cdna4; also the C++ kernels hk is measured against
scripts/        building MLIR, the dialect and the wheels; the GPU suite, the
                performance gate, and the GPU CI jobs that run them (ci/)
tools/hk-bench/ the benchmarks, the scoreboard, the performance gate and the
                engine harnesses behind every number here
tools/cdna-probes/
                the CDNA probes: coverage, the bind hazard, occupancy timing
tests/perf/baselines/
                the performance gate's measured ratios and tolerances
examples/       complete kernels, each checked against torch and timed
docs/           these pages, and the development record (docs/history)
```

The `hk/` paths are under `python/`. `tests/hk/` is the test suite
([testing](testing.md)).
