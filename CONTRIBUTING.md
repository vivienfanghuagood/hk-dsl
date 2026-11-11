# Contributing to hk-dsl

## Your first contribution

Most changes need no GPU: the language, the passes, the generated C++, the docs and their tests all run in
the IR tier, on any x86-64 Linux machine. On Ubuntu 22.04 with its Python 3.10, CI's prebuilt MLIR saves the
~20-minute LLVM build:

```bash
sudo apt-get install -y git cmake ninja-build g++ python3-pip python3-dev zstd curl
git clone --recursive https://github.com/vivienfanghuagood/hk-dsl.git && cd hk-dsl
sudo mkdir -p /opt/hk-llvm && curl -sSL \
  https://github.com/vivienfanghuagood/hk-dsl/releases/download/ci-mlir-23.1.3/mlir-23.1.3-install-cp310-ubuntu22.04-x86_64.tar.zst \
  | sudo tar -C /opt/hk-llvm -I zstd -xf -                       # 889 MB
python3 -m pip install "nanobind==2.15.0" numpy pybind11 typing_extensions pytest pytest-xdist
HK_LLVM_ROOT=/opt/hk-llvm PYTHON=/usr/bin/python3 scripts/build_hk_mlir.sh       # the dialect, ~1 min
PYTHONPATH=python python3 -m pytest -q -m "not gpu" -n auto tests/hk/ir           # the IR tier, ~1 min
```

In a fresh `ubuntu:22.04` container this took 17 minutes end to end, most of it the download; the dialect built
in 42 s and the IR tier ran 2,037 tests in 32 s. Another Python or distribution builds MLIR itself first
(`scripts/build_mlir.sh`, [building](docs/building.md#mlir-and-the-dialect)).

Then:
- **A docs change** runs the same tier, which executes or traces the code blocks of the language, quick-start and
  template pages. A change to a page that feeds `llms-full.txt` (the language, `hk.check`, the API pages) regenerates it with
  `python3 scripts/gen_api_docs.py`.
- **A kernel or route change** also needs the codegen tier (hipcc, no GPU) and, before it merges, the GPU tier.
  You can run it on your own AMD GPU ([testing](docs/testing.md#running-each-tier)). Pull requests run it on
  the maintainers' W7900D, R9700 and MI355X runners.
- **A performance result** for an issue or a PR measures both arms in one process, alternating, for several rounds.
  It reports the median and the range ([how numbers are taken](docs/performance/method.md)). A single run, or two
  runs in separate processes, cannot tell a 5% change from noise on these GPUs.
- **Questions:** in an issue. Include the `[hk]` line from the engine's log and the output of
  `python3 -m hk.integration --status` when it is about an engine.

## The rules

Issues and pull requests are welcome. These pages cover the mechanics:
- [testing](docs/testing.md): the tiers, and the gates for a change to the generated code;
- [the default-on rule](docs/testing.md#the-default-on-rule): what an engine route needs before it
  is on by default (a 1.15x time-weighted win at a real model's shapes, only on shapes where hk is
  faster; the correctness suite; one end-to-end smoke per engine);
- [how numbers are taken](docs/performance/method.md);
- [releasing](docs/releasing.md).

A change to a public signature or docstring regenerates the API reference
(`python3 scripts/gen_api_docs.py`). The IR tier fails if it is stale.

**A PR that adds an op or an engine route carries its op-catalog row.** The
catalog (`python/hk/ops/catalog.py`, `FAMILIES`) is written by hand. Each
family entry gives the op's entry points, GPUs, dtypes, routes and GPU tests.
`docs/ops.md` is generated from it. A new public `hk.ops` name, or a new route
in `hk.integration.switches`, goes into a family in the same PR:
- extend an existing family, or add a new one;
- run `python3 scripts/gen_hardware_docs.py`.

`tests/hk/ir/test_ops_catalog.py` fails on a name or route that is in no
family. The merge train regenerates generated files such as `docs/ops.md` on
its stacked tree, but it cannot write a catalog row.

`python3 scripts/check_public_hygiene.py` keeps internal host names, addresses and user paths out
of the repository.

## New kernels: no kernel-specific helpers, no one-off DSL ops

hk-dsl is the DSL and the kernels written in it. Each hard kernel used to bring its own C++
helper behind a DSL name, and the language grew one bespoke op per kernel. The design that
converges that surface, and the registry of what exists, is
[docs/design/dsl-surface.md](docs/design/dsl-surface.md).

The rule for a new kernel, or a new version of one:

1. **Write it in the existing abstractions.** Do not add a kernel-specific C++ helper to
   `include/hk/`, and do not add a DSL op, method or keyword that only this kernel needs. Look
   first at the registry's primitives:
   - strips and their integer and bit expressions;
   - register views and lane coordinates;
   - row index sources, element-indexed access, lane exchanges;
   - `hk.mma`.

   If something is missing from them, the fix is usually a general op, written for every kernel
   that could use it, tested on its own and documented in `docs/api`. It is not a helper shaped
   like this kernel.
2. **If a kernel-specific op is truly needed, it goes through review first**, before the kernel PR.
   - Open an issue or a design PR that explains, in writing, why the existing abstractions
     cannot express it.
   - Say what was tried and what it cost: the instruction count, the measured time, or the
     result that came out wrong.
   - Name the general op that would replace it, and say why that op is not being built now.
3. **It is registered as an escape hatch with a usage boundary.** The entry goes in the surface
   registry (`python/hk/dsl/surface.py`, `hk.dsl.surface`) and states:
   - its kind;
   - when it may be used and when not;
   - the tests that pin it;
   - the general op that would retire it.

   Each file in `hk.ops`, the templates and the examples that uses it has a line in the
   escape-hatch allowlist (`scripts/escape_hatches_allow.txt`) with a one-line reason and a
   ceiling: the most uses that file may have. CI fails on a use in an unlisted file or above the
   ceiling, so adding a line or raising a ceiling is part of the PR and gets reviewed. Fewer uses
   only print a hint: tighten the ceiling when you see one.
4. **Never edit the generated C++.** A `Kernel` subclass may not override `with_defines` or
   `source` to splice or rewrite code, and `with_defines` carries `#define NAME value` knobs only.
   Every past case of rewriting was a missing op: ask for the op. One exception predates the rule:
   `attn_k3.py`'s `PairPKernel` splices the RDNA3 Pᵀ exchange into its source. It is listed in the
   escape-hatch allowlist (the class, and its two `with_defines` uses) until S11 replaces it with
   an op, and the allowlist's ceilings fail CI on any new override.

A type-selected specialization behind a general op is allowed without an escape-hatch entry, under
one condition: it must produce bitwise the generic path's results. Examples are a wider store for
one fragment shape, or a permlane instead of an LDS shuffle. It is registered as *internal* with a
test that compares it to the generic path.

## Reviewer checklist

For every PR that touches `include/hk/`, `mlir/include/hk/Dialect/HK/IR/HKOps.td`,
`python/hk/dsl/` or `python/hk/ops/`:

- [ ] **No new kernel-specific helper or op slipped in.** Every new C++ function in
      `include/hk/`, dialect op or public DSL name is either general (usable by any kernel,
      tested on its own, documented in `docs/api`) or a registered escape hatch whose written
      justification was reviewed before this PR.
- [ ] **Every escape-hatch use is inside its boundary, and its file's allowlist line covers
      it.** This covers `hk.map_cpp`, `granules`, `with_defines`, `build_flags` and
      `hk.arch.agpr`. A new line or a raised ceiling in `scripts/escape_hatches_allow.txt` has a
      reason that holds.
- [ ] **No `Kernel` subclass rewrites generated C++**, and `with_defines` carries only `#define`
      knobs.
- [ ] **A new specialization behind a general op is chosen by type alone** and has a test
      showing it bitwise equal to the generic path.
- [ ] **A change to existing generated code states its gate** (A, B or C in
      [testing](docs/testing.md#the-gates-for-a-change-to-the-generated-code)), and the evidence
      is in the PR.
- [ ] **A public name that changes keeps a deprecation shim**, and the API reference is
      regenerated.
