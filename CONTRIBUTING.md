# Contributing to hk-dsl

Issues and pull requests are welcome. These pages cover the mechanics:
- [testing](docs/testing.md): the tiers, and the gates for a change to the generated code;
- [how numbers are taken](docs/performance.md#how-the-numbers-are-taken);
- [releasing](docs/releasing.md).

A change to a public signature or docstring regenerates the API reference
(`python3 scripts/gen_api_docs.py`). The IR tier fails if it is stale.

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
