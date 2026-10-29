<!-- What changed, why, and the evidence. For a change to the generated code, say which gate
(A, B or C in docs/testing.md) it passed and paste the snapshot diff (tools/surface/snapshot.py). -->

## Reviewer checklist (CONTRIBUTING.md)

- [ ] No kernel-specific C++ helper or one-off DSL op: every new name, dialect op or
      `include/hk/` function is a general primitive, or an escape hatch reviewed before this PR
      (python/hk/dsl/surface.py; tests/hk/ir/test_surface.py enforces the registration).
- [ ] Escape-hatch uses are inside their boundaries and in `scripts/escape_hatches_allow.txt`.
- [ ] No `Kernel` subclass rewrites generated C++; `with_defines` carries `#define` knobs only.
- [ ] A specialization behind a general op is chosen by type alone and tested bitwise against
      the generic path.
- [ ] A change to generated code states its gate and its evidence.
- [ ] A public name that changes keeps a deprecation shim; docs/api is regenerated.
