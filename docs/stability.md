# API stability

What hk-dsl promises to keep working from one release to the next, and what it
does not. The [API reference](api/README.md) marks each page with its level.

## Stable

A stable name keeps its meaning across minor releases. A breaking change to
one comes only in a minor version bump (0.9 to 0.10 while hk-dsl is below
1.0). The old form keeps working for one release behind a deprecation shim
that warns, as [contributing](../CONTRIBUTING.md) requires, and the CHANGELOG
says what changed and what to write instead.

- **The language**: `hk.kernel`, `hk.func`, `hk.Tensor`, the tile and strip
  types (`hk.RT`, `hk.RS`, `hk.ST`, `hk.SV`), loads and stores, `hk.mma`,
  `hk.range`, `hk.if_` and the dtypes ([`hk`](api/hk.md)), with
  [`hk.math`](api/hk.math.md), [`hk.arch`](api/hk.arch.md) and
  [`hk.lds`](api/hk.lds.md).
- **The templates**: [`hk.templates`](api/hk.templates.md), elementwise,
  rowwise and the GEMM with an epilogue.
- **The checker**: [`hk.check`](api/hk.check.md), its Python API and
  `python3 -m hk.check`.
- **The shipped kernels' documented entry points**: the functions on the
  [`hk.ops`](api/hk.ops.md) page above its "In the submodules" section. A name
  that page does not document, or one starting with `_`, is private.
- **The switches' names**: every `HK_*` variable that turns a route on or off,
  and `HK_DISABLE`. Which routes are **on by default is not part of this
  promise.** Defaults change from release to release under
  [the default-on rule](testing.md#the-default-on-rule), and the release notes
  list every change.

## Experimental

An experimental name may change or go in any release, minor or patch. The
CHANGELOG still says so when it does. These pages carry an
*Experimental* line in the API reference:

- [`hk.torch_op`](api/hk.md#hktorch_op), for 0.9.0. It becomes stable after
  one release of use.
- [`hk.comm`](api/hk.comm.md), [`hk.collective`](api/hk.collective.md),
  [`hk.placement`](api/hk.placement.md) and [`hk.group`](api/hk.group.md).
- [`hk.schedules`](api/hk.schedules.md), [`hk.tune`](api/hk.tune.md) and
  [`hk.profile`](api/hk.profile.md).
- [`hk.target`](api/hk.target.md): the architecture facts are data that grows
  with each GPU.
- The functions of `hk.ops`' submodules that the engine integrations call
  (the "In the submodules" section of [`hk.ops`](api/hk.ops.md)).
- `hk.integration`'s Python API. What the plugins do is decided by the
  switches above.
- `hk.runtime`'s internals and the escape hatches registered in
  `hk.dsl.surface`.

## Not covered

No release promises these, stable or not:
- **The generated code**: the HipKittens C++ a kernel translates to, the ISA
  hipcc makes of it, and anything that reads either.
- **The compile cache**: its layout on disk and its keys. A new release may
  rebuild every entry.
- **Performance**: a kernel may get faster or slower from one release to the
  next. The numbers the project quotes are on
  [the scorecard](performance/scorecard.md), each tied to the release and
  stack it was measured on.

## Platforms

Every release ships `manylinux_2_35` wheels (glibc 2.35 or newer) on Linux
x86_64, on PyPI and as release assets ([building](building.md#wheels)): 0.8.0
and 0.9.0 for CPython 3.10 and 3.12. Wheels for 3.11 and 3.13 are planned for a
0.9.x release; until then, build one from source. The GPUs and how
thoroughly each is tested are the tiers of
[supported hardware](supported-hardware.md).
