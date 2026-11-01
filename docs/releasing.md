# Releasing

How a version of hk-dsl goes out: the release PR, the documentation sync
that must pass before the tag, the wheels, the GitHub release and PyPI. The
wheels themselves (what they carry, how they are tagged manylinux) are in
[building](building.md#wheels).

- [The steps](#the-steps)
- [Documentation sync (must pass before tagging)](#documentation-sync-must-pass-before-tagging)
- [After the release](#after-the-release)

## The steps

1. **The release PR** (`release: X.Y.Z`, as #37 for 0.6.0 and #58 for 0.7.0):
   - `version` in `pyproject.toml` and `hk.__version__` (`python/hk/__init__.py`);
   - the dialect's semantic version (`mlirHKDialectVersion` in
     `mlir/lib/CAPI/HK.cpp`, part of the compile-cache key) if the dialect
     changed since the last release;
   - `CHANGELOG.md`: the `Unreleased` section becomes `## X.Y.Z`, with its
     install line, compatibility notes and known issues;
   - `scripts/gen_api_docs.py`: `RELEASE = "X.Y.Z"`, so the API reference says
     which release has the names it marks *New after* the last one;
   - the version and install facts on every page that states them (the
     [checklist](#documentation-sync-must-pass-before-tagging) lists them).

   The PR is tested on a head that contains the latest main (CI, the docs
   site, and both GPU tiers), and rc wheels are built from that head.
2. **The documentation sync gate**, [below](#documentation-sync-must-pass-before-tagging),
   on the commit to be tagged. **If any item fails, do not tag.**
3. **Tag** the release PR's merge commit (annotated, `vX.Y.Z`), after
   checking that main has not moved past it, and push only the tag.
4. **Build from the tag**, in a fresh worktree with the submodule: the cp310
   and cp312 wheels, then their manylinux copies (`scripts/manylinux_tag.sh`).
   Both are built on glibc 2.35 (cp312 in vLLM's ROCm image), so each is one
   `manylinux_2_35` wheel. Each wheel's glibc must be within its tag:
   `scripts/check_wheel_glibc.py dist/*.whl dist/manylinux/*.whl` (#66)
   fails a wheel whose highest `GLIBC_2.N` is above its `manylinux_2_M` tag.
   The build is reproducible ([building](building.md#reproducible-wheels)),
   so the tag's wheels are byte-identical to the rc's.
5. **Compare with the rc wheels.** The sha256 of each wheel: with the
   reproducible build they are equal. Where one differs, compare the sorted
   file lists and each file's sha256: a difference in a `.py`, `.so`, header
   or data file stops the release.
6. **Smoke test** each manylinux wheel in a clean venv, on each glibc it is
   tagged for (rocm/pytorch's Ubuntu 24.04, and vLLM's ROCm image on Ubuntu
   22.04): `hk.__version__`, a kernel against torch, and
   `python3 -m hk.integration --status` (the default-on routes on, all off
   with `HK_DISABLE=1`). `twine check` every wheel (it needs no credentials).
7. **GitHub release** `vX.Y.Z`: the notes are the CHANGELOG section at the
   tag under an install header that matches the wheels; the assets are the
   wheels, their manylinux copies and `SHA256SUMS`.
8. **PyPI.** The project owner uploads the manylinux wheels with twine,
   after checking their sha256 against the GitHub release
   (`.github/workflows/pypi.yml` cannot upload until PyPI's trusted
   publisher is configured, [building](building.md#wheels)).
9. **Verify:** pypi.org lists X.Y.Z with exactly the wheels built, its
   description's links open, and `pip install hk-dsl==X.Y.Z` in a clean venv
   picks the expected wheel on each glibc, imports and runs `--status`.

## Documentation sync (must pass before tagging)

Every item is checked on the commit to be tagged. **If any item fails, do
not tag**: fix it in the release PR (or a docs PR merged before it), and
check again.

- [ ] **Every user-visible feature since the last release has a page.** List
  them with `git log vPREV..HEAD --oneline` and the PR descriptions; each
  feature, fix and behaviour change has a page or a section of one (the
  language, the runtime, integration, check, testing, a design page).
- [ ] **It is reachable.** The page is in `mkdocs.yml`'s nav, and a headline
  feature is also in `README.md`. `mkdocs build --strict` passes with no
  warning: every page in the nav, every link and anchor resolves.
- [ ] **The API docs are regenerated and current.**
  `python3 scripts/gen_api_docs.py --check` passes (it needs `hk._mlir`; the
  IR tier runs it), `RELEASE` is X.Y.Z, and the baseline
  (`scripts/api_baseline_<PREV>.json`) is the previous release's wheel, so
  the *New after* marks are exactly what X.Y.Z adds.
- [ ] **The CHANGELOG has an entry per feature and per fix**, with its PR
  number, plus the compatibility notes and known issues; nothing is left
  under `Unreleased`.
- [ ] **The install matrix matches the wheels being built**: the Python
  versions × manylinux tags × minimum glibc stated in README's install
  section, [install](install.md), the CHANGELOG's
  install line, [building](building.md#wheels) and the release notes'
  install header are the wheels in `dist/manylinux/`, and each wheel passes
  `scripts/check_wheel_glibc.py` (its glibc within its tag). The ROCm / hipcc versions the install pages
  name are the ones CI and the GPU tiers run.
- [ ] **The version strings agree**: `pyproject.toml`, `hk.__version__`, the
  CHANGELOG heading, README, getting started, `docs/README.md`,
  `docs/api/README.md`, `llms.txt`, `llms-full.txt` and
  `skills/hk-kernel/SKILL.md`
  (`grep -rnE "0\.[0-9]+\.[0-9]+" README.md docs/*.md llms.txt skills/`).
- [ ] **The PyPI description is correct.** It is README.md with its links made
  absolute at the tag (`setup.py`'s `pypi_readme`,
  `tests/hk/ir/test_pypi_readme.py`), and it is frozen at upload: a wrong
  install fact on it stays wrong until the next release. Read README as
  PyPI will show it, and `twine check` the rc wheels.
- [ ] **The examples run**: `tests/hk/ir/test_examples.py`,
  `tests/hk/codegen/test_example_builds.py` and the GPU tier's
  `tests/hk/gpu/test_example_runs.py` pass on the release head, on gfx950
  and gfx1100; a new feature that has an example is in
  [`examples/README.md`](../examples/README.md).
- [ ] **Grep for internal IPs, host names and user paths finds nothing
  outside the allowlist.** The repository is public:
  `python3 scripts/check_public_hygiene.py --patterns <site-patterns>`
  passes, where `<site-patterns>` is the maintainers' local file of the
  site's own names (hosts, pods, nodes, proxies, logins, workspace paths),
  which is never committed. Without it the script checks only the generic
  rules (private IPv4, `/home/<user>`, `/root/`, `/dev/shm/<name>`,
  internal domains, PCI addresses, `/workspace/<name>`, `~/<dir>`, a
  container's hex hostname as a `"host"`), as the IR tier does on every PR,
  with the site's file too when the repository has it as the secret
  `HK_HYGIENE_PATTERNS`. A hit is replaced by a
  placeholder or an environment variable, or, if it is generic, added to
  `scripts/public_hygiene_allow.txt` with the reason. Anything sensitive
  that ever reached main is still in git history: rotate it, do not just
  remove it.

## After the release

Once the release is on PyPI, a PR on main:
- records the API baseline from the released wheel
  (`pip download hk-dsl==X.Y.Z --no-deps`, unzip,
  `python3 scripts/gen_api_docs.py --baseline DIR`), sets `PYPI = "X.Y.Z"`
  and `RELEASE = None` in `scripts/gen_api_docs.py` and regenerates, so that
  what lands next is marked *New after X.Y.Z*;
- opens an `## Unreleased` section at the top of `CHANGELOG.md`, where each
  PR adds its entry as it merges.

`pyproject.toml` and `hk.__version__` keep saying X.Y.Z until the next
release PR raises them ([building](building.md#wheels)).
