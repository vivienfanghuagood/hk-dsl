"""Write docs/support-matrix.md: what hk is tested with, per arch, from what CI runs.

Two tables, both read from the repository, never typed:
- **hk's own tests**: the image (ROCm, Python, torch) each GPU tier runs in, from the CI files
  (scripts/ci/k8s/gfx1100-ci.yaml, scripts/ci/gfx1201/run_in_container, scripts/ci/run_gfx950.sh,
  .github/workflows/gpu.yml), and the no-GPU tiers' toolchain (.github/workflows/ci.yml);
- **the serving engines**: every image recorded in tests/hk/ir/engine_facts/ (engine version, torch,
  HIP, Triton, Python), and for each route that is on by default there, whether its contract holds
  in it. The IR tier judges those same facts (tests/hk/ir/test_engine_facts.py).

    python3 scripts/gen_support_matrix.py            # write the page
    python3 scripts/gen_support_matrix.py --check    # exit 1 if it is stale (the IR tier runs this)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
OUT = REPO / "docs" / "support-matrix.md"
FACTS = REPO / "tests" / "hk" / "ir" / "engine_facts"
ARCHS = ("gfx1100", "gfx1201", "gfx950")
GPU = {"gfx1100": "Radeon PRO W7900", "gfx1201": "Radeon AI PRO R9700", "gfx950": "Instinct MI355X"}
ROLE = {"production": "an image hk serves in",
        "measured": "the engine version hk's routes were measured on, end to end",
        "development": "the newest image on this arch; no production image yet"}


def _find(path: str, pattern: str) -> str:
    """The first group of `pattern` in `path`; a CI file that no longer says it is an error."""
    m = re.search(pattern, (REPO / path).read_text(encoding="utf-8"), re.M)
    if m is None:
        raise SystemExit(f"gen_support_matrix: no match for {pattern!r} in {path}: update the reader")
    return m.group(1)


def _pytorch_image(image: str) -> dict:
    """rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 -> its versions."""
    m = re.search(r"rocm([\d.]+)_ubuntu([\d.]+)_py([\d.]+)_pytorch_release_([\d.]+)", image)
    if m is None:
        raise SystemExit(f"gen_support_matrix: not a rocm/pytorch image tag: {image}")
    return {"rocm": m.group(1), "os": f"Ubuntu {m.group(2)}", "python": m.group(3),
            "torch": m.group(4)}


def ci_tiers() -> list:
    """(tier, arch, image, versions) for each GPU tier, as CI's files configure it."""
    rows = []
    for tier, arch, path, pat in (
            ("gfx1100", "gfx1100", "scripts/ci/k8s/gfx1100-ci.yaml", r"^\s*image:\s*(\S+)"),
            ("gfx1201", "gfx1201", "scripts/ci/gfx1201/run_in_container", r"^IMAGE=\$\{HK_IMAGE:-([^}]+)\}"),
            ("gfx950", "gfx950", "scripts/ci/run_gfx950.sh", r"^IMAGE=\$\{HK_CI_IMAGE:-([^}]+)\}")):
        image = _find(path, pat)
        rows.append((tier, arch, image.split("/", 1)[-1] if image.startswith("docker.io/")
                     else image, _pytorch_image(image)))
    return rows


def ci_vllm_tier() -> str:
    """The gfx950-vllm tier's image (gpu.yml)."""
    return _find(".github/workflows/gpu.yml", r"^\s*HK_CI_IMAGE:\s*(\S+)")


def no_gpu_tiers() -> dict:
    ci = ".github/workflows/ci.yml"
    return {"os": _find(ci, r"^\s*runs-on:\s*ubuntu-([\d.]+)"),
            "rocm": _find(ci, r'^\s*ROCM_VERSION:\s*"?([\d.]+)"?'),
            "python": "3." + _find(ci, r"^\s*MLIR_ASSET:.*-cp3(\d+)-")}


def fixtures() -> list:
    out = [json.loads(p.read_text(encoding="utf-8")) | {"_name": p.stem}
           for p in sorted(FACTS.glob("*.json"))]
    order = {a: i for i, a in enumerate(ARCHS)}
    roles = {r: i for i, r in enumerate(ROLE)}
    return sorted(out, key=lambda f: (order[f["deployed_on"]], roles[f["role"]], f["_name"]))


def _rocm(stack: dict) -> str:
    """The ROCm the torch build names (`+rocm7.2`), else the HIP version's major.minor."""
    m = re.search(r"rocm(\d+\.\d+)", stack.get("torch", ""))
    if m:
        return m.group(1)
    hip = stack.get("hip") or ""
    return ".".join(hip.split(".")[:2]) or "?"


def _triton(stack: dict) -> str:
    for d in ("triton-rocm", "pytorch-triton-rocm", "triton"):
        if d in stack:
            return f"{d} {stack[d]}"
    return "--"


def _served(fx: dict) -> list:
    """The engines a fixture is recorded for: `serves`, else every installed one."""
    names = fx.get("serves") or list(fx["engines"])
    return [e for e in names if not fx["engines"][e].endswith("(version unknown)")]


def engine_rows() -> list:
    from hk.integration import contract, switches

    module = {"vllm": "vllm", "sglang": "sglang", "sglang_diffusion": "sglang"}
    rows = []
    for fx in fixtures():
        served = _served(fx)
        routes = [r for r in switches.ROUTES.values()
                  if r.default and fx["deployed_on"] in r.arches and module.get(r.engine) in served
                  and r.name in contract.CHECKS]
        cells = []
        for r in routes:
            problems = contract.judge(r.name, fx["facts"][r.name])
            cells.append(f"`{r.name}`" if not problems else f"~~`{r.name}`~~ (off: "
                         f"{fx['expected_refusals'].get(r.name) or problems[0]})")
        engines = ", ".join(fx["engines"][e] for e in served)
        if "sglang" in served and "sglang-radeon-rdna3" in fx["stack"]:
            engines += f" with sglang-radeon-rdna3 {fx['stack']['sglang-radeon-rdna3']}"
        rows.append({"fx": fx, "engines": engines,
                     "routes": cells})
    return rows


def render() -> str:
    import hk

    L = []
    w = L.append
    w("# Support matrix")
    w("")
    w("<!-- Generated by scripts/gen_support_matrix.py from CI's files and "
      "tests/hk/ir/engine_facts/. Do not edit; the IR tier fails when it is stale. -->")
    w("")
    w(f"What this checkout of hk-dsl (version {hk.__version__}, and main's changes after it) is "
      "tested with. A version that is not here may work, but nothing tests it. For the 0.8.0 wheel "
      "on PyPI, see this page in the 0.8.0 tag.")
    w("")
    w("## hk's own tests")
    w("")
    w("Every pull request runs the no-GPU tiers, and the GPU tier on each arch before it merges "
      "([testing](testing.md)).")
    w("")
    w("| arch | GPU | image | ROCm | Python | torch |")
    w("|---|---|---|---|---|---|")
    for tier, arch, image, v in ci_tiers():
        w(f"| {arch} | {GPU[arch]} | `{image}` | {v['rocm']} | {v['python']} | {v['torch']} |")
    n = no_gpu_tiers()
    w(f"| none | -- | Ubuntu {n['os']}, hipcc from ROCm {n['rocm']} (the IR and codegen tiers, "
      f"the wheel) | {n['rocm']} | {n['python']} | -- |")
    w("")
    w(f"gfx950 has a second GPU tier, gfx950-vllm: the gfx950 routes' tests and their engine "
      f"contract, in the vLLM image they were measured on (`{ci_vllm_tier()}`, below).")
    w("")
    w("`pip install hk-dsl` needs Python 3.10 or newer; torch is optional, and needed only to "
      "call kernels with torch tensors or to use [`hk.torch_op`](runtime.md#torch-ops-and-the-sdpa-drop-in).")
    w("")
    w("## Serving engines")
    w("")
    w("Each row is an image whose engines' contract facts are recorded in "
      "`tests/hk/ir/engine_facts/`. The IR tier holds every route that is on by default on "
      "that arch to them ([integration](integration.md#supported-versions-and-the-contract-tests)). "
      "A route whose contract an image breaks is off there, with a warning, and the others stay on; "
      "it is shown struck through, with the reason.")
    w("")
    for arch in ARCHS:
        rows = [r for r in engine_rows() if r["fx"]["deployed_on"] == arch]
        if not rows:
            continue
        w(f"### {arch} ({GPU[arch]})")
        w("")
        w("| image | role | engine | ROCm | torch | Triton | Python | routes on by default |")
        w("|---|---|---|---|---|---|---|---|")
        for r in rows:
            fx, s = r["fx"], r["fx"]["stack"]
            w(f"| {fx['label']} | {fx['role']} | {r['engines']} | {_rocm(s)} | "
              f"{s.get('torch', '--')} | {_triton(s)} | {s.get('python', '--')} | "
              f"{', '.join(r['routes']) or '--'} |")
        w("")
    w("Roles: " + "; ".join(f"**{k}**, {v}" for k, v in ROLE.items()) + ".")
    w("")
    w("To add an image, record it inside the image and commit the file "
      "([testing](testing.md)), then run `python3 scripts/gen_support_matrix.py`.")
    w("")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = render()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.relative_to(REPO)} is stale: run python3 scripts/gen_support_matrix.py")
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
