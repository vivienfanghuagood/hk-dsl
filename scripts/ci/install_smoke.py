#!/usr/bin/env python3
"""install_smoke.py WHEEL README [DEPS]  (HK_INSTALL_ARCH=gfx1100|gfx1201|gfx950, HK_INSTALL_TARGET=DIR): what a new user does, in a fresh container of a stock ROCm image.

The install-matrix job (.github/workflows/install.yml) runs this once per (image, GPU arch), in a
container that has nothing of hk but the wheel and this script:
1. `pip install WHEEL` with the image's own Python and pip, as root, as the README's quick start says
   (the container is --rm and its mounts read-only): wheels the image lacks (pybind11) come from DEPS,
   nothing from an index -- a CI container may have no PyPI. A venv is not used: rocm/pytorch 7.2's
   Python is itself a venv (a venv over it would not see its torch), and 10.1's has no venv module.
   Then `import hk` from a scratch directory: it must come from site-packages / dist-packages, not a
   checkout (a repository conftest would put python/ first and hide a broken wheel) or the mount.
   HK_INSTALL_TARGET=DIR (a shared host whose Python must not change: the gfx1100 CI pod) installs
   with `pip install --target DIR` instead, and every later step sees DIR alone on PYTHONPATH;
2. the README's quick start, as written: its `my_kernels.py` block and its `python3 -m hk.check`
   command, both read from README (so an edit that breaks the quick start fails here);
3. one GPU test with an exact answer: attention with V all ones returns exactly 1.0.
The JIT compile in steps 2 and 3 uses the image's own hipcc: a toolchain hk does not handle (the
clang 24 remark format of ROCm 10.1) fails here, not on a user's machine. Exit 0 only if all
three pass; each step prints one PASS / FAIL line. A container without the expected GPU fails at
once: hk.check without a GPU only traces and builds ("[build-only]") and exits 0, which would make
steps 2 and 3 pass vacuously.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile


def step(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        sys.exit(1)


def quick_start(readme: str):
    """(my_kernels.py source, the hk.check command) from README's "## Quick start" section."""
    text = open(readme, encoding="utf-8").read()
    m = re.search(r"^## Quick start\n(.*?)(?=^## )", text, re.S | re.M)
    if not m:
        raise SystemExit("README has no '## Quick start' section")
    blocks = re.findall(r"^```(\w+)\n(.*?)^```", m.group(1), re.S | re.M)
    code = next((b for lang, b in blocks if lang == "python" and "my_kernels.py" in b), None)
    cmd = next((b for lang, b in blocks if lang == "bash" and "hk.check" in b), None)
    if code is None or cmd is None:
        raise SystemExit("README's quick start lacks the my_kernels.py block or the hk.check command")
    return code, cmd.replace("\\\n", " ").strip()


def main(argv):
    wheel, readme = argv[1], argv[2]
    deps = argv[3] if len(argv) > 3 else None
    work = tempfile.mkdtemp(prefix="hk-install-")
    os.chdir(work)
    py = sys.executable
    target = os.environ.get("HK_INSTALL_TARGET")
    os.environ.pop("PYTHONPATH", None)
    os.environ["HK_CACHE_DIR"] = os.path.join(work, "hk-cache")    # a new user's: empty
    print(f"image Python {sys.version.split()[0]} at {py}", flush=True)
    pip = [py, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-index"]
    extra = []
    if target:   # --target resolves every dependency anew, ignoring the image's: take the image's numpy and
        # typing_extensions as they are, and DEPS' wheels (the root-install legs check the wheel's own requirements)
        pip += ["--target", target, "--no-deps"]
        extra = sorted(os.path.join(deps, f) for f in os.listdir(deps) if f.endswith(".whl")) if deps else []
    if deps:
        pip += ["--find-links", deps]
    r = subprocess.run(pip + [wheel] + extra, capture_output=True, text=True)
    step("pip install", r.returncode == 0, (r.stdout + r.stderr).strip()[-800:] if r.returncode else
         os.path.basename(wheel) + (f" (--target {target})" if target else ""))
    if target:
        os.environ["PYTHONPATH"] = target
    r = subprocess.run([py, "-c", "import hk, sys; print(hk.__file__); print(hk.__version__)"],
                       capture_output=True, text=True, cwd=work)
    where = r.stdout.split()
    installed = (where[0].startswith(os.path.abspath(target)) if target else
                 any(d in where[0] for d in ("/site-packages/", "/dist-packages/"))) if where else False
    step("import hk from the installed wheel", r.returncode == 0 and installed and not where[0].startswith(
             os.path.dirname(os.path.abspath(wheel))),
         " ".join(where) if r.returncode == 0 else r.stderr.strip()[-800:])

    arch = os.environ.get("HK_INSTALL_ARCH", "")
    r = subprocess.run([py, "-c", "import torch; p = torch.cuda.get_device_properties(0); "
                        "print(p.gcnArchName.split(':')[0], p.name, torch.__version__, torch.version.hip)"],
                       capture_output=True, text=True, cwd=work)
    seen = r.stdout.strip()
    step(f"GPU visible ({arch or 'any'})", r.returncode == 0 and (not arch or seen.split()[:1] == [arch]),
         seen or r.stderr.strip()[-400:])

    code, cmd = quick_start(readme)
    with open(os.path.join(work, "my_kernels.py"), "w") as f:
        f.write(code)
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=work)
    out = (r.stdout + r.stderr).strip()
    print(out[-3000:], flush=True)
    ran = "[build-only]" not in out and (not arch or f" {arch} " in out.splitlines()[0] if out else False)
    step("README quick start (hk.check my_kernels.py:rms_norm)", r.returncode == 0 and ran,
         f"rc {r.returncode}" if r.returncode else ("" if ran else "it did not run on the GPU (build-only)"))

    probe = (
        "import torch, hk\n"
        "torch.manual_seed(0)\n"
        "q = torch.randn(1, 4, 4096, 128, device='cuda', dtype=torch.bfloat16)\n"
        "k = torch.randn_like(q)\n"
        "got = hk.attention(q, k, torch.ones_like(q))\n"
        "dev = (got.float() - 1.0).abs().max().item()\n"
        "print('max deviation from 1.0:', dev)\n"
        "raise SystemExit(0 if torch.equal(got, torch.ones_like(got)) else 1)\n")
    r = subprocess.run([py, "-c", probe], capture_output=True, text=True, cwd=work)
    print((r.stdout + r.stderr).strip()[-2000:], flush=True)
    step("GPU test: attention with V = 1 is exactly 1.0", r.returncode == 0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
