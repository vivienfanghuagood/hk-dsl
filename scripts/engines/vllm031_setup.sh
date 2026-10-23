#!/bin/bash
# vLLM 0.31.0 from source for gfx1100, on the stack its numbers were measured on
# (docs/history/44-matched-stack.md): Python 3.12, torch 2.13.0+rocm7.2 from
# download.pytorch.org, and the Triton that torch pins, triton-rocm 3.7.1.
#
#   VENV=/workspace/venv-vllm031 WORK=/workspace/vllm031 scripts/engines/vllm031_setup.sh
#
# The first copy of this script (chapter 24's rebuild, /workspace/vllm031_setup.sh
# in the dev pod) installed the right Triton and then lost it: xgrammar, one of
# vLLM's requirements, requires `triton` with no version, triton-rocm is a
# different distribution name, and pip put PyPI's newest `triton` (3.8.0) over
# triton-rocm's files in the same package. Every measurement in that venv ran
# Triton 3.8.0 (chapters 24-38). So this one pins it three ways:
#   - the venv's pip.conf names a constraints file (vllm031_constraints.txt)
#     that allows no `triton` and no `pytorch-triton-rocm` at all, so any later
#     `pip install` that would bring one fails with ResolutionImpossible instead
#     of overwriting triton-rocm;
#   - xgrammar is installed with --no-deps (vllm031_reqs_nodeps.txt), never
#     from vllm031_reqs.txt;
#   - after the requirements and again at the end, check_stack stops the script
#     unless `import triton` is triton-rocm 3.7.1, torch is 2.13.0+rocm7.2, and
#     triton-rocm is the only distribution of the `triton` package.
# hk checks the same at run time: python3 -m hk.integration.stack.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${VENV:-/workspace/venv-vllm031}
WORK=${WORK:-/workspace/vllm031}
HK_WHEEL=${HK_WHEEL:-}            # a hk-dsl wheel to install into the venv, optional
TORCH=2.13.0+rocm7.2
TRITON=3.7.1
# triton_rocm-3.7.1-cp312-cp312-linux_x86_64.whl on the rocm7.2 index (index and file agree)
TRITON_SHA256=b200ade2418b8450e4e3e69c33c804e2cc07d1e36dc9678ac8ac1dd65f9e10e1
ROCM_INDEX=https://download.pytorch.org/whl/rocm7.2
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"
mkdir -p "$WORK"
step() { echo "=== $(date +%T) $*"; }

check_stack() {
    python - "$TORCH" "$TRITON" <<'PY'
import importlib.metadata as md, sys
torch_want, triton_want = sys.argv[1:]
import torch, triton
dists = {}
for d in ("triton", "triton-rocm", "pytorch-triton-rocm"):
    try:
        dists[d] = md.version(d)
    except md.PackageNotFoundError:
        pass
bad = []
if torch.__version__ != torch_want:
    bad.append(f"torch {torch.__version__}, want {torch_want}")
if triton.__version__.split("+")[0] != triton_want:
    bad.append(f"import triton gives {triton.__version__} ({triton.__file__}), want triton-rocm {triton_want}")
if dists != {"triton-rocm": triton_want}:
    bad.append(f"distributions of the triton package: {dists}, want only triton-rocm {triton_want}")
print(f"stack: torch {torch.__version__}, triton {triton.__version__} from {dists}")
if bad:
    sys.exit("STACK MISMATCH: " + "; ".join(bad))
PY
}

step venv "$VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
# The pin, before the first install: every pip run in this venv reads it.
cp "$HERE/vllm031_constraints.txt" "$VENV/constraints-stack.txt"
printf '[install]\nconstraint = %s/constraints-stack.txt\n' "$VENV" > "$VENV/pip.conf"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
pip install -q -U pip
pip config debug | grep -A2 '^site:' || true

step "torch $TORCH, triton-rocm $TRITON"
pip install -q "torch==$TORCH" "torchvision==0.28.0+rocm7.2" "triton-rocm==$TRITON" \
    --index-url "$ROCM_INDEX" --extra-index-url "$PIP_INDEX_URL"
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"

step requirements
pip install -q -r "$HERE/vllm031_reqs.txt"
pip install -q --no-deps -r "$HERE/vllm031_reqs_nodeps.txt"

step "triton: triton-rocm $TRITON and nothing over it"
# A no-op on a fresh venv with the constraints above. On a venv built before
# them it removes the `triton` that went over triton-rocm's files (its
# uninstall deletes files both own) and puts triton-rocm's back, by hash.
for d in triton pytorch-triton-rocm; do
    if pip show -q "$d" 2>/dev/null; then pip uninstall -y "$d"; fi
done
printf 'triton-rocm==%s --hash=sha256:%s\n' "$TRITON" "$TRITON_SHA256" > "$WORK/triton-pin.txt"
pip install -q --no-deps --force-reinstall --require-hashes -r "$WORK/triton-pin.txt" --index-url "$ROCM_INDEX"
check_stack

step sdist
# --no-binary :all: would extend to the build requirements, and pip then
# bootstraps cmake (and torch) from source in an isolated env.
[ -f "$WORK/vllm-0.31.0.tar.gz" ] || \
    pip download -q --no-deps --no-build-isolation --no-binary vllm "vllm==0.31.0" -d "$WORK"
[ -d "$WORK/vllm-0.31.0" ] || tar xzf "$WORK/vllm-0.31.0.tar.gz" -C "$WORK"

step build
# CMake would git-clone all of ROCm/triton for one Python directory (vLLM's
# vendored triton_kernels), which from here stalls; the commit's tarball from
# codeload is 7 MB. Under triton-rocm 3.7.1 vLLM logs that triton_kernels
# cannot be imported (MXFP4 MoE only; nothing hk patches uses them).
TK="$WORK/triton-src/python/triton_kernels/triton_kernels"
if [ ! -d "$TK" ]; then
    curl -sSfL --retry 5 -o "$WORK/triton-669b31a.tar.gz" \
        https://codeload.github.com/ROCm/triton/tar.gz/669b31acc1dd1b3fd93286afd5db67f65d9f7557
    mkdir -p "$WORK/triton-src"
    tar xzf "$WORK/triton-669b31a.tar.gz" -C "$WORK/triton-src" \
        --strip-components=1 --wildcards "*/python/triton_kernels/*"
fi
(cd "$WORK/vllm-0.31.0" && TRITON_KERNELS_SRC_DIR="$TK" PYTORCH_ROCM_ARCH=gfx1100 VLLM_TARGET_DEVICE=rocm \
    MAX_JOBS="${MAX_JOBS:-28}" pip install --no-build-isolation --no-deps -v . > "$WORK/build.log" 2>&1)
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

if [ -n "$HK_WHEEL" ]; then
    step hk
    pip install -q --no-deps "$HK_WHEEL"
    python -c "import hk; print('hk', hk.__version__)"
fi

step "the stack, once more"
check_stack
pip check || true
step done
