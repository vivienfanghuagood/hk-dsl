#!/bin/bash
# Inside rocm/pytorch (run_gfx950.sh) -- or the vLLM deployment image, as tier
# gfx950-vllm -- build hk._mlir for this image's CPython against the cp310 MLIR
# install, then run the GPU tier for gfx950 on the one GPU the container sees:
# all of it (scripts/gpu_suite.sh), or the test files in HK_CI_TESTS. The MLIR C++ libraries do not depend on the
# interpreter; the core Python extension is compiled here from MLIR's
# installed sources.
#   /ci        CI's own directory: hk-build-cp312, pylib, hk-cache, home
#   /ci/src    the checkout          /ci/run   the run's directory
#   /ci/mlir   the MLIR install, read-only
set -uo pipefail
# No core dumps: a faulting or killed GPU process writes host cores of its whole address space
# (100+ GB for vLLM) and ROCr GPU cores (gpucore.PID in the working directory) to a shared disk.
# ROCr honours RLIMIT_CORE and also HSA_DISABLE_COREDUMP_ON_EXCEPTION (ROCm 7.2.3 and 10).
ulimit -c 0
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
SHA=$1
GPU=$2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIER=${HK_CI_TIER:-gfx950}
DEVICE="MI355X (gfx950), GPU $GPU"
export HOME=/ci/home PYTHONPATH=/ci/pylib PATH=/ci/pylib/bin:$PATH
# JIT caches under /ci/home: the vLLM image's aiter puts FlyDSL's in its package directory, which
# the CI user cannot write (PermissionError on the first MoE call)
export FLYDSL_RUNTIME_CACHE_DIR=/ci/home/flydsl-cache TRITON_CACHE_DIR=/ci/home/triton-cache
mkdir -p "$HOME"

fail() {
    echo "error: $*"
    python3 "$HERE/summarize.py" --error "$*" --sha "$SHA" --tier "$TIER" --device "$DEVICE" --out /ci/run
    exit 1
}

[ -d /ci/pylib/nanobind ] || python3 -m pip install -q --target /ci/pylib \
    "nanobind==2.15.0" ninja cmake pytest >/ci/run/pip.log 2>&1 || fail "pip install (pip.log)"

{ cmake -G Ninja -S mlir -B /ci/hk-build-cp312 \
      -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
      -DMLIR_DIR=/ci/mlir/lib/cmake/mlir \
      -DPython3_EXECUTABLE="$(command -v python3)" \
      -DCMAKE_INSTALL_PREFIX=/ci/src/python &&
  cmake --build /ci/hk-build-cp312 -j 32 &&
  cmake --install /ci/hk-build-cp312 --component HKPythonModules; } >/ci/run/build.log 2>&1 \
    || fail "the dialect did not build (build.log)"

export HK_ARCH=gfx950 HK_CACHE_DIR=/ci/hk-cache
if [ -n "${HK_CI_TESTS:-}" ]; then
    # one group, in gpu_suite.sh's output format (a log and a status line)
    mkdir -p /ci/run/gpu
    { echo "gpu=0"; PYTHONPATH="python:$PYTHONPATH" HIP_VISIBLE_DEVICES=0 timeout --signal=KILL 5400 \
          python3 -m pytest -q -p no:cacheprovider $HK_CI_TESTS; } > "/ci/run/gpu/$TIER.log" 2>&1
    rc=$?; [ $rc = 5 ] && rc=0
    printf '%s exit %s\ndone\n' "$TIER" "$rc" > /ci/run/gpu/status
else
    HK_GPU=0 scripts/gpu_suite.sh /ci/run/gpu
fi
python3 "$HERE/summarize.py" /ci/run/gpu --sha "$SHA" --tier "$TIER" --device "$DEVICE" --out /ci/run
