#!/bin/bash
# Inside rocm/pytorch (local_run.sh): build hk._mlir for this image's CPython
# against the cp310 MLIR install, then run the GPU tier for gfx950 on the one
# GPU the container sees. The MLIR C++ libraries do not depend on the
# interpreter; the core Python extension is compiled here from MLIR's
# installed sources.
set -uo pipefail
SHA=$1
GPU=$2
DEVICE="MI355X (gfx950), GPU $GPU"
export HOME=/ci/home PYTHONPATH=/ci/pylib PATH=/ci/pylib/bin:$PATH
mkdir -p "$HOME"

fail() {
    echo "error: $*"
    python3 /ci/summarize.py --error "$*" --sha "$SHA" --tier gfx950 --device "$DEVICE" --out /ci/run
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
HK_GPU=0 scripts/gpu_suite.sh /ci/run/gpu
python3 /ci/summarize.py /ci/run/gpu --sha "$SHA" --tier gfx950 --device "$DEVICE" --out /ci/run
