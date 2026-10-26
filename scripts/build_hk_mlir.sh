#!/bin/bash
# Build the hk dialect against the MLIR scripts/build_mlir.sh installed, and
# install the hk._mlir Python package into python/hk/_mlir.
#
#   scripts/build_hk_mlir.sh
#   HK_LLVM_ROOT=/workspace/llvm PYTHON=/opt/venv/bin/python3 scripts/build_hk_mlir.sh
#   scripts/build_hk_mlir.sh -DCMAKE_CXX_FLAGS=-g     # arguments go to the cmake configure
#
# Then `ninja -C $BUILD check-hk` runs the lit tests. The binaries name no
# build-machine path (mlir/CMakeLists.txt maps the checkout, $BUILD, the MLIR
# install and nanobind to /hk/...), so where they live does not change a byte;
# -DHK_REPRODUCIBLE_PATHS=OFF keeps the real paths. docs/building.md has the
# reproducible release build.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
PYTAG="$("$PYTHON" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
ROOT="${HK_LLVM_ROOT:-/dev/shm/hk-llvm}"
MLIR_INSTALL="${LLVM_INSTALL:-$ROOT/install-$PYTAG}"
BUILD="${HK_MLIR_BUILD:-$ROOT/hk-build-$PYTAG}"
JOBS="${JOBS:-$(nproc)}"
LIT="${LIT:-$(command -v lit || true)}"

cmake -G Ninja -S "$REPO/mlir" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR="$MLIR_INSTALL/lib/cmake/mlir" \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DLLVM_EXTERNAL_LIT="$LIT" \
    -DCMAKE_INSTALL_PREFIX="$REPO/python" \
    "$@"

cmake --build "$BUILD" -j "$JOBS"
cmake --install "$BUILD" --component HKPythonModules > /dev/null
echo "hk._mlir for $PYTAG installed at $REPO/python/hk/_mlir (build: $BUILD)"
