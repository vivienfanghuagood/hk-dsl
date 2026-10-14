#!/bin/bash
# A wheel of hk for one interpreter: the package, the headers the generated
# kernels include at JIT time -- HipKittens' from the third_party/HipKittens
# submodule, and hk's own from include/ -- and hk._mlir built for that Python.
#
#   scripts/build_wheel.sh                  # python3, wheel in dist/
#   PYTHON=/opt/venv/bin/python3 ISOLATION=0 scripts/build_wheel.sh
#
# hk._mlir has to be the one for $PYTHON (scripts/build_hk_mlir.sh with the
# same PYTHON); a wheel carrying another interpreter's extension modules
# installs and then fails at import, so this refuses to build one. ISOLATION=0
# builds with the interpreter's own setuptools (>= 64), for a machine without
# an index to fetch one from.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
ISOLATION="${ISOLATION:-1}"
OUT="${OUT:-$REPO/dist}"
SOABI="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"

libs="$REPO/python/hk/_mlir/_mlir_libs"
if ! ls "$libs"/_mlir."$SOABI".so >/dev/null 2>&1; then
    echo "hk._mlir in $libs is not built for $SOABI:" >&2
    ls "$libs"/_mlir.*.so >&2 2>/dev/null || echo "  (not built at all)" >&2
    echo "run PYTHON=$PYTHON scripts/build_hk_mlir.sh first" >&2
    exit 1
fi

kittens="$REPO/third_party/HipKittens/include"
if [ ! -f "$kittens/kittens.cuh" ]; then
    echo "no HipKittens headers at $kittens: git submodule update --init" >&2
    exit 1
fi

# Staged, so the checkout never grows a copy of the headers inside the package.
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/python"
cp "$REPO/pyproject.toml" "$REPO/setup.py" "$REPO/README.md" "$REPO/LICENSE" \
   "$REPO/NOTICE" "$stage/"
cp -r "$REPO/python/hk" "$stage/python/hk"
cp -r "$kittens" "$stage/python/hk/include"
cp -r "$REPO/include/." "$stage/python/hk/include/"
find "$stage" -name __pycache__ -type d -prune -exec rm -rf {} +
# libHKPythonCAPI.so -> .so.23.1 is a link-time name; the loader asks for the
# versioned file, and a wheel would carry the symlink as a second 34 MB copy.
find "$stage/python/hk/_mlir" -type l -delete

flags=(--no-deps -w "$OUT")
[ "$ISOLATION" = 1 ] || flags+=(--no-build-isolation)
mkdir -p "$OUT"
(cd "$stage" && "$PYTHON" -m pip wheel "${flags[@]}" . >/dev/null)
ls -1 "$OUT"/hk_dsl-*-"$("$PYTHON" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"-*.whl
