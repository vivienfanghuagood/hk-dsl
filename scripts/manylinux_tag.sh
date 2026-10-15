#!/bin/bash
# Retag wheels from scripts/build_wheel.sh with the manylinux tag auditwheel
# says they already satisfy. PyPI refuses linux_x86_64, and nothing has to be
# grafted: the wheels link only libc, libm, libgcc_s and libstdc++, so the tag
# is the whole change and the binaries are not touched.
#
#   PYTHON=/path/with/auditwheel/python3 scripts/manylinux_tag.sh dist/*linux_x86_64.whl
#
# The tag is the glibc of the machine that built the wheel: manylinux_2_35 for
# one built on Ubuntu 22.04, manylinux_2_38 for the cp312 one built in the
# rocm/pytorch Ubuntu 24.04 pod. $PYTHON needs auditwheel and wheel >= 0.42.
set -euo pipefail

PYTHON="${PYTHON:-$(command -v python3)}"
OUT="${OUT:-dist/manylinux}"
mkdir -p "$OUT"

for whl in "$@"; do
    tag="$("$PYTHON" -m auditwheel show "$whl" 2>/dev/null \
        | grep -o -m1 '"manylinux_[0-9]*_[0-9]*_x86_64"' | tr -d '"')" || true
    if [ -z "$tag" ]; then
        echo "$whl: auditwheel finds no manylinux tag it satisfies" >&2
        "$PYTHON" -m auditwheel show "$whl" >&2 || true
        exit 1
    fi
    cp "$whl" "$OUT/"
    "$PYTHON" -m wheel tags --remove --platform-tag "$tag" "$OUT/$(basename "$whl")"
done
