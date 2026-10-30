# The build environment of the cp311 and cp313 release wheels (docs/building.md, "Wheels").
#
#   docker build -t hk-wheel-builder:jammy -f scripts/wheel-builder.Dockerfile scripts
#
# Ubuntu 22.04, like the cp310 and cp312 builds: glibc 2.35 and gcc 11.4, so these wheels are
# manylinux_2_35 too and link nothing newer. CPython 3.11 and 3.13 come from the deadsnakes PPA. Each
# has a venv with pinned setuptools and wheel, so the wheel's metadata does not drift with whatever
# is newest. hk._mlir is built in it against the same MLIR install as the cp310 / cp312 wheels.
FROM ubuntu:22.04
RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common gpg-agent \
 && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends build-essential git \
      ca-certificates binutils ninja-build python3.11 python3.11-dev python3.11-venv python3.13 \
      python3.13-dev python3.13-venv \
 && rm -rf /var/lib/apt/lists/*
RUN for v in 3.11 3.13; do \
      python$v -m venv /opt/py$v && /opt/py$v/bin/pip install -q --no-cache-dir setuptools==80.9.0 wheel==0.45.1; \
    done
