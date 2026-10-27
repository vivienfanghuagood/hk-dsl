#!/bin/bash
# For N in a range, compile a kernel that clobbers v0..v(N-1) and read LLVM's VGPRs/Occupancy remark.
# usage: vgpr_sweep.sh ARCH LO HI
set -u
ARCH=$1; LO=$2; HI=$3
D=$(mktemp -d)
for N in $(seq $LO $HI); do
  last=$((N-1))
  cat > $D/k$N.hip <<EOT
#include <hip/hip_runtime.h>
extern "C" __global__ __launch_bounds__(32) void k(float *p) {
  asm volatile("v_mov_b32 v$last, 0" ::: "v$last");
  p[threadIdx.x] = 1.f;
}
EOT
  out=$(hipcc -O3 --offload-arch=$ARCH --cuda-device-only -c $D/k$N.hip -o $D/k$N.o -Rpass-analysis=kernel-resource-usage 2>&1)
  v=$(grep -o "VGPRs: [0-9]*" <<<"$out" | head -1); o=$(grep -o "Occupancy \[waves/SIMD\]: [0-9]*" <<<"$out" | head -1)
  echo "N=$N $v $o"
done
rm -rf $D
