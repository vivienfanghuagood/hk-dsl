#!/bin/bash
# bench/profile/probe_groups.sh: does each of hk.profile's counter groups fit one rocprofv3 pass here?
# Runs a bf16 GEMM under `rocprofv3 --pmc GROUP` for every group in hk.profile.counters.PASSES of
# this arch, with a timeout (a group that does not fit aborts the program, which can then hang).
#   bash bench/profile/probe_groups.sh            # on the GPU, from the repo root
set -u
ARCH=$(python3 -c "from hk.target import current_arch; print(current_arch())")
OUT=${TMPDIR:-/tmp}/hk-probe-groups
mkdir -p "$OUT"
cat > "$OUT/mm.py" <<'EOF'
import torch
a = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(2):
    c = a @ b
torch.cuda.synchronize()
EOF
python3 - "$ARCH" <<'EOF' > "$OUT/groups.txt"
import sys
from hk.profile.counters import PASSES
for name, cs in PASSES.get(sys.argv[1], []):
    print(name, " ".join(cs))
EOF
while read -r name cs; do
  rm -rf "$OUT/$name"
  if timeout -s KILL 90 rocprofv3 --pmc $cs --output-format csv -d "$OUT/$name" -o run -- python3 "$OUT/mm.py" > "$OUT/$name.log" 2>&1 \
     && ls "$OUT/$name"/*counter_collection.csv > /dev/null 2>&1; then
    echo "$ARCH $name OK: $cs"
    python3 - "$OUT/$name"/*counter_collection.csv <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
last = max(int(r["Dispatch_Id"]) for r in rows)
print("   ", rows[-1]["Kernel_Name"][:60], {r["Counter_Name"]: float(r["Counter_Value"]) for r in rows if int(r["Dispatch_Id"]) == last})
EOF
  else
    echo "$ARCH $name FAILED: $cs"; grep -h -E "exceeds|rror code|not found|Invalid" "$OUT/$name.log" | head -3
  fi
done < "$OUT/groups.txt"
