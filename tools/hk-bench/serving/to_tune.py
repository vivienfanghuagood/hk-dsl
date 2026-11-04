"""to_tune.py OUT_CSV UNTUNED_CSV... -- offline TunableOp tuning of the GEMM shapes a measured run used.

The shapes come from a run in --mode to_record (TunableOp on, tuning off, PYTORCH_TUNABLEOP_RECORD_UNTUNED=1,
same engine flags and CUDA graphs as the measurement), so they include the padded CUDA-graph batch sizes.
Each shape is tuned once with TunableOp's default tuning budget on this GPU (hipBLASLt + rocBLAS
candidates), and the results file is written to OUT_CSV (copy it to each rank's file name)."""
import os, sys
import torch

out, files = sys.argv[1], sys.argv[2:]
t = torch.cuda.tunable
t.enable(True)
t.tuning_enable(True)
t.set_filename(out, insert_device_ordinal=False)
seen = set()
for f in files:
    for line in open(f):
        if line.strip() and line not in seen:
            seen.add(line)
merged = os.path.join(os.path.dirname(out) or ".", "untuned-merged.csv")
open(merged, "w").writelines(sorted(seen))
print(f"{len(seen)} unique untuned GEMMs", flush=True)
t.tune_gemm_in_file(merged)
if hasattr(t, "write_file"):
    t.write_file()   # newer torch writes the file in tune_gemm_in_file
print("results", len(t.get_results()), flush=True)
