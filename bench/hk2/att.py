"""Thread-trace one attention kernel (hk2 or Triton) and print per-class dynamic cycles per MFMA.

    python bench/hk2/att.py hk2|triton [D]          (inside x950: needs rocprofv3 and /dec)
Runs itself under rocprofv3 --att, then summarizes the busiest dispatch's code.json.
"""
import collections
import glob
import json
import os
import re
import subprocess
import sys
import tempfile


def cls(m):
    if m.startswith(("v_mfma", "v_smfmac", "v_wmma")):
        return "mfma"
    if m.startswith(("v_exp", "v_log", "v_rcp", "v_sqrt", "v_rsq")):
        return "trans"
    if m.startswith("v_"):
        return "valu"
    if m.startswith("ds_"):
        return "lds"
    if m.startswith(("buffer_", "global_", "flat_", "scratch_")):
        return "vmem"
    if m == "s_nop":
        return "s_nop"
    if m.startswith("s_wait"):
        return "wait"
    if m.startswith("s_barrier"):
        return "barrier"
    if m.startswith("s_"):
        return "salu"
    return "other"


def summarize(code_json):
    code = json.load(open(code_json))["code"]
    by_c = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in code:
        t = r[0]
        if t.startswith(";") or not t.strip():
            continue
        m = re.sub(r"_e(32|64)$", "", t.split()[0])
        c = cls(m)
        for i in range(4):
            by_c[c][i] += r[6 + i]
    mf = by_c["mfma"][0] or 1
    tot = sum(v[1] for v in by_c.values()) or 1
    print(f"{'class':8s} {'hits/mfma':>10s} {'lat/mfma':>9s} {'stall/mfma':>11s} {'share':>6s}")
    for c, v in sorted(by_c.items(), key=lambda kv: -kv[1][1]):
        print(f"{c:8s} {v[0] / mf:10.2f} {v[1] / mf:9.1f} {v[2] / mf:11.1f} {v[1] / tot:6.1%}")
    rows = [r for r in code if not r[0].startswith(";") and r[0].strip() and r[6]]
    print("top stalls (per MFMA):")
    for r in sorted(rows, key=lambda r: -r[8])[:14]:
        print(f"  {r[8] / mf:7.2f}  hits {r[6] / mf:5.2f}  {r[0][:90]}")


def child(which, d):
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    B, H, S = 1, 32, 4096
    q, k, v = (torch.randn(B, H, S, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    if which == "triton":
        import triton_ref
        f = lambda: triton_ref.attention(q, k, v)
    else:
        from hk2.kernels.attention import attention
        f = lambda: attention(q, k, v)
    for _ in range(3):
        f()
    torch.cuda.synchronize()


if __name__ == "__main__":
    which = sys.argv[1]
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    if os.environ.get("HK2_ATT_CHILD"):
        child(which, d)
        sys.exit(0)
    out = tempfile.mkdtemp(prefix="att-", dir="/hk2home")
    regex = "_fa" if which == "triton" else "attn_fwd"
    env = dict(os.environ, HK2_ATT_CHILD="1")
    cmd = ["rocprofv3", "--att", "--att-library-path", "/dec", "--kernel-include-regex", regex, "-d", out, "-o", "x",
           "--", sys.executable, __file__, which, str(d)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    jsons = sorted(glob.glob(os.path.join(out, "**", "ui_output_agent_*_dispatch_*", "code.json"), recursive=True))
    if not jsons:
        print(r.stdout[-2000:], r.stderr[-3000:])
        sys.exit(1)
    print(f"# {which} D={d}: {len(jsons)} traced dispatches; last one:")
    summarize(jsons[-1])
