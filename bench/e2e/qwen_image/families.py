"""families.py STATS_CSV [--top N]: rocprofv3 kernel_stats.csv -> device time by kernel family."""
import csv, re, sys
FAM = [("hk", r"_kernel\(globals\)|hk_flat|_kernel7globals|globals\)"),
       ("attention", r"attn|fmha|flash|sdpa|aotriton|attention|mha_"),
       ("gemm", r"Cijk|gemm|gemv|wvSplit|matmul|hipblaslt|mm_|ck::|kernel_gemm"),
       ("conv", r"conv|igemm|MIOpen|naive_conv|winograd|col2im|im2col"),
       ("norm", r"norm|rms|layer_norm|group_norm"),
       ("rope/complex", r"rope|complex|polar"),
       ("softmax", r"softmax"),
       ("copy/cat", r"copy|cat|Memcpy|memcpy|transpose|contiguous|direct_copy"),
       ("reduce", r"reduce|sum|mean|max"),
       ("elementwise", r"elementwise|vectorized|unrolled|pointwise|triton_poi|fused"),
       ]
def fam(n):
    for f, p in FAM:
        if re.search(p, n, re.I):
            return f
    return "other"
rows = list(csv.DictReader(open(sys.argv[1])))
tot = {}
top = []
for r in rows:
    ns = float(r.get("TotalDurationNs") or r.get("Total Duration(ns)") or 0)
    n = r.get("Name") or r.get("KernelName")
    f = fam(n)
    tot[f] = tot.get(f, 0) + ns
    top.append((ns, f, int(r.get("Calls") or 0), n[:110]))
T = sum(tot.values())
print(f"total device time {T/1e9:.2f} s")
for f, ns in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"  {f:14s} {ns/1e9:8.2f} s  {ns/T*100:5.1f}%")
N = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 25
for ns, f, c, n in sorted(top, reverse=True)[:N]:
    print(f"  {ns/1e9:7.3f} s {ns/T*100:5.1f}% {c:7d} [{f}] {n}")
