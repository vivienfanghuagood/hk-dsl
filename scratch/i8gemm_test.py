import torch, statistics
from hk.ops import gemm
from hk.types import i8
torch.manual_seed(0)
k = gemm.gemm_kernel("i8gemm_test", op_dtype=i8, **gemm.DEFAULT)
for M, N, K in ((256, 256, 256), (4096, 4096, 4096), (14720, 21504, 5376)):
    a = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    c = torch.empty(M, N, device="cuda", dtype=torch.int32)
    k(a, b, c); torch.cuda.synchronize()
    ref = torch._int_mm(a, b.t().contiguous()) if M > 16 else None
    ok = torch.equal(c, ref)
    reps = [k(a, b, c) for _ in range(3)]
    det = all(torch.equal(c, ref) for _ in range(1))
    def t(f, n=10):
        for _ in range(2): f()
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            e0,e1=torch.cuda.Event(True),torch.cuda.Event(True); e0.record(); f(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)
    th = t(lambda: k(a, b, c)); tr = t(lambda: torch._int_mm(a, b.t()))
    bt = t(lambda: gemm.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16))) if hasattr(gemm, "matmul") else 0
    fl = 2 * M * N * K
    print(f"{M}x{N}x{K}: exact {ok}  hk i8 {fl/th/1e9:.1f} TOP/s  hipBLASLt _int_mm {fl/tr/1e9:.1f} TOP/s", flush=True)
