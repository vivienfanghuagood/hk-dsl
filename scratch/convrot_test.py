import torch, statistics
from hk.ops import convrot, gemm as G
import hk
torch.manual_seed(0)
def t(f, n=7):
    for _ in range(2): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(); f(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)
for (M, N, K) in ((4096, 12288, 4096), (14720, 21504, 5376), (14720, 5376, 14336)):
    g = 256
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
    wq, ws = convrot.prepare_weight(w, g)
    y = convrot.linear(x, wq, ws, g)
    ref = convrot.reference(x, wq, ws, g)
    exact = (x.float() @ w.float().t())
    d = (y.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    e8 = (y.float() - exact).abs().max().item() / exact.abs().max().item()
    eb = (hk.ops.matmul(x, w).float() - exact).abs().max().item() / exact.abs().max().item()
    det = all(torch.equal(convrot.linear(x, wq, ws, g), y) for _ in range(3))
    tl = t(lambda: convrot.linear(x, wq, ws, g))
    tb = t(lambda: hk.ops.matmul(x, w))
    xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    tg = t(lambda: torch._int_mm(xq, wq.t()))
    tr = t(lambda: convrot.rotate(x, g)); tq = t(lambda: convrot.quantize_rows(convrot.rotate(x, g))) - tr
    fl = 2 * M * N * K
    print(f"{M}x{N}x{K}: vs-ref maxrel {d:.2e}  vs-fp32 int8 {e8:.2e} bf16 {eb:.2e}  det {det} | "
          f"convrot linear {tl:.2f} ms ({fl/tl/1e9:.0f} T-eq)  hk bf16 {tb:.2f} ms  hipBLASLt int8 gemm only {tg:.2f} ms | rotate {tr:.2f} quant {tq:.2f} ms", flush=True)
