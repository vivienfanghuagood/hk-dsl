# kernel-level, gfx1201, Qwen-Image 2.1's shapes: sgld_linear (hk's GEMM, the route's own _hk_linear) vs F.linear
# under TunableOp (the row's tuned file), and sgld_rope (the route's apply_rope) vs SGLang's apply_rope. Per call: median of
# 5 interleaved rounds of 50 calls (cuda events). usage: PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_FILENAME=<tuned%d.csv> python3 tools/hk-bench/sgld_kernels.py (SGLang installed)
import importlib, statistics, torch
import torch.nn.functional as F
import hk
from hk.integration import sglang_diffusion as sd
torch.cuda.tunable.enable(True); torch.cuda.tunable.tuning_enable(False)
dev, bf = "cuda", torch.bfloat16
def per_call(fns, n=50, rounds=5):
    for f in fns.values():
        for _ in range(5): f()
    t = {k: [] for k in fns}
    for _ in range(rounds):
        for k, f in fns.items():
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(n): f()
            b.record(); b.synchronize(); t[k].append(a.elapsed_time(b) * 1e3 / n)
    return {k: statistics.median(v) for k, v in t.items()}
print("hk", hk.__file__.split("/")[-3], "tunableop results", len(torch.cuda.tunable.get_results()), flush=True)
m = 4096
for n, k in ((12288, 4096), (4096, 4096), (4096, 12288)):
    for bias in (False, True):
        x = torch.randn(m, k, device=dev, dtype=bf); w = torch.randn(n, k, device=dev, dtype=bf) * 0.02
        b = torch.randn(n, device=dev, dtype=bf) * 0.02 if bias else None
        y = sd._hk_linear(x, w, b)
        if y is None:
            print(f"linear {m}x{n}x{k}{' +bias' if bias else ''}: the route leaves it to the engine", flush=True); continue
        err = ((y.float() - F.linear(x, w, b).float()).abs().max() / F.linear(x, w, b).float().abs().max()).item()
        t = per_call({"engine": lambda: F.linear(x, w, b), "hk": lambda: sd._hk_linear(x, w, b)})
        print(f"linear {m}x{n}x{k}{' +bias' if bias else ''}: F.linear {t['engine']:.0f} us, hk {t['hk']:.0f} us, "
              f"{t['engine'] / t['hk']:.2f}x; max rel err {err:.1e}; route decision {sd.DECISIONS.get((sd._bucket(m), n, k, bias), ('?',))[0]}", flush=True)
mod = importlib.import_module(sd.QWEN21_HOME)
orig = mod.apply_rope
sd.apply_rope(); hk_rope = mod.apply_rope
for s in (4141, 4096):
    x = torch.randn(1, s, 32, 128, device=dev, dtype=bf)
    ang = torch.rand(s, 64, device=dev) * 6.28
    rope = torch.polar(torch.ones_like(ang), ang).contiguous()
    same = torch.equal(orig(x, rope), hk_rope(x, rope))
    t = per_call({"engine": lambda: orig(x, rope), "hk": lambda: hk_rope(x, rope)})
    print(f"rope (1,{s},32,128): SGLang {t['engine']:.0f} us, hk {t['hk']:.0f} us, {t['engine'] / t['hk']:.2f}x; bit-identical {same}", flush=True)
sd.revert_rope()
