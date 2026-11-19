"""hk.ops.cross_entropy vs Megatron-LM's vocab_parallel_cross_entropy (TP=1, its torch ops) and TE's fused CE, against fp64; peaked targets, ignore_index, the in-step (S, B, V) layout. Needs ROCm transformer_engine."""
import statistics, torch
from hk.ops.cross_entropy import cross_entropy as hk_ce

class MegatronCE(torch.autograd.Function):   # megatron/core/tensor_parallel/cross_entropy.py at TP=1, no label smoothing
    @staticmethod
    def forward(ctx, vp_logits, target):
        logits_max = torch.max(vp_logits, dim=-1)[0]
        vp_logits = vp_logits - logits_max.unsqueeze(dim=-1)
        predicted = vp_logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        exp_logits = vp_logits
        torch.exp(vp_logits, out=exp_logits)
        sum_exp = exp_logits.sum(dim=-1)
        loss = torch.log(sum_exp) - predicted
        exp_logits.div_(sum_exp.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target)
        return loss
    @staticmethod
    def backward(ctx, g):
        softmax, target = ctx.saved_tensors
        grad = softmax
        grad.scatter_add_(-1, target.unsqueeze(-1), -torch.ones_like(target, dtype=grad.dtype).unsqueeze(-1))
        grad.mul_(g.unsqueeze(-1))
        return grad, None

def te_ce(logits, target):
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy
    return parallel_cross_entropy(logits.unsqueeze(0), target.unsqueeze(0), 0.0, False, None).view(-1)

def megatron(logits, target):   # language_module: logits.float() then the CE
    return MegatronCE.apply(logits.float(), target)

def t(f, reps=10):
    for _ in range(2): f()
    torch.cuda.synchronize(); r = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps): f()
        e1.record(); torch.cuda.synchronize(); r.append(e0.elapsed_time(e1) / reps)
    return statistics.median(r)

torch.manual_seed(0)
from hk.ops.cross_entropy import vocab_parallel_cross_entropy as hk_vp
for n, v, peak in ((4096, 50304, 20.0), (4096, 50304, 22.0), (4096, 50304, 24.0), (1000, 50256, 0.0)):
    logits = (torch.randn(n, v, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, v, (n,), device="cuda")
    if peak == -1.0:   # ignore_index rows (every 7th), compared against fp64 with ignore_index
        target[::7] = -100
    elif peak:   # make half the rows near-certain on their target (p_t -> 1): the failure mode of TE < 2.19
        rows = torch.arange(0, n, 2, device="cuda")
        logits[rows, target[rows]] = peak
    g = (torch.rand(n, device="cuda") + 0.5) / n
    # fp64 reference
    x64 = logits.double().requires_grad_(True)
    l64 = torch.nn.functional.cross_entropy(x64, target, reduction="none", ignore_index=-100)
    l64.backward(g.double())
    res = {}
    arms = (("hk", hk_ce), ("te", te_ce)) if peak == -1.0 else (("hk", hk_ce), ("megatron", megatron), ("te", te_ce))
    for name, fn in arms:
        x = logits.clone().requires_grad_(True)
        loss = fn(x, target)
        loss.backward(g)
        gr = x.grad
        lerr = ((loss.double() - l64).abs().max() / l64.abs().max()).item()
        gerr = ((gr.double() - x64.grad).norm() / x64.grad.norm()).item()
        tc = target.clamp_min(0)
        tgt = gr.gather(1, tc.view(-1, 1)).double().view(-1)
        tref = x64.grad.gather(1, tc.view(-1, 1)).view(-1)
        terr = ((tgt - tref).abs().max() / tref.abs().max()).item()
        zero_t = ((tgt == 0) & (tref.abs() > 1e-6 * tref.abs().max())).sum().item()
        res[name] = (lerr, gerr, terr, zero_t, gr)
    same = (res["hk"][4] == res.get("megatron", res["te"])[4]).float().mean().item()
    print(f"N{n} V{v} peak{peak}: " + " | ".join(f"{k}: loss {a:.1e} grad {b:.1e} target-grad {c:.1e} zeroed {d}" for k, (a, b, c, d, _) in res.items())
          + f" | grads bit-equal hk vs megatron/te {same*100:.3f}%", flush=True)
    if peak == -1.0:
        continue
    xh = logits.clone().requires_grad_(True); xm = logits.clone().requires_grad_(True)
    xt_ = logits.clone().requires_grad_(True)
    th = t(lambda: hk_ce(xh, target).backward(g))
    tm = t(lambda: megatron(xm, target).backward(g))
    tt = t(lambda: te_ce(xt_, target).backward(g))
    print(f"   fwd+bwd: megatron {tm:.3f} ms, TE fused {tt:.3f} ms, hk {th:.3f} ms -> {tm/th:.2f}x megatron, {tt/th:.2f}x TE (hk {3*n*v*2/th/1e6:.0f} GB/s)", flush=True)

# the in-step layout: (S, B, V) logits, labels (S, B) as Megatron's te branch makes them
S, B, V = 2048, 2, 50304
lg = (torch.randn(S, B, V, device="cuda") * 2).to(torch.bfloat16)
lab_bs = torch.randint(0, V, (B, S), device="cuda")
lab = lab_bs.transpose(0, 1).contiguous()
lab = torch.as_strided(lab, lab.size(), (lab.size()[1], 1))
g2 = torch.rand(S, B, device="cuda") / (S * B)
x1 = lg.clone().requires_grad_(True); l1 = hk_vp(x1, lab); l1.backward(g2)
x2 = lg.clone().requires_grad_(True); l2 = MegatronCE.apply(x2.float(), lab); l2.backward(g2)
print(f"in-step layout (S,B,V)=(2048,2,50304): loss max rel {((l1-l2).abs().max()/l2.abs().max()).item():.1e}, "
      f"grad rel L2 {((x1.grad.float()-x2.grad.float()).norm()/x2.grad.float().norm()).item():.1e}", flush=True)
