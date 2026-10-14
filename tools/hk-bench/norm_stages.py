#!/usr/bin/env python3
"""Where a one-row norm's five microseconds go: the same 1x4096 row through
kernels that each add one stage of rms_norm, timed in one process."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hk  # noqa: E402
from hk.dsl.collective import cross_warp, fold_rows  # noqa: E402

T = hk.RT(hk.bf16, 16, 64)
F = hk.RT(hk.fp32, 16, 64)


def one(p):
    return (1, 1, 1)


def stage(n, wide=False, w_late=False):
    """n: 0 copy, 1 +cvt, 2 +rowsum, 3 +xwarp, 4 +fold, 5 +weight."""
    def load(view, idx):
        if wide:
            return view[idx].load(F, bounded=True)
        return view[idx].load(T, bounded=True).to(hk.fp32)

    def body(x: hk.Tensor[hk.bf16], w: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.elem_coord(0, 0, 0, hk.arch.warp_id() * 64)
        if n == 0:
            o[idx].store(x[idx].load(T, bounded=True), bounded=True)
            return
        v = load(x, idx)
        g = load(w, idx) if n >= 5 and not w_late else None
        if n >= 2:
            sq = hk.zeros(hk.col_vec(F))
            sq += (v * v).sum(axis=1)
            if n >= 3:
                sq = cross_warp(sq, "add", 4)
            if n >= 4:
                sq = fold_rows(sq, "add", 4)
            sq += 1e-6
            v *= sq.rsqrt()
        if n >= 5:
            v *= g if g is not None else load(w, idx)
        o[idx].store(v if wide else v.to(hk.bf16), bounded=True)
    name = f"probe_s{n}{'w' if wide else ''}{'l' if w_late else ''}"
    body.__name__ = name
    return hk.kernel(body, warps=4, grid=one, name=name)


VARIANTS = {
    "hk copy": stage(0),
    "hk +cvt": stage(1),
    "hk +rowsum": stage(2),
    "hk +xwarp": stage(3),
    "hk +fold": stage(4),
    "hk +weight": stage(5),
    "hk +weight late": stage(5, w_late=True),
    "wide +fold": stage(4, wide=True),
    "wide +weight": stage(5, wide=True),
}


def code_sizes():
    """Bytes of .text per variant, from a device-only compile (no GPU)."""
    import re
    import subprocess
    import tempfile

    from hk.runtime import arch, compile as c

    fl = [f for f in c.flags(arch.get_target("gfx1100"), "probe")
          if f not in ("-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage")]
    out = {}
    for name, k in VARIANTS.items():
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "k.cpp"
            src.write_text(k.source())
            asm = Path(d) / "k.s"
            subprocess.run([c.hipcc(), *fl, "--offload-device-only", "-S", "-o",
                            str(asm), str(src)], check=True, capture_output=True)
            text = asm.read_text()
            n = len(re.findall(r"^\s+(?:v_|s_|ds_|buffer_|global_|flat_)", text, re.M))
            out[name] = n
    return out


def main():
    if "--static" in sys.argv:
        for k, v in code_sizes().items():
            print(f"{k:<16} {v:5d} instructions")
        return
    import torch  # noqa: PLC0415

    import scoreboard  # noqa: PLC0415
    import triton_ref  # noqa: PLC0415

    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)
    xf, wf, of = x.view(16, 256), w.view(16, 256), o.view(16, 256)
    fns = {name: (lambda k=k: k(xf, wf, of)) for name, k in VARIANTS.items()}
    fns["hk rms_norm"] = lambda: hk.ops.rms_norm(x, w, out=o)
    fns["triton rms"] = lambda: triton_ref.rms_norm(x, w)
    t = scoreboard.gpu_times(fns, rounds=5, calls=20)
    for k, v in t.items():
        print(f"{k:<16} {v * 1e3:7.2f} us")


if __name__ == "__main__":
    main()
