#!/usr/bin/env python3
"""End-to-end verification of hk.integration.vllm inside a real vLLM.

Three stages, each answering a question the one before it cannot:

  --probe   What does *this* vLLM look like? Version, which backend it picks
            for a ViT at each head_dim, which decoder backends exist. Every
            claim in `hk/integration/vllm.py`'s docstring is a claim about
            vLLM's source, and this is how it gets checked against the
            installed copy instead of against a file read months ago.

  --unit    The exact call the ViT path makes -- einops-permuted (B,H,N,D),
            bf16, scale set, dropout 0 -- against torch, for numbers and for
            time. Runs without an engine, so a failure here is this package's
            and not vLLM's.

  --e2e     A real multimodal request through a real engine, generated twice:
            once unpatched, once patched. Compares the decoded text and reads
            `hk.ops.sdpa.STATS` to prove which branch the server actually took.
            A patch that silently falls back produces identical text and no
            speedup, which is indistinguishable from a patch that works unless
            something counts.

The decoder's attention is NOT under test and cannot be: vLLM V1 keeps KV in a
paged cache and dispatches through AttentionImpl.forward(kv_cache, block_table,
...), so there is no dense tensor for an SDPA drop-in to intercept. --probe
prints the backend list that makes that concrete.

    python3 tools/hk-bench/vllm_verify.py --probe
    python3 tools/hk-bench/vllm_verify.py --unit
    python3 tools/hk-bench/vllm_verify.py --e2e llava-hf/llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

#: (name, heads, head_dim, tokens) for vision towers that matter here. The
#: token counts are the real ones: CLIP-L/14 at 336 px is 24x24 patches plus
#: the CLS token.
TOWERS = [
    ("CLIP ViT-L/14-336 (LLaVA)", 16, 64, 577),
    ("InternViT-300M", 16, 64, 1025),
    ("SigLIP so400m", 16, 72, 729),
    ("Qwen2.5-VL ViT", 16, 80, 1024),
]


def probe() -> int:
    import hk.integration.vllm as hkv

    print(json.dumps(hkv.probe(), indent=2, default=str))
    print()
    print(hkv.apply())
    print(json.dumps(hkv.probe(), indent=2, default=str))
    return 0


def unit(rounds: int = 5) -> int:
    """The ViT call shape, hk against torch, same process, interleaved."""
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _timing import interleave

    from hk.ops import sdpa as hk_sdpa

    rc = 0
    print(f"{'tower':>28} {'B,H,N,D':>18} {'path':<16} {'ms':>8} "
          f"{'vs fp32':>9} {'bf16 ulp':>9} {'speedup':>8}")
    for name, heads, head_dim, n in TOWERS:
        # Exactly what apply_sdpa hands over: a (b,s,h,d) tensor permuted to
        # (b,h,s,d), i.e. NOT contiguous. The drop-in has to cope with that,
        # and the copy it does is part of what is being timed.
        g = torch.Generator(device="cuda").manual_seed(0)
        qb, kb, vb = (torch.randn(1, n, heads, head_dim, device="cuda",
                                  dtype=torch.bfloat16, generator=g)
                      for _ in range(3))
        q, k, v = (x.permute(0, 2, 1, 3) for x in (qb, kb, vb))
        scale = head_dim ** -0.5

        # Both paths are bf16 approximations of the same fp32 answer, so
        # comparing them to each other only says they disagree, never which
        # one is wrong. The reference is fp32; the claim is that hk is no
        # further from it than torch is.
        ref = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), scale=scale)
        hk_sdpa.reset_stats()
        got = hk_sdpa.scaled_dot_product_attention(q, k, v, scale=scale)
        want = F.scaled_dot_product_attention(q, k, v, scale=scale)
        torch.cuda.synchronize()
        err = (got.float() - ref).abs().max().item()
        torch_err = (want.float() - ref).abs().max().item()
        used = "kernel" if hk_sdpa.STATS["kernel"] else "fallback"
        fits = head_dim in (64, 128)
        if used != ("kernel" if fits else "fallback"):
            print(f"  !! {name}: took the {used} branch, expected "
                  f"{'kernel' if fits else 'fallback'}")
            rc = 1
        # Both columns are distances from the same fp32 answer, printed so
        # the difference between them is visible rather than asserted away.
        # hk is consistently the further of the two -- a fused online softmax
        # rescales the accumulator where torch's two-pass form does not -- and
        # consistently within a small multiple of one bf16 ulp, which is the
        # resolution either path can round to anyway.
        #
        # This is a sanity bound, not the correctness gate. The gate is
        # tests/hk/gpu/test_attn.py (31 cases) and test_sdpa.py, which compare
        # against torch at rtol=atol=2e-2.
        ulp = ref.abs().max().item() * 2 ** -8
        if err > 4 * max(torch_err, ulp):
            print(f"  !! {name}: hk is {err:.5f} from fp32 against torch's "
                  f"{torch_err:.5f} (one bf16 ulp here is {ulp:.5f})")
            rc = 1

        best = interleave({
            "hk": lambda q=q, k=k, v=v, s=scale:
                hk_sdpa.scaled_dot_product_attention(q, k, v, scale=s),
            "torch": lambda q=q, k=k, v=v, s=scale:
                F.scaled_dot_product_attention(q, k, v, scale=s),
        }, rounds=rounds)
        shape = f"1,{heads},{n},{head_dim}"
        for path, e in (("hk", err), ("torch", torch_err)):
            tag = f"{path} ({used})" if path == "hk" else path
            print(f"{name:>28} {shape:>18} {tag:<16} {best[path]:8.4f} "
                  f"{e:9.5f} {ulp:9.5f} {best['torch'] / best[path]:8.2f}x")
        print()
    return rc


def _image(px: int = 336):
    from PIL import Image

    # Synthetic rather than downloaded: the content does not matter, only that
    # the vision tower runs, and a pod with no egress should still be able to
    # run this.
    import numpy as np

    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (px, px, 3), dtype="uint8"))


def e2e(model: str, *, max_tokens: int = 64, reps: int = 3) -> int:
    import torch  # noqa: F401 -- imported for the side effect of being first

    from vllm import LLM, SamplingParams

    import hk.integration.vllm as hkv
    from hk.ops import sdpa as hk_sdpa

    print(f"building engine for {model} with --mm-encoder-attn-backend "
          f"TORCH_SDPA")
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.85")),
        mm_encoder_attn_backend="TORCH_SDPA",
        enforce_eager=os.environ.get("HK_EAGER", "0") == "1",
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    req = {"prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
           "multi_modal_data": {"image": _image()}}

    def run():
        t0 = time.perf_counter()
        out = llm.generate([req], params)
        return out[0].outputs[0].text, time.perf_counter() - t0

    # Warm the engine (compile, capture, allocator) before either arm.
    run()

    hk_sdpa.reset_stats()
    base_text, _ = run()
    base = min(run()[1] for _ in range(reps))
    before = dict(hk_sdpa.STATS)

    print(hkv.apply())
    hk_sdpa.reset_stats()
    run()                       # let the kernel JIT and the graph re-warm
    hk_text, _ = run()
    hk_ms = min(run()[1] for _ in range(reps))
    after = dict(hk_sdpa.STATS)

    print(f"\nunpatched  {base * 1e3:8.1f} ms   sdpa calls {before}")
    print(f"patched    {hk_ms * 1e3:8.1f} ms   sdpa calls {after}")
    print(f"same text: {base_text == hk_text}")
    if base_text != hk_text:
        print(f"  unpatched: {base_text!r}")
        print(f"  patched:   {hk_text!r}")
    if hk_sdpa.FALLBACKS:
        print("fallbacks:")
        for shape, why in hk_sdpa.FALLBACKS.items():
            print(f"  {list(shape[0])} {shape[1]}: {why}")

    rc = 0
    if after["kernel"] == 0:
        print("\n!! the server never took the kernel branch -- the patch did "
              "not reach this model's attention")
        rc = 1
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--e2e", metavar="MODEL")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args(argv)

    rc = 0
    if args.probe:
        rc |= probe()
    if args.unit:
        rc |= unit(args.rounds)
    if args.e2e:
        rc |= e2e(args.e2e, max_tokens=args.max_tokens)
    if not (args.probe or args.unit or args.e2e):
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
