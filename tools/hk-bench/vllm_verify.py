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

  --e2e     A real multimodal request through a real engine, run patched and
            unpatched, interleaved. Reads `hk.ops.sdpa.STATS` to prove which
            branch the server took, and compares the decoded text against an
            unpatched run *repeated*, so that "the patched arm said something
            else" can be distinguished from "this engine is not reproducible".

            It is not a speed measurement and the number it prints is not one.
            The vision tower is ~2% of a 2B VLM's request and the per-request
            spread is +-25%; `--unit` is where the tower's time is.

            Two things it caught that nothing else could. vLLM V1 runs the
            model in a *spawned* process, so `apply()` in the process that
            built `LLM(...)` patches a process the model never runs in -- the
            counters read zero while the answers and the timings looked
            perfectly normal. And with `mm_processor_cache` and prefix caching
            on, the second request for an image already seen skips the vision
            tower entirely, so a warm-up consumed the only ViT pass. Hence
            `VLLM_ENABLE_V1_MULTIPROCESSING=0` here, the general plugin for
            production (`hk/integration/_vllm_plugin.py`), the caches off, and
            a distinct image per request.

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


def _image_data_url(seed: int = 0, px: int = 448) -> str:
    """A synthetic image, inline, so this needs no egress and no fixture.

    `seed` matters more than it looks. vLLM caches the vision encoder's output
    per image (`mm_processor_cache`) and caches prefixes, so sending the *same*
    image twice runs the tower exactly once -- which is how the first run of
    this script reported `{'kernel': 0, 'fallback': 0}` after a warm-up had
    already consumed the only ViT pass. Every request here gets its own image.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    # Smooth rather than white noise: a noise image makes a VLM produce
    # degenerate text, and degenerate text is a weak equality check.
    small = rng.integers(0, 255, (16, 16, 3), dtype="uint8")
    img = Image.fromarray(small).resize((px, px), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def e2e(model: str, *, max_tokens: int = 64, reps: int = 6) -> int:
    """One real multimodal request, generated unpatched and then patched.

    `llm.chat` rather than a hand-written prompt so this does not encode one
    model's placeholder convention; the model's own chat template inserts it.
    """
    import torch  # noqa: F401 -- torch first, as vLLM expects

    from vllm import LLM, SamplingParams

    import hk.integration.vllm as hkv
    from hk.ops import sdpa as hk_sdpa

    backend = os.environ.get("HK_VIT_BACKEND", "TORCH_SDPA")
    print(f"building {model} with mm_encoder_attn_backend={backend}")
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("HK_MAX_LEN", "4096")),
        gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.85")),
        mm_encoder_attn_backend=backend,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        # Both caches off. With them on, the second request for an image it has
        # already seen skips the vision tower entirely, and the thing under
        # test stops running without anything saying so.
        mm_processor_cache_gb=0,
        enable_prefix_caching=False,
        enforce_eager=os.environ.get("HK_EAGER", "0") == "1",
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def run(seed: int):
        messages = [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": _image_data_url(seed)}},
            {"type": "text", "text": "Describe this image in one sentence."},
        ]}]
        t0 = time.perf_counter()
        out = llm.chat(messages, params)
        return out[0].outputs[0].text, time.perf_counter() - t0

    run(0)                      # warm: compile, capture, allocator
    hkv.apply()
    run(0)                      # warm again: the kernel JITs on first call
    hkv.revert()

    # Image 1 is the one both arms are compared on.
    hk_sdpa.reset_stats()
    base_text, _ = run(1)
    # The same request again, still unpatched. Without this the comparison
    # below cannot be read: vLLM with chunked prefill and async scheduling is
    # not bitwise reproducible, so "the patched arm said something else" is
    # only evidence if the unpatched arm says the same thing twice.
    base_text2, _ = run(1)
    before = dict(hk_sdpa.STATS)

    print(hkv.apply())
    hk_sdpa.reset_stats()
    hk_text, _ = run(1)
    after = dict(hk_sdpa.STATS)
    hkv.revert()

    # Interleaved, because two blocks of requests are not comparable here.
    # The first version of this ran one arm and then the other and reported
    # 1.18x -- and the control that ran the *same* arm twice under a
    # different ViT backend reported 1.16x with the kernel never called at
    # all. Whatever that 16% is (allocator, caches, clocks), it belongs to
    # whichever block ran second, so the arms alternate and swap order
    # between rounds, and every request gets its own image.
    times = {"torch": [], "hk": []}
    seed = 100
    for r in range(reps):
        for arm in (["torch", "hk"] if r % 2 == 0 else ["hk", "torch"]):
            hkv.apply() if arm == "hk" else hkv.revert()
            times[arm].append(run(seed)[1])
            seed += 1
    hkv.revert()
    base, hk_s = min(times["torch"]), min(times["hk"])

    print(f"\nunpatched  {base * 1e3:8.1f} ms   sdpa calls {before}")
    print(f"patched    {hk_s * 1e3:8.1f} ms   sdpa calls {after}")
    # Printed, and not to be read as a speedup. A 2B VLM generating 64
    # tokens spends ~250 ms in the decoder and ~4 ms in the vision tower, so
    # even a tower that took no time at all would move this by under 2% --
    # and the per-request spread below is +-25%. The tower's own number is
    # `--unit`. What --e2e establishes is that the kernel is reached, that
    # the server does not crash, and what the model then says.
    print(f"request    {base / hk_s:.3f}x  (end to end -- NOISE, see below; "
          f"the tower is ~2% of this)")
    print(f"  torch ms: {[round(t * 1e3, 1) for t in times['torch']]}")
    print(f"  hk ms   : {[round(t * 1e3, 1) for t in times['hk']]}")
    print(f"unpatched reproducible: {base_text == base_text2}")
    print(f"same text as unpatched:  {base_text == hk_text}")
    print(f"  unpatched : {base_text!r}")
    print(f"  unpatched2: {base_text2!r}")
    print(f"  patched   : {hk_text!r}")
    if hk_sdpa.FALLBACKS:
        print("fallbacks:")
        for shape, why in hk_sdpa.FALLBACKS.items():
            print(f"  {shape}: {why}")

    rc = 0
    if after["kernel"] == 0:
        print("\n!! the server never took the kernel branch -- the patch did "
              "not reach this model's attention")
        rc = 1
    if base_text != hk_text and base_text == base_text2:
        # Only a finding if the baseline was reproducible. Greedy decoding on
        # an ambiguous image is a near-tie between tokens, and any kernel swap
        # -- aotriton for Triton, one vLLM version for the next -- can flip
        # one. The numeric gate is tests/hk/gpu/test_attn.py, not this.
        print("\n?? the arms differ while the unpatched arm repeated itself; "
              "worth a logprob comparison before trusting the kernel here")
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
