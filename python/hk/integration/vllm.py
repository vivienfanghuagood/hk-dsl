"""vLLM on the generated attention kernel.

    import vllm                      # let vLLM import its modules first
    import hk.integration.vllm as hkv
    print(hkv.apply())               # before the engine is built
    print(hkv.probe())               # what it can actually reach

**Where vLLM calls SDPA, and where it does not.** Read against vLLM at
1dfe9fb (Nov 2026); the shape of this has been stable for several releases.

The decoder's attention does **not** go through
`F.scaled_dot_product_attention` and cannot be made to. vLLM V1 keeps KV in a
paged cache and dispatches through an `AttentionImpl` whose `forward` takes
`kv_cache`, a `block_table` and `cu_seqlens` -- on ROCm that is
`RocmAttentionImpl` or `TritonAttentionImpl`, calling `unified_attention` on
paged blocks. There is no dense (B, H, N, D) tensor anywhere on that path for a
drop-in to intercept. Reaching it needs a *paged* kernel registered as an
attention backend, which is kernel work this package has not done; `hk.ops.attn`
is a dense prefill kernel.

The encoder's attention does. A multimodal model's vision tower runs
`mm_encoder_attention.forward_cuda` -> `_forward_sdpa` ->
`torch.ops.vllm.torch_sdpa_wrapper` -> `apply_sdpa`, whose body is a literal
`F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale,
enable_gqa=...)` on a dense `(B, H, N, D)` tensor. That is this kernel's exact
shape, and patching `F` reaches it because the call resolves the attribute on
the module at call time.

**It is not selected by default on RDNA3.** `RocmPlatform.get_vit_attn_backend`
prefers `FLASH_ATTN` (the Triton one) on gfx11xx whenever
`flash_attn_triton_available()` and the dtype is fp16/bf16, and falls back to
`TORCH_SDPA` only otherwise. Forcing it is a supported configuration:

    vllm serve <model> --mm-encoder-attn-backend TORCH_SDPA

`VLLM_ATTENTION_BACKEND` is **not** the lever -- the variable no longer exists
in current vLLM, and `AttentionBackendEnum.TORCH_SDPA` carries the comment
"this tag is only used for ViT". An earlier version of this file said it was;
it was wrong.

**Which vision towers fit.** `hk.ops.attn` instantiates head_dim 64 and 128, so

    CLIP ViT-L/14 (LLaVA, many adapters)   1024 / 16 heads = 64   fits
    InternViT-300M                         1024 / 16       = 64   fits
    Pixtral                                1024 / 16       = 64   fits
    SigLIP so400m                          1152 / 16       = 72   falls back
    Qwen2-VL / Qwen2.5-VL                  1280 / 16       = 80   falls back

A head_dim the kernel does not have is not an error: `hk.ops.sdpa` forwards it
to torch. `probe()` says which case you are in rather than leaving it to be
inferred from a benchmark.
"""

from __future__ import annotations

from . import _common

PREFIXES = ("vllm",)

_APPLIED = None


def available() -> bool:
    return _common.importable("vllm")


def apply() -> str:
    """Patch vLLM's attention. Returns what happened, in one line."""
    global _APPLIED
    if not available():
        return "vllm is not importable; nothing patched"
    if _APPLIED is not None:
        return f"already applied: {_APPLIED}"
    report, names = _common.patch_sdpa(PREFIXES)
    _APPLIED = report
    if not names:
        report += ("; no vllm module held its own reference, which is normal "
                   "-- the ViT path resolves F.scaled_dot_product_attention on "
                   "the module at call time and is patched regardless")
    return report


def revert() -> str:
    global _APPLIED
    out = _common.unpatch_sdpa(PREFIXES)
    _APPLIED = None
    return out


def status() -> str:
    from ..ops import sdpa  # noqa: PLC0415

    if not available():
        return "vllm not importable"
    return "patched" if sdpa.patched() else "not patched"


def probe() -> dict:
    """What this patch can actually reach in *this* vLLM, right now.

    Written to be run inside a live vLLM process, because every claim in this
    module's docstring is a claim about vLLM's source and the only way to stop
    one going stale is to ask the installed copy. Nothing here raises: a key
    whose answer could not be obtained says so.
    """
    out: dict = {}
    try:
        import torch  # noqa: PLC0415

        from ..ops import attn, sdpa  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return {"torch": f"not importable: {e}"}
    try:
        import vllm  # noqa: PLC0415

        out["vllm"] = getattr(vllm, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return {"vllm": f"not importable: {e}", "torch": torch.__version__}

    out["sdpa_patched"] = (
        torch.nn.functional.scaled_dot_product_attention
        is sdpa.scaled_dot_product_attention
    )

    # Does the installed copy still route the ViT path through F.sdpa?
    try:
        from vllm.v1.attention.ops import vit_attn_wrappers as w  # noqa: PLC0415

        out["vit_sdpa_wrapper"] = hasattr(w, "apply_sdpa")
        out["vit_uses_F_sdpa"] = (
            getattr(w, "F", None) is torch.nn.functional
        )
    except Exception as e:  # noqa: BLE001
        out["vit_sdpa_wrapper"] = f"unavailable: {e}"

    # Which backend this platform picks for a ViT, unforced, at our dtypes.
    try:
        from vllm.platforms import current_platform  # noqa: PLC0415

        picks = {}
        for hd in (64, 72, 80, 128):
            try:
                picks[hd] = str(current_platform.get_vit_attn_backend(
                    head_size=hd, dtype=torch.bfloat16))
            except Exception as e:  # noqa: BLE001
                picks[hd] = f"error: {e}"
        out["vit_backend_default"] = picks
    except Exception as e:  # noqa: BLE001
        out["vit_backend_default"] = f"unavailable: {e}"

    # The decoder side, stated rather than guessed: list the V1 backends the
    # installed copy has, none of which takes a dense tensor.
    try:
        from vllm.v1.attention.backends import registry as r  # noqa: PLC0415

        enum = getattr(r, "AttentionBackendEnum", None)
        out["decoder_backends"] = (
            sorted(m.name for m in enum) if enum is not None else "unavailable"
        )
    except Exception as e:  # noqa: BLE001
        out["decoder_backends"] = f"unavailable: {e}"

    out["head_dims_hk_has"] = list(attn.HEAD_DIMS)
    out["sdpa_calls"] = dict(sdpa.STATS)
    out["fallbacks"] = {f"{list(k[0])} {k[1]}": v
                        for k, v in sdpa.FALLBACKS.items()}
    return out
