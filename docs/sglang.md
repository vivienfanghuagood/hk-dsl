# hk in SGLang and SGLang-Diffusion

The routes hk applies inside SGLang (linear and MoE layers) and SGLang-Diffusion (attention and linear layers of text-to-image and video models). What "on by default" means, `--status`, the engine contract and the environment variables are in [integration](integration.md); the generated [route table](supported-hardware.md#engine-routes-on-each) says which routes are on for each GPU.

- [SGLang](#sglang)
- [SGLang-Diffusion (text to image and video)](#sglang-diffusion-text-to-image-and-video)

## SGLang

Launched with the plugin as in [on by default](integration.md#on-by-default-and-turning-it-off), or patched
in-process:

```python
import sglang                      # let SGLang import its modules first
import hk.integration.sglang as hks
print(hks.apply_linear())          # unquantized linear layers -> hk_vllm::linear
print(hks.apply())                 # SDPA, before the engine is built
```

**The linear layers are measured inside SGLang.** The setup was SGLang 0.5.18
with AMD-AIM/sglang-radeon 0.1.5 and its prebuilt gfx1100 kernel, torch
2.9.1 on ROCm 7.2.1, and Qwen3-8B in bf16, eager with Triton attention as
sglang-radeon runs it.
- `apply_linear()` rebinds `UnquantizedLinearMethod.apply`, whose ROCm path
  without aiter is `F.linear`, to `torch.ops.hk_vllm.linear`, routed by the
  SGLang table. Every call took an hk kernel.
- The patch does not change accuracy: hk's linear and F.linear land equally
  far from an fp64 matmul, to three digits. In a greedy check, two of three
  prompts matched stock over 48 tokens; the third took the other side of a
  near-tie at its ninth token.

**MoE models run, and the expert GEMMs are measured inside SGLang**.
- **Without the patch they do not run.** sgl_kernel's `topk_softmax` fails
  on gfx1100 on its own, with `unspecified launch failure`, before any
  expert GEMM. So no softmax-routed MoE model ran in SGLang on this card.
- **What the patch changes.** `sglang_moe.apply_topk()` computes the top-k
  in torch, as SGLang's own `fused_topk_torch_native` does, and with only
  that, OLMoE runs on SGLang's Triton MoE kernel. `sglang_moe.apply()` also
  routes the bf16 expert GEMMs through `hk.ops.moe`, asking for 16-row
  blocks up to 256 tokens a step.
- **Results.** Greedy output identical to Triton's GEMMs over SGLang's
  correctness prompts. Prefill steps of more than 256 tokens
  stay on Triton.

**Limits.**
- **No CUDA graphs.** sglang-radeon disables them on RDNA3 because HIP graph
  capture hangs there, so these are eager numbers. Even so, the decode step
  is GPU-bound: at batch 1 the linear layers' timings, measured one by one,
  add up to 42.1 ms of a 44.3 ms step.
- **Attention is not verified.** SGLang's RDNA3 path selects its Triton
  attention backend, which does not call `F.scaled_dot_product_attention`.
  On a text model `apply()` therefore reports that it rebound nothing; it
  bites with `--attention-backend torch_native` and in the vision encoders
  of multimodal models.

## SGLang-Diffusion (text to image and video)

```bash
python3 -m hk.integration sglang-diffusion install    # once per environment
python my_generate.py                                # attention and GEMM: on by default (gfx1100)
HK_SGLD_LINEAR=0 python my_generate.py               # attention only (HK_SGLD_ATTN=0: GEMM only)
HK_DISABLE=1 python my_generate.py                   # stock
python3 -m hk.integration sglang-diffusion status
```

SGLang-Diffusion (`sglang.multimodal_gen`) runs the DiT in worker processes
and loads no plugins there. So `install` writes `hk_sgld.pth`, and the boot it
imports, into site-packages. Unless `HK_DISABLE=1` or `HK_SGLD=0`, the boot
patches whichever process imports the SDPA backend. It checks the engine
contract first, as the plugins do. An install made before hk was on by
default keeps the old boot, which needs `HK_SGLD=1`; run `install` again
(`status` says so). Two patches are on by default:
- **attention**: `hk.ops.sdpa.patch()` behind the `torch_sdpa` backend (the
  only one that runs on RDNA3); masked and odd-head-dim calls go to torch;
- **linear**: `UnquantizedLinearMethod.apply` of
  `sglang.multimodal_gen.runtime.layers.linear` to `hk.ops.matmul` for bf16
  projections without bias and with at least `MIN_M` (1024) rows.

The third, ConvRot INT8 (hk's `convrot_int8` backend, for `--quantization
convrot_int8`), **stays opt-in**: it is lossy by design.
`HK_SGLD=attn,linear,convrot` or `HK_SGLD_CONVROT=1` turns it on.

Each patched process prints how many calls took hk's kernels: whenever the
counts changed, at most every `HK_SGLD_REPORT_S` seconds (60), and as it
exits. The worker that runs the DiT is torn down without running `atexit`,
so the line printed at exit is the parent's, whose counts stay 0; read the
worker's lines (their pid differs).
Checked on Qwen-Image 2.1 (1024^2, 30 steps, one W7900D): images visually
identical to stock's; attention reads SGLang's (B, N, H, D) views in place. With sglang-radeon 0.1.5 installed,
its `h3_denoise_platform_gate` shim must be narrowed first (it makes the global
platform claim CUDA, and SGLang then picks CUDA-only kernels on gfx1100); the
chapter has the hook.
