# SGLang routes: reference

To turn hk on in SGLang or SGLang-Diffusion and check it, read [hk in SGLang](sglang.md) first. This page is the reference for the routes hk applies inside SGLang (linear and MoE layers) and SGLang-Diffusion (attention and linear layers of text-to-image and video models). What "on by default" means, `--status`, the engine contract and the environment variables are in [integration](integration.md); the generated [route table](supported-hardware.md#engine-routes-on-each) says which routes are on for each GPU.

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

**How to read the numbers in this section.** They are measured in **eager** mode,
SGLang-Diffusion's default: through hk 0.9.20 the routes did not run under
`--enable-torch-compile` (every request failed; fixed after #384). Unless a
GPU is named they are from one W7900D. "x a video / a request" is end to end:
either hk against stock, or -- where it says so -- one serving process with
only that route toggled and every other hk route on in both arms (the
baseline is then hk with the route off). Per-call numbers ("a call", "a
conv") are kernel level at the model's shapes. The hk version and the
evidence of each route are in [integration](integration.md#on-by-default-and-turning-it-off).

SGLang-Diffusion (`sglang.multimodal_gen`) runs the DiT in worker processes
and loads no plugins there. So `install` writes `hk_sgld.pth`, and the boot it
imports, into site-packages. Unless `HK_DISABLE=1` or `HK_SGLD=0`, the boot
patches whichever process imports the SDPA backend. It checks the engine
contract first, as the plugins do. An install made before hk was on by
default keeps the old boot, which needs `HK_SGLD=1`; run `install` again
(`status` says so). Five patches are on by default:
- **attention**: `hk.ops.sdpa.patch()` behind the `torch_sdpa` backend (the
  only one that runs on RDNA3); masked and odd-head-dim calls go to torch;
- **linear**: `UnquantizedLinearMethod.apply` of
  `sglang.multimodal_gen.runtime.layers.linear` to hk's GEMM for bf16
  projections with at least `MIN_M` (1024) rows, contiguous or rows of a
  wider buffer passed with their row stride (a DiT's fc2 reading the
  activation half of its gate / up buffer: MiniMax H3's, 1.57-1.89x
  `F.linear` on the same view on a W7900D). A bias (Wan's projections
  have one) is added in the GEMM's store, `hk.templates.gemm`'s bias
  epilogue, as hipBLASLt does: 1.22-1.26x `F.linear` at Wan2.1's shapes,
  1.047x a Wan2.1 832x480x81 video. `LINEAR_BIAS = False` leaves biased
  projections to the engine. Each shape is decided once, by timing hk against
  `F.linear` (hk must win by `MARGIN`, 3%), and the decision is pinned in
  hk's cache directory (`sgld_linear_<arch>_<gpu>.json` under `HK_CACHE_DIR`),
  so every later process and every TP rank takes the same kernel and the same
  seed gives the same image; delete the file to decide again;
- **vaeconv**: Wan's VAE conv3d (`WanCausalConv3d` of
  `sglang.multimodal_gen.runtime.models.vaes.wanvae`) as calls of 2 output
  frames, each the engine's own conv on a slice of the input. On gfx1100
  MIOpen runs the decoder's full-resolution convs, which get 6 input frames,
  on its naive reference kernel: 9.5 s a call at 832x480, against 95 ms in
  2-frame calls. It also keeps the VAE in NCDHW: SGLang runs it channels-last
  when its conv weights are (a cuDNN path), and MIOpen converts every such
  conv to NCDHW and back, 10 s of copies a video; NCDHW, a Wan2.1 video is
  1.052x faster again. And each conv3d runs as one conv2d: the frames are
  the batch, an output frame's kt input frames are stacked on the channels
  against the weight's time taps side by side (`hk.ops.conv3d.as_conv2d`),
  so MIOpen uses its conv2d solvers instead of its conv3d im2col, with the
  sum over taps in one fp32 accumulation: 1.75-2.7x the 2-frame calls at the
  same error, 1.055x a video. `VAE_CONV2D = False` goes back to the 2-frame
  calls. On gfx1201 (on by default there too) MIOpen runs every
  channels-last VAE conv on its naive reference kernel: a 9-frame Wan2.1
  decode takes 2,680 s stock and 4.1 s with the route, closer to an fp32 VAE
  decode than stock (38.60 dB vs 38.53). No hk kernel; the route only steers
  MIOpen.
  Wan2.1-T2V-1.3B 832x480x81 on one W7900D: VAE decode 969 s -> 37.8 s, the
  generation 2.77x. 36.2 dB from stock and closer to an fp32 VAE decode than stock (36.9 dB vs 36.1);
- **rope**: Qwen-Image 2.1's complex RoPE (`apply_rope` of
  `sglang.multimodal_gen.runtime.models.dits.qwen_image21`, which runs as four
  torch launches: cast to fp32, complex multiply, cast back, copy) on
  `hk.ops.rotary_pairs`, one launch that reproduces torch's complex multiply
  bit for bit: 42 us instead of 446 us a call at 4141 tokens. Qwen-Image 2.1
  1024^2, 30 steps, one W7900D (same process, ABBA x3): 28.88 -> 28.15 s a
  request, 1.026x;
- **qknorm**: Qwen-Image 2.1's q / k RMSNorm and RoPE (`apply_qk_norm_rope`:
  SGLang's RMSNorm, native fp32 on ROCm, five launches, then the RoPE) as one
  `hk.ops.fused.rmsnorm_rotary_pairs` launch: 43 us instead of 1,166 us a call.
  The mean's sum is in hk's order, so a few elements differ from the engine's
  by a bf16 unit; the error against fp64 is the engine's. 1024^2, 30 steps:
  27.36 -> 26.22 s a request, 1.044x over the rope route alone. 37.9 dB from
  stock; torch's own norm with only its sum order changed is 36.5 dB from stock.
- **ditnorm**: the DiT block norms -- `ScaleResidualLayerNormScaleShift`
  (r' = r + x * gate, LayerNorm, * (1 + scale) + shift) and
  `LayerNormScaleShift` -- as one `hk.ops.norm.layernorm_scale_shift` launch
  each where SGLang would run its native chain (ROCm without FlyDSL: about ten
  passes over the activation). Each output keeps the dtype torch's chain gives
  it (Wan's fp32 gate promotes the residual and the norm to fp32). At Wan2.1's
  1x32760x1536 on a W7900D: 2.6 / 2.2 / 3.0 ms -> 0.36 / 0.89 / 0.59 ms a
  call; a Wan2.1 832x480x81 video 1.038x.

Two more are **opt-in** until an end-to-end image-edit smoke on MiniMax H3
(`HK_SGLD_H3VAE=1`, `HK_SGLD_PATCHEMBED=1`); neither is an hk kernel, both
steer the work to a cheaper exact form:

- **h3vae**: MiniMax H3's VAE convs (`BaseConv3d` of
  `sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.conv`) on a
  single frame -- every encoder conv of an image edit's reference image. The
  conv pads a one-frame input with `kt - 1` zero frames in front (causal), so
  its one output frame is a conv2d of the frame with the last time tap: exact
  up to summation order, `kt` x fewer FLOPs, and MIOpen's conv2d instead of
  its conv3d's im2col. 1.99-7.35x a conv at the encoder's 14 shapes on a
  W7900D (128 channels at 258^2 to 1024 at 18^2); weighted by an H3 edit's
  profile that is about 2.2 s -> 0.6 s an edit (an estimate, not yet
  measured end to end). More frames, replicate time padding or a pre-padded input keep the
  engine's. RDNA only: on gfx950 MIOpen's conv2d at these shapes has 1.6x
  the mean error of its conv3d, so the route leaves the engine's there.
- **patchembed**: Qwen3-VL's vision patch embedding
  (`Qwen3VLVisionPatchEmbed` of `sglang.srt.models.qwen3_vl`), whose conv3d has
  kernel == stride over each whole (C, T, P, P) patch, as the one F.linear it
  is -- SGLang's own `Conv3dLayer._forward_mulmat`, which MiniMax H3's vision
  tower turns off (`disable_linear=True`). 4096 patches on a W7900D: 538 ms of
  MIOpen conv3d -> 0.28 ms. Where SGLang already takes its linear path the
  route leaves it.

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
