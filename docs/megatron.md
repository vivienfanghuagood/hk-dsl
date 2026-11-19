# hk in Megatron-LM training on RDNA4 (gfx1201)

**Draft.** Lines marked TBD wait for the recipe's final A/B.

hk speeds up FP8 training with Megatron-LM and Transformer Engine on Radeon AI PRO R9700 / RX 9070 (gfx1201). It
replaces two of the step's kernels, and the rest is configuration. Megatron's and TE's sources are not changed: one
launcher installs hk's pieces and then runs `pretrain_gpt.py`.

| what | how | switch |
|---|---|---|
| attention, forward and backward | `hk.ops.attn_bwd.attention` in place of TE's unfused attention (gfx1201 has no fused one) | `HK_MEGATRON_ATTN` |
| the MLP's fc1 backward (FP8) | SwiGLU backward and the fp8 cast as one kernel, then fc1's dgrad and wgrad on hk's fp8 GEMM | `HK_MEGATRON_FC1_BWD` |
| the output layer's weight gradient | TE's GEMM instead of apex's (configuration) | `HK_MEGATRON_LMHEAD_WGRAD` |
| cross-entropy | TE's fused kernel: `--cross-entropy-loss-fusion --cross-entropy-fusion-impl te` (configuration) | Megatron flag |

## Result

TBD (final A/B): 1.28B Llama-style model, one R9700. Columns: BF16 + PyTorch SDPA (best stock: tuned GEMMs, TE fused
cross-entropy) against FP8 + hk, ms per iteration and tokens/s, plus loss agreement over N steps on real text.

## Requirements

- A gfx1201 GPU, ROCm 7.2 or newer, PyTorch 2.10 for ROCm. The image the numbers below come from is
  `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`.
- Transformer Engine for ROCm, built for gfx1201. TBD: stock ROCm TE does not yet enable gfx1201; the build steps go
  here once that is settled.
- Megatron-LM for ROCm at TBD (commit).
- `pip install hk-dsl` (TBD: the first release with `hk.integration.megatron`).

## Run

```bash
cd Megatron-LM
python3 -m hk.integration --status          # the megatron_* routes this GPU will get
HK_MEGATRON_ATTN=1 HK_MEGATRON_FC1_BWD=1 HK_MEGATRON_LMHEAD_WGRAD=1 \
torchrun --nproc-per-node 1 -m hk.integration.megatron pretrain_gpt.py \
    TBD: model and training flags \
    --fp8-format hybrid --fp8-recipe delayed \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl te \
    --disable-bias-linear --swiglu --attention-dropout 0 ...
```

Each piece is off until its variable is set, so a piece can be turned on, or left off, on its own.
`HK_MEGATRON_ATTN=sdpa` puts torch's SDPA in Megatron's core attention instead: the best stock attention, which is the
baseline arm. `HK_PATCH_MEGATRON=0` turns all of them off.

## Tune the GEMMs first

TBD: TE's hipBLASLt algorithm search (`TE_HIPBLASLT_TUNING_*`, 1024 candidates) and TunableOp for torch's matmuls,
how to save and reload the results, and how to check that they loaded. The baseline is measured with the same tuning.

## What each piece does, and its limits

- **Attention.** Supports causal or unmasked self-attention, no dropout, bf16, head_dim 64 or 128, and GQA.
  - Megatron's SBHD tensors are read in place at micro-batch 1.
  - It does not support context parallelism, attention bias, or packed sequences. A model that needs one of these
    stops with an error naming the switch to turn off.
  - The backward is deterministic.
  - Measured ceiling and what was not pursued: `tools/attn-bwd-gfx1201/README.md`.
- **fc1 backward.** Requires FP8 with delayed scaling, SwiGLU, and `--disable-bias-linear`.
  - The fp8 gradient is written with padded rows. On gfx1201, rows a power of two of bytes apart share a memory
    channel, which halves the GEMMs that read them.
  - Only hk's GEMM reads those rows. Every other GEMM stays on TE, and a padded tensor headed for TE raises an error.
- **Not covered:** tensor and pipeline parallel runs at scale (TBD: what was tested), MXFP8, current scaling,
  gfx1100.

## Measured and not taken

- **An hk cross-entropy kernel.** hk has one (two passes over the bf16 logits, fp32 log-sum-exp and gradient, the
  target entry's gradient in fp32; branch `claude/a-fused-ce`, not shipped). Against fp64, on one R9700 at
  (4096 x 50304), fwd+bwd:

  | | loss rel. error | grad rel. L2 | target grads zeroed (p_t 0.998-0.99999) | ms |
  |---|---|---|---|---|
  | Megatron, unfused (`vocab_parallel_cross_entropy`) | 7.8e-8 | 1.7e-3 | 0 | 19.9 |
  | Transformer Engine 2.21, fused | 7.4e-8 | 1.7e-3 | 0 | 5.08 |
  | hk | 7.4e-8 | 1.7e-3 | 0 | 4.31 |

  TE's fused kernel is already close to the bandwidth floor (about 2 ms), and hk would save about 3 ms an
  iteration, so the recipe uses TE's. Megatron-LM rejects TE's fused cross-entropy as unstable ("stability
  issues"). That covers TE before 2.19, which kept the gradient in bf16 and zeroed a near-certain target's
  (TransformerEngine #3193); 2.21 passes the same test. TBD: how the recipe selects it.
- **Attention backward beyond today's.** Measured ceiling and the levers left: `tools/attn-bwd-gfx1201/README.md`.

## Reproduce the A/B

TBD: `tools/megatron-rdna4/run.sh` (one command per arm, mock or real data, the same tuning), and the expected output.
