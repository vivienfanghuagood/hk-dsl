#!/bin/bash
# Megatron-LM FP8 training on AMD Radeon AI PRO R9700 (gfx1201) with hk: one arm per call.
#
#   tools/megatron-rdna4/run.sh <arm> [iters]
#     arm  = tune-bf16 | tune-fp8   (one-time: hipBLASLt + TunableOp tuning for this model/shape)
#            baseline                (BF16, torch SDPA attention, LM-head wgrad fix, fused CE)   <- fair baseline
#            recipe                  (FP8 delayed HYBRID + hk attention + hk fc1 backward + LM-head wgrad fix + fused CE)
#     iters default 60 (tuning arms use 12)
#
# Environment (all optional except MEGATRON):
#   MEGATRON   path to a Megatron-LM checkout (ROCm fork, commit 5933b58; no source changes needed)
#   DATA       Megatron indexed dataset prefix (…_text_document); default: --mock-data
#   OUT        directory for logs and tuning caches (default ./megatron-rdna4-out)
#   SEED       default 1234
#   MEGATRON_EXTRA  extra pretrain_gpt.py arguments (e.g. profiler flags)
# Requires: ROCm 7.2.x PyTorch 2.10 with gfx1201, hk (this repo) importable, and a TransformerEngine build with
# gfx1201 enabled (see docs/megatron.md, "TransformerEngine").  Run on one GPU: HIP_VISIBLE_DEVICES=<idx>.
set -euo pipefail
ARM=${1:?usage: run.sh tune-bf16|tune-fp8|baseline|recipe [iters]}
ITERS=${2:-60}
: "${MEGATRON:?set MEGATRON to the Megatron-LM checkout}"
OUT=${OUT:-$PWD/megatron-rdna4-out}; SEED=${SEED:-1234}; mkdir -p "$OUT"

# ~1.28B Llama-style GPT: 16 layers, hidden 2048, FFN 8192 (SwiGLU), 16 heads, RMSNorm, RoPE, vocab 50304, seq 2048.
MODEL="--num-layers 16 --hidden-size 2048 --num-attention-heads 16 --ffn-hidden-size 8192 --seq-length 2048
 --max-position-embeddings 2048 --normalization RMSNorm --swiglu --position-embedding-type rope
 --untie-embeddings-and-output-weights --disable-bias-linear --init-method-std 0.02
 --tokenizer-type NullTokenizer --vocab-size 50303"
# Training: mbs 2 x gbs 8 (4 micro-batches), no activation recompute (hk attention keeps memory low), no dropout.
TRAIN="--micro-batch-size 2 --global-batch-size 8 --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 30
 --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 --bf16 --attention-dropout 0 --hidden-dropout 0
 --transformer-impl transformer_engine --attention-backend unfused --log-interval 10 --eval-iters 0 --eval-interval 100000 --log-throughput
 --seed $SEED"
FP8="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max
 --keep-fp8-weight-transpose-cache-te"
if [ -n "${DATA:-}" ]; then DATAARGS="--data-path $DATA --split 100,0,0 --data-cache-path $OUT/data-cache"
else DATAARGS="--mock-data"; fi

export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=0          # TE's fused attention has no gfx120x kernels; hk/SDPA replace it
export HK_MEGATRON_LMHEAD_WGRAD=1                   # LM-head wgrad: apex 41 TF -> TE hipBLASLt, both arms
export PYTORCH_TUNABLEOP_ENABLED=1                  # torch matmuls (LM head fwd/dgrad) use the tuned table
# Fused cross-entropy (both arms): TE >= 2.19 parallel_cross_entropy. The ROCm fork asserts against
# --cross-entropy-fusion-impl te unconditionally (upstream gates it on TE < 2.19); until the fork takes the
# version-conditional assert, hk's launcher routes the allowed 'native' slot to TE's kernel.
export HK_MEGATRON_TE_CE=1
CE="--cross-entropy-loss-fusion --cross-entropy-fusion-impl native"

case "$ARM" in
  tune-bf16|tune-fp8)
    P=${ARM#tune-}; X=""; [ "$P" = fp8 ] && X="$FP8"
    export TE_HIPBLASLT_TUNING_RUN_COUNT=10 TE_HIPBLASLT_TUNING_ALGO_COUNT=1024 TE_HIPBLASLT_ALGO_SAVE=$OUT/te_algos_$P.csv
    export PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=$OUT/tunableop_$P.csv
    if [ "$P" = fp8 ]; then export HK_MEGATRON_ATTN=1 HK_MEGATRON_FC1_BWD=1; else export HK_MEGATRON_ATTN=sdpa; fi
    ITERS=12; EXTRA="--lr-decay-iters 1000" ;;
  baseline)
    export HK_MEGATRON_ATTN=sdpa HK_MEGATRON_FC1_BWD=0
    export TE_HIPBLASLT_ALGO_LOAD=$OUT/te_algos_bf16.csv PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$OUT/tunableop_bf16.csv
    X=""; EXTRA="" ;;
  recipe)
    export HK_MEGATRON_ATTN=1 HK_MEGATRON_FC1_BWD=1
    export TE_HIPBLASLT_ALGO_LOAD=$OUT/te_algos_fp8.csv PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$OUT/tunableop_fp8.csv
    X="$FP8"; EXTRA="" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
if [ -n "${TE_HIPBLASLT_ALGO_LOAD:-}" ] && [ ! -f "$TE_HIPBLASLT_ALGO_LOAD" ]; then
  P=bf16; [ "$ARM" = recipe ] && P=fp8
  echo "missing $TE_HIPBLASLT_ALGO_LOAD: run 'run.sh tune-$P' first" >&2; exit 1
fi

cd "$MEGATRON"
# python -m torch.distributed.run (not the torchrun script) so the active interpreter/venv is used.
python -m torch.distributed.run --nproc_per_node 1 -m hk.integration.megatron pretrain_gpt.py \
  $MODEL $TRAIN $CE $X $DATAARGS --train-iters "$ITERS" ${EXTRA:-} ${MEGATRON_EXTRA:-} 2>&1 | tee "$OUT/$ARM-seed$SEED.log" \
  | grep -E "iteration +[0-9]+/|Error|error:"
