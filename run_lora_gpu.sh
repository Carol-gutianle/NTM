#!/usr/bin/env bash
# Launch using the dedicated shared Conda environment; no conda init required.
set -euo pipefail
PROJECT="${PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
cd "$PROJECT"
export HF_HOME="${HF_HOME:-$PROJECT/artifacts/hf-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
exec "$PYTHON" train_lora.py \
  --model Qwen/Qwen2.5-3B \
  --train train.csv \
  --output artifacts/qlora_qwen25_3b_fold0 \
  --four-bit \
  --max-length 512 \
  --batch-size 1 \
  --eval-batch-size 2 \
  --grad-accum 16 \
  --rank 16 \
  --alpha 32 \
  --lr 1e-4 \
  --epochs 5 \
  "$@"
