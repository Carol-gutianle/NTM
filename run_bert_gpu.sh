#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-$PWD/artifacts/pretrained/bert-base-cased}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "[1/2] Check local BERT; download only if missing/incomplete."
"$PYTHON" prepare_bert.py --output "$MODEL"
[[ "${DOWNLOAD_ONLY:-0}" != 1 ]] || exit 0
echo "[2/2] Offline single-GPU BERT full fine-tuning."
exec "$PYTHON" train_bert.py --pretrained "$MODEL" "$@"
