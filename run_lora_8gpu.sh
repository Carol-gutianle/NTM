#!/usr/bin/env bash
# Single-node Qwen3-32B LoRA/QLoRA; only real training, no artificial GPU load.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash run_lora_8gpu.sh [train_lora.py options]
  MODEL=/local/HF/model         default: shared Qwen3-32B
  PYTHON=/env/bin/python        default: python on PATH
  MODE=qlora|lora               default: qlora (NF4)
  OUTPUT=artifacts/my_run       new directory, or previous directory for resume
  RESUME=auto|/path/checkpoint  only checkpoints with COMPLETE marker
  CUDA_VISIBLE_DEVICES=0,...,7  defaults to eight GPUs
  CHECK_ONLY=1                 model/GPU preflight, no training
  LOCAL_MODEL_CACHE=auto       default: stage on a verified local disk
  LOCAL_MODEL_CACHE=/nvme/cache  explicitly select node-local cache directory
  LOCAL_MODEL_CACHE=off        opt out (shared loading may be very slow)
  STAGE_ONLY=1                populate cache without GPU checks or training
  CACHE_VERIFY=1              rehash cached files (extra full local read)
  LOCAL_CACHE_ALLOW_UNVERIFIED=1  only if platform confirms disk locality
  GPU_MIN_UTIL=20 GPU_LOW_SECONDS=180 GPU_GRACE_SECONDS=180
  GPU_LOW_ACTION=warn|stop      warn also requests a checkpoint; no artificial load
Use MODEL/OUTPUT/RESUME/MODE environment variables rather than training flags.
Keep training options identical when resuming.
HELP
  exit 0
fi
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-/apdcephfs_sgfd2/share_303378293/hunyuan/common/Qwen3-32B}"
MODE="${MODE:-qlora}"
OUTPUT="${OUTPUT:-$PROJECT/artifacts/qwen3_32b_${MODE}_8gpu_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-$PROJECT/artifacts/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LORA_USE_LOCAL_DEPS="${LORA_USE_LOCAL_DEPS:-0}"
for arg in "$@"; do
  case "$arg" in
    --model|--model=*|--model-load-path|--model-load-path=*|--output|--output=*|--resume|--resume=*|--four-bit|--cpu)
      echo "Use MODEL/OUTPUT/RESUME/MODE environment variables, not $arg." >&2; exit 2 ;;
  esac
done
[[ "$MODE" == lora || "$MODE" == qlora ]] || { echo "MODE must be lora or qlora" >&2; exit 2; }
extra=()
[[ "$MODE" != qlora ]] || extra+=(--four-bit)
if [[ -n "${RESUME:-}" ]]; then
  checkpoint="$RESUME"
  if [[ "$RESUME" == auto ]]; then
    checkpoint="$("$PYTHON" - "$OUTPUT" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
items = [p for p in root.glob("checkpoint-*") if p.name[11:].isdigit()
         and (p / "COMPLETE").is_file() and (p / "trainer_state.json").is_file()]
if not items:
    sys.exit("No complete checkpoint found; inspect the output directory.")
print(max(items, key=lambda p: int(p.name[11:])))
PY
)"
  fi
  [[ -f "$checkpoint/COMPLETE" ]] || { echo "Incomplete checkpoint: $checkpoint" >&2; exit 2; }
  extra+=(--resume "$checkpoint")
elif [[ -d "$OUTPUT" && -n "$(ls -A -- "$OUTPUT")" ]]; then
  echo "Output is nonempty; use RESUME=auto or a new OUTPUT: $OUTPUT" >&2
  exit 2
fi
[[ ! -e "$OUTPUT/STOP_REQUEST" ]] || { echo "Remove $OUTPUT/STOP_REQUEST before resume" >&2; exit 2; }
if [[ "${STAGE_ONLY:-0}" != 1 ]]; then
  "$PYTHON" "$PROJECT/preflight_lora.py" --model "$MODEL" --mode "$MODE" --gpus 8
  [[ "${CHECK_ONLY:-0}" != 1 ]] || exit 0
fi
LOAD_MODEL="$MODEL"
if [[ "${LOCAL_MODEL_CACHE:-auto}" != off ]]; then
  cache_options=()
  [[ "${CACHE_VERIFY:-0}" != 1 ]] || cache_options+=(--verify)
  [[ "${LOCAL_CACHE_ALLOW_UNVERIFIED:-0}" != 1 ]] || cache_options+=(--allow-unverified-fs)
  LOAD_MODEL="$("$PYTHON" "$PROJECT/stage_lora_model.py" \
    --source "$MODEL" --cache-root "${LOCAL_MODEL_CACHE:-auto}" \
    --reserve-gib "${CACHE_RESERVE_GIB:-8}" "${cache_options[@]}")"
else
  echo "WARNING: local staging disabled; all eight ranks load directly from $MODEL" >&2
fi
[[ "${STAGE_ONLY:-0}" != 1 ]] || exit 0
mkdir -p -- "$(dirname -- "$OUTPUT")"
LOG="${OUTPUT%/}.launch.$(date +%Y%m%d_%H%M%S).log"
echo "MODEL=$MODEL LOAD_MODEL=$LOAD_MODEL MODE=$MODE OUTPUT=$OUTPUT"
echo "Log: $LOG"
echo "Cooperative stop: touch '$OUTPUT/STOP_REQUEST'"
set +e
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  --max-restarts=0 "$PROJECT/train_lora.py" \
  --model "$MODEL" --model-load-path "$LOAD_MODEL" --output "$OUTPUT" --offline \
  --train "$PROJECT/train.csv" --test "$PROJECT/test.csv" \
  --max-length 512 --batch-size 1 --eval-batch-size 1 --grad-accum 2 \
  --rank 16 --alpha 32 --lr 1e-4 --epochs 5 \
  --workers 2 --threads "$OMP_NUM_THREADS" --save-steps 25 \
  --monitor-gpu --gpu-min-util "${GPU_MIN_UTIL:-20}" \
  --gpu-low-seconds "${GPU_LOW_SECONDS:-180}" \
  --gpu-grace-seconds "${GPU_GRACE_SECONDS:-180}" \
  --gpu-low-action "${GPU_LOW_ACTION:-warn}" \
  "${extra[@]}" "$@" 2>&1 | tee "$LOG"
statuses=("${PIPESTATUS[@]}")
set -e
if [[ "${statuses[0]}" != 0 ]]; then
  echo "Training stopped/failed (${statuses[0]}). See $LOG; resume a COMPLETE checkpoint."
  exit "${statuses[0]}"
fi
[[ "${statuses[1]}" == 0 ]] || { echo "Log write failed" >&2; exit "${statuses[1]}"; }
[[ -f "$OUTPUT/SUCCESS.json" ]] || { echo "Missing SUCCESS.json; outputs may be incomplete" >&2; exit 1; }
echo "Done: $OUTPUT/adapter, $OUTPUT/validation_metrics.json (unless --full-train),"
echo "      $OUTPUT/submission.parquet (unless --submission overrides it)"
