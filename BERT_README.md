# Single-GPU BERT

```bash
conda activate /apdcephfs_sgfd2/share_303378293/tianlelegu/ntl
cd /apdcephfs_sgfd2/share_303378293/tianlelegu/name_that_llm
PYTHON="$CONDA_PREFIX/bin/python" CUDA_VISIBLE_DEVICES=0 bash run_bert_gpu.sh
```

- Full fine-tuning of `google-bert/bert-base-cased`, not LoRA.
- Reuses `artifacts/pretrained/bert-base-cased` if structurally valid. If missing
  or incomplete, downloads only config/tokenizer/safetensors files, not TF/Flax.
  Download needs internet access; subsequent training is strictly offline.
- This is a local filesystem path on shared storage, not necessarily node-local SSD.
  Set `MODEL=/actual/node-local/disk/bert-base-cased` to prepare there instead.
- Existing `ntl` environment is used without injecting repository `.deps`.
- GPU is required by default, BF16, batch 32, max length 512, learning rate 2e-5,
  up to 10 epochs, early stop after 3 non-improving epoch validations.
- Same seed-42 first StratifiedGroupKFold split as the baseline and default LoRA.
  Best checkpoint selection uses validation balanced accuracy; it is not an
  independent test score. No promise that BERT beats the 32B experiment.
- New timestamped output directory on each launch; old outputs are not overwritten.
  No optimizer resume supported.

Preparation without training:

```bash
DOWNLOAD_ONLY=1 bash run_bert_gpu.sh
```

Change training options:

```bash
bash run_bert_gpu.sh --batch-size 16 --lr 3e-5 --epochs 10
```

Outputs include `model/` (best full classifier and tokenizer), `metrics.json`,
`classification_report.json`, `confusion_matrix.csv`, `validation_logits.npy`,
`validation_predictions.csv`, `history.json`, `split.csv`, `length_stats.csv`
(per-class token lengths/truncation rates), `run_summary.json`, `submission.parquet`.
The included test.csv has only three examples: its predictions are a smoke output.
Length statistics use BERT's tokenizer, not Qwen's.

## Training-set diagnostics (no retraining required)

New training runs automatically evaluate both splits with the best checkpoint in
`eval()` / inference mode and save `diagnostics/summary.json`, per-split loss,
accuracy, balanced accuracy, macro F1, class reports, confusion matrices,
predictions and logits. Training inference loss is not the dropout-on training-loop loss.
Training-set performance is diagnostic, not a generalization estimate.

For an existing run:

```bash
CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" evaluate_bert.py \
  --run artifacts/bert_gpu_20261023_143300_319726
```

This uses the saved `model/`, `split.csv`, and original maximum length / precision.
It creates a new `diagnostics_TIMESTAMP_PID/` without altering the saved model or
old metrics. It checks ID/label/prompt consistency and, for new runs, the training
CSV checksum. Old runs lack that checksum, so text identity cannot be verified.
Use `--cpu` only for an explicit CPU evaluation.

## Exploratory BERT + TF-IDF score fusion (CPU, no retraining)

```bash
"$CONDA_PREFIX/bin/python" fuse_bert_tfidf.py \
  --bert-run artifacts/bert_gpu_20261023_143300_319726 \
  --tfidf-run artifacts/tfidf_char
```

Reads saved logits/margins and trusted local TF-IDF joblib class ordering.
Checks IDs, prompts, true labels and argmax against each saved prediction CSV,
then reorders score rows/columns. Each score row is standardized across classes
and softmaxed at temperature 1, then combined at BERT weights 0/.25/.5/.75/1.
This is heuristic normalization, NOT probability calibration. Both endpoints
preserve their respective single-model predictions.

New timestamped output includes the five candidate results, selected weights,
label ordering/method, predictions, scores, class report and confusion matrix.
Selection uses this same validation set, which also selected the BERT checkpoint:
this is an exploratory tuning result, NOT an independent evaluation.
Do not report it as CV or test performance. For confirmation, freeze the recipe
and retrain both constituent models with the same prompt-disjoint split on other
folds; for rigorous selection use nested/grouped validation. This script does
not generate a test submission or deploy the fusion.

Tests: `"$CONDA_PREFIX/bin/python" test_fusion.py`

Offline inference, using the same length as training:

```bash
BASELINES_USE_LOCAL_DEPS=0 python predict.py --method bert \
  --model artifacts/YOUR_RUN/model --max-length 512 \
  --test test.csv --output artifacts/YOUR_RUN/submission.parquet
```
