# Baseline results

> Historical baseline report (2026-10-22). For later BERT, Qwen LoRA,
> ensembles, augmentation and the complete snapshot, see `results/README.md`.
> Paths below describe the original machine. Local sample predictions are not
> hidden-test leaderboard results.

## Validation protocol

- 2,699 labeled examples; 150 prompts; 18 classes.
- First fold of `StratifiedGroupKFold(5, shuffle=True, random_state=42)`.
- 2,159 training examples (120 prompts); 540 validation examples (30 prompts).
- No prompt overlap. Only `text` is an input feature.
- This is a single holdout experiment, not a five-fold average or Kaggle score.
- All saved models were trained on the training split only, not the full dataset.

## Results

| Method | Balanced accuracy | Training + validation |
|---|---:|---:|
| Character TF-IDF + LinearSVC | 31.85% | 6.55 s |
| Word TF-IDF + LinearSVC | 26.85% | 0.94 s |
| Character + word TF-IDF + LinearSVC | 30.37% | 7.00 s |
| BERT base cased, best epoch (4) | 24.26% | 54.03 min for all 5 epochs |

BERT validation balanced accuracy by epoch: 5.56%, 12.78%, 19.26%, 24.26%, 24.07%.
BERT used CPU, 8 threads, BF16 autocast, batch size 16, learning rate 2e-5,
AdamW, 10% warmup, and a 256-token limit. This truncates 15.33% of training
examples and 13.33% of validation examples. This first configuration does not
establish that BERT architectures generally underperform TF-IDF.

## Reproduce

### Character TF-IDF: completed five-fold grouped validation

Parameters were kept unchanged from the initial character baseline. Each fold
holds out 30 entire prompts; vocabulary and IDF are fitted only on that fold's
training partition. Every example has exactly one out-of-fold prediction.

| Fold | Balanced accuracy |
|---|---:|
| 1 | 31.85% |
| 2 | 32.96% |
| 3 | 32.78% |
| 4 | 31.12% |
| 5 | 28.15% |

- Mean: **31.37%**; fold population standard deviation: **1.74 percentage points**.
- Pooled out-of-fold balanced accuracy: **31.37%**.
- Total runtime including artifact saving: **41.7 seconds**.
- Fold 1 predictions exactly reproduce the initial holdout result.
- The character configuration was selected using fold 1, so this is not an
  untouched final evaluation. BERT has only been evaluated on fold 1.
- No full-data refit or Kaggle submission has been performed.

Run with `python baselines.py tfidf-cv --threads 4` after the initial TF-IDF run.
This clones the saved character pipeline configuration, discards its fitted
state, and refits each fold from scratch. Results, models, fold assignments and
OOF predictions/scores are under `artifacts/tfidf_char_cv/`.

The Python environment used locally is:

```bash
PY=/data/home/tianlelegu/miniconda3/envs/vlm/bin/python
$PY baselines.py tfidf --threads 4
$PY baselines.py bert --threads 8 --max-length 256 --batch-size 16 --epochs 5 --bf16
```

The scripts prepend the project's `.deps` directory when present.
Dependencies are listed in `requirements-baselines.txt`. The run used
PyTorch 2.13.0+cu130 but no GPU was visible. Pretrained BERT files must already
be available at `artifacts/pretrained/bert-base-cased` (or use `--pretrained`).
Training and inference use local-only model loading.

## Saved artifacts and offline inference

- `artifacts/split.csv`: exact train/validation membership.
- `artifacts/split_metadata.json`: source data SHA256, labels, split details.
- `artifacts/tfidf_{char,word,combined}/`: model, metrics, confusion matrix,
  per-class report, validation predictions/scores, sample submission.
- `artifacts/bert/`: best model/tokenizer, configuration, per-epoch history,
  metrics, confusion matrix, validation predictions/logits, sample submission.

```bash
$PY predict.py --method tfidf --model artifacts/tfidf_char/model.joblib \
  --output artifacts/local_tfidf_submission.parquet
$PY predict.py --method bert --model artifacts/bert/model \
  --max-length 256 --output artifacts/local_bert_submission.parquet
```

`predict.py` defaults to the competition test path when present, otherwise the
local `test.csv`. Use `--test` and `--output` to override paths.
For Kaggle, attach only the needed model files, inference code and offline
dependencies. Never upload `.env` or API credentials.

The local test file contains only three examples. Generated Parquet files are
smoke-test outputs, not predictions for the hidden test set. No Kaggle submission
has been made. Before a competitive submission, validate more folds and refit
the selected configuration on all training data.
