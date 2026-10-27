# Hard-vote validation ensembles

No dependencies beyond Python's standard library, no training, no Kaggle submission.

```bash
python ensemble_vote.py
python test_ensemble_vote.py
```

Fixed groups:
- vote3: character SVM, Qwen LoRA, BERT.
- vote5: vote3 + XGBoost + TF-IDF MLP.
- vote7: vote5 + word SVM + combined character/word SVM.

One vote per classifier. Highest count wins (plurality). In multiclass tasks
even an odd number of voters can tie or lack a strict >50% majority.
Ties are resolved only among tied winning labels, using the listed member order.
That order is fixed, not chosen using the true label of each sample.
Existing BERT/SVM score fusion is NOT counted as an extra independent voter.

Inputs must have identical ID sets, prompt IDs and ground-truth labels.
Rows are aligned by ID. Input checksums and voting configurations are saved.
Per-group outputs contain predictions/member votes, tie and majority indicators,
metrics, class report and confusion matrix.

Custom predictions (also supports identically formatted future validation folds):

```bash
python ensemble_vote.py \
  --member first=/path/first/validation_predictions.csv \
  --member second=/path/second/validation_predictions.csv \
  --member third=/path/third/validation_predictions.csv \
  --priority first,second,third
```

This tool evaluates saved labeled predictions, not new/unlabeled test texts.
Deployment still requires running the constituent models and applying the saved
`hard_vote` rule. It does not upload models or submit a competition entry.

## Initial experiment, 2026-10-23

Same 540-sample prompt-disjoint validation set:

| Ensemble | Accuracy | Macro F1 | Tied samples |
|---|---:|---:|---:|
| 3 voters | 34.63% | 33.57% | 235 |
| 5 voters | 34.44% | 33.77% | 126 |
| 7 voters | 34.26% | 33.35% | 88 |

Saved under `artifacts/multi_vote_20261023_163628_2`.
None beats the previous BERT/SVM normalized-score fusion at 35.74% on this
validation set. These are exploratory comparisons on already reused validation
data, not independent test/CV scores. The SVM variants are correlated voters.
