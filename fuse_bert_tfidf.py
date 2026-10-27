"""Exploratory validation score fusion. No training, GPU, or calibrated probabilities."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from compare_predictions import read_predictions, compare


def normalized_scores(scores):
    """Per-row z-score then softmax; deterministic heuristic, NOT calibration."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Expected finite 2D scores")
    z = (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-12)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def align_scores(bert_rows, tfidf_rows, bert_scores, tfidf_scores, bert_labels, tfidf_labels):
    compare(bert_rows, tfidf_rows)  # Exact ID set, true label and prompt match.
    if len(set(bert_labels)) != len(bert_labels) or set(bert_labels) != set(tfidf_labels):
        raise ValueError("Class sets differ")
    for name, rows, scores, labels in [
        ("bert", bert_rows, bert_scores, bert_labels),
        ("tfidf", tfidf_rows, tfidf_scores, tfidf_labels),
    ]:
        if scores.shape != (len(rows), len(labels)) or not np.isfinite(scores).all():
            raise ValueError(f"Bad {name} score shape/values")
        pred = np.array(labels)[scores.argmax(axis=1)].tolist()
        if pred != [r["prediction"] for r in rows.values()]:
            raise ValueError(f"{name} score argmax differs from saved predictions; check row/class order")
    position = {uid: i for i, uid in enumerate(tfidf_rows)}
    row_indices = [position[uid] for uid in bert_rows]
    class_indices = [tfidf_labels.index(label) for label in bert_labels]
    return bert_scores, tfidf_scores[np.ix_(row_indices, class_indices)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bert-run", type=Path, required=True)
    p.add_argument("--tfidf-run", type=Path, default=Path("artifacts/tfidf_char"))
    p.add_argument("--output", type=Path, default=Path("artifacts") /
                   f"bert_tfidf_fusion_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output nonempty; use a new directory")
    config = json.loads((args.bert_run / "model/config.json").read_text())
    labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
    # Only load trusted, locally produced joblib artifacts (pickle is executable).
    import joblib
    tfidf = joblib.load(args.tfidf_run / "model.joblib")
    tfidf_labels = tfidf.classes_.tolist()
    del tfidf
    a = read_predictions(args.bert_run / "validation_predictions.csv")
    b = read_predictions(args.tfidf_run / "validation_predictions.csv")
    bs, ts = align_scores(
        a, b, np.load(args.bert_run / "validation_logits.npy", allow_pickle=False),
        np.load(args.tfidf_run / "validation_scores.npy", allow_pickle=False),
        labels, tfidf_labels)
    bp, tp = normalized_scores(bs), normalized_scores(ts)
    frame = pd.DataFrame(list(a.values()))[["id", "prompt_id", "model"]]
    truth = frame.model.to_numpy()
    experiments, mixtures = [], []
    for alpha in [0., .25, .5, .75, 1.]:
        scores = alpha * bp + (1 - alpha) * tp
        pred = np.array(labels)[scores.argmax(axis=1)]
        report = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
        row = dict(bert_weight=alpha, tfidf_weight=1-alpha,
                   accuracy=float(accuracy_score(truth, pred)),
                   balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                   macro_f1=report["macro avg"]["f1-score"])
        experiments.append(row)
        mixtures.append(scores)
        print(f"BERT={alpha:.2f}, TF-IDF={1-alpha:.2f}: "
              f"accuracy={row['accuracy']:.4%}, macro_F1={row['macro_f1']:.4%}", flush=True)
    # Ties keep the first candidate, i.e. the smaller BERT weight.
    winner = max(range(len(experiments)), key=lambda i: experiments[i]["balanced_accuracy"])
    pred = np.array(labels)[mixtures[winner].argmax(axis=1)]
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(experiments).to_csv(args.output / "weight_search.csv", index=False)
    frame["prediction"] = pred
    frame.to_csv(args.output / "validation_predictions.csv", index=False)
    np.save(args.output / "validation_fused_scores.npy", mixtures[winner])
    pd.DataFrame(confusion_matrix(truth, pred, labels=labels), index=labels, columns=labels
                 ).to_csv(args.output / "confusion_matrix.csv")
    report = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
    (args.output / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "best": experiments[winner], "rows": len(frame), "labels": labels,
        "method": "Per model and sample: z-score across classes (std floor 1e-12), "
                  "then softmax T=1; weighted average; argmax.",
        "bert_run": str(args.bert_run.resolve()), "tfidf_run": str(args.tfidf_run.resolve()),
        "note": "Exploratory selection on the SAME validation set used to select the BERT "
                "checkpoint and earlier baselines. Not independent/CV/test accuracy. "
                "Normalized scores are NOT calibrated probabilities. No test submission generated.",
    }
    (args.output / "fusion_config.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["best"], indent=2))
    print(summary["note"])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
