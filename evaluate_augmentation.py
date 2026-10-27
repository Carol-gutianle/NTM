"""Frozen-split controlled SVM comparison: original vs original + external candidates.

No automatic Kaggle submission, no changes to original data, no pseudo-labeling.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--augmentation", type=Path, default=ROOT /
                   "artifacts/augmentation200/augmentation_candidates.csv")
    p.add_argument("--split", type=Path, default=ROOT /
                   "artifacts/bert_gpu_20261023_143300_319726/split.csv")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" /
                   f"augmentation_eval_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output nonempty")
    print("Importing evaluation dependencies...", flush=True)
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import normalize
    from threadpoolctl import threadpool_limits
    from bert_diagnostics import load_split
    frames = load_split(ROOT / "train.csv", args.split)
    original, val = frames["train"], frames["validation"]
    aug = pd.read_csv(args.augmentation, dtype={"id": str, "prompt_id": str})
    labels = sorted(original.model.unique())
    required = ["id", "prompt_id", "model", "text"]
    if not len(aug) or aug[required].isna().any().any() or not aug.id.is_unique:
        p.error("Empty/invalid augmentation")
    if not set(aug.model) <= set(labels):
        p.error("Unknown labels")
    if set(aug.id) & set(pd.concat(frames.values()).id):
        p.error("Augmentation IDs overlap original")
    if set(aug.prompt_id) & set(pd.concat(frames.values()).prompt_id):
        p.error("Augmentation prompt IDs overlap original")
    # Exact and near-duplicate text checks against the held-out answers.
    # An unsupervised lexical screen does NOT prove absence of semantic leakage.
    canonical = lambda text: " ".join(text.lower().split())
    existing = {canonical(t) for f in frames.values() for t in f.text}
    drop_exact = aug.text.map(canonical).isin(existing) | aug.text.map(canonical).duplicated()
    aug = aug.loc[~drop_exact].reset_index(drop=True)
    if not len(aug):
        p.error("All candidates are exact duplicates")
    screen = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), max_features=50000,
                            min_df=1, lowercase=True, dtype=np.float32)
    with threadpool_limits(limits=args.threads):
        mat = screen.fit_transform(aug.text.tolist() + val.text.tolist())
        sim = mat[:len(aug)] @ mat[len(aug):].T
        max_sim = sim.max(axis=1).toarray().ravel()
    near = max_sim >= .9
    screened = aug.loc[near, ["id", "prompt_id", "model"]].copy()
    screened["max_validation_text_cosine"] = max_sim[near]
    aug = aug.loc[~near].reset_index(drop=True)
    if not len(aug):
        p.error("No candidates remain after duplicate screening")
    args.output.mkdir(parents=True, exist_ok=True)
    screened.to_csv(args.output / "excluded_near_duplicates.csv", index=False)
    aug.to_csv(args.output / "augmentation_used.csv", index=False)
    results = {}
    # Same hyperparameters as the original character SVM. New vocabulary/IDF
    # fitted independently on each experiment's training data.
    start = time.monotonic()
    for name, training in [
        ("original", original),
        ("augmented", pd.concat([original, aug], ignore_index=True)),
    ]:
        out = args.output / name
        out.mkdir()
        pipe = Pipeline([
            ("features", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), lowercase=False, min_df=2,
                sublinear_tf=True, max_features=200000, dtype=np.float32)),
            ("classifier", LinearSVC(C=1., random_state=42, max_iter=5000)),
        ])
        with threadpoolctl_context(args.threads):
            pipe.fit(training.text, training.model)
            scores = pipe.decision_function(val.text)
            pred = pipe.classes_[scores.argmax(axis=1)]
            train_pred = pipe.predict(training.text)
        if name == "original":
            reference = pd.read_csv(ROOT / "artifacts/tfidf_char/validation_predictions.csv", dtype={"id": str})
            reference = reference.set_index("id").loc[val.id]
            if reference.prediction.tolist() != pred.tolist():
                raise RuntimeError("Refitted baseline differs from previous predictions; investigate before comparing")
        result = {
            "training_rows": len(training),
            "training_class_counts": training.model.value_counts().sort_index().to_dict(),
            "validation_rows": len(val),
            "train_accuracy": float(accuracy_score(training.model, train_pred)),
            "accuracy": float(accuracy_score(val.model, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(val.model, pred)),
            "features": len(pipe.named_steps["features"].vocabulary_),
        }
        report = classification_report(val.model, pred, labels=labels, output_dict=True, zero_division=0)
        result["macro_f1"] = report["macro avg"]["f1-score"]
        results[name] = result
        joblib.dump(pipe, out / "model.joblib")
        np.save(out / "validation_scores.npy", scores)
        predictions = val[["id", "prompt_id", "model"]].copy()
        predictions["prediction"] = pred
        predictions.to_csv(out / "validation_predictions.csv", index=False)
        pd.DataFrame(confusion_matrix(val.model, pred, labels=labels), index=labels, columns=labels
                     ).to_csv(out / "confusion_matrix.csv")
        for filename, value in [("metrics.json", result), ("classification_report.json", report)]:
            (out / filename).write_text(json.dumps(value, indent=2))
        print(name, json.dumps(result), flush=True)
    # Paired counts provide more information than only a delta on 540 samples.
    old = pd.read_csv(args.output / "original/validation_predictions.csv")
    new = pd.read_csv(args.output / "augmented/validation_predictions.csv")
    before = old.model == old.prediction
    after = new.model == new.prediction
    summary = {
        "results": results,
        "accuracy_delta": results["augmented"]["accuracy"]-results["original"]["accuracy"],
        "fixed_wrong_to_right": int((~before & after).sum()),
        "regressed_right_to_wrong": int((before & ~after).sum()),
        "augmentation_rows_used": len(aug),
        "augmentation_counts": aug.model.value_counts().sort_index().to_dict(),
        "excluded_exact_duplicates": int(drop_exact.sum()),
        "excluded_near_duplicates": int(near.sum()),
        "missing_augmented_classes": sorted(set(labels) - set(aug.model)),
        "input_csv_sha256": hashlib.sha256(args.augmentation.read_bytes()).hexdigest(),
        "train_csv_sha256": hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic()-start,
        "note": "Same frozen validation prompts, same SVM settings, no external validation "
                "labels used for fitting. Validation has been reused in earlier experiments. "
                "Candidate filtering uses unsupervised lexical similarity only; no guarantee "
                "against semantic overlap or incorrect gateway provenance. Missing classes "
                "mean unequal augmentation; no class reweighting in this first comparison. "
                "Not Kaggle score or independent test. No submission.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {args.output}", flush=True)


def threadpoolctl_context(threads):
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=threads)


if __name__ == "__main__":
    main()
