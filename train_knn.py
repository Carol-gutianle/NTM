"""Frozen original grouped split: character TF-IDF + cosine KNN."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" /
                   f"knn_char_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output nonempty")
    if args.threads < 1:
        p.error("threads must be positive")
    print("Importing KNN dependencies...", flush=True)
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
    from threadpoolctl import threadpool_limits
    from bert_diagnostics import load_split
    split = ROOT / "artifacts/bert_gpu_20261023_143300_319726/split.csv"
    frames = load_split(ROOT / "train.csv", split)
    train, val = frames["train"], frames["validation"]
    source = joblib.load(ROOT / "artifacts/tfidf_char/model.joblib")
    # Clone unfitted features and refit on this run's train partition only.
    features = clone(source.named_steps["features"])
    start = time.monotonic()
    with threadpool_limits(limits=args.threads):
        x = features.fit_transform(train.text).tocsr()
        xv = features.transform(val.text).tocsr()
        old = source.named_steps["features"].transform(val.text).tocsr()
        diff = xv - old
        if diff.nnz and abs(diff.data).max() > 1e-6:
            raise RuntimeError("Refitted TF-IDF differs from baseline features")
        del source, old
        args.output.mkdir(parents=True, exist_ok=True)
        results, best = [], None
        for k in [1, 3, 5, 9, 15]:
            for weights in ["uniform", "distance"]:
                model = KNeighborsClassifier(n_neighbors=k, weights=weights,
                                             metric="cosine", algorithm="brute",
                                             n_jobs=args.threads)
                model.fit(x, train.model)
                prob = model.predict_proba(xv)
                pred = model.classes_[prob.argmax(1)]
                report = classification_report(val.model, pred, labels=model.classes_,
                                               output_dict=True, zero_division=0)
                row = {"k": k, "weights": weights,
                       "accuracy": float(accuracy_score(val.model, pred)),
                       "balanced_accuracy": float(balanced_accuracy_score(val.model, pred)),
                       "macro_f1": report["macro avg"]["f1-score"]}
                results.append(row)
                print(json.dumps(row), flush=True)
                if best is None or row["balanced_accuracy"] > best[0]["balanced_accuracy"]:
                    best = row, model, prob, pred, report
        row, model, prob, pred, report = best
        # Do NOT report resubstitution accuracy: K=1 would retrieve each row itself.
        # Validation queries never retrieve validation examples.
        distances, indices = model.kneighbors(xv, n_neighbors=15)
        records = []
        for i, (_, query) in enumerate(val.iterrows()):
            for rank, (dist, j) in enumerate(zip(distances[i], indices[i]), 1):
                neighbor = train.iloc[j]
                records.append({
                    "id": query.id, "prompt_id": query.prompt_id, "model": query.model,
                    "prediction": pred[i], "correct": bool(pred[i] == query.model),
                    "query_text": query.text, "neighbor_rank": rank,
                    "neighbor_id": neighbor.id, "neighbor_prompt_id": neighbor.prompt_id,
                    "neighbor_model": neighbor.model, "cosine_distance": float(dist),
                    "neighbor_text": neighbor.text,
                })
        pd.DataFrame(records).to_csv(args.output / "validation_neighbors.csv", index=False)
        pd.DataFrame(results).to_csv(args.output / "search_results.csv", index=False)
        predictions = val[["id", "prompt_id", "model"]].copy()
        predictions["prediction"] = pred
        predictions.to_csv(args.output / "validation_predictions.csv", index=False)
        np.save(args.output / "validation_probabilities.npy", prob)
        pd.DataFrame(confusion_matrix(val.model, pred, labels=model.classes_),
                     index=model.classes_, columns=model.classes_).to_csv(
                         args.output / "confusion_matrix.csv")
        (args.output / "classification_report.json").write_text(json.dumps(report, indent=2))
        joblib.dump({"features": features, "classifier": model}, args.output / "model.joblib")
        # Check serialization on held-out queries without refitting.
        loaded = joblib.load(args.output / "model.joblib")
        np.testing.assert_allclose(
            loaded["classifier"].predict_proba(loaded["features"].transform(val.text.iloc[:10])),
            prob[:10], rtol=1e-6, atol=1e-7)
        test = pd.read_csv(ROOT / "test.csv", dtype={"id": str})
        test_pred = model.predict(features.transform(test.text))
        pd.DataFrame({"id": test.id, "model": test_pred}).to_parquet(
            args.output / "submission.parquet", index=False)
    manifest = {
        "best": row, "labels": model.classes_.tolist(),
        "train_rows": len(train), "validation_rows": len(val), "features": x.shape[1],
        "split": str(split), "train_sha256": hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic()-start,
        "reload_check": "passed",
        "note": "Original data only; same grouped split and refitted train-only character TF-IDF. "
                "Select among 10 candidates on reused validation data, not independent test/CV. "
                "Class-score ties use sorted class order; neighbor boundary ties can depend on "
                "training row order. No misleading self-neighbor train accuracy reported. "
                "Local test predictions are only a smoke output, no Kaggle submission.",
    }
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
