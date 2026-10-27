"""Character TF-IDF + XGBoost; saved prompt-disjoint split, no Kaggle submission."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if (ROOT / ".xgb-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".xgb-deps"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=ROOT / "train.csv")
    p.add_argument("--test", type=Path, default=ROOT / "test.csv")
    p.add_argument("--split", type=Path, default=ROOT /
                   "artifacts/bert_gpu_20261023_143300_319726/split.csv")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" /
                   f"xgboost_char_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--trees", type=int, default=400, help="Boosting rounds; up to 18 trees/round")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-features", type=int, default=30000)
    p.add_argument("--lr", type=float, default=.05)
    args = p.parse_args()
    if min(args.threads, args.trees, args.patience, args.max_depth, args.max_features) < 1 or args.lr <= 0:
        p.error("Training parameters must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output is nonempty; choose a new directory")
    print("Importing XGBoost and text dependencies...", flush=True)
    import joblib
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from threadpoolctl import threadpool_limits
    from bert_diagnostics import load_split, save_diagnostics
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            p.error("CUDA requested but unavailable; explicitly use --device cpu")
    frames = load_split(args.train, args.split)
    labels = sorted(frames["train"].model.unique())
    label2id = {s: i for i, s in enumerate(labels)}
    y = {k: np.array([label2id[s] for s in f.model]) for k, f in frames.items()}
    start = time.monotonic()
    # Feature cap differs from original SVM: intended as a cheap first experiment.
    # Fit vocabulary and IDF on training texts ONLY.
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=False, min_df=2,
        sublinear_tf=True, max_features=args.max_features, dtype=np.float32)
    with threadpool_limits(limits=args.threads):
        x_train = vectorizer.fit_transform(frames["train"].text).tocsr()
        x_val = vectorizer.transform(frames["validation"].text).tocsr()
    print(f"train={x_train.shape}, validation={x_val.shape}, device={args.device}", flush=True)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(labels), tree_method="hist",
        device=args.device, n_jobs=args.threads, n_estimators=args.trees,
        max_depth=args.max_depth, learning_rate=args.lr, min_child_weight=5,
        subsample=.8, colsample_bytree=.5, reg_lambda=5, max_bin=64,
        eval_metric="mlogloss", early_stopping_rounds=args.patience, random_state=42)
    model.fit(x_train, y["train"], eval_set=[(x_val, y["validation"])], verbose=25)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output / "model.ubj")
    joblib.dump(vectorizer, args.output / "vectorizer.joblib")
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "labels": labels, "xgboost_version": xgb.__version__,
        "features": x_train.shape[1], "best_iteration": model.best_iteration,
        "best_validation_mlogloss": float(model.best_score),
        "sha256_train_csv": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "note": "Same saved prompt-disjoint split; vocabulary/IDF fitted on train only. "
                "30k feature cap by default differs from SVM's 200k cap. "
                "Best round selected by validation mlogloss, not accuracy. "
                "Sparse absent entries treated as missing by tree booster. Not independent test.",
    }
    # Native Booster requires explicit best-iteration range for portable inference.
    logits = {k: model.predict(x, output_margin=True)
              for k, x in [("train", x_train), ("validation", x_val)]}
    summary = save_diagnostics(args.output / "diagnostics", frames, logits, labels, config)
    # Match existing baseline filenames for downstream comparison/fusion.
    val = frames["validation"][["id", "prompt_id", "model"]].copy()
    val["prediction"] = np.array(labels)[logits["validation"].argmax(1)]
    val.to_csv(args.output / "validation_predictions.csv", index=False)
    np.save(args.output / "validation_logits.npy", logits["validation"])
    (args.output / "metrics.json").write_text(json.dumps(summary["validation"], indent=2))
    split = pd.concat([f[["id", "prompt_id", "model"]].assign(split=k)
                       for k, f in frames.items()], ignore_index=True)
    split.to_csv(args.output / "split.csv", index=False)
    (args.output / "history.json").write_text(json.dumps(model.evals_result(), indent=2))
    test = pd.read_csv(args.test, dtype={"id": str})
    if not len(test) or not test.id.is_unique or test[["id", "text"]].isna().any().any():
        raise ValueError("Invalid test data")
    pred = model.predict(vectorizer.transform(test.text).tocsr()).astype(int)
    pd.DataFrame({"id": test.id, "model": np.array(labels)[pred]}).to_parquet(
        args.output / "submission.parquet", index=False)
    config["elapsed_seconds"] = time.monotonic() - start
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Done: {args.output}; seconds={config['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
