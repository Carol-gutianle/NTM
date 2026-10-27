"""Offline inference for train_xgboost.py output; respects best boosting iteration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / ".xgb-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".xgb-deps"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--test", type=Path, default=ROOT / "test.csv")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import joblib
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    config = json.loads((args.model_dir / "config.json").read_text())
    vectorizer = joblib.load(args.model_dir / "vectorizer.joblib")
    booster = xgb.Booster()
    booster.load_model(args.model_dir / "model.ubj")
    booster.set_param({"device": "cpu", "nthread": 4})
    test = pd.read_csv(args.test, dtype={"id": str})
    if not len(test) or not test.id.is_unique or test[["id", "text"]].isna().any().any():
        raise ValueError("Invalid test data")
    x = vectorizer.transform(test.text).tocsr()
    scores = booster.predict(xgb.DMatrix(x), iteration_range=(0, config["best_iteration"] + 1))
    pred = np.array(config["labels"])[scores.argmax(1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test.id, "model": pred}).to_parquet(args.output, index=False)
    print(f"Saved {len(test)} predictions: {args.output}")


if __name__ == "__main__":
    main()
