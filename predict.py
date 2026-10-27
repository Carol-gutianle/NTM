"""Offline inference for saved baseline artifacts (local or Kaggle)."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if os.environ.get("BASELINES_USE_LOCAL_DEPS", "1") == "1" and (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd


def main():
    kaggle = Path("/kaggle/input/competitions/name-that-llm/test.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tfidf", "bert"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=kaggle if kaggle.exists() else ROOT / "test.csv")
    parser.add_argument("--output", type=Path, default=Path("submission.parquet"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    test = pd.read_csv(args.test)
    if args.method == "tfidf":
        import joblib
        model = joblib.load(args.model)
        labels = set(model.classes_)
        predictions = model.predict(test.text)
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(args.threads)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, local_files_only=True,
        ).to(device).eval()
        labels = set(model.config.id2label.values())
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(test), args.batch_size):
                tokens = tokenizer(
                    test.text.iloc[start:start + args.batch_size].tolist(),
                    padding=True, truncation=True, max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                ids = model(**tokens).logits.argmax(-1).tolist()
                predictions.extend(model.config.id2label[i] for i in ids)
    result = pd.DataFrame({"id": test.id, "model": predictions})
    assert len(result) == len(test)
    assert result.id.is_unique and not result.isna().any().any()
    assert set(result.model) <= labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
