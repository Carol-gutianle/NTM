"""Compare saved validation predictions without loading models or fitting weights."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_predictions(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "prompt_id", "model", "prediction"} <= set(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {path}")
        rows = list(reader)
    result = {r["id"]: r for r in rows}
    if not rows or len(result) != len(rows):
        raise ValueError(f"Empty data or duplicate IDs: {path}")
    if any(not r[k] for r in rows for k in ["id", "prompt_id", "model", "prediction"]):
        raise ValueError(f"Missing values: {path}")
    return result


def compare(a, b):
    if set(a) != set(b):
        raise ValueError("Validation ID sets differ; refusing partial comparison.")
    counts = Counter({k: 0 for k in ["both_correct", "bert_only_correct",
                                     "tfidf_only_correct", "both_wrong"]})
    rows = []
    for uid, x in a.items():
        y = b[uid]
        if (x["model"], x["prompt_id"]) != (y["model"], y["prompt_id"]):
            raise ValueError(f"Label/prompt mismatch for {uid}")
        ac, bc = x["prediction"] == x["model"], y["prediction"] == y["model"]
        group = ("both_correct" if ac and bc else "bert_only_correct" if ac
                 else "tfidf_only_correct" if bc else "both_wrong")
        counts[group] += 1
        rows.append(dict(id=uid, prompt_id=x["prompt_id"], model=x["model"],
                         bert_prediction=x["prediction"], tfidf_prediction=y["prediction"],
                         category=group))
    n = len(rows)
    summary = {
        "rows": n, **counts,
        "bert_accuracy": (counts["both_correct"] + counts["bert_only_correct"]) / n,
        "tfidf_accuracy": (counts["both_correct"] + counts["tfidf_only_correct"]) / n,
        "prediction_agreement": sum(r["bert_prediction"] == r["tfidf_prediction"] for r in rows) / n,
        "oracle_selection_accuracy": 1 - counts["both_wrong"] / n,
        "note": "Oracle uses ground truth to pick the correct model per row. Not an achievable "
                "ensemble score or an upper bound for arbitrary score fusion. No fusion fitted.",
    }
    return summary, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bert-run", type=Path, required=True)
    p.add_argument("--tfidf-run", type=Path, default=Path("artifacts/tfidf_char"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    output = args.output or args.bert_run / "tfidf_comparison"
    if output.exists() and any(output.iterdir()):
        p.error("Output is nonempty; choose a new --output directory.")
    summary, rows = compare(
        read_predictions(args.bert_run / "validation_predictions.csv"),
        read_predictions(args.tfidf_run / "validation_predictions.csv"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
