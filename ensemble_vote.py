"""Hard-vote existing classifiers on a strictly aligned validation set.

No training or Kaggle submission. Fixed plurality voting; report strict-majority
and tie frequencies separately. The truth is never passed to the voting rule.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

from compare_predictions import read_predictions

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "char_svm": ROOT / "artifacts/tfidf_char/validation_predictions.csv",
    "qwen_lora": ROOT / "artifacts/qwen3_32b_lora_8gpu_20261023_120419_38821/validation_predictions.csv",
    "bert": ROOT / "artifacts/bert_gpu_20261023_143300_319726/validation_predictions.csv",
    "xgboost": ROOT / "artifacts/xgboost_char_first/validation_predictions.csv",
    "tfidf_mlp": ROOT / "artifacts/tfidf_mlp/validation_predictions.csv",
    "word_svm": ROOT / "artifacts/tfidf_word/validation_predictions.csv",
    "combined_svm": ROOT / "artifacts/tfidf_combined/validation_predictions.csv",
}
# Fixed before running this experiment. This is not fitted on each row's truth.
PRIORITY = ["char_svm", "qwen_lora", "bert", "xgboost", "tfidf_mlp",
            "word_svm", "combined_svm"]
PRESETS = {
    "vote3": PRIORITY[:3],
    "vote5": PRIORITY[:5],
    "vote7": PRIORITY,
}


def hard_vote(predictions, priority):
    if not predictions or len(priority) != len(set(priority)) or set(priority) != set(predictions):
        raise ValueError("Priority must contain every member exactly once")
    counts = Counter(predictions.values())
    top_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == top_count}
    # Choose only among tied winning labels, not an arbitrary preferred prediction.
    choice = next(predictions[name] for name in priority if predictions[name] in tied)
    return choice, top_count, len(tied) > 1, top_count > len(predictions) / 2


def align(members):
    first = next(iter(members.values()))
    if not first:
        raise ValueError("Empty prediction set")
    labels = sorted({r["model"] for r in first.values()})
    for name, rows in members.items():
        if set(rows) != set(first):
            raise ValueError(f"{name}: validation ID sets differ; no partial joins allowed")
        for uid, reference in first.items():
            r = rows[uid]
            if (r["model"], r["prompt_id"]) != (reference["model"], reference["prompt_id"]):
                raise ValueError(f"{name}: label/prompt mismatch for {uid}")
            if r["prediction"] not in labels:
                raise ValueError(f"{name}: unknown predicted label {r['prediction']}")
    return list(first), labels


def metrics(truth, predictions, labels):
    if not truth or len(truth) != len(predictions):
        raise ValueError("Invalid predictions")
    report = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predictions))
        support = truth.count(label)
        predicted = predictions.count(label)
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        report[label] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
    return {
        "rows": len(truth),
        "accuracy": sum(t == p for t, p in zip(truth, predictions)) / len(truth),
        "balanced_accuracy": sum(x["recall"] for x in report.values()) / len(labels),
        "macro_f1": sum(x["f1-score"] for x in report.values()) / len(labels),
    }, report


def save_csv(path, rows, fields=None):
    import csv
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(paths, presets, output):
    members = {name: read_predictions(path) for name, path in paths.items()}
    ids, labels = align(members)
    first = next(iter(members.values()))
    truth = [first[uid]["model"] for uid in ids]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, rows in members.items():
        result, _ = metrics(truth, [rows[uid]["prediction"] for uid in ids], labels)
        results.append({"name": name, "kind": "single", **result,
                        "tie_rows": 0, "strict_majority_rows": ""})
    for name, priority in presets.items():
        records = []
        for uid in ids:
            predictions = {n: members[n][uid]["prediction"] for n in priority}
            pred, count, tie, majority = hard_vote(predictions, priority)
            records.append({
                "id": uid, "prompt_id": first[uid]["prompt_id"], "model": first[uid]["model"],
                "prediction": pred, "winning_votes": count, "tie": tie,
                "strict_majority": majority,
                **{f"pred_{n}": predictions[n] for n in priority},
            })
        pred = [r["prediction"] for r in records]
        result, report = metrics(truth, pred, labels)
        result.update(tie_rows=sum(r["tie"] for r in records),
                      strict_majority_rows=sum(r["strict_majority"] for r in records))
        results.append({"name": name, "kind": "hard_vote", **result})
        dest = output / name
        dest.mkdir()
        save_csv(dest / "validation_predictions.csv", records)
        (dest / "metrics.json").write_text(json.dumps(result, indent=2))
        (dest / "classification_report.json").write_text(json.dumps(report, indent=2))
        matrix = Counter(zip(truth, pred))
        save_csv(dest / "confusion_matrix.csv",
                 [{"true_label": t, **{p: matrix[t, p] for p in labels}} for t in labels])
        (dest / "voting_config.json").write_text(json.dumps({
            "members": priority, "tie_priority": priority,
            "rule": "One vote per member; max vote count; tied labels resolved by member "
                    "priority. Not guaranteed to have >50% votes in multiclass problems.",
        }, indent=2))
    save_csv(output / "comparison.csv", results)
    (output / "manifest.json").write_text(json.dumps({
        "inputs": {k: {"path": str(v.resolve()), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
                   for k, v in paths.items()},
        "labels": labels, "presets": presets,
        "note": "Exploratory evaluation on previously used validation data; neither independent "
                "test nor CV. No extra vote for existing BERT/SVM fusion. Reuse the saved "
                "voting rule without true labels at inference. No Kaggle submission.",
    }, indent=2))
    for r in results:
        print(f"{r['name']:15s} accuracy={r['accuracy']:.4%} "
              f"macro_F1={r['macro_f1']:.4%} ties={r['tie_rows']}", flush=True)
    print(f"Saved: {output}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["all", *PRESETS], default="all")
    p.add_argument("--member", action="append", help="Custom NAME=prediction.csv; repeat per member")
    p.add_argument("--priority", help="Custom comma-separated names, every member exactly once")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" /
                   f"multi_vote_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output nonempty; choose a new output directory")
    if args.member:
        paths = {}
        for item in args.member:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in paths:
                p.error("Use unique NAME=prediction.csv entries")
            paths[name] = Path(path)
        priority = args.priority.split(",") if args.priority else list(paths)
        hard_vote({n: "check" for n in paths}, priority)
        presets = {"custom_vote": priority}
    else:
        if args.priority:
            p.error("--priority requires custom --member entries")
        presets = PRESETS if args.preset == "all" else {args.preset: PRESETS[args.preset]}
        selected = {n for names in presets.values() for n in names}
        paths = {k: v for k, v in DEFAULTS.items() if k in selected}
    run(paths, presets, args.output)


if __name__ == "__main__":
    main()
