"""Comparable train/validation diagnostics for one fixed, best BERT checkpoint."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix


def load_split(data_path, split_path):
    """Use saved membership, never silently regenerate a split."""
    data = pd.read_csv(data_path, dtype={"id": str, "prompt_id": str})
    membership = pd.read_csv(split_path, dtype={"id": str, "prompt_id": str})
    for name, frame in [("data", data), ("split", membership)]:
        required = {"id", "prompt_id", "model"} | ({"text"} if name == "data" else {"split"})
        if not required <= set(frame.columns) or frame[list(required)].isna().any().any():
            raise ValueError(f"Missing columns/values in {name}")
        if not frame.id.is_unique:
            raise ValueError(f"Duplicate IDs in {name}")
    if set(data.id) != set(membership.id):
        raise ValueError("Data IDs differ from saved split")
    data = data.set_index("id").loc[membership.id].reset_index()
    for column in ["model", "prompt_id"]:
        if data[column].tolist() != membership[column].tolist():
            raise ValueError(f"{column} differs from saved split")
    if set(membership["split"]) != {"train", "validation"}:
        raise ValueError("Expected exactly train and validation membership")
    frames = {split: data.loc[membership["split"].to_numpy() == split].copy()
              for split in ["train", "validation"]}
    if set(frames["train"].prompt_id) & set(frames["validation"].prompt_id):
        raise ValueError("Prompt overlap between training and validation")
    return frames


def save_diagnostics(output, frames, logits_by_split, labels, metadata=None):
    """Save inference-mode metrics; loss is mean, unweighted cross entropy."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    label2id = {label: i for i, label in enumerate(labels)}
    summary = {"metadata": metadata or {}}
    for split in ["train", "validation"]:
        frame = frames[split]
        logits = np.asarray(logits_by_split[split], dtype=np.float64)
        if logits.shape != (len(frame), len(labels)) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid logits for {split}")
        if set(frame.model) != set(labels):
            raise ValueError(f"Label mismatch for {split}")
        targets = np.array([label2id[x] for x in frame.model])
        shifted = logits - logits.max(axis=1, keepdims=True)
        loss = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(frame)), targets])
        predictions = np.array(labels)[logits.argmax(axis=1)]
        report = classification_report(frame.model, predictions, labels=labels,
                                       output_dict=True, zero_division=0)
        metrics = {
            "rows": len(frame), "loss": float(loss),
            "accuracy": float(accuracy_score(frame.model, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(frame.model, predictions)),
            "macro_f1": report["macro avg"]["f1-score"],
        }
        summary[split] = metrics
        for name, value in [("metrics", metrics), ("classification_report", report)]:
            (output / f"{split}_{name}.json").write_text(
                json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        pd.DataFrame(confusion_matrix(frame.model, predictions, labels=labels),
                     index=labels, columns=labels).to_csv(output / f"{split}_confusion_matrix.csv")
        result = frame[["id", "prompt_id", "model"]].copy()
        result["prediction"] = predictions
        result.to_csv(output / f"{split}_predictions.csv", index=False)
        np.save(output / f"{split}_logits.npy", logits)
    summary["train_minus_validation_accuracy"] = summary["train"]["accuracy"] - summary["validation"]["accuracy"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary
