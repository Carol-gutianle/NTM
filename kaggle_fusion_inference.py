"""Offline fixed 50/50 BERT + character TF-IDF; dynamic hidden-test row count."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def normalized_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Invalid model scores")
    z = (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-12)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def unique(paths, description):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {description}, got {paths}")
    return paths[0]


torch.set_num_threads(4)
root = Path("/kaggle/input")
test_path = unique((p for p in root.rglob("test.csv") if "name-that-llm" in str(p)), "test.csv")
model_path = unique((p.parent for p in root.rglob("config.json")
                     if "ntl-bert-baseline-weights-20261023" in str(p)
                     and (p.parent / "model.safetensors").is_file()), "BERT model")
tfidf_path = unique(root.rglob("tfidf_config.json"), "TF-IDF config").parent
config = json.loads((tfidf_path / "tfidf_config.json").read_text())
labels_tfidf = config.pop("labels")
config["ngram_range"] = tuple(config["ngram_range"])
vectorizer = TfidfVectorizer(**config, dtype=np.float32)
with np.load(tfidf_path / "tfidf_arrays.npz", allow_pickle=False) as a:
    vectorizer.idf_ = a["idf"]
    coef, intercept = a["coef"], a["intercept"]
fixtures = json.loads((tfidf_path / "tfidf_fixtures.json").read_text())
np.testing.assert_allclose(
    vectorizer.transform(fixtures) @ coef.T + intercept,
    np.load(tfidf_path / "tfidf_fixture_scores.npy", allow_pickle=False),
    rtol=1e-5, atol=1e-6)
print("TF-IDF runtime compatibility check passed.", flush=True)
test = pd.read_csv(test_path, dtype={"id": str})
assert len(test) and test.id.is_unique and not test[["id", "text"]].isna().any().any()
device = "cuda" if torch.cuda.is_available() else "cpu"
bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, local_files_only=True, attn_implementation="sdpa").to(device).eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
assert len(set(labels)) == len(labels) == len(labels_tfidf) and set(labels) == set(labels_tfidf)
order = [labels_tfidf.index(label) for label in labels]
predictions = []
with torch.inference_mode():
    for start in range(0, len(test), 32):
        texts = test.text.iloc[start:start+32].tolist()
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
            bert_scores = model(**tokens).logits.float().cpu().numpy()
        tfidf_scores = (vectorizer.transform(texts) @ coef.T + intercept)[:, order]
        fused = .5 * normalized_scores(bert_scores) + .5 * normalized_scores(tfidf_scores)
        predictions.extend(np.array(labels)[fused.argmax(1)].tolist())
        if start % 1024 == 0:
            print(f"{min(start+32, len(test))}/{len(test)}", flush=True)
result = pd.DataFrame({"id": test.id, "model": predictions})
assert len(result) == len(test) and not result.isna().any().any()
result.to_parquet("/kaggle/working/submission.parquet", index=False)
Path("/kaggle/working/fusion_run.json").write_text(json.dumps({
    "rows": len(test), "bert_weight": .5, "tfidf_weight": .5,
    "device": device, "bf16": bf16, "max_length": 512,
    "normalization": "per-row z-score, softmax T=1; heuristic, not calibration",
}, indent=2))
print(f"Saved {len(result)} fused predictions on {device}; bf16={bf16}", flush=True)
