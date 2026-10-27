"""Export a portable TF-IDF + linear classifier, verify it, package private notebook."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
base = ROOT / "artifacts/kaggle-fusion-v1"
dataset, notebook = base / "dataset", base / "notebook"
dataset.mkdir(parents=True, exist_ok=True)
notebook.mkdir(parents=True, exist_ok=True)
pipe = joblib.load(ROOT / "artifacts/tfidf_char/model.joblib")
branches = pipe.named_steps["features"].transformer_list
assert len(branches) == 1 and branches[0][0] == "char"
vec = branches[0][1]
clf = pipe.named_steps["classifier"]
# Export explicit arrays/config instead of unpickling sklearn objects on Kaggle.
settings = {
    "analyzer": vec.analyzer, "ngram_range": list(vec.ngram_range),
    "lowercase": vec.lowercase, "sublinear_tf": vec.sublinear_tf,
    "norm": vec.norm, "use_idf": vec.use_idf, "smooth_idf": vec.smooth_idf,
    "binary": vec.binary, "strip_accents": vec.strip_accents,
    "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
    "labels": clf.classes_.tolist(),
}
assert vec.preprocessor is None and vec.tokenizer is None
(dataset / "tfidf_config.json").write_text(json.dumps(settings, ensure_ascii=False))
np.savez_compressed(dataset / "tfidf_arrays.npz", idf=vec.idf_,
                    coef=clf.coef_, intercept=clf.intercept_)
restored = TfidfVectorizer(
    **{k: (tuple(v) if k == "ngram_range" else v)
       for k, v in settings.items() if k != "labels"}, dtype=np.float32)
restored.idf_ = vec.idf_
frame = pd.read_csv(ROOT / "train.csv", dtype={"id": str})
pred = pd.read_csv(ROOT / "artifacts/tfidf_char/validation_predictions.csv",
                   dtype={"id": str})
texts = frame.set_index("id").loc[pred.id, "text"].tolist()
scores = restored.transform(texts) @ clf.coef_.T + clf.intercept_
expected = np.load(ROOT / "artifacts/tfidf_char/validation_scores.npy")
np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)
assert np.array_equal(clf.classes_[scores.argmax(1)], pred.prediction)
print("Portable TF-IDF matches saved validation scores and all predictions.", flush=True)
# Non-secret reference fixtures permit a tiny pre-inference runtime check.
fixtures = ["Hello world! This is a test.", "A  B\nC\tD. Unicode: café 中文."]
np.save(dataset / "tfidf_fixture_scores.npy", pipe.decision_function(fixtures))
(dataset / "tfidf_fixtures.json").write_text(json.dumps(fixtures))
(dataset / "dataset-metadata.json").write_text(json.dumps({
    "title": "NTL TFIDF fusion weights 20261023",
    "id": "gutianle0110/ntl-tfidf-fusion-weights-20261023",
    "licenses": [{"name": "other"}],
    "description": "Private portable character TF-IDF and linear classifier parameters. "
                   "No raw training samples, credentials or validation labels.",
}, indent=2))
source = (ROOT / "kaggle_fusion_inference.py").read_text()
(notebook / "inference.ipynb").write_text(json.dumps({
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
               "outputs": [], "source": source}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4,
}, indent=2))
(notebook / "kernel-metadata.json").write_text(json.dumps({
    "id": "gutianle0110/ntl-bert-tfidf-fusion-20261023",
    "title": "NTL BERT TFIDF Fusion 20261023", "code_file": "inference.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["gutianle0110/ntl-bert-baseline-weights-20261023",
                        "gutianle0110/ntl-tfidf-fusion-weights-20261023"],
    "competition_sources": ["name-that-llm"], "kernel_sources": [],
}, indent=2))
print(f"Prepared: {base}", flush=True)
