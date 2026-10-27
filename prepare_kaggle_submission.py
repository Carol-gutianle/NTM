"""Package only explicitly allowed model files and credential-free notebook."""
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
base = root / "artifacts/kaggle-bert-v1"
dataset = base / "dataset"
notebook = base / "notebook"
dataset.mkdir(parents=True, exist_ok=True)
notebook.mkdir(parents=True, exist_ok=True)
model = root / "artifacts/bert_gpu_20261023_143300_319726/model"
target = dataset / "bert-model"
target.mkdir(exist_ok=True)
for name in ["config.json", "model.safetensors", "special_tokens_map.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.txt"]:
    shutil.copy2(model / name, target / name)
(dataset / "dataset-metadata.json").write_text(json.dumps({
    "title": "NTL BERT baseline weights 20261023",
    "id": "gutianle0110/ntl-bert-baseline-weights-20261023",
    "licenses": [{"name": "other"}],
    "description": "Private fine-tuned BERT classifier for name-that-llm. "
                   "Original backbone google-bert/bert-base-cased. No training data or credentials.",
}, indent=2))
source = (root / "kaggle_inference.py").read_text()
book = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
               "outputs": [], "source": source}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 4,
}
(notebook / "inference.ipynb").write_text(json.dumps(book, indent=2))
(notebook / "kernel-metadata.json").write_text(json.dumps({
    "id": "gutianle0110/ntl-bert-offline-baseline-20261023",
    "title": "NTL BERT Offline Baseline 20261023",
    "code_file": "inference.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["gutianle0110/ntl-bert-baseline-weights-20261023"],
    "competition_sources": ["name-that-llm"], "kernel_sources": [],
}, indent=2))
print(f"Prepared private dataset and notebook: {base}")
