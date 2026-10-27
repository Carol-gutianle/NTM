"""Offline Kaggle BERT submission. No credentials, training data or network."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(4)
root = Path("/kaggle/input")
tests = list(root.rglob("test.csv"))
tests = [p for p in tests if "name-that-llm" in str(p)]
if len(tests) != 1:
    raise RuntimeError(f"Expected one competition test.csv, found {tests}")
models = [p.parent for p in root.rglob("config.json")
          if "ntl-bert-baseline-weights-20261023" in str(p)
          and (p.parent / "model.safetensors").is_file()]
if len(models) != 1:
    raise RuntimeError(f"Expected one bert-model directory, found {models}")
test = pd.read_csv(tests[0], dtype={"id": str})
assert test.id.is_unique and not test[["id", "text"]].isna().any().any()
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(models[0], local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    models[0], local_files_only=True, attn_implementation="sdpa").to(device).eval()
predictions = []
with torch.inference_mode():
    for start in range(0, len(test), 32):
        tokens = tokenizer(test.text.iloc[start:start+32].tolist(), padding=True,
                           truncation=True, max_length=512, return_tensors="pt").to(device)
        ids = model(**tokens).logits.argmax(-1).tolist()
        predictions.extend(model.config.id2label[i] for i in ids)
        if start % 1024 == 0:
            print(f"{min(start+32, len(test))}/{len(test)}", flush=True)
result = pd.DataFrame({"id": test.id, "model": predictions})
assert len(result) == len(test) and not result.isna().any().any()
result.to_parquet("/kaggle/working/submission.parquet", index=False)
print(f"Saved {len(result)} predictions on {device}")
