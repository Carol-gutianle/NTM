"""Shared offline-capable loading and data helpers for Qwen sequence classification."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for directory in (ROOT / ".deps", ROOT / ".lora-deps") if os.environ.get("LORA_USE_LOCAL_DEPS", "1") == "1" else ():
    if directory.exists():
        sys.path.insert(0, str(directory))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def resolve_dtype(name, cpu=False):
    if name == "auto":
        name = "fp32" if cpu else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    if cpu and name != "fp32":
        raise ValueError("CPU smoke tests require fp32.")
    if not cpu and name == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; choose --precision fp16.")
    return name, {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def load_tokenizer(path, offline=False, revision=None):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=offline, revision=revision)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError("Tokenizer needs an existing EOS or PAD token.")
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base(path, labels, tokenizer, dtype, four_bit=False, offline=False,
              revision=None, attention="sdpa", device=None):
    kwargs = dict(
        num_labels=len(labels),
        label2id={label: i for i, label in enumerate(labels)},
        id2label={i: label for i, label in enumerate(labels)},
        torch_dtype=dtype, local_files_only=offline, revision=revision,
        attn_implementation=attention, low_cpu_mem_usage=True,
    )
    if device is not None:
        # Single-device placement, NOT device_map="auto" model parallelism.
        kwargs["device_map"] = {"": device}
    if four_bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit QLoRA requires a supported CUDA GPU.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            # Keep the randomly initialized, trainable classification head unquantized.
            llm_int8_skip_modules=["score", "classifier"],
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model


def read_data(path, labeled=False):
    df = pd.read_csv(path, dtype={"id": str})
    required = ["id", "text"] + (["model", "prompt_id"] if labeled else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not len(df) or df[required].isna().any().any() or not df.id.is_unique:
        raise ValueError("Empty dataset, missing values or duplicate IDs.")
    if not df.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Every text must be a non-empty string.")
    return df


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, frame, tokenizer, max_length, label2id=None):
        self.enc = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length,
                             return_token_type_ids=False)
        self.labels = None if label2id is None else [label2id[x] for x in frame.model]

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def write_submission(frame, predictions, labels, output):
    result = pd.DataFrame({"id": frame.id, "model": predictions})
    if len(result) != len(frame) or result.isna().any().any() or not result.id.is_unique:
        raise ValueError("Invalid submission rows.")
    if not set(result.model) <= set(labels):
        raise ValueError("Invalid predicted labels.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
