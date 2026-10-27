"""Offline evaluation of a saved best BERT on its original train/validation split."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path, help="Existing bert_gpu_* directory")
    p.add_argument("--train", type=Path, default=ROOT / "train.csv")
    p.add_argument("--output", type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        p.error("Batch size and threads must be positive")
    output = args.output or args.run / f"diagnostics_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    if output.exists() and any(output.iterdir()):
        p.error("Output is nonempty; choose a new directory")
    config = json.loads((args.run / "experiment_config.json").read_text())
    checksum = hashlib.sha256(args.train.read_bytes()).hexdigest()
    if config.get("sha256_train_csv") and config["sha256_train_csv"] != checksum:
        p.error("Training CSV checksum differs from original run")
    if not config.get("sha256_train_csv"):
        print("Older run: no data checksum; checking saved IDs, labels and prompt membership.", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Importing evaluation dependencies...", flush=True)
    import torch
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from bert_diagnostics import load_split, save_diagnostics
    torch.set_num_threads(args.threads)
    if not args.cpu and not torch.cuda.is_available():
        p.error("CUDA unavailable; use --cpu for explicit CPU evaluation")
    device = "cpu" if args.cpu else "cuda"
    bf16 = bool(config.get("bf16", False))
    if bf16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        p.error("Original run used BF16, unsupported on this GPU")
    frames = load_split(args.train, args.run / "split.csv")
    model_path = args.run / "model"
    print(f"Loading best model: {model_path}; device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, attn_implementation="sdpa").to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    results = {}
    with torch.inference_mode():
        for split, frame in frames.items():
            chunks = []
            for start in range(0, len(frame), args.batch_size):
                inputs = tokenizer(frame.text.iloc[start:start + args.batch_size].tolist(),
                                   padding=True, truncation=True, max_length=config["max_length"],
                                   return_tensors="pt").to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                    logits = model(**inputs).logits
                chunks.append(logits.float().cpu().numpy())
                if start == 0 or start // args.batch_size % 20 == 0:
                    print(f"{split}: {min(start + args.batch_size, len(frame))}/{len(frame)}", flush=True)
            results[split] = np.concatenate(chunks)
    save_diagnostics(output, frames, results, labels, {
        "model": str(model_path.resolve()), "max_length": config["max_length"],
        "bf16": bf16, "device": device, "sha256_train_csv": checksum,
        "note": "Same saved best model, eval mode. Validation used for model selection.",
    })
    print(f"Diagnostics saved: {output}", flush=True)


if __name__ == "__main__":
    main()
