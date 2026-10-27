"""Offline LoRA inference. Requires BOTH the adapter and its original base weights."""
import argparse
import json
from pathlib import Path

from lora_common import (
    ROOT, TextDataset, load_base, load_tokenizer, read_data, resolve_dtype,
    write_submission,
)
import numpy as np
import torch
from peft import PeftModel
from transformers import DataCollatorWithPadding


def main():
    kaggle = Path("/kaggle/input/competitions/name-that-llm/test.csv")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--base-model", required=True, help="Local directory of the ORIGINAL base model.")
    p.add_argument("--test", type=Path, default=kaggle if kaggle.exists() else ROOT / "test.csv")
    p.add_argument("--output", type=Path, default=Path("submission.parquet"))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--logits-output", type=Path)
    args = p.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        p.error("CUDA GPU not found. CPU mode is only suitable for small models.")
    if args.cpu and args.four_bit:
        p.error("4-bit mode requires CUDA.")
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    config = json.loads((args.adapter / "inference_config.json").read_text())
    labels = config["labels"]
    torch.set_num_threads(args.threads)
    _, dtype = resolve_dtype(args.precision, args.cpu)
    tokenizer = load_tokenizer(args.adapter, offline=True)
    base = load_base(args.base_model, labels, tokenizer, dtype,
                     four_bit=args.four_bit, offline=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False,
                                      local_files_only=True)
    if not args.four_bit:
        model.to("cpu" if args.cpu else "cuda")
    model.eval()
    device = model.get_input_embeddings().weight.device
    df = read_data(args.test)
    data = TextDataset(df, tokenizer, config["max_length"])
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=device.type == "cuda" and dtype != torch.float32):
                outputs.append(model(**batch).logits.float().cpu().numpy())
    logits = np.concatenate(outputs)
    write_submission(df, np.array(labels)[logits.argmax(1)], labels, args.output)
    if args.logits_output:
        args.logits_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.logits_output, logits)
    print(f"Wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
