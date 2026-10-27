"""Single-GPU full fine-tuning using the existing prompt-disjoint BERT baseline."""
import argparse
from datetime import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", type=Path, default=ROOT / "artifacts/pretrained/bert-base-cased")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" /
                   f"bert_gpu_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3, help="Non-improving epochs, not steps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--cpu", action="store_true", help="Explicit CPU fallback; normally use GPU")
    args = p.parse_args()
    if not 1 <= args.max_length <= 512:
        p.error("This BERT supports max-length 1..512, including special tokens.")
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        p.error("Epochs, patience, batch size, threads and learning rate must be positive.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Single-process trainer: use python, not torchrun.")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output is nonempty; use a new directory (no resume support).")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.require_gpu = not args.cpu
    args.bf16 = args.precision == "bf16"
    from prepare_bert import validate
    validate(args.pretrained)
    os.environ["BASELINES_USE_LOCAL_DEPS"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Importing training dependencies (shared-storage imports may take time)...", flush=True)
    from baselines import bert
    bert(args)
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
