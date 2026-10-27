"""Reuse a structurally valid local BERT; download missing files explicitly."""
import argparse
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
FILES = ["config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt"]


def validate(path):
    """Check JSON, BERT configuration and safetensors offsets/file size.

    This catches missing/truncated files, not arbitrary corruption of tensor bytes.
    """
    for name in FILES:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError(f"Missing/empty {path / name}")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "bert":
        raise ValueError("Expected a BERT model")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        json.loads((path / name).read_text())
    weights = path / "model.safetensors"
    with weights.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= size <= 100_000_000:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(f.read(size))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    end = 0
    for tensor in sorted(tensors, key=lambda x: x["data_offsets"][0]):
        start, stop = tensor["data_offsets"]
        if start != end or stop < start:
            raise ValueError("Invalid safetensors offsets")
        end = stop
    if not tensors or weights.stat().st_size != 8 + size + end:
        raise ValueError("Truncated/invalid safetensors file")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/pretrained/bert-base-cased")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    try:
        validate(args.output)
    except (OSError, ValueError, KeyError, struct.error) as exc:
        if args.check_only:
            p.error(str(exc))
        print(f"Local BERT needs preparation: {exc}", flush=True)
        # Must precede huggingface_hub import; training itself stays offline.
        os.environ["HF_HUB_OFFLINE"] = "0"
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="google-bert/bert-base-cased", local_dir=str(args.output),
            allow_patterns=FILES, max_workers=2, force_download=True,
        )
        validate(args.output)
        print(f"Downloaded and checked: {args.output.resolve()}", flush=True)
    else:
        print(f"Local BERT already present; no download needed: {args.output.resolve()}", flush=True)
    print("Structural check passed (not a full weight checksum).", flush=True)


if __name__ == "__main__":
    main()
