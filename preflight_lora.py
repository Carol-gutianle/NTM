"""Read-only local-model / GPU checks before starting eight workers."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--mode", choices=["lora", "qlora"], required=True)
    args = p.parse_args()
    if not (args.model / "config.json").is_file():
        p.error(f"Local model config not found: {args.model / 'config.json'}; check the mount.")
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") not in ("qwen3", "qwen2", "llama"):
        p.error(f"Unsupported model_type: {config.get('model_type')}")
    if config.get("quantization_config"):
        p.error("Use original unquantized HF base weights, not a prequantized release.")
    index = args.model / "model.safetensors.index.json"
    if index.exists():
        manifest = json.loads(index.read_text())
        shards = list({args.model / name for name in manifest["weight_map"].values()})
    else:
        shards = [args.model / "model.safetensors"]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        p.error(f"Missing HF safetensors weights: {missing[:5]}")

    # Import shared helpers first so dependency resolution matches training.
    from lora_common import load_tokenizer
    import torch
    import transformers
    import peft
    import accelerate
    from transformers import AutoConfig, AutoModelForSequenceClassification
    auto_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if type(auto_config) not in AutoModelForSequenceClassification._model_mapping:
        p.error("Installed transformers does not support this classifier architecture.")
    load_tokenizer(args.model, offline=True)
    if args.mode == "qlora":
        import bitsandbytes
        print(f"bitsandbytes={bitsandbytes.__version__}")
    print(f"torch={torch.__version__}, transformers={transformers.__version__}, "
          f"peft={peft.__version__}, accelerate={accelerate.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != args.gpus:
        p.error(f"Expected exactly {args.gpus} visible CUDA GPUs, got {torch.cuda.device_count()}.")
    total_bytes = sum(path.stat().st_size for path in shards)
    # A rough screening estimate, not an OOM guarantee; assumes 16-bit source weights.
    estimate = total_bytes / (4 if args.mode == "qlora" else 1) + 6 * 1024**3
    for i in range(args.gpus):
        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, free={free / 1024**3:.1f} GiB, "
              f"total={total / 1024**3:.1f} GiB")
        if free < estimate:
            print(f"WARNING: GPU {i} free memory below rough {args.mode} estimate "
                  f"{estimate / 1024**3:.1f} GiB; reduce batch/length or use QLoRA.")
    print(f"Local HF model OK: {args.model} ({config['model_type']}); "
          "DDP replicates the full base on EACH GPU.")


if __name__ == "__main__":
    main()
