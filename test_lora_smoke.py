"""No-download integration test with a random tiny Qwen2 and synthetic 18-class data."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from lora_common import ROOT
import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification
from sklearn.model_selection import StratifiedGroupKFold


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True,
                   env={**os.environ, "HF_HUB_OFFLINE": "1", "WANDB_DISABLED": "true"})


def main():
    torch.set_num_threads(2)
    work = Path(tempfile.mkdtemp(prefix="lora-smoke-", dir="/tmp"))
    base = work / "base"
    labels = sorted(pd.read_csv(ROOT / "train.csv").model.unique())
    words = ["[PAD]", "[UNK]", "[EOS]", "sample", "text", "author", "answer", "short"]
    words += [f"class{i}" for i in range(18)] + [f"prompt{i}" for i in range(10)]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.save_pretrained(base)
    torch.manual_seed(42)
    config_class, model_class = Qwen2Config, Qwen2ForSequenceClassification
    if os.environ.get("SMOKE_QWEN3") == "1":
        from transformers import Qwen3Config, Qwen3ForSequenceClassification
        config_class, model_class = Qwen3Config, Qwen3ForSequenceClassification
    tiny = model_class(config_class(
        vocab_size=len(words), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128, num_labels=18, pad_token_id=0,
        eos_token_id=2, attention_dropout=0.,
    ))
    tiny.save_pretrained(base)
    rows = []
    for prompt in range(10):
        for i, label in enumerate(labels):
            rows.append({"id": f"{prompt:02d}-{i:02d}", "prompt_id": f"p{prompt}",
                         "text": f"sample text prompt{prompt} author class{i} " + "answer " * (i % 4 + 1),
                         "model": label})
    frame = pd.DataFrame(rows)
    train = work / "train.csv"
    frame.to_csv(train, index=False)
    test_input = work / "test_input.csv"
    frame[["id", "text"]].iloc[:7].to_csv(test_input, index=False)
    output = work / "run"
    common = [
        "train_lora.py", "--model", base, "--train", train, "--output", output,
        "--cpu", "--offline", "--precision", "fp32", "--epochs", "2",
        "--batch-size", "18", "--eval-batch-size", "18", "--grad-accum", "2",
        "--max-length", "32", "--rank", "2", "--alpha", "4", "--threads", "2",
        "--test", test_input, "--save-steps", "2",
    ]
    if os.environ.get("SMOKE_DDP") == "1":
        common = ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", *common]
    run(*common)
    assert (output / "SUCCESS.json").exists()
    submission = pd.read_parquet(output / "submission.parquet")
    assert submission.id.tolist() == frame.id.iloc[:7].tolist()
    config = __import__("json").loads((output / "run_config.json").read_text())
    assert config["world_size"] == (2 if os.environ.get("SMOKE_DDP") == "1" else 1)
    metadata = __import__("json").loads((output / "trainable_parameters.json").read_text())
    assert any("score" in name for name in metadata)
    fold = pd.read_csv(output / "split.csv")
    assert set(fold[fold.split == "train"].prompt_id).isdisjoint(fold[fold.split == "validation"].prompt_id)
    real = pd.read_csv(ROOT / "train.csv")
    _, va = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
        real.text, real.model, groups=real.prompt_id))
    if (ROOT / "artifacts/split.csv").exists():
        existing = pd.read_csv(ROOT / "artifacts/split.csv")
        assert real.iloc[va].id.tolist() == existing[existing.split == "validation"].id.tolist()
    val = pd.read_csv(output / "validation_predictions.csv")
    test = work / "test.csv"
    val[["id", "text"]].to_csv(test, index=False)
    infer_logits = work / "reloaded_logits.npy"
    run("predict_lora.py", "--adapter", output / "adapter", "--base-model", base,
        "--test", test, "--output", work / "submission.parquet", "--cpu",
        "--batch-size", "18", "--threads", "2", "--logits-output", infer_logits)
    np.testing.assert_allclose(np.load(infer_logits), np.load(output / "validation_logits.npy"),
                               atol=1e-5, rtol=1e-4)
    predictions = pd.read_parquet(work / "submission.parquet")
    assert predictions.id.tolist() == val.id.tolist()
    assert predictions.model.tolist() == val.prediction.tolist()
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    assert all((x / "COMPLETE").exists() for x in checkpoints)
    run(*common, "--resume", checkpoints[0])
    # Exercise no-validation, fixed-epoch full-data training too.
    full = [*common]
    full[full.index("--output") + 1] = work / "full"
    full[full.index("--epochs") + 1] = "1"
    run(*full, "--full-train")
    assert (work / "full/adapter/adapter_model.safetensors").exists()
    print(f"PASS: training, grouped split, adapter/head reload, offline inference, resume, full train. {work}")


if __name__ == "__main__":
    main()
