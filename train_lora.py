"""Single-GPU / torchrun DDP LoRA classification, grouped evaluation, resumable."""
import argparse
import hashlib
import importlib.metadata
import os
import json
import time
import faulthandler
from datetime import timedelta
from pathlib import Path

from lora_common import (
    ROOT, TextDataset, load_base, load_tokenizer, read_data, resolve_dtype, write_json,
    write_submission,
)
from lora_runtime import RuntimeGuard, CheckpointTrainer
from lora_compat import patch_ddp_adapter_reload
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback, set_seed,
)


def main():
    patch_ddp_adapter_reload()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/apdcephfs_sgfd2/share_303378293/hunyuan/common/Qwen3-32B")
    p.add_argument("--model-load-path", type=Path,
                   help="Verified local cache for loading; --model remains the stable source identity.")
    p.add_argument("--revision", help="Pin a Hugging Face commit for reproducibility.")
    p.add_argument("--train", type=Path, default=ROOT / "train.csv")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/lora_qwen3_32b")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--full-train", action="store_true", help="No validation; use preselected epoch count.")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Only for tiny smoke tests.")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--resume", type=Path, help="Path to checkpoint-N from this exact run.")
    p.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=0, help="0: each epoch; >0: eval/save interval.")
    p.add_argument("--test", type=Path, help="Also export test predictions after training.")
    p.add_argument("--submission", type=Path)
    p.add_argument("--monitor-gpu", action="store_true")
    p.add_argument("--gpu-poll-seconds", type=int, default=10)
    p.add_argument("--gpu-min-util", type=float, default=20)
    p.add_argument("--gpu-low-seconds", type=int, default=180)
    p.add_argument("--gpu-grace-seconds", type=int, default=180)
    p.add_argument("--gpu-low-action", choices=["warn", "stop"], default="warn")
    p.add_argument("--ddp-timeout-seconds", type=int, default=1800,
                   help="Collective timeout, including waiting for slower model-loading ranks.")
    p.add_argument("--load-debug-seconds", type=int, default=0,
                   help="Dump each rank's Python stack repeatedly during model loading; 0 disables.")
    args = p.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not args.cpu and not torch.cuda.is_available():
        p.error("CUDA GPU not detected. --cpu is for tiny smoke tests only.")
    if args.ddp_timeout_seconds <= 0 or args.load_debug_seconds < 0:
        p.error("ddp-timeout-seconds must be positive; load-debug-seconds must be nonnegative.")
    if world_size > 1:
        distributed_kwargs = dict(
            backend="gloo" if args.cpu else "nccl",
            timeout=timedelta(seconds=args.ddp_timeout_seconds),
        )
        if not args.cpu:
            distributed_kwargs["device_id"] = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(**distributed_kwargs)
    if args.cpu and args.four_bit:
        p.error("--cpu and --four-bit cannot be combined.")
    if min(args.batch_size, args.eval_batch_size, args.grad_accum, args.max_length,
           args.rank, args.epochs) <= 0:
        p.error("Batch sizes, length, rank and epochs must be positive.")
    if args.workers < 0 or args.save_steps < 0 or args.threads < 1:
        p.error("workers/save-steps must be nonnegative; threads must be positive.")
    if (min(args.gpu_poll_seconds, args.gpu_low_seconds) <= 0
            or args.gpu_grace_seconds < 0 or not 0 <= args.gpu_min_util <= 100):
        p.error("Invalid GPU monitor thresholds.")
    args.output.mkdir(parents=True, exist_ok=True)
    # Synchronize the emptiness check, before rank zero writes anything.
    nonempty = any(args.output.iterdir())
    if world_size > 1:
        torch.distributed.barrier()
    if nonempty and not args.resume:
        p.error("Output directory is not empty. Choose a new output or use --resume.")
    if (args.output / "STOP_REQUEST").exists():
        p.error("Remove STOP_REQUEST before resuming.")
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    precision, dtype = resolve_dtype(args.precision, args.cpu)
    df = read_data(args.train, labeled=True)
    labels = sorted(df.model.unique().tolist())
    if len(labels) != 18:
        p.error(f"Expected 18 model labels, got {len(labels)}.")
    label2id = {label: i for i, label in enumerate(labels)}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    tr, va = list(splitter.split(df.text, df.model, groups=df.prompt_id))[args.fold]
    if not set(df.iloc[tr].prompt_id).isdisjoint(df.iloc[va].prompt_id):
        raise AssertionError("Prompt leakage")
    if args.full_train:
        tr, va = np.arange(len(df)), np.array([], dtype=int)
    elif set(df.iloc[tr].model) != set(labels) or set(df.iloc[va].model) != set(labels):
        raise ValueError("Each split must contain every class.")
    train_hash = hashlib.sha256(args.train.read_bytes()).hexdigest()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(labels=labels, resolved_precision=precision, train_sha256=train_hash,
                  train_rows=len(tr), validation_rows=len(va), world_size=world_size,
                  effective_batch_size=args.batch_size * args.grad_accum * world_size)
    if args.resume:
        if not args.resume.is_dir():
            p.error("Resume checkpoint does not exist.")
        previous = json.loads((args.output / "run_config.json").read_text())
        for key in ["model", "revision", "train_sha256", "fold", "full_train", "seed",
                    "rank", "alpha", "dropout", "target_modules", "max_length",
                    "batch_size", "grad_accum", "lr", "epochs", "four_bit",
                    "weight_decay", "no_gradient_checkpointing", "resolved_precision",
                    "world_size", "save_steps"]:
            if config[key] != previous.get(key, {"world_size": 1, "save_steps": 0}.get(key)):
                p.error(f"Resume config differs: {key}. Start a new output directory instead.")
    if is_main:
        (args.output / "SUCCESS.json").unlink(missing_ok=True)
        write_json(args.output / "run_config.json", config)
    membership = df[["id", "prompt_id", "model"]].copy()
    membership["split"] = "train"
    membership.loc[va, "split"] = "validation"
    if is_main:
        membership.to_csv(args.output / "split.csv", index=False)
    load_path = str(args.model_load_path) if args.model_load_path else args.model
    tokenizer = load_tokenizer(load_path, args.offline, args.revision)
    if is_main:
        lengths = np.array([len(x) for x in tokenizer(df.text.tolist(), truncation=False)["input_ids"]])
        write_json(args.output / "length_stats.json", {
            "max_length": args.max_length, "max_observed": int(lengths.max()),
            "train_truncated_fraction": float((lengths[tr] > args.max_length).mean()),
            "validation_truncated_fraction": float((lengths[va] > args.max_length).mean()) if len(va) else None,
        })
    rank = os.environ.get("RANK", "0")
    print(f"[rank {rank}] Loading base: {load_path}; source={args.model}; device="
          f"{'cpu' if args.cpu else f'cuda:{local_rank}'}, four_bit={args.four_bit}. "
          "The shard bar advances only after a complete shard is processed.", flush=True)
    started = time.monotonic()
    if args.load_debug_seconds:
        faulthandler.dump_traceback_later(args.load_debug_seconds, repeat=True)
    try:
        model = load_base(load_path, labels, tokenizer, dtype, args.four_bit,
                          args.offline, args.revision, args.attention,
                          device=None if args.cpu else local_rank)
    finally:
        if args.load_debug_seconds:
            faulthandler.cancel_dump_traceback_later()
    print(f"[rank {rank}] Base loaded in {time.monotonic() - started:.1f}s.", flush=True)
    # Do not export an ephemeral node-local path as the original base identity.
    model.name_or_path = args.model
    model.config._name_or_path = args.model
    checkpointing = not args.no_gradient_checkpointing
    if args.four_bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    # Qwen/Llama sequence classification heads are called score. Save/train it too.
    if not hasattr(model, "score"):
        p.error("Expected a Qwen/Llama-style sequence classifier with a 'score' head.")
    modules = args.target_modules if args.target_modules == "all-linear" else args.target_modules.split(",")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=args.alpha,
        lora_dropout=args.dropout, target_modules=modules,
        modules_to_save=["score"], bias="none",
    ))
    # AMP GradScaler cannot unscale FP16 trainable gradients. Keep adapters and
    # the saved classifier in FP32; autocast handles computation in the forward.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()
    trainable = [n for n, x in model.named_parameters() if x.requires_grad]
    assert any("lora_" in n for n in trainable)
    assert any("score" in n and "modules_to_save" in n for n in trainable)
    assert all("lora_" in n or "modules_to_save" in n for n in trainable)
    if is_main:
        write_json(args.output / "trainable_parameters.json", trainable)
    train_set = TextDataset(df.iloc[tr], tokenizer, args.max_length, label2id)
    val_set = TextDataset(df.iloc[va], tokenizer, args.max_length, label2id) if len(va) else None
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=.1, lr_scheduler_type="cosine", optim="adamw_torch",
        max_grad_norm=1., gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=precision == "fp16", bf16=precision == "bf16", use_cpu=args.cpu,
        eval_strategy=("steps" if args.save_steps else "epoch") if val_set is not None else "no",
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500, eval_steps=args.save_steps or 500,
        load_best_model_at_end=val_set is not None,
        metric_for_best_model="balanced_accuracy", greater_is_better=True,
        save_total_limit=2, logging_steps=10, logging_first_step=True,
        report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=args.workers, dataloader_pin_memory=not args.cpu,
        dataloader_persistent_workers=args.workers > 0,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        ddp_broadcast_buffers=False if world_size > 1 else None,
        ddp_timeout=args.ddp_timeout_seconds,
        eval_accumulation_steps=8, label_names=["labels"],
    )
    if args.cpu:
        # Accelerate 1.14 assigns cpu:0 under CPU DDP, which torch.load cannot
        # restore as a map_location. CPU devices are not indexed.
        training_args.distributed_state.device = torch.device("cpu")
        training_args.__dict__["_setup_devices"] = torch.device("cpu")

    def compute_metrics(result):
        logits = result.predictions[0] if isinstance(result.predictions, tuple) else result.predictions
        predictions = logits.argmax(-1)
        return {"balanced_accuracy": balanced_accuracy_score(result.label_ids, predictions),
                "accuracy": accuracy_score(result.label_ids, predictions)}

    guard = RuntimeGuard(
        args.output, args.gpu_poll_seconds, args.gpu_min_util,
        args.gpu_low_seconds, args.gpu_grace_seconds, args.gpu_low_action,
        enabled=args.monitor_gpu,
    )
    trainer = CheckpointTrainer(
        model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
        callbacks=([EarlyStoppingCallback(early_stopping_patience=args.patience)]
                   if val_set is not None else []) + [guard],
    )
    try:
        train_result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    finally:
        guard.close()
    if guard.interrupted:
        if is_main:
            print("已协作保存训练检查点并停止；移除 STOP_REQUEST（若存在）后续训。", flush=True)
        if world_size > 1:
            torch.distributed.destroy_process_group()
        raise SystemExit(75)
    adapter_dir = args.output / "adapter"
    trainer.save_model(str(adapter_dir))
    # Prediction collectives must be called by ALL ranks, not only rank zero.
    result = trainer.predict(val_set) if val_set is not None else None
    test_frame = read_data(args.test) if args.test else None
    test_result = trainer.predict(
        TextDataset(test_frame, tokenizer, args.max_length), metric_key_prefix="submission"
    ) if test_frame is not None else None
    if not is_main:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return
    tokenizer.save_pretrained(adapter_dir)
    # Training does not export the frozen base weights. Store explicit reload metadata.
    write_json(adapter_dir / "inference_config.json", {
        "base_model": args.model, "revision": args.revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "labels": labels, "max_length": args.max_length,
        "four_bit": args.four_bit, "precision": precision,
    })
    write_json(args.output / "training_metrics.json", train_result.metrics)
    write_json(args.output / "history.json", trainer.state.log_history)
    write_json(args.output / "versions.json", {
        pkg: importlib.metadata.version(pkg)
        for pkg in ["torch", "transformers", "peft", "accelerate", "scikit-learn", "pandas"]
    })
    if val_set is not None:
        logits = result.predictions[0] if isinstance(result.predictions, tuple) else result.predictions
        predicted = np.array(labels)[logits.argmax(-1)]
        np.save(args.output / "validation_logits.npy", logits)
        rows = df.iloc[va][["id", "prompt_id", "text", "model"]].copy()
        rows["prediction"] = predicted
        rows.to_csv(args.output / "validation_predictions.csv", index=False)
        write_json(args.output / "validation_metrics.json", {
            **result.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
            "note": "Validation is also used for checkpoint selection; not an independent test.",
        })
        write_json(args.output / "classification_report.json",
                   classification_report(df.iloc[va].model, predicted, labels=labels,
                                         output_dict=True, zero_division=0))
        pd.DataFrame(confusion_matrix(df.iloc[va].model, predicted, labels=labels),
                     index=labels, columns=labels).to_csv(args.output / "confusion_matrix.csv")
        print(result.metrics)
    if test_result is not None:
        logits = test_result.predictions[0] if isinstance(test_result.predictions, tuple) else test_result.predictions
        submission = args.submission or args.output / "submission.parquet"
        write_submission(test_frame, np.array(labels)[logits.argmax(-1)], labels, submission)
        print(f"Saved {len(test_frame)} test predictions: {submission}")
    write_json(args.output / "SUCCESS.json", {
        "adapter": str(adapter_dir), "global_step": trainer.state.global_step,
        "submission": str(args.submission or args.output / "submission.parquet") if args.test else None,
    })
    print(f"Saved adapter + classifier + tokenizer: {adapter_dir}")
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
