"""Reproducible prompt-disjoint TF-IDF/BERT holdout experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if os.environ.get("BASELINES_USE_LOCAL_DEPS", "1") == "1" and (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold

ART = ROOT / "artifacts"
SEED = 42


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def get_split():
    df = pd.read_csv(ROOT / "train.csv")
    labels = sorted(df.model.unique())
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(splitter.split(df.text, df.model, groups=df.prompt_id))
    assert not set(df.iloc[tr].prompt_id) & set(df.iloc[va].prompt_id)
    assert set(df.iloc[tr].model) == set(df.iloc[va].model) == set(labels)
    ART.mkdir(exist_ok=True)
    split = df[["id", "prompt_id", "model"]].copy()
    split["split"] = "train"
    split.loc[va, "split"] = "validation"
    path = ART / "split.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), split)
    else:
        split.to_csv(path, index=False)
    write_json(ART / "split_metadata.json", {
        "seed": SEED, "method": "StratifiedGroupKFold, first fold of 5",
        "train_rows": len(tr), "validation_rows": len(va),
        "train_prompts": df.iloc[tr].prompt_id.nunique(),
        "validation_prompts": df.iloc[va].prompt_id.nunique(),
        "sha256_train_csv": hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest(),
        "labels": labels,
    })
    return df, tr, va, labels


def evaluate(frame, predictions, labels, out, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        frame.model, predictions, labels=labels, output_dict=True, zero_division=0,
    )
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(frame.model, predictions),
        "accuracy": accuracy_score(frame.model, predictions),
        "validation_rows": len(frame),
        **(extra or {}),
    }
    write_json(out / "metrics.json", metrics)
    write_json(out / "classification_report.json", report)
    pd.DataFrame(confusion_matrix(frame.model, predictions, labels=labels),
                 index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")
    result = frame[["id", "prompt_id", "model"]].copy()
    result["prediction"] = predictions
    result.to_csv(out / "validation_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def save_submission(test, predictions, labels, path):
    result = pd.DataFrame({"id": test.id, "model": predictions})
    assert len(result) == len(test) and result.id.is_unique
    assert not result.isna().any().any()
    assert set(result.model) <= set(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)


def tfidf(args):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC
    from threadpoolctl import threadpool_limits

    df, tr, va, labels = get_split()
    results = {}
    test = pd.read_csv(ROOT / "test.csv")
    with threadpool_limits(limits=args.threads):
        for kind in ["char", "word", "combined"]:
            branches = []
            if kind in ["char", "combined"]:
                branches.append(("char", TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), lowercase=False,
                    min_df=2, sublinear_tf=True, max_features=200_000,
                    dtype=np.float32,
                )))
            if kind in ["word", "combined"]:
                branches.append(("word", TfidfVectorizer(
                    ngram_range=(1, 2), lowercase=False, min_df=2,
                    sublinear_tf=True, max_features=100_000, dtype=np.float32,
                )))
            pipe = Pipeline([
                ("features", FeatureUnion(branches)),
                ("classifier", LinearSVC(C=1.0, random_state=SEED, max_iter=5000)),
            ])
            start = time.monotonic()
            pipe.fit(df.iloc[tr].text, df.iloc[tr].model)
            pred = pipe.predict(df.iloc[va].text)
            out = ART / f"tfidf_{kind}"
            results[kind] = evaluate(df.iloc[va], pred, labels, out, {
                "train_and_eval_seconds": time.monotonic() - start,
                "features": pipe.named_steps["classifier"].coef_.shape[1],
                "config": pipe.get_params()["classifier"].get_params(),
                "training_scope": "holdout training split only",
            })
            joblib.dump(pipe, out / "model.joblib")
            save_submission(test, pipe.predict(test.text), labels, out / "submission.parquet")
            np.save(out / "validation_scores.npy", pipe.decision_function(df.iloc[va].text))
    write_json(ART / "tfidf_comparison.json", results)


def tfidf_cv(args):
    import joblib
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits

    df, first_tr, first_va, labels = get_split()
    # Clone the exact previously evaluated configuration, discarding fitted state.
    template = joblib.load(ART / "tfidf_char" / "model.joblib")
    out = ART / "tfidf_char_cv"
    out.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_scores = np.full((len(df), len(labels)), np.nan)
    fold_ids = np.full(len(df), -1, dtype=int)
    coverage = np.zeros(len(df), dtype=int)
    metrics = []
    start = time.monotonic()
    with threadpool_limits(limits=args.threads):
        for fold, (tr, va) in enumerate(
            splitter.split(df.text, df.model, groups=df.prompt_id), 1
        ):
            assert set(df.iloc[tr].prompt_id).isdisjoint(df.iloc[va].prompt_id)
            if fold == 1:
                assert np.array_equal(tr, first_tr) and np.array_equal(va, first_va)
            model = clone(template)
            fold_start = time.monotonic()
            model.fit(df.iloc[tr].text, df.iloc[tr].model)
            assert model.classes_.tolist() == labels
            scores = model.decision_function(df.iloc[va].text)
            predictions = np.array(labels)[scores.argmax(1)]
            oof_scores[va] = scores
            fold_ids[va] = fold
            coverage[va] += 1
            fold_out = out / f"fold_{fold}"
            metrics.append(evaluate(df.iloc[va], predictions, labels, fold_out, {
                "fold": fold,
                "train_rows": len(tr),
                "train_prompts": df.iloc[tr].prompt_id.nunique(),
                "validation_prompts": df.iloc[va].prompt_id.nunique(),
                "train_and_eval_seconds": time.monotonic() - fold_start,
                "features": model.named_steps["classifier"].coef_.shape[1],
            }))
            joblib.dump(model, fold_out / "model.joblib")
            if fold == 1:
                previous = pd.read_csv(ART / "tfidf_char" / "validation_predictions.csv")
                assert previous.id.tolist() == df.iloc[va].id.tolist()
                assert previous.prediction.tolist() == predictions.tolist()
    assert (coverage == 1).all() and np.isfinite(oof_scores).all()
    membership = df[["id", "prompt_id", "model"]].copy()
    membership["validation_fold"] = fold_ids
    assert membership.groupby("prompt_id").validation_fold.nunique().eq(1).all()
    membership.to_csv(out / "fold_assignments.csv", index=False)
    np.save(out / "oof_scores.npy", oof_scores)
    oof_predictions = np.array(labels)[oof_scores.argmax(1)]
    pooled = evaluate(df, oof_predictions, labels, out / "oof")
    values = [m["balanced_accuracy"] for m in metrics]
    write_json(out / "summary.json", {
        "seed": SEED,
        "method": "StratifiedGroupKFold, 5 folds, fixed character TF-IDF configuration",
        "folds": metrics,
        "mean_balanced_accuracy": float(np.mean(values)),
        "std_balanced_accuracy_population": float(np.std(values)),
        "min_balanced_accuracy": float(np.min(values)),
        "max_balanced_accuracy": float(np.max(values)),
        "pooled_oof_balanced_accuracy": pooled["balanced_accuracy"],
        "total_seconds": time.monotonic() - start,
        "labels": labels,
        "sha256_train_csv": hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest(),
        "note": "Configuration selected on fold 1 previously; CV is not an untouched final test.",
    })
    print((out / "summary.json").read_text(), flush=True)


def bert(args):
    import random
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, get_linear_schedule_with_warmup,
    )

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "require_gpu", False) and device.type != "cuda":
        raise RuntimeError("CUDA unavailable; refusing to silently train on CPU.")
    if args.bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; use --precision fp32.")
    df, tr, va, labels = get_split()
    label2id = {name: i for i, name in enumerate(labels)}
    out = getattr(args, "output", None) or ART / "bert"
    out.mkdir(parents=True, exist_ok=True)
    source = args.pretrained
    print(f"Loading local BERT: {source}; device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    full_lengths = np.array([len(x) for x in tokenizer(df.text.tolist(), truncation=False)["input_ids"]])
    config = {
        "source": str(source), "seed": SEED, "max_length": args.max_length,
        "sha256_train_csv": hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest(),
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "epochs": args.epochs, "patience": args.patience, "device": str(device),
        "threads": args.threads, "bf16": args.bf16,
        "training_scope": "holdout training split only",
        "truncated_fraction_train_split": float((full_lengths[tr] > args.max_length).mean()),
        "truncated_fraction_validation": float((full_lengths[va] > args.max_length).mean()),
        "token_length_quantiles": np.quantile(full_lengths, [0, .5, .9, .95, 1]).tolist(),
    }
    write_json(out / "experiment_config.json", config)
    membership = df[["id", "prompt_id", "model"]].copy()
    membership["split"] = "train"
    membership.loc[va, "split"] = "validation"
    membership.to_csv(out / "split.csv", index=False)
    length_rows = []
    for split, indices in [("train", tr), ("validation", va)]:
        for label in labels:
            values = full_lengths[indices[df.iloc[indices].model.to_numpy() == label]]
            length_rows.append({
                "split": split, "model": label, "count": len(values),
                "mean_tokens": float(values.mean()),
                "median_tokens": float(np.median(values)),
                "p95_tokens": float(np.quantile(values, .95)),
                "max_tokens": int(values.max()),
                "truncated_fraction": float((values > args.max_length).mean()),
            })
    pd.DataFrame(length_rows).to_csv(out / "length_stats.csv", index=False)
    print(json.dumps(config, indent=2), flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(labels), label2id=label2id,
        id2label={i: label for label, i in label2id.items()}, local_files_only=True,
        attn_implementation="sdpa",
    ).to(device)
    collator = DataCollatorWithPadding(tokenizer)

    def loader(frame, shuffle=False, labeled=True):
        enc = tokenizer(frame.text.tolist(), truncation=True, max_length=args.max_length)
        rows = [{key: values[i] for key, values in enc.items()} for i in range(len(frame))]
        if labeled:
            for row, label in zip(rows, frame.model):
                row["labels"] = label2id[label]
        return DataLoader(rows, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collator, num_workers=0,
                          generator=torch.Generator().manual_seed(SEED))

    train_loader = loader(df.iloc[tr], shuffle=True)
    val_loader = loader(df.iloc[va])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * .1), num_training_steps=total_steps,
    )

    def predict(batches):
        model.eval()
        outputs = []
        with torch.inference_mode():
            for batch in batches:
                batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    logits = model(**batch).logits
                outputs.append(logits.float().cpu().numpy())
        return np.concatenate(outputs)

    history = []
    best = -1
    best_epoch = 0
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            n = len(batch["labels"])
            total_loss += loss.item() * n
            count += n
            if step == 1 or step % 20 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={total_loss/count:.4f} elapsed={time.monotonic()-start:.1f}s", flush=True)
        logits = predict(val_loader)
        pred = np.array(labels)[logits.argmax(1)]
        score = balanced_accuracy_score(df.iloc[va].model, pred)
        history.append({
            "epoch": epoch, "train_loss": total_loss / count,
            "balanced_accuracy": score,
            "accuracy": accuracy_score(df.iloc[va].model, pred),
            "elapsed_seconds": time.monotonic() - start,
        })
        write_json(out / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            model.save_pretrained(out / "model")
            tokenizer.save_pretrained(out / "model")
            np.save(out / "validation_logits.npy", logits)
            evaluate(df.iloc[va], pred, labels, out, {
                "best_epoch": best_epoch,
                "train_and_eval_seconds": time.monotonic() - start,
                "config": config,
            })
        if epoch - best_epoch >= args.patience:
            break
    # Free optimizer state before loading the best checkpoint.
    del optimizer, scheduler, model
    model = AutoModelForSequenceClassification.from_pretrained(
        out / "model", local_files_only=True, attn_implementation="sdpa",
    ).to(device)
    # Evaluate BOTH splits in eval mode with this same best checkpoint.
    # This is distinct from training-loop loss (dropout on, changing weights).
    from bert_diagnostics import save_diagnostics
    print("Evaluating best BERT on training and validation splits...", flush=True)
    frames = {"train": df.iloc[tr], "validation": df.iloc[va]}
    logits_by_split = {split: predict(loader(frame)) for split, frame in frames.items()}
    save_diagnostics(out / "diagnostics", frames, logits_by_split, labels, {
        "model": str((out / "model").resolve()), "best_epoch": best_epoch,
        "max_length": args.max_length, "bf16": args.bf16, "device": str(device),
        "note": "Same best checkpoint, eval mode. Validation used for model selection.",
    })
    test = pd.read_csv(ROOT / "test.csv")
    pred = np.array(labels)[predict(loader(test, labeled=False)).argmax(1)]
    save_submission(test, pred, labels, out / "submission.parquet")
    write_json(out / "run_summary.json", {
        "best_epoch": best_epoch, "best_balanced_accuracy": best,
        "epochs_completed": len(history), "total_seconds": time.monotonic() - start,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["tfidf", "tfidf-cv", "bert", "split"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pretrained", default=str(ART / "pretrained" / "bert-base-cased"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    {"tfidf": tfidf, "tfidf-cv": tfidf_cv, "bert": bert,
     "split": lambda _: get_split()}[args.method](args)


if __name__ == "__main__":
    main()
