"""Sparse character TF-IDF -> trainable MLP; same grouped holdout as SVM."""
import argparse
import copy
import time

from baselines import ART, ROOT, SEED, evaluate, get_split, save_submission, write_json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits


class SparseMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout):
        super().__init__()
        self.first = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        hidden = torch.sparse.mm(x, self.first.weight.T) + self.first.bias
        return self.output(self.dropout(self.activation(hidden)))


def sparse_tensor(matrix):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
        torch.from_numpy(coo.data.astype(np.float32, copy=False)),
        size=coo.shape,
    ).coalesce()


def predict_logits(model, matrix, batch_size):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, matrix.shape[0], batch_size):
            parts.append(model(sparse_tensor(matrix[start:start + batch_size])).numpy())
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--model-dir", type=str, default=str(ART / "tfidf_mlp"))
    parser.add_argument("--test", type=str, default=str(ROOT / "test.csv"))
    parser.add_argument("--output", type=str, default="submission.parquet")
    args = parser.parse_args()
    from pathlib import Path
    out = Path(args.model_dir)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.predict:
        checkpoint = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
        model = SparseMLP(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["state_dict"])
        features = joblib.load(out / "features.joblib")
        test = pd.read_csv(args.test)
        logits = predict_logits(model, features.transform(test.text), args.batch_size)
        labels = checkpoint["labels"]
        save_submission(test, np.array(labels)[logits.argmax(1)], labels, Path(args.output))
        print(f"Wrote {len(test)} predictions to {args.output}", flush=True)
        return

    start = time.monotonic()
    df, tr, va, labels = get_split()
    source = joblib.load(ART / "tfidf_char" / "model.joblib")
    # This fitted vectorizer used only the exact same training fold.
    previous = pd.read_csv(ART / "tfidf_char" / "validation_predictions.csv")
    assert previous.id.tolist() == df.iloc[va].id.tolist()
    features = source.named_steps["features"]
    with threadpool_limits(limits=args.threads):
        x_train = features.transform(df.iloc[tr].text).tocsr()
        x_val = features.transform(df.iloc[va].text).tocsr()
    assert np.array_equal(source.predict(df.iloc[va].text), previous.prediction.values)
    label2id = {label: i for i, label in enumerate(labels)}
    y_train = torch.tensor(df.iloc[tr].model.map(label2id).values, dtype=torch.long)
    y_val = torch.tensor(df.iloc[va].model.map(label2id).values, dtype=torch.long)
    architecture = dict(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim,
                        num_labels=len(labels), dropout=args.dropout)
    model = SparseMLP(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    config = {
        **vars(args), "seed": SEED, "architecture": architecture,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": "cpu", "training_rows": len(tr), "validation_rows": len(va),
        "features": "Exact fitted character TF-IDF from initial SVM training fold",
        "selection": "Best validation balanced accuracy; same validation set used for early stopping",
        "training_scope": "2159-row training partition only, no full-data refit",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    print(config, flush=True)
    # Regression test: sparse first layer agrees with the dense equivalent.
    model.eval()
    tiny = sparse_tensor(x_train[:2])
    with torch.inference_mode():
        sparse_output = model(tiny)
        dense_output = model.output(model.activation(model.first(tiny.to_dense())))
    torch.testing.assert_close(sparse_output, dense_output, atol=1e-6, rtol=1e-5)

    rng = np.random.default_rng(SEED)
    history = []
    best_score, best_epoch, best_state = -1, 0, None
    joblib.dump(features, out / "features.joblib")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.
        order = rng.permutation(len(tr))
        for start_idx in range(0, len(tr), args.batch_size):
            idx = order[start_idx:start_idx + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(sparse_tensor(x_train[idx]))
            loss = criterion(logits, y_train[idx])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        train_logits = predict_logits(model, x_train, args.batch_size)
        val_logits = predict_logits(model, x_val, args.batch_size)
        score = balanced_accuracy_score(y_val.numpy(), val_logits.argmax(1))
        row = {
            "epoch": epoch, "training_loss_with_dropout": total_loss / len(tr),
            "train_loss_eval": criterion(torch.from_numpy(train_logits), y_train).item(),
            "validation_loss": criterion(torch.from_numpy(val_logits), y_val).item(),
            "train_balanced_accuracy": balanced_accuracy_score(y_train.numpy(), train_logits.argmax(1)),
            "validation_balanced_accuracy": score,
            "validation_accuracy": accuracy_score(y_val.numpy(), val_logits.argmax(1)),
            "elapsed_seconds": time.monotonic() - start,
        }
        history.append(row)
        print(row, flush=True)
        write_json(out / "history.json", history)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_state, "architecture": architecture, "labels": labels},
                       out / "model.pt")
            np.save(out / "validation_logits.npy", val_logits)
        if epoch - best_epoch >= args.patience:
            print("Early stopping", flush=True)
            break
    model.load_state_dict(best_state)
    val_logits = predict_logits(model, x_val, args.batch_size)
    evaluate(df.iloc[va], np.array(labels)[val_logits.argmax(1)], labels, out, {
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "total_seconds": time.monotonic() - start,
        "best_epoch_train_balanced_accuracy": history[best_epoch - 1]["train_balanced_accuracy"],
    })
    test = pd.read_csv(args.test)
    predictions = np.array(labels)[predict_logits(model, features.transform(test.text), args.batch_size).argmax(1)]
    save_submission(test, predictions, labels, out / "submission.parquet")


if __name__ == "__main__":
    main()
