"""Export an explicit, text-free allowlist of existing experiment evidence.

No training, network calls, credentials, model deserialization or API requests.
Run: python export_results.py --source /path/to/original/project
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QWEN = "qwen3_32b_lora_8gpu_20261023_120419_38821"
BERT = "bert_gpu_20261023_143300_319726"
FUSION = "bert_tfidf_fusion_20261023_150134_2"
VOTE = "multi_vote_20261023_163628_2"
RUNS = [
    ("Character TF-IDF + LinearSVC", "tfidf_char", "fixed baseline"),
    ("Word TF-IDF + LinearSVC", "tfidf_word", "fixed baseline"),
    ("Character + word TF-IDF + LinearSVC", "tfidf_combined", "fixed baseline"),
    ("TF-IDF MLP", "tfidf_mlp", "checkpoint selected on validation"),
    ("BERT CPU / 256 tokens", "bert", "checkpoint selected on validation"),
    ("BERT GPU / 512 tokens", BERT, "checkpoint selected on validation"),
    ("Qwen3-32B LoRA / 8 GPUs", QWEN, "checkpoint selected on validation"),
    ("XGBoost / character TF-IDF", "xgboost_char_first", "reused validation"),
    ("KNN / character TF-IDF", "knn_char_first", "10 candidates; validation-selected"),
    ("BERT + character SVM score fusion", FUSION, "5 weights; validation-selected"),
    ("Hard vote / 3 models", f"{VOTE}/vote3", "exploratory reused validation"),
    ("Hard vote / 5 models", f"{VOTE}/vote5", "exploratory reused validation"),
    ("Hard vote / 7 models", f"{VOTE}/vote7", "exploratory reused validation"),
    ("Character SVM + 1,050 external rows", "augmentation_interim_tfidf/augmented",
     "partial external augmentation"),
    ("Character SVM + 2,025 external rows", "augmentation200_evaluation/augmented",
     "partial external augmentation"),
]
# Only these directories and basenames may be exported. Never traverse model,
# checkpoint, gateway response, dataset, tokenizer, or raw notebook directories.
DIRECTORIES = [r[1] for r in RUNS] + [
    "tfidf_char_cv", "tfidf_char_cv/oof",
    *[f"tfidf_char_cv/fold_{i}" for i in range(1, 6)],
    "bert_tfidf_comparison_20261023", VOTE,
    f"{BERT}/diagnostics_20261023_145126_336640",
    "xgboost_char_first/diagnostics",
    "augmentation_interim_tfidf", "augmentation_interim_tfidf/original",
    "augmentation200_evaluation", "augmentation200_evaluation/original",
    "augmentation_eval_pipeline_check",
    "augmentation_eval_pipeline_check/original",
    "augmentation_eval_pipeline_check/augmented",
    "qwen3_32b_first",
]
JSON_NAMES = {
    "metrics.json", "validation_metrics.json", "training_metrics.json",
    "train_metrics.json", "classification_report.json",
    "validation_classification_report.json", "train_classification_report.json",
    "summary.json", "config.json", "experiment_config.json", "run_config.json",
    "run_summary.json", "history.json", "length_stats.json", "versions.json",
    "SUCCESS.json", "fusion_config.json", "voting_config.json",
}
CSV_NAMES = {
    "validation_predictions.csv", "train_predictions.csv", "confusion_matrix.csv",
    "validation_confusion_matrix.csv", "train_confusion_matrix.csv",
    "split.csv", "fold_assignments.csv", "search_results.csv",
    "weight_search.csv", "length_stats.csv",
}
EXTRA_FILES = [
    "split.csv", "split_metadata.json", "tfidf_comparison.json",
    "augmentation200/progress.json", "augmentation200/evaluation_status.json",
    "decoder_qwen_v8/preflight.json", "decoder_qwen_v8/provenance.json",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def calculate_metrics(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty predictions or duplicate IDs")
    labels = sorted({r["model"] for r in rows})
    if any(r["prediction"] not in labels for r in rows):
        raise ValueError("Prediction outside label vocabulary")
    actual = Counter(r["model"] for r in rows)
    predicted = Counter(r["prediction"] for r in rows)
    correct = Counter(r["model"] for r in rows if r["model"] == r["prediction"])
    return {
        "rows": len(rows),
        "accuracy": sum(correct.values()) / len(rows),
        "balanced_accuracy": sum(correct[k] / actual[k] for k in labels) / len(labels),
        "macro_f1": sum(2 * correct[k] / (actual[k] + predicted[k])
                        for k in labels) / len(labels),
    }


def sanitize(value, source):
    if isinstance(value, dict):
        return {k: sanitize(v, source) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, source) for v in value]
    if isinstance(value, str):
        # Existing experiments were also copied between the two original hosts.
        for prefix in (
            str(source.resolve()),
            "/apdcephfs_sgfd2/share_303378293/tianlelegu/name_that_llm",
            "/data/home/tianlelegu/name_that_llm",
        ):
            value = value.replace(prefix + "/", "")
        value = value.replace(
            "/apdcephfs_sgfd2/share_303378293/hunyuan/common/Qwen3-32B",
            "<LOCAL_QWEN3_32B>")
    return value


def export(source, output):
    artifacts = source / "artifacts"
    files = {artifacts / p for p in EXTRA_FILES}
    for directory in sorted(set(DIRECTORIES)):
        folder = artifacts / directory
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files.update(p for p in folder.iterdir()
                     if p.name in JSON_NAMES | CSV_NAMES and p.is_file())
    manifest = []
    for path in sorted(files):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked evidence: {path}")
        rel = path.relative_to(artifacts)
        raw = path.read_bytes()
        transform = "none"
        if path.suffix == ".json":
            original = json.loads(raw)
            clean = sanitize(original, source)
            data = (json.dumps(clean, ensure_ascii=False, indent=2) + "\n").encode()
            transform = "JSON formatting; local paths normalized" if clean != original else "JSON formatting"
        else:
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
            if "text" in (reader.fieldnames or []):
                fields = [f for f in reader.fieldnames if f != "text"]
                buffer = io.StringIO(newline="")
                writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows({f: r[f] for f in fields} for r in reader)
                data = buffer.getvalue().encode()
                transform = "removed original competition text column"
            else:
                data = raw
        destination = output / "evidence" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        manifest.append({
            "source": f"artifacts/{rel.as_posix()}",
            "destination": f"evidence/{rel.as_posix()}",
            "source_sha256": digest(raw), "export_sha256": digest(data),
            "bytes": len(data), "transformation": transform,
        })
    (output / "manifest.json").write_text(json.dumps({
        "snapshot_date": "2026-10-27",
        "experiment_dates": ["2026-10-22", "2026-10-23"],
        "policy": "Explicit allowlist; no weights, raw texts, API responses or credentials.",
        "files": manifest,
    }, ensure_ascii=False, indent=2) + "\n")
    build_summary(output)


def build_summary(output):
    evidence = output / "evidence"
    split = csv_rows(evidence / "split.csv")
    validation = {r["id"]: r for r in split if r["split"] == "validation"}
    training = [r for r in split if r["split"] == "train"]
    if {r["prompt_id"] for r in training} & {r["prompt_id"] for r in validation.values()}:
        raise ValueError("Prompt leakage in frozen split")
    summary = []
    for name, directory, selection in RUNS:
        predictions = csv_rows(evidence / directory / "validation_predictions.csv")
        if {r["id"] for r in predictions} != set(validation):
            raise ValueError(f"Validation ID mismatch: {directory}")
        for row in predictions:
            if any(row[k] != validation[row["id"]][k] for k in ("model", "prompt_id")):
                raise ValueError(f"Label/prompt mismatch: {directory}")
        metrics = calculate_metrics(predictions)
        report = evidence / directory / "classification_report.json"
        if report.exists():
            saved = json.loads(report.read_text())
            if "accuracy" in saved and abs(saved["accuracy"] - metrics["accuracy"]) > 1e-10:
                raise ValueError(f"Saved report mismatch: {directory}")
            labels = sorted({r["model"] for r in predictions})
            for key, field in (("balanced_accuracy", "recall"), ("macro_f1", "f1-score")):
                mean = sum(saved[label][field] for label in labels) / len(labels)
                if abs(mean - metrics[key]) > 1e-10:
                    raise ValueError(f"Saved {key} mismatch: {directory}")
        for filename in ("metrics.json", "validation_metrics.json"):
            saved_path = evidence / directory / filename
            if saved_path.exists():
                saved = json.loads(saved_path.read_text())
                for key in ("accuracy", "balanced_accuracy", "macro_f1"):
                    value = saved.get(key, saved.get("test_" + key))
                    if value is not None and abs(value - metrics[key]) > 1e-10:
                        raise ValueError(f"Saved metric mismatch: {directory}/{filename}: {key}")
        summary.append({"method": name, **metrics, "protocol": "grouped_holdout_fold_1",
                        "selection": selection,
                        "evidence": f"evidence/{directory}/validation_predictions.csv"})
    summary.sort(key=lambda r: -r["balanced_accuracy"])
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cv = json.loads((evidence / "tfidf_char_cv/summary.json").read_text())
    lines = [
        "# 实验结果汇总",
        "",
        "整理日期：2026-10-27；实验记录：2026-10-22 至 2026-10-23。本次仅整理存量结果，没有重训或提交竞赛。",
        "",
        "## 验证协议",
        "",
        "- 2,699 条有标签样本、150 个 prompt、18 类；仅使用回答文本作为模型输入。",
        "- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` 第一折："
        "2,159 条训练 / 540 条验证，120 / 30 个 prompt，无 prompt 交集。",
        "- 下表指标由已归档预测重新计算，并校验 ID、标签、prompt 和可用的分类报告。"
        "540 条验证数据每类 30 条，因此 Accuracy 与 Balanced accuracy 相同。",
        "- 同一验证集反复用于 checkpoint、超参数和融合权重选择；这些是探索性验证分数，"
        "**不是独立测试、五折均值或 Kaggle 榜单分数**。",
        "",
        "## 同一 holdout 对比",
        "",
        "| 方法 | Balanced accuracy | Macro F1 | 选择/实验说明 |",
        "|---|---:|---:|---|",
    ]
    for row in summary:
        lines.append(f"| {row['method']} | {row['balanced_accuracy']:.2%} | "
                     f"{row['macro_f1']:.2%} | {row['selection']} |")
    lines += [
        "",
        "精确值与逐项证据路径见 `summary.csv` / `summary.json`。",
        "",
        "## 五折 grouped CV（单独报告）",
        "",
        "| Fold | 验证样本数 | Balanced accuracy |",
        "|---|---:|---:|",
    ]
    for fold in cv["folds"]:
        lines.append(f"| {fold['fold']} | {fold['validation_rows']} | {fold['balanced_accuracy']:.2%} |")
    lines += [
        "",
        f"- 折均值 **{cv['mean_balanced_accuracy']:.2%}**；总体标准差 "
        f"**{100 * cv['std_balanced_accuracy_population']:.2f} 个百分点**。",
        f"- pooled OOF balanced accuracy：**{cv['pooled_oof_balanced_accuracy']:.2%}**；"
        "每条样本有一次 OOF 预测。",
        "- 字符 TF-IDF 配置事先根据第一折选择，因此五折结果也不是完全未触碰的最终评估。",
        "",
        "## 结论与限制",
        "",
        "- 当前 holdout 最高是 BERT + 字符 SVM 分数融合：35.74%，BERT 权重 0.5，SVM 权重 0.5。"
        "分数经逐样本标准化、softmax 后融合，并非概率校准。",
        "- Qwen3-32B LoRA 为单模型最高：35.19%；8 GPU、BF16、非 4-bit、rank 16、"
        "alpha 32、max length 512，选择 checkpoint-300。日志键 `test_*` 实际指上述验证集。",
        "- 3/5/7 模型硬投票均未超过分数融合。BERT/SVM oracle 47.78% 使用真实标签选模型，"
        "不可作为实际可部署成绩。",
        "- 2,025 条外部增强覆盖 12/18 类，未达到每类 200 条；8 类达到 200 条。"
        "加入增强后 31.85% → 28.33%（下降 3.52 个百分点）；修正 24 条、退化 43 条。"
        "这是特定增强配方的负结果，不证明所有增强都无效。",
        "- 增强生成记录 2,144 次请求/响应、733,495 reported tokens、"
        "reported cost 61.124810224（币种未知）；存在未知 reasoning usage 和网关模型身份未独立核验的问题。",
        "- BERT GPU 训练集 accuracy 65.68%，验证 27.59%；训练诊断不能当泛化能力。"
        "其余训练诊断保存在对应 evidence 子目录。",
        "- LLM Decoder Qwen v8 仅完成源码准备和预检；预检未通过，缺核心模型/数据附件，"
        "**没有端到端复现成绩**。上游 Notebook、代码和权重不随本仓库重新发布。",
        "- `qwen3_32b_first` 仅留有配置/长度信息，不能视为完成训练；"
        "`augmentation_eval_pipeline_check` 是流程检查，不列入主结果表。",
        "- 本地 test.csv 仅 3 条样例；本次不发布这些 smoke submission，"
        "归档也不提供可核验的隐藏测试榜单分数。",
        "",
        "## 归档与复核",
        "",
        "- `evidence/` 保留原 run 目录名，包括指标、分类报告、混淆矩阵、"
        "去文本的预测、训练配置/历史、CV 分折和增强完成状态。",
        "- `manifest.json` 记录原文件与导出文件 SHA-256、大小和变换说明；"
        "Qwen 预测移除了 `text` 列，JSON 中机器路径已规范化。",
        "- 原始训练/测试文本、生成文本/API 响应、凭证、二进制分数、模型权重、"
        "依赖缓存和第三方 Notebook 均不上传。",
        "- 标准库复核：`python -m unittest -v test_results_archive.py`。",
        "- 重新导出：`python export_results.py --source /path/to/original/project`；"
        "依赖原始 artifacts，不会训练或请求 API。快照日期在导出器中明确固定。",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    export(args.source, args.output)
    print(f"Exported verified results to {args.output}")
