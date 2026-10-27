# NTM — Name That LLM

根据回答文本识别生成模型的实验代码与结果归档。

**2026-10-27 更新：**整理了 2026-10-22 至 2026-10-23 的现有实验；
本次没有重新训练、调用付费 API 或提交 Kaggle 预测。

## 结果入口

- **`results/README.md`**：完整对比、验证协议、负结果和未完成事项。
- `results/summary.csv` / `results/summary.json`：15 组同一 holdout 实验的机器可读汇总。
- `results/evidence/`：指标、去文本预测、混淆矩阵、配置和训练历史。
- `results/manifest.json`：原始文件及导出文件校验和。

| 实验 | Balanced accuracy | 范围 |
|---|---:|---|
| BERT + 字符 SVM 分数融合 | **35.74%** | 同一验证集上选权重 |
| Qwen3-32B LoRA | **35.19%** | 同一验证集上选 checkpoint |
| 字符 TF-IDF + LinearSVC | **31.85%** | 第一折 holdout |
| 字符 TF-IDF + LinearSVC | **31.37% ± 1.74 pp** | 五折均值 ± 总体标准差 |
| 字符 SVM + 2,025 条外部增强 | **28.33%** | 第一折 holdout，增强未覆盖全部类 |

数据共 2,699 条、150 个 prompt、18 类，按 prompt 分组划分。
Holdout 为 2,159 条训练 / 540 条验证，无 prompt 交集。
同一验证集被多次用于模型选择；**以上均不是隐藏测试集或 Kaggle 榜单分数**。
五折配置也曾用第一折选择，不能视为完全未触碰的最终评估。
模型名称在此为本地数据集标签或配置记录，不代表对网关后端身份的独立认证。

## 目录与文档

| 内容 | 代码 / 文档 |
|---|---|
| TF-IDF / BERT 基线与 CV | `baselines.py`、`BASELINE_RESULTS.md` |
| TF-IDF MLP / XGBoost / KNN | `tfidf_mlp.py`、`train_xgboost.py`、`train_knn.py` |
| 单 GPU BERT、诊断与分数融合 | `BERT_README.md` |
| Qwen LoRA、多 GPU 启动与缓存 | `LORA_README.md` |
| 硬投票融合 | `ENSEMBLE_README.md` |
| 外部数据增强及固定 split 对比 | `AUGMENTATION_README.md` |
| Decoder Qwen v8 复现准备（未完成） | `DECODER_REPRODUCTION.md` |
| 原运行环境记录 | `SHARED_ENV.md` |
| 结果导出与离线校验 | `export_results.py`、`test_results_archive.py` |

根目录实验说明保留历史运行语境；**跨实验的状态和完整结果以 `results/README.md` 为准**。
历史路径、默认模型位置和依赖版本并不保证适用于另一台机器。

## 只查看和核验结果（无需模型、原始数据或 GPU）

```bash
python -m unittest -v test_results_archive.py
```

导出的 Qwen 验证预测已去除 `text` 列；所有 holdout 指标均从预测重新计算，
并核对相同的 ID、prompt、真实标签以及可用分类报告。

## 重跑实验

1. 自行从有权限的数据来源取得 `train.csv`、`test.csv`，放到项目根目录。
   训练数据需包含 `id,prompt_id,text,model`，测试数据至少包含 `id,text`。
   历史训练数据 SHA-256 记录在 `results/evidence/split_metadata.json`。
2. 在独立 Python 环境安装所选实验的依赖；先安装适合硬件的 PyTorch。
   `requirements-baselines.txt` 与 `requirements-lora.txt` 为不同历史环境记录，
   **不要直接合并安装**（部分版本约束不同）。XGBoost 实验另需安装 XGBoost。
3. 按对应文档准备模型，检查机器路径、GPU 分配和参数后再启动。
   不随仓库发布训练权重，已有结果不能直接用于新文本推理。

例如重跑字符、词、组合 TF-IDF 及字符五折 CV：

```bash
python baselines.py tfidf --threads 4
python baselines.py tfidf-cv --threads 4
```

训练输出写入被忽略的 `artifacts/`；只读历史证据位于 `results/evidence/`，
不是模型运行目录，不能将两者混用。

## 发布边界

不发布 `.env`、API 凭证、原始竞赛文本、生成文本/API 响应、依赖缓存、
模型/checkpoint、大型二进制分数及第三方 Notebook。
原始数据、模型与第三方资产仍受各自授权约束，本仓库不对它们重新授权。
保留去文本的样本 ID、prompt ID、标签和预测，仅用于核验本地实验。

归档导出命令：

```bash
python export_results.py --source /path/to/original/project
```

此操作不训练、不反序列化模型、不访问网络，也不会读取 `.env`。
