# 实验结果汇总

整理日期：2026-10-27；实验记录：2026-10-22 至 2026-10-23。本次仅整理存量结果，没有重训或提交竞赛。

## 验证协议

- 2,699 条有标签样本、150 个 prompt、18 类；仅使用回答文本作为模型输入。
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` 第一折：2,159 条训练 / 540 条验证，120 / 30 个 prompt，无 prompt 交集。
- 下表指标由已归档预测重新计算，并校验 ID、标签、prompt 和可用的分类报告。540 条验证数据每类 30 条，因此 Accuracy 与 Balanced accuracy 相同。
- 同一验证集反复用于 checkpoint、超参数和融合权重选择；这些是探索性验证分数，**不是独立测试、五折均值或 Kaggle 榜单分数**。

## 同一 holdout 对比

| 方法 | Balanced accuracy | Macro F1 | 选择/实验说明 |
|---|---:|---:|---|
| BERT + character SVM score fusion | 35.74% | 34.55% | 5 weights; validation-selected |
| Qwen3-32B LoRA / 8 GPUs | 35.19% | 32.78% | checkpoint selected on validation |
| Hard vote / 3 models | 34.63% | 33.57% | exploratory reused validation |
| Hard vote / 5 models | 34.44% | 33.77% | exploratory reused validation |
| Hard vote / 7 models | 34.26% | 33.35% | exploratory reused validation |
| Character TF-IDF + LinearSVC | 31.85% | 30.86% | fixed baseline |
| TF-IDF MLP | 31.67% | 30.96% | checkpoint selected on validation |
| Character + word TF-IDF + LinearSVC | 30.37% | 28.96% | fixed baseline |
| Character SVM + 1,050 external rows | 28.70% | 28.11% | partial external augmentation |
| Character SVM + 2,025 external rows | 28.33% | 27.54% | partial external augmentation |
| BERT GPU / 512 tokens | 27.59% | 25.44% | checkpoint selected on validation |
| Word TF-IDF + LinearSVC | 26.85% | 24.93% | fixed baseline |
| BERT CPU / 256 tokens | 24.26% | 19.69% | checkpoint selected on validation |
| XGBoost / character TF-IDF | 22.78% | 22.73% | reused validation |
| KNN / character TF-IDF | 15.74% | 14.32% | 10 candidates; validation-selected |

精确值与逐项证据路径见 `summary.csv` / `summary.json`。

## 五折 grouped CV（单独报告）

| Fold | 验证样本数 | Balanced accuracy |
|---|---:|---:|
| 1 | 540 | 31.85% |
| 2 | 540 | 32.96% |
| 3 | 540 | 32.78% |
| 4 | 539 | 31.12% |
| 5 | 540 | 28.15% |

- 折均值 **31.37%**；总体标准差 **1.74 个百分点**。
- pooled OOF balanced accuracy：**31.37%**；每条样本有一次 OOF 预测。
- 字符 TF-IDF 配置事先根据第一折选择，因此五折结果也不是完全未触碰的最终评估。

## 结论与限制

- 当前 holdout 最高是 BERT + 字符 SVM 分数融合：35.74%，BERT 权重 0.5，SVM 权重 0.5。分数经逐样本标准化、softmax 后融合，并非概率校准。
- Qwen3-32B LoRA 为单模型最高：35.19%；8 GPU、BF16、非 4-bit、rank 16、alpha 32、max length 512，选择 checkpoint-300。日志键 `test_*` 实际指上述验证集。
- 3/5/7 模型硬投票均未超过分数融合。BERT/SVM oracle 47.78% 使用真实标签选模型，不可作为实际可部署成绩。
- 2,025 条外部增强覆盖 12/18 类，未达到每类 200 条；8 类达到 200 条。加入增强后 31.85% → 28.33%（下降 3.52 个百分点）；修正 24 条、退化 43 条。这是特定增强配方的负结果，不证明所有增强都无效。
- 增强生成记录 2,144 次请求/响应、733,495 reported tokens、reported cost 61.124810224（币种未知）；存在未知 reasoning usage 和网关模型身份未独立核验的问题。
- BERT GPU 训练集 accuracy 65.68%，验证 27.59%；训练诊断不能当泛化能力。其余训练诊断保存在对应 evidence 子目录。
- LLM Decoder Qwen v8 仅完成源码准备和预检；预检未通过，缺核心模型/数据附件，**没有端到端复现成绩**。上游 Notebook、代码和权重不随本仓库重新发布。
- `qwen3_32b_first` 仅留有配置/长度信息，不能视为完成训练；`augmentation_eval_pipeline_check` 是流程检查，不列入主结果表。
- 本地 test.csv 仅 3 条样例；本次不发布这些 smoke submission，归档也不提供可核验的隐藏测试榜单分数。

## 归档与复核

- `evidence/` 保留原 run 目录名，包括指标、分类报告、混淆矩阵、去文本的预测、训练配置/历史、CV 分折和增强完成状态。
- `manifest.json` 记录原文件与导出文件 SHA-256、大小和变换说明；Qwen 预测移除了 `text` 列，JSON 中机器路径已规范化。
- 原始训练/测试文本、生成文本/API 响应、凭证、二进制分数、模型权重、依赖缓存和第三方 Notebook 均不上传。
- 标准库复核：`python -m unittest -v test_results_archive.py`。
- 重新导出：`python export_results.py --source /path/to/original/project`；依赖原始 artifacts，不会训练或请求 API。快照日期在导出器中明确固定。
