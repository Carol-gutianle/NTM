# LoRA / QLoRA：18 类文本来源分类

这里训练的是 **Qwen 底座 + LoRA + 18 类分类头**，不是让模型生成标签字符串。
输入只有原始 `text`，不加聊天模板、不使用 `id` 或 `prompt_id` 作为特征。
8 卡启动脚本默认使用本地 `Qwen3-32B`（见下一节）；文末历史单卡示例使用
`Qwen/Qwen2.5-3B`，也可以替换成兼容的
Qwen/Llama sequence-classification 模型；不是任意架构都兼容。

## 文件

- `run_lora_8gpu.sh`：本地 8 卡启动、环境预检、日志和完整检查点恢复。
- `train_lora.py`：单 GPU / DDP LoRA / NF4 4-bit QLoRA 训练与验证。
- `lora_runtime.py`：真实 GPU 遥测、持续低利用率告警、协作保存/退出。
- `predict_lora.py`：严格本地加载的推理脚本，输出 `submission.parquet`。
- `lora_common.py`：公共加载、分词和数据校验。
- `requirements-lora.txt`：实验依赖。
- `test_lora_smoke.py`：无需下载的微型随机 Qwen2 集成测试。

## 本地 8 卡首跑：Qwen3-32B

默认 backbone（严格离线，不会退回网上下载）：

```text
/apdcephfs_sgfd2/share_303378293/hunyuan/common/Qwen3-32B
```

先在**实际 GPU 节点**确认这个目录挂载可见，且为原始 Hugging Face 格式：
`config.json`、tokenizer 和完整 safetensors 权重分片。
当前开发会话无法访问该挂载、没有 GPU，不能据此保证 32B/8 卡可运行或不会 OOM。

```bash
# 已激活完整 GPU Python 环境时可以省略 PYTHON。
# 替换为 GPU 节点上实际存在、已安装 requirements-lora.txt 的解释器：
export PYTHON=/path/to/your/gpu-env/bin/python

# 仅检查本地模型、依赖和 8 张可见 GPU，不加载完整底座训练：
CHECK_ONLY=1 bash run_lora_8gpu.sh

# 正式训练 + 分组验证 + test.csv 预测
OUTPUT=artifacts/qwen3_32b_first bash run_lora_8gpu.sh
```

### 大模型加载加速：先缓存到节点本地盘

启动器现在默认 `LOCAL_MODEL_CACHE=auto`：在 torchrun 启动前由**一个进程**
将底座复制到节点本地缓存，再让 8 个 rank 加载这一个本地副本。
不会修改源模型，也不会改变 QLoRA/LoRA 算法。

自动选择只接受 `findmnt` 报告为 ext2/ext3/ext4/xfs/btrfs/zfs 的候选目录，
不默认相信 `/tmp`、overlay、tmpfs 或 FUSE 是高速本地盘。
这项检查也不能证明底层硬件一定是 NVMe；实际存储类型仍应向平台确认。
没有合适空间会明确报错，不会悄悄退回慢速共享盘。
源模型约 62 GB，另默认保留 8 GiB 空闲空间；不会自动删除其他缓存。

**建议显式指定平台提供的本地 NVMe 路径**（下面 `/local_nvme` 只是示例）：

```bash
# 可在拿到 GPU 训练时段前预缓存；不需要 torch/CUDA，Python 标准库即可
PYTHON=python LOCAL_MODEL_CACHE=/local_nvme/lora-models STAGE_ONLY=1 \
  bash run_lora_8gpu.sh

# 正式训练复用缓存，MODEL 仍是原共享路径，不要手工改为临时缓存路径
LOCAL_MODEL_CACHE=/local_nvme/lora-models OUTPUT=artifacts/qwen3_32b_cached \
  bash run_lora_8gpu.sh

# 续训时可以更换本地缓存根目录；必须保持 MODEL 和训练参数不变
LOCAL_MODEL_CACHE=/local_nvme/lora-models OUTPUT=artifacts/qwen3_32b_cached \
  RESUME=auto bash run_lora_8gpu.sh
```

缓存机制：
- 以源路径、文件尺寸和修改时间构建独立缓存标识；源变化使用新目录。
- 锁保证同一缓存同一时刻只有一个复制进程。
- 单文件顺序流式复制，每约 10 秒输出已复制字节和速度，不会再开 8 路远端读取。
- 复制时计算 SHA-256 并落盘；临时分片原子替换后才登记完成。
- 中断后跳过已完成且尺寸/时间一致的文件；未完成分片重新复制，不做字节级续传。
- 默认复用检查尺寸/时间；`CACHE_VERIFY=1` 会完整重算缓存哈希，代价是额外一次本地读盘。
- 训练保存原始 `MODEL` 作为底座标识，`model_load_path` 只记录这次的实际加载位置。

若平台确认容器 overlay 的底层确实是本地盘，可显式设置
`LOCAL_CACHE_ALLOW_UNVERIFIED=1`，但这不验证本地性，也不保证速度。
如没有本地盘且愿意接受原来的慢速加载，可设 `LOCAL_MODEL_CACHE=off`。
`CHECK_ONLY=1` 仍只执行模型/GPU 预检，不复制 62 GB；
`STAGE_ONLY=1` 才会填充缓存。

**缓存只能改善远端读取瓶颈，不能消除在线 4-bit 量化耗时。**
首次复制也需要时间，最好先于 GPU 训练时段完成；不会制造 GPU 负载保活。
启动前缓存日志输出到 stderr；需要保留时可在命令末尾加 `2>stage.log`。
正在运行的旧任务不受影响，新逻辑只在下次启动生效。

默认 **8 卡 QLoRA（NF4 4-bit）**，每卡 batch=1，梯度累积=2，
有效 batch=16；max_length=512，LoRA rank=16，5 epochs，
每 25 个 optimizer steps 验证并保存，保留最佳模型和最近检查点。
默认 2 个 DataLoader worker/卡，SDPA、动态 padding、梯度检查点。
这些只是保守起点，不是基于实际卡型调优过的最佳配置。

仓库固定依赖 `peft==0.19.1` / `transformers==4.51.3` 存在 DDP adapter
重载时错误进入 TP helper、导入缺失符号的问题。`lora_compat.py` 仅针对这对版本
跳过**无 tensor-parallel device mesh** 时本来就不需要的权重分片分支，
不修改 adapter 加载；检测到真实 TP 则拒绝。这条路径包含在双进程微型测试中。

**DDP 每张卡保存一份完整底座，不会把底座显存除以 8。**
如果卡的显存充足，需要未量化的普通 LoRA：

```bash
MODE=lora OUTPUT=artifacts/qwen3_32b_bf16 \
  bash run_lora_8gpu.sh --precision bf16 --batch-size 1
```

32B 未量化 BF16 权重本身约 60 GiB/卡，此外还有激活、CUDA、LoRA 和优化器开销。
预检里的显存估算仅供告警，不能保证不 OOM；本脚本没有实现 FSDP/ZeRO 参数分片。
首跑显存有余量后，再提高 `--batch-size`、相应降低 `--grad-accum`；
如需保持有效 batch=16，可用每卡 batch=2、累积=1。
不要为提高利用率盲目增加训练轮数或加入无效计算。

训练成功后检查 `SUCCESS.json`，结果在 `OUTPUT` 下：

```text
adapter/                     最佳 LoRA、18 类 score 分类头和 tokenizer
validation_metrics.json      本地验证准确率、balanced accuracy 等
validation_predictions.csv   验证集预测
submission.parquet           对 test.csv 的预测（不是隐藏集评分）
checkpoint-*/COMPLETE        所有 rank 完成保存的标记
gpu_metrics.jsonl            每张可见卡的真实利用率/显存采样
SUCCESS.json                 训练与结果导出均成功的标记
```

启动日志在输出目录旁边：`<OUTPUT>.launch.<时间>.log`。
`MODEL`、`OUTPUT`、`MODE`、`RESUME` 用环境变量设置，其他训练参数直接追加到命令。
不要让两个任务共享同一个输出目录。

### 平台 GPU 低利用率查杀

加载阶段若停在 `Loading checkpoint shards: 0/17`，不能只据此判定死锁：
进度条以完整分片为单位更新，8 个 rank 各自读取底座，QLoRA 还需量化。
可追加 `--load-debug-seconds 120`，在加载期间每 120 秒打印各 rank 的
Python 调用栈（诊断输出，不是额外的错误/超时退出）。
日志会显示每个 rank 的加载开始与完成耗时。
NCCL 初始化已显式绑定 `LOCAL_RANK` 对应 GPU；默认通信等待超时为 1800 秒，
可通过 `--ddp-timeout-seconds` 调整，但延长超时不能修复磁盘或通信故障。

监控线程只调用 `nvidia-smi`，**不会执行空转 kernel、伪造利用率或绕过调度器**。
默认从训练开始预留 180 秒 grace，之后每 10 秒采样；只在训练阶段统计，
任何一张可见卡持续低于 20% 达 180 秒，就告警并在下一个 optimizer step
边界请求 checkpoint。默认继续真实训练，也可选择保存后退出：

```bash
GPU_MIN_UTIL=30 GPU_LOW_SECONDS=120 GPU_GRACE_SECONDS=300 \
GPU_LOW_ACTION=stop OUTPUT=artifacts/qwen3_32b_guard \
  bash run_lora_8gpu.sh
```

这些是**用户可调的监控参数，不是已经查明的平台阈值**。
需向平台确认：逐卡还是平均利用率、统计窗口、初始化宽限期、SIGTERM
到 SIGKILL 间隔。模型加载（尤其 8 份 32B 共享盘读取）、验证和保存均可能低利用率；
监控不覆盖模型加载前的阶段，也不承诺能阻止平台回收。
优先申请初始化宽限期/正规长任务配额，把模型/数据预先放到节点高速存储。

不在训练中自动改变 batch/卡数：这会改变优化语义、采样和 DDP 同步；
先根据遥测及每步吞吐调参，再新建实验。不会无限自动重启失败任务。

手动安全停止（无需给 torchrun agent 发信号）：

```bash
touch artifacts/qwen3_32b_first/STOP_REQUEST
```

各 rank 在下一 optimizer step 边界同步退出请求，完成验证/保存后停止。
worker 收到 SIGTERM/SIGINT 也会请求此流程，但 torchrun/platform 可能在保存前
强制杀掉 worker；**SIGKILL、硬件故障、DDP 死锁无法保证现场保存**。
直接 `kill` agent 不等价于有充足宽限期的协作停止。

恢复时保留完全相同的训练参数及 8 卡数，删除 `STOP_REQUEST`（若存在）：

```bash
OUTPUT=artifacts/qwen3_32b_first RESUME=auto bash run_lora_8gpu.sh
```

只恢复最高 step 且有 `COMPLETE` 标记的 checkpoint，不使用可能被中断的保存。
停止/失败会返回非零退出码，不生成成功标记。验证用来选最佳 checkpoint，
所以验证分数不是独立测试分数。

无需下载的测试（仅 CPU 微型模型，不代表真实 32B 显存/精度）：

```bash
LORA_USE_LOCAL_DEPS=0 "$PYTHON" test_lora_runtime.py
LORA_USE_LOCAL_DEPS=0 SMOKE_QWEN3=1 "$PYTHON" test_lora_smoke.py
LORA_USE_LOCAL_DEPS=0 SMOKE_QWEN3=1 SMOKE_DDP=1 "$PYTHON" test_lora_smoke.py
```

## 1. GPU 环境

建议使用独立 Python 3.10/3.11 环境，并先安装适合服务器 CUDA/驱动的 PyTorch。
不要盲目替换服务器已经可用的 CUDA PyTorch；requirements 不固定 CUDA wheel。

```bash
python -m venv .venv-lora
source .venv-lora/bin/activate
# 先安装适合你的 CUDA/驱动的 PyTorch，再执行：
python -m pip install -r requirements-lora.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

默认 `precision=auto`：支持 BF16 时使用 BF16，否则 GPU 用 FP16。
FP16 下可训练参数保留 FP32，避免 GradScaler 对 FP16 梯度 unscale 报错。
不依赖 FlashAttention 安装，默认使用 SDPA。

**当前机器没有 GPU。CPU 微型模型测试验证代码链路，不代表真实 3B 模型精度、
GPU 显存占用、CUDA 混合精度或 bitsandbytes 4-bit 路径已实测。**
如需先核对服务器环境，可运行：

```bash
python test_lora_smoke.py
```

## 2. 首次训练：QLoRA

以下是低显存起始配置，不保证在任何显卡上都不 OOM。先跑一个 epoch
观察显存和速度，然后用新的输出目录正式跑 5 个 epoch。
首次使用 Hugging Face 模型 ID 需要联网下载公开底座；脚本不使用 Kaggle Token。

```bash
CUDA_VISIBLE_DEVICES=0 python train_lora.py \
  --model Qwen/Qwen2.5-3B \
  --train train.csv \
  --output artifacts/qlora_qwen25_3b_fold0 \
  --four-bit \
  --max-length 512 \
  --batch-size 1 \
  --eval-batch-size 2 \
  --grad-accum 16 \
  --rank 16 \
  --alpha 32 \
  --lr 1e-4 \
  --epochs 5
```

默认有效 batch size = 1 × 16 = 16；训练 LoRA 的 attention
`q_proj,k_proj,v_proj,o_proj` 和完整 `score` 分类头。
`score` 会和 adapter 一起保存，防止重载时丢失随机初始化后学到的分类头。
可用 `--target-modules all-linear` 扩大 LoRA 覆盖，作为后续实验，不是默认方案。

若显存充足，去掉 `--four-bit` 就是普通 LoRA。想改底座直接换 `--model`；
更换模型、参数或训练轮数请使用新的 `--output`。

若 OOM：先把 `--eval-batch-size` 改为 1，再考虑更小底座或缩短 max-length。
缩短输入会截断文本，截断比例记录在 `length_stats.json`。
支持 `torchrun` DDP，但多卡不会自动合并底座显存；单卡示例仍然只用一张卡。

## 3. 验证方式与输出

默认 `--fold 0 --seed 42`：与此前 TF-IDF/BERT 第一折完全相同的
`StratifiedGroupKFold(n_splits=5)`，2,159 条训练、540 条验证。
同一 `prompt_id` 的回答不会跨训练与验证。

每轮验证，按 Balanced Accuracy 保存最佳模型；连续 2 轮没有提升可提前停止。
验证集参与 checkpoint 选择，结果不是独立测试或 Kaggle 分数。

输出目录包含：

```text
adapter/                    # 最佳 LoRA + 分类头 + 分词器 + 推理配置
checkpoint-N/               # 每轮保存的训练状态，可续训；限制保留数量
run_config.json             # 参数、数据哈希、标签映射、分割信息
versions.json
split.csv
length_stats.json
history.json
training_metrics.json
validation_metrics.json
validation_predictions.csv
validation_logits.npy
classification_report.json
confusion_matrix.csv
```

`validation_metrics.json` 中的 `test_balanced_accuracy` 是 Trainer.predict 的
字段前缀；这里实际评估的是本地验证集，**不是隐藏测试集成绩**。

五折训练：

```bash
for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 python train_lora.py \
    --model Qwen/Qwen2.5-3B --four-bit --fold "$fold" \
    --output "artifacts/qlora_fold${fold}" --epochs 5
done
```

### 断点续训

保留原参数和输出目录，追加 checkpoint 路径：

```bash
CUDA_VISIBLE_DEVICES=0 python train_lora.py \
  --model Qwen/Qwen2.5-3B --four-bit \
  --output artifacts/qlora_qwen25_3b_fold0 \
  --epochs 5 \
  --resume artifacts/qlora_qwen25_3b_fold0/checkpoint-135
```

checkpoint 编号仅为例子，以实际目录为准。这里用于恢复中断的同一次训练，
不是把已经完成的 5 轮直接改成 10 轮；关键配置不一致会拒绝续训。

## 4. 离线推理 / Kaggle

**adapter 不包含冻结底座权重！** 必须同时准备与训练完全相同版本的
原始底座目录和 `adapter/`。不要把别的模型、量化发布版或不同版本权重混用。
模型提交版本可通过训练的 `--revision <commit>` 固定；adapter 的
`inference_config.json` 也记录了解析出的 commit。

如要统一使用本地模型目录，可在联网环境先下载（需准备足够磁盘）：

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-3B",
    local_dir="artifacts/pretrained/Qwen2.5-3B",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
```

之后训练用 `--model artifacts/pretrained/Qwen2.5-3B --offline`。
将该目录原样作为正式推理的底座。

```bash
CUDA_VISIBLE_DEVICES=0 python predict_lora.py \
  --adapter artifacts/qlora_qwen25_3b_fold0/adapter \
  --base-model artifacts/pretrained/Qwen2.5-3B \
  --four-bit \
  --test test.csv \
  --output submission.parquet
```

建议推理的 `--four-bit` 与训练保持一致；切换量化模式可能改变结果。
最大文本长度自动读取训练保存的配置，不会默默换成其他长度。
所有推理权重与分词器加载均使用 `local_files_only=True`。

Kaggle 中，把上述模型和必要脚本作为输入附件，并提前准备可离线安装的依赖。
Notebook 调用示例（替换 BASE / ADAPTER / CODE 实际目录）：

```python
import subprocess, sys
subprocess.run([
    sys.executable, "/kaggle/input/CODE/predict_lora.py",
    "--adapter", "/kaggle/input/ADAPTER/adapter",
    "--base-model", "/kaggle/input/BASE",
    "--four-bit",
    "--test", "/kaggle/input/competitions/name-that-llm/test.csv",
    "--output", "/kaggle/working/submission.parquet",
], check=True)
```

`lora_common.py` 应和 `predict_lora.py` 在同一目录。
不要上传整个工作目录，尤其不要上传 `.env`、API Token 或不需要的 checkpoint。
正式隐藏集耗时需在实际 GPU 上评估，不能用 3 条示例的运行时间代替。

## 5. 全量重训

先通过分组验证选择配置和训练轮数，再使用 `--full-train`：

```bash
# 3 轮仅是示例，请换成验证确定的轮数。
CUDA_VISIBLE_DEVICES=0 python train_lora.py \
  --model Qwen/Qwen2.5-3B --four-bit \
  --full-train --epochs 3 --output artifacts/qlora_full
```

该模式没有验证集，不早停，保存最后一轮 adapter；不要把训练指标当作验证分数。
代码不会自动提交 Kaggle。
