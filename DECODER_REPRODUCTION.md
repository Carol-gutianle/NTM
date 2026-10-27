# LLM Decoder Qwen v8 复现

## 范围与状态

复现 `lynnsakurai/llm-decoder-qwen-ver` **v8** 的原版推理流程：
四视角 SVC、two-stage LoRA、同题 Gaussian、辅助模型集成、ATLAS
混合融合与最终类别平衡。不是重新训练全部模型，也不是简化替代模型。

`reproduce_decoder.py` 校验原始 Notebook SHA-256，仅替换 `/kaggle/input`
路径；按原顺序执行代码单元，包括其中通过字符串嵌入的 Gaussian 集成。
模型代码只有显式执行 `run` 才会运行，`prepare` 不执行下载的代码。

**当前不能承诺逐位一致或同榜单成绩。** API 没有固定所有数据集和
上游 Notebook 输出版本；还必须取得原权重，匹配原运行环境并实际推理。
公开 `test.csv` 只是样例，不能据此计算隐藏测试集分数。
原始文件及其作者归属保留在 `artifacts/decoder_qwen_v8/`，未加入版本控制；
本项目不对上游文件重新授权。

## 1. 获取固定源码

```bash
python reproduce_decoder.py prepare
# 或使用已下载的 API 响应，完全离线：
python reproduce_decoder.py prepare --api-response /tmp/llm-decoder-qwen.json
```

下载的版本或哈希发生变化会报错，不会默默使用新版。
生成 `original.ipynb`、便于检查的 `original.py`、
`public_gaussian_stack.py`、`metadata.json` 和 `provenance.json`。

## 2. 准备输入

默认输入目录：

```text
artifacts/decoder_qwen_v8/input/
  name-that-llm/
    train.csv
    test.csv
  ...其他数据集、基础模型和训练输出...
```

也可用 `--input-root /your/input` 指定任意独立输入目录。
允许软链接，但不要用整个项目或整个共享盘作输入根目录：
原版递归查找并有“只允许一个 ATLAS/完整 LoRA24”的检查。

精确输入来源见 `metadata.json` 中以下字段：

- `datasetDataSources`
- `kernelDataSources`
- `modelDataSources`
- `competitionDataSources`

核心附件包括：

| 类别 | 内容 |
|---|---|
| 数据 | 官方 train/test、`pool_v1.parquet`、辅助集成的公共增强文本 |
| two-stage LoRA | `lora8b_v1`、`lora14b_v1` 完整目录 |
| 原始 LoRA | `lora24/final`、seed7、Claude head、Qwen2.5 instyle |
| 基础模型 | Qwen3 4B/8B/14B、Qwen2.5 7B、GLM4 9B、Qwen3.5 |
| Encoder | DeBERTa-v3-large、ELECTRA-large、Desklib |
| 其他分类器 | Qwen/DeBERTa token-attention、UltraBERT 五折 |
| ATLAS | `atlas_bundle.json` 及其列出的文件和包 |

**上游依赖存在缺口：** API 的 `kernelDataSources` 有一个空字符串。
原 Notebook 的错误提示指向
`lightningv08/ntl-lora24-fivefold-training` 的完整 `lora24/final` 输出。
该提示并不证明此输出仍可访问；下载前需要核实，不能用本地其他 LoRA
冒充这个 checkpoint。缺少 `COMPLETE.json` 本身可以接受，但模型、
heads、readouts、anchor features、tokenizer 必须完整。

目前不会自动下载全部模型：体积很大，上游版本也未全部固定。
本工具不会自动读凭证、提交竞赛预测或改动既有训练环境。

## 3. 环境与检查

推荐在独立环境中复现原 Kaggle 镜像；镜像摘要记录在 `provenance.json`。
不要直接覆盖项目原有 `transformers` 等依赖。

运行涉及 PyTorch、Transformers、PEFT、Accelerate、NumPy、pandas、
SciPy、scikit-learn、LightGBM、safetensors 和 pyarrow。
还要求环境能够识别原版使用的模型配置，包括 `qwen3_5` 和 `ultrabert`。
不能仅凭“安装了 Transformers”判断兼容；ATLAS 也可能有额外要求。
这里没有编造一份声称可精确复现的版本锁定文件。

```bash
python reproduce_decoder.py check
# 实际用哪个 Python 推理，就用哪个 Python 检查：
/path/to/gpu-env/bin/python reproduce_decoder.py check --input-root /your/input
```

检查缺输入时退出码为 2，并写入 `preflight.json`。这是初步检查，
不替代原 Notebook 的完整权重、标签顺序、ATLAS 哈希和 anchor parity 检查。

## 4. GPU 推理

```bash
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  /path/to/gpu-env/bin/python -u reproduce_decoder.py run \
  --input-root /your/input \
  --output artifacts/decoder_qwen_v8/run1
```

原版按 GPU 数量分配部分大模型，实际显存、速度和数值兼容性需要实测。
有其他任务时请先分配好 `CUDA_VISIBLE_DEVICES`，不要直接占用全部 GPU。
输出目录不能已有 `submission.parquet`；脚本不会覆盖已有预测。
原版数值计算、分类权重、分组、融合和类别平衡没有简化或重拟合。

最后生成 `submission.parquet`。本地运行成功只证明推理执行成功，
不等于隐藏榜单成绩复现。该 Notebook 不包含完整上游 LoRA 训练流程。

## 5. 本地测试

```bash
python -m unittest -v test_reproduce_decoder.py
```

测试覆盖源码完整性检查、内嵌源码路径替换、LoRA 必需文件检查和
原 Notebook 全部代码单元的编译，不会运行 GPU 模型，也不代表预测通过验证。
