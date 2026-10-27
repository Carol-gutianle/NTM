# Shared GPU environment

- Environment: `/apdcephfs_sgfd2/share_303378293/tianlelegu/ntl`
- Project: `/apdcephfs_sgfd2/share_303378293/tianlelegu/name_that_llm`
- Python 3.10; CUDA-enabled PyTorch pinned by `constraints-gpu-cu124.txt`.
- LoRA dependencies: `requirements-lora.txt`.
- No GPU is present on the installation node. Run GPU checks on the actual GPU node.

```bash
cd /apdcephfs_sgfd2/share_303378293/tianlelegu/name_that_llm
PY=/apdcephfs_sgfd2/share_303378293/tianlelegu/ntl/bin/python
$PY -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
bash run_lora_gpu.sh
```

The launcher uses absolute paths, so activation is not required. For an interactive shell:

```bash
conda activate /apdcephfs_sgfd2/share_303378293/tianlelegu/ntl
```

Additional launcher arguments override defaults, for example:

```bash
bash run_lora_gpu.sh --model /path/to/local/Qwen2.5-3B --offline
```

HF cache defaults to `artifacts/hf-cache` on shared storage. The Qwen base model
has not been downloaded by environment setup; first training downloads it unless
a local base model path and `--offline` are supplied.

The project was copied from `/data/home/tianlelegu/name_that_llm`, including
local dependency directories, training data, existing model artifacts, and
`.env` with mode 600. The original project and environment remain unchanged.
Never publish `.env` or upload the entire project as a Kaggle dataset.

Use the CUDA driver on the actual training node; a CUDA-enabled PyTorch wheel
does not install or replace the NVIDIA driver. If GPU checks fail, diagnose the
driver/GPU compatibility before starting training.
