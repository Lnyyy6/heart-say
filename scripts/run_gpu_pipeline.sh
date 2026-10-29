#!/usr/bin/env bash
# 在 GPU 节点上跑训练/合并/量化。不会自动启动 vLLM 或评测，避免端口/权限混乱。
#
# 用法：
#   bash scripts/run_gpu_pipeline.sh all
#   bash scripts/run_gpu_pipeline.sh train
#   bash scripts/run_gpu_pipeline.sh merge
#   bash scripts/run_gpu_pipeline.sh quant
#
# 可用环境变量覆盖：
#   SFT_CONFIG=training/configs/sft_qwen3b.yaml
#   DPO_CONFIG=training/configs/dpo_qwen7b.yaml
#   MERGED_DIR=outputs/merged-qwen7b
#   AWQ_DIR=outputs/qwen7b-awq
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SFT_CONFIG="${SFT_CONFIG:-training/configs/sft_qwen7b.yaml}"
DPO_CONFIG="${DPO_CONFIG:-training/configs/dpo_qwen7b.yaml}"
MERGED_DIR="${MERGED_DIR:-outputs/merged-qwen7b}"
AWQ_DIR="${AWQ_DIR:-outputs/qwen7b-awq}"
STAGE="${1:-all}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "未检测到 nvidia-smi。请在 Linux + CUDA 的 GPU 机器上运行。" >&2
  exit 1
fi

echo "[1/4] 清洗与合并训练数据"
python scripts/build_training_data.py

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  echo "[2/4] QLoRA SFT（$SFT_CONFIG）"
  python training/sft_qlora.py --config "$SFT_CONFIG"

  echo "[3/4] DPO（$DPO_CONFIG）"
  python training/dpo.py --config "$DPO_CONFIG"
fi

BASE_MODEL="$(DPO_CONFIG="$DPO_CONFIG" python -c 'import os,yaml;print(yaml.safe_load(open(os.environ["DPO_CONFIG"],encoding="utf-8"))["base_model"])')"
ADAPTER="$(DPO_CONFIG="$DPO_CONFIG" python -c 'import os,yaml;print(yaml.safe_load(open(os.environ["DPO_CONFIG"],encoding="utf-8"))["output_dir"])')"

if [[ "$STAGE" == "all" || "$STAGE" == "merge" ]]; then
  echo "[4/4] 合并 LoRA → $MERGED_DIR"
  python training/merge_lora.py --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$MERGED_DIR"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "quant" ]]; then
  echo "[5/5] AWQ 4-bit → $AWQ_DIR"
  python training/quantize_awq.py --model "$MERGED_DIR" --output "$AWQ_DIR"
fi

cat <<EOF

训练阶段完成。下一步：

1) 启动 vLLM：
   bash scripts/serve_vllm.sh "$AWQ_DIR" heart-say-qwen

2) 用同一套评测对比 base / SFT / DPO / AWQ：
   LLM_BACKEND=openai_compatible LLM_BASE_URL=http://localhost:8001/v1 LLM_MODEL=<served-name> \\
   QDRANT_PATH= QDRANT_URL= python -m heart_say.evaluation.runner --name <model-label>

3) 安全与性能：
   LLM_BACKEND=openai_compatible ... python scripts/run_redteam.py --name <model-label>
   python scripts/benchmark_inference.py --mode openai --url http://localhost:8001/v1 \\
     --model <served-name> --label <model-label>-awq

详见 docs/GPU_RUNBOOK.md。
EOF
