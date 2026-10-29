#!/usr/bin/env bash
# 用 vLLM 提供 AWQ 模型。需先训练+量化，或在 HuggingFace 有现成 AWQ 模型。
set -euo pipefail

MODEL="${1:-outputs/qwen7b-awq}"
MODEL_NAME="${2:-heart-say-qwen}"
PORT="${PORT:-8001}"
GPU_MEMORY="${GPU_MEMORY:-0.9}"

if command -v vllm >/dev/null 2>&1; then
  exec vllm serve "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --quantization awq \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization "$GPU_MEMORY"
else
  echo "vLLM 未安装。如需云 GPU 部署：pip install vllm" >&2
  exit 1
fi