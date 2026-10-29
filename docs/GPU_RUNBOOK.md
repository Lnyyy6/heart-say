# GPU 运行手册：SFT / DPO / AWQ / vLLM / 评测

> 当前开发机没有 CUDA GPU。1.5B SFT/DPO 已在 Apple M1 Pro 的 MPS 上真实跑通（数字见 0.1）；7B、AWQ 与 vLLM 性能仍需下面的单卡 24GB Linux 环境，跑完把真实数字填进表格，不要用估算值。

## 0.1 无 CUDA：Apple Silicon / MPS 真实训练（已执行）

```bash
# SFT：fp32 LoRA，r16/α32，batch 1 + grad 2，seq 512
python training/sft_qlora.py --config training/configs/sft_qwen1.5b.yaml \
  --device mps --no-4bit --max-steps 30 --max-seq-length 512

# DPO：在 SFT adapter 上继续训练
python training/dpo.py --config training/configs/dpo_qwen1.5b.yaml \
  --device mps --no-4bit --max-steps 15 --max-length 384 --max-prompt-length 192

# 用同一套 100 条评测（base / SFT / DPO）
LLM_BACKEND=transformers TRANSFORMERS_ADAPTER_PATH= \
  PYTHONPATH=src python -m heart_say.evaluation.runner --name base_mps --concurrency 1
LLM_BACKEND=transformers TRANSFORMERS_ADAPTER_PATH=outputs/sft-qwen1.5b \
  PYTHONPATH=src python -m heart_say.evaluation.runner --name sft_mps --concurrency 1
LLM_BACKEND=transformers TRANSFORMERS_ADAPTER_PATH=outputs/dpo-qwen1.5b \
  PYTHONPATH=src python -m heart_say.evaluation.runner --name dpo_mps --concurrency 1
```

实测（M1 Pro 16GB，fp32 LoRA，非 4-bit）：

| 阶段 | 数据 | 实际训练 | 耗时 | 峰值显存 | train loss | 评测均分 | 通过率 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 60 条 | 1 epoch / 30 step | 23.6 min | 10.31 GB | 2.315 | 4.33 | 96% |
| DPO | 30 对 | 1 epoch / 15 step | 1.1 min | 9.47 GB | 0.015 | 4.31 | 98% |

注意：这是 `transformers` + MPS 的本地串行推理，不是 vLLM 性能；P95 受共享机器负载影响，只作相对参考。产物在 `outputs/`（gitignore），报告在 `reports/eval_*_mps.md`。

## 0.2 环境（CUDA）

```bash
nvidia-smi
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r training/requirements-train.txt
pip install vllm autoawq
```

验收：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python training/sft_qlora.py --config training/configs/sft_qwen7b.yaml --dry-run
python training/dpo.py --config training/configs/dpo_qwen7b.yaml --dry-run
```

## 1. 数据

```bash
python scripts/data_report.py --check
python scripts/build_training_data.py
```

当前 v0：60 条 SFT / 30 条偏好对 / 80 篇知识卡 / 100 条评测 / 50 条红队。

目标规模仍差：SFT 1k–2k、偏好 500–1k、知识片段 500–1000。补齐方式：

```bash
# 用强模型扩充（需要 OpenAI 兼容 API Key）
OPENAI_API_KEY=... python scripts/synthesize_data.py \
  --base-url https://api.deepseek.com/v1 --model deepseek-chat --kind sft --n 100 --batches 10
OPENAI_API_KEY=... python scripts/synthesize_data.py \
  --base-url https://api.deepseek.com/v1 --model deepseek-chat --kind pairs --n 50 --batches 10

# 再清洗、去重、PII 脱敏
python scripts/build_training_data.py
python scripts/data_report.py --check
```

> 合成后必须人工抽检 10–20%，并检查危机/用药/诊断边界；不要直接喂。

## 2. 训练

小显存降级：

```bash
# 3B
SFT_CONFIG=training/configs/sft_qwen3b.yaml \
DPO_CONFIG=training/configs/dpo_qwen3b.yaml \
bash scripts/run_gpu_pipeline.sh train

# 1.5B
SFT_CONFIG=training/configs/sft_qwen1.5b.yaml \
DPO_CONFIG=training/configs/dpo_qwen1.5b.yaml \
bash scripts/run_gpu_pipeline.sh train
```

7B 全流程：

```bash
bash scripts/run_gpu_pipeline.sh all
```

或分步：

```bash
python training/sft_qlora.py --config training/configs/sft_qwen7b.yaml
python training/dpo.py --config training/configs/dpo_qwen7b.yaml
python training/merge_lora.py --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/dpo-qwen7b --output outputs/merged-qwen7b
python training/quantize_awq.py --model outputs/merged-qwen7b --output outputs/qwen7b-awq
```

每个阶段会产出：

- `outputs/sft-qwen7b/metrics.jsonl`：step / loss / lr / elapsed / peak_vram
- `outputs/sft-qwen7b/training_summary.json`
- `outputs/dpo-qwen7b/training_summary.json` + reward history
- `outputs/merged-qwen7b/`、`outputs/qwen7b-awq/`

DPO 实验：固定 SFT 与数据，只扫 beta：

```bash
python training/dpo.py --config training/configs/dpo_qwen7b.yaml --beta 0.05
python training/dpo.py --config training/configs/dpo_qwen7b.yaml --beta 0.1
python training/dpo.py --config training/configs/dpo_qwen7b.yaml --beta 0.3
```

如果 DPO 没提升，保留负结果：对比 reward accuracy、偏好对噪声、SFT 是否饱和、评测是否敏感。不要删失败记录。

## 3. 启动 vLLM

```bash
bash scripts/serve_vllm.sh outputs/qwen7b-awq heart-say-qwen
# 默认 http://localhost:8001/v1
```

确认：

```bash
curl -s http://localhost:8001/v1/models
```

## 4. 评测：base / SFT / DPO / AWQ

每次换模型后，用同一批 100 条评测与 50 条红队。

```bash
# 以 SFT 为例；base / dpo / awq 只换 --name 与 vLLM served-model-name
LLM_BACKEND=openai_compatible \
LLM_BASE_URL=http://localhost:8001/v1 \
LLM_MODEL=heart-say-qwen \
LLM_API_KEY=EMPTY \
QDRANT_PATH= QDRANT_URL= \
python -m heart_say.evaluation.runner --name sft

# 安全回归
LLM_BACKEND=openai_compatible LLM_BASE_URL=http://localhost:8001/v1 \
LLM_MODEL=heart-say-qwen LLM_API_KEY=EMPTY \
QDRANT_PATH= QDRANT_URL= \
python scripts/run_redteam.py --name sft

# 性能
python scripts/benchmark_inference.py --mode openai \
  --url http://localhost:8001/v1 --model heart-say-qwen \
  --label sft-awq --requests 30 --concurrency 4
```

推荐四组：

```bash
# base 模型（未 SFT）先跑一次
python -m heart_say.evaluation.runner --name base
python scripts/run_redteam.py --name base

# SFT / DPO / AWQ 各跑一次
python -m heart_say.evaluation.runner --name sft
python -m heart_say.evaluation.runner --name dpo
python -m heart_say.evaluation.runner --name awq
```

生成对比：

```bash
python -m heart_say.evaluation.compare \
  --baseline reports/eval_base.json --rag reports/eval_sft.json \
  --left-label Base --right-label SFT --title "Base vs SFT" \
  --output reports/base_vs_sft.md

python -m heart_say.evaluation.compare \
  --baseline reports/eval_sft.json --rag reports/eval_dpo.json \
  --left-label SFT --right-label DPO --title "SFT vs DPO" \
  --output reports/sft_vs_dpo.md
```

## 5. 结果表（跑完后填写）

| 模型 | 量化 | Eval 均分 | 通过率 | 事实/引用 | 红队 | TTFT P95 | 总延迟 P95 | 显存 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B base | - | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| SFT | - | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| DPO | - | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| DPO + AWQ | 4-bit | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 6. 验收清单

- [ ] `metrics.jsonl` 有真实 loss / 显存 / 时长
- [ ] SFT、DPO 各有一份 `training_summary.json`
- [ ] base / SFT / DPO / AWQ 四组 eval 与 redteam
- [ ] `reports/base_vs_sft.md`、`reports/sft_vs_dpo.md`
- [ ] AWQ 的 TTFT/P95/吞吐/显存报告
- [ ] 若 DPO 负结果，写清实验条件与原因
- [ ] 人工抽检 10–20% 并记录一致性

## 7. 常见问题

- **OOM**：把 `per_device_train_batch_size` 降到 1，增大 `gradient_accumulation_steps`；或换 3B/1.5B 配置。
- **bitsandbytes 报错**：确认 CUDA 版本与 torch 匹配；优先 Linux，不要用 macOS 跑 4-bit。
- **DPO 报 tokenizer 弃用**：当前脚本按 TRL 0.12.2 固定，若升级 TRL 需同步改参数名。
- **AWQ 校准太慢/显存不足**：减少校准样本或换 3B；量化后必须重跑 eval/redteam。
- **vLLM 端口冲突**：`PORT=8002 bash scripts/serve_vllm.sh ...`，同时改 `LLM_BASE_URL`。
