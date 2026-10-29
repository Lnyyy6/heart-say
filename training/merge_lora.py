"""把 LoRA adapter 合并回 fp16 基座，供 AWQ 量化或直接部署。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="outputs/dpo-qwen7b")
    parser.add_argument("--output", default="outputs/merged-qwen7b")
    args = parser.parse_args()

    print("加载基座（fp16）…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print("合并 adapter…")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
