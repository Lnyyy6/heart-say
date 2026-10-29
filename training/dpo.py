"""DPO：在 SFT adapter 上继续对齐偏好。

用法：
  python training/dpo.py --config training/configs/dpo_qwen7b.yaml --dry-run
  python training/dpo.py --config training/configs/dpo_qwen7b.yaml --beta 0.05

实验建议：固定 SFT 模型与数据，只扫 beta（0.05 / 0.1 / 0.3），
每个配置跑完整 eval，若没有提升就如实记录负结果。
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from training_utils import (
    JsonlMetricsCallback,
    gpu_peak_memory_gb,
    load_yaml,
    validate_preference,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5 DPO")
    parser.add_argument("--config", default="training/configs/dpo_qwen7b.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    count, problems = validate_preference(config["dataset_path"])
    print(f"偏好对：{count}；问题：{len(problems)}")
    for problem in problems[:10]:
        print(" -", problem)
    if args.dry_run:
        if problems:
            raise SystemExit("数据校验未通过")
        print("dry-run 通过。beta =", args.beta or config["training"]["beta"])
        return
    if problems:
        raise SystemExit("数据校验未通过")

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    train_cfg = config["training"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    beta = args.beta or train_cfg["beta"]

    is_mps = args.device == "mps" or (
        args.device == "auto" and torch.backends.mps.is_available() and not torch.cuda.is_available()
    )
    use_4bit = not args.no_4bit and torch.cuda.is_available() and not is_mps
    device_name = "mps" if is_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DPO device={device_name}，4bit={use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if is_mps:
            model = model.to("mps")
    model = PeftModel.from_pretrained(model, config["sft_adapter"], is_trainable=True)
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
    print(f"偏好对：{len(dataset)}，beta={beta}")

    max_length = args.max_length or train_cfg["max_length"]
    max_prompt_length = args.max_prompt_length or train_cfg["max_prompt_length"]
    if is_mps:
        max_length = min(max_length, 512)
        max_prompt_length = min(max_prompt_length, 256)
    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        beta=beta,
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=1 if is_mps else train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=train_cfg["bf16"] and not is_mps,
        optim="adamw_torch" if is_mps else train_cfg["optim"],
        report_to="none",
        remove_unused_columns=False,
        seed=config.get("seed", 42),
        dataloader_pin_memory=not is_mps,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.add_callback(JsonlMetricsCallback(str(output_dir / "metrics.jsonl")))
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    shutil.copy2(args.config, output_dir / "run_config.yaml")

    rewards = [
        {
            "step": item.get("step"),
            "rewards_chosen": item.get("rewards/chosen"),
            "rewards_rejected": item.get("rewards/rejected"),
            "reward_accuracy": item.get("rewards/accuracies"),
        }
        for item in trainer.state.log_history
        if "rewards/chosen" in item
    ]
    summary = {
        "stage": "dpo",
        "base_model": config["base_model"],
        "sft_adapter": config["sft_adapter"],
        "dataset_size": len(dataset),
        "beta": beta,
        "train_loss": round(result.training_loss, 5),
        "elapsed_sec": round(elapsed, 1),
        "peak_vram_gb": gpu_peak_memory_gb(device_name),
        "device": device_name,
        "use_4bit": use_4bit,
        "max_length": max_length,
        "reward_history": rewards[-20:],
        "output_dir": str(output_dir),
    }
    write_summary(str(output_dir / "training_summary.json"), summary)
    print("DPO 完成：", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
