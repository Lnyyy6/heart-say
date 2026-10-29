"""Qwen2.5 QLoRA SFT。

用法：
  python training/sft_qlora.py --config training/configs/sft_qwen7b.yaml --dry-run
  python training/sft_qlora.py --config training/configs/sft_qwen7b.yaml

24GB 显存建议 7B + r=16 + batch=1 + grad_accum=8；不够时把 config 的
model_name_or_path 换成 Qwen2.5-3B-Instruct / 1.5B-instruct。
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

from training_utils import (
    JsonlMetricsCallback,
    build_sft_text,
    gpu_peak_memory_gb,
    load_yaml,
    validate_sft,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5 QLoRA SFT")
    parser.add_argument("--config", default="training/configs/sft_qwen7b.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只校验数据与配置，不加载模型")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no-4bit", action="store_true", help="禁用 4-bit（MPS/CPU 无 bitsandbytes 时使用）")
    parser.add_argument("--max-seq-length", type=int, default=0, help="覆盖配置的最大序列长度")
    return parser.parse_args()


def load_records(paths: list[str]) -> list[dict]:
    records = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"[warn] 数据文件不存在，跳过：{path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = config["dataset_paths"]
    count, problems = validate_sft(paths)
    print(f"数据条数：{count}；问题：{len(problems)}")
    for problem in problems[:10]:
        print(" -", problem)
    if args.dry_run:
        if problems:
            raise SystemExit("数据校验未通过")
        print("dry-run 通过。配置：", json.dumps(config, ensure_ascii=False, indent=2))
        return
    if problems:
        raise SystemExit("数据校验未通过，请先修复")
    if count == 0:
        raise SystemExit("没有训练数据")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    model_name = config["model_name_or_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    is_mps = args.device == "mps" or (
        args.device == "auto" and torch.backends.mps.is_available() and not torch.cuda.is_available()
    )
    use_4bit = not args.no_4bit and torch.cuda.is_available() and not is_mps
    device_name = "mps" if is_mps else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"加载 tokenizer / model：{model_name}（device={device_name}，4bit={use_4bit}）")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
            model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.float32 if is_mps or device_name == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if is_mps:
            model = model.to("mps")

    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True)
        )
    elif train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    records = load_records(paths)
    dataset = Dataset.from_list([{"text": build_sft_text(tokenizer, record)} for record in records])
    print(f"训练样本：{len(dataset)}")

    sft_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": train_cfg["num_train_epochs"],
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1 if is_mps else train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "logging_steps": train_cfg["logging_steps"],
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg["save_total_limit"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "bf16": train_cfg["bf16"] and not is_mps,
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "optim": "adamw_torch" if is_mps else train_cfg["optim"],
        "report_to": "none",
        "seed": config.get("seed", 42),
        "dataset_text_field": "text",
        "dataloader_pin_memory": not is_mps,
    }
    parameters = inspect.signature(SFTConfig.__init__).parameters
    length_key = "max_seq_length" if "max_seq_length" in parameters else "max_length"
    max_seq_length = args.max_seq_length or train_cfg["max_seq_length"]
    if is_mps:
        max_seq_length = min(max_seq_length, 768)
    sft_kwargs[length_key] = max_seq_length
    sft_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.add_callback(JsonlMetricsCallback(str(output_dir / "metrics.jsonl")))
    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    shutil.copy2(args.config, output_dir / "run_config.yaml")

    losses = [
        record["loss"]
        for record in trainer.state.log_history
        if "loss" in record
    ]
    summary = {
        "stage": "sft",
        "model": model_name,
        "dataset_size": len(dataset),
        "epochs": train_cfg["num_train_epochs"],
        "max_steps": args.max_steps,
        "train_loss": round(train_result.training_loss, 5),
        "last_logged_loss": round(losses[-1], 5) if losses else None,
        "elapsed_sec": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 1),
        "peak_vram_gb": gpu_peak_memory_gb(device_name),
        "device": device_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Apple MPS" if is_mps else "cpu",
        "max_seq_length": max_seq_length,
        "use_4bit": use_4bit,
        "lora": lora_cfg,
        "output_dir": str(output_dir),
    }
    write_summary(str(output_dir / "training_summary.json"), summary)
    print("训练完成：", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
