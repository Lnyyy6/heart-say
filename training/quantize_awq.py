"""AWQ 4-bit 量化（需要 GPU 与 autoawq）。

用法：
  python training/quantize_awq.py --model outputs/merged-qwen7b --output outputs/qwen7b-awq
  vllm serve outputs/qwen7b-awq --served-model-name heart-say-qwen --quantization awq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


CALIBRATION = [
    "我女儿初三，最近天天失眠，我比她还焦虑。",
    "孩子把自己关在房间里，不愿意跟我说话。",
    "我一考试就紧张到手抖，怕考不好让爸妈失望。",
    "孩子被同学起外号，回家一直哭，我该怎么安慰？",
    "我说什么孩子都顶嘴，亲子关系越来越僵。",
    "最近总是睡不着，白天上课也集中不了注意力。",
    "孩子说不想上学了，我既生气又害怕。",
    "我控制不住吼孩子，事后又特别自责。",
    "孩子开始在意身材，经常不吃饭，我很担心。",
    "我和伴侣经常吵架，孩子会不会受影响？",
    "孩子说同学都在玩游戏，不玩就融不进去。",
    "我不知道怎么跟孩子谈性教育和身体边界。",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ 4-bit 量化")
    parser.add_argument("--model", default="outputs/merged-qwen7b")
    parser.add_argument("--output", default="outputs/qwen7b-awq")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    print(f"加载模型 {args.model}")
    model = AutoAWQForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": "GEMM",
    }
    print("量化中…")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=CALIBRATION)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)
    Path(args.output, "quantization_meta.json").write_text(
        json.dumps(quant_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存 AWQ 模型到 {args.output}")


if __name__ == "__main__":
    main()
