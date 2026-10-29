"""训练脚本共用的配置、校验与指标回调。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

SFT_SYSTEM_PROMPT = (
    "你是“心语”，面向中国家庭的青少年心理教育陪伴助手。"
    "你不做诊断、不推荐药物或剂量、不承诺保密；先共情再给具体的小步骤，不说教。"
    "遇到自伤、自杀、虐待、暴力或药物过量，先确认当下安全，鼓励联系可信任的成年人，"
    "提供全国心理援助热线 12356，紧急危险拨打 120/110。"
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_sft(paths: list[str]) -> tuple[int, list[str]]:
    count = 0
    problems: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = record.get("messages") or []
                if len(messages) < 4:
                    problems.append(f"{path}:{line_no} 少于 2 轮")
                roles = [m.get("role") for m in messages]
                if roles and roles[0] != "user":
                    problems.append(f"{path}:{line_no} 首条不是 user")
                if any(
                    roles[i] == roles[i + 1] for i in range(len(roles) - 1)
                ):
                    problems.append(f"{path}:{line_no} 角色未交替")
                count += 1
    return count, problems


def validate_preference(path: str) -> tuple[int, list[str]]:
    path_obj = Path(path)
    count = 0
    problems: list[str] = []
    if not path_obj.exists():
        return 0, [f"文件不存在：{path}"]
    with path_obj.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("prompt", "chosen", "rejected"):
                if not record.get(field):
                    problems.append(f"{path}:{line_no} 缺少 {field}")
            if record.get("chosen") == record.get("rejected"):
                problems.append(f"{path}:{line_no} chosen 与 rejected 相同")
            count += 1
    return count, problems


def build_sft_text(tokenizer, record: dict[str, Any]) -> str:
    messages = record["messages"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SFT_SYSTEM_PROMPT}] + messages
    return tokenizer.apply_chat_template(messages, tokenize=False)


class JsonlMetricsCallback:
    """延迟导入 transformers，避免数据校验阶段依赖 GPU 环境。"""

    def __new__(cls, path: str):
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self):
                self.path = Path(path)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.started = time.time()

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs or "loss" not in logs:
                    return
                payload = {
                    "step": state.global_step,
                    "epoch": round(state.epoch or 0, 3),
                    "loss": round(float(logs["loss"]), 5),
                    "learning_rate": logs.get("learning_rate"),
                    "elapsed_sec": round(time.time() - self.started, 1),
                    "peak_vram_gb": gpu_peak_memory_gb(),
                }
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return _Callback()


def gpu_peak_memory_gb(device: str = "") -> float | None:
    try:
        import torch

        if device == "mps" or (torch.backends.mps.is_available() and not torch.cuda.is_available()):
            return round(torch.mps.driver_allocated_memory() / 1024**3, 2)
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        pass
    return None


def write_summary(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
