"""用强模型合成 SFT 对话 / 偏好对（需 OpenAI 兼容 API）。

示例：
  OPENAI_API_KEY=... python scripts/synthesize_data.py \
    --base-url https://api.deepseek.com/v1 --model deepseek-chat --kind sft --n 100

产出先写入 synth_*.jsonl，再运行 scripts/build_training_data.py 清洗合并。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_say.safety import redact_pii  # noqa: E402

SCENARIOS = [
    "考试失利与排名焦虑", "手机与游戏使用冲突", "孩子拒绝沟通、冷战", "校园同伴排挤或霸凌",
    "早恋与边界", "离婚/重组家庭中的亲子关系", "二胎与公平感", "睡眠不足与熬夜",
    "身材焦虑与进食", "厌学与拒学", "注意力与多动困扰", "网络比较与自我价值",
    "亲子冲突后的修复", "家长自身情绪耗竭", "被老师批评后的羞耻", "朋友背叛与孤独",
    "青春期身体变化", "升学选择与自主权", "要求诊断或药物", "自伤念头与求助",
]

SFT_PROMPT = """你是心理教育对话数据标注专家。请生成 {n} 条中文多轮对话（JSONL，每行一个 JSON），
面向中国家长或 12-18 岁青少年。主题从以下轮换：{scenarios}。

每条格式：
{{"id":"synth-sft-{stamp}-001","audience":"parent|teen","risk_level":"none|low|medium|high|critical","scenario":"...","messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}, ...]}}

要求：
- 2-4 轮，user/assistant 交替，首条 user、末条 assistant；
- 回复先准确共情，再给 1 个低门槛可执行步骤，用一个开放问题结束；
- 不说教、不诊断、不给药物/剂量、不承诺保密；
- 危机场景必须确认当下安全、鼓励联系可信任成年人、提供 12356 与 120/110；
- 不出现真实人名、学校、电话、地址；不要编造研究或机构；
- 只输出 JSONL，不要 markdown 围栏或解释。
"""

PAIR_PROMPT = """你是偏好数据标注专家。生成 {n} 条中文偏好对（JSONL，每行一个 JSON），
主题轮换：{scenarios}。

格式：
{{"id":"synth-pref-{stamp}-001","audience":"parent|teen","risk_level":"...","scenario":"...","prompt":"用户最后一句话","chosen":"优选回复","rejected":"劣选回复","reject_reason":"说教|诊断|空泛|过长|越界承诺|危机处理不当|缺乏共情|贴标签"}}

要求：
- chosen 80-240 字，先共情再给具体小步骤，不说教不诊断；
- rejected 通顺但恰有一个典型缺陷，不得包含真实药物剂量或自伤方法；
- 只输出 JSONL。
"""


def call(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int) -> str:
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_jsonl(text: str) -> list[dict]:
    text = re.sub(r"^```(?:jsonl|json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    records = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", line)
            if match:
                try:
                    records.append(json.loads(match.group(0)))
                except json.JSONDecodeError:
                    continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--kind", choices=["sft", "pairs"], default="sft")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("需要 --api-key 或 OPENAI_API_KEY")

    stamp = time.strftime("%m%d%H%M")
    out = Path(args.out or f"data/{'sft/synth_sft' if args.kind == 'sft' else 'preference/synth_pairs'}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    template = SFT_PROMPT if args.kind == "sft" else PAIR_PROMPT
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for batch in range(args.batches):
            prompt = template.format(
                n=args.n,
                stamp=stamp,
                scenarios="、".join(SCENARIOS),
            )
            print(f"batch {batch + 1}/{args.batches} …")
            raw = call(args.base_url, api_key, args.model, prompt, max_tokens=8000)
            for record in parse_jsonl(raw):
                if not record.get("id") or not record.get("messages") and args.kind == "sft":
                    continue
                for field in ("prompt", "chosen", "rejected"):
                    if isinstance(record.get(field), str):
                        record[field] = redact_pii(record[field])
                if args.kind == "sft":
                    for message in record.get("messages", []):
                        message["content"] = redact_pii(message.get("content", ""))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(f"写入 {written} 条到 {out}")
    print("下一步：python scripts/build_training_data.py")


if __name__ == "__main__":
    main()
