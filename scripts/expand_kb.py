"""把 80 张知识卡拆成要点/家长问答/青少年问答/信号片段，扩充到 500+ 检索片段。

原始卡片保留；派生片段共享同一来源与主题，便于引用核对。生成后需要重新
`make index` 并跑同一套 RAG 评测，确认没有拉低检索质量。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_say.rag.knowledge import load_records  # noqa: E402

KB = ROOT / "data" / "knowledge" / "kb.jsonl"
BACKUP = ROOT / "data" / "knowledge" / "kb_v0.jsonl"


def sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[。！？])", text) if item.strip()]


def compact(text: str, limit: int = 150) -> str:
    return text if len(text) <= limit else text[:limit].rstrip("，。；") + "。"


def derive(record: dict) -> list[dict]:
    text = record["text"]
    points = [point for point in record.get("key_points", []) if point]
    sents = sentences(text)
    first = sents[0] if sents else text
    second = "".join(sents[1:3]) if len(sents) > 1 else first
    base = {
        "topic": record["topic"],
        "audience": record["audience"],
        "risk_level": record["risk_level"],
        "source_name": record["source_name"],
        "source_url": record.get("source_url", ""),
        "language": record.get("language", "zh"),
    }
    derived: list[dict] = []
    for index, point in enumerate(points[:3], start=1):
        derived.append({
            **base,
            "id": f"{record['id']}-kp{index}",
            "title": f"{record['title']}·要点{index}",
            "key_points": [point],
            "text": compact(f"{point}。{first}"),
        })
    parent_answer = "；".join(points[:3]) or first
    derived.append({
        **base,
        "id": f"{record['id']}-qa-parent",
        "title": f"{record['title']}·家长问答",
        "audience": "parent",
        "key_points": points[:3],
        "text": compact(f"家长可以怎么做？{parent_answer}。{second}"),
    })
    derived.append({
        **base,
        "id": f"{record['id']}-qa-teen",
        "title": f"{record['title']}·青少年问答",
        "audience": "teen",
        "key_points": points[:3],
        "text": compact(f"青少年可以怎么办？{parent_answer}。{second}"),
    })
    signal = next((sentence for sentence in sents if re.search(r"(如果|信号|注意|观察|可能|建议)", sentence)), second)
    derived.append({
        **base,
        "id": f"{record['id']}-signal",
        "title": f"{record['title']}·信号",
        "key_points": [signal],
        "text": compact(f"需要留意的信号：{signal}"),
    })
    return derived


def main() -> None:
    if not BACKUP.exists():
        BACKUP.write_text(KB.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"备份原始知识库 → {BACKUP}")
    records = load_records(BACKUP)
    print(f"从原始备份读取 {len(records)} 张来源卡片")
    expanded: list[dict] = []
    for record in records:
        expanded.append(record)
        expanded.extend(derive(record))
    KB.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in expanded) + "\n",
        encoding="utf-8",
    )
    print(f"来源卡片 {len(records)} → 检索片段 {len(expanded)}")
    print("下一步：make index && 重新跑 RAG 评测")


if __name__ == "__main__":
    main()
