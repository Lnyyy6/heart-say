"""知识库加载与切块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import get_settings
from .text import iter_chunks

REQUIRED_FIELDS = {"id", "title", "topic", "text", "source_name"}


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or get_settings().knowledge_path
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} 缺少字段 {sorted(missing)}")
            if record["id"] in seen:
                raise ValueError(f"{path}:{line_no} 重复 id {record['id']}")
            seen.add(record["id"])
            record.setdefault("key_points", [])
            record.setdefault("audience", "both")
            record.setdefault("risk_level", "none")
            record.setdefault("source_url", "")
            records.append(record)
    return records


def load_chunks(max_chars: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    records = load_records()
    return list(iter_chunks(records, max_chars=max_chars or settings.max_doc_chars))


def embedding_text(chunk: dict[str, Any]) -> str:
    tips = "；".join(chunk.get("key_points") or [])
    return f"{chunk['title']}\n{tips}\n{chunk['text']}"
