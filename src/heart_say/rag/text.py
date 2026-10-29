"""中文分词、分句与分块。"""

from __future__ import annotations

import re
from typing import Any, Iterable

import jieba

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")
_STOPWORDS = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "和", "与", "也", "就", "都",
    "而", "及", "或", "一个", "这个", "那个", "什么", "怎么", "为什么", "可以", "但是", "因为",
    "所以", "如果", "还是", "已经", "没有", "不是", "自己", "孩子", "家长", "我们",
}


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 字符二元组，兼顾中文短语与未登录词。"""
    text = re.sub(r"\s+", "", text or "")
    words = [w for w in jieba.lcut(text) if len(w.strip()) > 1 and w not in _STOPWORDS]
    chars = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    bigrams = [chars[i : i + 2] for i in range(len(chars) - 1)]
    return words + bigrams


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    return parts or [text]


def chunk_record(record: dict[str, Any], max_chars: int = 360) -> list[dict[str, Any]]:
    """按句切块，块间保留一句重叠，避免把建议拦腰截断。"""
    text = (record.get("text") or "").strip()
    if not text:
        return []
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if size + len(sentence) > max_chars and current:
            chunks.append("".join(current))
            current = [current[-1]] if len(current) > 1 else []
            size = sum(len(s) for s in current)
        current.append(sentence)
        size += len(sentence)
    if current:
        chunks.append("".join(current))

    # 极短正文不拆分
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        chunks = [text]

    results: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = record["id"] if len(chunks) == 1 else f"{record['id']}-c{i}"
        results.append(
            {
                "chunk_id": chunk_id,
                "doc_id": record["id"],
                "title": record.get("title", ""),
                "topic": record.get("topic", "general"),
                "audience": record.get("audience", "both"),
                "risk_level": record.get("risk_level", "none"),
                "source_name": record.get("source_name", ""),
                "source_url": record.get("source_url", ""),
                "key_points": record.get("key_points", []),
                "text": chunk,
            }
        )
    return results


def iter_chunks(records: Iterable[dict[str, Any]], max_chars: int = 360) -> Iterable[dict[str, Any]]:
    for record in records:
        yield from chunk_record(record, max_chars=max_chars)
