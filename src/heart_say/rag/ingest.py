"""知识库入库 CLI：切块 → 向量化 → 写入 Qdrant/内存 → 记录 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from ..config import get_settings
from .embedder import get_embedder
from .knowledge import embedding_text, load_chunks, load_records
from .store import get_store, write_manifest


def _kb_sha256(path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(
    recreate: bool = True,
    store: Any | None = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"知识库为空或不存在：{settings.knowledge_path}")

    embedder = embedder or get_embedder()
    store = store or get_store()
    vectors = embedder.encode([embedding_text(chunk) for chunk in chunks])
    store.ensure_collection(dim=embedder.dim, recreate=recreate)
    store.upsert(chunks, vectors)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(load_records()),
        "chunks": len(chunks),
        "dim": embedder.dim,
        "embedding_backend": settings.embedding_backend,
        "reranker_backend": settings.reranker_backend,
        "store_mode": store.mode,
        "collection": settings.qdrant_collection,
        "kb_sha256": _kb_sha256(settings.knowledge_path),
        "topics": sorted({chunk["topic"] for chunk in chunks}),
    }
    # 内存库常用于评测/测试，不覆盖生产 Qdrant 的 manifest
    if store.mode != "memory":
        write_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="构建心语知识库索引")
    parser.add_argument("--no-recreate", action="store_true", help="增量写入而不清空 collection")
    parser.add_argument("--json", action="store_true", help="输出 JSON 统计")
    args = parser.parse_args()

    manifest = build_index(recreate=not args.no_recreate)
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(
            f"入库完成：{manifest['documents']} 篇文档 / {manifest['chunks']} 个片段，"
            f"向量维度 {manifest['dim']}，存储 {manifest['store_mode']}"
        )


if __name__ == "__main__":
    main()
