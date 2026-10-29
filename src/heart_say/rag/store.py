"""向量库：Qdrant（服务端/嵌入式）与内存降级实现。"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("8f1d6b40-3f1a-4a5a-9a4a-6d2f0f3c9a10")


class MemoryVectorStore:
    """进程内余弦检索，用于 Qdrant 不可用或单元测试。"""

    mode = "memory"

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        if recreate:
            self.chunks = []
            self.vectors = np.zeros((0, dim), dtype=np.float32)

    def upsert(self, chunks: list[dict[str, Any]], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        if self.vectors.size == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        if not self.chunks or self.vectors.size == 0:
            return []
        scores = self.vectors @ vector.astype(np.float32)
        order = np.argsort(-scores)[:limit]
        return [
            {"payload": self.chunks[i], "score": float(scores[i])}
            for i in order
        ]

    def count(self) -> int:
        return len(self.chunks)

    def close(self) -> None:
        return None


class QdrantVectorStore:
    mode = "qdrant"

    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        settings = get_settings()
        self.collection = settings.qdrant_collection
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=20
            )
            self.mode = "qdrant-server"
            # Compose 启动顺序不保证 Qdrant 已就绪；最多等 60 秒
            for _ in range(30):
                try:
                    self.client.get_collections()
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError("Qdrant 服务在 60 秒内未就绪")
        else:
            path = settings.qdrant_local_path
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.mode = "qdrant-embedded"

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("创建 Qdrant collection=%s dim=%s", self.collection, dim)

    def upsert(self, chunks: list[dict[str, Any]], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, chunk["chunk_id"])),
                vector=vector.tolist(),
                payload=chunk,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 128):
            self.client.upsert(collection_name=self.collection, points=points[start : start + 128])

    def search(self, vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                limit=limit,
                with_payload=True,
            )
            points = result.points
        else:  # 兼容旧版 qdrant-client
            points = self.client.search(
                collection_name=self.collection,
                query_vector=vector.tolist(),
                limit=limit,
                with_payload=True,
            )
        return [{"payload": point.payload or {}, "score": float(point.score)} for point in points]

    def count(self) -> int:
        try:
            return int(self.client.count(self.collection).count)
        except Exception:
            return 0

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def get_store() -> MemoryVectorStore | QdrantVectorStore:
    settings = get_settings()
    if settings.qdrant_url or settings.qdrant_path:
        try:
            return QdrantVectorStore()
        except Exception as exc:  # 文件锁、依赖缺失、服务不可用
            logger.warning("Qdrant 不可用，降级到内存向量库：%s", exc)
    return MemoryVectorStore()


def write_manifest(payload: dict[str, Any]) -> Path:
    path = get_settings().root / "data" / "index" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_manifest() -> dict[str, Any]:
    path = get_settings().root / "data" / "index" / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
