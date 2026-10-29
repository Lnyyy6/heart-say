"""向量编码器。

默认 HashEmbedder：不依赖下载、确定性、CPU 毫秒级，用于演示/CI/离线评测。
生产切换 BGE-M3：设置 EMBEDDING_BACKEND=bge-m3 并安装 sentence-transformers。
两者的接口一致，余弦相似度可直接切换。
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from ..config import get_settings
from .text import tokenize


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """特征哈希 + 词/二元组 TF，L2 归一化。

    这是可解释的降级方案，不代表最终检索质量；它的价值是让 RAG 全链路
    在没有 GPU、没有模型下载的环境里可运行、可复现。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _hash(token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % (2**32), 1.0 if value & 1 else -1.0

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            index, sign = self._hash(token)
            vector[index % self.dim] += sign * (1.0 + np.log1p(count))
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode_one(text) for text in texts])


class BgeM3Embedder:
    """BAAI/bge-m3，1024 维，归一化后使用内积/余弦。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_backend == "bge-m3":
        return BgeM3Embedder(settings.embedding_model)
    return HashEmbedder(dim=settings.embedding_dim)
