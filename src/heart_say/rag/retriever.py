"""混合检索：Dense + BM25 → RRF → Rerank，输出带来源的片段。"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import get_settings
from ..schemas import RetrievedDoc
from .embedder import get_embedder
from .knowledge import embedding_text, load_chunks
from .store import get_store, read_manifest
from .text import tokenize

logger = logging.getLogger(__name__)


class HeuristicReranker:
    """无模型 CrossEncoder 的可解释替代：词面覆盖 + 标题/要点命中 + 场景先验。"""

    def score_many(
        self, query: str, docs: list[dict[str, Any]], base_scores: list[float], audience: str
    ) -> list[float]:
        return [
            self._score_one(query, doc, base, audience)
            for doc, base in zip(docs, base_scores, strict=True)
        ]

    def _score_one(
        self, query: str, doc: dict[str, Any], base_score: float, audience: str
    ) -> float:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return base_score
        title_tokens = set(tokenize(doc.get("title", "")))
        point_text = "；".join(doc.get("key_points") or [])
        point_tokens = set(tokenize(point_text))
        doc_tokens = set(tokenize(doc.get("text", "")))

        coverage = len(query_tokens & doc_tokens) / (len(query_tokens) ** 0.5 + 1e-6)
        title_hit = len(query_tokens & title_tokens) / (len(title_tokens) + 1e-6)
        point_hit = len(query_tokens & point_tokens) / (len(point_tokens) + 1e-6)

        score = (
            0.30 * base_score
            + 0.40 * min(coverage, 1.0)
            + 0.20 * min(title_hit * 2, 1.0)
            + 0.10 * min(point_hit * 2, 1.0)
        )
        doc_audience = doc.get("audience", "both")
        if audience in ("parent", "teen") and doc_audience in (audience, "both"):
            score += 0.05
        if doc.get("topic") == "crisis" and doc.get("risk_level") in ("high", "critical"):
            score -= 0.03
        return score


class BgeReranker:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import CrossEncoder

        if device == "auto":
            import torch

            device = "mps" if torch.backends.mps.is_available() else None
        self.model = CrossEncoder(model_name, device=device)

    def score_many(
        self, query: str, docs: list[dict[str, Any]], base_scores: list[float], audience: str
    ) -> list[float]:
        if not docs:
            return []
        pairs = [(query, f"{doc.get('title', '')}\n{doc.get('text', '')}") for doc in docs]
        raw = self.model.predict(pairs, show_progress_bar=False)
        # CrossEncoder 输出 logits，用 sigmoid 归一到 0-1
        normalized = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))
        return [
            0.85 * float(score) + 0.15 * base
            for score, base in zip(normalized, base_scores, strict=True)
        ]


class Retriever:
    def __init__(self, auto_build: bool = True):
        settings = get_settings()
        self.settings = settings
        self.embedder = get_embedder()
        self.store = get_store()
        self.chunks = load_chunks()
        self._by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self._bm25, self._bm25_ids = self._build_bm25()
        self.reranker = self._build_reranker()
        self._ensure_index(auto_build)

    def _build_bm25(self) -> tuple[BM25Okapi | None, list[str]]:
        if not self.chunks:
            return None, []
        corpus = [tokenize(embedding_text(chunk)) for chunk in self.chunks]
        ids = [chunk["chunk_id"] for chunk in self.chunks]
        return BM25Okapi(corpus), ids

    def _build_reranker(self):
        if self.settings.reranker_backend == "bge":
            try:
                return BgeReranker(self.settings.reranker_model, device=self.settings.reranker_device)
            except Exception as exc:
                logger.warning("BGE reranker 加载失败，使用启发式：%s", exc)
        return HeuristicReranker()

    def _ensure_index(self, auto_build: bool) -> None:
        manifest = read_manifest()
        count = self.store.count()
        stale = bool(manifest) and (
            manifest.get("chunks") != len(self.chunks)
            or manifest.get("dim") != self.embedder.dim
            or manifest.get("embedding_backend") != self.settings.embedding_backend
            or (
                getattr(self.store, "collection", None)
                and manifest.get("collection") != self.store.collection
            )
        )
        if manifest and not stale and count >= len(self.chunks):
            return
        if not auto_build:
            return
        try:
            from .ingest import build_index

            logger.info("索引缺失或过期，正在构建…")
            build_index(recreate=True, store=self.store, embedder=self.embedder)
        except Exception as exc:
            logger.warning("自动建索引失败：%s", exc)

    def _dense(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = self.embedder.encode([query])[0]
        hits = self.store.search(vector, limit)
        results: list[tuple[str, float]] = []
        for hit in hits:
            payload = hit["payload"]
            results.append((payload.get("chunk_id", ""), float(hit["score"])))
        return results

    def _sparse(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:limit]
        return [(self._bm25_ids[i], float(scores[i])) for i in order if scores[i] > 0]

    @staticmethod
    def _rrf(
        rankings: list[list[tuple[str, float]]], weights: list[float], k: int = 60
    ) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking, weight in zip(rankings, weights, strict=True):
            for rank, (chunk_id, _) in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
        return fused

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        audience: str = "unknown",
        risk_level: str = "none",
    ) -> list[RetrievedDoc]:
        query = (query or "").strip()
        if not query or not self.chunks:
            return []
        settings = self.settings
        top_k = top_k or settings.top_k_final

        dense = self._dense(query, settings.top_k_dense)
        sparse = self._sparse(query, settings.top_k_sparse)
        # hash 向量在无模型环境下噪声较大，BM25 给更高权重；BGE-M3 时仍保留混合
        fused = self._rrf([dense, sparse], weights=[0.35, 0.65])
        if not fused:
            return []

        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        max_rrf = fused[ranked_ids[0]] or 1.0
        dense_scores = dict(dense)
        sparse_scores = dict(sparse)
        max_sparse = max(sparse_scores.values(), default=1.0) or 1.0
        max_dense = max(dense_scores.values(), default=1.0) or 1.0

        query_tokens = set(tokenize(query))
        rows: list[tuple[float, dict[str, Any]]] = []
        for chunk_id in ranked_ids[: settings.top_k_dense + settings.top_k_sparse]:
            chunk = self._by_id.get(chunk_id)
            if not chunk:
                continue
            # 非危机场景不混入危机处理卡片
            if (
                chunk.get("topic") == "crisis"
                and risk_level not in ("high", "critical")
                and not re.search(r"(自杀|自伤|自残|不想活|遗书|虐待|家暴)", query)
            ):
                continue
            # 受众错位的卡片不进入候选（青少年提问时不推家长操作指南）
            if audience == "teen" and chunk.get("audience") == "parent":
                continue
            if audience == "parent" and chunk.get("audience") == "teen":
                continue
            lexical = len(query_tokens & set(tokenize(chunk["text"] + chunk["title"])))
            if lexical == 0 and sparse_scores.get(chunk_id, 0.0) == 0:
                continue
            base = (
                0.65 * sparse_scores.get(chunk_id, 0.0) / max_sparse
                + 0.35 * dense_scores.get(chunk_id, 0.0) / max_dense
            )
            rows.append((0.7 * base + 0.3 * (fused[chunk_id] / max_rrf), chunk))

        reranked = self.reranker.score_many(
            query, [chunk for _, chunk in rows], [base for base, _ in rows], audience
        )
        candidates = list(zip(reranked, [chunk for _, chunk in rows], strict=True))
        candidates.sort(key=lambda item: item[0], reverse=True)

        # 同文档去重，保证引用多样性；不足时再回填
        selected: list[tuple[float, dict[str, Any]]] = []
        seen_docs: set[str] = set()
        for score, chunk in candidates:
            doc_id = chunk["doc_id"]
            if doc_id in seen_docs:
                continue
            selected.append((score, chunk))
            seen_docs.add(doc_id)
            if len(selected) >= top_k:
                break

        results: list[RetrievedDoc] = []
        for score, chunk in selected:
            results.append(
                RetrievedDoc(
                    id=chunk["chunk_id"],
                    title=chunk["title"],
                    topic=chunk["topic"],
                    audience=chunk["audience"],
                    risk_level=chunk["risk_level"],
                    source_name=chunk["source_name"],
                    source_url=chunk["source_url"],
                    text=chunk["text"],
                    key_points=chunk.get("key_points", []),
                    score=round(fused.get(chunk["chunk_id"], 0.0), 6),
                    rerank_score=round(float(score), 6),
                )
            )
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(auto_build=True)


def to_citations(docs: list[RetrievedDoc]) -> list[Any]:
    from ..schemas import Citation

    citations = []
    for index, doc in enumerate(docs, start=1):
        citations.append(
            Citation(
                index=index,
                doc_id=doc.id,
                title=doc.title,
                topic=doc.topic,
                source_name=doc.source_name,
                source_url=doc.source_url,
                snippet=re.sub(r"\s+", " ", doc.text)[:140],
                score=doc.score,
                rerank_score=doc.rerank_score,
            )
        )
    return citations
