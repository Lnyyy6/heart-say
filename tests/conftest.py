"""测试默认使用内存向量库，避免依赖外部服务与文件锁。"""

from __future__ import annotations

import os

os.environ.setdefault("QDRANT_PATH", "")
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("LLM_BACKEND", "offline")
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("RERANKER_BACKEND", "heuristic")
