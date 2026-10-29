"""运行时配置。所有路径默认相对仓库根目录解析。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "心语"
    app_env: str = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8010
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # 生成模型
    llm_backend: Literal["offline", "openai_compatible", "transformers"] = "offline"
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen2.5-7B-Instruct-AWQ"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 600
    llm_timeout: float = 60.0

    # 本地 Transformers 推理（MPS/CPU 训练后评估；无 CUDA 时的降级路径）
    transformers_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    transformers_adapter_path: str = "outputs/sft-qwen1.5b"
    transformers_device: str = "auto"
    transformers_max_new_tokens: int = 260

    # 评测 judge（llm 需要单独配置更强的模型；默认离线规则评分）
    judge_backend: Literal["heuristic", "llm"] = "heuristic"
    judge_base_url: str = ""
    judge_api_key: str = ""
    judge_model: str = ""

    # RAG
    embedding_backend: Literal["hash", "bge-m3"] = "hash"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_dim: int = 512
    reranker_backend: Literal["heuristic", "bge"] = "heuristic"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "heart_say_kb"
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 4
    max_doc_chars: int = 360

    # 数据与记忆
    knowledge_file: str = "data/knowledge/kb.jsonl"
    session_dir: str = "data/sessions"
    redis_url: str = ""
    max_history_turns: int = 8

    # 安全
    enable_output_guard: bool = True
    crisis_hotline: str = "12356"

    # 常用绝对路径
    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def knowledge_path(self) -> Path:
        return self.resolve(self.knowledge_file)

    @property
    def sessions_path(self) -> Path:
        return self.resolve(self.session_dir)

    @property
    def qdrant_local_path(self) -> Path:
        return self.resolve(self.qdrant_path)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def report_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    @property
    def eval_cases_path(self) -> Path:
        return PROJECT_ROOT / "data" / "eval" / "cases.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
